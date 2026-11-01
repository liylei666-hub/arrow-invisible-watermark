# 微信云托管部署 ARROW 解码

把已经在仓库里的 FastAPI（`services/decode_api/`）交到微信云托管服务 **`arrow-decode`** 上构建。本文是控制台填写清单和 `callContainer` 示例。镜像由云托管从 GitHub 拉取代码后构建。算法和接受门不在这里改。

本地「代码包 / 文件夹」上传上限约 **2 MiB**。本仓库用 **GitHub 代码库拉取** 构建，不走 zip 上传。

只新建 **`arrow-decode`**。控制台里其他已经在跑的服务不要打开、不要改镜像、端口、环境变量或流水线。

TI-ONE 那条线仍然看 [services/decode_api/TI-ONE.md](../services/decode_api/TI-ONE.md)。小程序怎么读返回 JSON，看 [miniprogram-integration.md](miniprogram-integration.md)。

## 镜像怎么被构建

微信云托管的「目标目录」就是 Docker 构建上下文，Dockerfile 按这个目录为根来找。下面两种填法打出来的是同一份镜像（三份 Dockerfile 内容相同，CI 会核对）：

| | 目标目录 | Dockerfile 名称 |
|---|---|---|
| 推荐 | 留空（仓库根） | `Dockerfile` |
| 备选 | `wxcloudrun` | `Dockerfile` |

`wxcloudrun/` 里放了一份构建所需源码（`arrow_decode.py`、`arrow_wm.py`、`arrow_inner.py`、`ARROW_clean_pro.png`、`services/decode_api/app.py` 等），和仓库根是同一份内容。改解码或 HTTP 服务时两处一起改，否则 CI 失败。

容器进程监听 **`0.0.0.0:8501`**（环境变量 `PORT` 未设置时）。`GET /health` 返回 `{"status":"ok"}`，不校验令牌。`POST /decode` 的 multipart 行为与原来一致。

本地核对构建：

```bash
docker build -t arrow-decode-api .
docker build -t arrow-decode-api wxcloudrun
docker run --rm -p 8501:8501 arrow-decode-api
```

## 控制台清单

路径：微信云托管控制台 → 服务管理。下列名字和数字按字面填写。

### 服务

| 项 | 值 |
|---|---|
| 服务名称 | `arrow-decode` |
| 说明 | 名称要和下面 `X-WX-SERVICE` 完全一致，包括大小写和连字符 |

### 版本上传方式

| 项 | 值 |
|---|---|
| 上传方式 | **代码库拉取** → GitHub |
| 仓库 | 本仓库 |
| 分支 | 含这次改动的分支。合并前可以用该分支做一次验证发布 |
| 推送触发 | 先不勾。`/health` 确认通过之后再决定要不要 push 自动发布 |
| 有无 Dockerfile | **有 Dockerfile**。不要选「无 Dockerfile」的 Python 模板（那个模板会另写启动命令） |
| 目标目录 | 留空，或填 `wxcloudrun`（见上表，二选一） |
| Dockerfile 名称 | `Dockerfile` |
| 启动命令 | 留空，用镜像里的 uvicorn |

首次绑定 GitHub 时按控制台提示授权。授权只给云托管拉取代码。

### 监听端口、规格、副本

| 项 | 值 |
|---|---|
| 监听端口 | `8501` |
| CPU | `0.5` 核 |
| 内存 | `1` GB |
| 实例副本数最小值 | `1` |
| 实例副本数最大值 | `2` |
| 扩缩容条件 | CPU 使用率 `60`。表单若还有内存一条，填 `60` |
| 启动检测延迟 | `120` 秒（字段名可能写作 `initialDelaySeconds`，在高级设置里） |
| 日志采集路径 | `stdout` |

监听端口不要留默认的 `80`。端口填错时版本会创建失败：平台把流量转到你填的端口，进程实际听的是 8501。

最小副本数用 `1` 是因为 `wx.cloud.callContainer` 的超时上限是 15 秒，而本进程在打出日志 `[decode_api] ready` 之前不接受连接（要先加载 SIFT 模板）。副本最小值为 `0` 时，大约 30 分钟无请求会缩到 0，下一次调用会在冷启动上超时（`errCode: 102002`）。最小值为 1 会持续占 0.5 核 / 1 GB。

最大副本数用 `2`。解码在单个进程里是串行的，0.5 核很容易被打满；默认最大 50 会扩出很多份各自带模板的实例。

启动检测延迟用 120 秒：端口在模板加载完成前是关的。日志里出现 `[decode_api] ready` 才算就绪。表单没有这一项时，以版本状态和这份日志为准。

若实例在加载模板时因内存被杀掉，只把 **本服务** 的内存调到 `2GB`（0.5 核允许 1–4 GB），或只在本服务增加环境变量 `DECODE_WARMUP=edge`（只预热边缘模板）。

### 环境变量

只加在 **`arrow-decode`** 上。

| 键 | 值 |
|---|---|
| `DECODE_TOKEN` | 自行生成的共享密钥 |

- 值不要写入 git、Dockerfile、`container.config.json` 或小程序源码仓库的公开文件。
- 设置之后，`POST /decode` 要求请求头 `X-Decode-Token` 与该值完全一致，否则 `401`，正文 `{"detail":"unauthorized"}`。
- 不设置或留空则 `/decode` 不鉴权。
- `GET /health` 始终不读这个变量。
- 不要另设 `PORT`。未设置时进程监听 8501。若必须设置，值只能是 `8501`，并且和监听端口相同。

### 发布

- 版本构建成功后，把流量分给这个版本（发布时可以直接 100%）。没有线上版本时，`callContainer` 会报服务不存在（`EnvDbId [], ServiceName [] not exist`）。
- 公网访问可以关闭。小程序走 `callContainer`，不需要公网域名。默认公网域名只适合临时 `curl`，不要写进小程序的 request 合法域名当生产入口。

发布后若开着默认公网域名，可以先打健康检查（域名以控制台显示的为准，调用方不写端口）：

```bash
curl -s "https://<控制台给出的默认域名>/health"
```

期望：`{"status":"ok"}`。

## 小程序调用

基础库不低于 `2.23.0`，并在小程序管理后台把「基础库最低版本」设到 `2.23.0` 或更高，否则没有 `wx.cloud.callContainer`。

`callContainer` 的 `header` 里带 `X-WX-SERVICE: arrow-decode`。这是云托管网关用来选服务的头。容器里的路径仍是 **`/decode`**。

`callContainer` 组 multipart 不方便（默认 `Content-Type` 是 `application/json`，文件字段经常到不了容器）。`POST /decode` 因此同时接受三种正文，返回 JSON 相同。multipart 的字段、状态码和错误正文没有改。

| Content-Type | 正文 | `carrier` |
|---|---|---|
| `multipart/form-data` | 文件字段 `image`（以及原来的别名） | 查询参数优先，否则表单字段，默认 `edge` |
| `application/json` | `{"image_base64":"<base64>","carrier":"edge"}` | 查询参数优先，否则 JSON 字段，默认 `edge` |
| `application/octet-stream` 或 `image/jpeg` / `image/png` 等 | 图片原始字节 | 只看查询参数，默认 `edge` |

平台限制（调用还没进容器就会被拒）：

- 文本请求体约 **100 KiB**。`application/json` 算文本，手机原图的 base64 通常会触发 `errCode: -606001`（请求包超过 100KB）。JSON 只适合压得很小的图。
- 二进制请求体约 **20 MiB**，和本服务自己的 20 MB 上限对齐。照片用 `ArrayBuffer` 走原始字节。
- 响应约 1 MB。本接口的 JSON 远小于这个上限。
- `timeout` 最大 **15 秒**（15000 毫秒）。服务已经常驻（最小副本数 1）时，单次解码大约 0.5–2.5 秒，放得下。

返回字段、`accepted` / `digits` / `reason_text` 的用法与 [miniprogram-integration.md](miniprogram-integration.md) 相同。`candidate_digits` 不能当号码展示。

### 照片：原始字节

`FileSystemManager.readFileSync` 不传 `encoding` 时得到 `ArrayBuffer`。必须自己写上 `content-type`，否则 `callContainer` 会按 JSON 发送。

```javascript
wx.cloud.init({ env: "替换成云托管环境 ID" });

const filePath = tempFilePath; // wx.chooseMedia / chooseImage 的临时路径
const data = wx.getFileSystemManager().readFileSync(filePath);

wx.cloud.callContainer({
  path: "/decode?carrier=edge",
  method: "POST",
  timeout: 15000,
  header: {
    "X-WX-SERVICE": "arrow-decode",
    "content-type": "application/octet-stream",
    "X-Decode-Token": "<与服务环境变量 DECODE_TOKEN 相同>",
  },
  data: data,
  success(res) {
    const body = typeof res.data === "string" ? JSON.parse(res.data) : res.data;
    if (body.accepted && body.digits) {
      // 展示 body.digits
    } else {
      // 展示 body.reason_text，不要用 body.candidate_digits
    }
  },
});
```

未设置 `DECODE_TOKEN` 时不要带 `X-Decode-Token`。只拍内孔印版时把查询改成 `carrier=inner`。默认不要在界面上做载体开关。

### 小图：JSON / base64

对象形式的 `data` 会被当成 `application/json`。整段请求（base64 之后）需要留在约 100 KiB 以内，否则平台直接返回 `-606001`，容器日志里看不到这次调用。

```javascript
wx.cloud.callContainer({
  path: "/decode",
  method: "POST",
  timeout: 15000,
  header: {
    "X-WX-SERVICE": "arrow-decode",
    "X-Decode-Token": "<与服务环境变量 DECODE_TOKEN 相同>",
  },
  data: {
    image_base64: wx.getFileSystemManager().readFileSync(filePath, "base64"),
    carrier: "edge",
  },
  success(res) {
    const body = typeof res.data === "string" ? JSON.parse(res.data) : res.data;
    if (body.accepted && body.digits) {
      // 展示 body.digits
    } else {
      // 展示 body.reason_text
    }
  },
});
```

`image_base64` 也可以写成带 `data:image/jpeg;base64,` 前缀的字符串。缺字段、非法 base64、空图是 `400`。解码后超过 20 MB 是 `413`。令牌不对是 `401`。这些都不是解码结果。

公网域名上的 multipart 仍然可用（仅调试）：

```bash
curl -s -H "X-Decode-Token: <DECODE_TOKEN>" \
  -F "image=@samples/edge/ARROW_13800138000.png" \
  "https://<控制台给出的默认域名>/decode"
```
