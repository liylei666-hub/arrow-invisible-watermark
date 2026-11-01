# ARROW 解码 HTTP 服务

把仓库根目录的 `arrow_decode.Decoder` 包成一个小 HTTP 服务，给微信小程序和腾讯云 TI-ONE 在线服务用。算法和接受门都不在这里改：默认载体仍是边缘（`edge`），`inner` 只是显式选项。

契约见 [docs/miniprogram-integration.md](../../docs/miniprogram-integration.md)。TI-ONE 上怎么挂、以及不要动其他已有服务，见 [TI-ONE.md](TI-ONE.md)。微信云托管（服务名 `arrow-decode`、GitHub 构建、`callContainer`）见 [docs/wechat-cloudrun.md](../../docs/wechat-cloudrun.md)。

## 端口

容器监听 **8501**。这是 TI-ONE 在线服务的 REST 默认端口，平台健康检查默认也打 8501。本地和镜像都用这个端口，避免控制台「端口」和进程不一致时探针一直失败。

要用别的端口：环境变量 `PORT`，并且在 TI-ONE 服务配置里把「端口」设成同一个数。不要用 8500（gRPC）或保留端口 8502–8510、6006、9092。

## 本地运行

在仓库根目录：

```bash
pip install -r requirements.txt
python -m uvicorn services.decode_api.app:app --host 127.0.0.1 --port 8501
```

也可以在服务目录里启动。`app.py` 会把仓库根加进 `sys.path`，不依赖当前工作目录：

```bash
cd services/decode_api
python -m uvicorn app:app --host 127.0.0.1 --port 8501
```

进程起来后会先加载边缘和内孔两个 `Decoder`（SIFT 模板）。日志出现 `[decode_api] ready` 之后 `/health` 才会通。只预热边缘可以设 `DECODE_WARMUP=edge`。

健康检查：

```bash
curl -s http://127.0.0.1:8501/health
```

期望：`{"status":"ok"}`。

解码（字段名用 `image`；`file` / `img` / `photo` / `picture` 也能认）。默认载体是 `edge`：

```bash
curl -s -F "image=@samples/edge/ARROW_13800138000.png" \
  http://127.0.0.1:8501/decode
```

指定内孔，查询参数和表单字段都可以，查询参数优先：

```bash
curl -s -F "image=@samples/inner/ARROW_13800138000.png" \
  "http://127.0.0.1:8501/decode?carrier=inner"

curl -s -F "image=@photo.jpg" -F "carrier=edge" \
  http://127.0.0.1:8501/decode
```

成功**只**认 `accepted === true`，等价于 `status === "accepted"`。这时 `digits` 是 11 位号码，`reason_text` 为「接受」。拒绝时 HTTP 仍是 200，`status` 为 `rejected`，`accepted` 为 `false`，`digits` 为 `null`，用 `reason` / `reason_text` 提示用户，文案按「结果不可信」处理。`ok` 不是成功位，无码图也可以是 `true`。

`candidate_digits` 默认不返回。只有同时满足下面三条才出现，出现了也不能当号码：

1. 查询参数 `debug=1`
2. 服务设置了环境变量 `DECODE_TOKEN`
3. 请求头 `X-Decode-Token` 与该值一致（否则是 401，`status` 为 `error`）

请求本身不合法（400 / 401 / 413）时正文是 `{"status":"error","detail":"..."}`，不是一次解码拒绝。`GET /health` 的 `{"status":"ok"}` 只表示进程活着。

`POST /decode` 还接受另外两种正文，返回和鉴权与 multipart 相同：

- `application/json`：`{"image_base64":"<base64>","carrier":"edge"}`。查询参数 `carrier` 优先于 JSON 字段。
- `application/octet-stream` 或 `image/jpeg` / `image/png`：正文就是图片字节，`carrier` 只看查询参数。

微信 `wx.cloud.callContainer` 用这两种，因为组 multipart 不稳定。控制台步骤和示例在 [docs/wechat-cloudrun.md](../../docs/wechat-cloudrun.md)。

无码对照应被拒绝（`digits` 为 `null`）：

```bash
curl -s -F "image=@samples/edge/ARROW_无码对照.png" \
  http://127.0.0.1:8501/decode
```

## 可选鉴权

环境变量 `DECODE_TOKEN` **未设置或为空**时不鉴权，方便本地调试。设置之后，只有请求头 `X-Decode-Token` 与它完全一致时 `POST /decode` 才通过；否则 `401`，正文 `{"detail":"unauthorized"}`。`GET /health` 始终不鉴权，给 TI-ONE 探针用。

不要把令牌写进仓库或镜像。本地临时试：

```bash
DECODE_TOKEN=devsecret python -m uvicorn services.decode_api.app:app \
  --host 127.0.0.1 --port 8501

curl -s -H "X-Decode-Token: devsecret" \
  -F "image=@samples/edge/ARROW_13800138000.png" \
  http://127.0.0.1:8501/decode
```

TI-ONE 控制台自己的「服务鉴权」（`Authorization`）是另一层，和这个头互不替代。两层都开时，小程序要两个头都带。

## 镜像

两份构建上下文都可以，Dockerfile 内容相同：

```bash
# 上下文 = 仓库根（微信云托管目标目录留空）
docker build -t arrow-decode-api .
docker build -f services/decode_api/Dockerfile -t arrow-decode-api .

# 上下文 = wxcloudrun/（目标目录填 wxcloudrun，Dockerfile 在该目录根上）
docker build -t arrow-decode-api wxcloudrun

docker run --rm -p 8501:8501 arrow-decode-api
```

`wxcloudrun/` 里的源码是仓库根那几份的副本（母版 PNG、`arrow_wm.py`、HTTP 服务）。CI 要求两边一致。镜像里只装 `services/decode_api/requirements.txt`（numpy、opencv-python-headless、fastapi、uvicorn、python-multipart），不装 matplotlib。启动命令是单 worker 的 uvicorn，监听 `0.0.0.0:${PORT:-8501}`。

同一进程内解码是串行的。不要把 `--workers` 加大来赌吞吐：每个进程都会再加载一套模板。
