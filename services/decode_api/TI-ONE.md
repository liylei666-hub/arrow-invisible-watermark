# TI-ONE 挂载说明

这份说明只覆盖 **ARROW 解码** 这一条在线服务。账号里已经在跑的其他 TI-ONE 模型服务 / 在线服务 **不要改**：不要换它们的镜像，不要改它们的端口、环境变量、启动命令、健康检查、资源组、网关鉴权或调用地址。

现有小程序直连的是陶瓷解码那条，形状为：

```text
https://ms-2w5g828w-100050884192.gw.ap-shanghai.ti.tencentcs.com/ms-2w5g828w/decode
```

公网网关的规则是：主机名和路径前缀都带着服务 id（`ms-<id>`），前缀后面的才是容器里的应用路径。上面这个 URL 里，网关前缀是 `/ms-2w5g828w`，应用路由是 `POST /decode`。

本服务在容器内只注册两条路由：

| 方法 | 容器路径 | 作用 |
|---|---|---|
| `POST` | `/decode` | 解码。multipart 字段 `image`，可选 `carrier`（默认 `edge`） |
| `GET` | `/health` | `{"status":"ok"}`。不校验 `X-Decode-Token` |

公网调用时把前缀接回去即可：

```text
客户端   POST https://ms-<id>-<uin>.gw.<region>.ti.tencentcs.com/ms-<id>/decode
网关     路径上的 /ms-<id> 是服务路由；转给容器的是剩余路径 /decode
容器     FastAPI  POST /decode     端口 8501
```

健康检查同理：`GET https://<主机>/ms-<id>/health` → 容器 `GET /health`。

官方文档里，公网网关一般会在进容器前去掉这一个 `/ms-<id>` 前缀（调用示例都是 `${SERVER_URL}/v1/...`，而 `SERVER_URL` 已经含 `/ms-<id>`）。少数链路（例如 CLB 直连示例）会把 `/ms-<id>` 再带进容器，甚至带两次。本服务若看到路径以 `/ms-<id>` 开头，会把这些前缀段剥掉再进 `/decode` 和 `/health`，所以两种转发方式都能打到同一路由。应用本身不要再挂一层 `/ms-...` 前缀。

新建服务后，**以控制台「服务调用」页展示的公网地址为准**，不要手改主机名里的 uin 或地域。小程序里的 `direct-decoder.local.js` 只改这一条 ARROW 的 url。

## 端口：8501

依据是 TI-ONE 在线服务文档（产品文档「在线服务部署」「在线服务相关」）：

- 容器端口可配置，范围 1024–65535，**排除** 8502–8510、6006、9092。
- **8500 是 gRPC 默认端口，8501 是 REST 默认端口**，不能混用。
- 自定义镜像若监听端口和平台不一致，服务会一直「等待中」：平台健康检查默认打 **8501**。处理办法是让进程听 8501，或把控制台端口改成进程实际端口。两者必须相同。

因此镜像 `EXPOSE 8501`，默认 `PORT=8501`。创建服务时：

- **端口**填 `8501`
- 若必须沿用某个已有端口（且该端口允许使用），启动前设环境变量 `PORT` 为同一个值，并在控制台「端口」填同一个值。不要为了迁就别的服务去改那些服务。

健康检查（新建服务的表单里单独配，不要改别人的探针）：

| 项 | 值 |
|---|---|
| 检测方式 | HTTP GET |
| 调用路径 | `/health` |
| 端口 | `8501`（与「端口」一致） |
| 等待服务启动的延时 | `120` 秒。边缘 + 内孔模板加载完之前进程不接受连接；日志里出现 `[decode_api] ready` 才算好。若探针在 ready 之前失败，把延时加大，不要改算法 |
| 轮询间隔 / 超时 / 失败次数 | 存活探针可按平台示例：间隔 10 秒、超时 5 秒、失败 3 次。就绪探针可以更勤 |

`/health` 不读 `DECODE_TOKEN`，探针不用带业务密钥。

## 推荐做法：新建一条模型服务

这是满足「别的服务已经在 TI-ONE 上，不能动」的做法。

1. 用本目录的 Dockerfile，在**仓库根**构建镜像并推到本账号的 TCR。镜像是自包含的（代码和 `ARROW_clean_pro.png` 都在里面），模型来源选 **容器镜像**。不要选其他服务正在用的 CFS / COS / GooseFSx 路径，也不要往 `/data/model` 放代码：平台在选择文件型模型来源时会把模型挂到 `/data/model` 并覆盖那里的内容。
2. **新建**一条在线服务，拿到新的服务 id（`ms-<新id>`）。不要点进 `ms-2w5g828w` 或其他已有服务的「更新」。
3. 运行环境选这块新镜像。启动命令留空，用镜像自带的 uvicorn（单 worker，`--host 0.0.0.0 --port 8501`）。CPU 规格即可，这个解码不用 GPU。资源从空闲量里新申请，不要把别的服务的实例缩掉来腾卡。
4. 环境变量只加在**这条新服务**上。需要应用层共享密钥时设置 `DECODE_TOKEN`（值不要进 git、不要打进镜像层）。不设置则 `/decode` 不对这个头做校验。
5. 等 `/health` 就绪后，用控制台给出的公网地址试：

```bash
curl -s "https://ms-<新id>-<uin>.gw.ap-shanghai.ti.tencentcs.com/ms-<新id>/health"

curl -s -F "image=@samples/edge/ARROW_13800138000.png" \
  "https://ms-<新id>-<uin>.gw.ap-shanghai.ti.tencentcs.com/ms-<新id>/decode"
```

6. 只改小程序 `direct-decoder.local.js` 里 ARROW 解码的 url，指向 `https://ms-<新id>-<uin>.gw.<region>.ti.tencentcs.com/ms-<新id>/decode`。微信公众平台把这个新主机名加进 request 合法域名。旧的 `ms-2w5g828w` 域名如果还有别的页面在用，留着。
7. `ms-2w5g828w` 以及任何非 ARROW 的服务 id 保持原镜像、原路径、原配置。

这样陶瓷解码（或别的业务）继续走原来的 MS，ARROW 走新的 MS。两边都是 `POST .../decode`，只是 `ms-<id>` 不同。

## 若必须沿用现在的 MS id

只有在明确要让 **这一条** 现有调用地址改成 ARROW 解码时才这么做。仍然只动这一条服务，不动其他 id。

- 服务 id 仍是 `ms-2w5g828w`（或控制台里确认「当前确实是陶瓷解码」的那一个 id，不要凭记忆改到旁边的服务上）。
- 只替换 **该服务** 的部署镜像 / 制品为本 FastAPI 镜像。容器继续提供 `POST /decode` 和 `GET /health`，端口按上一节对齐。
- 小程序 url 可以继续是  
  `https://ms-2w5g828w-100050884192.gw.ap-shanghai.ti.tencentcs.com/ms-2w5g828w/decode`  
  网关前缀不变，容器内仍是 `/decode`。
- 换镜像之后，这条 URL 不再做原来的陶瓷解码。调用方要按 [小程序契约](../../docs/miniprogram-integration.md) 改字段：成功只认 `accepted === true`（或 `status === "accepted"`），拒绝时展示 `reason_text` 并标明结果不可信。不要用 `ok`，不要用 `candidate_digits`（默认响应里没有这个字段）。
- 不要在这次替换里改其他服务的路由、镜像或资源。

## 平台侧和本服务各管一层

| 层 | 谁配置 | 说明 |
|---|---|---|
| 公网网关、服务 id、端口、探针 | TI-ONE 控制台，只针对这一条新服务（或上节那一个旧 id） | 密钥用控制台「服务鉴权」，头一般是 `Authorization` |
| `POST /decode` 的可选共享密钥 | 该服务的环境变量 `DECODE_TOKEN` | 请求头 `X-Decode-Token`。未设置则不启用 |
| 号码能不能信 | `arrow_decode` 的接受门 | 只认 `accepted===true` / `status==="accepted"`。HTTP 200 且 `status=rejected` 是一次正常的拒绝，不是网关故障。400/401/413 的 `status=error` 才是请求错误 |

控制台自带的「在线测试」偏向 JSON 请求体，不适合 multipart 图片。用上面的 `curl -F` 或小程序 `wx.uploadFile`（`name` 填 `image`）来验。

## 明确不要做的事

- 不要更新、重启、改端口或改镜像到 **ARROW 这一条之外** 的在线服务。
- 不要把新镜像推上去之后，在服务列表里批量滚动更新。
- 不要复用别的服务的 CLB 监听或共享路径来「顺便」挂 `/decode`。这条服务用它自己的公网网关地址。
- 不要把 `DECODE_TOKEN`、TCR 密码、TI-ONE 鉴权密钥写进仓库。
