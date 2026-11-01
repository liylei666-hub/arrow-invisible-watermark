# 小程序接入 ARROW 解码

HTTP 服务实现在 `services/decode_api/`。默认载体是内孔水印 `inner`。旧边缘印版传 `carrier=edge`。生产判定来自 `arrow_decode.DecodeResult`。

**成功只有一种写法：`accepted === true`，等价于 `status === "accepted"`。** 不要用 `ok`。不要用 `digits || candidate_digits`。`ok === true` 只说明研究解码器做完了一次同步，无码图也会这样。`candidate_digits` 默认不在 JSON 里。

TI-ONE 上怎么部署、以及为什么要新建一条模型服务而不是改已有服务，见 `services/decode_api/TI-ONE.md`。摘要：账号里其他在线服务不要动；小程序 `direct-decoder.local.js` 的 url 指到 **新的** `ms-<id>/decode`。现有陶瓷解码地址是：

```text
https://ms-2w5g828w-100050884192.gw.ap-shanghai.ti.tencentcs.com/ms-2w5g828w/decode
```

网关前缀 `/ms-2w5g828w` 后面的 `/decode` 才是应用路由。新服务换成新的 id，路径仍以 `/decode` 结尾。

微信云托管上的控制台参数和 `wx.cloud.callContainer` 示例在 [wechat-cloudrun.md](wechat-cloudrun.md)。服务名 `arrow-decode`，头 `X-WX-SERVICE: arrow-decode`，路径仍是 `/decode`。

## 请求

`POST /decode`。原来的 `multipart/form-data` 不变。另外也接受 JSON `image_base64`，以及 `application/octet-stream` / `image/jpeg` / `image/png` 的原始图片字节（`carrier` 放查询参数）。`callContainer` 默认发 JSON，组 multipart 不可靠，照片优先用原始字节。细节和 100 KiB 文本上限见 [wechat-cloudrun.md](wechat-cloudrun.md)。

multipart 的约定：

| 位置 | 名字 | 必填 | 说明 |
|---|---|---|---|
| 文件字段 | `image` | 是 | 照片。也接受 `file`、`img`、`photo`、`picture`，或其他任意一个文件字段。小程序 `wx.uploadFile` 的 `name` 请用 `image` |
| 查询或表单 | `carrier` | 否 | `inner`（默认）或 `edge`。查询参数优先于表单字段。不要传 `both`。旧边缘印版才传 `edge` |
| 头 | `X-Decode-Token` | 仅当服务设置了环境变量 `DECODE_TOKEN` | 与变量值一致。未设置该变量时不要带头，服务不鉴权 |

微信侧用 `wx.uploadFile`：`url` 为网关完整地址（含 `/ms-<id>/decode`），`name` 为 `image`，`formData` 可带 `carrier: "inner"`（不带则服务端默认内孔）。旧边缘印版才传 `carrier: "edge"`。若控制台另开了 TI-ONE 服务鉴权，还要按平台要求带 `Authorization`，那是网关的头，不是 `X-Decode-Token`。

图片请控制在 20 MB 以内。空文件、缺字段、`carrier` 不是 `edge`/`inner`、或正文类型无法识别，返回 400，`{"status":"error","detail":"..."}`。令牌不对返回 401，`{"status":"error","detail":"unauthorized"}`。过大返回 413，同样是 `status:"error"`。这些不是解码结果，小程序按网络错误处理即可。`GET /health` 的 `{"status":"ok"}` 只表示进程活着，不是解码成功。

## 响应

解码本身（包括「这张图没有可信号码」）是 **HTTP 200**。正文里 `status` 只能是 `accepted` 或 `rejected`。请求不合法或服务内部失败时才是 `status=error`，并且 HTTP 不是 200。没有 `raw`，也没有旧接口的 `detected` / `ecc_valid` / `crc_valid` / `business_id`。默认没有 `candidate_digits`。

| 字段 | 类型 | 含义 |
|---|---|---|
| `status` | string | `accepted`、`rejected` 或（非 200）`error`。成功即 `status === "accepted"` |
| `accepted` | bool | 与 `status === "accepted"` 同一件事。`true` 才可以展示号码 |
| `digits` | string 或 null | 采纳时的 11 位号码。拒绝时一定是 `null` |
| `candidate_digits` | 默认不返回 | 只有查询参数 `debug=1` **并且**服务配置了 `DECODE_TOKEN`、请求带对了 `X-Decode-Token` 时才出现。出现了也禁止当号码展示或落库 |
| `carrier` | string | 实际使用的载体：`edge` 或 `inner` |
| `margin` / `margin_mean` | number | 十组码字间隙的平均。采纳线默认 ≥ 1.8。这是有码/无码门 |
| `margin_min` | number | 最弱一组的间隙。采纳线默认 ≥ 0.55。均值高但这一组打平时会拒绝 |
| `cc` | number | 配准相关。采纳线默认 ≥ 0.85 |
| `logo_scale` | number 或 null | 相对母版的缩放 |
| `mean_conf` | number | 研究解码器的平均置信度。不要用它判断有没有水印 |
| `payload_valid` | bool | 读数是不是合法 11 位数字 |
| `ok` | bool | 研究解码器是否跑完一次同步。可以为 `true` 同时 `status` 为 `rejected`。**不是成功位** |
| `reason` | string | 机器可读原因，逗号分隔。采纳时为 `""` |
| `reason_text` | string | 给界面的中文。采纳时为「接受」，否则是原因的中文，多项用顿号连接 |
| `margin_threshold` | number | 本次使用的均值门，默认 1.8 |
| `min_group_margin_threshold` | number | 本次使用的单组最小门，默认 0.55 |
| `cc_threshold` | number 或 null | 本次使用的配准门，默认 0.85 |

`reason` 与 `reason_text` 的对应（未知代码会原样出现在 `reason_text` 里）：

| `reason` 中的代码 | `reason_text` |
|---|---|
| `low_margin` | 码字裕度不足 |
| `low_group_margin` | 单组裕度不足 |
| `pad_mismatch` | 补零不符 |
| `low_cc` | 配准 cc 不足 |
| `low_logo_scale` | 画面中 Logo 过小 |
| `invalid_payload` | 码字越界 |
| `unreadable` | 读不到图像 |
| `sync_failed` | 同步失败 |
| `decode_failed` | 解码失败 |
| `rectify_failed` | 矫正失败 |

多项同时出现时按 `reason` 里的顺序用「，」接起来，例如 `low_cc,low_margin` → `配准 cc 不足，码字裕度不足`。

采纳示例（默认 `inner`，例如 `samples/inner/ARROW_13800138000.png`）：

```json
{
  "accepted": true,
  "status": "accepted",
  "digits": "13800138000",
  "carrier": "inner",
  "margin": 5.7,
  "margin_mean": 5.7,
  "margin_min": 5.2,
  "cc": 0.99,
  "logo_scale": 1.0,
  "mean_conf": 0.04,
  "payload_valid": true,
  "ok": true,
  "reason": "",
  "reason_text": "接受",
  "margin_threshold": 1.8,
  "min_group_margin_threshold": 0.55,
  "cc_threshold": 0.85
}
```

拒绝示例（无码图，默认 `inner`）。HTTP 仍然是 200。`ok` 可以是 `true`，但 `status` 是 `rejected`，`digits` 是 `null`，响应里没有 `candidate_digits`。界面如果自己再去拼一个号码，就会把一次拒绝当成成功：

```json
{
  "accepted": false,
  "status": "rejected",
  "digits": null,
  "carrier": "inner",
  "margin": 0.96,
  "margin_mean": 0.96,
  "margin_min": 0.2,
  "cc": 0.99,
  "logo_scale": 1.0,
  "mean_conf": 0.004,
  "payload_valid": true,
  "ok": true,
  "reason": "low_margin,low_group_margin",
  "reason_text": "码字裕度不足，单组裕度不足",
  "margin_threshold": 1.8,
  "min_group_margin_threshold": 0.55,
  "cc_threshold": 0.85
}
```

## 从旧字段迁到新界面

旧陶瓷解码结果里常见的是 `detected`、`ecc_valid`、`crc_valid`、`business_id` 等。新 JSON **没有这些键**。界面只做下面这一件事：

- `accepted === true`（也就是 `status === "accepted"`）：展示 `digits`。
- 否则：不要展示号码。提示用 `reason_text`（没有的话再退回 `reason`）。不要读 `ok`，不要读 `candidate_digits`。

建议的对应关系，避免把旧判断原样套上来：

| 旧字段 | 新结果里怎么处理 |
|---|---|
| `business_id` | 仅当 `accepted === true` 时取 `digits`。拒绝时视为没有号码 |
| `detected` | 等于 `status === "accepted"`。不要映射到 `ok` |
| `ecc_valid` | 不再单列。配准没过门时 `accepted` 为假，`reason` 含 `low_cc` |
| `crc_valid` | 不再单列。码字不合法时 `accepted` 为假，`reason` 含 `invalid_payload`（或其他失败代码） |

伪代码：

```javascript
const body = JSON.parse(res.data);
// 成功只认 accepted===true（等价于 status==="accepted"）。
// 不要写 body.ok、不要写 body.digits || body.candidate_digits。
if (body.accepted === true && body.digits) {
  showPhone(body.digits);
} else {
  showReject(body.reason_text || body.reason || "未识别");
}
```

默认不要在界面上提供载体开关；服务端默认就是 `inner`。只有拍的是旧边缘印版时才传 `carrier=edge`。
