# 小程序接入 ARROW 解码

HTTP 服务实现在 `services/decode_api/`。默认载体是边缘水印 `edge`。生产判定来自 `arrow_decode.DecodeResult`：只有 `accepted === true` 时 `digits` 才是可信号码。

TI-ONE 上怎么部署、以及为什么要新建一条模型服务而不是改已有服务，见 `services/decode_api/TI-ONE.md`。摘要：账号里其他在线服务不要动；小程序 `direct-decoder.local.js` 的 url 指到 **新的** `ms-<id>/decode`。现有陶瓷解码地址是：

```text
https://ms-2w5g828w-100050884192.gw.ap-shanghai.ti.tencentcs.com/ms-2w5g828w/decode
```

网关前缀 `/ms-2w5g828w` 后面的 `/decode` 才是应用路由。新服务换成新的 id，路径仍以 `/decode` 结尾。

## 请求

`POST /decode`，`Content-Type: multipart/form-data`。

| 位置 | 名字 | 必填 | 说明 |
|---|---|---|---|
| 文件字段 | `image` | 是 | 照片。也接受 `file`、`img`、`photo`、`picture`，或其他任意一个文件字段。小程序 `wx.uploadFile` 的 `name` 请用 `image` |
| 查询或表单 | `carrier` | 否 | `edge`（默认）或 `inner`。查询参数优先于表单字段。不要传 `both` |
| 头 | `X-Decode-Token` | 仅当服务设置了环境变量 `DECODE_TOKEN` | 与变量值一致。未设置该变量时不要带头，服务不鉴权 |

微信侧用 `wx.uploadFile`：`url` 为网关完整地址（含 `/ms-<id>/decode`），`name` 为 `image`，`formData` 可带 `carrier: "edge"`（不带则服务端默认边缘）。若控制台另开了 TI-ONE 服务鉴权，还要按平台要求带 `Authorization`，那是网关的头，不是 `X-Decode-Token`。

图片请控制在 20 MB 以内。空文件、缺字段、`carrier` 不是 `edge`/`inner`、或正文不是 multipart，返回 400，`{"detail":"..."}`。令牌不对返回 401，`{"detail":"unauthorized"}`。过大返回 413。这些不是解码结果，小程序按网络错误处理即可。

## 响应

解码本身（包括「这张图没有可信号码」）是 **HTTP 200**，正文是 `DecodeResult.public_dict()` 再加一个中文 `reason_text`。没有 `raw`，也没有旧接口的 `detected` / `ecc_valid` / `crc_valid` / `business_id`。

| 字段 | 类型 | 含义 |
|---|---|---|
| `accepted` | bool | **唯一的采纳位**。`true` 才可以展示号码 |
| `digits` | string 或 null | 采纳时的 11 位号码。拒绝时一定是 `null` |
| `candidate_digits` | string 或 null | 研究解码器的原始读数，仅供排查。**禁止当作手机号展示或落库** |
| `carrier` | string | 实际使用的载体：`edge` 或 `inner` |
| `margin` | number | 码字裕度。采纳线默认 ≥ 1.8 |
| `cc` | number | 配准相关。采纳线默认 ≥ 0.85 |
| `logo_scale` | number 或 null | 相对母版的缩放 |
| `mean_conf` | number | 研究解码器的平均置信度。不要用它判断有没有水印 |
| `payload_valid` | bool | 读数是不是合法 11 位数字 |
| `ok` | bool | 研究解码器是否跑完一次同步。可以为 `true` 同时 `accepted` 为 `false`（例如无码图）。**不要把 `ok` 当成旧字段 `detected`** |
| `reason` | string | 机器可读原因，逗号分隔。采纳时为 `""` |
| `reason_text` | string | 给界面的中文。采纳时为「接受」，否则是原因的中文，多项用顿号连接 |
| `margin_threshold` | number | 本次使用的裕度门，默认 1.8 |
| `cc_threshold` | number 或 null | 本次使用的配准门，默认 0.85 |

`reason` 与 `reason_text` 的对应（未知代码会原样出现在 `reason_text` 里）：

| `reason` 中的代码 | `reason_text` |
|---|---|
| `low_margin` | 码字裕度不足 |
| `low_cc` | 配准 cc 不足 |
| `invalid_payload` | 码字越界 |
| `unreadable` | 读不到图像 |
| `sync_failed` | 同步失败 |
| `decode_failed` | 解码失败 |
| `rectify_failed` | 矫正失败 |

多项同时出现时按 `reason` 里的顺序用「，」接起来，例如 `low_cc,low_margin` → `配准 cc 不足，码字裕度不足`。

采纳示例（`samples/edge/ARROW_13800138000.png`，默认 `edge`）：

```json
{
  "accepted": true,
  "digits": "13800138000",
  "candidate_digits": "13800138000",
  "carrier": "edge",
  "margin": 5.290382769170719,
  "cc": 0.9985074655284963,
  "logo_scale": 0.851334810256958,
  "mean_conf": 0.038704290986061096,
  "payload_valid": true,
  "ok": true,
  "reason": "",
  "reason_text": "接受",
  "margin_threshold": 1.8,
  "cc_threshold": 0.85
}
```

拒绝示例（`samples/edge/ARROW_无码对照.png`）。研究解码器仍吐出了一个 11 位数，但裕度不够，所以 `digits` 是 `null`。界面如果误用 `candidate_digits`，就会把 `01074536704` 当成号码：

```json
{
  "accepted": false,
  "digits": null,
  "candidate_digits": "01074536704",
  "carrier": "edge",
  "margin": 0.9603219536756346,
  "cc": 0.9986710686975407,
  "logo_scale": 0.8512288331985474,
  "mean_conf": 0.00429149204865098,
  "payload_valid": true,
  "ok": true,
  "reason": "low_margin",
  "reason_text": "码字裕度不足",
  "margin_threshold": 1.8,
  "cc_threshold": 0.85
}
```

## 从旧字段迁到新界面

旧陶瓷解码结果里常见的是 `detected`、`ecc_valid`、`crc_valid`、`business_id` 等。新 JSON **没有这些键**。界面只做下面这一件事：

- `accepted` 为真：展示 `digits`，当作业务号码（原来 `business_id` 的位置）。
- `accepted` 为假：不要展示号码。提示用 `reason_text`（没有的话再退回 `reason`）。不要读 `candidate_digits`。

建议的对应关系，避免把旧判断原样套上来：

| 旧字段 | 新结果里怎么处理 |
|---|---|
| `business_id` | 仅当 `accepted` 时取 `digits`。拒绝时视为没有号码，而不是改去读 `candidate_digits` |
| `detected` | 不要映射到 `ok`。`ok` 只表示同步过程跑完了，无码图也会是 `true`。用户看得见的「识别到了」应等于 `accepted` |
| `ecc_valid` | 不再单列。配准没过门时 `accepted` 为假，`reason` 含 `low_cc` |
| `crc_valid` | 不再单列。码字不合法时 `accepted` 为假，`reason` 含 `invalid_payload`（或其他失败代码） |

伪代码：

```javascript
const body = JSON.parse(res.data);
if (body.accepted && body.digits) {
  showPhone(body.digits);
} else {
  showReject(body.reason_text || body.reason || "解码失败");
  // 不要使用 body.candidate_digits
}
```

默认不要在界面上提供载体开关；服务端默认就是 `edge`。只有拍的是内孔印版时才传 `carrier=inner`。
