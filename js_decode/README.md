# 纯 JS 边缘解码（干净印版）

微信小程序可用的边缘载体解码：`arrow_edge.js` 与 `layout.js`。ES5、CommonJS（`require` / `module.exports`），不使用 DOM、Node 内置模块、OpenCV 或 WASM。

```javascript
var edge = require("./arrow_edge.js");
var result = edge.decodeCleanPlate(gray, 2200, 600);
if (result.accepted) {
  var phone = result.digits;          // 11 位，仅此时可信号码
} else {
  var phone = null;                   // result.digits 一定是 null
  // result.candidate_digits 不可当号码
  // result.reason: low_margin / low_cc / invalid_payload / sync_failed / ...
}
```

`gray` 是行优先的灰度数组（`Uint8Array` 或普通数组），长度 `2200 * 600`，取值 0–255，与 `cv2.IMREAD_GRAYSCALE` 同一平面。

接受门与 `arrow_decode.apply_gate` 相同：同步成功、11 位合法号码、`margin ≥ 1.8`、`cc ≥ 0.85`。任一不过则 `accepted=false` 且 `digits=null`。

## 对照 Python

在仓库根目录（需要 Python 依赖和 Node）：

```bash
python js_decode/compare_accuracy.py
```

脚本用 `arrow_wm` 的嵌入器生成干净边缘印版（`gen_plate.py` 用的就是它），并用 `gen_plate.py` 走一遍 PNG 回读。Python 侧是生产入口 `arrow_decode`（SIFT/几何同步 + ECC + 硬门）。JS 侧只吃已经对齐的画布。

12 个固定号码上的实测：

| | 结果 |
|---|---|
| Python 接受且 JS 号码一致 | **12/12** |
| 同一张未矫正印版，宽度特征最大绝对差 | **3.0×10⁻⁷** |
| 相对生产解码的裕度绝对差 | 最大 0.029，均值 0.008（真码约 5.2–10.6，门限 1.8） |
| 相对生产解码的 `cc` 绝对差 | 最大 0.0013（两边都在 0.997 附近） |
| 无码印版 | 两边都是 `low_margin`，`digits` 为 null |
| 画布内平移 15 px | Python 仍接受原号；JS 拒绝（`low_cc,low_margin`） |

## 这一版做了什么 / 没做什么

做了：载荷与 Hamming(7,4)、140 个码元上的半高宽和正弦匹配滤波、差分软判决、码字裕度、与生产入口相同的硬门。码元几何是 `arrow_wm.build_layout` 的预计算导出，不在小程序里重跑母版栅格化。

没做，实拍仍然需要：

- 在照片里找到 Logo（亮底卡片、杂乱背景）
- SIFT 或字形质心单应，把透视/缩放/滚转矫正回 2200×600
- ECC 亚像素精配（Python 的 `cc` 来自 `findTransformECC`）
- 小图放大后的边缘恢复

JS 的 `cc` 是画布相对未加水印模板的 1/4 分辨率归一化相关（4×4 盒式平均，与 OpenCV `INTER_AREA` 在这张母版上一致），字段 `cc_method` 为 `quarter_res_ncc`。`logo_scale` 是设计凸包宽度 / 2200（恒等单应），不是 ECC 之后的尺度。干净印版上这两个数和 Python 只差千分位，但平移 15 px 时 JS 的 `cc` 掉到 0.81，门会把它拒掉，而 Python 仍能解。

内孔载体不在这里。

## 改布局之后

```bash
python js_decode/export_layout.py
```

会重写 `layout.js`（140 个 cell，以及 1/4 分辨率模板的行程编码）。不要手改这个文件。
