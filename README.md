# ARROW 隐形水印 · Sub-pixel Geometric Watermark

> 把 11 位号码写进 Logo 笔画的**亚像素宽度**里，用手机拍照就能盲解。
> 信息不写在灰度上，写在几何尺寸上——所以打印、油墨扩散、曝光、反光都拿它没办法。

![同一号码的三版对照](samples/figures/14_same_payload_3way.png)

*第 1–3 行：原图 / 边缘载体 / 内孔载体（同一号码 `13800138000`）。第 4–5 行：笔画边缘与内孔的 6× 放大。*

---

## 一句话原理

把号码编码成 70 bit，再用**相邻两个码元一宽一窄**的方式，写进笔画边缘的亚像素位移里。
解码时靠字母自身的几何形状把照片掰正，逐行量出亚像素宽度差，再解回号码。

整个方案**没有任何定位标记**——图上只有 ARROW 五个字母。

## 巧在哪

**① 写在几何上，不写在灰度上。**
打印-拍摄链路里灰度会被油墨扩散、曝光、白平衡、釉面反光彻底改变；但"这条笔画有多宽"相对稳定。这决定了方案能否在实物上活下来。

**② 写在"宽度"上，不写在"位置"上。**
位置需要绝对参考系，一拍歪就全废；宽度是**局部差分**，不需要参考系。这是能盲解的根本原因。

**③ 差分对抵消一切慢变量。**
相邻码元一加宽一收窄，解码时相减。油墨整体扩散、光照梯度、笔画本身的粗细漂移都是共模的，相减就没了。调制幅度只有 ±1.8 px（笔画宽 71 px 的 2.5%），差分后有 5.6 px 的净信号。

**④ 50% 交叉点对模糊免疫。**
高斯模糊是对称的：边缘过渡变宽，但 50% 高度处的横坐标**完全不动**。所以用半高宽而不是"第一个暗像素"，失焦几乎不影响测量。

**⑤ 嵌入与检测共用同一基函数。**
嵌入是 `d·sin(πt)`，解码就用同一个 `sin(πt)` 做投影——匹配滤波，加性噪声下是最大似然最优。正弦窗两端为零，相邻码元无缝衔接。

**⑥ 几何基准来自字形自身。**
五字母质心 / 凸包四角 / 二阶矩仿射 / 墨迹包围盒 / 内孔质心 / SIFT 特征点，六类候选一起用 NCC 打分择优，再由 ECC 亚像素精配收尾。

## 参数

| 项目 | 值 |
|---|---|
| 载体 | 白底深灰字单色「ARROW」无衬线粗体，2200 × 600 px（= 2.2 cm × 0.6 cm，1000 px/cm 制版） |
| 载荷 | 任意 11 位十进制数字（≤ 2³⁷，37 bit） |
| 纠错 | Hamming(7,4)，37 → 70 bit |
| 调制 | 差分宽度调制，140 码元 / 70 差分对，边沿正弦偏移 ±1.8 px（±0.018 mm） |
| 解码 | SIFT+RANSAC 定位 → 单应矫正 → ECC 精配 → 匹配滤波半高宽 → 软判决最大似然 |
| 依赖 | numpy / opencv-python / matplotlib / pillow |

## 实测性能

仿真器模拟陶瓷拍摄链路（透视 + 离焦 + 光照不均 + 釉面高光 + 噪点 + JPEG），**50 组随机号码 × 5 档扰动**：

| 扰动档位 | 完整号码正确率 | 原始误码率（纠错前） |
|---|---|---|
| 无扰动 | **100%** | 0.00% |
| 轻度 | **100%** | 0.06% |
| 中度 | **100%** | 0.26% |
| 重度 | **100%** | 1.06% |
| 极限 | **74%** | 7.49% |

- 单次嵌入约 35–60 ms，单次盲解码 0.5–2.5 s（视是否触发粗搜抢救）
- 打印分辨率不是瓶颈：等效 635 dpi 与 2540 dpi 的结果完全相同（调制是 18 px 周期的平滑正弦）
- 极限档的失败经真值单应对照确认：约一半是配准、一半是信号不足

## 快速开始

```bash
pip install -r requirements.txt
```

`requirements.txt` 使用 `opencv-python-headless`（CI 与服务器无图形界面时用这个）。本机若已装带 GUI 的 `opencv-python`，不必再装 headless，二者都提供 `cv2`。

**生成印版**

```bash
# 单张，边缘载体，输出到当前目录
python gen_plate.py 13800138000

# 内孔载体
python gen_plate.py 13800138000 --carrier inner -o out.png

# 批量，两版都出，附带 1bit 二值版
python gen_plate.py 13800138000 13912345678 18600001234 --carrier both -d out --1bit

# 号码写在文本文件里（每行一个，# 开头为注释）
python gen_plate.py --file numbers.txt --carrier edge -d out
```

生成后会自动解一遍并打印 OK / FAIL，不用再单独验。

**解码照片**

```bash
# 边缘载体（可一次给多张）
python arrow_wm.py --decode photo1.jpg photo2.jpg

# 内孔载体
python arrow_inner.py --decode photo1.jpg

# 单张 + 核验图（输入/检测框、矫正图、70 bit 逐比特置信度）
python verify_decode.py photo.jpg --digits 13800138000

# 批量扫目录，出表格 + CSV
python batch_decode.py photos/ --csv result.csv --save-figs
```

**库接口（生产解码）**

命令行和 `from arrow_decode import decode` 走同一道门。默认载体是**边缘**；内孔要显式选择，本接口不改内孔算法，也不抬极限档成功率。

```python
from arrow_decode import decode

result = decode(image)                    # carrier="edge"
# result = decode(image, carrier="inner")
# result = decode("photo.jpg")            # 路径也可以

if result.accepted:
    phone = result.digits                 # 仅此时是可信号码
    # result.margin / result.cc / result.logo_scale / result.mean_conf
else:
    phone = None                          # result.digits 一定是 None
    # result.candidate_digits 是研究解码器的原始读数，不可当号码使用
    # result.reason：low_margin / low_cc / invalid_payload / sync_failed / ...
```

接受条件（两个门同时过，外加码字本身是 11 位合法号码）：

| 门 | 默认 | 依据 |
|---|---|---|
| 码字裕度 `margin` | ≥ 1.8 | 真码 ≥ 2.0，噪声/无码 ≤ 1.4 |
| 配准 `cc` | ≥ 0.85 | 与解码器内部「这条路径能不能信」的门相同；低于此值时置信度可能虚高 |

任一不过：`accepted=False`，`digits=None`。批量 CSV 的「号码」列同样留空。研究压测（`arrow_wm.py --bench`、`arrow_inner.py --bench`）仍直接调用原来的解码器，**不经过这道门**，已发表的成功率数字不变。`arrow_wm.py --no-ecc` 不算配准分，这时只保留裕度门。

本地自检（也是 CI 在跑的那条）：

```bash
python ci_smoke.py
```

**纯 JS 边缘解码（微信小程序，只覆盖已对齐的干净印版）**

`js_decode/` 把边缘载体的生产解码收成两个 CommonJS 文件（`arrow_edge.js` + `layout.js`）：ES5、无 DOM、无 Node API、无 OpenCV / WASM。半高宽、差分、Hamming 软判决、裕度和接受门与 Python 相同。同步只认 **2200×600** 灰度画布（`gen_plate.py` 写出的那张）；`cc` 是 1/4 分辨率上与未加水印模板的归一化相关，**不是** `findTransformECC`。没有 SIFT、单应或 ECC。

```bash
python js_decode/compare_accuracy.py
```

需要本机有 Node。脚本用 `arrow_wm` 生成干净边缘印版（并经 `gen_plate.py` 走一遍 PNG），分别用 `arrow_decode` 和 JS 解码。

12 个固定号码（含 `13800138000`）的干净印版：**Python 接受时 JS 也接受，号码 12/12 一致**。同一张未做 ECC 的印版上，JS 宽度特征与 Python `extract_widths` 的最大绝对差是 **3.0×10⁻⁷**。相对走完 ECC 的生产结果，裕度绝对差最大 **0.029**（均值 0.008；真码裕度约 5.2–10.6，门限 1.8），`cc` 绝对差最大 **0.0013**（两边都在 0.997 附近）。无码印版两边都因 `low_margin` 拒绝，`digits` 为 `null`（`candidate_digits` 只是诊断，不能当号码）。

**不要把它接成拍照盲解。** 干净印版在画布内平移 15 px：Python 仍解出原号（裕度 5.29，cc 0.997），JS 拒绝（裕度 0.88，cc 0.81，`digits` 为 `null`）。实拍还要透视、缩放、滚转和 ECC 亚像素精配。照片若还没被矫正回 2200×600，继续走云端 Python。母版或布局变了之后，用 `python js_decode/export_layout.py` 重导 `layout.js`。

**验真模式**（照片糊到盲解失败、但你知道嫌疑号码时）

```bash
python verify_decode.py photo.jpg --suspect 13800138000
```

不做纠错，只算测量结果与目标码字的相关性，输出 z 值。实测标定：真号码 5.0–8.1σ，假号码最高 3.8σ，门限取 4.5σ。

**压测**

```bash
python arrow_wm.py --bench -n 50 --seed 6000
```

## 两种载体

| | 边缘载体 `arrow_wm.py` | 内孔载体 `arrow_inner.py` |
|---|---|---|
| 调制位置 | 外轮廓 | 字母内部的孔（O 内环、两个 R 字怀、A 上三角） |
| 外轮廓 | 有细微波浪 | **逐像素与原图相同** |
| 重度 / 极限成功率 | **100% / 74%** | 90% / 35% |
| 适用 | 要求最高解码率 | 客户质检要求"外轮廓和设计稿一模一样" |

![载体对比](samples/figures/12_carrier_cmp.png)

*第 4 行是 R 竖干左边缘的亚像素位置随行号变化：原图与内孔版两条线精确重合，边缘版在 509.66–513.32 之间摆动（峰峰 3.66 px）。*

## 目录

```
arrow_wm.py            边缘载体核心（嵌入 / 解码 / 仿真 / 压测）
arrow_inner.py         内孔载体核心
arrow_decode.py        生产解码 API（边缘默认，内孔可选，裕度 + cc 硬门）
js_decode/             边缘载体的纯 JS 干净印版解码 + 与 Python 的对照脚本
gen_plate.py           生成印版（单张 / 批量 / 两种载体）
verify_decode.py       单张解码 + 核验图 + 验真模式
batch_decode.py        批量解码（扫目录 / CSV / 批量出图）
ci_smoke.py            嵌入/解码冒烟 + 接受门单测（CI 入口）
requirements.txt       运行与 CI 依赖
.github/workflows/ci.yml  安装依赖并跑 ci_smoke.py
make_test_samples.py   生成 10 组随机号码实测样张 + A4 打印排版图
compare_carrier.py     两种载体的定量对比
compare_same_payload.py 同一号码的三版对照图
sample_gallery.py      样张总览图
export_print_plates.py 批量导出送印图
ablation_interior.py   内孔载体消融实验
ARROW_clean_pro.png    字形母版（必需，不能删）
docs/技术报告.md        完整技术报告（原理、实验、踩坑记录）
samples/               样张与插图
```

## 已知局限

- **容量由边缘长度决定**，不是面积。2200 px 宽的 Logo 有效边缘约 2000 px，70 bit 就是每 bit 28 px。想多写只能加大调制幅度（视觉更明显）或换更大的载体。
- **极限档 74%** 是物理信噪比的边界。其中约一半失败来自配准、一半来自信号不足；后者受限于调制幅度，加大就破坏隐蔽性。
- **拍摄建议**：距离 8–15 cm，让 Logo 占画面 1/3 以上，避开强反光（最主要的干扰源）。倾斜 30° 以内能自动矫正。低于原图 0.2 倍缩放时信号会不足。
- **研究解码器对无码图仍会吐出一个号码**——判决器总要给出 0/1。生产入口 `arrow_decode.decode` 用硬门把它拒掉：码字裕度低于 1.8，或配准 cc 低于 0.85，或码字不是合法 11 位号码时，`accepted=False` 且 `digits=None`。真码裕度 ≥ 2.0，噪声/无码 ≤ 1.4。不要用 `mean|conf|` 做有无水印的判据（误报可达 40% 以上）；内孔命令行以前的 0.012 置信度提示已改成与边缘相同的裕度 + cc 门。
- **`js_decode/` 只解已经对齐的 2200×600 干净印版。** 平移 15 px 就会和 Python 拉开（Python 仍接受，JS 拒绝）。手机照片的同步和 ECC 不在这个模块里。

## 理论出处

没有一篇"本方案出自某某论文"——这是一套针对具体场景拼出来的工程组合。各部分的理论源头：

- 用字形**几何变形**承载信息：Brassil, Low, Maxemchuk, O'Gorman, *Electronic marking and identification techniques to discourage document copying*, IEEE JSAC 1995
- 几何同步是打印-拍摄水印的核心难题：O'Ruanaidh & Pun, ICIP 1997；Kutter, SPIE 1998
- SIFT：Lowe, *Distinctive image features from scale-invariant keypoints*, IJCV 2004
- 扩频水印（对照，本方案**不走**这条路）：Cox et al., IEEE TIP 1997

自行设计并实现的部分：差分对 + 正弦窗 + 亚像素半宽的调制与检测组合、140 码元布局从字形掩膜自动推导、六类自同步候选池 + NCC 择优、内孔载体、软判决 ML 替代硬判决 Hamming。

## License

未附开源许可，保留所有权利。商业使用请联系作者。
