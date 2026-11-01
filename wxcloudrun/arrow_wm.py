#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ARROW 陶瓷 Logo 隐形水印 —— 生成 / 物理信道扰动模拟 / 盲解码验证 闭环系统
================================================================================

场景
----
白底深灰字的单色 "ARROW" 无衬线粗体图案（物理尺寸 0.6cm x 2.2cm），
印在陶瓷釉面上，用手机微距拍摄后需要盲还原出 11 位十进制数字（如手机号）。

核心思想（差分宽度调制 / Differential Width Modulation）
--------------------------------------------------------
1. 载体是 2200 x 600 px 的高精度底版（等效 ~2540 dpi 制版）。字形母版直接取自
   设计稿（1bit PNG 内嵌，逐像素差异 0.002%），保证"原图"与设计稿完全一致。
2. 在字母笔画的"竖直类"区段（竖干、碗形侧壁、斜笔画的可分离段）内，
   沿纵向切出 140 个固定码元区域（Cell），两个相邻 Cell 组成一个"差分对"。
3. 每个差分对里，一个 Cell 的笔画被横向加宽 +d，另一个被收窄 -d。
       (宽, 窄) -> bit = 1
       (窄, 宽) -> bit = 0
   这就是 Manchester 思想的空域版本：信息承载在"相邻两格的相对差异"上，
   因此全局油墨浓淡（整体膨胀/收缩）、全局曝光、全局模糊都会同时作用于
   一对 Cell 的两侧，在差分中被抵消。
4. 解码端不测"墨量"，而测笔画的**半高宽**（亚像素）：高斯模糊是对称卷积，
   半高宽几乎不变；墨量积分则会被模糊后的窗口截断影响。实测在极限扰动下
   半高宽的 BER 比墨量积分低一个数量级。
5. 37 bit 载荷 -> Hamming(7,4) -> 70 bit -> 70 个差分对 -> 140 个 Cell。

四大模块
--------
  WatermarkEmbedder    : 底版绘制 + 布局推导 + 差分宽度调制嵌入
  CeramicCamSimulator  : 油墨扩散/透视/高光/失焦/噪声+JPEG 五级复合退化
  WatermarkDecoder     : 字形自同步 -> 单应矫正 -> ECC 精配 -> 匹配滤波半高宽 -> 纠错
  BenchmarkRunner      : 50 组随机号码 x 5 档扰动的 BER / 成功率压测

依赖: numpy, opencv-python, matplotlib, pillow, scipy（均为常规库，无 C 扩展自编译）
运行:
    python arrow_wm.py --all            # 演示 + 压测（默认）
    python arrow_wm.py --demo           # 只跑单条演示
    python arrow_wm.py --bench -n 50    # 只跑压测
================================================================================
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2

# matplotlib 只在出图时用。原先注释写着"延迟导入"，实际是模块级立即导入，
# 光 import 就要 0.3~0.5 s —— 纯解码/批量解码完全用不到它。改成真延迟。
_PLT = None


def _pyplot():
    """首次出图时才导入 matplotlib（约 0.4 s），解码路径完全不付这个代价。"""
    global _PLT
    if _PLT is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _PLT = plt
    return _PLT

# ==============================================================================
# 0. 全局常量
# ==============================================================================

CANVAS_W: int = 2200          # 底版宽（对应 2.2 cm）
CANVAS_H: int = 600           # 底版高（对应 0.6 cm）
SS: int = 4                   # 超采样倍率（4x 绘制 -> 面积平均降采样，获得 16 级抗锯齿）

INK_LEVEL: int = 58           # 深灰油墨（对应参考图 #3A3A3A 左右）
BG_LEVEL: int = 255           # 白瓷底

CAP_TOP: int = 108            # 字母上沿
CAP_BASE: int = 508           # 字母基线
CAP_H: int = CAP_BASE - CAP_TOP   # 字高 400 px

TEXT_X0: int = 100             # 文字带左边界
TEXT_X1: int = 2100           # 文字带右边界

ANCHOR_SIZE: int = 0          # 已彻底移除：白底上只保留纯 ARROW 字母，无任何黑点
# （保留常量占位以兼容旧接口；0 表示不绘制任何定位块）

N_BITS: int = 70              # 编码后比特数 = 10 组 Hamming(7,4)
N_CELLS: int = N_BITS * 2     # 每 bit 一个差分对 -> 140 个 Cell

WINDOW_POW: float = 1.0       # 调制窗形状：1 = sin(πt)（斜率不连续），2 = sin²(πt)（C1 连续）
MOD_DEPTH: float = 6.0        # 单侧边沿峰值偏移（px @ 底版尺度，约 0.060 mm）
# 与内孔载体 DEPTH 对齐。±0.060 mm 在 2.2 cm 的 logo 上仍低于约 0.1 mm 的
# 视距分辨；打印 1200 dpi 时约 2.8 个墨点。早期 1.8 px 的已发表压测数字
# 是布局 v1 的工作点，不是当前默认。
CELL_MARGIN: int = 10         # 解码测量窗相对笔画边缘的余量
# 测量窗伸出笔画边缘 d+CELL_MARGIN，对向调制再让出一个 d，并留 2px。
# d=6 时 max(20, ceil(2*6+10+2)) = 24。v1 印版不走这条，锁 min_clear=20。


def min_clearance_for_depth(d: float) -> int:
    """v2 净空下限，随调制深度推导。"""
    return max(20, int(math.ceil(2.0 * float(d) + CELL_MARGIN + 2)))


def min_stroke_w_for_depth(d: float) -> float:
    """v2 笔画宽度下限 = 4*d。d=6 时为 24px。"""
    return 4.0 * float(d)


def end_trim_for_depth(d: float) -> int:
    """v2 区段两端裁掉的行数 = ceil(d)。"""
    return int(math.ceil(float(d)))


MIN_CLEARANCE: int = min_clearance_for_depth(MOD_DEPTH)  # 24 @ d=6
MIN_CELL_H: int = 14
LAYOUT_VERSION: int = 2
V1_DEPTH: float = 1.8
V1_MIN_CLEARANCE: int = 20
PAIR_PERM_ID_V1: str = "v1-identity"
PAIR_PERM_ID_V2: str = "v2-snake-r1"
# 路径择优用的归一化软值门。depth=1.8 时 mean|conf|≈0.030 约是满调制的 0.75。
# 这里改看 soft_n，避免 6px 下沿用 1.8 时期的绝对 conf 门。
SOFT_N_PATH_GATE: float = 0.70

PAYLOAD_DIGITS: int = 11
PAYLOAD_BITS: int = 37        # 10^11 - 1 < 2^37

OUT_DIR: str = "arrow_wm_out"

# 攻击强度档位（数值均为"物理信道"参数）
#   ink      : 油墨扩散/烧制形变的形态学半径（px @ 底版尺度，正=膨胀 负=腐蚀）
#   tilt     : 相机绕水平轴倾角（度）
#   roll     : 绕光轴滚转（度）
#   spec     : 釉面高光强度（0~1，把像素往 255 拉的比例）
#   defocus  : 高斯失焦 sigma
#   motion   : 运动模糊核长度（px）
#   noise    : 传感器噪声 sigma（灰度级）
#   jpeg     : JPEG 质量
#   shade    : 环境光渐变强度
LEVELS: Dict[str, Dict] = {
    "clean": dict(ink=(0.0, 0.0), tilt=(0.0, 0.0), roll=(0.0, 0.0), spec=(0.0, 0.0),
                  defocus=(0.0, 0.0), motion=(0, 0), noise=(0.0, 0.0),
                  jpeg=(98, 98), shade=(0.0, 0.0)),
    "light": dict(ink=(0.0, 0.8), tilt=(6.0, 11.0), roll=(-4.0, 4.0), spec=(0.10, 0.30),
                  defocus=(0.35, 0.85), motion=(0, 3), noise=(0.6, 1.8),
                  jpeg=(88, 95), shade=(0.02, 0.06)),
    "moderate": dict(ink=(0.6, 1.5), tilt=(11.0, 18.0), roll=(-7.0, 7.0), spec=(0.25, 0.50),
                     defocus=(0.85, 1.45), motion=(2, 6), noise=(1.8, 3.8),
                     jpeg=(72, 86), shade=(0.05, 0.12)),
    "heavy": dict(ink=(1.2, 2.4), tilt=(18.0, 25.0), roll=(-10.0, 10.0), spec=(0.40, 0.72),
                  defocus=(1.40, 2.30), motion=(5, 10), noise=(3.8, 7.5),
                  jpeg=(50, 70), shade=(0.10, 0.20)),
    "extreme": dict(ink=(2.0, 3.2), tilt=(24.0, 30.0), roll=(-14.0, 14.0), spec=(0.60, 0.92),
                    defocus=(2.20, 3.40), motion=(9, 16), noise=(7.0, 12.0),
                    jpeg=(28, 50), shade=(0.18, 0.32)),
}
LEVEL_ORDER = ["clean", "light", "moderate", "heavy", "extreme"]
LEVEL_CN = {"clean": "无扰动", "light": "轻度", "moderate": "中度",
            "heavy": "重度", "extreme": "极限"}


# ==============================================================================
# 1. 小工具
# ==============================================================================

def _dwidth(s: str) -> int:
    """终端显示宽度（中文算 2 列）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _pad(s: str, width: int, align: str = "<") -> str:
    d = max(0, width - _dwidth(s))
    return s + " " * d if align == "<" else " " * d + s


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _arc(cx: float, cy: float, rx: float, ry: float,
         a0: float, a1: float, n: int = 10) -> List[Tuple[float, float]]:
    """椭圆弧采样点（角度制，图像坐标系 y 向下）。"""
    ang = np.deg2rad(np.linspace(a0, a1, n))
    return [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in ang]


def _rounded_rect(x0: float, y0: float, x1: float, y1: float,
                  rx: float, ry: float, n: int = 8) -> List[Tuple[float, float]]:
    """圆角矩形多边形（顺时针，y 向下）。"""
    pts: List[Tuple[float, float]] = []
    pts.append((x0 + rx, y0))
    pts.append((x1 - rx, y0))
    pts += _arc(x1 - rx, y0 + ry, rx, ry, -90, 0, n)
    pts.append((x1, y1 - ry))
    pts += _arc(x1 - rx, y1 - ry, rx, ry, 0, 90, n)
    pts.append((x0 + rx, y1))
    pts += _arc(x0 + rx, y1 - ry, rx, ry, 90, 180, n)
    pts.append((x0, y0 + ry))
    pts += _arc(x0 + rx, y0 + ry, rx, ry, 180, 270, n)
    return pts


# ==============================================================================
# 2. Hamming(7,4) 纯 Python 实现
# ==============================================================================
# 码字位置 1..7，校验位在 1,2,4，数据位在 3,5,6,7
#   p1 = d1^d2^d4     覆盖 1,3,5,7
#   p2 = d1^d3^d4     覆盖 2,3,6,7
#   p4 = d2^d3^d4     覆盖 4,5,6,7
# 伴随式 s = s1 + 2*s2 + 4*s4 直接给出出错位置（1-indexed）

def hamming74_encode(data_bits: Sequence[int]) -> List[int]:
    """每 4 bit -> 7 bit。要求 len(data_bits) % 4 == 0。"""
    if len(data_bits) % 4 != 0:
        raise ValueError("Hamming(7,4) 需要 4 的整数倍长度")
    out: List[int] = []
    for i in range(0, len(data_bits), 4):
        d1, d2, d3, d4 = (int(b) & 1 for b in data_bits[i:i + 4])
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p4 = d2 ^ d3 ^ d4
        out += [p1, p2, d1, p4, d2, d3, d4]
    return out


def hamming74_decode(bits: Sequence[int]) -> Tuple[List[int], int, List[bool]]:
    """每 7 bit -> 4 bit，纠正 1 位错。返回 (数据, 纠正次数, 每组是否发生过纠正)。"""
    if len(bits) % 7 != 0:
        raise ValueError("Hamming(7,4) 码字长度必须为 7 的整数倍")
    out: List[int] = []
    n_corr = 0
    flags: List[bool] = []
    for i in range(0, len(bits), 7):
        b = [int(x) & 1 for x in bits[i:i + 7]]
        s1 = b[0] ^ b[2] ^ b[4] ^ b[6]
        s2 = b[1] ^ b[2] ^ b[5] ^ b[6]
        s4 = b[3] ^ b[4] ^ b[5] ^ b[6]
        syn = s1 + 2 * s2 + 4 * s4
        hit = syn != 0
        if hit:
            b[syn - 1] ^= 1
            n_corr += 1
        flags.append(hit)
        out += [b[2], b[4], b[5], b[6]]
    return out, n_corr, flags


def payload_to_bits(digits: str) -> List[int]:
    """11 位十进制 -> 37 bit（MSB first）。"""
    if len(digits) != PAYLOAD_DIGITS or not digits.isdigit():
        raise ValueError("载荷必须是 11 位十进制数字")
    v = int(digits)
    if v >= (1 << PAYLOAD_BITS):
        raise ValueError("载荷超出 37 bit 表示范围")
    return [(v >> (PAYLOAD_BITS - 1 - i)) & 1 for i in range(PAYLOAD_BITS)]


def bits_to_payload(bits37: Sequence[int]) -> Optional[str]:
    """37 bit -> 11 位号码字符串。

    37 bit 能表示到 1.37e11，比 11 位号码的上限 1e11 大，因此约 27% 的
    码字空间对应"不存在的号码"。原实现用取模把它们折回合法区间，
    结果是**任何垃圾输入都能吐出一个人模人样的 11 位号码**，
    只能靠置信度区分——这对验真场景是危险的。
    现在越界直接返回 None，让调用方明确知道"这段码字不合法"。
    """
    v = 0
    for b in bits37:
        v = (v << 1) | (int(b) & 1)
    if v >= 10 ** PAYLOAD_DIGITS:
        return None
    return f"{v:011d}"


def encode_payload(digits: str) -> List[int]:
    """11 位数字 -> 70 bit 码字。37 bit 前补 3 个 0 凑 40 bit -> 10 组 (7,4)。"""
    b37 = payload_to_bits(digits)
    b40 = [0, 0, 0] + b37
    code = hamming74_encode(b40)
    assert len(code) == N_BITS
    return code


_HAMMING74_CODEBOOK: Optional[np.ndarray] = None


def hamming_codebook() -> np.ndarray:
    """16 个合法码字（各 7 bit），用于软判决最大似然译码。"""
    global _HAMMING74_CODEBOOK
    if _HAMMING74_CODEBOOK is None:
        rows = []
        for v in range(16):
            d = [(v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1]
            rows.append(hamming74_encode(d))
        _HAMMING74_CODEBOOK = np.array(rows, np.int8)
    return _HAMMING74_CODEBOOK


def decode_payload_soft(soft: Sequence[float]) -> Tuple[str, int]:
    """软判决最大似然译码（每 7 bit 独立，枚举 16 个合法码字取相关最大者）。

    硬判决 + 伴随式纠错只能纠正 1 位错，一组内出现 2 位错就会**误纠**成
    另一个合法码字（数据静默出错）。软判决把每个比特的可靠度都用上，
    等价于把判决门限从"逐比特"提升到"逐码字"，对模糊/噪声的容忍度更高。

    参数
    ----
    soft : 长度 70 的软信息，正值表示判为 1 的置信度

    返回
    ----
    (11 位号码, 与硬判决相比被改动的比特数)
    """
    cb = hamming_codebook()
    sgn = (cb.astype(np.float64) * 2.0 - 1.0)          # (16,7) -> ±1
    soft = np.asarray(soft, np.float64).ravel()
    hard = (soft > 0).astype(np.int8)
    data: List[int] = []
    n_chg = 0
    for g in range(len(soft) // 7):
        sl = slice(g * 7, (g + 1) * 7)
        best = int(np.argmax(sgn @ soft[sl]))
        code = cb[best]
        data += [int(code[2]), int(code[4]), int(code[5]), int(code[6])]
        n_chg += int(np.sum(code != hard[sl]))
    return bits_to_payload(data[3:]), n_chg


def group_codeword_margins(soft: Sequence[float]) -> np.ndarray:
    """每组 Hamming 码字的 best-vs-2nd 间隙，再除以整段软信息的标准差。

    返回长度 = 组数（70 bit 时为 10）。平均值是「有没有水印」的门；
    最小值才会暴露「其中一组选得很勉强」。单组选错或打平时，平均值
    仍可能被其余组抬过 1.8。
    """
    cb = hamming_codebook()
    sgn = (cb.astype(np.float64) * 2.0 - 1.0)
    soft = np.asarray(soft, np.float64).ravel()
    n = int(soft.size) // 7
    if n <= 0:
        return np.zeros(0, np.float64)
    sd = float(np.std(soft)) or 1e-9
    ms = np.empty(n, np.float64)
    for g in range(n):
        v = sgn @ soft[g * 7:(g + 1) * 7]
        # partition 只保证这两位是最大的两个，彼此无序，必须再取 max-min。
        top = np.partition(v, len(v) - 2)[-2:]
        ms[g] = float(np.max(top) - np.min(top)) / sd
    return ms


def codeword_margin(soft: Sequence[float]) -> float:
    """各组码字间隙的平均。真码明显高于噪声，但是读对/读错还要看最小值。

    早期边缘载体（调制 ±1.8 px）实测：
        真码 clean 5.28 / heavy 4.35 / extreme 2.04
        随机噪声 0.88~1.40，无码印版 0.96
    mean|conf| 分不开有码和无码；这个平均值可以。它盖不住单组选错。
    """
    ms = group_codeword_margins(soft)
    return float(ms.mean()) if ms.size else 0.0


def group0_pad_bits(soft: Sequence[float]) -> Optional[Tuple[int, int, int]]:
    """第 0 组软判决的前 3 个数据位。编码时这 3 位被强制写成 000。"""
    soft = np.asarray(soft, np.float64).ravel()
    if soft.size < 7:
        return None
    cb = hamming_codebook()
    sgn = (cb.astype(np.float64) * 2.0 - 1.0)
    best = int(np.argmax(sgn @ soft[:7]))
    code = cb[best]
    return (int(code[2]), int(code[4]), int(code[5]))


def pad_bits_from_hard(bits: Sequence[int]) -> Optional[Tuple[int, int, int]]:
    """第 0 组硬判决的前 3 个数据位。软判决关闭时用这个核对补零。"""
    raw = [int(b) & 1 for b in list(bits)[:7]]
    if len(raw) < 7:
        return None
    data, _, _ = hamming74_decode(raw)
    return (int(data[0]), int(data[1]), int(data[2]))


def expected_differential_px(depth: float) -> float:
    """满调制时一对码元的匹配滤波差分（px）。

    单侧边沿峰值是 depth，笔画两侧一起动，宽度峰值约 2·depth。
    窗是 sin，投影按 Σs 归一（不是 Σs²），单格输出约 (π/2)·depth，
    对向的另一格再减一次，差分约 π·depth。soft_n = 实测差分 / 这个期望。
    """
    return math.pi * float(depth)


def readout_qc(conf: Sequence[float], bits: Sequence[int], *,
               soft: bool) -> Dict:
    """生产门要用的裕度与补零。``margin`` 仍是各组间隙的平均。

    ``conf`` 在边缘路径上应是按期望幅度归一化后的 soft_n，不是像素差分。
    间隙再除以 soft 的标准差，所以它不是 px 门限。
    """
    ms = group_codeword_margins(conf)
    mean = float(ms.mean()) if ms.size else 0.0
    minimum = float(ms.min()) if ms.size else 0.0
    pad = group0_pad_bits(conf) if soft else pad_bits_from_hard(bits)
    return {
        "margin": mean,
        "margin_mean": mean,
        "margin_min": minimum,
        "group_margins": [float(x) for x in ms],
        "pad_bits": None if pad is None else [int(b) for b in pad],
    }


def decode_payload(code_bits: Sequence[int]) -> Tuple[str, int, List[bool]]:
    data40, n_corr, flags = hamming74_decode(code_bits)
    return bits_to_payload(data40[3:]), n_corr, flags


# ==============================================================================
# 3. ARROW 字形设计（纯几何绘制，不依赖任何字体文件）
# ==============================================================================
# 局部坐标系：字高 = 1.0，y 向下（0 = 上沿，1 = 基线），x 从 0 到字宽。
# 五个字母按参考图比例：A 0.90 / R 0.85 / R 0.85 / O 0.95 / W 1.15
# 笔画粗细 t = 0.14 字高（中粗无衬线），字间距均分至 TEXT 带宽度。

T_WEIGHT: float = 0.162          # 笔画水平宽度（占字高比例），由参考图实测


def _superellipse(cx: float, cy: float, rx: float, ry: float,
                  n: float = 2.0, steps: int = 72) -> List[Tuple[float, float]]:
    """超椭圆 |x/rx|^n + |y/ry|^n = 1。n=2 为椭圆，n 越大越接近圆角矩形。"""
    th = np.linspace(0, 2 * np.pi, steps, endpoint=False)
    c, s_ = np.cos(th), np.sin(th)
    x = np.sign(c) * (np.abs(c) ** (2.0 / n)) * rx + cx
    y = np.sign(s_) * (np.abs(s_) ** (2.0 / n)) * ry + cy
    return list(zip(x.tolist(), y.tolist()))


def _glyph_A() -> Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
    """按参考图实测：宽 0.965，顶点平台 0.18，横杠 0.61~0.74，字怀顶点 0.188。"""
    w = 0.965
    half = w / 2.0
    apex_half = 0.090
    slope = half - apex_half                    # 外沿水平斜率
    dt = T_WEIGHT                               # 水平截宽（实测）
    y_c = (dt - apex_half) / slope              # 字怀顶点
    poly = [
        (half - apex_half, 0.0),
        (half + apex_half, 0.0),
        (w, 1.0),
        (w - dt, 1.0),
        (half, y_c),
        (dt, 1.0),
        (0.0, 1.0),
    ]
    yb0, yb1 = 0.610, 0.740
    def _xl(y): return (half - apex_half) * (1.0 - y)      # 左外沿
    def _xr(y): return (half + apex_half) + slope * y       # 右外沿
    bar = [(_xl(yb0), yb0), (_xr(yb0), yb0), (_xr(yb1), yb1), (_xl(yb1), yb1)]
    return [poly, bar], []


def _glyph_R() -> Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
    """宽 0.852；碗高 0.52、外沿最右 0.817、圆角 rx=0.27/ry=0.26；腿为楔形。"""
    w = 0.852
    t = T_WEIGHT
    bowl_bot = 0.505
    rx, ry = 0.270, 0.260
    stem = [(0.0, 0.0), (t, 0.0), (t, 1.0), (0.0, 1.0)]
    bowl = [(t, 0.0), (0.817 - rx, 0.0)]
    bowl += _arc(0.817 - rx, ry, rx, ry, -90, 0, 14)
    bowl += [(0.817, bowl_bot - ry)]
    bowl += _arc(0.817 - rx, bowl_bot - ry, rx, ry, 0, 90, 14)
    bowl += [(t, bowl_bot)]
    # 字怀
    hx, hy = 0.160, 0.140
    hx1 = 0.662
    hole = [(t, 0.160), (hx1 - hx, 0.160)]
    hole += _arc(hx1 - hx, 0.160 + hy, hx, hy, -90, 0, 12)
    hole += [(hx1, 0.440 - hy)]
    hole += _arc(hx1 - hx, 0.440 - hy, hx, hy, 0, 90, 12)
    hole += [(t, 0.440)]
    # 楔形腿（左沿陡、右沿缓，实测）
    leg = [(0.225, 0.500), (0.400, 0.500), (0.852, 1.0), (0.677, 1.0)]
    return [stem, bowl, leg], [hole]


def _glyph_O() -> Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
    """宽 0.93；外轮廓近似椭圆（n=2.2），内孔更方（n=3.0）—— 实测结果。"""
    w = 0.93
    outer = _superellipse(w / 2.0, 0.5, w / 2.0, 0.5, n=2.05, steps=96)
    hole = _superellipse(w / 2.0, 0.5, w / 2.0 - T_WEIGHT,
                         0.5 - T_WEIGHT - 0.018, n=3.0, steps=96)
    return [outer], [hole]


def _glyph_W() -> Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
    """宽 1.197；五顶点 (0.074,0)(0.30,1)(0.595,0)(0.88,1)(1.12,0)，各笔水平宽实测。"""
    centers = [(0.074, 0.0), (0.300, 1.0), (0.595, 0.0), (0.880, 1.0), (1.120, 0.0)]
    hw = [0.074, 0.088, 0.088, 0.0775]      # 各笔半宽（水平）
    polys: List[List[Tuple[float, float]]] = []
    for i in range(4):
        (x1, y1), (x2, y2) = centers[i], centers[i + 1]
        h = hw[i]
        polys.append([(x1 - h, y1), (x1 + h, y1), (x2 + h, y2), (x2 - h, y2)])
    return polys, []


def _glyph_sequence() -> List[Tuple[str, float]]:
    """返回 [(字形名, 局部宽度), ...]，宽度和用于均分字间距。"""
    return [("A", 0.965), ("R", 0.852), ("R", 0.852), ("O", 0.930), ("W", 1.197)]


GLYPH_BUILDERS = {"A": _glyph_A, "R": _glyph_R, "O": _glyph_O, "W": _glyph_W}


_GLYPH_MASK_B64: str = (
    "iVBORw0KGgoAAAANSUhEUgAAAsIAAACMCAAAAABTtJAwAAAH0klEQVR42u2d647bOgyEM4Tf/5WnKHqA4hTZRLZ4F/VzNzYl"
    "8uOYsmULr8eNP/0Dr73Gl35DIfuqXoPq6RCKyftTXa8zGn04+mgfzvRuDRysEtrLxG95OUY383RNICIfJhYqTGSmGI2SiHXy"
    "16zh8vDeQGxhn7YJZF5JaCmdeNdmGSBmA/sPT8Ky0vMjVvI6sYVDHDgA37HT0gt/ZqoSFYRXcYgic2iTwp3DE8ZdOgJSoYuh"
    "MsrqCqSAcAMXFGVYib6Fs6CIk6UnH20LYnYZ/f3B0BfhEWIbihg++mTihKcIs9pIm6QcO2Vw8kJiZNjxSmqVEqjhXzk4fYsx"
    "zIbj1+iqdIWjG8NG0y/W9j4eItyqzuXYLx59aRubqdrD48TiCA8civYZ0gGUCNNVJ+YIRjTQvq3LA9d3a5i+WnrmH/IY2U8F"
    "+3yWZbSJ1PKv6RHbJwgXrHER3OtN+7xP798/UZ/hY6ZzuWAHYvu5YX9lRc7PpwewlXv5ZyLnTOdQ1D630mOR4vSXVioiHP08"
    "yUIIGcwwbYZ150d0pU7Vo0fdVENB81QAeMU4M7FpWkhwGHY1Tx2Ab/7WUYTpjvBJwCdgWHUpGe6bynKngnoIM+lcaTOKLGn/"
    "ttM/CzGVEit7IfGTM1FhsOUatQuDVt/zgSLCKDfs+vahfxjTpamBCvOQ/M4vwnBCH2FkJpnOTSWRT8w1HlMyTy7fRJjq/swS"
    "fGaEj+49CBEcbo5DrD0zMvw0jGZVG8K4NLEsr3BVmObNIIJCbLNMRPQ6NhO6IgQHGrPouv0aidHufH5Dg5HiAcJfJ3Mjw2V0"
    "cXfRHhMljmRNshHhEgqaIOaSQROm5ZFyliuGRTOx5r6afWQ7Ssbem1fzHQlnMNyW0dqOmNE+xn2El57MoXzGn5NrMHSQq5Nl"
    "sJlW+3onexTOZM436VEXusfLPKmE8F4/mZ6NA+sYu0qCrp2f6dw0X0zVLySy1WFUqyyi66FF+z51BBokhJ8Ks76nzp3XIne+"
    "yI7rq03mhmCXbjrvyyROKTv31ZR2lphi+N9zyCDkhcb619S98r1CMfzdF7JxfpSS4ZL71bZphtkipzg/+vsBlb9fgNS5Km6J"
    "FZoIrLfdciDqzJQRX88gGXuozg9D+xtt38h0lg2hr/CcbF6CHlcCu38h8zLIYLwPm/ueJNwcR3X7imrNxApwFU/vjvbbC7du"
    "RshTh3ZZZonD7VcshqF1R+J2WDgE79hHXa5Nuy6+qTat57Uscthi0okyMoxJoTR+4kNiZCI4GWStNTA9nTwbDIbguQa8gyPg"
    "QiuuweEQPFWM9tDl3Mlc9D6KGGd9g40KCD8eUHoZBo5OoMJDxS2EOW4dgu/wEQGMRA/6TAmeT8iswEE7hKH0m6mCZz63H6fr"
    "uDpiSohn3V6BQfnpsp0KY+dXwWkxJUQ7CbpOEuERYPXq9flGump0Xe6BIIbfaYpsyMRs2lLoFsIZUgp/QrhZHTEv0eXqFLQO"
    "uwxNK79Dh80Q7FYw2CSALXd6QfjprtoijDt9pXptg1u+2gx3zRSw7/QV0CX9WGCRIysKFu2z9/yAMaeT2hn4P2sL91xZwX5S"
    "zpH0fBIRbMZRxPAsmrYTHygUEtlrZAR38Iv9dnO6vwPyf3f5swqzKMGvb89w2d5+HzlKUAtHuQ6IDZuRfTfgkKEYZm6EQ6ce"
    "PMB+12IY6wgztKPmV/N60/f8zEWVwo2/I4FYGcSGfTjT17EW5jCc0f4RJQjvjlViOzoX80q5n/PhhnTOdnST4cSyyTDmJXbg"
    "h8rZ0wM5IThpOpdAhsPtl4wPFRDmMHxCOc77fb5ZGkG/c6lUmEczxMxOqyVIMm7KmEKlbqbQ3T1ZauFYGY7eqjtchtklJfp/"
    "5X3uDjv4K7AUfoMws2ZbUevQPYqhw8+oBwfstZFUhpMmMHyHgv3fykzmDpXhxOnIW+e7dOLAR8uXnd7Cid4TWtm+Rr+pm2+h"
    "Vz/RySfMlKpS/cP0XOrXwsjmSg2/R7/IM2suVcgSH++MDGfKfJaME7ZUGFmD0VmGYeW2Zo/2JTiFpmXS8IgoQRfhtnfUun1R"
    "j2EEI503JBiAM57QPaxkLBjup1ISnkQjw54osp+PZGF8SBiKJjK8cRRzexvGrglQ4ZnQaSNF6ppD0TjKCZO5DJ6HwVEMIDjK"
    "3UigwodXEg/tKzLMphIlAQk1Ezo9HaZWsqBsZMRvMhf9ClDOCd32w2Kq/KywTkmdbBsZfgwxqeibkAcgWEH41LXu6WV4YROa"
    "j4iSvUMrnvkUvdqmcRX/m1O+/TOVLKS9Pl4HcVT27Y21XeP/Yxj341l7wn5lGB3nsUdkYQSdVIzKCd/7wjkvWPlvSqBxUJa6"
    "gAWEz37A0KIcrkuwqQqjl7+SzicZ6Lj6NZyMDNYQU6MtnGBxECIQju5L8NehCsiwTTBQI7eRZzp3/JJL5PJdi2hIkjqCR5tf"
    "tK9MnNX2kohA+JxLV81lwybQdbkgykzmCsmw5rrBXARjE+EMacr8vkphXwk9VHIiviLMUjE8XIZVIDaVYO8Ay3EMVZfh/VOg"
    "120h8Y/1yPC2faAdwNhB+LTJXIs9kPCQxBSltK6xK49YzpLLu5FlRvDcoygBIpy1Gi23Fd0tKQaaSkSmHY94CsOaQ1kjswC/"
    "jzt4UfuMU0p8j5bu9jl4+X80GHnS/Relejxr0+N6pAAAAABJRU5ErkJggg=="
)

_EXTERNAL_MASTERS = ("ARROW_clean_pro.png",)   # 只用脚本产出的原图


def _p97(p: np.ndarray) -> float:
    """与 np.percentile(p, 97.0) 逐位等价的快速实现。

    np.percentile 内部要排序，在 107 点的小数组上耗时约 40 us，是逐行宽度
    测量的最大单项开销；np.partition 只分出所需的两阶统计量即可，快 20 倍
    以上，且语义完全相同（同为 linear 插值）。
    """
    n = p.size
    if n == 0:
        return 0.0
    k = 0.97 * (n - 1)
    lo = int(np.floor(k))
    hi = int(np.ceil(k))
    if lo == hi:
        return float(np.partition(p, lo)[lo])
    part = np.partition(p, (lo, hi))
    return float(part[lo] + (part[hi] - part[lo]) * (k - lo))


def imread_u(path: str, flags: int = cv2.IMREAD_GRAYSCALE) -> Optional[np.ndarray]:
    """Unicode 安全的读图（Windows 下 cv2.imread 不支持中文路径）。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, flags)
    except Exception:
        return None


def imwrite_u(path: str, img: np.ndarray) -> bool:
    """Unicode 安全的写图（Windows 下 cv2.imwrite 不支持中文路径）。"""
    try:
        ext = os.path.splitext(path)[1] or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False


def _load_master() -> Optional[np.ndarray]:
    """加载字形母版：优先外部文件（脚本产出的原图），其次内嵌母版。

    返回裁到墨迹包围盒的灰度图。
    """
    cands = []
    # 先找模块所在目录：这样无论从哪个工作目录调用都能拿到随包分发的母版
    here = os.path.dirname(os.path.abspath(__file__))
    for nm in _EXTERNAL_MASTERS:
        cands += [os.path.join(here, nm), nm, os.path.join("arrow_wm_out", nm)]
    for path in cands:
        if os.path.exists(path):
            img = imread_u(path)
            if img is not None:
                m = (img < 128)                      # 白底深字 -> 墨迹为 True
                if m.sum() > 1000:
                    ys, xs = np.nonzero(m)
                    # 统一成"墨 = 255"的二值掩膜，与内嵌母版同极性
                    return (m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                            .astype(np.uint8) * 255)
    return _decode_glyph_mask()


def _decode_glyph_mask() -> Optional[np.ndarray]:
    """解码内嵌的字形母版（1bit PNG，逐像素与设计稿一致）。

    母版直接从设计稿提取：红色标注线按颜色剔除，二值化后裁到文字包围盒，
    与设计稿的像素差异为 0.000%。运行时用 CUBIC 放大到 4x 制版尺度再阈值化，
    得到平滑的亚像素轮廓。
    """
    try:
        raw = base64.b64decode(_GLYPH_MASK_B64)
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        return arr
    except Exception:
        return None



def order_quad(q: np.ndarray) -> np.ndarray:
    """把 4 个点整理为 TL / TR / BR / BL 顺序（图像坐标，y 向下）。"""
    q = np.asarray(q, np.float32).reshape(4, 2)
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]
    s = q[:, 0] + q[:, 1]
    q = np.roll(q, -int(np.argmin(s)), axis=0)
    if q[1, 0] < q[3, 0]:
        q = q[[0, 3, 2, 1]]
    return q.astype(np.float32)


# ==============================================================================
# 4. 底版设计对象（嵌入器与解码器共用同一份确定性几何 -> 布局严格对齐）
# ==============================================================================

@dataclass
class Cell:
    """一个码元区域（差分对的一半）。

    x_mid / slope 描述该笔画中心线随 y 的线性漂移：
        x_center(y) = x_mid + slope * (y - y_mid)
    嵌入器与解码器使用完全相同的数值，因此坐标系严格一致。
    """
    idx: int
    track: int
    y0: int
    y1: int
    x_mid: float
    y_mid: float
    slope: float
    width: float
    half_win: int
    clearance: float = 0.0   # 该段与邻笔的最小净空（px）
    letter: str = ""
    letter_i: int = -1
    interval: int = -1
    iv_y0: int = 0
    iv_y1: int = 0
    depth_scale: float = 1.0
    pair_risk: float = 0.0

    def center_at(self, y: float) -> float:
        return self.x_mid + self.slope * (y - self.y_mid)

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def dist_to_iv_end(self) -> float:
        if self.iv_y1 <= self.iv_y0:
            return 0.0
        return float(min(self.y0 - self.iv_y0, self.iv_y1 - self.y1))


class ArrowDesign:
    """ARROW 底版：字形栅格化 + 自同步基准四边形 + 码元布局推导。

    设计上不再有任何定位黑块。解码所需的空间基准完全由**字母自身的凸包**提供：
    "ARROW" 的凸包稳定收敛为 4 个角点（W 右上尖、W 右下脚、A 左脚、A 顶点），
    这 4 点与设计稿一一对应，构成 findHomography 的基准。
    """

    def __init__(self, verbose: bool = False, depth: float = MOD_DEPTH,
                 layout_version: int = LAYOUT_VERSION,
                 window_pow: float = WINDOW_POW):
        self.verbose = verbose
        self.layout_version = int(layout_version)
        if self.layout_version not in (1, 2):
            raise ValueError(f"layout_version 只能是 1 或 2，收到 {layout_version}")
        # v1 的码元位置锁在 min_clear=20 的旧算法上。depth 仍可取 6，
        # 只影响调制幅度和测量窗，不挪码元。这是消融里的 v1@6。
        self.depth = float(depth)
        self.window_pow = float(window_pow)
        self.text_mask4 = self._rasterize()
        self.mask4 = self.text_mask4          # 无锚点：掩膜即纯文字
        self.anchor_mask4 = np.zeros_like(self.text_mask4)
        # 1x 二值掩膜（用于布局推导）
        small = cv2.resize(self.text_mask4, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
        self.mask1 = (small > 127).astype(np.uint8)
        self.ref_quad = self._compute_ref_quad()
        self.letter_centroids = self._compute_letter_centroids()
        self.hole_centroids = self._compute_hole_centroids()
        _ys, _xs = np.nonzero(self.mask1)
        self.ink_bbox = (float(_xs.min()), float(_ys.min()),
                         float(_xs.max()), float(_ys.max()))
        _p = np.column_stack(np.nonzero(self.mask1)[::-1]).astype(np.float64)
        self.mom_mu = _p.mean(axis=0)
        self.mom_C = np.cov((_p - self.mom_mu).T)
        self.cells: List[Cell] = []
        self.pair_perm: Tuple[int, ...] = tuple(range(N_BITS))
        self.pair_perm_id: str = PAIR_PERM_ID_V1
        self._build_cells()

    # ---------------- 栅格化 ----------------
    def _place_letters(self) -> List[Tuple[str, float]]:
        specs = _glyph_sequence()
        total_w = sum(w for _, w in specs) * CAP_H
        gap = (TEXT_X1 - TEXT_X0 - total_w) / (len(specs) - 1)
        out = []
        x = float(TEXT_X0)
        for name, w in specs:
            out.append((name, x))
            x += w * CAP_H + gap
        return out

    def _rasterize(self) -> np.ndarray:
        """栅格化字形到 4x 制版尺度。

        母版来源优先级：外部 `ARROW_clean_pro.png`（脚本产出的原图）→ 内嵌 1bit 母版
        → 几何多边形构造。母版按宽度铺满文字带、纵向以字高中心对齐，保持原始宽高比。
        """
        h4, w4 = CANVAS_H * SS, CANVAS_W * SS
        text = np.zeros((h4, w4), np.uint8)
        mask = _load_master()
        if mask is not None:
            mh, mw = mask.shape
            tw = TEXT_X1 - TEXT_X0
            th = int(round(tw * mh / float(mw)))
            cy = CAP_TOP + CAP_H // 2
            y0 = cy - th // 2
            big = cv2.resize(mask, (tw * SS, th * SS), interpolation=cv2.INTER_CUBIC)
            big = (big > 127).astype(np.uint8) * 255
            text[y0 * SS:(y0 + th) * SS, TEXT_X0 * SS:(TEXT_X0 + tw) * SS] = big
            return text
        # ---- 回退：几何构造 ----
        solids: List[np.ndarray] = []
        holes: List[np.ndarray] = []
        for name, x_off in self._place_letters():
            polys, holes_l = GLYPH_BUILDERS[name]()
            for p in polys:
                pts = np.array([[(x_off + u * CAP_H) * SS, (CAP_TOP + v * CAP_H) * SS]
                                for (u, v) in p], np.int32)
                solids.append(pts)
            for p in holes_l:
                pts = np.array([[(x_off + u * CAP_H) * SS, (CAP_TOP + v * CAP_H) * SS]
                                for (u, v) in p], np.int32)
                holes.append(pts)
        for pts in solids:
            cv2.fillPoly(text, [pts], 255, lineType=cv2.LINE_8)
        for pts in holes:
            cv2.fillPoly(text, [pts], 0, lineType=cv2.LINE_8)
        return text

    def _compute_letter_centroids(self) -> List[Tuple[float, float]]:
        """五个字母各自的墨迹质心 —— 解码端最强的自同步基准（5 点超定单应）。"""
        out: List[Tuple[float, float]] = []
        for (name, x_off), (_, w) in zip(self._place_letters(), _glyph_sequence()):
            x0 = int(round(x_off))
            x1 = int(round(x_off + w * CAP_H))
            sub = self.mask1[:, max(0, x0):min(CANVAS_W, x1)]
            ys, xs = np.nonzero(sub)
            if len(xs) == 0:
                out.append((x_off + w * CAP_H * 0.5, CAP_TOP + CAP_H * 0.5))
            else:
                out.append((x0 + float(xs.mean()), float(ys.mean())))
        return out

    def _compute_hole_centroids(self) -> List[Tuple[float, float]]:
        """4 个内孔（O 内环 / 两个 R 字怀 / A 上三角）的质心，按 x 排序。

        内孔是**被墨迹完全包围**的背景区域，其质心不受外轮廓模糊、油墨胀缩影响，
        在极端透视下比"字母墨迹质心"更稳，是 4 点精确单应的理想基准。
        """
        bg = (1 - self.mask1).astype(np.uint8)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(bg, 4)
        H, W = self.mask1.shape
        hs: List[Tuple[float, float, int]] = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if x == 0 or y == 0 or x + w >= W or y + h >= H:
                continue
            if a < 200:
                continue
            hs.append((float(cents[i][0]), float(cents[i][1]), int(a)))
        hs.sort(key=lambda t: t[0])
        return [(x, y) for x, y, _ in hs]

    def _compute_ref_quad(self) -> np.ndarray:
        """由字形凸包提取 4 个自同步基准角点（与拍摄端同一算法）。"""
        pts = np.column_stack(np.nonzero(self.mask1)[::-1]).astype(np.int32)  # x,y
        hull = cv2.convexHull(pts)
        peri = cv2.arcLength(hull, True)
        for eps in (0.03, 0.02, 0.04, 0.05, 0.06, 0.015, 0.08):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                return order_quad(ap.reshape(4, 2).astype(np.float32))
        rect = cv2.minAreaRect(pts)
        return order_quad(cv2.boxPoints(rect).astype(np.float32))

    # ---------------- 码元布局 ----------------
    def _build_cells(self) -> None:
        self.cells = build_layout(self.mask1, n_cells=N_CELLS, d=self.depth,
                                  margin=CELL_MARGIN, version=self.layout_version)
        self._annotate_letters()
        if self.layout_version == 1:
            self.pair_perm = tuple(range(N_BITS))
            self.pair_perm_id = PAIR_PERM_ID_V1
        else:
            derived = derive_pair_perm(self.cells)
            # 默认 v2@6 的置换是冻结常量。其它深度仍用同一规则现算，
            # 保证编码器和解码器对得上，同时默认印版不随实现漂移。
            if abs(self.depth - MOD_DEPTH) < 1e-9:
                if derived != PAIR_PERM_V2:
                    raise RuntimeError(
                        "v2@6 的 PAIR_PERM 与冻结常量不一致。"
                        "布局改了就要有意识地更新 PAIR_PERM_V2。")
                self.pair_perm = PAIR_PERM_V2
            else:
                self.pair_perm = derived
            self.pair_perm_id = PAIR_PERM_ID_V2
        if self.verbose:
            print(f"[Design] layout=v{self.layout_version} depth={self.depth:g} "
                  f"码元数 = {len(self.cells)}，"
                  f"平均高度 = {np.mean([c.height for c in self.cells]):.1f}px，"
                  f"平均笔画宽 = {np.mean([c.width for c in self.cells]):.1f}px "
                  f"perm={self.pair_perm_id}")

    def _annotate_letters(self) -> None:
        """按五个字母的水平跨度给码元打字母。两个 R 记成 R1 / R2。"""
        names = ("A", "R1", "R2", "O", "W")
        spans: List[Tuple[str, int, float, float]] = []
        for i, ((name, x_off), (_, w)) in enumerate(
                zip(self._place_letters(), _glyph_sequence())):
            label = names[i] if i < len(names) else name
            spans.append((label, i, float(x_off), float(x_off + w * CAP_H)))
        for c in self.cells:
            hit = None
            best = None
            best_d = 1e18
            for label, idx, x0, x1 in spans:
                if x0 <= c.x_mid <= x1:
                    hit = (label, idx)
                    break
                dist = min(abs(c.x_mid - x0), abs(c.x_mid - x1))
                if dist < best_d:
                    best_d = dist
                    best = (label, idx)
            label, idx = hit if hit is not None else best
            c.letter = label
            c.letter_i = idx

    def render(self, mask4: np.ndarray) -> np.ndarray:
        """把 4x 二值掩膜渲染成 8bit 灰度底版（白底深灰字）。"""
        cov = cv2.resize(mask4, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
        cov_f = cov.astype(np.float32) / 255.0
        return np.clip(BG_LEVEL - cov_f * (BG_LEVEL - INK_LEVEL), 0, 255).astype(np.uint8)

    @property
    def template(self) -> np.ndarray:
        """未加水印的原始底版（解码器用作配准模板）。"""
        return self.render(self.mask4)

def layout_geometry_token(cells: Sequence[Cell], *, with_window: bool = True) -> str:
    """码元几何的稳定文本。v1 黄金哈希用它，半窗随 depth 变。"""
    lines = []
    for c in cells:
        row = (f"{c.idx},{c.track},{c.y0},{c.y1},"
               f"{c.x_mid:.6f},{c.y_mid:.6f},{c.slope:.8f},{c.width:.4f}")
        if with_window:
            row += f",{c.half_win}"
        lines.append(row)
    return "\n".join(lines)


def layout_geometry_sha1(cells: Sequence[Cell], *, with_window: bool = True) -> str:
    token = layout_geometry_token(cells, with_window=with_window)
    return hashlib.sha1(token.encode("utf-8")).hexdigest()


def _ink_rows(mask1: np.ndarray) -> List[List[Tuple[int, int]]]:
    rows: List[List[Tuple[int, int]]] = []
    for y in range(mask1.shape[0]):
        r = mask1[y].astype(np.int8)
        dd = np.diff(np.concatenate(([0], r, [0])))
        s = np.where(dd == 1)[0]
        e = np.where(dd == -1)[0]
        rows.append(list(zip(s.tolist(), e.tolist())))
    return rows


def _link_tracks(rows: Sequence[Sequence[Tuple[int, int]]]) -> List[List[Tuple[int, int, int, int]]]:
    tracks: List[List[Tuple[int, int, int, int]]] = []
    active: List[Tuple[int, int, int]] = []
    for y, runs in enumerate(rows):
        new_active: List[Tuple[int, int, int]] = []
        used = set()
        for ri, (s, e) in enumerate(runs):
            best, best_ov = -1, 0.0
            for (ti, ps, pe) in active:
                if ti in used:
                    continue
                ov = min(e, pe) - max(s, ps)
                m = min(e - s, pe - ps)
                if m <= 0:
                    continue
                if ov > 0.5 * m and ov > best_ov:
                    best_ov, best = ov, ti
            if best >= 0:
                tracks[best].append((y, ri, s, e))
                used.add(best)
                new_active.append((best, s, e))
            else:
                tracks.append([(y, ri, s, e)])
                new_active.append((len(tracks) - 1, s, e))
        active = new_active
    return tracks


def _collect_intervals(rows, tracks, *, min_clear: float, min_width: float,
                       min_track: int, end_trim: int, min_len: int):
    """连续可用行。点是 (y, center, width, clearance)。"""
    intervals: Dict[int, List[List[Tuple[int, float, float, float]]]] = {}
    for ti, pts in enumerate(tracks):
        if len(pts) < min_track:
            continue
        widths = np.array([e - s for (_, _, s, e) in pts], float)
        med = float(np.median(widths))
        if med < 8:
            continue
        ok: List[Tuple[int, float, float, float]] = []
        for (y, ri, s, e) in pts:
            rr = rows[y]
            cl = (s - rr[ri - 1][1]) if ri > 0 else 10 ** 9
            cr = (rr[ri + 1][0] - e) if ri + 1 < len(rr) else 10 ** 9
            gap = float(min(cl, cr))
            if gap < min_clear:
                continue
            wv = float(e - s)
            if wv < min_width:
                continue
            if not (0.85 * med <= wv <= 1.18 * med):
                continue
            ok.append((y, 0.5 * (s + e), wv, gap))
        cur: List[Tuple[int, float, float, float]] = []
        for item in ok:
            if cur and item[0] == cur[-1][0] + 1:
                cur.append(item)
            else:
                _store_interval(intervals, ti, cur, end_trim, min_len)
                cur = [item]
        _store_interval(intervals, ti, cur, end_trim, min_len)
    return intervals


def _store_interval(intervals, ti, cur, end_trim: int, min_len: int) -> None:
    """两端按 END_TRIM 裁，但不要把还能放下一个差分对的区段裁没。"""
    if len(cur) < min_len:
        return
    if end_trim > 0:
        max_trim = max(0, (len(cur) - min_len) // 2)
        trim = min(int(end_trim), max_trim)
        if trim > 0:
            cur = cur[trim:-trim]
    if len(cur) >= min_len:
        intervals.setdefault(ti, []).append(cur)


def _cells_from_segment(ti: int, seg: Sequence[Tuple], d: float, margin: int,
                        idx: int, interval: int, iv_y0: int, iv_y1: int,
                        risk: float) -> Cell:
    ys = np.array([p[0] for p in seg], float)
    cs = np.array([p[1] for p in seg], float)
    ws = np.array([p[2] for p in seg], float)
    slope, intercept = np.polyfit(ys, cs, 1)
    y0 = int(ys[0])
    y1 = int(ys[-1]) + 1
    y_mid = 0.5 * (y0 + y1)
    x_mid = float(slope * y_mid + intercept)
    wv = float(np.median(ws))
    hw = int(round(wv * 0.5 + d + margin))
    clearance = float(min(p[3] for p in seg)) if seg and len(seg[0]) >= 4 else 0.0
    return Cell(idx=idx, track=ti, y0=y0, y1=y1, x_mid=x_mid, y_mid=y_mid,
                slope=float(slope), width=wv, half_win=hw, clearance=clearance,
                interval=interval, iv_y0=iv_y0, iv_y1=iv_y1, pair_risk=risk)


def _build_layout_v1(mask1: np.ndarray, n_cells: int, d: float, margin: int,
                     min_track: int) -> List[Cell]:
    """布局 v1：锁死 min_clear=20 的旧切分。

    码元怎么选、按什么顺序取前 n_cells，与 depth=1.8 的历史印版一致。
    ``d`` 只进入测量窗 half_win，所以 v1@6 的位置不变、窗口跟得上 ±6px。
    差分对仍是轨迹顺序上的相邻格（恒等 PAIR_PERM），允许跨区段。
    """
    min_clear = V1_MIN_CLEARANCE
    rows = _ink_rows(mask1)
    tracks = _link_tracks(rows)
    intervals = _collect_intervals(
        rows, tracks, min_clear=min_clear, min_width=0.0,
        min_track=min_track, end_trim=0, min_len=min_track)
    if not intervals:
        raise RuntimeError("布局失败：找不到任何可用的笔画区段")

    _allw = [p[2] for segs in intervals.values() for seg in segs for p in seg]
    gmed = float(np.median(_allw)) if _allw else 0.0

    chosen: List[Tuple] = []
    for h_try in range(28, 6, -1):
        cells_seg: List[Tuple] = []
        for ti in sorted(intervals):
            segs: List[Tuple] = []
            for iv_i, iv in enumerate(intervals[ti]):
                n = max(1, int(round(len(iv) / float(h_try))))
                cuts = np.linspace(0, len(iv), n + 1)
                iv_y0 = int(iv[0][0])
                iv_y1 = int(iv[-1][0]) + 1
                for k in range(n):
                    i0, i1 = int(round(cuts[k])), int(round(cuts[k + 1]))
                    if i1 - i0 < 6:
                        continue
                    seg = iv[i0:i1]
                    ws = [p[2] for p in seg]
                    med = float(np.median(ws))
                    if med <= 0 or (max(ws) - min(ws)) / med > 0.10:
                        continue
                    if gmed > 0 and abs(med - gmed) / gmed > 0.22:
                        continue
                    segs.append((iv_i, iv_y0, iv_y1, seg))
            if len(segs) % 2 == 1:
                segs = segs[:-1]
            cells_seg.extend((ti, item) for item in segs)
        if len(cells_seg) >= n_cells:
            chosen = cells_seg[:n_cells]
            break
    if not chosen:
        cells_seg = []
        for ti in sorted(intervals):
            for iv_i, iv in enumerate(intervals[ti]):
                cells_seg.append((ti, (iv_i, int(iv[0][0]), int(iv[-1][0]) + 1, iv)))
        if len(cells_seg) % 2 == 1:
            cells_seg = cells_seg[:-1]
        chosen = cells_seg
        if len(chosen) < n_cells:
            raise RuntimeError(f"布局容量不足：仅 {len(chosen)} 个码元 < {n_cells}")

    out: List[Cell] = []
    for i, (ti, (iv_i, iv_y0, iv_y1, seg)) in enumerate(chosen):
        out.append(_cells_from_segment(
            ti, seg, d, margin, i, iv_i, iv_y0, iv_y1, risk=0.0))
    return out


def _seg_acceptable(seg: Sequence[Tuple], gmed: float) -> bool:
    if len(seg) < MIN_CELL_H:
        return False
    ws = [p[2] for p in seg]
    med = float(np.median(ws))
    if med <= 0 or (max(ws) - min(ws)) / med > 0.10:
        return False
    if gmed > 0 and abs(med - gmed) / gmed > 0.22:
        return False
    return True


def _walk_cells(iv: Sequence[Tuple], h_try: int, gmed: float) -> List[Sequence[Tuple]]:
    """把区段均分成高度约 h_try 的码元，相邻两格高度最多差 1。

    余数摊进前面的格子，避免「短格配一个被尾巴撑长的格」使 |Δy| 超 1.2h。
    """
    n = len(iv)
    target = max(MIN_CELL_H, int(h_try))
    k = n // target
    while k > 1 and n / float(k) < MIN_CELL_H:
        k -= 1
    if k < 1 or n // k < MIN_CELL_H:
        return []
    base, extra = divmod(n, k)
    segs: List[Sequence[Tuple]] = []
    i = 0
    for c in range(k):
        sz = base + (1 if c < extra else 0)
        seg = iv[i:i + sz]
        i += sz
        if _seg_acceptable(seg, gmed):
            segs.append(seg)
    return segs


def _predicted_q(seg_a: Sequence[Tuple], seg_b: Sequence[Tuple], depth: float,
                 iv_y0: int, iv_y1: int) -> float:
    """几何代理的配对质量。越大越稳。斜笔画和区段端降权，容量不够时不硬删。"""
    def _one(seg):
        ys = np.array([p[0] for p in seg], float)
        cs = np.array([p[1] for p in seg], float)
        slope = float(np.polyfit(ys, cs, 1)[0]) if len(ys) >= 2 else 0.0
        clr = float(min(p[3] for p in seg))
        return abs(slope), clr, float(ys[-1] - ys[0] + 1.0), float(ys[0]), float(ys[-1])
    sl_a, cl_a, h_a, y0a, y1a = _one(seg_a)
    sl_b, cl_b, h_b, y0b, y1b = _one(seg_b)
    h = min(h_a, h_b)
    clr = min(cl_a, cl_b)
    slope = max(sl_a, sl_b)
    end = min(y0a - iv_y0, (iv_y1 - 1) - y1a, y0b - iv_y0, (iv_y1 - 1) - y1b)
    floor = max(2.0 * float(depth) + CELL_MARGIN + 2.0, 1.0)
    q = (clr / floor) * (h / float(MIN_CELL_H))
    q /= (1.0 + 4.0 * max(0.0, slope - 0.15))
    if slope > 0.7:
        q *= 0.45
    if end < float(depth):
        q *= 0.55
    return float(q)


def _trim_for_pairs(iv: Sequence[Tuple], end_trim: int, min_h: int, min_len: int):
    """在 0..END_TRIM 里选裁切：先保差分对数量，再尽量裁满。

    整段都裁满 END_TRIM 时，若干区段会剩奇数个码元，配不成对，140 会不够。
    只在多裁会少一对时才少裁。
    """
    n = len(iv)
    hi = min(int(end_trim), max(0, (n - min_len) // 2))
    best_t = 0
    best_key = None
    for t in range(0, hi + 1):
        n_cells = (n - 2 * t) // min_h
        key = (n_cells // 2, t)
        if best_key is None or key > best_key:
            best_key = key
            best_t = t
    if best_t:
        return list(iv[best_t:n - best_t])
    return list(iv)


def _build_layout_v2(mask1: np.ndarray, n_cells: int, d: float, margin: int,
                     min_track: int) -> List[Cell]:
    """布局 v2：同区段相邻配对，净空/笔画宽/端裁随深度推导。"""
    if n_cells % 2:
        raise RuntimeError("n_cells 必须是偶数")
    n_pairs = n_cells // 2
    min_clear = float(min_clearance_for_depth(d))
    min_w = float(min_stroke_w_for_depth(d))
    trim = end_trim_for_depth(d)
    rows = _ink_rows(mask1)
    tracks = _link_tracks(rows)
    raw_intervals = _collect_intervals(
        rows, tracks, min_clear=min_clear, min_width=min_w,
        min_track=min_track, end_trim=0, min_len=2 * MIN_CELL_H)
    intervals = {}
    for ti, segs in raw_intervals.items():
        kept = []
        for seg in segs:
            trimmed = _trim_for_pairs(seg, trim, MIN_CELL_H, 2 * MIN_CELL_H)
            if len(trimmed) >= 2 * MIN_CELL_H:
                # 端距相对裁切前的区段，这样 END_TRIM 会计进 dist_to_iv_end。
                kept.append((trimmed, int(seg[0][0]), int(seg[-1][0]) + 1))
        if kept:
            intervals[ti] = kept
    if not intervals:
        raise RuntimeError("v2 布局失败：找不到满足净空/笔画宽的区段")

    _allw = [p[2] for segs in intervals.values() for iv, _y0, _y1 in segs for p in iv]
    gmed = float(np.median(_allw)) if _allw else 0.0

    best = None
    for h_try in range(28, MIN_CELL_H - 1, -1):
        pairs = []
        for ti in sorted(intervals):
            for iv_i, (iv, iv_y0, iv_y1) in enumerate(intervals[ti]):
                segs = _walk_cells(iv, h_try, gmed)
                k = 0
                while k + 1 < len(segs):
                    a, b = segs[k], segs[k + 1]
                    touching = int(b[0][0]) == int(a[-1][0]) + 1
                    if not touching:
                        k += 1
                        continue
                    q = _predicted_q(a, b, d, iv_y0, iv_y1)
                    pairs.append((q, ti, iv_i, iv_y0, iv_y1, a, b))
                    k += 2
        if len(pairs) < n_pairs:
            continue
        ranked = sorted(pairs, key=lambda t: (t[0], t[1], t[5][0][0]))
        kept = ranked[len(ranked) - n_pairs:]
        kept.sort(key=lambda t: (t[1], t[5][0][0], t[2]))
        qs = sorted(t[0] for t in kept)
        worst_i = max(0, int(math.ceil(0.10 * len(qs))) - 1)
        key = (qs[worst_i], float(np.median(qs)), h_try)
        if best is None or key > best[0]:
            best = (key, kept, h_try)
    if best is None:
        raise RuntimeError(
            f"v2 布局容量不足：同区段、h≥{MIN_CELL_H}、净空≥{min_clear:g}、"
            f"笔画宽≥{min_w:g} 的差分对少于 {n_pairs}")

    _key, kept, _h = best
    out: List[Cell] = []
    for q, ti, iv_i, iv_y0, iv_y1, seg_a, seg_b in kept:
        risk = 1.0 / max(q, 1e-3)
        out.append(_cells_from_segment(
            ti, seg_a, d, margin, len(out), iv_i, iv_y0, iv_y1, risk))
        out.append(_cells_from_segment(
            ti, seg_b, d, margin, len(out), iv_i, iv_y0, iv_y1, risk))

    _assert_v2(out, d)
    return out


def _assert_v2(cells: Sequence[Cell], d: float) -> None:
    if len(cells) != N_CELLS:
        raise RuntimeError(f"v2 码元数 {len(cells)} != {N_CELLS}")
    min_clear = float(min_clearance_for_depth(d))
    min_w = float(min_stroke_w_for_depth(d))
    for i in range(0, len(cells), 2):
        a, b = cells[i], cells[i + 1]
        if a.track != b.track or a.interval != b.interval:
            raise RuntimeError(f"v2 差分对 {i // 2} 跨了区段或轨迹")
        h = min(a.height, b.height)
        if h < MIN_CELL_H:
            raise RuntimeError(f"v2 码元高度 {h} < {MIN_CELL_H}")
        if abs(a.y_mid - b.y_mid) > 1.2 * h + 1e-6:
            raise RuntimeError(
                f"v2 差分对 {i // 2} 的 |Δy|={abs(a.y_mid - b.y_mid):.2f} > 1.2h")
        if min(a.clearance, b.clearance) + 1e-6 < min_clear:
            raise RuntimeError(
                f"v2 净空 {min(a.clearance, b.clearance):.1f} < {min_clear:.0f}")
        if min(a.width, b.width) + 1e-6 < min_w:
            raise RuntimeError(
                f"v2 笔画宽 {min(a.width, b.width):.1f} < {min_w:.0f}")


def derive_pair_perm(cells: Sequence[Cell]) -> Tuple[int, ...]:
    """把几何差分对分进 10 个 Hamming 组。

    最脆的对先发牌，每组先各拿一个，再尽量补不同轨迹、不同字母。
    返回值 PAIR_PERM[bit] = 几何对编号。v1 不用这个，v1 是恒等置换。
    组内不再做第二套交织。
    """
    n_pairs = len(cells) // 2
    if n_pairs != N_BITS:
        raise RuntimeError(f"差分对 {n_pairs} != {N_BITS}")
    risks = [max(cells[2 * p].pair_risk, cells[2 * p + 1].pair_risk)
             for p in range(n_pairs)]
    order = sorted(range(n_pairs), key=lambda p: (-risks[p], p))
    groups: List[List[int]] = [[] for _ in range(N_BITS // 7)]
    for p in order:
        a = cells[2 * p]
        cands = [g for g in range(len(groups)) if len(groups[g]) < 7]

        def _score(g: int, aa=a, pp=p) -> Tuple:
            tracks = {cells[2 * q].track for q in groups[g]}
            letters = {cells[2 * q].letter_i for q in groups[g]}
            new_tr = 0 if aa.track in tracks else 1
            new_lt = 0 if aa.letter_i in letters else 1
            return (new_tr, new_lt, -len(groups[g]), -g)

        groups[max(cands, key=_score)].append(p)
    perm: List[int] = []
    for group in groups:
        perm.extend(sorted(group))
    if sorted(perm) != list(range(n_pairs)):
        raise RuntimeError("PAIR_PERM 不是 70 长置换")
    return tuple(perm)


# 默认 v2@6 的冻结置换。由 derive_pair_perm 在该布局上生成一次后写死。
# 测试会核对它仍等于当前布局的推导结果；对不上就不要悄悄改印版。
PAIR_PERM_V2: Tuple[int, ...] = (
    10, 24, 31, 37, 53, 56, 62,
    6, 20, 33, 41, 55, 61, 67,
    9, 18, 26, 32, 38, 47, 50,
    2, 19, 27, 35, 40, 57, 59,
    5, 13, 14, 25, 48, 64, 68,
    1, 23, 30, 39, 45, 66, 69,
    0, 8, 28, 29, 46, 51, 63,
    3, 11, 16, 22, 36, 60, 65,
    7, 15, 17, 21, 42, 49, 58,
    4, 12, 34, 43, 44, 52, 54,
)
# v1 @ depth 1.8 的码元几何（含测量窗）。旧印版对得上这个哈希。
V1_GEOMETRY_SHA1: str = "ed2c4135922d67afd4f507f052ddfc2f7e3f895f"


def build_layout(mask1: np.ndarray, n_cells: int = N_CELLS, d: float = MOD_DEPTH,
                 margin: int = CELL_MARGIN, min_clear: int = MIN_CLEARANCE,
                 version: int = LAYOUT_VERSION, min_track: int = 50) -> List[Cell]:
    """从二值字形掩膜推导码元布局。

    version=1 锁死历史印版（min_clear=20，恒等配对），忽略 min_clear。
    version=2 是默认：同区段配对，净空/笔画宽/端裁由 d 推导。
    """
    version = int(version)
    if version == 1:
        return _build_layout_v1(mask1, n_cells=n_cells, d=float(d),
                                margin=int(margin), min_track=min_track)
    if version == 2:
        return _build_layout_v2(mask1, n_cells=n_cells, d=float(d),
                                margin=int(margin), min_track=min_track)
    raise ValueError(f"未知布局版本 {version}")


# ==============================================================================
# 5. 模块 1：水印嵌入器
# ==============================================================================
# ==============================================================================

class WatermarkEmbedder:
    """绘制 ARROW 底版，并把 70 bit 码字以**平滑正弦宽度调制**的方式嵌入。

    调制方式（对每个差分对 p，两个 Cell 记为 A、B）：
        bit = 1  ->  A 的宽度按 +sin 起伏, B 按 -sin 起伏
        bit = 0  ->  反之

    关键实现：不做矩形方波切割，而是把边沿偏移写成 y 的连续函数
        e(y) = sign * d * sin(pi * t),  t = (y - y0) / (y1 - y0)
    在 Cell 边界处 e = 0，因此相邻 Cell 之间**没有台阶**，整条边呈连续流线型微波。
    实现上先用全局膨胀/腐蚀各做一次得到 e=+d 与 e=-d 两个端点形态，再按逐行
    系数做亚像素混合 —— 混合比例直接等价于边沿位移量，精度可达 1/255 px。
    """

    def __init__(self, design: Optional[ArrowDesign] = None, verbose: bool = True,
                 depth: Optional[float] = None, layout_version: Optional[int] = None,
                 window_pow: Optional[float] = None):
        if design is None:
            kw: Dict = {"verbose": verbose}
            if depth is not None:
                kw["depth"] = depth
            if layout_version is not None:
                kw["layout_version"] = layout_version
            if window_pow is not None:
                kw["window_pow"] = window_pow
            design = ArrowDesign(**kw)
        elif depth is not None and abs(float(depth) - float(design.depth)) > 1e-6:
            raise ValueError(
                f"depth={depth} 与 design.depth={design.depth} 不一致")
        self.design = design
        self.depth = float(design.depth)
        self.verbose = verbose
        if len(self.design.cells) != N_CELLS:
            raise RuntimeError(f"布局码元数 {len(self.design.cells)} != {N_CELLS}")

    # -------- 单 Cell 的平滑调制 --------
    def _apply_cell(self, mask4: np.ndarray, cell: Cell, sign: int, d: float) -> None:
        """把边沿偏移写成连续正弦，再按行做**真实几何位移**。

        e(y) = sign * d * sin(pi * t)   （t 为该行在 Cell 内的归一化位置）
        在 Cell 两端 e=0，因此相邻 Cell 之间没有台阶，整条边是连续流线型微波。

        实现要点：偏移量量化到 1/SS 像素（4x 域 1px = 1x 域 0.25px），
        按相同量化值把连续行分组成带，每带做一次水平膨胀/腐蚀。
        这样得到的是**真实的边沿位置移动**（而不是半透明环带），
        半高宽测量才能线性地反映出来。
        """
        pad = d + 3.0
        ys = np.array([cell.y0, cell.y1], float)
        cs = cell.center_at(ys)
        X0 = int(math.floor((cs.min() - cell.width * 0.5 - pad) * SS))
        X1 = int(math.ceil((cs.max() + cell.width * 0.5 + pad) * SS))
        Y0 = int(round(cell.y0 * SS))
        Y1 = int(round(cell.y1 * SS))
        X0 = max(0, X0)
        X1 = min(mask4.shape[1], X1)
        Y0 = max(0, Y0)
        Y1 = min(mask4.shape[0], Y1)
        n = Y1 - Y0
        if X1 - X0 < 3 or n < 3:
            return
        t = (np.arange(n, dtype=np.float32) + 0.5) / float(n)
        pow_ = float(self.design.window_pow)
        kk = np.rint(sign * d * np.sin(np.pi * t) ** pow_ * SS).astype(np.int32)
        i = 0
        while i < n:
            j = i + 1
            while j < n and kk[j] == kk[i]:
                j += 1
            kv = int(kk[i])
            if kv != 0:
                y0 = Y0 + i
                y1 = Y0 + j
                roi = mask4[y0:y1, X0:X1]
                ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * abs(kv) + 1, 1))
                mask4[y0:y1, X0:X1] = (cv2.dilate(roi, ker) if kv > 0
                                       else cv2.erode(roi, ker))
            i = j

    # -------- 主入口 --------
    def embed(self, digits: str) -> Dict:
        bits = encode_payload(digits)
        mask4 = self.design.mask4.copy()
        cells = self.design.cells
        perm = self.design.pair_perm
        for i, b in enumerate(bits):
            # 只在这里做一次置换。v1 是恒等，不会和别的交织叠两次。
            p = int(perm[i])
            cA = cells[2 * p]
            cB = cells[2 * p + 1]
            scale = 0.5 * (float(cA.depth_scale) + float(cB.depth_scale))
            d = self.depth * scale
            sign = 1 if b == 1 else -1
            self._apply_cell(mask4, cA, sign, d)
            self._apply_cell(mask4, cB, -sign, d)
        plate = self.design.render(mask4)
        if self.verbose:
            print(f"[Embed] 载荷 {digits} -> 70 bit 码字 -> {N_CELLS} 个码元，"
                  f"layout=v{self.design.layout_version} "
                  f"峰值 ±{self.depth:g}px  perm={self.design.pair_perm_id}")
        return {"digits": digits, "bits": bits, "plate": plate, "mask4": mask4,
                "depth": self.depth, "layout_version": self.design.layout_version,
                "pair_perm_id": self.design.pair_perm_id,
                "window_pow": self.design.window_pow}


# ==============================================================================
# 6. 模块 2：陶瓷印制 + 手机拍摄物理退化模拟器
# ==============================================================================

class CeramicCamSimulator:
    """把"底版"投到陶瓷釉面上，再模拟手机微距拍摄的 5 类复合扰动。

    物理链路：
        制版图 --(a 油墨扩散/烧制形变)--> 承印层
               --(贴到陶瓷砖 + 环境光)--> 场景
               --(b 三维透视/单应变换)--> 相机传感器
               --(c 釉面高光 + 渐变阴影)--> 辐照度
               --(d 微距失焦 + 运动模糊)--> 光学模糊
               --(e 传感器噪声 + JPEG)--> 最终 JPG
    """

    def __init__(self, scene_w: int = 2800, scene_h: int = 1700,
                 out_w: int = 2600, out_h: int = 1600, seed: int = 20240501):
        self.scene_w = scene_w
        self.scene_h = scene_h
        self.out_w = out_w
        self.out_h = out_h
        self.seed = seed
        self._bg_cache: Optional[np.ndarray] = None

    # ---------- 陶瓷底 ----------
    def _ceramic_bg(self) -> np.ndarray:
        """陶瓷底纹。用固定种子生成，保证 simulate() 每次调用互相独立、可复现。"""
        h, w = self.scene_h, self.scene_w
        if self._bg_cache is not None:
            return self._bg_cache.copy()
        rng = np.random.default_rng(90210)
        low = rng.normal(0, 1, (h // 10 + 2, w // 10 + 2)).astype(np.float32)
        low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
        low = cv2.GaussianBlur(low, (0, 0), 11)
        low /= (low.std() + 1e-6)
        bg = 246.5 + low * 2.0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w * 0.5) / (w * 0.5)) ** 2 + ((yy - h * 0.5) / (h * 0.5)) ** 2)
        bg -= 2.5 * np.clip(r - 0.55, 0, None)
        bg = np.clip(bg, 0, 255)
        self._bg_cache = bg
        return bg.copy()

    # ---------- 透视投影 ----------
    @staticmethod
    def _project_quad(sw: int, sh: int, tilt_deg: float, roll_deg: float,
                      rng: np.random.Generator) -> np.ndarray:
        """针孔相机模型：平面绕水平轴倾斜 tilt，再绕光轴滚转 roll。

        物点 (X,Y,0)，相机在 -Z 方向距离 1，则
            u = X / (1 - Y sin t),  v = Y cos t / (1 - Y sin t)
        这是一个真正的单应（keystone），而非简单的仿射缩放。
        """
        t = math.radians(tilt_deg)
        ph = math.radians(roll_deg)
        hs = sh / float(sw)
        pts = []
        for X, Y in [(-1.0, -hs), (1.0, -hs), (1.0, hs), (-1.0, hs)]:
            den = 1.0 - Y * math.sin(t)
            u = X / den
            v = Y * math.cos(t) / den
            ur = u * math.cos(ph) - v * math.sin(ph)
            vr = u * math.sin(ph) + v * math.cos(ph)
            pts.append((ur, vr))
        return np.array(pts, np.float32)

    # ---------- 运动模糊核 ----------
    @staticmethod
    def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
        length = max(1, int(length))
        if length <= 1:
            return np.ones((1, 1), np.float32)
        k = np.zeros((length, length), np.float32)
        cv2.line(k, (0, length // 2), (length - 1, length // 2), 1.0, 1)
        M = cv2.getRotationMatrix2D((length * 0.5 - 0.5, length * 0.5 - 0.5), angle_deg, 1.0)
        k = cv2.warpAffine(k, M, (length, length))
        s = k.sum()
        return k / s if s > 0 else np.ones((1, 1), np.float32) / 1.0

    # ---------- 主入口 ----------
    def simulate(self, plate: np.ndarray, level: str = "moderate",
                 rng: Optional[np.random.Generator] = None) -> Dict:
        if level not in LEVELS:
            raise ValueError(f"未知扰动等级 {level}")
        rng = rng or np.random.default_rng(self.seed)
        p = LEVELS[level]

        def U(key, is_int=False):
            lo, hi = p[key]
            if hi <= lo:
                v = lo
            else:
                v = rng.uniform(lo, hi)
            return int(round(v)) if is_int else float(v)

        ink_r = U("ink")
        tilt = U("tilt")
        roll = U("roll")
        spec = U("spec")
        defocus = U("defocus")
        motion = U("motion", True)
        noise = U("noise")
        jpeg_q = U("jpeg", True)
        shade = U("shade")
        m_ang = rng.uniform(0, 180)

        # ---------- (a) 油墨扩散 / 烧制形变 ----------
        cov = (BG_LEVEL - plate.astype(np.float32)) / float(BG_LEVEL - INK_LEVEL)
        cov = np.clip(cov, 0.0, 1.0)
        if abs(ink_r) > 0.05:
            r = int(round(abs(ink_r)))
            if r >= 1:
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                cov = cv2.dilate(cov, ker) if ink_r > 0 else cv2.erode(cov, ker)
            else:
                # 亚像素级：用模糊近似
                s = abs(ink_r) * 1.2
                cov = cv2.GaussianBlur(cov, (0, 0), s)
        # 印刷 MTF（油墨边缘不会是数学上的硬边）
        cov = cv2.GaussianBlur(cov, (0, 0), 0.65)

        # ---------- 贴到陶瓷砖 ----------
        bg = self._ceramic_bg()
        ox = (self.scene_w - CANVAS_W) // 2 + int(rng.integers(-40, 41))
        oy = (self.scene_h - CANVAS_H) // 2 + int(rng.integers(-30, 31))
        ox = max(0, min(self.scene_w - CANVAS_W, ox))
        oy = max(0, min(self.scene_h - CANVAS_H, oy))
        # 油墨颜色批次差异（深黑 / 浅灰）
        ink_val = float(rng.uniform(46, 82))
        patch = bg[oy:oy + CANVAS_H, ox:ox + CANVAS_W]
        patch *= (1.0 - cov)
        patch += cov * ink_val
        bg[oy:oy + CANVAS_H, ox:ox + CANVAS_W] = patch

        # ---------- (b) 三维透视 ----------
        quad = self._project_quad(self.scene_w, self.scene_h, tilt, roll, rng)
        # 映射到输出画布：先归一化到输出尺度，保证画面被完全覆盖
        bw = quad[:, 0].max() - quad[:, 0].min()
        bh = quad[:, 1].max() - quad[:, 1].min()
        scale = max(self.out_w / bw, self.out_h / bh) * 1.06
        cx, cy = quad[:, 0].mean(), quad[:, 1].mean()
        dst = (quad - np.array([cx, cy], np.float32)) * scale
        dst += np.array([self.out_w * 0.5, self.out_h * 0.5], np.float32)
        src = np.array([[0, 0], [self.scene_w, 0],
                        [self.scene_w, self.scene_h], [0, self.scene_h]], np.float32)
        Hm = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        img = cv2.warpPerspective(bg, Hm, (self.out_w, self.out_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

        # ---------- (c) 釉面高光 + 环境光渐变 ----------
        h, w = img.shape
        if spec > 0.01:
            for _ in range(int(rng.integers(1, 3))):
                gx = rng.uniform(0.10, 0.90) * w
                gy = rng.uniform(0.10, 0.90) * h
                sx = rng.uniform(0.08, 0.26) * w
                sy = rng.uniform(0.05, 0.18) * h
                ang = rng.uniform(0, math.pi)
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                xr = (xx - gx) * math.cos(ang) + (yy - gy) * math.sin(ang)
                yr = -(xx - gx) * math.sin(ang) + (yy - gy) * math.cos(ang)
                g = np.exp(-(xr ** 2 / (2 * sx ** 2) + yr ** 2 / (2 * sy ** 2)))
                amp = spec * float(rng.uniform(0.7, 1.0))
                img += (g * amp * (255.0 - img)).astype(np.float32)
        if shade > 0.01:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            a = rng.uniform(0, 2 * math.pi)
            grad = (np.cos(a) * (xx / w - 0.5) + np.sin(a) * (yy / h - 0.5))
            grad = grad / (np.abs(grad).max() + 1e-6)
            img *= (1.0 + shade * grad)
        img = np.clip(img, 0, 255)

        # ---------- (d) 微距失焦 + 运动模糊 ----------
        if defocus > 0.05:
            img = cv2.GaussianBlur(img, (0, 0), defocus)
        if motion >= 2:
            img = cv2.filter2D(img, -1, self._motion_kernel(motion, m_ang))

        # ---------- (e) 传感器噪声 + JPEG ----------
        if noise > 0.05:
            # 泊松散粒噪声 + 高斯读出噪声
            gain = 26.0
            img = rng.poisson(np.clip(img, 1, 255) * gain / 255.0).astype(np.float32) * 255.0 / gain
            img += rng.normal(0, noise, img.shape).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return {"image": img, "level": level, "plate_origin": (ox, oy),
                "params": dict(ink=ink_r, tilt=tilt, roll=roll, spec=spec,
                               defocus=defocus, motion=motion, noise=noise,
                               jpeg=jpeg_q, shade=shade, motion_angle=m_ang),
                "homography": Hm}


# ==============================================================================
# 7. 模块 3：鲁棒透视矫正 + 亚像素盲解码器
# ==============================================================================

# OpenCV 整数倍 INTER_LINEAR / INTER_CUBIC 放大后，行和列会落进线性插值子空间。
# 真照片不会。logo_scale 用单应在上传像素上的宽度，再除以这个倍数，
# 避免「把小图插值放大」绕过 0.1。阈值卡在「干净插值」和「只是有点糊」之间：
#   真 4× 线性 < 0.006，2× 大画幅在 4× 检验上约 0.020（不能当成 4×）；
#   真 2× 线性 < 0.006，门控原图约 0.053，再高斯糊 σ≈1.2 仍约 0.013。
# σ≈1.6 的重糊会碰到 2× 门，那种帧波形已经没了，不会因此被接受。
_UPSCALE_SCORE_MAX = {4: 0.018, 2: 0.012}


def _solve_tridiag(diag: np.ndarray, off: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a symmetric tridiagonal system. ``off`` is the lower/upper diagonal."""
    n = int(diag.shape[0])
    d = diag.copy()
    b = rhs.copy()
    for i in range(n - 1):
        if abs(d[i]) < 1e-12:
            d[i] = 1e-12
        w = off[i] / d[i]
        d[i + 1] -= w * off[i]
        b[i + 1] -= w * b[i]
    x = np.zeros(n, np.float64)
    x[-1] = b[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - off[i] * x[i + 1]) / d[i]
    return x


def _linear_resample_residual(row: np.ndarray, factor: int) -> Optional[float]:
    """RMS residual / std after projecting one line onto OpenCV's linear resize.

    ``cv2.resize(..., INTER_LINEAR)`` samples at
    ``(i + 0.5) * (src/dst) - 0.5``. A line produced that way fits with a
    residual around quantization; a native line does not. Returns None when
    the line is too short or too flat to judge.
    """
    f = int(factor)
    n = (int(row.shape[0]) // f) * f
    if n < f * 16:
        return None
    src_n = n // f
    y = np.ascontiguousarray(row[:n], dtype=np.float64)
    sd = float(y.std())
    if sd < 1.5:
        return None
    scale = src_n / float(n)
    sx = (np.arange(n, dtype=np.float64) + 0.5) * scale - 0.5
    sx = np.clip(sx, 0.0, src_n - 1.0)
    i0 = np.floor(sx).astype(np.int32)
    a = sx - i0
    i1 = np.minimum(i0 + 1, src_n - 1)
    w0 = 1.0 - a
    w1 = a
    same = i1 == i0
    diag = np.zeros(src_n, np.float64)
    off = np.zeros(max(src_n - 1, 0), np.float64)
    rhs = np.zeros(src_n, np.float64)
    np.add.at(diag, i0, w0 * w0)
    np.add.at(rhs, i0, w0 * y)
    if np.any(~same):
        np.add.at(diag, i1[~same], (w1 * w1)[~same])
        np.add.at(rhs, i1[~same], (w1 * y)[~same])
        idx = i0[~same]
        ok = idx < src_n - 1
        if np.any(ok):
            np.add.at(off, idx[ok], (w0 * w1)[~same][ok])
    coef = _solve_tridiag(diag, off, rhs)
    recon = w0 * coef[i0]
    if np.any(~same):
        recon = recon.copy()
        recon[~same] += (w1 * coef[i1])[~same]
    if np.any(same):
        recon = np.array(recon, dtype=np.float64, copy=True)
        recon[same] += (w1 * coef[i0])[same]
    res = y - recon
    return float(np.sqrt(np.mean(res * res)) / sd)


def _upscale_axis_score(gray: np.ndarray, factor: int, horizontal: bool,
                        n_lines: int = 24) -> float:
    h, w = gray.shape[:2]
    length = w if horizontal else h
    other = h if horizontal else w
    if length < factor * 16 or other < 8:
        return 9.0
    if horizontal:
        stds = gray.std(axis=1)
        idx = np.argsort(stds)[-min(n_lines, other):]
        ratios = [_linear_resample_residual(gray[int(i)], factor) for i in idx]
    else:
        stds = gray.std(axis=0)
        idx = np.argsort(stds)[-min(n_lines, other):]
        ratios = [_linear_resample_residual(gray[:, int(i)], factor) for i in idx]
    ratios = [r for r in ratios if r is not None]
    if not ratios:
        return 9.0
    return float(np.median(ratios))


def interpolation_upscale_factor(image: np.ndarray) -> int:
    """Return 4, 2, or 1.

    4 or 2 means the whole frame is an integer linear/cubic upsample.
    Callers divide the homography's logo_scale by this so a client-side
    interpolate-to-inflate does not move the 0.1 gate. A native photo,
    and a real downscale such as the 0.077 tiny logo, returns 1.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    h, w = gray.shape[:2]
    for factor, limit in ((4, _UPSCALE_SCORE_MAX[4]), (2, _UPSCALE_SCORE_MAX[2])):
        if min(h, w) < factor * 32:
            continue
        rx = _upscale_axis_score(gray, factor, True)
        ry = _upscale_axis_score(gray, factor, False)
        if max(rx, ry) < limit:
            return factor
    return 1


class WatermarkDecoder:
    """盲解码：字形自同步 → 单应矫正 → ECC 亚像素精配 → 法向亚像素积分 → 差分判决。

    全程不依赖任何人工标记。空间基准来自 "ARROW" 字母自身的凸包四角，
    与设计稿的同一算法结果一一对应。
    """

    def __init__(self, design: ArrowDesign, use_ecc_refine: bool = True,
                 verbose: bool = False, use_soft_decode: bool = True,
                 use_sift: bool = True):
        self.design = design
        self.use_soft_decode = use_soft_decode
        self.use_sift = use_sift
        self.cells = design.cells
        self.ref_quad = design.ref_quad.astype(np.float32)
        self.tmpl = design.template.astype(np.float32) / 255.0
        self.use_ecc_refine = use_ecc_refine
        self.verbose = verbose
        # 半分辨率模板（ECC 精配用）
        self._ecc_scale = 0.5
        self._ecc_iters = 30
        self._ecc_eps = 1e-6
        tw = int(round(CANVAS_W * self._ecc_scale))
        th = int(round(CANVAS_H * self._ecc_scale))
        self._tmpl_half = cv2.resize(self.tmpl, (tw, th), interpolation=cv2.INTER_AREA)
        # 1/4 分辨率模板向量（候选四边形 NCC 打分用）
        qs = 0.25
        self._quad_scale = qs
        self._tmpl_q = cv2.resize(self.tmpl, (int(round(CANVAS_W * qs)),
                                              int(round(CANVAS_H * qs))),
                                  interpolation=cv2.INTER_AREA)
        _v = self._tmpl_q.ravel() - self._tmpl_q.mean()
        self._tmpl_vec = _v
        self._tmpl_norm = float(np.linalg.norm(_v))
        # 特征点定位（SIFT + RANSAC）：与亮度、背景、构图完全无关，
        # 是最稳的一条定位路径。模板描述子只算一次。
        self._sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.01) \
            if use_sift else None
        self._sift_tpl = (self._sift.detectAndCompute(design.template, None)
                          if self._sift is not None else ([], None))
        self._bf = cv2.BFMatcher(cv2.NORM_L2)
        # 粗搜抢救用的 1/8 模板只算一次（每次 recover 再 resize 是纯浪费）
        _rsc = 0.125
        self._recover_sc = _rsc
        self._recover_tmpl = cv2.resize(
            self.tmpl, (int(round(CANVAS_W * _rsc)), int(round(CANVAS_H * _rsc))),
            interpolation=cv2.INTER_AREA).astype(np.float32)
        # 参考四边形的宽高比（用于候选合理性检查）
        w, h = self._quad_wh(self.ref_quad)
        self._ref_ar = w / max(h, 1e-6)

    # ---------------- 基础几何 ----------------
    @staticmethod
    def _quad_wh(q: np.ndarray) -> Tuple[float, float]:
        w = 0.5 * (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[2] - q[3]))
        h = 0.5 * (np.linalg.norm(q[1] - q[2]) + np.linalg.norm(q[3] - q[0]))
        return float(w), float(h)

    def _plausible(self, q: np.ndarray) -> bool:
        w, h = self._quad_wh(q)
        if h < 2.0 or w < 8.0:
            return False
        ar = w / h
        # 透视会把 5:1 压到 ~3:1，放宽到 [0.5x, 2.2x] 参考宽高比
        return 0.45 * self._ref_ar < ar < 2.2 * self._ref_ar

    # ---------------- 二值化 ----------------
    @staticmethod
    def _background_diff(gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """形态学闭运算估计"无墨背景"，返回 (bg-g, 缩放系数)。

        闭运算会填平所有比结构元窄的暗结构（笔画），于是 bg 就是
        "假如这里没有印字会是什么样"。差分 bg-g 即纯墨量，对釉面渐变、
        局部高光、整体曝光都不敏感。

        这一步与阈值无关，却被多阈值循环调用了三遍——缩图、模糊、两次
        大核形态学是全流程最贵的算子之一。拆出来只算一次。
        """
        sc = 0.5
        sm = cv2.resize(gray, (int(gray.shape[1] * sc), int(gray.shape[0] * sc)),
                        interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (3, 3), 0)
        k = int(min(sm.shape) * 0.15) | 1
        k = max(31, min(201, k))
        bg = cv2.morphologyEx(sm, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
        bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
        return cv2.subtract(bg, sm), sc

    @staticmethod
    def _thresh_ink(diff: np.ndarray, thr: float) -> np.ndarray:
        """把背景差分图按阈值二值化成墨迹掩膜（与原实现同序同参）。"""
        ink = (diff > thr).astype(np.uint8) * 255
        return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    @staticmethod
    def _binarize(gray: np.ndarray, thr: float = 38.0) -> Tuple[np.ndarray, float]:
        """单次二值化（保留给需要独立调用的地方）。"""
        diff, sc = WatermarkDecoder._background_diff(gray)
        return WatermarkDecoder._thresh_ink(diff, thr), sc

    # ---------------- 字形自同步 ----------------
    def _ncc_H(self, g_small: np.ndarray, Hm: np.ndarray) -> float:
        """用给定单应把 1/4 分辨率图矫正到标准坐标系，与模板算归一化互相关。"""
        qs = self._quad_scale
        th, tw = self._tmpl_q.shape
        S = np.array([[qs, 0, 0], [0, qs, 0], [0, 0, 1]], np.float32)
        Hs = (S @ Hm.astype(np.float32) @ np.linalg.inv(S)).astype(np.float32)
        w = cv2.warpPerspective(g_small, Hs, (tw, th), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        a = w.ravel() - w.mean()
        na = float(np.linalg.norm(a))
        if na < 1e-6:
            return -1.0
        return float(a.dot(self._tmpl_vec) / (na * self._tmpl_norm + 1e-9))

    def _sift_homography(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """用 SIFT 特征点 + RANSAC 直接估出 image->plate 的单应。

        这条路不依赖二值化、不依赖背景明暗、不依赖 logo 是否占满画面，
        对缩放/旋转/透视/画面杂物都免疫，是最可靠的定位方式。
        返回 None 表示特征点不够（画面太小、过糊或被大面积遮挡）。
        """
        if self._sift is None:
            return None
        kp1, des1 = self._sift_tpl
        if des1 is None or len(kp1) < 8:
            return None
        # SIFT 本身尺度无关，大图先降采样再跑。目标长边实测：1400 最好；
        # 降到 1000 内点数看着更多，但极限档成功率 74%→66%，故维持 1400。
        sc = 1.0
        small = gray
        long_side = max(gray.shape[:2])
        if long_side > 1400:
            sc = 1400.0 / long_side
            small = cv2.resize(gray, (int(round(gray.shape[1] * sc)),
                                      int(round(gray.shape[0] * sc))),
                               interpolation=cv2.INTER_AREA)
        kp2, des2 = self._sift.detectAndCompute(small, None)
        if des2 is None or len(kp2) < 8:
            return None
        try:
            matches = self._bf.knnMatch(des1, des2, k=2)
        except cv2.error:
            return None
        good = [a for a, b in matches if a.distance < 0.75 * b.distance]
        if len(good) < 8:
            return None
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        Hpi, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if Hpi is None or mask is None or int(mask.sum()) < 8:
            return None
        # 把"plate->缩小图"换算回"plate->原图"，再取逆得到 image->plate
        if sc != 1.0:
            D = np.diag([1.0 / sc, 1.0 / sc, 1.0])
            Hpi = (D @ Hpi.astype(np.float64)).astype(np.float32)
        try:
            Hm = np.linalg.inv(Hpi.astype(np.float64)).astype(np.float32)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(Hm)):
            return None
        return Hm

    def _find_transforms(self, gray: np.ndarray) -> List[np.ndarray]:
        """无标记自同步：产生若干候选单应，按 1/4 分辨率 NCC 排序。

        首选基准是**五个字母各自的墨迹质心**（5 点超定单应，最稳）：
        质心是全局积分量，对模糊、油墨胀缩、局部高光都不敏感。
        字母被糊成一团时，退回"整排字母凸包四角"（4 点）方案，
        且凸包简化在主轴坐标系里进行，避免滚转导致四点对应错位。

        顺序：先跑几何候选（便宜）。若最佳 NCC≥0.92，说明字形自同步已经
        足够准，跳过 SIFT（省 ~60–120 ms）。只有几何候选偏弱时才启用 SIFT——
        这与"SIFT 命中后跳过其他候选"不同，后者会掉样本。
        """
        qs = self._quad_scale
        g_small = cv2.resize(gray, (int(gray.shape[1] * qs), int(gray.shape[0] * qs)),
                             interpolation=cv2.INTER_AREA)
        pool: List[Tuple[float, np.ndarray]] = []
        dst_c = np.array(self.design.letter_centroids, np.float32)

        diff, sc = self._background_diff(gray)
        for thr in (38.0, 26.0, 18.0):
            ink = self._thresh_ink(diff, thr)
            n, lab, stats, cents = cv2.connectedComponentsWithStats(ink, 8)
            if n < 3:
                continue
            areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
            amax = float(areas.max()) if areas.size else 0.0
            if amax < 200:
                continue
            ids = [i for i in range(1, n) if areas[i - 1] >= 0.03 * amax]

            # ---- 候选 A：五个字母质心（5 点超定单应） ----
            if len(ids) == 5:
                ids_s = sorted(ids, key=lambda i: cents[i][0])
                src = np.array([cents[i] for i in ids_s], np.float32) / sc
                Hm, _ = cv2.findHomography(src, dst_c, cv2.RANSAC, 4.0)
                if Hm is not None and np.all(np.isfinite(Hm)):
                    pool.append((self._ncc_H(g_small, Hm), Hm.astype(np.float32)))

            # ---- 候选 E：内孔质心（4 点精确单应，极端透视下最稳） ----
            if len(self.design.hole_centroids) == 4:
                bgm = (ink < 128).astype(np.uint8)
                nb, labb, sb, cb = cv2.connectedComponentsWithStats(bgm, 4)
                Hb, Wb = bgm.shape
                hb: List[Tuple[float, float, int]] = []
                for j in range(1, nb):
                    xb, yb, wb, hbb, ab = sb[j]
                    if xb == 0 or yb == 0 or xb + wb >= Wb or yb + hbb >= Hb:
                        continue
                    if ab < 800:
                        continue
                    hb.append((float(cb[j][0]), float(cb[j][1]), int(ab)))
                if len(hb) == 4:
                    hb.sort(key=lambda t: t[0])
                    src = np.array([[t[0], t[1]] for t in hb], np.float32) / sc
                    dst_h = np.array(self.design.hole_centroids, np.float32)
                    Hm, _ = cv2.findHomography(src, dst_h, 0)
                    if Hm is not None and np.all(np.isfinite(Hm)):
                        pool.append((self._ncc_H(g_small, Hm), Hm.astype(np.float32)))

            # ---- 候选 C：二阶矩仿射（天然含剪切，抗局部形变） ----
            lutc = np.zeros(n, np.uint8)
            for i in ids:
                lutc[i] = 255
            yc_, xc_ = np.nonzero(lutc[lab])
            if len(xc_) > 500:
                po = np.column_stack([xc_, yc_]).astype(np.float64) / sc
                mu_o = po.mean(axis=0)
                C_o = np.cov((po - mu_o).T)
                Ld, Vd = np.linalg.eigh(self.design.mom_C)
                Lo, Vo = np.linalg.eigh(C_o)
                Ld = np.maximum(Ld, 1e-6)
                Lo = np.maximum(Lo, 1e-6)
                Dd = np.diag(1.0 / np.sqrt(Ld))
                Do = np.diag(np.sqrt(Lo))
                for sgn in (1.0, -1.0):
                    Qm = np.diag([1.0, sgn])
                    A = Vo @ Do @ Qm @ Dd @ Vd.T
                    if abs(np.linalg.det(A)) < 1e-6:
                        continue
                    # 上式解出的是 模板->照片（p_obs = A·p_design + tt），
                    # 而候选池约定的是 照片->模板，必须取逆再入池。
                    tt = mu_o - A @ self.design.mom_mu
                    Hpi = np.array([[A[0, 0], A[0, 1], tt[0]],
                                    [A[1, 0], A[1, 1], tt[1]],
                                    [0.0, 0.0, 1.0]], np.float64)
                    if abs(np.linalg.det(Hpi)) < 1e-9:
                        continue
                    Hm = np.linalg.inv(Hpi).astype(np.float32)
                    if not np.all(np.isfinite(Hm)):
                        continue
                    pool.append((self._ncc_H(g_small, Hm), Hm))

            # ---- 候选 D：墨迹包围盒 -> 设计墨迹包围盒 ----
            # 对干净印版而言这就是恒等变换；对拍摄图是一个稳健的粗对齐。
            lutb = np.zeros(n, np.uint8)
            for i in ids:
                lutb[i] = 255
            yb_, xb_ = np.nonzero(lutb[lab])
            if len(xb_) > 500:
                ox0, ox1 = xb_.min() / sc, xb_.max() / sc
                oy0, oy1 = yb_.min() / sc, yb_.max() / sc
                bx0, by0, bx1, by1 = self.design.ink_bbox
                if (ox1 - ox0) > 10 and (oy1 - oy0) > 5:
                    sx = (bx1 - bx0) / (ox1 - ox0)
                    sy = (by1 - by0) / (oy1 - oy0)
                    Hm = np.array([[sx, 0.0, bx0 - sx * ox0],
                                   [0.0, sy, by0 - sy * oy0],
                                   [0.0, 0.0, 1.0]], np.float32)
                    pool.append((self._ncc_H(g_small, Hm), Hm))

            # ---- 候选 B：整排字母凸包四角 ----
            lut = np.zeros(n, np.uint8)
            for i in ids:
                lut[i] = 255
            ys, xs = np.nonzero(lut[lab])
            if len(xs) < 800:
                continue
            pts = np.column_stack([xs, ys]).astype(np.float32)
            rect = cv2.minAreaRect(pts.astype(np.int32))
            ang = float(rect[2])
            if ang < -45.0:
                ang += 90.0
            elif ang > 45.0:
                ang -= 90.0
            M = cv2.getRotationMatrix2D((float(rect[0][0]), float(rect[0][1])), ang, 1.0)
            pts_r = cv2.transform(pts.reshape(1, -1, 2), M).reshape(-1, 2)
            hull = cv2.convexHull(pts_r.astype(np.int32))
            peri = cv2.arcLength(hull, True)
            Minv = cv2.invertAffineTransform(M)
            for eps in (0.03, 0.02, 0.04, 0.05, 0.06, 0.08):
                ap = cv2.approxPolyDP(hull, eps * peri, True)
                if len(ap) != 4:
                    continue
                qr = cv2.transform(ap.reshape(1, -1, 2).astype(np.float32), Minv).reshape(4, 2)
                q = order_quad(qr / sc)
                if not self._plausible(q):
                    continue
                Hm = cv2.getPerspectiveTransform(q.astype(np.float32), self.ref_quad)
                pool.append((self._ncc_H(g_small, Hm), Hm.astype(np.float32)))

        best_geo = max((s for s, _ in pool), default=-1.0)
        # ---- 候选 F：SIFT + RANSAC（几何候选不够准时才跑） ----
        # 干净印版/轻中度扰动下几何 NCC 经常 ≥0.92，此时 SIFT 是纯开销。
        # 重度/极限或构图杂乱时几何偏弱，SIFT 才是主力。
        if best_geo < 0.92:
            Hs = self._sift_homography(gray)
            if Hs is not None:
                pool.append((self._ncc_H(g_small, Hs), Hs))

        if not pool:
            return []
        pool.sort(key=lambda t: -t[0])
        out: List[np.ndarray] = []
        for _, Hm in pool:
            if all(float(np.abs(Hm - o).max()) > 1e-3 for o in out):
                out.append(Hm)
            if len(out) >= 4:
                break
        return out

    # ---------------- ECC 亚像素精配 ----------------
    def _ecc_refine(self, warped: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """用模板做 ECC 精配，先试 8 自由度单应，失败回退欧氏变换。"""
        sc = self._ecc_scale
        th, tw = self._tmpl_half.shape
        w_small = cv2.resize(warped.astype(np.float32) / 255.0, (tw, th),
                             interpolation=cv2.INTER_AREA)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self._ecc_iters, self._ecc_eps)
        S = np.array([[sc, 0, 0], [0, sc, 0], [0, 0, 1]], np.float32)
        Si = np.linalg.inv(S)
        # 单应已经拿到可用解（cc≥0.55）就不必再跑欧氏——欧氏几乎不会更好，
        # 却要再付一次 findTransformECC。
        motions = ((cv2.MOTION_HOMOGRAPHY, (3, 3)),
                   (cv2.MOTION_EUCLIDEAN, (2, 3)))
        for motion, wshape in motions:
            try:
                wm = np.eye(wshape[0], wshape[1], dtype=np.float32)
                cc, wm = cv2.findTransformECC(self._tmpl_half, w_small, wm,
                                              motion, crit, None, 5)
            except Exception:
                continue
            if not np.isfinite(cc) or cc < 0.40:
                continue
            if motion == cv2.MOTION_HOMOGRAPHY:
                corners = np.array([[0, 0], [tw, 0], [tw, th], [0, th]],
                                   np.float32).reshape(1, -1, 2)
                moved = cv2.perspectiveTransform(corners, wm).reshape(-1, 2)
                if float(np.abs(moved - corners.reshape(-1, 2)).max()) > 24.0:
                    continue
                Hf = (Si @ wm.astype(np.float32) @ S).astype(np.float32)
                warped = cv2.warpPerspective(
                    warped, Hf, (CANVAS_W, CANVAS_H),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE)
                shift = float(np.abs(moved - corners.reshape(-1, 2)).max()) / float(sc)
                return warped, {"cc": float(cc), "motion": int(motion),
                                "residual": shift}
            else:
                wf = wm.copy()
                wf[0, 2] /= sc
                wf[1, 2] /= sc
                warped = cv2.warpAffine(
                    warped, wf, (CANVAS_W, CANVAS_H),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE)
                residual = float(math.hypot(float(wf[0, 2]), float(wf[1, 2])))
                return warped, {"cc": float(cc), "motion": int(motion),
                                "residual": residual}
        return warped, {"rejected": True}

    def _recover_alignment(self, warped: np.ndarray) -> Tuple[np.ndarray, float]:
        """粗对齐抢救：1/8 分辨率上由粗到细搜索，扩大 ECC 收敛域。

        只在 ECC 相关性偏低时启用。原实现是 7×5×5=175 次全组合；
        改为两阶段：先搜旋转×缩放（剪切=0），再在最优点附近扫剪切与
        微调角度——同等覆盖、约 1/3 代价。
        """
        sc = self._recover_sc
        t = self._recover_tmpl
        th, tw = t.shape
        w = cv2.resize(warped.astype(np.float32) / 255.0, (tw, th),
                       interpolation=cv2.INTER_AREA)
        pad = 26
        cx, cy = tw * 0.5, th * 0.5
        best_s, best_T = -2.0, None
        best_ang, best_scl = 0.0, 1.0

        def _score(M3: np.ndarray) -> Tuple[float, np.ndarray]:
            w2 = cv2.warpAffine(w, M3[:2], (tw, th), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
            w2p = cv2.copyMakeBorder(w2, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            res = cv2.matchTemplate(w2p, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, mloc = cv2.minMaxLoc(res)
            T = np.array([[1, 0, mloc[0] - pad], [0, 1, mloc[1] - pad],
                          [0, 0, 1]], np.float32)
            return float(mx), T @ M3

        # 阶段 1：旋转 × 缩放（35 次）
        for ang in (-9, -6, -3, 0, 3, 6, 9):
            for scl in (0.90, 0.95, 1.0, 1.05, 1.10):
                M0 = cv2.getRotationMatrix2D((cx, cy), ang, scl)
                M3 = np.vstack([M0, [0, 0, 1]]).astype(np.float32)
                mx, T = _score(M3)
                if mx > best_s:
                    best_s, best_T, best_ang, best_scl = mx, T, float(ang), float(scl)
        # 阶段 2：在最优点附近扫剪切 + 微角度（5×5=25 次）
        for sh in (-0.14, -0.07, 0.0, 0.07, 0.14):
            for dang in (-2.0, -1.0, 0.0, 1.0, 2.0):
                M0 = cv2.getRotationMatrix2D((cx, cy), best_ang + dang, best_scl)
                Sh = np.array([[1.0, sh, -sh * cy], [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]], np.float32)
                M3 = Sh @ np.vstack([M0, [0, 0, 1]]).astype(np.float32)
                mx, T = _score(M3)
                if mx > best_s:
                    best_s, best_T = mx, T
        if best_T is None:
            return warped, -1.0
        S = np.array([[sc, 0, 0], [0, sc, 0], [0, 0, 1]], np.float32)
        Hf = (np.linalg.inv(S) @ best_T @ S).astype(np.float32)
        out = cv2.warpPerspective(warped, Hf, (CANVAS_W, CANVAS_H),
                                  flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE)
        return out, best_s

    def _logo_scale(self, H: np.ndarray) -> float:
        """由 image→plate 单应反推照片里 logo 相对底版的尺度（宽比）。"""
        try:
            Hi = np.linalg.inv(H.astype(np.float64)).astype(np.float32)
            q = cv2.perspectiveTransform(
                self.ref_quad.reshape(1, -1, 2), Hi).reshape(-1, 2)
            w = 0.5 * (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[3] - q[2]))
            return float(w / CANVAS_W)
        except Exception:
            return 1.0

    @staticmethod
    def _restore_edges(warped: np.ndarray, scale: float) -> np.ndarray:
        """真实尺寸拍摄时 logo 往往只占画面一小块，矫正回 2200×600 等于强放大，
        双线性会抹掉亚像素调制（幅度见 MOD_DEPTH）。轻度反锐化遮罩把边缘对比度拉回来。
        scale≥0.45（接近半幅以上）时不处理，避免动干净印版的数值路径。
        """
        if scale >= 0.45:
            return warped
        up = 1.0 / max(scale, 1e-6)
        amt = min(1.35, 0.30 + 0.22 * (up - 2.0))
        if amt <= 0.05:
            return warped
        sig = min(1.7, 0.75 + 0.10 * up)
        blur = cv2.GaussianBlur(warped, (0, 0), sig)
        return np.clip(cv2.addWeighted(warped, 1.0 + amt, blur, -amt, 0),
                       0, 255).astype(np.uint8)

    def sift_roi_box(self, gray: np.ndarray, pad: float = 0.2
                     ) -> Optional[Tuple[int, int, int, int]]:
        """Full-resolution crop box from a coarse SIFT homography.

        SIFT itself may run on a long-side-1400 downsample inside
        ``_sift_homography``; the homography is mapped back before this box
        is measured. ``pad`` expands the projected logo quad (default 0.2)
        and the box is clipped to ``gray``. Callers must crop at this
        resolution and warp the original image once — not resize the crop.
        """
        Hm = self._sift_homography(gray)
        if Hm is None:
            return None
        try:
            Hi = np.linalg.inv(Hm.astype(np.float64))
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(Hi)):
            return None
        quad = cv2.perspectiveTransform(
            self.ref_quad.reshape(1, -1, 2), Hi.astype(np.float32)).reshape(-1, 2)
        if quad.shape != (4, 2) or not np.all(np.isfinite(quad)):
            return None
        x0, y0 = [float(v) for v in quad.min(axis=0)]
        x1, y1 = [float(v) for v in quad.max(axis=0)]
        bw, bh = x1 - x0, y1 - y0
        if bw < 8.0 or bh < 4.0:
            return None
        m = float(pad)
        x0 -= m * bw
        x1 += m * bw
        y0 -= m * bh
        y1 += m * bh
        h, w = gray.shape[:2]
        ix0 = max(0, int(np.floor(x0)))
        iy0 = max(0, int(np.floor(y0)))
        ix1 = min(int(w), int(np.ceil(x1)))
        iy1 = min(int(h), int(np.ceil(y1)))
        if ix1 - ix0 < 32 or iy1 - iy0 < 16:
            return None
        return ix0, iy0, ix1, iy1

    def rectify(self, gray: np.ndarray, source: Optional[np.ndarray] = None,
                origin: Tuple[float, float] = (0.0, 0.0),
                upscale_factor: Optional[int] = None) -> Dict:
        """多起点矫正 + 收敛域抢救，取相关性最高的结果。

        ``source`` + ``origin``：在 1:1 裁切 ``gray`` 上估计单应，再按
        H∘平移从 ``source``（上传原图）一次采样到版面坐标。不要先把裁切
        放大再 warp。logo_scale 是这条单应在 ``source`` 像素里的尺度，
        并除掉整幅整数插值倍数。
        """
        Hs = self._find_transforms(gray)
        if not Hs:
            return {"ok": False, "reason": "sync_failed"}
        ox, oy = float(origin[0]), float(origin[1])
        use_source = source is not None
        T = None
        if use_source:
            T = np.array([[1.0, 0.0, -ox],
                          [0.0, 1.0, -oy],
                          [0.0, 0.0, 1.0]], np.float64)
        best = None
        for k, Hm in enumerate(Hs):
            if use_source:
                H_use = (Hm.astype(np.float64) @ T).astype(np.float32)
                src = source
            else:
                H_use = Hm
                src = gray
            warped = cv2.warpPerspective(src, H_use, (CANVAS_W, CANVAS_H),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
            ecc: Dict = {}
            if self.use_ecc_refine:
                warped, ecc = self._ecc_refine(warped)
                # 粗搜只给排名前两的起点：后面的候选就算拉回来，
                # 也很少超过已经试过的更好起点，却会再付 1–2 次 ECC。
                if float(ecc.get("cc", 0.0)) < 0.80 and k < 2:
                    # ECC 没收敛好：先用粗搜把它拉回正确盆地，再精配一次。
                    # 注意图像与分数必须成对替换——曾经只换了图不换分，
                    # 导致候选之间拿"旧图的分数"比"新图的画质"。
                    base_warped, base_cc = warped, float(ecc.get("cc", 0.0))
                    rec_warped, ms = self._recover_alignment(warped)
                    if ms > 0.3:
                        rec_warped, ecc2 = self._ecc_refine(rec_warped)
                        if float(ecc2.get("cc", 0.0)) > base_cc:
                            warped, ecc = rec_warped, ecc2
                        else:
                            warped, ecc = base_warped, ecc
            cc = float(ecc.get("cc", 0.0))
            if best is None or cc > best["cc"]:
                best = {"ok": True, "warped": warped, "H": H_use, "ecc": ecc,
                        "cc": cc, "tries": k + 1}
            # 与解码入口的 CC_GATE=0.85 对齐：到门限即可停，不必追到 0.88
            if cc >= 0.85:
                break
        # 小尺寸实拍：边缘恢复后再交给宽度测量。
        # 尺度来自原图像素上的 H。整幅被 2×/4× 插值放大时除掉这个倍数，
        # 否则 logo_scale 会跟着像素网格变大、绕过 0.1。
        sc = self._logo_scale(best["H"])
        if upscale_factor is None:
            scale_img = source if use_source else gray
            upscale_factor = interpolation_upscale_factor(scale_img)
        fac = int(upscale_factor or 1)
        if fac > 1:
            sc = sc / float(fac)
        best["logo_scale"] = sc
        best["warped"] = self._restore_edges(best["warped"], sc)
        best.pop("cc", None)
        return best

    # ---------------- 法向亚像素积分 ----------------
    @staticmethod
    def _cell_levels(img: np.ndarray, cell: Cell) -> Tuple[float, float]:
        """在**整块码元**上估计背景灰度与墨迹对比度（而不是逐行估计）。

        逐行估计时，JPEG 块效应与陶瓷纹理会让阈值逐行抖动，抖动直接变成
        宽度噪声——而宽度噪声正是判决的错误来源。整块估计把 ~16 行平均掉，
        抖动降到 1/4；剩下的恒定偏置会被正弦匹配滤波完全消掉。
        """
        H, W = img.shape
        bgs: List[float] = []
        prof: List[np.ndarray] = []
        for y in range(max(0, cell.y0), min(H, cell.y1)):
            cx = cell.center_at(y)
            x0 = max(0, int(round(cx - cell.half_win)))
            x1 = min(W, int(round(cx + cell.half_win)) + 1)
            if x1 - x0 < 8:
                continue
            seg = img[y, x0:x1]
            bgs.append(float(np.median(np.concatenate([seg[:5], seg[-5:]]))))
            prof.append(seg)
        if not bgs:
            return 0.0, 0.0
        bg = float(np.median(bgs))
        P = np.clip(bg - np.concatenate(prof), 0, None)
        ct = float(np.percentile(P, 97.0)) if P.size else 0.0
        return bg, ct

    @staticmethod
    def _row_width(img: np.ndarray, cell: Cell, y: int,
                   levels: Optional[Tuple[float, float]] = None) -> Optional[float]:
        """在单行上测笔画水平宽度（亚像素，半高宽法）。

        levels 给出 (bg, ct) 时使用整块估计的固定阈值；否则退回逐行估计。
        """
        W = img.shape[1]
        cx = cell.center_at(y)
        x0 = int(round(cx - cell.half_win))
        x1 = int(round(cx + cell.half_win)) + 1
        x0 = max(0, x0)
        x1 = min(W, x1)
        if x1 - x0 < 8:
            return None
        seg = img[y, x0:x1]
        if levels is not None:
            bg, ct = levels
            if ct < 18.0:
                return None
            p = np.clip(bg - seg, 0, None)
        else:
            bg = float(np.median(np.concatenate([seg[:5], seg[-5:]])))
            p = np.clip(bg - seg, 0, None)
        if p.size >= 3:
            p = np.convolve(p, np.ones(3, np.float32) / 3.0, mode="same")
        if levels is None:
            ct = _p97(p)
            if ct < 18.0:                  # 该行被高光吃掉 / 无墨
                return None
        t = 0.5 * ct
        idx = np.where(p >= t)[0]
        if len(idx) < 2:
            return None
        l = float(idx[0])
        r = float(idx[-1])
        if l > 0:
            den = p[int(l)] - p[int(l) - 1]
            l = l - 1.0 + (t - p[int(l) - 1]) / (den if abs(den) > 1e-6 else 1e-6)
        if r < p.size - 1:
            den = p[int(r)] - p[int(r) + 1]
            r = r + (p[int(r)] - t) / (den if abs(den) > 1e-6 else 1e-6)
        return r - l + 1.0

    def extract_widths(self, warped: np.ndarray) -> np.ndarray:
        """亚像素宽度 + 正弦匹配滤波 + 成对差分。

        默认走向量化实现（整数列 + INTER_NEAREST，与逐行路径逐位一致）。
        早期双线性 remap 版本曾让中度档掉到 96%，已修正；20×5 回归上
        成功率不低于逐行，耗时约 1/2.5。
        """
        return self._extract_widths_fast(warped)

    def _extract_widths_slow(self, warped: np.ndarray) -> np.ndarray:
        """逐行参考实现（调试/对照用）。"""
        img = warped.astype(np.float32)
        H = img.shape[0]
        prof: List[Tuple[np.ndarray, np.ndarray]] = []
        for cell in self.cells:
            ys: List[float] = []
            ws: List[float] = []
            for y in range(max(0, cell.y0), min(H, cell.y1)):
                w = self._row_width(img, cell, y)
                if w is None:
                    continue
                ys.append(float(y))
                ws.append(float(w))
            prof.append((np.array(ys), np.array(ws)))

        allw = np.concatenate([p[1] for p in prof]) if prof else np.array([1.0])
        self._wmean = float(np.median(allw)) if allw.size else 1.0

        out = np.zeros(N_CELLS, np.float32)
        perm = self.design.pair_perm
        pow_ = float(self.design.window_pow)
        for i in range(N_BITS):
            p = int(perm[i])
            ka, kb = 2 * p, 2 * p + 1
            ya, wa = prof[ka]
            yb, wb = prof[kb]
            if len(wa) < 4 or len(wb) < 4:
                out[2 * i] = float(wa.mean()) if len(wa) else 0.0
                out[2 * i + 1] = float(wb.mean()) if len(wb) else 0.0
                continue
            base = 0.5 * (wa.mean() + wb.mean())
            ta = (ya - self.cells[ka].y0 + 0.5) / float(self.cells[ka].height)
            tb = (yb - self.cells[kb].y0 + 0.5) / float(self.cells[kb].height)
            sa = np.sin(np.pi * ta) ** pow_
            sb = np.sin(np.pi * tb) ** pow_
            out[2 * i] = float(((wa - base) * sa).sum() / max(sa.sum(), 1e-6))
            out[2 * i + 1] = float(((wb - base) * sb).sum() / max(sb.sum(), 1e-6))
        return out

    def _extract_widths_fast(self, warped: np.ndarray) -> np.ndarray:
        """向量化的逐行半高宽测量（与 `_extract_widths_slow` 等价，快约 2.5×）。

        原实现每个码元每行调一次 `_row_width`，140×16 ≈ 2200 次 Python 调用；
        这里把每个码元的采样网格一次性 remap 出来，再用 numpy 批量算阈值与
        亚像素跳变沿。数学与原实现一致：逐行背景取窗口两端中位数，
        对比度取该行 97 分位，阈值 0.5·ct，边缘线性插值。
        """
        img = warped.astype(np.float32)
        H, W = img.shape
        profs: List[Tuple[np.ndarray, np.ndarray]] = []
        for cell in self.cells:
            y0, y1 = max(0, cell.y0), min(H, cell.y1)
            n = y1 - y0
            if n < 2:
                profs.append((np.zeros(0), np.zeros(0)))
                continue
            ys = np.arange(y0, y1, dtype=np.float32)
            cx = cell.x_mid + cell.slope * (ys - cell.y_mid)
            half = int(cell.half_win)
            # 必须按**整数列**取窗口（与原逐行实现一致）。用亚像素 remap 会引入
            # 双线性模糊，让半高宽产生 0.3~0.5 px 的偏差，足以翻掉边缘比特。
            x0i = np.clip(np.rint(cx - half).astype(np.int32), 0, max(W - 1, 0))
            xs = np.clip(x0i[:, None] + np.arange(0, 2 * half + 2)[None, :],
                         0, W - 1).astype(np.float32)
            patch = cv2.remap(img, xs, np.repeat(ys[:, None], xs.shape[1], 1),
                              interpolation=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_REPLICATE)
            bg = np.median(np.concatenate([patch[:, :5], patch[:, -5:]], axis=1), axis=1)
            pv = np.clip(bg[:, None] - patch, 0, None)
            # 必须用 CONSTANT 边界：默认的 REFLECT_101 在窗口首尾列与
            # np.convolve(mode='same') 的零填充不一致，会带来亚像素偏差
            pv = cv2.blur(pv, (3, 1), borderType=cv2.BORDER_CONSTANT)
            ct = np.percentile(pv, 97.0, axis=1)
            thr = 0.5 * ct
            ok = ct >= 18.0
            m = pv >= thr[:, None]
            has = m.any(axis=1) & ok
            rows = np.arange(pv.shape[0])
            nc = pv.shape[1]
            first = np.argmax(m, axis=1)
            last = nc - 1 - np.argmax(m[:, ::-1], axis=1)
            # 亚像素插值：frac 必须钳在 [0,1]。
            # 当墨迹顶到窗口首/末列时（first==0 或 last==nc-1），取到的两点
            # 并非真实跳变沿，分母可能为 0 —— 不钳位会算出 1e5 量级的宽度。
            i0 = np.clip(first, 1, nc - 1)
            v0 = pv[rows, i0 - 1]
            v1 = pv[rows, i0]
            dl = v1 - v0
            frac_l = np.clip((thr - v0) / np.where(np.abs(dl) > 1e-3, dl, 1e-3), 0.0, 1.0)
            left = np.where(first > 0, (i0 - 1) + frac_l, 0.0)
            j0 = np.clip(last, 0, nc - 2)
            w0 = pv[rows, j0]
            w1 = pv[rows, j0 + 1]
            dr = w0 - w1
            frac_r = np.clip((w0 - thr) / np.where(np.abs(dr) > 1e-3, dr, 1e-3), 0.0, 1.0)
            right = np.where(last < nc - 1, j0 + frac_r, float(nc - 1))
            wd = right - left + 1.0
            profs.append((ys[has], wd[has]))

        # 置信度归一化用的笔画中位宽度（原逐行实现里也要维护这个量）
        allw = np.concatenate([pr[1] for pr in profs]) if profs else np.array([1.0])
        self._wmean = float(np.median(allw)) if allw.size else 1.0

        out = np.zeros(N_CELLS, np.float32)
        perm = self.design.pair_perm
        pow_ = float(self.design.window_pow)
        for i in range(N_BITS):
            p = int(perm[i])
            ka, kb = 2 * p, 2 * p + 1
            ya, wa = profs[ka]
            yb, wb = profs[kb]
            if len(wa) < 4 or len(wb) < 4:
                out[2 * i] = float(wa.mean()) if len(wa) else 0.0
                out[2 * i + 1] = float(wb.mean()) if len(wb) else 0.0
                continue
            base = 0.5 * (wa.mean() + wb.mean())
            ta = (ya - self.cells[ka].y0 + 0.5) / float(self.cells[ka].height)
            tb = (yb - self.cells[kb].y0 + 0.5) / float(self.cells[kb].height)
            sa = np.sin(np.pi * ta) ** pow_
            sb = np.sin(np.pi * tb) ** pow_
            out[2 * i] = float(((wa - base) * sa).sum() / max(sa.sum(), 1e-6))
            out[2 * i + 1] = float(((wb - base) * sb).sum() / max(sb.sum(), 1e-6))
        return out

    def extract_masses(self, warped: np.ndarray) -> np.ndarray:
        """墨量积分（消融对照用，非主路径）。"""
        img = warped.astype(np.float32)
        H, W = img.shape
        masses = np.zeros(len(self.cells), np.float32)
        for ci, cell in enumerate(self.cells):
            guard = max(1, min(3, cell.height // 5))
            acc = 0.0
            for y in range(max(0, cell.y0 + guard), min(H, cell.y1 - guard)):
                cx = cell.center_at(y)
                x0 = max(0, int(round(cx - cell.half_win)))
                x1 = min(W, int(round(cx + cell.half_win)) + 1)
                if x1 - x0 < 8:
                    continue
                seg = img[y, x0:x1]
                bg = float(np.median(np.concatenate([seg[:5], seg[-5:]])))
                acc += float(np.clip(bg - seg, 0, None).sum())
            masses[ci] = acc
        return masses

    def decide_bits(self, feat: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """差分判决：M[2i] - M[2i+1] > 0 -> bit 1。

        ``conf`` 仍是像素差分除以笔画中位宽度，留给旧日志。
        ``_soft_n`` 再除以期望幅度 π·depth·scale，满调制约 ±1。
        生产门和软判决用 ``_soft_n``，不拿 1.8px 时期的绝对 conf 当门限。
        """
        bits: List[int] = []
        conf: List[float] = []
        soft_n: List[float] = []
        wm_ = max(float(getattr(self, "_wmean", 1.0)), 1.0)
        perm = self.design.pair_perm
        depth = float(self.design.depth)
        for i in range(N_BITS):
            a = float(feat[2 * i])
            b = float(feat[2 * i + 1])
            dlt = a - b
            bits.append(1 if dlt > 0 else 0)
            conf.append(dlt / (2.0 * wm_))
            p = int(perm[i])
            cA = self.cells[2 * p]
            cB = self.cells[2 * p + 1]
            scale = 0.5 * (float(cA.depth_scale) + float(cB.depth_scale))
            exp = expected_differential_px(depth * scale)
            soft_n.append(dlt / exp if exp > 1e-9 else 0.0)
        self._soft_n = np.array(soft_n, np.float32)
        return bits, np.array(conf, np.float32)

    # ---------------- 主入口 ----------------
    @staticmethod
    def _plate_candidates(gray: np.ndarray, k: int = 3) -> List[np.ndarray]:
        """定位候选：把"亮底卡片"的几种可能都裁出来（按面积从大到小）。

        真实构图里 logo 可能是：整幅就是底版、白底标签贴在深色产品/屏幕上、
        画面里还有别的亮区（窗口、台面）。这里取最大的 k 个亮连通域，
        各自填掉内孔、域外置白、裁出包围盒，交给解码端逐个尝试。
        """
        m = (gray > 170).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n < 2:
            return []
        order = sorted(range(1, n),
                       key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:max(k, 1)]
        out: List[np.ndarray] = []
        for i in order:
            x, y, w, h, a = stats[i]
            if a < 1500 or h < 20:
                continue
            ar = w / float(h)
            if not (2.0 < ar < 6.0):       # 底版宽高比 3.67
                continue
            plate = (lab == i).astype(np.uint8)
            cnts, _ = cv2.findContours(plate, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            filled = np.zeros_like(plate)
            cv2.drawContours(filled, cnts, -1, 1, -1)
            pad = 6
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
            img = gray.copy()
            img[filled == 0] = 255
            out.append(img[y0:y1, x0:x1])
        return out

    @staticmethod
    def _isolate_plate(gray: np.ndarray) -> Optional[np.ndarray]:
        """把"亮底小卡片 + 暗背景"的照片预处理成白底黑字。

        真实拍摄里 logo 常常是白底卡片/标签贴在深色产品或屏幕上。此时
        形态学背景估计会把四周的大片暗区误判成墨迹，自同步直接失效。
        这里取最大的亮连通域、填掉内孔、把域外全部置白，再交给正常流程。
        """
        cands = WatermarkDecoder._plate_candidates(gray, k=1)
        return cands[0] if cands else None

    def _decode_gray(self, gray: np.ndarray, return_debug: bool = False,
                     upscale_factor: Optional[int] = None) -> Dict:
        r = self.rectify(gray, upscale_factor=upscale_factor)
        if not r.get("ok"):
            return {"ok": False, "reason": r.get("reason", "rectify_failed"),
                    "digits": None, "bits": None, "ber": 1.0, "conf": 0.0,
                    "soft_n": None, "mean_conf": 0.0, "mean_soft_n": 0.0,
                    "n_corrected": 0, "reg_ok": False, "reg_residual": 1.0}
        warped = r["warped"]
        feat = self.extract_widths(warped)
        bits, conf = self.decide_bits(feat)
        soft_n = np.asarray(self._soft_n, np.float32)
        digits_hard, n_corr, flags = decode_payload(bits)
        if self.use_soft_decode:
            # 软判决用按期望幅度归一化后的 soft_n。scale 恒为 1 时与 conf 只差一个正的常数。
            digits, n_corr = decode_payload_soft(soft_n)
        else:
            digits = digits_hard
        ecc = dict(r.get("ecc", {}))
        ecc.setdefault("cc", 0.0)          # ECC 被拒时也给出明确的 0
        cc = float(ecc.get("cc", 0.0))
        if ecc.get("residual") is None:
            residual = max(0.0, 1.0 - cc)
        else:
            residual = float(ecc.get("residual"))
        reg_ok = bool(cc >= 0.85) if self.use_ecc_refine else True
        out = {"ok": True, "digits": digits, "digits_hard": digits_hard,
               "payload_valid": digits is not None,
               "bits": bits, "conf": conf, "soft_n": soft_n,
               "widths": feat, "n_corrected": n_corr,
               "mean_conf": float(np.mean(np.abs(conf))),
               "mean_soft_n": float(np.mean(np.abs(soft_n))),
               "ecc": ecc, "cc": cc,
               "reg_ok": reg_ok, "reg_residual": residual,
               "layout_version": self.design.layout_version,
               "depth": float(self.design.depth),
               "pair_perm_id": self.design.pair_perm_id,
               "logo_scale": float(r.get("logo_scale", 1.0)),
               "warped": warped if return_debug else None}
        out.update(readout_qc(soft_n, bits, soft=self.use_soft_decode))
        if return_debug:
            out["H"] = r["H"]
        return out

    def decode(self, image: np.ndarray, return_debug: bool = False) -> Dict:
        """解码入口：**先定位再解码**，而不是先硬解再补救。

        定位分两路产出假设，各自完整解一次，取置信度更高的：

          路径 A：整幅图直接解（logo 本来就占满画面时最合适）
          路径 B：先隔离"亮底卡片"再解（logo 是白底标签贴在深色产品/屏幕上时）

        真实尺寸微距拍摄时，2.2 cm logo 在大画幅里往往只是一张小白卡。
        这时先跑路径 B 能少付一次失败的整幅配准；反之 logo 已占满画面
        时仍先走路径 A。路径 B 只在隔离区明显小于原图时才跑。
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # 整幅一次判定。裁切候选沿用同一倍数，避免裁切块自己检不出放大。
        upscale_factor = interpolation_upscale_factor(gray)

        # 择优规则：配准不可信（cc 过低）的解，置信度再高也不能信 ——
        # 实测：一条错误路径把整幅裁错，宽度测量全乱，conf 虚高到 0.12，
        # 但它的 cc 是 0.000。所以先按 cc 过闸，再比置信度。
        CC_GATE = 0.85

        def _rank(d):
            return (1 if d.get("cc", 0.0) >= CC_GATE else 0,
                    d.get("mean_soft_n", 0.0))

        plates = self._plate_candidates(gray, k=3)
        # 小卡片判定：最大亮连通域面积 < 画面 45%，且画幅明显大于底版
        plate_first = False
        if plates:
            frac = plates[0].size / float(max(gray.size, 1))
            big = max(gray.shape[:2]) >= 1600
            plate_first = frac < 0.45 and big

        best: Optional[Dict] = None

        def _consider(res: Dict, path: str, isolated: bool = False) -> None:
            nonlocal best
            res = dict(res)
            res["path"] = path
            if isolated:
                res["isolated"] = True
            if best is None or _rank(res) > _rank(best):
                best = res

        def _good_enough(gate_cc: float, gate_soft_n: float) -> bool:
            return (best is not None
                    and best.get("cc", 0.0) >= gate_cc
                    and best.get("mean_soft_n", 0.0) >= gate_soft_n)

        if not plate_first:
            _consider(self._decode_gray(gray, return_debug, upscale_factor), "full")
            # 整幅已经解得又稳又准，就不必再花时间定位了
            if _good_enough(0.95, SOFT_N_PATH_GATE):
                return best

        for k, cand in enumerate(plates):
            # 隔离区没比原图小多少 -> 说明整幅就是底版，没必要再解一次
            if cand.size / float(max(gray.size, 1)) >= 0.85:
                continue
            _consider(self._decode_gray(cand, return_debug, upscale_factor),
                      f"plate#{k}", isolated=True)
            if _good_enough(CC_GATE, SOFT_N_PATH_GATE):
                return best

        # 小卡片优先时，若卡片路径不够好，再补整幅兜底
        if plate_first:
            _consider(self._decode_gray(gray, return_debug, upscale_factor), "full")

        if best is not None:
            return best
        return {
            "ok": False, "reason": "decode_failed", "digits": None,
            "bits": None, "ber": 1.0, "conf": 0.0, "mean_conf": 0.0,
            "n_corrected": 0, "path": "none",
        }

def make_payloads(n_payloads: int, seed: int) -> List[str]:
    """压测号码。首尾不用 0，同一 seed 在各消融配置之间可以对照。"""
    rng = np.random.default_rng(seed)
    digits_list: List[str] = []
    while len(digits_list) < n_payloads:
        d = "".join(str(x) for x in rng.integers(0, 10, size=PAYLOAD_DIGITS))
        if d[0] == "0":
            d = "1" + d[1:]
        digits_list.append(d)
    return digits_list


PAIR_DUMP_FIELDS = (
    "seed", "level", "payload", "layout", "depth", "reg_ok", "reg_residual",
    "pair_id", "bit_idx", "group_id", "cell_a", "cell_b", "track", "letter",
    "y_mid", "h", "slope", "width", "clr_min", "dist_to_iv_end", "straddle",
    "soft", "soft_n", "bit_err", "group_gap",
)


def pair_dump_rows(res: Dict, tx_bits: Sequence[int], *, seed: int, level: str,
                   payload: str, design: "ArrowDesign") -> List[Dict]:
    """一个样本的 70 行。配准失败也落盘，信号型统计时再按 reg_ok 过滤。"""
    perm = design.pair_perm
    cells = design.cells
    conf = res.get("conf")
    conf_a = np.zeros(N_BITS, np.float64) if conf is None else np.asarray(conf, np.float64).ravel()
    soft = res.get("soft_n")
    soft_a = np.zeros(N_BITS, np.float64) if soft is None else np.asarray(soft, np.float64).ravel()
    bits = res.get("bits") or []
    gaps = res.get("group_margins") or []
    rows: List[Dict] = []
    for i in range(N_BITS):
        p = int(perm[i])
        a = cells[2 * p]
        b = cells[2 * p + 1]
        if res.get("ok") and i < len(bits) and i < len(tx_bits):
            bit_err = int(int(bits[i]) != int(tx_bits[i]))
        else:
            bit_err = 1
        g = i // 7
        rows.append({
            "seed": int(seed),
            "level": level,
            "payload": payload,
            "layout": int(design.layout_version),
            "depth": float(design.depth),
            "reg_ok": int(bool(res.get("reg_ok"))),
            "reg_residual": float(res.get("reg_residual") or 0.0),
            "pair_id": p,
            "bit_idx": i,
            "group_id": g,
            "cell_a": int(a.idx),
            "cell_b": int(b.idx),
            "track": int(a.track),
            "letter": a.letter,
            "y_mid": round(0.5 * (a.y_mid + b.y_mid), 3),
            "h": int(min(a.height, b.height)),
            "slope": round(max(abs(a.slope), abs(b.slope)), 4),
            "width": round(min(a.width, b.width), 3),
            "clr_min": round(min(a.clearance, b.clearance), 3),
            "dist_to_iv_end": round(min(a.dist_to_iv_end(), b.dist_to_iv_end()), 3),
            "straddle": int(a.track != b.track or a.interval != b.interval),
            "soft": round(float(conf_a[i]) if i < conf_a.size else 0.0, 6),
            "soft_n": round(float(soft_a[i]) if i < soft_a.size else 0.0, 6),
            "bit_err": bit_err,
            "group_gap": round(float(gaps[g]) if g < len(gaps) else 0.0, 4),
        })
    return rows


def summarize_pair_rows(rows: Sequence[Dict]) -> Dict:
    """只在配准合格的行上算 μ/σ/Q。aligned soft_n 朝真值比特取正。"""
    ok = [r for r in rows if int(r.get("reg_ok", 0)) == 1]
    by_pair: Dict[int, List[float]] = {}
    for r in ok:
        # 对齐到真值方向：判对取 |soft_n|，判错取负。真 bit 为 0 时 soft_n 本身是负的。
        measured = abs(float(r["soft_n"]))
        aligned = measured if int(r["bit_err"]) == 0 else -measured
        by_pair.setdefault(int(r["pair_id"]), []).append(aligned)
    stats = []
    for pid, vals in sorted(by_pair.items()):
        arr = np.asarray(vals, np.float64)
        mu = float(arr.mean()) if arr.size else 0.0
        sd = float(arr.std()) if arr.size else 0.0
        q = mu / sd if sd > 1e-9 else (99.0 if mu > 0 else 0.0)
        stats.append({"pair_id": pid, "mu": mu, "sigma": sd, "q": q, "n": int(arr.size)})
    mus = [s["mu"] for s in stats]
    qs = sorted(s["q"] for s in stats)
    worst_i = max(0, int(math.ceil(0.10 * len(qs))) - 1) if qs else 0
    return {
        "n_rows_reg_ok": len(ok),
        "n_pairs": len(stats),
        "mu_median": float(np.median(mus)) if mus else 0.0,
        "mu_min": float(min(mus)) if mus else 0.0,
        "q_worst10": float(qs[worst_i]) if qs else 0.0,
        "pairs": stats,
    }


def _production_gate(res: Dict, decoder: "WatermarkDecoder"):
    """把研究解码器的一次结果送进 ``arrow_decode`` 的接受门。

    ``--no-ecc`` 时研究解码器不算配准分，cc 保持 0，生产门因此跳过默认 cc 门，
    裕度均值、单组最小裕度和补零一致性仍然生效。
    """
    import arrow_decode as AD
    cc_thr = AD.effective_cc_threshold(
        AD.CC_THRESHOLD, bool(getattr(decoder, "use_ecc_refine", True)))
    if not isinstance(res, dict):
        res = {"ok": False, "reason": "decode_failed", "digits": None}
    return AD.apply_gate(res, carrier="edge", cc_threshold=cc_thr)


class BenchmarkRunner:
    """50 组随机号码 x 5 档扰动。统计走生产接受门，并单列接受且错号。"""

    def __init__(self, embedder: WatermarkEmbedder, sim: CeramicCamSimulator,
                 decoder: WatermarkDecoder, out_dir: str = OUT_DIR):
        self.embedder = embedder
        self.sim = sim
        self.decoder = decoder
        self.out_dir = out_dir

    def run(self, n_payloads: int = 50, seed: int = 7,
            levels: Sequence[str] = LEVEL_ORDER,
            payloads: Optional[Sequence[str]] = None,
            dump_pairs: Optional[str] = None) -> Dict:
        digits_list = list(payloads) if payloads is not None else make_payloads(
            n_payloads, seed)
        n_payloads = len(digits_list)
        design = self.embedder.design

        stats = {lv: {"n": 0, "bits": 0, "err": 0, "ok": 0, "conf": [],
                      "fail": 0, "corr": 0, "accept_correct": 0,
                      "accept_wrong": 0, "rejected": 0, "reg_ok": 0,
                      "reg_ok_correct": 0, "residuals": []} for lv in levels}
        t0 = time.time()
        total = n_payloads * len(levels)
        done = 0
        old_verbose = self.embedder.verbose
        self.embedder.verbose = False          # 压测期间关闭逐条日志，避免打断进度条
        dump_fh = None
        writer = None
        if dump_pairs:
            _ensure_dir(os.path.dirname(os.path.abspath(dump_pairs)) or ".")
            dump_fh = open(dump_pairs, "w", encoding="utf-8", newline="")
            writer = csv.DictWriter(dump_fh, fieldnames=PAIR_DUMP_FIELDS)
            writer.writeheader()
        try:
            for pi, digits in enumerate(digits_list):
                wm = self.embedder.embed(digits)
                tx_bits = np.array(wm["bits"], np.int8)
                for lv in levels:
                    srng = np.random.default_rng(seed * 100003 + pi * 97 +
                                                 LEVEL_ORDER.index(lv) * 13 + 5)
                    cam = self.sim.simulate(wm["plate"], lv, srng)["image"]
                    res = self.decoder.decode(cam)
                    st = stats[lv]
                    st["n"] += 1
                    st["bits"] += N_BITS
                    if res.get("reg_ok"):
                        st["reg_ok"] += 1
                        st["residuals"].append(float(res.get("reg_residual") or 0.0))
                        if res.get("digits") == digits:
                            st["reg_ok_correct"] += 1
                    if res["ok"]:
                        rx = np.array(res["bits"], np.int8)
                        st["err"] += int((rx != tx_bits).sum())
                        st["conf"].append(res["mean_conf"])
                        st["corr"] += res["n_corrected"]
                        if res["digits"] == digits:
                            st["ok"] += 1
                        else:
                            st["fail"] += 1
                    else:
                        st["err"] += N_BITS
                        st["fail"] += 1
                        st["conf"].append(0.0)
                    gated = _production_gate(res, self.decoder)
                    if gated.accepted and gated.digits == digits:
                        st["accept_correct"] += 1
                    elif gated.accepted:
                        st["accept_wrong"] += 1
                    else:
                        st["rejected"] += 1
                    if writer is not None:
                        for row in pair_dump_rows(
                                res, tx_bits, seed=seed, level=lv,
                                payload=digits, design=design):
                            writer.writerow(row)
                    done += 1
                    if done % 25 == 0 or done == total:
                        el = time.time() - t0
                        sys.stdout.write(f"\r  [压测] {done}/{total}  已用 {el:5.1f}s  "
                                         f"剩余约 {el / done * (total - done):5.1f}s   ")
                        sys.stdout.flush()
        finally:
            self.embedder.verbose = old_verbose
            if dump_fh is not None:
                dump_fh.close()
        sys.stdout.write("\n")
        elapsed = time.time() - t0

        report = {"n_payloads": n_payloads, "levels": list(levels),
                  "elapsed": elapsed, "seed": seed,
                  "layout": design.layout_version, "depth": design.depth,
                  "window_pow": design.window_pow,
                  "pair_perm_id": design.pair_perm_id,
                  "dump_pairs": dump_pairs, "stats": {}}
        for lv in levels:
            st = stats[lv]
            n = max(1, st["n"])
            resids = st["residuals"]
            report["stats"][lv] = {
                "n": st["n"],
                "ber": st["err"] / max(1, st["bits"]),
                # success：生产门接受且号码正确。研究解码器「位数对上」另记。
                "success": st["accept_correct"] / n,
                "accept_and_wrong": st["accept_wrong"],
                "accept_and_wrong_rate": st["accept_wrong"] / n,
                "rejected": st["rejected"],
                "research_success": st["ok"] / n,
                "reg_ok": st["reg_ok"],
                "reg_ok_rate": st["reg_ok"] / n,
                "reg_ok_correct": st["reg_ok_correct"],
                "reg_ok_correct_rate": (st["reg_ok_correct"] / st["reg_ok"]
                                        if st["reg_ok"] else 0.0),
                "reg_residual_p50": (float(np.percentile(resids, 50))
                                     if resids else None),
                "reg_residual_p95": (float(np.percentile(resids, 95))
                                     if resids else None),
                "mean_conf": float(np.mean(st["conf"])) if st["conf"] else 0.0,
                "mean_corrected": st["corr"] / max(1, st["n"]),
            }
        report["gate"] = "arrow_decode"
        report["accept_and_wrong_total"] = int(sum(
            report["stats"][lv]["accept_and_wrong"] for lv in levels))
        return report

    # ---------------- 终端表格 ----------------
    @staticmethod
    def print_report(report: Dict) -> None:
        cols = [("扰动等级", 14), ("样本", 6), ("BER", 9),
                ("接受且对", 10), ("接受且错", 10), ("拒绝", 8),
                ("研究正确", 10)]
        line = "+" + "+".join("-" * (c[1] + 2) for c in cols) + "+"
        print("\n" + "=" * 96)
        print("  压力测试（生产门）  接受且对 = accepted 且号码正确；"
              "接受且错 = accepted 但号码不是真值")
        print("=" * 96)
        print(line)
        print("|" + "|".join(" " + _pad(c[0], c[1]) + " " for c in cols) + "|")
        print(line)
        wrong = 0
        for lv in report["levels"]:
            s = report["stats"][lv]
            n = max(1, s["n"])
            n_ok = int(round(s["success"] * s["n"]))
            n_wrong = int(s.get("accept_and_wrong", 0))
            wrong += n_wrong
            n_rej = int(s.get("rejected", s["n"] - n_ok - n_wrong))
            n_research = int(round(s.get("research_success", 0.0) * s["n"]))
            row = [
                f"{LEVEL_CN.get(lv, lv)} {lv}",
                f"{s['n']}",
                f"{s['ber'] * 100:.2f}%",
                f"{n_ok}/{s['n']}",
                f"{n_wrong}/{s['n']}",
                f"{n_rej}",
                f"{n_research}/{s['n']}",
            ]
            print("|" + "|".join(" " + _pad(v, cols[i][1], ">" if i > 0 else "<") + " "
                                 for i, v in enumerate(row)) + "|")
        print(line)
        print(f"  接受且错号合计 {wrong}    "
              f"总耗时 {report['elapsed']:.1f}s，"
              f"平均 {report['elapsed'] / max(1, report['n_payloads'] * len(report['levels'])):.2f}s/次")
        print("=" * 96 + "\n")


# ==============================================================================
# 9. 可视化
# ==============================================================================

def _setup_cn_font() -> None:
    plt = _pyplot()
    try:
        from matplotlib import font_manager
        names = {f.name for f in font_manager.fontManager.ttflist}
        for cand in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                     "Source Han Sans SC", "WenQuanYi Zen Hei"]:
            if cand in names:
                plt.rcParams["font.sans-serif"] = [cand]
                plt.rcParams["axes.unicode_minus"] = False
                return
    except Exception:
        pass


def figure_compare(orig: np.ndarray, wm: np.ndarray, cells: List[Cell],
                   digits: str, out_path: str) -> str:
    """1_original_vs_watermarked.png：全图 + 5x 细节放大 + 差值图。"""
    from matplotlib.patches import Rectangle
    plt = _pyplot()
    _setup_cn_font()
    # 选一个"干净竖直笔画"的码元做 5x 放大：宽度接近标称、斜率接近 0
    cand = [c for c in cells if abs(c.slope) < 0.05 and 55.0 < c.width < 72.0]
    if not cand:
        cand = [c for c in cells if abs(c.slope) < 0.05] or list(cells)
    cand.sort(key=lambda c: (c.x_mid, c.y0))
    cell = cand[len(cand) // 2]
    cy = int((cell.y0 + cell.y1) * 0.5)
    cx = int(round(cell.center_at(cy) - cell.width * 0.5))   # 以左边缘为中心取景
    zh, zw = 80, 180
    y0 = max(0, min(CANVAS_H - zh, cy - zh // 2))
    x0 = max(0, min(CANVAS_W - zw, cx - zw // 2))
    crop_o = orig[y0:y0 + zh, x0:x0 + zw]
    crop_w = wm[y0:y0 + zh, x0:x0 + zw]
    z = 5
    crop_o5 = cv2.resize(crop_o, (zw * z, zh * z), interpolation=cv2.INTER_NEAREST)
    crop_w5 = cv2.resize(crop_w, (zw * z, zh * z), interpolation=cv2.INTER_NEAREST)
    diff = np.abs(wm.astype(np.int16) - orig.astype(np.int16)).astype(np.float32)
    diff_vis = np.clip(diff * 10, 0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(18.5, 13.5), facecolor="white")
    gs = fig.add_gridspec(4, 2, height_ratios=[2.35, 2.35, 3.0, 3.0],
                          hspace=0.30, wspace=0.10,
                          left=0.03, right=0.98, top=0.925, bottom=0.03)

    ax = fig.add_subplot(gs[0, :])
    ax.imshow(orig, cmap="gray", vmin=0, vmax=255)
    ax.set_title("(a) 原始 ARROW 底版  2200x600 px = 2.2 cm × 0.6 cm（无任何定位标记）",
                 fontsize=12, pad=6)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, :])
    ax.imshow(wm, cmap="gray", vmin=0, vmax=255)
    _mm = MOD_DEPTH * 0.01                     # 1000 px/cm -> 1px = 0.01 mm
    ax.set_title(f"(b) 嵌入水印后（载荷 {digits}，边沿平滑偏移 ±{MOD_DEPTH:g}px = ±{_mm:.3f} mm，墨量守恒）",
                 fontsize=12, pad=6)
    ax.axis("off")

    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(crop_o5, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"(c) 原始 · 局部 5x 放大 ({zw}x{zh}px)", fontsize=12)
    ax.axis("off")

    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(crop_w5, cmap="gray", vmin=0, vmax=255)
    ax.set_title("(d) 水印版 · 同一区域 5x 放大（正弦微波，无直角台阶）", fontsize=12)
    ax.axis("off")

    ax = fig.add_subplot(gs[3, 0])
    ax.imshow(diff_vis, cmap="inferno", vmin=0, vmax=255)
    ax.set_title("(e) 像素差值 x10（调制能量只落在笔画两侧边缘）", fontsize=12)
    ax.axis("off")

    ax = fig.add_subplot(gs[3, 1])
    ax.imshow(wm, cmap="gray", vmin=0, vmax=255)
    rect = Rectangle((x0, y0), zw, zh, fill=False, edgecolor="#ff3355", lw=2)
    ax.add_patch(rect)
    for c in cells:
        xc = c.center_at((c.y0 + c.y1) * 0.5)
        ax.plot([xc, xc], [c.y0, c.y1], color="#00d0ff", lw=0.5, alpha=0.55)
    ax.set_title(f"(f) 140 个码元（青色）：{digits} / 70 bit 差分码", fontsize=12)
    ax.axis("off")

    fig.suptitle("ARROW 陶瓷 Logo 隐形水印 —— 嵌入前后对比", fontsize=16, y=0.975)
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)
    return out_path


def figure_attacks(samples: List[Dict], out_path: str) -> str:
    plt = _pyplot()
    """2_simulated_attacks.png：五档复合扰动 + 解码结果。"""
    _setup_cn_font()
    n = len(samples)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig = plt.figure(figsize=(20, 4.6 * rows + 0.8), facecolor="white")
    gs = fig.add_gridspec(rows, cols, hspace=0.20, wspace=0.05,
                          left=0.015, right=0.985, top=0.90, bottom=0.02)
    for i, s in enumerate(samples):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.imshow(s["image"], cmap="gray", vmin=0, vmax=255)
        p = s["params"]
        ok = s["decode_ok"]
        color = "#00a050" if ok else "#d02020"
        ax.set_title(
            f"[{LEVEL_CN[s['level']]}] {s['level']}\n"
            f"tilt={p['tilt']:.1f}° roll={p['roll']:.1f}° ink={p['ink']:+.1f}px\n"
            f"blur={p['defocus']:.2f} mot={p['motion']}px σn={p['noise']:.1f} "
            f"JPEG={p['jpeg']}\n→ {s['decoded']}  [{'OK' if ok else 'FAIL'}]",
            fontsize=9.5, color=color)
        ax.axis("off")
    fig.suptitle("陶瓷印制 + 手机微距拍摄 物理信道退化模拟（五档复合扰动）",
                 fontsize=15, y=0.965)
    fig.savefig(out_path, dpi=105, facecolor="white")
    plt.close(fig)
    return out_path


def figure_decode_debug(warped: np.ndarray, cells: List[Cell], bits: List[int],
                        truth_bits: List[int], out_path: str,
                        title: str = "") -> str:
    """3_decode_debug.png：矫正后的图 + 码元测量窗 + 局部 4x 放大 + 逐比特判决。"""
    from matplotlib.patches import Rectangle
    plt = _pyplot()
    _setup_cn_font()
    fig = plt.figure(figsize=(18, 10.2), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.0, 0.62],
                          hspace=0.16, wspace=0.10,
                          left=0.02, right=0.985, top=0.90, bottom=0.06)

    ax = fig.add_subplot(gs[0, :])
    ax.imshow(warped, cmap="gray", vmin=0, vmax=255)
    for i, c in enumerate(cells):
        bi = i // 2                      # 两个相邻码元 -> 一个比特
        ok = (bits[bi] == truth_bits[bi]) if bi < len(bits) else False
        col = "#00c060" if ok else "#ff2040"
        yc = 0.5 * (c.y0 + c.y1)
        xc = c.center_at(yc)
        ax.add_patch(Rectangle((xc - c.half_win, c.y0), 2 * c.half_win, c.y1 - c.y0,
                               fill=False, edgecolor=col, lw=0.9, alpha=0.95))
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(CANVAS_H, 0)
    ax.set_title(title or "矫正后图像 + 140 个码元测量窗（绿=判决正确，红=错误）", fontsize=12)
    ax.axis("off")

    # 局部 4x 放大：选一条竖直宽笔画
    by_track: Dict[int, List[Cell]] = {}
    for c in cells:
        by_track.setdefault(c.track, []).append(c)
    vert = [k for k, g in by_track.items()
            if float(np.median([abs(c.slope) for c in g])) < 0.06 and len(g) >= 8]
    pool = vert if vert else list(by_track.keys())
    bt = max(pool, key=lambda k: float(np.median([c.width for c in by_track[k]])))
    grp = by_track[bt]
    c0 = grp[len(grp) // 2]
    cy = int((c0.y0 + c0.y1) * 0.5)
    cx = int(c0.center_at(cy))
    zh, zw = 76, 120
    y0 = max(0, min(CANVAS_H - zh, cy - zh // 2))
    x0 = max(0, min(CANVAS_W - zw, cx - zw // 2))
    crop = warped[y0:y0 + zh, x0:x0 + zw]
    z = 4
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(cv2.resize(crop, (zw * z, zh * z), interpolation=cv2.INTER_NEAREST),
              cmap="gray", vmin=0, vmax=255)
    for c in cells:
        if c.y1 <= y0 or c.y0 >= y0 + zh:
            continue
        yc2 = 0.5 * (c.y0 + c.y1)
        xc2 = c.center_at(yc2)
        if xc2 < x0 - 60 or xc2 > x0 + zw + 60:
            continue
        ax.add_patch(Rectangle(((xc2 - c.half_win - x0) * z, (c.y0 - y0) * z),
                               2 * c.half_win * z, (c.y1 - c.y0) * z,
                               fill=False, edgecolor="#00e0ff", lw=1.2))
    ax.set_xlim(0, zw * z)
    ax.set_ylim(zh * z, 0)
    ax.set_title("码元测量窗局部 4x（青色框 = 解码器采样范围）", fontsize=11)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    err = [i for i in range(min(len(bits), len(truth_bits))) if bits[i] != truth_bits[i]]
    ax.bar(np.arange(len(truth_bits)), np.ones(len(truth_bits)),
           color="#c8d8e8", width=0.9)
    for i in err:
        ax.bar([i], [1], color="#ff3050", width=0.9)
    ax.set_xlim(-0.6, len(truth_bits) - 0.4)
    ax.set_yticks([])
    ax.set_xlabel(f"比特索引（共 {len(truth_bits)} bit，错误 {len(err)} bit）", fontsize=11)
    ax.set_title("逐比特判决结果", fontsize=11)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    txt = (
        "解码流程\n"
        "─────────────────────────\n"
        "1  形态学背景差分二值化\n"
        "2  五字母质心 / 凸包四角 自同步\n"
        "3  候选单应 NCC 择优\n"
        "4  ECC 亚像素精配（含粗搜抢救）\n"
        "5  法向亚像素投影\n"
        "6  匹配滤波 + 逐行半高宽（亚像素）\n"
        "7  相邻码元差分判决\n"
        "8  Hamming(7,4) 纠错 → 11 位数字"
    )
    ax.text(0.0, 0.98, txt, va="top", ha="left", fontsize=11.5,
            family=plt.rcParams["font.sans-serif"][0], linespacing=1.7)
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)
    return out_path


def figure_benchmark(report: Dict, out_path: str) -> str:
    plt = _pyplot()
    _setup_cn_font()
    lv = report["levels"]
    ber = [report["stats"][l]["ber"] * 100 for l in lv]
    succ = [report["stats"][l]["success"] * 100 for l in lv]
    x = np.arange(len(lv))
    fig, ax1 = plt.subplots(figsize=(11, 5.6), facecolor="white")
    ax1.bar(x - 0.2, succ, width=0.4, color="#2f6fb5", label="生产门接受且正确 (%)")
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("成功率 (%)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{LEVEL_CN[l]}\n{l}" for l in lv], fontsize=11)
    ax2 = ax1.twinx()
    ax2.plot(x, ber, "o-", color="#d03030", lw=2, ms=8, label="BER (%)")
    ax2.set_ylabel("BER (%)", fontsize=12)
    ax2.set_ylim(0, max(12, max(ber) * 1.35 if max(ber) > 0 else 1))
    for xi, v in zip(x, succ):
        ax1.text(xi - 0.2, v + 2, f"{v:.1f}%", ha="center", fontsize=10, color="#2f6fb5")
    for xi, v in zip(x, ber):
        ax2.text(xi, v + 0.45, f"{v:.2f}%", ha="center", fontsize=10, color="#d03030",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    wrong = sum(int(report["stats"][l].get("accept_and_wrong", 0)) for l in lv)
    ax1.set_title(
        f"压力测试（生产门接受且正确）：{report['n_payloads']} 组 x {len(lv)} 档"
        f"    接受且错号 {wrong}",
        fontsize=14, pad=34)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, fontsize=11, frameon=False)
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor="white")
    plt.close(fig)
    return out_path


# ==============================================================================
# 10. 主流程
# ==============================================================================

def run_demo(embedder: WatermarkEmbedder, sim: CeramicCamSimulator,
             decoder: WatermarkDecoder, digits: str, out_dir: str) -> Dict:
    print("\n" + "-" * 96)
    print(f"  【演示】载荷 = {digits}")
    print("-" * 96)
    t0 = time.time()
    wm = embedder.embed(digits)
    t_emb = time.time() - t0
    orig = embedder.design.template

    p1 = os.path.join(out_dir, "1_original_vs_watermarked.png")
    figure_compare(orig, wm["plate"], embedder.design.cells, digits, p1)
    print(f"  [输出] {p1}")

    # 无扰动自检
    res0 = decoder.decode(wm["plate"], return_debug=True)
    ber0 = float(np.mean(np.array(res0["bits"]) != np.array(wm["bits"])))
    print(f"  [自检] 无扰动直读 -> {res0['digits']}  BER={ber0 * 100:.2f}%  "
          f"置信度={res0['mean_conf']:.3f}  {'通过' if res0['digits'] == digits else '失败'}")
    p3 = os.path.join(out_dir, "3_decode_debug.png")
    figure_decode_debug(res0["warped"], embedder.design.cells, res0["bits"],
                        wm["bits"], p3,
                        title=f"无扰动自检：矫正图 + 码元窗（载荷 {digits}，BER={ber0 * 100:.2f}%）")
    print(f"  [输出] {p3}")

    samples = []
    for i, lv in enumerate(LEVEL_ORDER):
        srng = np.random.default_rng(20240501 + i * 17)
        cam = sim.simulate(wm["plate"], lv, srng)
        t1 = time.time()
        r = decoder.decode(cam["image"])
        dt = time.time() - t1
        ber = (float(np.mean(np.array(r["bits"]) != np.array(wm["bits"])))
               if r["ok"] else 1.0)
        samples.append(dict(image=cam["image"], level=lv, params=cam["params"],
                            decoded=r["digits"] or "—", decode_ok=(r["digits"] == digits)))
        print(f"  [{LEVEL_CN[lv]:<3}] {lv:<9} -> {str(r['digits']):<11} "
              f"BER={ber * 100:6.2f}%  置信={r.get('mean_conf', 0):.3f}  {dt * 1000:6.0f}ms  "
              f"{'OK' if r['digits'] == digits else 'FAIL'}")

    p2 = os.path.join(out_dir, "2_simulated_attacks.png")
    figure_attacks(samples, p2)
    print(f"  [输出] {p2}")
    print(f"  嵌入耗时 {t_emb * 1000:.0f} ms")
    return {"digits": digits, "samples": samples, "self_ber": ber0}


def _print_decode_line(name: str, res, dt: float) -> None:
    """Print one production decode. Rejected rows never lead with a phone number."""
    import arrow_decode as AD
    scale = f"{res.logo_scale:.2f}" if res.logo_scale is not None else "—"
    tail = (f"裕度均值 {res.margin_mean:.2f}  最小 {res.margin_min:.2f}  "
            f"cc {res.cc:.3f}  scale {scale}  置信度 {res.mean_conf:.4f}   "
            f"{dt * 1000:.0f} ms")
    if res.accepted:
        print(f"  {name:<32} -> {res.digits}   {tail}  接受")
        return
    if res.candidate_digits:
        extra = f"候选 {res.candidate_digits}（不可信）"
    elif res.ok:
        extra = "候选 码字越界（不可信）"
    else:
        extra = "无候选"
    print(f"  {name:<32} -> 拒绝（{AD.reason_text(res.reason)}）  {extra}   {tail}")


# 同一批号码、同一个种子。pow2 只改窗，不降 depth。
ABLATION_CONFIGS = (
    ("v1@1.8", 1, V1_DEPTH, 1.0),
    ("v1@6", 1, MOD_DEPTH, 1.0),
    ("v2@6", 2, MOD_DEPTH, 1.0),
    ("v2@6+pow2", 2, MOD_DEPTH, 2.0),
)


def run_ablation(n_payloads: int, seed: int, levels: Sequence[str], out_dir: str,
                 use_ecc_refine: bool = True) -> List[Dict]:
    """v1@1.8 / v1@6 / v2@6 / v2@6+pow2。号码列表共用。"""
    payloads = make_payloads(n_payloads, seed)
    sim = CeramicCamSimulator(seed=seed)
    reports: List[Dict] = []
    for name, layout, depth, pow_ in ABLATION_CONFIGS:
        print(f"\n==== 消融 {name}  layout={layout} depth={depth:g} window={pow_:g} ====")
        design = ArrowDesign(verbose=False, depth=depth, layout_version=layout,
                             window_pow=pow_)
        embedder = WatermarkEmbedder(design, verbose=False)
        decoder = WatermarkDecoder(design, use_ecc_refine=use_ecc_refine, verbose=False)
        dump = os.path.join(out_dir, f"pairs_{name.replace('@', '_').replace('+', '_')}.csv")
        rep = BenchmarkRunner(embedder, sim, decoder, out_dir).run(
            n_payloads, seed=seed, levels=levels, payloads=payloads, dump_pairs=dump)
        rep["name"] = name
        rows = []
        with open(dump, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        rep["pairs"] = summarize_pair_rows(rows)
        reports.append(rep)
        BenchmarkRunner.print_report(rep)
        print(f"  配对准样本 μ 中位 {rep['pairs']['mu_median']:.3f}  "
              f"最差 {rep['pairs']['mu_min']:.3f}  "
              f"最差 10% Q {rep['pairs']['q_worst10']:.3f}")
    return reports


def _parse_levels(text: Optional[str]) -> List[str]:
    if not text:
        return list(LEVEL_ORDER)
    out = []
    for part in text.split(","):
        name = part.strip()
        if not name:
            continue
        if name not in LEVELS:
            raise SystemExit(f"未知扰动档 {name}，可选 {', '.join(LEVEL_ORDER)}")
        out.append(name)
    return out or list(LEVEL_ORDER)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="ARROW 陶瓷 Logo 隐形水印：嵌入 / 物理信道模拟 / 盲解码")
    ap.add_argument("--demo", action="store_true", help="只跑单条演示")
    ap.add_argument("--bench", action="store_true", help="只跑压力测试")
    ap.add_argument("--ablation", action="store_true",
                    help="跑 v1@1.8 / v1@6 / v2@6 / v2@6+pow2")
    ap.add_argument("--all", action="store_true", help="演示 + 压力测试（默认）")
    ap.add_argument("-n", "--num", type=int, default=50, help="压测号码组数（默认 50）")
    ap.add_argument("--digits", type=str, default="13800138000", help="演示用 11 位号码")
    ap.add_argument("--seed", type=int, default=7, help="压测随机种子")
    ap.add_argument("--out", type=str, default=OUT_DIR, help="输出目录")
    ap.add_argument("--depth", type=float, default=None,
                    help=f"边缘调制深度 px，默认 {MOD_DEPTH:g}")
    ap.add_argument("--layout", default=None,
                    help="嵌入 1 或 2（默认 2）；解码还可以是 auto，先试 v2")
    ap.add_argument("--window-pow", type=float, default=None,
                    help="调制窗指数，1=sin，2=sin^2。默认 1")
    ap.add_argument("--levels", default=None,
                    help="逗号分隔的扰动档，默认 clean,light,moderate,heavy,extreme")
    ap.add_argument("--dump-pairs", default=None,
                    help="把每个差分对的几何和 soft_n 写成 CSV")
    ap.add_argument("--no-ecc", action="store_true", help="关闭 ECC 亚像素精配")
    ap.add_argument("--decode", metavar="IMAGE", nargs="+", default=None,
                    help="解码实拍照片（可一次给多张）")
    args = ap.parse_args(argv)

    explicit = args.demo or args.bench or args.ablation or args.decode
    do_demo = args.demo or args.all or not explicit
    do_bench = args.bench or args.all or not explicit
    if args.ablation or args.decode:
        if not args.demo:
            do_demo = False
        if not args.bench and not args.all:
            do_bench = False

    depth = MOD_DEPTH if args.depth is None else float(args.depth)
    window_pow = WINDOW_POW if args.window_pow is None else float(args.window_pow)
    layout_text = "2" if args.layout is None else str(args.layout).strip().lower()

    if args.decode:
        # 生产门限走 arrow_decode。--no-ecc 关闭 ECC，研究解码器不计算 cc
        # （保持 0）。Decoder 因此跳过默认 cc≥0.85 门，只留裕度门；
        # 默认路径生产门全开。压测在 BenchmarkRunner 里对同一次研究解码
        # 再过 arrow_decode.apply_gate，并统计接受且错号。
        import arrow_decode as AD
        if layout_text not in ("1", "2", "auto"):
            raise SystemExit("--layout 解码时只能是 1、2 或 auto")
        sessions = {}

        def _session(layout_version: int, used_depth: float):
            key = (layout_version, used_depth)
            if key not in sessions:
                sessions[key] = AD.Decoder(
                    carrier="edge",
                    layout_version=layout_version,
                    depth=used_depth,
                    window_pow=window_pow,
                    use_ecc_refine=not args.no_ecc)
            return sessions[key]

        print("=" * 74)
        if args.no_ecc:
            cc_note = "cc 门关闭（--no-ecc 不计算配准分）"
        else:
            cc_note = f"cc ≥ {AD.CC_THRESHOLD:g}"
        print(f"  实拍照片解码（边缘载体）  layout={layout_text}  depth={depth:g}"
              f"  裕度均值 ≥ {AD.MARGIN_THRESHOLD:g}"
              f"  单组最小 ≥ {AD.MIN_GROUP_MARGIN:g}  {cc_note}")
        print("=" * 74)
        for path in args.decode:
            img = imread_u(path)
            if img is None:
                print(f"  {path}  ->  读不到文件")
                continue
            t0 = time.time()
            if layout_text == "auto":
                res = _session(2, depth).decode(img)
                if not res.accepted:
                    older = _session(1, V1_DEPTH).decode(img)
                    if older.accepted or older.margin > res.margin:
                        res = older
            else:
                used = V1_DEPTH if layout_text == "1" and args.depth is None else depth
                res = _session(int(layout_text), used).decode(img)
            dt = time.time() - t0
            _print_decode_line(os.path.basename(path), res, dt)
        print("=" * 74)
        return 0

    if layout_text not in ("1", "2"):
        raise SystemExit("嵌入 / 压测的 --layout 只能是 1 或 2")
    layout_version = int(layout_text)
    if layout_version == 1 and args.depth is None and not args.ablation:
        # 不显式给深度时，v1 印版用冻结的 1.8。v1@6 要写 --layout 1 --depth 6。
        depth = V1_DEPTH

    out_dir = _ensure_dir(args.out)
    print("=" * 96)
    print("  ARROW 陶瓷 Logo 隐形水印系统 —— 差分宽度调制 / Print-Cam 鲁棒盲解码")
    print("=" * 96)
    print(f"  OpenCV {cv2.__version__}   NumPy {np.__version__}")
    print(f"  底版 {CANVAS_W}x{CANVAS_H}px（{CANVAS_W / 1000:.1f}cm x {CANVAS_H / 1000:.1f}cm @1000px/cm）")
    print(f"  载荷 {PAYLOAD_BITS} bit -> Hamming(7,4) -> {N_BITS} bit -> {N_CELLS} 个码元")

    if args.ablation:
        levels = _parse_levels(args.levels)
        reports = run_ablation(args.num, args.seed, levels, out_dir,
                               use_ecc_refine=not args.no_ecc)
        wrong = sum(int(r.get("accept_and_wrong_total", 0)) for r in reports)
        slim = []
        for rep in reports:
            item = {k: rep[k] for k in (
                "name", "n_payloads", "levels", "elapsed", "seed", "layout",
                "depth", "window_pow", "pair_perm_id", "accept_and_wrong_total",
                "stats")}
            item["mu_median"] = rep["pairs"]["mu_median"]
            item["mu_min"] = rep["pairs"]["mu_min"]
            item["q_worst10"] = rep["pairs"]["q_worst10"]
            slim.append(item)
        with open(os.path.join(out_dir, "ablation_report.json"), "w", encoding="utf-8") as f:
            json.dump({"accept_and_wrong_total": wrong, "runs": slim},
                      f, ensure_ascii=False, indent=2)
        print(f"\n消融接受且错号合计 {wrong}。输出目录：{os.path.abspath(out_dir)}\n")
        return 0

    t0 = time.time()
    design = ArrowDesign(verbose=True, depth=depth, layout_version=layout_version,
                         window_pow=window_pow)
    embedder = WatermarkEmbedder(design, verbose=True)
    sim = CeramicCamSimulator(seed=20240501)
    decoder = WatermarkDecoder(design, use_ecc_refine=not args.no_ecc, verbose=True)
    print(f"  初始化耗时 {time.time() - t0:.2f}s")
    print(f"  layout=v{design.layout_version}  depth={design.depth:g}  "
          f"window={design.window_pow:g}  perm={design.pair_perm_id}")

    if do_demo:
        run_demo(embedder, sim, decoder, args.digits, out_dir)

    if do_bench:
        levels = _parse_levels(args.levels)
        runner = BenchmarkRunner(embedder, sim, decoder, out_dir)
        rep = runner.run(args.num, seed=args.seed, levels=levels,
                         dump_pairs=args.dump_pairs)
        runner.print_report(rep)
        p4 = os.path.join(out_dir, "4_benchmark_report.png")
        figure_benchmark(rep, p4)
        print(f"  [输出] {p4}")
        with open(os.path.join(out_dir, "benchmark_report.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        if args.dump_pairs:
            print(f"  [输出] {args.dump_pairs}")

    print(f"\n全部完成，输出目录：{os.path.abspath(out_dir)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
