#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比实验：边缘宽度调制 vs 笔画内部灰度调制

目标：回答一个问题——把调制从"边缘"移到"笔画内部"，
      在真实拍摄链路（透视+模糊+JPEG+噪声）下还能不能解出来？

两种方案共用同一套：码元布局、Hamming(7,4)、扰动模拟器、解码框架。
唯一差别是"信息写在哪里"。
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import cv2

import arrow_wm as A


# ==============================================================================
# 内部灰度调制（外轮廓 0 改动）
# ==============================================================================
def embed_interior(design: A.ArrowDesign, digits: str, delta: float,
                   sigma: float = 3.2) -> Tuple[np.ndarray, List[int]]:
    """在每个码元中心叠一个平滑高斯鼓包，A 加 B 减。

    外轮廓完全不参与，笔画边缘保持绝对笔直。
    """
    bits = A.encode_payload(digits)
    plate = design.template.astype(np.float32).copy()

    # 每个码元的"内核"：取笔画中心 ±(半宽-保护带)
    guard = 14
    r = 10
    for i, cell in enumerate(design.cells):
        b = bits[i // 2]
        sgn = 1.0 if (b == 1) == (i % 2 == 0) else -1.0
        yc = int(round((cell.y0 + cell.y1) * 0.5))
        xc = int(round(cell.center_at(yc)))
        half = int(cell.width * 0.5) - guard
        if half < 4:
            continue
        x0, x1 = max(0, xc - half), min(plate.shape[1], xc + half + 1)
        y0, y1 = max(0, yc - r), min(plate.shape[0], yc + r + 1)
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-(((xx - xc) ** 2 + (yy - yc) ** 2) / (2 * sigma ** 2)))
        # 只在墨迹内部生效
        sub = plate[y0:y1, x0:x1]
        plate[y0:y1, x0:x1] = sub + sgn * delta * g
    return np.clip(plate, 0, 255).astype(np.uint8), bits


def decode_interior(design: A.ArrowDesign, warped: np.ndarray,
                    sigma: float = 3.2) -> Tuple[List[int], np.ndarray]:
    """在矫正图上做**匹配滤波**：用与嵌入端同形状的高斯核加权相关。"""
    tmpl = design.template.astype(np.float32)
    diff = warped.astype(np.float32) - tmpl
    guard = 14
    feat = np.zeros(len(design.cells), np.float32)
    for i, cell in enumerate(design.cells):
        yc = int(round((cell.y0 + cell.y1) * 0.5))
        xc = int(round(cell.center_at(yc)))
        half = int(cell.width * 0.5) - guard
        if half < 4:
            continue
        r = 9
        y0, y1 = max(0, yc - r), min(diff.shape[0], yc + r + 1)
        x0, x1 = max(0, xc - r), min(diff.shape[1], xc + r + 1)
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        patch = diff[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-(((xx - xc) ** 2 + (yy - yc) ** 2) / (2 * sigma ** 2)))
        g = g / g.sum()
        p = patch - patch.mean()          # 去局部均值，抗油墨整体深浅
        feat[i] = float((p * g).sum())
    bits, conf = [], []
    wm_ = max(float(np.median(np.abs(feat))) * 3.0, 1e-6)
    for i in range(A.N_BITS):
        d = float(feat[2 * i] - feat[2 * i + 1])
        bits.append(1 if d > 0 else 0)
        conf.append(d / wm_)
    return bits, np.array(conf, np.float32)


def run(scheme: str, levels: List[str], n: int = 10, delta: float = 8.0):
    design = A.ArrowDesign()
    sim = A.CeramicCamSimulator()
    dec = A.WatermarkDecoder(design)
    emb = A.WatermarkEmbedder(design, verbose=False)
    print(f"\n===== 方案：{scheme} =====")
    for lv in levels:
        bers, oks, t0 = [], 0, time.time()
        for sd in range(n):
            digits = "1%010d" % (sd * 7919 % 10 ** 10)
            if scheme == "edge":
                wm = emb.embed(digits)
                plate, tb = wm["plate"], wm["bits"]
            else:
                plate, tb = embed_interior(design, digits, delta)
            cam = sim.simulate(plate, lv, np.random.default_rng(6000 + sd))
            r = dec.rectify(cam["image"])
            if not r.get("ok"):
                bers.append(1.0)
                continue
            if scheme == "edge":
                f = dec.extract_widths(r["warped"])
                bits, _ = dec.decide_bits(f)
            else:
                bits, _ = decode_interior(design, r["warped"])
            tb = np.array(tb)
            bers.append(float(np.mean(np.array(bits) != tb)))
            dg, _, _ = A.decode_payload(bits)
            oks += (dg == digits)
        print(f"  {lv:<9} 成功率 {oks:2d}/{n}   BER {np.mean(bers) * 100:5.2f}%"
              f"   {(time.time() - t0) / n:.2f}s/次")


if __name__ == "__main__":
    lv = ["light", "moderate", "heavy", "extreme"]
    run("edge", lv, n=10)
    for d in (6.5, 10.0, 14.0):
        run("interior", lv, n=10, delta=d)
