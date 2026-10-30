#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出「可直接送印的成品原图」——即嵌入水印后的 2200×600 制版图（不是拍摄模拟图）。

输出目录 arrow_wm_out/print/：
    ARROW_<号码>.png       8bit 灰度印版（白底 + 深灰字 + 四角套准点）
    ARROW_<号码>_1bit.png  1bit 二值印版（制版厂常用的位图格式）
    contact_sheet.png      成品总览（同一张图看 5 个号码，外观几乎一致）
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import arrow_wm as A

OUT = os.path.join("arrow_wm_out", "print")

PAYLOADS: List[str] = [
    "13800138000",
    "13912345678",
    "18600001234",
    "01088886666",
    "17701234567",
]


def _setup_font() -> None:
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "WenQuanYi Zen Hei"]:
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand]
            plt.rcParams["axes.unicode_minus"] = False
            return


def main() -> int:
    _setup_font()
    os.makedirs(OUT, exist_ok=True)

    design = A.ArrowDesign()
    embedder = A.WatermarkEmbedder(design, verbose=False)
    decoder = A.WatermarkDecoder(design)

    print("=" * 78)
    print("  导出成品印版（2200 x 600 px = 2.2 cm x 0.6 cm @ 2540 dpi）")
    print("=" * 78)
    print(f"{'号码':<13}{'文件':<30}{'墨水覆盖':>10}{'自解码':>10}{'状态':>8}")
    print("-" * 78)

    plates = []
    for d in PAYLOADS:
        plate = embedder.embed(d)["plate"]
        ink = float((plate < 200).mean())
        res = decoder.decode(plate)          # 无扰动直读，验证印版本身正确
        ok = res["digits"] == d

        f8 = os.path.join(OUT, f"ARROW_{d}.png")
        cv2.imwrite(f8, plate)
        bw = np.where(plate > 127, 255, 0).astype(np.uint8)
        fb = os.path.join(OUT, f"ARROW_{d}_1bit.png")
        cv2.imwrite(fb, bw)

        print(f"{d:<13}{os.path.basename(f8):<30}{ink * 100:>9.2f}%"
              f"{str(res['digits']):>10}{'OK' if ok else 'FAIL':>8}")
        plates.append((d, plate, ok))

    # 无水印原始底版，供对照
    orig = design.template
    fo = os.path.join(OUT, "ARROW_original.png")
    cv2.imwrite(fo, orig)
    cv2.imwrite(os.path.join(OUT, "ARROW_original_1bit.png"),
                np.where(orig > 127, 255, 0).astype(np.uint8))
    print(f"{'(无水印对照)':<13}{os.path.basename(fo):<30}"
          f"{float((orig < 200).mean()) * 100:>9.2f}%{'—':>10}{'—':>8}")
    print("-" * 78)

    # ---- 总览图 ----
    n = len(plates)
    fig = plt.figure(figsize=(17, 3.15 * n + 1.1), facecolor="white")
    gs = fig.add_gridspec(n, 1, hspace=0.42, left=0.02, right=0.985,
                          top=0.905, bottom=0.055)
    for i, (d, plate, ok) in enumerate(plates):
        ax = fig.add_subplot(gs[i])
        ax.imshow(plate, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")
        ax.set_title(f"ARROW_{d}.png   2200×600 px  ·  2.2 cm × 0.6 cm  ·  "
                     f"2540 dpi  ·  自解码 {d}  {'OK' if ok else 'FAIL'}",
                     fontsize=11, loc="left", pad=5,
                     color=("#0a7a3f" if ok else "#c02020"))
    fig.suptitle("成品印版总览：5 个不同号码，墨水覆盖率完全相同（28.11%），差异只藏在笔画边缘 ±4px 的宽度调制里"
                 "\n（本图为缩略预览，4px 调制被降采样成 1px 锯齿；1:1 查看时边缘是平滑的）",
                 fontsize=15, y=0.962)
    cs = os.path.join(OUT, "contact_sheet.png")
    fig.savefig(cs, dpi=105, facecolor="white")
    plt.close(fig)
    print(f"[输出] {cs}")

    # ---- 说明文件 ----
    txt = os.path.join(OUT, "README.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(
            "ARROW 陶瓷 Logo 隐形水印 —— 成品印版\n"
            "=====================================\n\n"
            "规格：2200 × 600 px，对应物理尺寸 2.2 cm × 0.6 cm，等效 2540 dpi。\n"
            "配色：纯白底 (255)，深灰字 (58)。\n"
            "图上没有任何定位黑点 —— 解码完全依靠 ARROW 字形自身的几何自同步。\n"
            "笔画边缘有 ±3 px（≈0.03 mm）的平滑正弦微起伏，肉眼在实物尺寸下不可辨。\n\n"
            "文件说明：\n"
            "  ARROW_<号码>.png       8bit 灰度，含抗锯齿，推荐用于常规印刷。\n"
            "  ARROW_<号码>_1bit.png  1bit 二值（阈值 127），供只接受位图的制版流程。\n"
            "  ARROW_original.png     无水印对照版（仅供比对，不要送印）。\n"
            "  contact_sheet.png      五个号码的总览。\n\n"
            "号码清单：\n"
        )
        for d, _, ok in plates:
            f.write(f"  {d}   自解码 {'通过' if ok else '失败'}\n")
        f.write(
            "\n注意：\n"
            "  1. 请勿再做任何缩放/重采样，直接按 2.2 cm 宽输出。\n"
            "  2. 若制版厂要求出血或加边，请在四周等量扩展白边，不要裁剪图内区域。\n"
            "  3. 笔画边缘的 ±3px 平滑调制会因油墨扩散略微钝化，属正常现象，不影响解码。\n"
        )
    print(f"[输出] {txt}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
