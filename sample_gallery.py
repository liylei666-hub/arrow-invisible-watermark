#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成「样本图 ↔ 编码」对照可视化（依赖 arrow_wm.py）

输出：
    arrow_wm_out/5_sample_gallery.png   6 组样本：拍摄图 + 码字条 + 号码
    arrow_wm_out/6_code_breakdown.png   单个载荷的完整编码展开
    arrow_wm_out/samples_codes.md       样本与编码的文本对照表
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import arrow_wm as A

OUT = "arrow_wm_out"

# (11 位号码, 扰动等级)
SAMPLES: List[Tuple[str, str]] = [
    ("13800138000", "light"),
    ("13912345678", "moderate"),
    ("18600001234", "heavy"),
    ("01088886666", "light"),
    ("17701234567", "moderate"),
    ("19999999999", "heavy"),
]

C_ONE = "#1f4e79"     # bit = 1
C_ZERO = "#cfe0f0"    # bit = 0
C_PAR = "#e8a33d"     # 校验位
C_DAT = "#2f6fb5"     # 数据位


def _setup_font() -> None:
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "WenQuanYi Zen Hei"]:
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand]
            plt.rcParams["axes.unicode_minus"] = False
            return


def bits_str(bits: Sequence[int], group: int = 4, sep: str = " ") -> str:
    s = "".join(str(int(b)) for b in bits)
    return sep.join(s[i:i + group] for i in range(0, len(s), group))


def crop_logo(img: np.ndarray, dec: A.WatermarkDecoder) -> np.ndarray:
    """用解码器同一套二值化把 logo 区域裁出来，便于观察。"""
    ink, sc = dec._binarize(img, 38.0)
    ys, xs = np.nonzero(ink)
    if len(xs) == 0:
        return img
    x0, x1 = xs.min() / sc, xs.max() / sc
    y0, y1 = ys.min() / sc, ys.max() / sc
    px, py = (x1 - x0) * 0.10, (y1 - y0) * 0.45
    x0 = int(max(0, x0 - px))
    x1 = int(min(img.shape[1], x1 + px))
    y0 = int(max(0, y0 - py))
    y1 = int(min(img.shape[0], y1 + py))
    return img[y0:y1, x0:x1]


def draw_bit_strip(ax, bits: Sequence[int], block: int = 7,
                   title: str = "", fontsize: int = 8) -> None:
    """把比特流画成一条色带（深=1，浅=0），每 block 位加分隔线。"""
    arr = np.array(bits, np.int8).reshape(1, -1)
    ax.imshow(arr, aspect="auto", cmap=matplotlib.colors.ListedColormap([C_ZERO, C_ONE]),
              vmin=0, vmax=1, interpolation="nearest")
    for k in range(block, len(bits), block):
        ax.axvline(k - 0.5, color="white", lw=1.4)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=fontsize)


# ==============================================================================
# 图 5：样本画廊
# ==============================================================================
def make_gallery(embedder, sim, decoder, out_path: str) -> List[dict]:
    n = len(SAMPLES)
    rows = (n + 1) // 2
    fig = plt.figure(figsize=(19, 4.9 * rows + 1.5), facecolor="white")
    gs = fig.add_gridspec(rows * 2, 2,
                          height_ratios=sum([[4.6, 0.72] for _ in range(rows)], []),
                          hspace=0.34, wspace=0.05,
                          left=0.02, right=0.985, top=0.905, bottom=0.075)

    records = []
    for i, (digits, level) in enumerate(SAMPLES):
        wm = embedder.embed(digits)
        cam = sim.simulate(wm["plate"], level, np.random.default_rng(9000 + i * 31))
        res = decoder.decode(cam["image"])
        ok = res["digits"] == digits
        crop = crop_logo(cam["image"], decoder)

        r0 = (i // 2) * 2
        ax = fig.add_subplot(gs[r0, i % 2])
        ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        p = cam["params"]
        ax.set_title(
            f"{digits}   [{A.LEVEL_CN[level]}]   tilt={p['tilt']:.0f}°  "
            f"blur={p['defocus']:.1f}  spec={p['spec']:.2f}  JPEG={p['jpeg']}   →   "
            f"解码 {res['digits']}   BER {np.mean(np.array(res['bits']) != np.array(wm['bits'])) * 100:.1f}%   "
            f"{'OK' if ok else 'FAIL'}",
            fontsize=12.5, pad=10, color=("#0a7a3f" if ok else "#c02020"))

        axb = fig.add_subplot(gs[r0 + 1, i % 2])
        draw_bit_strip(axb, wm["bits"], block=7)
        axb.set_xticks([0, 10, 20, 30, 40, 50, 60, 69])
        axb.set_xticklabels(["0", "10", "20", "30", "40", "50", "60", "70"], fontsize=8)
        axb.tick_params(axis="x", length=2, pad=1)

        records.append(dict(digits=digits, level=level, ok=ok,
                            decoded=res["digits"], bits=wm["bits"],
                            ber=float(np.mean(np.array(res["bits"]) != np.array(wm["bits"]))),
                            conf=res.get("mean_conf", 0.0)))

    fig.suptitle("样本图 与 编码 对照：11 位号码 → 70 bit 码字 → 嵌入 → 拍摄 → 盲解码",
                 fontsize=16, y=0.972)
    fig.text(0.5, 0.028,
             "每张图下方的色带 = 该号码的 70 bit 码字（深色 = 1，浅色 = 0）；"
             "白色竖线为 Hamming(7,4) 分组边界（每 7 bit 一组，共 10 组）",
             ha="center", fontsize=11, color="#444444")
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)
    return records


# ==============================================================================
# 图 6：编码展开
# ==============================================================================
def make_breakdown(digits: str, out_path: str) -> dict:
    bits37 = A.payload_to_bits(digits)
    bits40 = [0, 0, 0] + bits37
    code = A.hamming74_encode(bits40)
    blocks = [code[i:i + 7] for i in range(0, 70, 7)]

    fig = plt.figure(figsize=(17, 12.2), facecolor="white")
    gs = fig.add_gridspec(4, 1, height_ratios=[1.05, 0.55, 2.5, 1.7],
                          hspace=0.46, left=0.085, right=0.975, top=0.905, bottom=0.055)

    # ---- ① 文字区 ----
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    lines = [
        f"载荷（11 位十进制）      {digits}",
        f"整数值                  {int(digits):,}",
        f"37 bit 载荷             {bits_str(bits37, 4)}",
        f"40 bit（前补 3 个 0）    {bits_str(bits40, 4)}",
        f"70 bit 码字             {bits_str(code, 7)}",
    ]
    ax.text(0.0, 0.96, "\n".join(lines), va="top", ha="left", fontsize=11.5,
            linespacing=2.1)
    ax.set_title("① 载荷 → 二进制 → Hamming(7,4) 编码", fontsize=13, loc="left", pad=6)

    # ---- ② 37 bit 载荷条 ----
    ax = fig.add_subplot(gs[1])
    draw_bit_strip(ax, bits37, block=4)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 32, 36])
    ax.set_xticklabels(["0", "4", "8", "12", "16", "20", "24", "28", "32", "37"],
                       fontsize=8)
    ax.tick_params(axis="x", length=2, pad=1)
    ax.set_title("② 37 bit 原始载荷（每 4 bit 为一组，共 10 组，最后一组 1 位）",
                 fontsize=12.5, loc="left", pad=6)

    # ---- ③ Hamming 分组网格 ----
    ax = fig.add_subplot(gs[2])
    for r in range(10):
        for c in range(7):
            is_par = c in (0, 1, 3)
            v = blocks[r][c]
            face = (C_PAR if is_par else C_DAT) if v else "#f2f4f7"
            ax.add_patch(Rectangle((c + 0.04, r + 0.06), 0.92, 0.88,
                                   facecolor=face, edgecolor="#c8ccd2", lw=0.8))
            ax.text(c + 0.5, r + 0.5, str(v), ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color="white" if v else "#6b7480")
    ax.set_xlim(0, 7)
    ax.set_ylim(10, 0)
    ax.set_xticks([c + 0.5 for c in range(7)])
    ax.set_xticklabels(["p1", "p2", "d1", "p4", "d2", "d3", "d4"], fontsize=13)
    ax.set_yticks([r + 0.5 for r in range(10)])
    ax.set_yticklabels([f"组{r + 1}   d={bits_str(bits40[r * 4:r * 4 + 4], 4, '')}"
                        for r in range(10)], fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("③ Hamming(7,4) 分组：橙 = 校验位（位置 1/2/4），蓝 = 数据位（位置 3/5/6/7），"
                 "灰 = 0", fontsize=12.5, loc="left", pad=8)

    # ---- ④ 差分对宽度 ----
    ax = fig.add_subplot(gs[3])
    W_NOM, D = 63.0, 4.0
    for i in range(140):
        b = code[i // 2]
        wide = (b == 1 and i % 2 == 0) or (b == 0 and i % 2 == 1)
        h = (W_NOM + D) if wide else (W_NOM - D)
        col = C_ONE if b == 1 else "#9dc0dd"
        ax.add_patch(Rectangle((i + 0.05, 0), 0.90, h,
                               facecolor=col, edgecolor="none"))
    for k in range(0, 140, 2):
        ax.axvline(k + 0.5, color="#333333", lw=0.5, alpha=0.30)
    for k in range(0, 140, 14):
        ax.axvline(k + 0.5, color="#c02020", lw=1.0, alpha=0.55)
    ax.axhline(W_NOM, color="#c02020", lw=0.9, ls="--", alpha=0.7)
    ax.text(139.5, W_NOM + 1.2, "标称笔画宽 63 px", ha="right", fontsize=10,
            color="#c02020")
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 74)
    ax.set_ylabel("笔画宽度 (px)", fontsize=11)
    ax.set_xlabel("码元序号（每 2 个码元 = 1 bit；细线 = 差分对边界，红线 = 每 7 bit 一组）",
                  fontsize=11)
    ax.set_title("④ 70 个差分对 → 140 个码元：同一对内一宽（67px）一窄（59px），"
                 "深色 = bit 1，浅色 = bit 0", fontsize=12.5, loc="left", pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle(f"编码展开：{digits}  →  37 bit  →  Hamming(7,4)  →  70 bit  →  140 个码元",
                 fontsize=16, y=0.962)
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)
    return dict(digits=digits, bits37=bits37, bits40=bits40, code=code)


# ==============================================================================
# Markdown 对照表
# ==============================================================================
def write_md(records: List[dict], brk: dict, path: str) -> None:
    lines = []
    lines.append("# 样本与编码对照表\n")
    lines.append("> 由 `sample_gallery.py` 自动生成。码字为 Hamming(7,4) 编码后的 70 bit，"
                 "每 7 bit 一组（共 10 组）。\n")
    lines.append("## 一、样本清单\n")
    lines.append("| # | 号码 | 扰动 | 解码结果 | BER | 置信度 | 状态 |")
    lines.append("|---|---|---|---|---:|---:|---|")
    for i, r in enumerate(records, 1):
        lines.append(f"| {i} | `{r['digits']}` | {A.LEVEL_CN[r['level']]} | "
                     f"`{r['decoded']}` | {r['ber'] * 100:.2f}% | {r['conf']:.4f} | "
                     f"{'OK' if r['ok'] else 'FAIL'} |")
    lines.append("\n## 二、逐条编码\n")
    for i, r in enumerate(records, 1):
        b37 = A.payload_to_bits(r["digits"])
        lines.append(f"### {i}. `{r['digits']}`\n")
        lines.append(f"- 十进制值：`{int(r['digits'])}`")
        lines.append(f"- 37 bit 载荷：`{bits_str(b37, 4)}`")
        lines.append(f"- 70 bit 码字：`{bits_str(r['bits'], 7)}`\n")
    lines.append("## 三、编码展开示例\n")
    lines.append(f"载荷 `{brk['digits']}`：\n")
    lines.append(f"- 37 bit：`{bits_str(brk['bits37'], 4)}`")
    lines.append(f"- 40 bit（前补 3 个 0）：`{bits_str(brk['bits40'], 4)}`")
    lines.append(f"- 70 bit 码字：`{bits_str(brk['code'], 7)}`\n")
    lines.append("Hamming 分组（p=校验位，d=数据位）：\n")
    lines.append("| 组 | p1 | p2 | d1 | p4 | d2 | d3 | d4 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k in range(10):
        blk = brk["code"][k * 7:(k + 1) * 7]
        lines.append(f"| {k + 1} | " + " | ".join(str(x) for x in blk) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    _setup_font()
    os.makedirs(OUT, exist_ok=True)
    print("构建设计中 ...")
    design = A.ArrowDesign()
    embedder = A.WatermarkEmbedder(design, verbose=False)
    sim = A.CeramicCamSimulator(seed=20240501)
    decoder = A.WatermarkDecoder(design)

    p1 = os.path.join(OUT, "5_sample_gallery.png")
    recs = make_gallery(embedder, sim, decoder, p1)
    print(f"[输出] {p1}")
    for r in recs:
        print(f"   {r['digits']}  [{A.LEVEL_CN[r['level']]}]  -> {r['decoded']}  "
              f"BER={r['ber'] * 100:5.2f}%  {'OK' if r['ok'] else 'FAIL'}")

    p2 = os.path.join(OUT, "6_code_breakdown.png")
    brk = make_breakdown(recs[0]["digits"], p2)
    print(f"[输出] {p2}")

    p3 = os.path.join(OUT, "samples_codes.md")
    write_md(recs, brk, p3)
    print(f"[输出] {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
