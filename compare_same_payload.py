"""同一号码下，「原图 / 边缘载体 / 内孔载体」三版并排对照。"""
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import arrow_wm as A
import arrow_inner as I

OUT = "arrow_wm_out"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DIGITS = "13800138000"


def make(digits: str = DIGITS) -> None:
    design = A.ArrowDesign()
    edge = A.WatermarkEmbedder(design, verbose=False)
    inner = I.InnerCarrier(design)

    p0 = design.template
    p1 = edge.embed(digits)["plate"]
    p2 = inner.embed(digits)["plate"]

    names = ["(a) 原图（无码）",
             f"(b) 边缘载体  D={A.MOD_DEPTH:g}px",
             f"(c) 内孔载体  D={inner.depth:g}px"]
    imgs = [p0, p1, p2]

    # 只保留文字带，去掉大片白边，便于并排看
    y0, y1 = 85, 515
    band = [im[y0:y1] for im in imgs]

    fig = plt.figure(figsize=(18.4, 15.6), facecolor="white")
    gs = fig.add_gridspec(5, 3, height_ratios=[1.0, 1.0, 1.0, 1.15, 1.15],
                          hspace=0.30, wspace=0.05,
                          left=0.015, right=0.985, top=0.935, bottom=0.02)

    # ---- 三行整幅 ----
    for r, (b, nm) in enumerate(zip(band, names)):
        ax = fig.add_subplot(gs[r, :])
        ax.imshow(b, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.set_title(nm + f"      载荷 {digits}", fontsize=13, pad=4, loc="left")
        ax.axis("off")

    # ---- R 竖干外轮廓 6× ----
    def crop(img, x, y, w, h, z):
        return cv2.resize(img[y:y + h, x:x + w], (w * z, h * z),
                          interpolation=cv2.INTER_NEAREST)

    for j, (im, nm) in enumerate(zip(imgs, names)):
        ax = fig.add_subplot(gs[3, j])
        ax.imshow(crop(im, 496, 158, 60, 176, 6), cmap="gray",
                  vmin=0, vmax=255)
        ax.set_title(f"R 竖干外轮廓 6×   {nm.split('（')[0]}", fontsize=11)
        ax.axis("off")

    # ---- A 三角孔 5× ----
    for j, (im, nm) in enumerate(zip(imgs, names)):
        ax = fig.add_subplot(gs[4, j])
        ax.imshow(crop(im, 226, 186, 146, 178, 5), cmap="gray",
                  vmin=0, vmax=255)
        ax.set_title(f"A 三角孔（内孔）5×   {nm.split('（')[0]}", fontsize=11)
        ax.axis("off")

    fig.suptitle(f"同一张水印图的三版对照 —— 载荷 {digits}"
                 f"    上三行：整幅；下两行：局部放大",
                 fontsize=16, y=0.968)
    path = os.path.join(OUT, "14_same_payload_3way.png")
    fig.savefig(path, dpi=95, facecolor="white")
    plt.close(fig)

    # ---- 客观指标 ----
    t0 = (p0 < 128).astype(np.uint8)
    holes = [(1304, 166, 232, 284), (582, 170, 200, 112),
             (942, 170, 200, 112), (234, 193, 128, 163)]
    m = np.ones(p0.shape, bool)
    for hx, hy, hw, hh in holes:
        m[max(0, hy - 4):hy + hh + 4, max(0, hx - 4):hx + hw + 4] = False
    print(f"{'版本':<10}{'二值差异':>10}{'孔外最大差':>12}{'墨水覆盖率':>12}")
    for nm, im in (("边缘版", p1), ("内孔版", p2)):
        d = np.abs(im.astype(np.int16) - p0.astype(np.int16))
        t1 = (im < 128).astype(np.uint8)
        print(f"{nm:<10}{int((t0 != t1).sum()):>10}{int(d[m].max()):>12}"
              f"{t1.mean() * 100:>11.2f}%")
    # ---- 实物尺寸观感（2.2 cm 宽） ----
    fig = plt.figure(figsize=(7.2, 5.0), facecolor="white")
    gs = fig.add_gridspec(3, 1, hspace=0.40, left=0.02, right=0.98,
                          top=0.78, bottom=0.05)
    for r, (im, nm) in enumerate(zip(imgs, names)):
        small = cv2.resize(im, (330, 90), interpolation=cv2.INTER_AREA)
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(small, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.set_title(nm, fontsize=10, pad=3, loc="left")
        ax.axis("off")
    fig.suptitle("实物尺寸观感：整幅宽 2.2 cm\n"
                 "本图按 330 px 渲染，约为实物尺寸的 4 倍，比实物更严苛\n"
                 "波动幅度：边缘版 ±0.018 mm，内孔版 ±0.022 mm",
                 fontsize=9.5, y=0.985)
    p2path = os.path.join(OUT, "15_physical_labeled.png")
    fig.savefig(p2path, dpi=150, facecolor="white")
    plt.close(fig)

    print("已输出", path, "/", p2path)


if __name__ == "__main__":
    make()
