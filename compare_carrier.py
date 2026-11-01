"""生成「边缘载体 vs 内孔载体」的视觉 + 定量对照图。"""
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


def imwrite_u(path: str, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())


def edge_x(img: np.ndarray, y: int, x0: int, x1: int, thr: float):
    row = img[y, x0:x1]
    idx = np.where(row < thr)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(x0)
    v0, v1 = row[i - 1], row[i]
    return float(x0 + i - 1 + (v0 - thr) / max(v0 - v1, 1e-6))


def make(digits: str = "13800138000") -> None:
    design = A.ArrowDesign()
    edge = A.WatermarkEmbedder(design, verbose=False)
    inner = I.InnerCarrier(design)
    p0 = design.template
    p1 = edge.embed(digits)["plate"]
    p2 = inner.embed(digits)["plate"]

    def crop(img, x, y, w, h, z):
        return cv2.resize(img[y:y + h, x:x + w], (w * z, h * z),
                          interpolation=cv2.INTER_NEAREST)

    fig = plt.figure(figsize=(18.6, 19.5), facecolor="white")
    gs = fig.add_gridspec(5, 3, height_ratios=[1.75, 2.5, 2.5, 2.3, 1.55],
                          hspace=0.33, wspace=0.06,
                          left=0.02, right=0.985, top=0.945, bottom=0.02)
    titles = ["(a) 原图（无码）",
              f"(b) 边缘载体 D={A.MOD_DEPTH:g}px",
              f"(c) 内孔载体 D={inner.depth:g}px"]
    imgs = [p0, p1, p2]

    ax = fig.add_subplot(gs[0, :])
    ax.imshow(p0, cmap="gray", vmin=0, vmax=255)
    ax.set_title("整幅印版 2200×600 px = 2.2 cm × 0.6 cm（缩到屏幕宽度时三者无法区分）",
                 fontsize=12, pad=6)
    ax.axis("off")

    # ---- R 竖干外轮廓（最敏感于游标锐度） ----
    x, y, w, h, z = 496, 158, 62, 178, 6
    for j, (img, tt) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(crop(img, x, y, w, h, z), cmap="gray", vmin=0, vmax=255)
        ax.set_title(tt + "\nR 竖干外轮廓 6×（游标锐度最敏感处）", fontsize=11)
        ax.axis("off")
        ax.plot([0, w * z], [0, 0], color="none")

    # ---- O 内环（内孔） ----
    x, y, w, h = 1298, 168, 132, 150
    for j, (img, tt) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(gs[2, j])
        ax.imshow(crop(img, x, y, w, h, 5), cmap="gray", vmin=0, vmax=255)
        ax.set_title(tt + "\nO 内环（内孔）5×", fontsize=11)
        ax.axis("off")

    # ---- 边缘位移曲线 ----
    ax = fig.add_subplot(gs[3, :])
    ys = np.arange(160, 336)
    for img, tt, col in zip(imgs, titles, ["#111111", "#d62728", "#1f77b4"]):
        xs = np.array([edge_x(img.astype(np.float32), int(v), 495, 560, 158.0)
                       for v in ys], dtype=object)
        xs = np.array([np.nan if v is None else v for v in xs], np.float64)
        ax.plot(ys, xs, color=col, lw=1.8 if "内孔" in tt else 1.4,
                label=tt, alpha=0.9 if "边缘" not in tt else 0.75)
    ax.set_ylabel("R 竖干左边缘的亚像素 x 坐标 (px)", fontsize=11)
    ax.set_xlabel("行号 y", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=11, ncol=3)
    ax.set_title(f"同一条外轮廓边的亚像素位置：原图与内孔版完全重合（峰峰值 0.00 px），"
                 f"边缘版起伏 {2 * A.MOD_DEPTH:.2f} px", fontsize=12)
    finite = []
    for line in ax.get_lines():
        ys = np.asarray(line.get_ydata(), np.float64)
        finite.append(ys[np.isfinite(ys)])
    if finite:
        stacked = np.concatenate(finite)
        mid = float(np.median(stacked))
        half = max(4.0, float(A.MOD_DEPTH) + 2.0)
        ax.set_ylim(mid - half, mid + half)

    # ---- 差值 ×12 ----
    for j, (img, tt) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(gs[4, j])
        d = np.abs(img.astype(np.int16) - p0.astype(np.int16)).astype(np.float32)
        ax.imshow(np.clip(d * 12, 0, 255), cmap="inferno", vmin=0, vmax=255)
        ax.set_title("|差值| ×12  " + tt, fontsize=11)
        ax.axis("off")

    fig.suptitle(f"载体对比：{digits}   —— 边缘版把波浪写在外轮廓，内孔版把波浪关进字母内部",
                 fontsize=17, y=0.973)
    fig.savefig(os.path.join(OUT, "12_carrier_cmp.png"), dpi=100, facecolor="white")
    plt.close(fig)

    # ---- 物理尺寸预览 ----
    phys = [cv2.resize(p, (330, 90), interpolation=cv2.INTER_AREA) for p in imgs]
    sep = np.full((6, 330), 255, np.uint8)
    stack = np.vstack([phys[0], sep, phys[1], sep, phys[2]])
    imwrite_u(os.path.join(OUT, "13_physical_scale.png"),
              cv2.resize(stack, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))

    # ---- 客观指标 ----
    t0 = (p0 < 128).astype(np.uint8)
    holes = [(1304, 166, 232, 284), (582, 170, 200, 112),
             (942, 170, 200, 112), (234, 193, 128, 163)]
    m = np.ones(p0.shape, bool)
    for hx, hy, hw, hh in holes:
        m[max(0, hy - 4):hy + hh + 4, max(0, hx - 4):hx + hw + 4] = False
    for nm, img in (("边缘版", p1), ("内孔版", p2)):
        t1 = (img < 128).astype(np.uint8)
        d = np.abs(img.astype(np.int16) - p0.astype(np.int16))
        print(f"{nm}: 二值差异 {int((t0 != t1).sum()):5d} px  "
              f"灰度差 max={d.max():3d} mean={d.mean():.3f}  "
              f"孔外最大差={d[m].max():3.0f}  覆盖率 {t0.mean():.4f}->{t1.mean():.4f}")
    print("已输出 12_carrier_cmp.png / 13_physical_scale.png")


if __name__ == "__main__":
    make()
