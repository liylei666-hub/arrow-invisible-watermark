"""生成 10 组随机号码的实测样张（边缘版 + 内孔版各一套）。"""
import csv
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import arrow_wm as A
import arrow_inner as I

ROOT = "实测样张"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261029
N = 10


def gen_payloads(n: int = N, seed: int = SEED):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        d = "".join(str(x) for x in rng.integers(0, 10, size=11))
        if d[0] == "0":
            continue                       # 首位为 0 的号码不符合手机号习惯
        if d in out:
            continue
        out.append(d)
    return out


def main() -> None:
    payloads = gen_payloads()
    design = A.ArrowDesign()
    edge = A.WatermarkEmbedder(design, verbose=False)
    inner = I.InnerCarrier(design)
    dec_e = A.WatermarkDecoder(design)
    dec_i = I.InnerDecoder(inner)

    d_edge = os.path.join(ROOT, "边缘版")
    d_inner = os.path.join(ROOT, "内孔版")
    os.makedirs(d_edge, exist_ok=True)
    os.makedirs(d_inner, exist_ok=True)

    A.imwrite_u(os.path.join(ROOT, "ARROW_无码对照.png"), design.template)

    rows = []
    print(f"{'#':<3}{'号码':<14}{'边缘版':>16}{'内孔版':>16}")
    print("-" * 52)
    for i, dg in enumerate(payloads, 1):
        wm_e = edge.embed(dg)
        wm_i = inner.embed(dg)
        A.imwrite_u(os.path.join(d_edge, f"ARROW_{dg}.png"), wm_e["plate"])
        A.imwrite_u(os.path.join(d_edge, f"ARROW_{dg}_1bit.png"),
                    np.where(wm_e["plate"] > 127, 255, 0).astype(np.uint8))
        A.imwrite_u(os.path.join(d_inner, f"ARROW_{dg}.png"), wm_i["plate"])
        A.imwrite_u(os.path.join(d_inner, f"ARROW_{dg}_1bit.png"),
                    np.where(wm_i["plate"] > 127, 255, 0).astype(np.uint8))

        re_ = dec_e.decode(wm_e["plate"])
        ri = dec_i.decode(wm_i["plate"])
        ok_e = re_.get("digits") == dg
        ok_i = ri.get("digits") == dg
        ce = re_.get("mean_conf", 0.0)
        ci = ri.get("mean_conf", 0.0)
        tag_e = ("OK  " if ok_e else "FAIL") + f" {ce:7.4f}"
        tag_i = ("OK  " if ok_i else "FAIL") + f" {ci:7.4f}"
        print(f"{i:<3}{dg:<14}{tag_e:>16}{tag_i:>16}")
        rows.append([i, dg, "OK" if ok_e else "FAIL", f"{ce:.4f}",
                     "OK" if ok_i else "FAIL", f"{ci:.4f}"])
    print("-" * 52)

    # ---- 答案对照表 ----
    key = os.path.join(ROOT, "答案对照表.csv")
    with open(key, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["序号", "号码", "边缘版自检", "边缘版置信度",
                    "内孔版自检", "内孔版置信度"])
        w.writerows(rows)

    # ---- 视觉索引图（缩略，仅用于辨认） ----
    fig = plt.figure(figsize=(13.5, 9.2), facecolor="white")
    gs = fig.add_gridspec(5, 2, hspace=0.42, wspace=0.06,
                          left=0.03, right=0.98, top=0.93, bottom=0.02)
    for i, dg in enumerate(payloads):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        small = cv2.resize(edge.embed(dg)["plate"], (880, 240),
                           interpolation=cv2.INTER_AREA)
        ax.imshow(small, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.set_title(f"#{i + 1}   {dg}", fontsize=11, pad=2, loc="left")
        ax.axis("off")
    fig.suptitle("10 组实测样张（缩略索引，仅供辨认；送印请用 1:1 原图）",
                 fontsize=14, y=0.975)
    fig.savefig(os.path.join(ROOT, "索引图_边缘版.png"), dpi=110,
                facecolor="white")
    plt.close(fig)
    print("索引图 ->", os.path.join(ROOT, "索引图_边缘版.png"))


if __name__ == "__main__":
    main()


def make_a4_sheet(payloads, dpi: int = 600) -> None:
    """A4 @600dpi 排版：每个 logo 精确 2.2 cm 宽，2 列 × 5 行。"""
    W, H = 4961, 7016                      # A4 @600dpi
    LOGO_W = 520                           # 2.2 cm @600dpi
    LOGO_H = int(round(LOGO_W * 600 / 2200.0))
    sheet = np.full((H, W), 255, np.uint8)
    design = A.ArrowDesign()
    edge = A.WatermarkEmbedder(design, verbose=False)

    xs = [int(W * 0.10), int(W * 0.54)]
    ys = [int(H * 0.09) + i * int(H * 0.178) for i in range(5)]
    for i, dg in enumerate(payloads):
        plate = edge.embed(dg)["plate"]
        small = cv2.resize(plate, (LOGO_W, LOGO_H), interpolation=cv2.INTER_AREA)
        x, y = xs[i % 2], ys[i // 2]
        sheet[y:y + LOGO_H, x:x + LOGO_W] = small
        # 标签：序号 + 号码（不想看答案可打印后遮住）
        cv2.putText(sheet, f"#{i + 1}   {dg}", (x, y + LOGO_H + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.9, 60, 4, cv2.LINE_AA)
        # 2.2 cm 比例尺
        cv2.line(sheet, (x, y + LOGO_H + 128), (x + LOGO_W, y + LOGO_H + 128), 140, 5)
        cv2.putText(sheet, "2.2 cm", (x + LOGO_W + 30, y + LOGO_H + 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.35, 140, 3, cv2.LINE_AA)

    cv2.putText(sheet, "ARROW invisible watermark - 10 test samples (A4 @600dpi, logo = 2.2 cm)",
                (int(W * 0.10), int(H * 0.055)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.9, 40, 4, cv2.LINE_AA)
    out = os.path.join(ROOT, "打印排版图_A4_600dpi.png")
    A.imwrite_u(out, sheet)
    print("A4 排版图 ->", out, f"({W}x{H})")
