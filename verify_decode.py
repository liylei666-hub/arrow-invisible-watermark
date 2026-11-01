"""实拍照片解码 + 核验图（边缘载体）。

用法：
    python verify_decode.py 照片.jpg
    python verify_decode.py 照片.jpg --digits 40881529652
    python verify_decode.py 照片.jpg --suspect 40881529652   （验真模式）
    python verify_decode.py 照片.jpg --out 核验.png
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import arrow_decode as AD
import arrow_wm as A

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def quad_from_H(design, H):
    Hi = np.linalg.inv(H.astype(np.float64)).astype(np.float32)
    return cv2.perspectiveTransform(
        design.ref_quad.astype(np.float32).reshape(1, -1, 2), Hi).reshape(-1, 2)


def render(img, res, out, digits=None, *, accepted=None, reason=""):
    """用**已经算好的**解码结果画核验图。

    抽成独立函数是为了让 batch_decode 复用同一个结果——
    之前批量出图会为每张图再完整解码一次，白跑一遍。

    ``accepted is False`` 时标题写明拒绝，不把研究解码器的候选号当成结果。
    """
    design = A.ArrowDesign()
    shown = img
    if res.get("isolated"):
        alt = A.WatermarkDecoder._isolate_plate(
            img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if alt is not None:
            shown = alt
    H = res.get("H")
    bits = np.array(res["bits"])
    conf = np.array(res["conf"])
    cc = float(res.get("cc", (res.get("ecc") or {}).get("cc", 0.0)))
    truth = np.array(A.encode_payload(digits)) if digits else None
    bad = np.where(bits != truth)[0] if truth is not None else np.array([], int)

    fig = plt.figure(figsize=(15, 10.2), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.95, 1.5],
                          hspace=0.36, left=0.035, right=0.965, top=0.90,
                          bottom=0.06)

    ax = fig.add_subplot(gs[0])
    ax.imshow(shown, cmap="gray", vmin=0, vmax=255)
    if H is not None:
        ax.add_patch(plt.Polygon(quad_from_H(design, H), closed=True,
                                 fill=False, edgecolor="#ff2d55", lw=2.2))
    tag = "（已自动隔离亮底卡片）" if res.get("isolated") else ""
    ax.set_title(f"(a) 输入照片 {shown.shape[1]}×{shown.shape[0]} px{tag}    "
                 f"红框 = 自同步检测到的 ARROW 区域", fontsize=12)
    ax.axis("off")

    ax = fig.add_subplot(gs[1])
    if res.get("warped") is not None:
        ax.imshow(res["warped"], cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax.set_title(f"(b) 透视矫正 + ECC 精配（2200×600）   cc = {cc:.3f}", fontsize=12)
    ax.axis("off")

    ax = fig.add_subplot(gs[2])
    x = np.arange(len(conf))
    cols = ["#d62728" if i in bad else ("#2ca02c" if bits[i] else "#1f77b4")
            for i in x]
    ax.bar(x, conf, color=cols, width=0.78)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xlim(-0.7, len(conf) - 0.3)
    ax.set_xlabel("比特序号（每 7 bit 一组 Hamming 码，虚线为组边界）", fontsize=11)
    ax.set_ylabel("软信息（正 = 判为 1）", fontsize=11)
    for k in range(0, len(conf) + 1, 7):
        ax.axvline(k - 0.5, color="#999", lw=0.8, ls=":")
    ax.grid(axis="y", alpha=0.25)
    sub = f"红=与真值不符 {len(bad)} 个" if truth is not None else "绿=1  蓝=0"
    if accepted is False:
        why = AD.reason_text(reason) if reason else "拒绝"
        decoded_line = f"拒绝（{why}），结果不可信"
        headline = "实拍解码：拒绝（结果不可信）"
    elif accepted is True:
        decoded_line = f"解码 {res.get('digits')}"
        headline = f"实拍解码：{res.get('digits')}"
    else:
        decoded_line = "解码结果未经过生产门"
        headline = "实拍解码：结果未判定"
    ax.set_title(f"(c) 比特判决与置信度   {sub}   "
                 f"{decoded_line}   平均置信度 {res['mean_conf']:.4f}",
                 fontsize=12)

    fig.suptitle(headline, fontsize=16, y=0.955)
    fig.savefig(out, dpi=105, facecolor="white")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--digits", default=None, help="已知号码，用于逐比特核验")
    ap.add_argument("--suspect", default=None,
                    help="验真模式：判定图中是否包含该号码（盲解失败时用）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    img = A.imread_u(args.image)
    if img is None:
        print("读不到文件：", args.image)
        return 1

    session = AD.Decoder(carrier="edge")
    result = session.decode(img, return_debug=True)
    res = result.raw
    if not result.ok:
        print("解码失败：", AD.reason_text(result.reason))
        return 2

    if args.suspect:
        if "conf" not in res:
            print("验真失败：没有软信息")
            return 2
        scored = AD.score_suspect(np.array(res["conf"], float), args.suspect)
        print(f"验真号码  {scored.suspect}")
        print(f"相关分    {scored.score:.4f}   z = {scored.z:.1f} σ")
        neighbor = scored.best_neighbor or "无"
        print(f"最近邻号  {neighbor}   z = {scored.neighbor_z:.1f} σ")
        print(f"间隙      {scored.gap:.2f} σ"
              f"（强证据还需 ≥ {AD.SUSPECT_GAP_STRONG:g}，"
              f"且 z ≥ {AD.SUSPECT_Z_STRONG:g}）")
        print(f"判定      {scored.verdict}")
        blind = ("接受 " + result.digits) if result.accepted else (
            f"拒绝（{AD.reason_text(result.reason)}），结果不可信")
        print(f"（盲解{blind}，裕度均值 {result.margin_mean:.2f}，"
              f"最小 {result.margin_min:.2f}，"
              f"置信度 {result.mean_conf:.4f}，仅作参考）")
        return 0

    print(f"文件      {os.path.basename(args.image)}")
    print(f"尺寸      {img.shape[1]}×{img.shape[0]} px")
    if result.accepted:
        print(f"解码号码  {result.digits}")
        print(f"判定      接受")
    else:
        cand = result.candidate_digits or "无"
        print(f"解码号码  （拒绝，不作为可信号码）")
        print(f"候选      {cand}（不可信）")
        print(f"判定      拒绝（{AD.reason_text(result.reason)}）")
    scale = f"{result.logo_scale:.3f}" if result.logo_scale is not None else "—"
    print(f"置信度    {result.mean_conf:.4f}   "
          f"裕度均值 {result.margin_mean:.2f}   最小 {result.margin_min:.2f}   "
          f"配准 cc = {result.cc:.3f}   logo_scale = {scale}")
    if args.digits:
        truth = np.array(A.encode_payload(args.digits))
        bad = np.where(np.array(res["bits"]) != truth)[0]
        print(f"逐比特    错 {len(bad)}/{len(truth)}"
              + (f"  位置 {bad.tolist()}" if len(bad) else "  （完美）"))
        got = result.digits if result.accepted else result.candidate_digits
        note = "" if result.accepted else "（候选，未接受）"
        print(f"号码比对  {'一致' if got == args.digits else '不一致！'}{note}")

    out = args.out or os.path.join(
        "arrow_wm_out",
        f"decode_{os.path.splitext(os.path.basename(args.image))[0][:24]}.png")
    render(img, res, out, args.digits,
           accepted=result.accepted, reason=result.reason)
    print("核验图    ", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
