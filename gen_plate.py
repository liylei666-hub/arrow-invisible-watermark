"""生成 ARROW 隐形水印印版（单张 / 批量，两种载体）。

用法示例
--------
# 单张，内孔载体（默认）。边缘幅度仍是 arrow_wm.MOD_DEPTH（当前 6 px）
python gen_plate.py 13800138000

# 指定输出文件（默认仍是内孔）
python gen_plate.py 13800138000 -o 13800138000.png

# 批量，两版都出，输出到 out 目录（同时生成 1bit 二值版）
python gen_plate.py 13800138000 13912345678 18600001234 --carrier both -d out --1bit

# 从文本文件读号码（每行一个，空行与 # 开头的行忽略）。旧边缘印版显式 --carrier edge
python gen_plate.py --file 号码.txt --carrier edge -d out
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import arrow_decode as AD
import arrow_wm as A

CARRIERS = ("edge", "inner")


def load_payloads(args) -> list:
    raw = list(args.digits or [])
    if args.file:
        with open(args.file, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    raw.append(line)
    out = []
    for d in raw:
        d = d.strip().replace(" ", "").replace("-", "")
        if not d.isdigit() or len(d) != 11:
            raise SystemExit(f"号码格式不对：{d!r}（需要 11 位纯数字）")
        if int(d) >= (1 << 37):
            raise SystemExit(f"号码超出 37 bit 范围：{d}")
        if d not in out:
            out.append(d)
    if not out:
        raise SystemExit("没有给出任何号码，见 --help")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="生成 ARROW 隐形水印印版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("digits", nargs="*", help="11 位号码，可给多个")
    ap.add_argument("--file", default=None, help="号码清单文件（每行一个）")
    ap.add_argument("--carrier", choices=("edge", "inner", "both"), default="inner",
                    help="inner=内孔载体（默认）；edge=边缘载体（旧印版，需显式指定）；both=两版都出")
    ap.add_argument("-o", "--out", default=None, help="单张时的输出文件名")
    ap.add_argument("-d", "--outdir", default=".", help="批量时的输出目录")
    ap.add_argument("--1bit", dest="onebit", action="store_true",
                    help="同时输出 1bit 二值版（供只收位图的制版流程）")
    ap.add_argument("--depth", type=float, default=None,
                    help=f"边缘调制深度，默认 {A.MOD_DEPTH:g}；--layout 1 且不写时用 {A.V1_DEPTH:g}")
    ap.add_argument("--layout", type=int, choices=(1, 2), default=2,
                    help="边缘布局版本，1=冻结旧印版，2=当前默认")
    ap.add_argument("--inner-layout", type=int, choices=(2, 3), default=3,
                    help="内孔版式，3=交织+CRC-7（默认，新印版），2=旧印版")
    ap.add_argument("--no-check", action="store_true", help="跳过生成后的自检解码")
    args = ap.parse_args(argv)

    payloads = load_payloads(args)
    carriers = ["edge", "inner"] if args.carrier == "both" else [args.carrier]
    if len(payloads) > 1 and args.out:
        raise SystemExit("多个号码时请用 -d 指定输出目录，不要用 -o")

    depth = args.depth
    if depth is None and args.layout == 1:
        depth = A.V1_DEPTH
    elif depth is None:
        depth = A.MOD_DEPTH
    design = A.ArrowDesign(depth=depth, layout_version=args.layout, verbose=False)
    emb = {"edge": A.WatermarkEmbedder(design, verbose=False)}
    # 自检走生产门：号码对得上且 accepted 才算 OK。干净印版应同时过裕度与 cc。
    dec = {"edge": AD.Decoder(carrier="edge", design=design)}
    if "inner" in carriers:
        import arrow_inner as I
        emb["inner"] = I.InnerCarrier(design, layout_version=args.inner_layout)
        dec["inner"] = AD.Decoder(carrier="inner", design=design,
                                  layout_version=args.inner_layout)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"{'载体':<6}{'号码':<14}{'文件':<34}{'自检':>6}{'置信度':>10}{'耗时':>8}")
    print("-" * 80)
    n_ok = 0
    for dg in payloads:
        for c in carriers:
            t0 = time.time()
            wm = emb[c].embed(dg)
            dt = (time.time() - t0) * 1000
            if args.out and len(payloads) == 1 and len(carriers) == 1:
                path = args.out
            else:
                tag = "" if args.carrier != "both" else ("_边缘" if c == "edge" else "_内孔")
                path = os.path.join(args.outdir, f"ARROW_{dg}{tag}.png")
            A.imwrite_u(path, wm["plate"])
            if args.onebit:
                base, ext = os.path.splitext(path)
                A.imwrite_u(base + "_1bit" + ext,
                            np.where(wm["plate"] > 127, 255, 0).astype(np.uint8))
            if args.no_check:
                mark, conf = "—", 0.0
            else:
                r = dec[c].decode(wm["plate"])
                ok = bool(r.accepted and r.digits == dg)
                n_ok += int(ok)
                mark = "OK" if ok else "FAIL"
                conf = r.mean_conf
            print(f"{c:<6}{dg:<14}{os.path.basename(path):<34}"
                  f"{mark:>6}{conf:>10.4f}{dt:>7.0f}ms")
    print("-" * 80)
    if not args.no_check:
        print(f"自检通过 {n_ok}/{len(payloads) * len(carriers)}")
    print(f"输出目录：{os.path.abspath(args.outdir)}")
    print("\n解码：")
    if "inner" in carriers:
        print(f"  {sys.executable} arrow_inner.py --decode 照片1.jpg            # 内孔载体")
    if "edge" in carriers:
        print(f"  {sys.executable} arrow_wm.py --decode 照片1.jpg 照片2.jpg      # 边缘载体")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
