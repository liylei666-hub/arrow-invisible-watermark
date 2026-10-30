"""生成 ARROW 隐形水印印版（单张 / 批量，两种载体）。

用法示例
--------
# 单张，边缘载体，输出到当前目录
python gen_plate.py 13800138000

# 单张，内孔载体，指定输出文件
python gen_plate.py 13800138000 --carrier inner -o 13800138000_内孔.png

# 批量，两版都出，输出到 out 目录（同时生成 1bit 二值版）
python gen_plate.py 13800138000 13912345678 18600001234 --carrier both -d out --1bit

# 从文本文件读号码（每行一个，空行与 # 开头的行忽略）
python gen_plate.py --file 号码.txt --carrier edge -d out
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

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
    ap.add_argument("--carrier", choices=("edge", "inner", "both"), default="edge",
                    help="edge=边缘载体（默认，鲁棒性最好）；inner=内孔载体（外轮廓不变）")
    ap.add_argument("-o", "--out", default=None, help="单张时的输出文件名")
    ap.add_argument("-d", "--outdir", default=".", help="批量时的输出目录")
    ap.add_argument("--1bit", dest="onebit", action="store_true",
                    help="同时输出 1bit 二值版（供只收位图的制版流程）")
    ap.add_argument("--no-check", action="store_true", help="跳过生成后的自检解码")
    args = ap.parse_args(argv)

    payloads = load_payloads(args)
    carriers = ["edge", "inner"] if args.carrier == "both" else [args.carrier]
    if len(payloads) > 1 and args.out:
        raise SystemExit("多个号码时请用 -d 指定输出目录，不要用 -o")

    design = A.ArrowDesign()
    emb = {"edge": A.WatermarkEmbedder(design, verbose=False)}
    dec = {"edge": A.WatermarkDecoder(design)}
    if "inner" in carriers:
        import arrow_inner as I
        emb["inner"] = I.InnerCarrier(design)
        dec["inner"] = I.InnerDecoder(emb["inner"])

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
                ok = r.get("digits") == dg
                n_ok += int(ok)
                mark = "OK" if ok else "FAIL"
                conf = r.get("mean_conf", 0.0)
            print(f"{c:<6}{dg:<14}{os.path.basename(path):<34}"
                  f"{mark:>6}{conf:>10.4f}{dt:>7.0f}ms")
    print("-" * 80)
    if not args.no_check:
        print(f"自检通过 {n_ok}/{len(payloads) * len(carriers)}")
    print(f"输出目录：{os.path.abspath(args.outdir)}")
    print("\n解码：")
    print(f"  {sys.executable} arrow_wm.py --decode 照片1.jpg 照片2.jpg      # 边缘载体")
    if "inner" in carriers:
        print(f"  {sys.executable} arrow_inner.py --decode 照片1.jpg            # 内孔载体")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
