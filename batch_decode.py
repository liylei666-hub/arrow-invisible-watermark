"""批量解码：扫一个目录（或给一批文件），逐张解码并汇总成表格 / CSV。

用法
----
# 解码目录下所有图片（默认边缘载体）
python batch_decode.py 照片目录

# 内孔载体 + 输出 CSV
python batch_decode.py 照片目录 --carrier inner --csv 结果.csv

# 直接给若干文件
python batch_decode.py a.jpg b.jpg c.jpg

# 逐张出核验图
python batch_decode.py 照片目录 --save-figs
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

import arrow_wm as A

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def collect(paths) -> list:
    files = []
    for p in paths:
        if os.path.isdir(p):
            for nm in sorted(os.listdir(p)):
                if nm.lower().endswith(EXTS):
                    files.append(os.path.join(p, nm))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"[跳过] 找不到 {p}")
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="批量解码 ARROW 隐形水印",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("paths", nargs="+", help="图片文件或目录，可混给多个")
    ap.add_argument("--carrier", choices=("edge", "inner"), default="edge")
    ap.add_argument("--csv", default=None, help="把结果写成 CSV")
    ap.add_argument("--save-figs", action="store_true", help="逐张输出核验图")
    ap.add_argument("--thresh", type=float, default=None,
                    help="码字裕度门限（边缘默认 1.8；噪声/无码实测 ≤1.4）")
    args = ap.parse_args(argv)

    files = collect(args.paths)
    if not files:
        print("没有找到任何图片")
        return 1

    design = A.ArrowDesign()
    if args.carrier == "edge":
        dec = A.WatermarkDecoder(design)
        default_thr = 1.8
    else:
        import arrow_inner as I
        dec = I.InnerDecoder(I.InnerCarrier(design))
        # 内孔版现已返回码字裕度（与边缘版同一口径）。
        # 实测：真码 ≥3.3（含 0.13× 小图），无码 ≈0.4，噪声 ≤1.2；门限取 1.8。
        default_thr = 1.8
    thr = args.thresh if args.thresh is not None else default_thr

    print(f"载体 {args.carrier}   共 {len(files)} 张   裕度门限 {thr:g}")
    print("=" * 92)
    print(f"{'文件':<40}{'解码号码':<14}{'裕度':>9}{'cc':>8}{'耗时':>8}  判定")
    print("-" * 92)

    rows = []
    n_ok = 0
    for path in files:
        img = A.imread_u(path)
        name = os.path.basename(path)
        if img is None:
            print(f"{name:<40}{'读不到':<14}{'':>9}{'':>8}{'':>8}  跳过")
            rows.append([name, "", "", "", "", "读不到"])
            continue
        t0 = time.time()
        # 要出核验图时必须带 debug 信息（矫正图与单应），否则画不出来
        r = dec.decode(img, return_debug=bool(args.save_figs))
        dt = (time.time() - t0) * 1000
        if not r.get("ok"):
            print(f"{name:<40}{'解码失败':<14}{'':>9}{'':>8}{dt:>7.0f}ms  {r.get('reason')}")
            rows.append([name, "", "", "", f"{dt:.0f}", r.get("reason", "fail")])
            continue
        conf = r["mean_conf"]
        mg = float(r.get("margin", 0.0))
        cc = r["ecc"].get("cc", 0.0) if isinstance(r.get("ecc"), dict) else 0.0
        ok = mg >= thr
        n_ok += int(ok)
        tag = "有码" if ok else "裕度不足"
        shown = r["digits"] if r.get("payload_valid", True) else "码字越界"
        print(f"{name:<40}{shown:<14}{mg:>9.2f}{cc:>8.3f}{dt:>7.0f}ms  {tag}")
        rows.append([name, shown, f"{mg:.2f}", f"{cc:.3f}", f"{dt:.0f}", tag])

        if args.save_figs:
            try:
                import verify_decode as V
                # 复用上面已经算好的结果，不再重复解码一遍
                V.render(img, r, os.path.join(
                    os.path.dirname(path) or ".",
                    "核验_" + os.path.splitext(name)[0] + ".png"))
            except Exception as e:      # 出图失败不影响解码结果
                print(f"    （核验图生成失败：{e}）")

    print("=" * 92)
    print(f"有效解码 {n_ok}/{len(files)}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["文件", "号码", "码字裕度", "cc", "耗时ms", "判定"])
            w.writerows(rows)
        print("CSV ->", os.path.abspath(args.csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
