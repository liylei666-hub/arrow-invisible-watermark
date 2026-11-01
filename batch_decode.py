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

import arrow_decode as AD
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
                    help="码字裕度均值门限（默认 1.8；噪声/无码实测 ≤1.4）")
    ap.add_argument("--min-margin", type=float, default=None,
                    help="单组最小裕度门限（默认见 arrow_decode.MIN_GROUP_MARGIN）")
    ap.add_argument("--cc-thresh", type=float, default=None,
                    help="配准 cc 门限（默认 0.85，与解码器内部择优门相同）")
    args = ap.parse_args(argv)

    files = collect(args.paths)
    if not files:
        print("没有找到任何图片")
        return 1

    design = A.ArrowDesign()
    margin_thr = AD.MARGIN_THRESHOLD if args.thresh is None else args.thresh
    min_thr = AD.MIN_GROUP_MARGIN if args.min_margin is None else args.min_margin
    cc_thr = AD.CC_THRESHOLD if args.cc_thresh is None else args.cc_thresh
    dec = AD.Decoder(carrier=args.carrier, design=design,
                     margin_threshold=margin_thr, min_group_margin=min_thr,
                     cc_threshold=cc_thr)

    print(f"载体 {args.carrier}   共 {len(files)} 张   "
          f"裕度均值 ≥ {margin_thr:g}   单组最小 ≥ {min_thr:g}   cc ≥ {cc_thr:g}")
    print("=" * 104)
    print(f"{'文件':<36}{'解码号码':<14}{'均值':>8}{'最小':>8}{'cc':>8}{'耗时':>8}  判定")
    print("-" * 104)

    rows = []
    n_ok = 0
    for path in files:
        img = A.imread_u(path)
        name = os.path.basename(path)
        if img is None:
            print(f"{name:<36}{'读不到':<14}{'':>8}{'':>8}{'':>8}{'':>8}  跳过")
            rows.append([name, "", "", "", "", "", "读不到"])
            continue
        t0 = time.time()
        # 要出核验图时必须带 debug 信息（矫正图与单应），否则画不出来
        r = dec.decode(img, return_debug=bool(args.save_figs))
        dt = (time.time() - t0) * 1000
        if not r.ok:
            why = AD.reason_text(r.reason)
            print(f"{name:<36}{'解码失败':<14}{'':>8}{'':>8}{'':>8}{dt:>7.0f}ms  {why}")
            rows.append([name, "", "", "", "", f"{dt:.0f}", why])
            continue
        tag = "有码" if r.accepted else AD.reason_text(r.reason)
        # 号码列只写接受结果。拒绝时留空，避免把候选号码当成可信结果写进 CSV。
        shown = r.digits if r.accepted else ""
        n_ok += int(r.accepted)
        print(f"{name:<36}{(shown or '拒绝'):<14}{r.margin_mean:>8.2f}"
              f"{r.margin_min:>8.2f}{r.cc:>8.3f}{dt:>7.0f}ms  {tag}")
        rows.append([name, shown, f"{r.margin_mean:.2f}", f"{r.margin_min:.2f}",
                     f"{r.cc:.3f}", f"{dt:.0f}", tag])

        if args.save_figs and r.raw.get("bits") is not None:
            try:
                import verify_decode as V
                # 复用上面已经算好的结果，不再重复解码一遍。
                # 拒绝时标题写「结果不可信」，不把候选号画成成功解码。
                V.render(img, r.raw, os.path.join(
                    os.path.dirname(path) or ".",
                    "核验_" + os.path.splitext(name)[0] + ".png"),
                    accepted=r.accepted, reason=r.reason)
            except Exception as e:      # 出图失败不影响解码结果
                print(f"    （核验图生成失败：{e}）")

    print("=" * 104)
    print(f"有效解码 {n_ok}/{len(files)}（只计 accepted；拒绝行的号码列为空）")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["文件", "号码", "裕度均值", "裕度最小", "cc", "耗时ms", "判定"])
            w.writerows(rows)
        print("CSV ->", os.path.abspath(args.csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
