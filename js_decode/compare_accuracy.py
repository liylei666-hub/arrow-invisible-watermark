#!/usr/bin/env python3
"""Compare the JS clean-plate edge decoder with production ``arrow_decode``.

Generates clean edge plates with the same embedder ``gen_plate.py`` uses,
decodes each with Python (full sync + ECC + gate) and with
``js_decode/arrow_edge.js`` (canvas already aligned), and prints match rate,
margin / cc deltas, and the cases a phone photo would still miss.

    python js_decode/compare_accuracy.py

Exit status is 0 when every Python-accepted clean plate is accepted by JS
with the same 11 digits, unmarked and noise plates are rejected, and the
codec bits match. A translated plate is reported and does not fail the run:
that gap is the missing homography.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import arrow_decode as AD  # noqa: E402
import arrow_wm as A  # noqa: E402
import gen_plate  # noqa: E402

NODE = shutil.which("node")
CLI = os.path.join(ROOT, "js_decode", "decode_cli.js")

# Fixed 11-digit payloads. All sit inside the 37-bit field.
PAYLOADS = [
    "13800138000",
    "13912345678",
    "18600001234",
    "10000000000",
    "19999999999",
    "00000000001",
    "12345678901",
    "15500001111",
    "18888888888",
    "13000000000",
    "17712344321",
    "10987654321",
]


def _js_encode(digits: str) -> list:
    proc = subprocess.run(
        [NODE, CLI, "--encode", digits],
        check=True, capture_output=True, text=True)
    text = proc.stdout.strip()
    return [int(ch) for ch in text]


def _js_decode(plate: np.ndarray, debug: bool = False) -> dict:
    if plate.ndim != 2:
        raise ValueError("plate must be grayscale")
    h, w = plate.shape
    blob = struct.pack("<II", int(w), int(h)) + np.ascontiguousarray(plate, np.uint8).tobytes()
    fd, path = tempfile.mkstemp(prefix="arrow_js_", suffix=".raw")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        cmd = [NODE, CLI, path]
        if debug:
            cmd.append("--debug")
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    finally:
        os.remove(path)
    return json.loads(proc.stdout)


def _fmt(value, spec: str) -> str:
    if value is None:
        return "—"
    return format(value, spec)


def _row(name: str, py: AD.DecodeResult, js: dict) -> str:
    py_digits = py.digits if py.accepted else None
    js_digits = js.get("digits")
    flag = "OK" if py_digits == js_digits and bool(py.accepted) == bool(js.get("accepted")) else "DIFF"
    # Both rejected is a match on the trusted field (both None), even if the
    # candidate strings differ. Flag that separately only when one accepts.
    if (not py.accepted) and (not js.get("accepted")) and js_digits is None and py.digits is None:
        flag = "OK"
    return (
        f"{name:<16}{flag:>4}  "
        f"py={py_digits or '—':<12} js={js_digits or '—':<12} "
        f"margin {py.margin:6.3f}/{_fmt(js.get('margin'), '6.3f')}  "
        f"cc {py.cc:6.4f}/{_fmt(js.get('cc'), '6.4f')}  "
        f"py:{py.reason or 'accepted':<16} js:{js.get('reason') or 'accepted'}"
    )


def main() -> int:
    if not NODE:
        print("node is not on PATH", file=sys.stderr)
        return 2
    if not os.path.isfile(CLI):
        print(f"missing {CLI}", file=sys.stderr)
        return 2

    design = A.ArrowDesign(verbose=False)
    emb = A.WatermarkEmbedder(design, verbose=False)
    research = A.WatermarkDecoder(design, verbose=False)
    prod = AD.Decoder(carrier="edge", design=design)

    print("codec")
    codec_bad = 0
    for digits in PAYLOADS:
        py_bits = list(A.encode_payload(digits))
        js_bits = _js_encode(digits)
        if py_bits != js_bits:
            codec_bad += 1
            print(f"  BIT MISMATCH {digits}")
    print(f"  Hamming bits match {len(PAYLOADS) - codec_bad}/{len(PAYLOADS)}")

    print("\nclean plates  (Python = full sync+ECC+gate, JS = aligned canvas)")
    print(f"{'payload':<16}{'':>4}  {'python':<15} {'js':<15} margin py/js        cc py/js")
    match = 0
    py_accept = 0
    feature_max = 0.0
    margin_deltas = []
    cc_deltas = []
    conf_deltas = []
    scale_deltas = []
    failures = []
    for digits in PAYLOADS:
        plate = emb.embed(digits)["plate"]
        py = prod.decode(plate)
        js = _js_decode(plate, debug=True)
        feat = research.extract_widths(plate)
        widths = np.asarray(js.get("widths", []), np.float64)
        if widths.shape == feat.shape:
            feature_max = max(feature_max, float(np.max(np.abs(widths - feat))))
        else:
            feature_max = max(feature_max, float("inf"))
        if py.accepted:
            py_accept += 1
            margin_deltas.append(abs(py.margin - float(js["margin"])))
            cc_deltas.append(abs(py.cc - float(js["cc"])))
            conf_deltas.append(abs(py.mean_conf - float(js["mean_conf"])))
            if js.get("logo_scale") is not None and py.logo_scale is not None:
                scale_deltas.append(abs(py.logo_scale - float(js["logo_scale"])))
            same = bool(js.get("accepted") and js.get("digits") == digits and py.digits == digits)
            if same:
                match += 1
            else:
                failures.append(digits)
        elif js.get("digits") is not None:
            failures.append(digits + " (js exposed digits on a python reject)")
        print("  " + _row(digits, py, js))

    print("\nfile round-trip  (gen_plate.py PNG)")
    fd, png = tempfile.mkstemp(prefix="arrow_plate_", suffix=".png")
    os.close(fd)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = gen_plate.main([PAYLOADS[0], "-o", png, "--no-check"])
        if rc not in (0, None):
            print(f"  gen_plate exit {rc}")
        loaded = A.imread_u(png)
        py_png = prod.decode(loaded)
        js_png = _js_decode(loaded, debug=False)
        print("  " + _row(os.path.basename(png), py_png, js_png))
        png_ok = bool(py_png.accepted and py_png.digits == PAYLOADS[0]
                      and js_png.get("accepted") and js_png.get("digits") == PAYLOADS[0])
    finally:
        os.remove(png)

    print("\nrejects the gate must hold")
    blank = design.template
    py_blank = prod.decode(blank)
    js_blank = _js_decode(blank)
    print("  " + _row("unmarked", py_blank, js_blank))
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, blank.shape, dtype=np.uint8)
    py_noise = prod.decode(noise)
    js_noise = _js_decode(noise)
    print("  " + _row("noise", py_noise, js_noise))
    rejects_ok = (
        (not py_blank.accepted) and py_blank.digits is None
        and (not js_blank.get("accepted")) and js_blank.get("digits") is None
        and (not py_noise.accepted) and py_noise.digits is None
        and (not js_noise.get("accepted")) and js_noise.get("digits") is None
    )

    print("\nblocker  (same canvas, content shifted 15 px — needs homography)")
    shifted = np.full_like(blank, 255)
    shifted[:, 15:] = emb.embed(PAYLOADS[0])["plate"][:, :-15]
    py_shift = prod.decode(shifted)
    js_shift = _js_decode(shifted)
    print("  " + _row("shift+15px", py_shift, js_shift))
    shift_gap = bool(py_shift.accepted and py_shift.digits == PAYLOADS[0]
                     and not (js_shift.get("accepted") and js_shift.get("digits") == PAYLOADS[0]))

    print("\nsummary")
    print(f"  clean digit matches     {match}/{py_accept} python-accepted")
    print(f"  raw-plate feature max|Δ| {feature_max:.3e}  "
          f"(JS vs Python extract_widths, before ECC)")
    if margin_deltas:
        print(f"  |margin py−js|         max {max(margin_deltas):.4f}  "
              f"mean {float(np.mean(margin_deltas)):.4f}")
        print(f"  |cc py−js|              max {max(cc_deltas):.4f}  "
              f"mean {float(np.mean(cc_deltas)):.4f}")
        print(f"  |mean_conf py−js|       max {max(conf_deltas):.5f}")
        if scale_deltas:
            print(f"  |logo_scale py−js|      max {max(scale_deltas):.5f}  "
                  f"(JS uses identity homography)")
    print(f"  png round-trip          {'OK' if png_ok else 'FAIL'}")
    print(f"  unmarked+noise rejected {'OK' if rejects_ok else 'FAIL'}")
    print(f"  15px shift              "
          f"{'JS misses, Python accepts' if shift_gap else 'see row above'}")
    if failures:
        print("  failures: " + ", ".join(failures))

    clean_ok = match == py_accept == len(PAYLOADS) and not failures and codec_bad == 0 and png_ok and rejects_ok
    print("\nrecommendation")
    if clean_ok:
        print("  Clean aligned plates match. The half-width / soft-decode / gate")
        print("  port is faithful. Do not wire it up as a blind camera decoder:")
        print("  a 15 px shift is enough to leave the precomputed cells, and real")
        print("  phone photos still need SIFT homography + ECC (or a cloud call).")
    else:
        print("  Clean-plate accuracy does not match Python. Do not wire this")
        print("  module into the mini program. Prefer the hosted Python decoder")
        print("  until the gap above is closed.")
    return 0 if clean_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
