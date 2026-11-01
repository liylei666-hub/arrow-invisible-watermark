"""Production decode API for the ARROW watermark.

The research decoders in ``arrow_wm`` / ``arrow_inner`` always emit a digit
string once sync succeeds. This module is the integrator-facing wrapper:

* default carrier is the inner-hole watermark (``carrier="inner"``)
* ``carrier="edge"`` selects the edge watermark for legacy plates; its algorithm is unchanged
* inner layout defaults to ``auto``: decode v3 (interleaved, CRC-7) first and
  stop when the CRC passes; otherwise decode v2 (legacy plates, group-0 pad).
  Both layouts share one registration pass. ``layout_version=2`` or ``3``
  pins one layout.
* a hard gate sets ``accepted=False`` and ``digits=None`` unless the readout
  clears the mean codeword-margin threshold, the per-group minimum margin,
  the integrity check (v3 CRC-7, or the v2 group-0 pad ``000``), and, when
  ECC refinement is on, the registration ``cc`` gate.
  ``use_ecc_refine=False`` (``--no-ecc``) skips only the cc gate: that path
  does not produce a registration score (``cc`` stays 0). The margin gates
  and the other accept rules still apply. Default remains ECC on.

``margin`` / ``margin_mean`` is the average of the ten group gaps. One
wrong or tied group is hidden by that average, so ``margin_min`` is a
separate hard gate.

``candidate_digits`` on the Python object is the research decoder's raw
read. It is not a trusted phone number. The HTTP service omits it unless
debug mode and a valid token are both present.

Clients decide success only with ``accepted is True`` or
``status == "accepted"``. ``ok`` means the research decoder finished a
sync; it is true for unmarked plates that this gate rejects. ``status``
is ``accepted``, ``rejected``, or (HTTP layer only) ``error``.

Batch callers should reuse :class:`Decoder`. :func:`decode` builds a new
session on every call (template + SIFT init).
"""
from __future__ import annotations

import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# Production default. Edge stays selectable for legacy plates and is not tuned here.
DEFAULT_CARRIER = "inner"
CARRIERS = ("edge", "inner")

# These cuts are z-scores of amplitude-normalized soft values (soft_n),
# not pixel offsets and not the old 1.8 px modulation depth.
# Edge decode divides the matched-filter differential by π·depth before
# the gap/std margin is computed. A global scale does not move the z-score,
# so the numeric cuts stay put when depth changes; per-bit depth_scale would.
# v2@6 was checked against the 1.8 mean cut (see README ablation).
# True codes still sit at or above ~2.0 and unmarked plates at or below ~1.4.
# The 1.8 cut is a watermark-present gate, not a digit-correct gate.
MARGIN_THRESHOLD = 1.8

# Hard gate on the worst Hamming group (best-vs-2nd gap / std of soft_n).
# A single near-tie group can leave the mean above 1.8. Confidently wrong
# groups (large gap on the wrong codeword) are not caught here; that needs
# a payload CRC, which is out of scope.
# The old 1.0 cut sat between clean-plate accepts (margin_min >= 1.24 on the
# pre-v2 ±6 px probe) and two wrong reads at 0.17 and 0.32. A real inner
# photo of 17701234567 reads that number with margin_min ≈ 0.604. The
# near-tie smoke (one group at amplitude 0.04) stays near 0.325, so 0.55
# still rejects that tie and the 0.17 / 0.32 wrong reads. Not a pixel cut.
MIN_GROUP_MARGIN = 0.55

# --suspect: absolute z is not enough. The closest one-digit neighbor is
# Hamming distance 3, so ideal soft info still scores that neighbor at
# about z - 0.72. Strong evidence needs the gap as well as z.
SUSPECT_Z_STRONG = 4.5
SUSPECT_Z_MAYBE = 3.0
SUSPECT_GAP_STRONG = 0.45
SUSPECT_GAP_MAYBE = 0.15
PAD_BITS = (0, 0, 0)

# Both research decoders already refuse to *prefer* a path whose ECC
# correlation is below 0.85, because a bad crop can inflate mean |conf|.
# The production gate treats that same cut as a hard reject.
CC_THRESHOLD = 0.85

# At ~0.08× the plate width the wave is a fraction of a camera pixel
# (±6 px at scale 0.077 is ±0.46 px); that documented tiny frame stays
# rejected. Extreme sims in this repo sit around 0.7–0.9, so the cut does
# not touch that bench. Two real inner photos of 17701234567 bracket the cut:
#   scale ≈ 0.160 -> correct, margin 6.07 (original 0.160 sample)
#   scale ≈ 0.131 -> correct, margin 2.82, margin_min 0.66, cc 0.976
# The 0.131 frame was rejected only by this cut, with every signal gate
# comfortably green. The product owner set the cut to 0.10 (from 0.15):
# at 0.10 the ±6 px modulation is ±0.60 camera px per side (1.20 px
# differential), and the margin / pad / cc gates still decide readability.
# 0.077 remains under the cut, so the documented tiny frame is still rejected.
# A frame under the cut is rejected even when registration cc is high.
# It does not relax the margin gates, and a larger digital upscale of the
# same pixels is a different image that still has to clear margin / pad.
LOGO_SCALE_MIN = 0.10

ImageLike = Union[np.ndarray, str, os.PathLike]

# ---- 共享构造缓存 ----------------------------------------------------------
# ArrowDesign 构造一次约 200 ms（栅格化 + 布局推导），Decoder 还要再花
# 250–500 ms 建模板 / SIFT / 内孔映射。一次性接口 decode() 原先每次调用都
# 重建一遍，单次调用里 3/4 的时间花在这上面。设计对象构造后只读，可以安全
# 复用；会话本身有内部状态（OpenCV matcher 不做重入保证），所以缓存里同时
# 存一把锁，同一会话的并发调用仍然串行。
_DESIGN_CACHE_MAX = 8
_SESSION_CACHE_MAX = 8
_design_cache: "OrderedDict[tuple, object]" = OrderedDict()
_session_cache: "OrderedDict[tuple, Tuple[object, threading.RLock]]" = OrderedDict()
# RLock：_shared_session 持锁构造 Decoder，而 Decoder 内部又会调
# _shared_design 取同一把锁。普通 Lock 会在这里自锁。
_cache_lock = threading.RLock()


def _design_key(kwargs: Dict) -> tuple:
    import arrow_wm as A
    return (int(kwargs.get("layout_version", A.LAYOUT_VERSION)),
            float(kwargs.get("depth", A.MOD_DEPTH)),
            float(kwargs.get("window_pow", A.WINDOW_POW)))


def _shared_design(**kwargs):
    """Return a process-wide ArrowDesign for this geometry.

    ``verbose=True`` always builds a fresh one so the layout prints still
    happen; it is a debugging path, not the hot one.
    """
    import arrow_wm as A
    if kwargs.get("verbose"):
        return A.ArrowDesign(**kwargs)
    key = _design_key(kwargs)
    with _cache_lock:
        design = _design_cache.get(key)
        if design is None:
            design = A.ArrowDesign(**kwargs)
            _design_cache[key] = design
            while len(_design_cache) > _DESIGN_CACHE_MAX:
                _design_cache.popitem(last=False)
        else:
            _design_cache.move_to_end(key)
        return design


def _session_key(carrier: str, margin_threshold: float,
                 min_group_margin: float, cc_threshold: Optional[float],
                 design, depth: Optional[float], layout_version: Optional[int],
                 window_pow: Optional[float], use_ecc_refine: bool,
                 verbose: bool) -> tuple:
    return (carrier, float(margin_threshold), float(min_group_margin),
            None if cc_threshold is None else float(cc_threshold),
            None if design is None else id(design),
            None if depth is None else float(depth),
            None if layout_version is None else int(layout_version),
            None if window_pow is None else float(window_pow),
            bool(use_ecc_refine), bool(verbose))


def _shared_session(**kwargs) -> Tuple["Decoder", threading.RLock]:
    """A cached Decoder plus the lock that serializes its use.

    Keyed on every constructor argument that changes behavior, thresholds
    included, so a call can never inherit another call's gate settings.
    """
    key = _session_key(
        kwargs.get("carrier", DEFAULT_CARRIER),
        kwargs.get("margin_threshold", MARGIN_THRESHOLD),
        kwargs.get("min_group_margin", MIN_GROUP_MARGIN),
        kwargs.get("cc_threshold", CC_THRESHOLD),
        kwargs.get("design"), kwargs.get("depth"), kwargs.get("layout_version"),
        kwargs.get("window_pow"), kwargs.get("use_ecc_refine", True),
        kwargs.get("verbose", False))
    with _cache_lock:
        entry = _session_cache.get(key)
        if entry is None:
            session = Decoder(**kwargs)
            entry = (session, threading.RLock())
            _session_cache[key] = entry
            while len(_session_cache) > _SESSION_CACHE_MAX:
                _session_cache.popitem(last=False)
        else:
            _session_cache.move_to_end(key)
        return entry

REASON_TEXT = {
    "low_margin": "码字裕度不足",
    "low_group_margin": "单组裕度不足",
    "pad_mismatch": "补零不符",
    "crc_mismatch": "校验码不符",
    "low_cc": "配准 cc 不足",
    "low_logo_scale": "画面中 Logo 过小",
    "invalid_payload": "码字越界",
    "unreadable": "读不到图像",
    "sync_failed": "同步失败",
    "decode_failed": "解码失败",
    "rectify_failed": "矫正失败",
}


def reason_text(reason: str) -> str:
    """Chinese label for a machine-readable ``reason`` code (or comma list)."""
    if not reason or reason == "accepted":
        return "接受"
    parts = [REASON_TEXT.get(p, p) for p in reason.split(",") if p]
    return "，".join(parts) if parts else "拒绝"


@dataclass(frozen=True)
class DecodeResult:
    """One production decode.

    ``digits`` is set only when ``accepted`` is True. On reject it is None
    even if the research decoder proposed a number (see ``candidate_digits``).
    """

    accepted: bool
    digits: Optional[str]
    candidate_digits: Optional[str]
    carrier: str
    margin: float
    margin_mean: float
    margin_min: float
    cc: float
    logo_scale: Optional[float]
    mean_conf: float
    payload_valid: bool
    ok: bool
    reason: str
    status: str
    margin_threshold: float
    min_group_margin_threshold: float
    cc_threshold: Optional[float]
    raw: Dict = field(repr=False, compare=False)

    def public_dict(self) -> Dict:
        """JSON-safe fields. Omits the research decoder dict (``raw``).

        ``candidate_digits`` is included here for in-process diagnostics.
        The HTTP service strips it unless debug and a valid token are set.
        Success is ``accepted`` / ``status == "accepted"`` only.
        """
        return {
            "accepted": self.accepted,
            "status": self.status,
            "digits": self.digits,
            "candidate_digits": self.candidate_digits,
            "carrier": self.carrier,
            "margin": self.margin,
            "margin_mean": self.margin_mean,
            "margin_min": self.margin_min,
            "cc": self.cc,
            "logo_scale": self.logo_scale,
            "mean_conf": self.mean_conf,
            "payload_valid": self.payload_valid,
            "ok": self.ok,
            "reason": self.reason,
            "margin_threshold": self.margin_threshold,
            "min_group_margin_threshold": self.min_group_margin_threshold,
            "cc_threshold": self.cc_threshold,
        }


def _finite(value, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _optional_finite(value) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _candidate_digits(raw: Dict) -> Optional[str]:
    digits = raw.get("digits")
    if digits is None:
        return None
    text = str(digits).strip()
    return text or None


def _legal_payload(digits: Optional[str]) -> bool:
    return digits is not None and len(digits) == 11 and digits.isdigit()


def _resolve_margins(raw: Dict) -> Tuple[float, float]:
    """Return ``(margin_mean, margin_min)``.

    When the research decoder did not report a per-group minimum, the mean
    is reused so older raw dicts keep the mean gate's behavior. A real
    decode always fills ``margin_min`` (or ``conf``).
    """
    mean = _finite(raw.get("margin_mean", raw.get("margin", 0.0)), 0.0)
    if raw.get("margin_min") is not None:
        return mean, _finite(raw.get("margin_min"), 0.0)
    groups = raw.get("group_margins")
    if groups:
        vals = [_finite(v, 0.0) for v in groups]
        if vals:
            return mean, min(vals)
    conf = raw.get("conf")
    if conf is not None:
        try:
            import arrow_wm as A
            arr = np.asarray(conf, np.float64).ravel()
        except (TypeError, ValueError):
            arr = None
        if arr is not None and arr.size >= 7:
            ms = A.group_codeword_margins(arr)
            if ms.size:
                if "margin" not in raw and "margin_mean" not in raw:
                    mean = float(ms.mean())
                return mean, float(ms.min())
    return mean, mean


def _resolve_pad(raw: Dict) -> Optional[Tuple[int, int, int]]:
    """Group-0 pad bits, or None when this readout has nothing to check."""
    if "pad_bits" in raw and raw.get("pad_bits") is not None:
        try:
            bits = [int(b) for b in list(raw["pad_bits"])]
        except (TypeError, ValueError):
            return (-1, -1, -1)
        if len(bits) < 3:
            return (-1, -1, -1)
        return (bits[0] & 1, bits[1] & 1, bits[2] & 1)
    conf = raw.get("conf")
    if conf is None:
        return None
    import arrow_wm as A
    return A.group0_pad_bits(conf)


@dataclass(frozen=True)
class SuspectScore:
    """Blind check of one candidate number against measured soft bits.

    ``z`` is the absolute correlation in units of ``std(conf)``. ``gap`` is
    ``z`` minus the best one-digit neighbor. A neighbor can clear the
    absolute cut while losing the gap; that is not strong evidence.
    """

    suspect: str
    z: float
    score: float
    best_neighbor: Optional[str]
    neighbor_z: float
    gap: float
    verdict: str


def _suspect_verdict(z: float, gap: float) -> str:
    if z >= SUSPECT_Z_STRONG and gap >= SUSPECT_GAP_STRONG:
        return "包含（强证据）"
    if z >= SUSPECT_Z_MAYBE and gap >= SUSPECT_GAP_MAYBE:
        return "疑似"
    return "不包含"


def score_suspect(conf: Sequence[float], suspect: str) -> SuspectScore:
    """Score ``suspect`` against one-digit neighbors. Does not run ECC."""
    import arrow_wm as A
    suspect = str(suspect).strip()
    if len(suspect) != A.PAYLOAD_DIGITS or not suspect.isdigit():
        raise ValueError("suspect 必须是 11 位数字")
    soft = np.asarray(conf, np.float64).ravel()
    if soft.size != A.N_BITS:
        return SuspectScore(suspect, 0.0, 0.0, None, 0.0, 0.0, "不包含")
    sd = float(np.std(soft)) or 1e-9

    def z_of(digits: str) -> Tuple[float, float]:
        cw = np.asarray(A.encode_payload(digits), np.float64)
        sgn = cw * 2.0 - 1.0
        sc = float(np.dot(soft, sgn) / math.sqrt(float(sgn.size)))
        return sc / sd, sc

    z, sc = z_of(suspect)
    best_n: Optional[str] = None
    best_z = -1e30
    for i, ch in enumerate(suspect):
        for digit in "0123456789":
            if digit == ch:
                continue
            neighbor = suspect[:i] + digit + suspect[i + 1:]
            nz, _ = z_of(neighbor)
            if nz > best_z:
                best_z = nz
                best_n = neighbor
    gap = float(z - best_z)
    return SuspectScore(
        suspect=suspect,
        z=float(z),
        score=float(sc),
        best_neighbor=best_n,
        neighbor_z=float(best_z),
        gap=gap,
        verdict=_suspect_verdict(float(z), gap),
    )


def effective_cc_threshold(cc_threshold: Optional[float],
                           use_ecc_refine: bool) -> Optional[float]:
    """Registration gate for one session, or None to skip it.

    ``None`` always skips. With ECC refinement off the decoder does not
    compute a registration score (``cc`` stays 0), so the default cut of
    ``CC_THRESHOLD`` (0.85) is not applied. That is the ``--no-ecc`` rule.
    Any other numeric threshold is kept. ECC on keeps the given threshold,
    including the default 0.85.
    """
    if cc_threshold is not None:
        cc_threshold = float(cc_threshold)
    if not use_ecc_refine and (cc_threshold is None or cc_threshold == CC_THRESHOLD):
        return None
    return cc_threshold


def apply_gate(raw: Dict, *, carrier: str = DEFAULT_CARRIER,
               margin_threshold: float = MARGIN_THRESHOLD,
               min_group_margin: float = MIN_GROUP_MARGIN,
               cc_threshold: Optional[float] = CC_THRESHOLD) -> DecodeResult:
    """Apply the production accept/reject gate to a research decoder dict.

    ``cc_threshold=None`` skips the registration gate. ``Decoder`` passes
    None for the default 0.85 cut when ECC refinement is off, because that
    path does not compute ``cc``. The mean-margin gate, the per-group
    minimum, and the group-0 pad check still apply.
    """
    if carrier not in CARRIERS:
        raise ValueError(f"carrier must be one of {CARRIERS}, got {carrier!r}")
    margin_threshold = float(margin_threshold)
    min_group_margin = float(min_group_margin)
    if cc_threshold is not None:
        cc_threshold = float(cc_threshold)

    ok = bool(raw.get("ok"))
    candidate = _candidate_digits(raw)
    legal = _legal_payload(candidate)
    payload_flag = raw.get("payload_valid", legal)
    payload_valid = bool(payload_flag) and legal
    margin, margin_min = _resolve_margins(raw)
    cc = _finite(raw.get("cc", 0.0), 0.0)
    logo_scale = _optional_finite(raw.get("logo_scale"))
    mean_conf = _finite(raw.get("mean_conf", 0.0), 0.0)

    reasons = []
    if not ok:
        reasons.append(str(raw.get("reason") or "decode_failed"))
    else:
        if not payload_valid:
            reasons.append("invalid_payload")
        if cc_threshold is not None and cc < cc_threshold:
            reasons.append("low_cc")
        if logo_scale is not None and logo_scale < LOGO_SCALE_MIN:
            reasons.append("low_logo_scale")
        if margin < margin_threshold:
            reasons.append("low_margin")
        if margin_min < min_group_margin:
            reasons.append("low_group_margin")
        # v3 用 CRC-7 取代固定补零：crc_ok 存在时以它为准，不再看补零。
        crc_ok = raw.get("crc_ok")
        if crc_ok is not None:
            if not bool(crc_ok):
                reasons.append("crc_mismatch")
        else:
            pad = _resolve_pad(raw)
            if pad is not None and pad != PAD_BITS:
                reasons.append("pad_mismatch")

    accepted = not reasons
    return DecodeResult(
        accepted=accepted,
        digits=candidate if accepted else None,
        candidate_digits=candidate,
        carrier=carrier,
        margin=margin,
        margin_mean=margin,
        margin_min=margin_min,
        cc=cc,
        logo_scale=logo_scale,
        mean_conf=mean_conf,
        payload_valid=payload_valid,
        ok=ok,
        reason="" if accepted else ",".join(reasons),
        status="accepted" if accepted else "rejected",
        margin_threshold=margin_threshold,
        min_group_margin_threshold=min_group_margin,
        cc_threshold=cc_threshold,
        raw=raw,
    )


def _load_image(image: ImageLike) -> Optional[np.ndarray]:
    """Return a decoded array, or None when a path cannot be read.

    Empty arrays are treated as unreadable. A non-array, non-path value
    raises TypeError so a programming error is not reported as a reject.
    """
    if isinstance(image, np.ndarray):
        if image.size == 0:
            return None
        return image
    if isinstance(image, (str, os.PathLike)):
        import arrow_wm as A
        return A.imread_u(os.fspath(image))
    raise TypeError(
        "image must be a numpy array or a filesystem path, "
        f"got {type(image).__name__}")


def _unreadable(carrier: str, margin_threshold: float,
                min_group_margin: float,
                cc_threshold: Optional[float], detail: str) -> DecodeResult:
    raw = {"ok": False, "reason": "unreadable", "digits": None, "detail": detail}
    return apply_gate(raw, carrier=carrier, margin_threshold=margin_threshold,
                      min_group_margin=min_group_margin,
                      cc_threshold=cc_threshold)


def _normalize_inner_layout(value) -> Union[int, str]:
    """内孔版式：None / "auto" 自动，或显式 2 / 3。

    自动模式先解 v3（CRC-7），通过即定案；不过再解 v2，兼容已印出的旧版。
    """
    if value is None:
        return "auto"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "auto"):
            return "auto"
        value = text
    layout = int(value)
    if layout not in (2, 3):
        raise ValueError(f"inner layout 只能是 2、3 或 auto，收到 {value!r}")
    return layout


class _InnerAuto:
    """内孔自动版式：v3 优先，v2 兜底。"""

    def __init__(self, design, depth: float, verbose: bool,
                 use_ecc_refine: bool):
        import arrow_inner as I
        self._v3 = I.InnerDecoder(
            I.InnerCarrier(design, depth=depth, verbose=verbose,
                           layout_version=I.LAYOUT_V3),
            use_ecc_refine=use_ecc_refine)
        self._v2 = I.InnerDecoder(
            I.InnerCarrier(design, depth=depth, verbose=verbose,
                           layout_version=I.LAYOUT_V2),
            use_ecc_refine=use_ecc_refine)

    def _pair(self, gray: np.ndarray, return_debug: bool):
        """同一画面、同一次配准，分别按 v3 / v2 测量。

        v3 的 CRC 通过就不必再解 v2；否则才付第二遍测量的代价
        （配准仍然只做一次）。
        """
        r = self._v3._reg.rectify(gray)
        if not r.get("ok"):
            return self._v3._rectify_failure(r), None
        a = self._v3.decode_rectified(r, return_debug)
        if a.get("crc_ok"):
            return a, None
        return a, self._v2.decode_rectified(r, return_debug)

    def decode(self, image: np.ndarray, return_debug: bool = False):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        plates = self._v3._reg._plate_candidates(gray, k=3)
        plate_first = False
        if plates:
            frac = plates[0].size / float(max(gray.size, 1))
            plate_first = frac < 0.45 and max(gray.shape[:2]) >= 1600

        best = None
        best_key = None

        def consider(a, b, path: str, isolated: bool = False) -> None:
            nonlocal best, best_key
            for res in (a, b):
                if not res or not res.get("ok"):
                    continue
                valid = bool(res.get("crc_ok")) or res.get("pad_bits") == [0, 0, 0]
                key = (1 if valid else 0,
                       1 if float(res.get("cc", 0.0)) >= 0.85 else 0,
                       float(res.get("mean_conf", 0.0)))
                if best is None or key > best_key:
                    best = dict(res)
                    best_key = key
                    best["path"] = path
                    if isolated:
                        best["isolated"] = True

        def good_enough() -> bool:
            return best is not None and best_key[0] == 1 and best_key[1] == 1

        if not plate_first:
            consider(*self._pair(gray, return_debug), "full")
            if good_enough():
                return best
        for k, cand in enumerate(plates):
            if cand.size / float(max(gray.size, 1)) >= 0.85:
                continue
            consider(*self._pair(cand, return_debug), f"plate#{k}", isolated=True)
            if good_enough():
                return best
        if plate_first:
            consider(*self._pair(gray, return_debug), "full")
        if best is not None:
            return best
        return {"ok": False, "reason": "decode_failed", "digits": None,
                "bits": None, "mean_conf": 0.0, "margin": 0.0, "cc": 0.0,
                "layout_version": None, "crc_ok": None, "path": "none"}


class Decoder:
    """Reusable decode session for one carrier.

    Constructing the session loads the ARROW master and the SIFT template.
    Call :meth:`decode` once per image.

    ``use_ecc_refine=False`` skips the default cc gate (registration ``cc``
    stays 0). The margin gates, the pad check, and the other accept rules
    still apply.
    """

    def __init__(self, carrier: str = DEFAULT_CARRIER, *,
                 margin_threshold: float = MARGIN_THRESHOLD,
                 min_group_margin: float = MIN_GROUP_MARGIN,
                 cc_threshold: Optional[float] = CC_THRESHOLD,
                 design=None,
                 depth: Optional[float] = None,
                 layout_version: Optional[int] = None,
                 window_pow: Optional[float] = None,
                 use_ecc_refine: bool = True,
                 verbose: bool = False):
        if carrier not in CARRIERS:
            raise ValueError(f"carrier must be one of {CARRIERS}, got {carrier!r}")
        self.carrier = carrier
        self.margin_threshold = float(margin_threshold)
        self.min_group_margin = float(min_group_margin)
        self.use_ecc_refine = bool(use_ecc_refine)
        # ECC off leaves cc at 0. Skip the default 0.85 gate; keep an
        # explicit non-default threshold, and always keep the margin gate.
        self.cc_threshold = effective_cc_threshold(
            cc_threshold, self.use_ecc_refine)

        import arrow_wm as A
        if design is not None:
            self.design = design
        elif carrier == "edge":
            kw = {"verbose": verbose}
            if layout_version is not None:
                kw["layout_version"] = int(layout_version)
            if depth is not None:
                kw["depth"] = float(depth)
            elif layout_version == 1:
                # Old plates were printed at 1.8. v1@6 passes depth explicitly.
                kw["depth"] = A.V1_DEPTH
            if window_pow is not None:
                kw["window_pow"] = float(window_pow)
            self.design = _shared_design(**kw)
        else:
            self.design = _shared_design(verbose=verbose)
        if carrier == "edge":
            self._impl = A.WatermarkDecoder(
                self.design, use_ecc_refine=self.use_ecc_refine, verbose=verbose)
            self._inner = None
        else:
            import arrow_inner as I
            # Default radial depth stays at arrow_inner.DEPTH. This wrapper
            # does not retune inner robustness.
            used_depth = I.DEPTH if depth is None else float(depth)
            self.inner_layout = _normalize_inner_layout(layout_version)
            if self.inner_layout == "auto":
                self._inner = None
                self._impl = _InnerAuto(self.design, used_depth, verbose,
                                        self.use_ecc_refine)
            else:
                self._inner = I.InnerCarrier(
                    self.design, depth=used_depth, verbose=verbose,
                    layout_version=self.inner_layout)
                self._impl = I.InnerDecoder(
                    self._inner, use_ecc_refine=self.use_ecc_refine)

    def decode(self, image: ImageLike, *, return_debug: bool = False) -> DecodeResult:
        """Decode one image or file path.

        ``return_debug=True`` asks the research decoder for the rectified
        plate and homography (used by the verification figure). Those arrays
        live on ``result.raw`` and are not part of the stable field set.
        """
        loaded = _load_image(image)
        if loaded is None:
            detail = os.fspath(image) if isinstance(image, (str, os.PathLike)) else "empty array"
            return _unreadable(self.carrier, self.margin_threshold,
                               self.min_group_margin, self.cc_threshold, detail)
        raw = self._impl.decode(loaded, return_debug=return_debug)
        if not isinstance(raw, dict):
            raw = {"ok": False, "reason": "decode_failed", "digits": None}
        return apply_gate(
            raw, carrier=self.carrier,
            margin_threshold=self.margin_threshold,
            min_group_margin=self.min_group_margin,
            cc_threshold=self.cc_threshold)


def decode(image: ImageLike, carrier: str = DEFAULT_CARRIER, *,
           margin_threshold: float = MARGIN_THRESHOLD,
           min_group_margin: float = MIN_GROUP_MARGIN,
           cc_threshold: Optional[float] = CC_THRESHOLD,
           return_debug: bool = False,
           design=None,
           depth: Optional[float] = None,
           layout_version: Optional[int] = None,
           window_pow: Optional[float] = None,
           use_ecc_refine: bool = True,
           verbose: bool = False) -> DecodeResult:
    """Decode one image. Carrier defaults to ``"inner"``; pass ``"edge"`` for legacy plates.

    On reject, ``accepted`` is False, ``status`` is ``"rejected"``, and
    ``digits`` is None. Do not promote ``candidate_digits`` or ``ok``.

    ``use_ecc_refine=False`` matches ``--no-ecc``: the default cc ≥ 0.85
    check is skipped (``cc`` stays 0). Mean margin, minimum group margin,
    and the pad check still apply.
    """
    session, lock = _shared_session(
        carrier=carrier,
        margin_threshold=margin_threshold,
        min_group_margin=min_group_margin,
        cc_threshold=cc_threshold,
        design=design,
        depth=depth,
        layout_version=layout_version,
        window_pow=window_pow,
        use_ecc_refine=use_ecc_refine,
        verbose=verbose,
    )
    with lock:
        return session.decode(image, return_debug=return_debug)
