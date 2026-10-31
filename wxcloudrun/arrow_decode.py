"""Production decode API for the ARROW watermark.

The research decoders in ``arrow_wm`` / ``arrow_inner`` always emit a digit
string once sync succeeds. This module is the integrator-facing wrapper:

* default carrier is the edge watermark (``carrier="edge"``)
* ``carrier="inner"`` selects the inner-hole carrier; its algorithm is unchanged
* a hard gate sets ``accepted=False`` and ``digits=None`` unless the readout
  clears both the codeword-margin threshold and the registration ``cc`` gate

Rejected calls still expose ``candidate_digits`` for diagnostics. That field
is not a trusted phone number.

Batch callers should reuse :class:`Decoder`. :func:`decode` builds a new
session on every call (template + SIFT init).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np

# Production default. Inner stays selectable and is not tuned here.
DEFAULT_CARRIER = "edge"
CARRIERS = ("edge", "inner")

# README / batch_decode: true codes sit at or above ~2.0, noise and unmarked
# plates at or below ~1.4. The same 1.8 cut is used for edge and inner.
MARGIN_THRESHOLD = 1.8

# Both research decoders already refuse to *prefer* a path whose ECC
# correlation is below 0.85, because a bad crop can inflate mean |conf|.
# The production gate treats that same cut as a hard reject.
CC_THRESHOLD = 0.85

ImageLike = Union[np.ndarray, str, os.PathLike]

REASON_TEXT = {
    "low_margin": "码字裕度不足",
    "low_cc": "配准 cc 不足",
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
    cc: float
    logo_scale: Optional[float]
    mean_conf: float
    payload_valid: bool
    ok: bool
    reason: str
    margin_threshold: float
    cc_threshold: Optional[float]
    raw: Dict = field(repr=False, compare=False)

    def public_dict(self) -> Dict:
        """JSON-safe fields. Omits the research decoder dict (``raw``)."""
        return {
            "accepted": self.accepted,
            "digits": self.digits,
            "candidate_digits": self.candidate_digits,
            "carrier": self.carrier,
            "margin": self.margin,
            "cc": self.cc,
            "logo_scale": self.logo_scale,
            "mean_conf": self.mean_conf,
            "payload_valid": self.payload_valid,
            "ok": self.ok,
            "reason": self.reason,
            "margin_threshold": self.margin_threshold,
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


def apply_gate(raw: Dict, *, carrier: str = DEFAULT_CARRIER,
               margin_threshold: float = MARGIN_THRESHOLD,
               cc_threshold: Optional[float] = CC_THRESHOLD) -> DecodeResult:
    """Apply the production accept/reject gate to a research decoder dict.

    ``cc_threshold=None`` skips the registration gate. That is only for the
    research ``--no-ecc`` switch, which does not compute ``cc``.
    """
    if carrier not in CARRIERS:
        raise ValueError(f"carrier must be one of {CARRIERS}, got {carrier!r}")
    margin_threshold = float(margin_threshold)
    if cc_threshold is not None:
        cc_threshold = float(cc_threshold)

    ok = bool(raw.get("ok"))
    candidate = _candidate_digits(raw)
    legal = _legal_payload(candidate)
    payload_flag = raw.get("payload_valid", legal)
    payload_valid = bool(payload_flag) and legal
    margin = _finite(raw.get("margin", 0.0), 0.0)
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
        if margin < margin_threshold:
            reasons.append("low_margin")

    accepted = not reasons
    return DecodeResult(
        accepted=accepted,
        digits=candidate if accepted else None,
        candidate_digits=candidate,
        carrier=carrier,
        margin=margin,
        cc=cc,
        logo_scale=logo_scale,
        mean_conf=mean_conf,
        payload_valid=payload_valid,
        ok=ok,
        reason="" if accepted else ",".join(reasons),
        margin_threshold=margin_threshold,
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
                cc_threshold: Optional[float], detail: str) -> DecodeResult:
    raw = {"ok": False, "reason": "unreadable", "digits": None, "detail": detail}
    return apply_gate(raw, carrier=carrier, margin_threshold=margin_threshold,
                      cc_threshold=cc_threshold)


class Decoder:
    """Reusable decode session for one carrier.

    Constructing the session loads the ARROW master and the SIFT template.
    Call :meth:`decode` once per image.
    """

    def __init__(self, carrier: str = DEFAULT_CARRIER, *,
                 margin_threshold: float = MARGIN_THRESHOLD,
                 cc_threshold: Optional[float] = CC_THRESHOLD,
                 design=None,
                 depth: Optional[float] = None,
                 use_ecc_refine: bool = True,
                 verbose: bool = False):
        if carrier not in CARRIERS:
            raise ValueError(f"carrier must be one of {CARRIERS}, got {carrier!r}")
        self.carrier = carrier
        self.margin_threshold = float(margin_threshold)
        self.cc_threshold = None if cc_threshold is None else float(cc_threshold)
        self.use_ecc_refine = bool(use_ecc_refine)

        import arrow_wm as A
        self.design = design if design is not None else A.ArrowDesign(verbose=verbose)
        if carrier == "edge":
            self._impl = A.WatermarkDecoder(
                self.design, use_ecc_refine=self.use_ecc_refine, verbose=verbose)
            self._inner = None
        else:
            import arrow_inner as I
            # Default radial depth stays at arrow_inner.DEPTH. This wrapper
            # does not retune inner robustness.
            used_depth = I.DEPTH if depth is None else float(depth)
            self._inner = I.InnerCarrier(self.design, depth=used_depth,
                                         verbose=verbose)
            self._impl = I.InnerDecoder(self._inner,
                                        use_ecc_refine=self.use_ecc_refine)

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
                               self.cc_threshold, detail)
        raw = self._impl.decode(loaded, return_debug=return_debug)
        if not isinstance(raw, dict):
            raw = {"ok": False, "reason": "decode_failed", "digits": None}
        return apply_gate(
            raw, carrier=self.carrier,
            margin_threshold=self.margin_threshold,
            cc_threshold=self.cc_threshold)


def decode(image: ImageLike, carrier: str = DEFAULT_CARRIER, *,
           margin_threshold: float = MARGIN_THRESHOLD,
           cc_threshold: Optional[float] = CC_THRESHOLD,
           return_debug: bool = False,
           design=None,
           depth: Optional[float] = None,
           use_ecc_refine: bool = True,
           verbose: bool = False) -> DecodeResult:
    """Decode one image. Carrier defaults to ``"edge"``; pass ``"inner"`` to opt in.

    On reject, ``accepted`` is False and ``digits`` is None. Do not promote
    ``candidate_digits`` to a phone number.
    """
    session = Decoder(
        carrier=carrier,
        margin_threshold=margin_threshold,
        cc_threshold=cc_threshold,
        design=design,
        depth=depth,
        use_ecc_refine=use_ecc_refine,
        verbose=verbose,
    )
    return session.decode(image, return_debug=return_debug)
