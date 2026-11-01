"""HTTP wrapper around ``arrow_decode`` for the mini-program and TI-ONE.

One :class:`arrow_decode.Decoder` is cached per carrier. Decode calls take a
process-wide lock so OpenCV's matcher is not used re-entrantly. The accept gate lives in ``arrow_decode``. This module maps it to JSON:
``status`` is ``accepted``, ``rejected``, or ``error``. ``ok`` is not a
success bit. ``candidate_digits`` is omitted unless ``debug`` is set and
``DECODE_TOKEN`` is configured (the request must already have matched it).

Listen on ``0.0.0.0:$PORT`` (default **8501**). ``POST /decode`` keeps
multipart, and also accepts JSON ``image_base64`` or a raw image body so
``wx.cloud.callContainer`` can send a photo without building a form.
"""
from __future__ import annotations

import base64
import json
import os
import re
import secrets
import sys
import threading
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

# Repo root holds arrow_decode.py / arrow_wm.py / arrow_inner.py. Insert it so
# `uvicorn app:app` from this directory and `uvicorn services.decode_api.app:app`
# from the repo root both import the library.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import arrow_decode as AD

REPO_ROOT = _ROOT

# Preferred multipart field, then common client aliases. Any other file part
# is still accepted so a renamed form field does not 400.
IMAGE_FIELDS = ("image", "file", "img", "photo", "picture")

# Phone photos are a few megabytes. Caps are abuse limits, not decode knobs.
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000

# Base64 expands by 4/3. Cap the text before decoding so a 1GB instance is
# not asked to materialize an unbounded body. Raw image bodies use
# MAX_IMAGE_BYTES directly.
_BASE64_CHAR_CAP = MAX_IMAGE_BYTES * 2

# callContainer can hand the photo over as an ArrayBuffer. These media types
# are the body itself (carrier stays on the query string).
_RAW_MEDIA_TYPES = frozenset({
    "application/octet-stream",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
})

_MS_SEGMENT = re.compile(r"^ms-[A-Za-z0-9_-]+$")

# RLock so decode_image can call get_decoder while already holding the lock.
# The lock stays held for the whole decode: OpenCV's matcher is not re-entrant.
_lock = threading.RLock()
_sessions: dict[str, AD.Decoder] = {}


def strip_ms_prefix(path: str) -> str:
    """Drop leading ``/ms-<id>`` path segments.

    The public gateway URL is ``https://<host>/ms-<id>/decode``. TI-ONE usually
    strips that prefix before the container. A few routes (some CLB examples)
    forward it once or twice. The app's own routes stay ``/decode`` and
    ``/health``.
    """
    rest = path or "/"
    while True:
        parts = rest.split("/", 2)
        if len(parts) < 2 or not _MS_SEGMENT.match(parts[1] or ""):
            return rest or "/"
        if len(parts) == 2 or not parts[2]:
            return "/"
        rest = "/" + parts[2]


class GatewayPrefixMiddleware:
    """Rewrite ``/ms-<id>/...`` onto the app path. Does not read the body."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            path = scope.get("path") or "/"
            stripped = strip_ms_prefix(path)
            if stripped != path:
                scope = dict(scope)
                scope["path"] = stripped
                scope["raw_path"] = stripped.encode("utf-8")
        await self.app(scope, receive, send)


def _warmup_carriers() -> list[str]:
    """Which sessions to build before serving.

    ``DECODE_WARMUP=1`` (default) loads edge and inner.
    ``DECODE_WARMUP=edge`` loads only the production default.
    ``DECODE_WARMUP=0`` skips warmup (tests).
    """
    flag = os.environ.get("DECODE_WARMUP", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return []
    if flag in ("edge", "default"):
        return [AD.DEFAULT_CARRIER]
    return [AD.DEFAULT_CARRIER, "inner"]


def get_decoder(carrier: str) -> AD.Decoder:
    """Return the cached session for ``carrier``, building it on first use."""
    if carrier not in AD.CARRIERS:
        raise ValueError(f"carrier must be one of {AD.CARRIERS}, got {carrier!r}")
    with _lock:
        session = _sessions.get(carrier)
        if session is None:
            session = AD.Decoder(carrier=carrier)
            _sessions[carrier] = session
        return session


def decode_image(carrier: str, gray: np.ndarray) -> AD.DecodeResult:
    """Run one production decode under the process-wide lock."""
    with _lock:
        return get_decoder(carrier).decode(gray)


def candidate_debug_allowed(debug: bool) -> bool:
    """Candidate digits leave the process only for an authenticated debug call.

    ``DECODE_TOKEN`` unset means there is no valid token, so debug cannot
    reveal a candidate. ``check_token`` has already rejected a mismatch.
    """
    if not debug:
        return False
    return bool(os.environ.get("DECODE_TOKEN", "").strip())


def result_payload(result: AD.DecodeResult, *,
                   include_candidate: bool = False) -> dict:
    """``DecodeResult.public_dict()`` plus Chinese ``reason_text``.

    ``candidate_digits`` is removed unless ``include_candidate``. Clients
    must treat only ``accepted === true`` (equivalently ``status ==
    "accepted"``) as a phone number. ``ok`` is the research decoder's sync
    flag and is true on many rejects.
    """
    body = result.public_dict()
    body["reason_text"] = AD.reason_text(result.reason)
    if not include_candidate:
        body.pop("candidate_digits", None)
    return body


class UTF8JSONResponse(JSONResponse):
    """UTF-8 JSON so ``reason_text`` is not ``\\u`` escaped. Rejects NaN."""

    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")


def check_token(provided: Optional[str]) -> None:
    """Require ``X-Decode-Token`` only when ``DECODE_TOKEN`` is set.

    Unset or blank means no auth (local dev). ``GET /health`` does not call
    this, so a platform probe does not need the secret.
    """
    expected = os.environ.get("DECODE_TOKEN", "").strip()
    if not expected:
        return
    got = "" if provided is None else provided.strip()
    if not secrets.compare_digest(got, expected):
        raise HTTPException(status_code=401, detail="unauthorized")


def ensure_upload_size(nbytes: int) -> None:
    if nbytes > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image too large")


def bytes_to_gray(data: bytes) -> np.ndarray:
    """Grayscale image, the same conversion as ``arrow_wm.imread_u``.

    Undecodable bytes become an empty array. ``Decoder`` then returns
    ``reason=unreadable`` instead of raising.
    """
    if not data:
        return np.empty((0,), np.uint8)
    buf = np.frombuffer(data, dtype=np.uint8)
    gray = cv2.imdecode(np.ascontiguousarray(buf), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.size == 0:
        return np.empty((0,), np.uint8)
    if int(gray.size) > MAX_PIXELS:
        raise HTTPException(status_code=413, detail="image too large")
    return gray


def media_type(content_type: str) -> str:
    """``Content-Type`` without parameters, lowercased."""
    return (content_type or "").split(";", 1)[0].strip().lower()


def carrier_from_value(raw: Optional[str]) -> str:
    """Normalize a carrier string. Blank means the production default."""
    text = "" if raw is None else str(raw).strip().lower()
    if not text:
        return AD.DEFAULT_CARRIER
    if text not in AD.CARRIERS:
        raise HTTPException(
            status_code=400, detail="carrier must be 'edge' or 'inner'")
    return text


def pick_carrier(query_value: Optional[str], form) -> str:
    """Query ``carrier`` wins over the form field. Default is ``edge``."""
    raw = query_value
    if raw is None or str(raw).strip() == "":
        item = form.get("carrier")
        if item is not None and not isinstance(item, UploadFile):
            raw = str(item)
    return carrier_from_value(None if raw is None else str(raw))


def decode_base64_image(text: str) -> bytes:
    """Decode a base64 image. A leading ``data:*;base64,`` prefix is stripped."""
    raw = text.strip()
    if raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[-1]
    raw = "".join(raw.split())
    if len(raw) > _BASE64_CHAR_CAP:
        raise HTTPException(status_code=413, detail="image too large")
    pad = (-len(raw)) % 4
    if pad:
        raw += "=" * pad
    try:
        data = base64.b64decode(raw, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image_base64")
    return data


def declared_length(request: Request) -> Optional[int]:
    raw = request.headers.get("content-length")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


async def finish_decode(carrier: str, data: bytes, *,
                        include_candidate: bool = False):
    """Shared tail for every ``POST /decode`` body shape."""
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    ensure_upload_size(len(data))
    gray = bytes_to_gray(data)
    try:
        result = await run_in_threadpool(decode_image, carrier, gray)
    except HTTPException:
        raise
    except Exception:
        return UTF8JSONResponse(
            {"status": "error", "detail": "decode error"}, status_code=500)
    return UTF8JSONResponse(
        result_payload(result, include_candidate=include_candidate))


async def decode_json_body(request: Request, query_carrier: Optional[str], *,
                           include_candidate: bool = False):
    """``{"image_base64": "...", "carrier": "edge"}``. Query carrier wins."""
    length = declared_length(request)
    if length is not None and length > _BASE64_CHAR_CAP + 4096:
        raise HTTPException(status_code=413, detail="image too large")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="expected JSON object")
    raw_carrier = query_carrier
    if raw_carrier is None or str(raw_carrier).strip() == "":
        if payload.get("carrier") is not None:
            raw_carrier = payload.get("carrier")
    chosen = carrier_from_value(None if raw_carrier is None else str(raw_carrier))
    image = payload.get("image_base64")
    if not isinstance(image, str):
        image = payload.get("image")
    if not isinstance(image, str) or not image.strip():
        raise HTTPException(status_code=400, detail="missing image_base64")
    return await finish_decode(
        chosen, decode_base64_image(image), include_candidate=include_candidate)


async def decode_raw_body(request: Request, query_carrier: Optional[str], *,
                          include_candidate: bool = False):
    """Raw image bytes. ``carrier`` is the query param (default ``edge``)."""
    length = declared_length(request)
    if length is not None and length > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image too large")
    data = await request.body()
    return await finish_decode(
        carrier_from_value(query_carrier), data,
        include_candidate=include_candidate)


def select_upload(form) -> Optional[UploadFile]:
    for name in IMAGE_FIELDS:
        item = form.get(name)
        if isinstance(item, UploadFile):
            return item
    for _key, item in form.multi_items():
        if isinstance(item, UploadFile):
            return item
    return None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    carriers = _warmup_carriers()
    for carrier in carriers:
        print(f"[decode_api] loading {carrier} decoder", flush=True)
        get_decoder(carrier)
    if carriers:
        print("[decode_api] ready", flush=True)
    yield


app = FastAPI(title="ARROW decode", version="1.0.0", lifespan=lifespan)
app.add_middleware(GatewayPrefixMiddleware)


@app.exception_handler(HTTPException)
async def http_error(_request: Request, exc: HTTPException):
    """Bad requests and auth failures are ``status=error``, not a decode."""
    return UTF8JSONResponse(
        {"status": "error", "detail": exc.detail},
        status_code=exc.status_code,
        headers=exc.headers,
    )


@app.get("/health")
def health():
    # Liveness only. This "ok" is not a decode success.
    return {"status": "ok"}


@app.post("/decode", response_model=None)
async def decode_endpoint(
    request: Request,
    carrier: Optional[str] = Query(default=None),
    debug: bool = Query(default=False),
    x_decode_token: Optional[str] = Header(default=None, alias="X-Decode-Token"),
):
    check_token(x_decode_token)
    include_candidate = candidate_debug_allowed(debug)
    kind = media_type(request.headers.get("content-type", ""))
    if kind == "application/json":
        return await decode_json_body(
            request, carrier, include_candidate=include_candidate)
    if kind in _RAW_MEDIA_TYPES:
        return await decode_raw_body(
            request, carrier, include_candidate=include_candidate)
    if kind != "multipart/form-data":
        raise HTTPException(status_code=400, detail="expected multipart/form-data")
    try:
        form = await request.form()
    except Exception:
        raise HTTPException(status_code=400, detail="expected multipart/form-data")
    try:
        chosen = pick_carrier(carrier, form)
        upload = select_upload(form)
        if upload is None:
            raise HTTPException(
                status_code=400,
                detail="missing image; send multipart field 'image'",
            )
        data = await upload.read()
    finally:
        await form.close()
    return await finish_decode(
        chosen, data, include_candidate=include_candidate)
