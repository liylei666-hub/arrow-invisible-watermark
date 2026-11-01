"""HTTP wrapper around ``arrow_decode`` for the mini-program and TI-ONE.

One :class:`arrow_decode.Decoder` is cached per carrier. Decode calls take a
process-wide lock so OpenCV's matcher is not used re-entrantly. The accept
gate itself is unchanged: this module only moves bytes in and JSON out.

Listen on ``0.0.0.0:$PORT`` (default **8501**, the TI-ONE REST port).
"""
from __future__ import annotations

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


def result_payload(result: AD.DecodeResult) -> dict:
    """``DecodeResult.public_dict()`` plus Chinese ``reason_text``."""
    body = result.public_dict()
    body["reason_text"] = AD.reason_text(result.reason)
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


def pick_carrier(query_value: Optional[str], form) -> str:
    """Query ``carrier`` wins over the form field. Default is ``edge``."""
    raw = query_value
    if raw is None or str(raw).strip() == "":
        item = form.get("carrier")
        if item is not None and not isinstance(item, UploadFile):
            raw = str(item)
    text = "" if raw is None else str(raw).strip().lower()
    if not text:
        return AD.DEFAULT_CARRIER
    if text not in AD.CARRIERS:
        raise HTTPException(
            status_code=400, detail="carrier must be 'edge' or 'inner'")
    return text


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/decode", response_model=None)
async def decode_endpoint(
    request: Request,
    carrier: Optional[str] = Query(default=None),
    x_decode_token: Optional[str] = Header(default=None, alias="X-Decode-Token"),
):
    check_token(x_decode_token)
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type.lower():
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
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    ensure_upload_size(len(data))
    gray = bytes_to_gray(data)
    result = await run_in_threadpool(decode_image, chosen, gray)
    return UTF8JSONResponse(result_payload(result))
