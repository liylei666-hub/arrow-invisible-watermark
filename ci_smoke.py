"""Fast production-decode smoke. Used by GitHub Actions and local checks.

Covers the accept/reject gate without a robustness bench, plus one clean
embed/decode roundtrip on the edge carrier and one on the inner carrier
(the production default). Inner extreme-tier rates are intentionally not measured.
"""
from __future__ import annotations

import base64
import inspect
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import arrow_decode as AD


class GateTests(unittest.TestCase):
    def test_default_carrier_is_inner(self):
        self.assertEqual(AD.DEFAULT_CARRIER, "inner")
        self.assertEqual(
            inspect.signature(AD.decode).parameters["carrier"].default, "inner")
        self.assertEqual(
            inspect.signature(AD.Decoder).parameters["carrier"].default, "inner")

    def test_unknown_carrier_rejected(self):
        with self.assertRaises(ValueError):
            AD.Decoder(carrier="both")

    def test_low_margin_is_not_a_phone_number(self):
        raw = {
            "ok": True, "digits": "13800138000", "payload_valid": True,
            "margin": 1.2, "cc": 0.97, "mean_conf": 0.04, "logo_scale": 1.0,
        }
        result = AD.apply_gate(raw, carrier="edge")
        self.assertFalse(result.accepted)
        self.assertEqual(result.status, "rejected")
        self.assertIsNone(result.digits)
        self.assertEqual(result.candidate_digits, "13800138000")
        self.assertIn("low_margin", result.reason)
        self.assertIsNone(result.public_dict()["digits"])

    def test_low_cc_is_not_a_phone_number(self):
        raw = {
            "ok": True, "digits": "13800138000", "payload_valid": True,
            "margin": 4.0, "cc": 0.2, "mean_conf": 0.08, "logo_scale": 0.3,
        }
        result = AD.apply_gate(raw, carrier="inner")
        self.assertFalse(result.accepted)
        self.assertIsNone(result.digits)
        self.assertEqual(result.carrier, "inner")
        self.assertIn("low_cc", result.reason)
        self.assertEqual(result.logo_scale, 0.3)

    def test_invalid_payload_is_not_padded_into_a_number(self):
        raw = {
            "ok": True, "digits": None, "payload_valid": False,
            "margin": 5.0, "cc": 0.99, "mean_conf": 0.05, "logo_scale": 1.0,
        }
        result = AD.apply_gate(raw)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.digits)
        self.assertIn("invalid_payload", result.reason)

    def test_sync_failure_stays_rejected(self):
        raw = {"ok": False, "reason": "sync_failed", "digits": None}
        result = AD.apply_gate(raw)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.digits)
        self.assertEqual(result.reason, "sync_failed")

    def test_ecc_off_skips_default_cc_gate_only(self):
        # ECC off does not produce a registration score. The default 0.85
        # cut must not reject a legal high-margin read, and must not let a
        # low-margin read through.
        high = {
            "ok": True, "digits": "13800138000", "payload_valid": True,
            "margin": 4.0, "cc": 0.0, "mean_conf": 0.05, "logo_scale": 1.0,
        }
        skipped = AD.effective_cc_threshold(AD.CC_THRESHOLD, use_ecc_refine=False)
        self.assertIsNone(skipped)
        accepted = AD.apply_gate(high, cc_threshold=skipped)
        self.assertTrue(accepted.accepted, accepted.public_dict())
        self.assertEqual(accepted.digits, "13800138000")
        self.assertEqual(accepted.cc, 0.0)
        self.assertLess(accepted.cc, AD.CC_THRESHOLD)
        self.assertIsNone(accepted.cc_threshold)
        self.assertGreaterEqual(accepted.margin, AD.MARGIN_THRESHOLD)
        self.assertNotIn("low_cc", accepted.reason)

        rejected = AD.apply_gate(dict(high, margin=1.2), cc_threshold=skipped)
        self.assertFalse(rejected.accepted, rejected.public_dict())
        self.assertIsNone(rejected.digits)
        self.assertEqual(rejected.candidate_digits, "13800138000")
        self.assertIn("low_margin", rejected.reason)
        self.assertNotIn("low_cc", rejected.reason)
        self.assertLess(rejected.margin, AD.MARGIN_THRESHOLD)

        # Explicit non-default threshold still applies with ECC off.
        # None always skips, including when ECC stays on.
        self.assertEqual(
            AD.effective_cc_threshold(0.9, use_ecc_refine=False), 0.9)
        self.assertIsNone(
            AD.effective_cc_threshold(None, use_ecc_refine=False))
        self.assertIsNone(
            AD.effective_cc_threshold(None, use_ecc_refine=True))
        self.assertEqual(
            AD.effective_cc_threshold(AD.CC_THRESHOLD, use_ecc_refine=True),
            AD.CC_THRESHOLD)
        still_gated = AD.apply_gate(high, cc_threshold=0.9)
        self.assertFalse(still_gated.accepted)
        self.assertIn("low_cc", still_gated.reason)

    def test_tiny_logo_is_rejected_without_becoming_a_number(self):
        raw = {
            "ok": True, "digits": "61960679562", "payload_valid": True,
            "margin": 4.2, "margin_min": 3.0, "cc": 0.97,
            "mean_conf": 0.01, "logo_scale": 0.077,
        }
        result = AD.apply_gate(raw, carrier="edge")
        self.assertFalse(result.accepted, result.public_dict())
        self.assertIsNone(result.digits)
        self.assertEqual(result.candidate_digits, "61960679562")
        self.assertIn("low_logo_scale", result.reason)
        self.assertGreaterEqual(result.margin, AD.MARGIN_THRESHOLD)
        self.assertLess(result.logo_scale, AD.LOGO_SCALE_MIN)

    def test_logo_scale_cut_accepts_0131_and_rejects_0077(self):
        """Second real inner photo of 17701234567 sets the accepted edge.

        It reads the correct number at logo_scale ≈ 0.131 with margin 2.82,
        margin_min 0.66 and cc 0.976 — every signal gate green — and was
        rejected only by the old 0.15 scale cut. The cut is 0.10; the
        documented 0.077 tiny frame stays under it and is still rejected.
        """
        base = {
            "ok": True, "digits": "17701234567", "payload_valid": True,
            "margin": 2.82, "margin_min": 0.66, "cc": 0.976,
            "mean_conf": 0.02, "pad_bits": [0, 0, 0],
        }
        kept = AD.apply_gate(dict(base, logo_scale=0.131), carrier="inner")
        self.assertTrue(kept.accepted, kept.public_dict())
        self.assertEqual(kept.digits, "17701234567")
        # 门是包含下界：恰好 0.10 接受，0.099 拒绝。
        edge = AD.apply_gate(dict(base, logo_scale=AD.LOGO_SCALE_MIN),
                             carrier="inner")
        self.assertTrue(edge.accepted, edge.public_dict())
        under = AD.apply_gate(dict(base, logo_scale=0.077), carrier="inner")
        self.assertFalse(under.accepted, under.public_dict())
        self.assertIn("low_logo_scale", under.reason)
        self.assertIsNone(under.digits)

    def test_gate_photo_inner_fields_are_accepted(self):
        # Real inner photo 17701234567: correct candidate, previously rejected
        # only by logo_scale≈0.1595 and margin_min≈0.604. Those cuts are now
        # 0.10 and 0.55. The 0.077 tiny-logo case and the ~0.325 near-tie stay
        # rejected by the tests above and below.
        raw = {
            "ok": True, "digits": "17701234567", "payload_valid": True,
            "margin": 2.80, "margin_min": 0.604, "cc": 0.975,
            "mean_conf": 1.75, "logo_scale": 0.1595, "pad_bits": [0, 0, 0],
        }
        result = AD.apply_gate(raw, carrier="inner")
        self.assertTrue(result.accepted, result.public_dict())
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.digits, "17701234567")
        self.assertEqual(result.candidate_digits, "17701234567")
        self.assertGreaterEqual(result.logo_scale, AD.LOGO_SCALE_MIN)
        self.assertGreaterEqual(result.margin_min, AD.MIN_GROUP_MARGIN)

    def test_clear_read_is_accepted(self):
        raw = {
            "ok": True, "digits": "13800138000", "payload_valid": True,
            "margin": 4.2, "cc": 0.96, "mean_conf": 0.05, "logo_scale": 1.0,
        }
        result = AD.apply_gate(raw)
        self.assertTrue(result.accepted)
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.digits, "13800138000")
        self.assertEqual(result.reason, "")
        self.assertGreaterEqual(result.margin, AD.MARGIN_THRESHOLD)
        self.assertEqual(result.margin, result.margin_mean)
        self.assertGreaterEqual(result.margin_min, AD.MIN_GROUP_MARGIN)
        self.assertGreaterEqual(result.cc, AD.CC_THRESHOLD)

    def test_high_mean_low_group_margin_is_rejected(self):
        # Nine confident groups keep the mean above 1.8. One near-tie group
        # must still reject. The averaged margin is not a digit-correct gate.
        import arrow_wm as A
        cb = A.hamming_codebook()
        soft = np.zeros(A.N_BITS, np.float64)
        for g in range(10):
            sgn = cb[0].astype(np.float64) * 2.0 - 1.0
            amp = 0.04 if g == 4 else 2.5
            soft[g * 7:(g + 1) * 7] = sgn * amp
        ms = A.group_codeword_margins(soft)
        self.assertGreaterEqual(float(ms.mean()), AD.MARGIN_THRESHOLD)
        self.assertLess(float(ms.min()), AD.MIN_GROUP_MARGIN)
        digits, _ = A.decode_payload_soft(soft)
        raw = {
            "ok": True, "digits": digits, "payload_valid": True,
            "margin": float(ms.mean()), "margin_min": float(ms.min()),
            "cc": 0.99, "mean_conf": 0.05, "logo_scale": 1.0,
            "pad_bits": [0, 0, 0],
        }
        result = AD.apply_gate(raw)
        self.assertFalse(result.accepted, result.public_dict())
        self.assertEqual(result.status, "rejected")
        self.assertIsNone(result.digits)
        self.assertIn("low_group_margin", result.reason)
        self.assertNotIn("low_margin", result.reason)
        self.assertNotIn("pad_mismatch", result.reason)
        self.assertGreaterEqual(result.margin_mean, AD.MARGIN_THRESHOLD)
        self.assertLess(result.margin_min, result.min_group_margin_threshold)

    def test_pad_mismatch_is_rejected(self):
        # Group 0's first three data bits are the encoder's pad. They must
        # be 000 even when every margin gate is comfortable.
        import arrow_wm as A
        data = [1, 0, 0, 0] + [0] * 36
        code = np.array(A.hamming74_encode(data), np.float64)
        soft = (code * 2.0 - 1.0) * 2.0
        pad = A.group0_pad_bits(soft)
        self.assertEqual(pad, (1, 0, 0))
        ms = A.group_codeword_margins(soft)
        digits, _ = A.decode_payload_soft(soft)
        self.assertEqual(digits, "00000000000")
        raw = {
            "ok": True, "digits": digits, "payload_valid": True,
            "margin": float(ms.mean()), "margin_min": float(ms.min()),
            "cc": 0.99, "mean_conf": 0.05, "logo_scale": 1.0,
            "pad_bits": list(pad),
        }
        result = AD.apply_gate(raw)
        self.assertFalse(result.accepted, result.public_dict())
        self.assertEqual(result.status, "rejected")
        self.assertIsNone(result.digits)
        self.assertEqual(result.candidate_digits, "00000000000")
        self.assertIn("pad_mismatch", result.reason)
        self.assertGreaterEqual(result.margin_mean, AD.MARGIN_THRESHOLD)
        self.assertGreaterEqual(result.margin_min, AD.MIN_GROUP_MARGIN)

        ok_pad = AD.apply_gate(dict(raw, pad_bits=[0, 0, 0]))
        self.assertTrue(ok_pad.accepted, ok_pad.public_dict())
        self.assertEqual(ok_pad.status, "accepted")
        self.assertNotIn("pad_mismatch", ok_pad.reason)

    def test_suspect_gap_blocks_a_near_number(self):
        # Ideal soft info aligned to the true number. A one-digit neighbor
        # at Hamming distance 3 still has z above 4.5, so absolute z would
        # call it strong evidence. The gap is negative, so the verdict is not.
        import arrow_wm as A
        true = "13800138000"
        code = np.array(A.encode_payload(true), np.float64)
        conf = (code * 2.0 - 1.0) * 0.08
        own = AD.score_suspect(conf, true)
        self.assertGreaterEqual(own.z, AD.SUSPECT_Z_STRONG)
        self.assertGreaterEqual(own.gap, AD.SUSPECT_GAP_STRONG)
        self.assertEqual(own.verdict, "包含（强证据）")

        found = False
        for i, ch in enumerate(true):
            for digit in "0123456789":
                if digit == ch:
                    continue
                suspect = true[:i] + digit + true[i + 1:]
                scored = AD.score_suspect(conf, suspect)
                if scored.z < AD.SUSPECT_Z_STRONG:
                    continue
                found = True
                self.assertLess(scored.gap, 0.0, scored)
                self.assertNotEqual(scored.verdict, "包含（强证据）")
                self.assertNotEqual(scored.verdict, "疑似")
                break
            if found:
                break
        self.assertTrue(found, "expected a one-digit neighbor whose absolute z still clears 4.5")


class RoundtripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.edge = AD.Decoder(carrier="edge")
        cls.digits = "13800138000"

    def test_edge_clean_roundtrip_accepts(self):
        import arrow_wm as A
        plate = A.WatermarkEmbedder(self.edge.design, verbose=False).embed(self.digits)["plate"]
        result = self.edge.decode(plate)
        self.assertTrue(result.accepted, result.public_dict())
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.digits, self.digits)
        self.assertEqual(result.carrier, "edge")
        self.assertGreaterEqual(result.margin, AD.MARGIN_THRESHOLD)
        self.assertGreaterEqual(result.margin_min, AD.MIN_GROUP_MARGIN)
        self.assertEqual(result.raw.get("pad_bits"), [0, 0, 0])
        self.assertGreaterEqual(result.cc, AD.CC_THRESHOLD)
        self.assertEqual(result.cc_threshold, AD.CC_THRESHOLD)
        self.assertTrue(self.edge.use_ecc_refine)
        self.assertIsNotNone(result.logo_scale)

        # Same plate via the function entry and via a file path.
        via_fn = AD.decode(plate, carrier="edge", design=self.edge.design)
        self.assertTrue(via_fn.accepted, via_fn.public_dict())
        self.assertEqual(via_fn.digits, self.digits)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            import arrow_wm as A
            self.assertTrue(A.imwrite_u(path, plate))
            via_path = self.edge.decode(path)
            self.assertTrue(via_path.accepted, via_path.public_dict())
            self.assertEqual(via_path.digits, self.digits)
        finally:
            os.remove(path)

    def test_unmarked_plate_is_rejected(self):
        result = self.edge.decode(self.edge.design.template)
        self.assertFalse(result.accepted, result.public_dict())
        self.assertIsNone(result.digits)
        self.assertTrue(result.ok, result.public_dict())
        self.assertLess(result.margin, AD.MARGIN_THRESHOLD, result.public_dict())
        self.assertTrue(result.candidate_digits is None or len(result.candidate_digits) == 11)

    def test_inner_clean_roundtrip_accepts(self):
        import arrow_inner as I
        session = AD.Decoder(carrier="inner", design=self.edge.design)
        plate = I.InnerCarrier(self.edge.design).embed(self.digits)["plate"]
        result = session.decode(plate)
        self.assertEqual(result.carrier, "inner")
        self.assertTrue(result.accepted, result.public_dict())
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.digits, self.digits)
        self.assertGreaterEqual(result.margin, AD.MARGIN_THRESHOLD)
        self.assertGreaterEqual(result.margin_min, AD.MIN_GROUP_MARGIN)
        # 默认新印版是 v3：CRC-7 取代固定补零。
        self.assertEqual(result.raw.get("layout_version"), I.LAYOUT_V3)
        self.assertTrue(result.raw.get("crc_ok"))
        self.assertIsNone(result.raw.get("pad_bits"))
        self.assertGreaterEqual(result.cc, AD.CC_THRESHOLD)

    def test_no_ecc_clean_accepts_unmarked_rejects_on_margin(self):
        # Equivalent to arrow_wm.py --no-ecc: do not pass cc_threshold.
        # A clean plate must accept with cc left at 0; an unmarked plate
        # must still fail the margin gate and must not be rejected for cc.
        session = AD.Decoder(
            carrier="edge", design=self.edge.design, use_ecc_refine=False)
        self.assertFalse(session.use_ecc_refine)
        self.assertIsNone(session.cc_threshold)
        self.assertEqual(session.margin_threshold, AD.MARGIN_THRESHOLD)

        import arrow_wm as A
        plate = A.WatermarkEmbedder(
            self.edge.design, verbose=False).embed(self.digits)["plate"]
        result = session.decode(plate)
        self.assertTrue(result.accepted, result.public_dict())
        self.assertEqual(result.digits, self.digits)
        self.assertEqual(result.carrier, "edge")
        self.assertLess(result.cc, AD.CC_THRESHOLD, result.public_dict())
        self.assertGreaterEqual(result.margin, AD.MARGIN_THRESHOLD)
        self.assertGreaterEqual(result.margin_min, AD.MIN_GROUP_MARGIN)
        self.assertIsNone(result.cc_threshold)
        self.assertNotIn("low_cc", result.reason)

        unmarked = session.decode(self.edge.design.template)
        self.assertFalse(unmarked.accepted, unmarked.public_dict())
        self.assertIsNone(unmarked.digits)
        self.assertTrue(unmarked.ok, unmarked.public_dict())
        self.assertIn("low_margin", unmarked.reason)
        self.assertNotIn("low_cc", unmarked.reason)
        self.assertLess(unmarked.margin, AD.MARGIN_THRESHOLD, unmarked.public_dict())

        via_fn = AD.decode(
            plate, carrier="edge", design=self.edge.design, use_ecc_refine=False)
        self.assertTrue(via_fn.accepted, via_fn.public_dict())
        self.assertEqual(via_fn.digits, self.digits)
        self.assertIsNone(via_fn.cc_threshold)
        self.assertLess(via_fn.cc, AD.CC_THRESHOLD)


class ServiceContractTests(unittest.TestCase):
    """HTTP contract around arrow_decode. Does not load a SIFT template."""

    @classmethod
    def setUpClass(cls):
        cls._prev_warm = os.environ.get("DECODE_WARMUP")
        cls._prev_token = os.environ.get("DECODE_TOKEN")
        os.environ["DECODE_WARMUP"] = "0"
        os.environ.pop("DECODE_TOKEN", None)
        try:
            import fastapi  # noqa: F401
            from fastapi.testclient import TestClient
        except ImportError:  # pragma: no cover - local run without the service extras
            raise unittest.SkipTest(
                "fastapi/httpx not installed; run pip install -r requirements.txt")
        import services.decode_api.app as svc

        cls.svc = svc
        cls._orig_decode = svc.decode_image
        cls.client = TestClient(svc.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        client = getattr(cls, "client", None)
        if client is not None:
            client.__exit__(None, None, None)
        if getattr(cls, "svc", None) is not None and getattr(cls, "_orig_decode", None) is not None:
            cls.svc.decode_image = cls._orig_decode
        if cls._prev_warm is None:
            os.environ.pop("DECODE_WARMUP", None)
        else:
            os.environ["DECODE_WARMUP"] = cls._prev_warm
        if cls._prev_token is None:
            os.environ.pop("DECODE_TOKEN", None)
        else:
            os.environ["DECODE_TOKEN"] = cls._prev_token

    def setUp(self):
        self.calls = []
        self.raw = {
            "ok": True, "digits": "13800138000", "payload_valid": True,
            "margin": 4.2, "cc": 0.96, "mean_conf": 0.05, "logo_scale": 1.0,
        }

        def fake(carrier, gray, layout="2"):
            self.calls.append((carrier, int(gray.size), layout))
            if gray.size == 0:
                raw = {"ok": False, "reason": "unreadable", "digits": None}
            else:
                raw = self.raw
            return AD.apply_gate(raw, carrier=carrier)

        self.svc.decode_image = fake
        os.environ.pop("DECODE_TOKEN", None)

    def tearDown(self):
        self.svc.decode_image = self._orig_decode
        os.environ.pop("DECODE_TOKEN", None)

    def _post(self, path="/decode", **kwargs):
        files = kwargs.pop("files", {"image": ("a.png", _PNG_1X1, "image/png")})
        return self.client.post(path, files=files, **kwargs)

    def test_repo_root_resolves_from_service_package(self):
        self.assertTrue(os.path.isfile(os.path.join(self.svc.REPO_ROOT, "arrow_decode.py")))
        self.assertTrue(os.path.isfile(os.path.join(self.svc.REPO_ROOT, "arrow_wm.py")))

    def test_import_from_service_directory(self):
        service_dir = os.path.join(self.svc.REPO_ROOT, "services", "decode_api")
        code = (
            "import app\n"
            "import os\n"
            "assert os.path.isfile(os.path.join(app.REPO_ROOT, 'arrow_decode.py'))\n"
            "import arrow_decode\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=service_dir,
            env={**os.environ, "DECODE_WARMUP": "0"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_health_and_gateway_prefix(self):
        self.assertEqual(self.svc.strip_ms_prefix("/decode"), "/decode")
        self.assertEqual(self.svc.strip_ms_prefix("/ms-2w5g828w/decode"), "/decode")
        self.assertEqual(
            self.svc.strip_ms_prefix("/ms-a/ms-b/health"), "/health")
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertEqual(
            self.client.get("/ms-2w5g828w/health").json(), {"status": "ok"})
        prefixed = self._post("/ms-newid/ms-newid/decode")
        self.assertEqual(prefixed.status_code, 200, prefixed.text)
        self.assertEqual(self.calls[0][0], "inner")

    def test_default_carrier_and_public_dict_shape(self):
        response = self._post()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("接受".encode("utf-8"), response.content)
        for key in (
            "accepted", "status", "digits", "carrier", "margin", "margin_mean",
            "margin_min", "cc", "logo_scale", "mean_conf", "payload_valid",
            "ok", "reason", "reason_text", "margin_threshold",
            "min_group_margin_threshold", "cc_threshold",
        ):
            self.assertIn(key, body)
        self.assertNotIn("raw", body)
        self.assertNotIn("candidate_digits", body)
        self.assertTrue(body["accepted"])
        self.assertEqual(body["status"], "accepted")
        self.assertEqual(body["digits"], "13800138000")
        self.assertEqual(body["carrier"], "inner")
        self.assertEqual(body["reason"], "")
        self.assertEqual(body["reason_text"], "接受")
        self.assertTrue(body["ok"])
        self.assertEqual(self.calls, [("inner", 1, "2")])

    def test_reject_omits_candidate_and_ok_is_not_success(self):
        self.raw = {
            "ok": True, "digits": "13800138000", "payload_valid": True,
            "margin": 1.2, "cc": 0.97, "mean_conf": 0.04, "logo_scale": 1.0,
        }
        response = self._post()
        body = response.json()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(body["accepted"])
        self.assertEqual(body["status"], "rejected")
        self.assertIsNone(body["digits"])
        self.assertNotIn("candidate_digits", body)
        self.assertTrue(body["ok"])
        self.assertNotEqual(body["ok"], body["accepted"])
        self.assertEqual(body["reason"], "low_margin")
        self.assertEqual(body["reason_text"], "码字裕度不足")

    def test_candidate_only_with_debug_and_token(self):
        self.raw = {
            "ok": True, "digits": "13800138000", "payload_valid": True,
            "margin": 1.2, "cc": 0.97, "mean_conf": 0.04, "logo_scale": 1.0,
        }
        files = {"image": ("a.png", _PNG_1X1, "image/png")}
        open_debug = self.client.post("/decode?debug=1", files=files)
        self.assertEqual(open_debug.status_code, 200, open_debug.text)
        self.assertNotIn("candidate_digits", open_debug.json())
        self.assertEqual(open_debug.json()["status"], "rejected")

        os.environ["DECODE_TOKEN"] = "devsecret"
        denied = self.client.post(
            "/decode?debug=1",
            files=files,
            headers={"X-Decode-Token": "nope"},
        )
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(denied.json()["status"], "error")
        self.assertEqual(denied.json()["detail"], "unauthorized")
        self.assertNotIn("candidate_digits", denied.json())

        allowed = self.client.post(
            "/decode?debug=1",
            files=files,
            headers={"X-Decode-Token": "devsecret"},
        )
        self.assertEqual(allowed.status_code, 200, allowed.text)
        body = allowed.json()
        self.assertEqual(body["status"], "rejected")
        self.assertFalse(body["accepted"])
        self.assertIsNone(body["digits"])
        self.assertEqual(body["candidate_digits"], "13800138000")
        self.assertTrue(body["ok"])

    def test_http_errors_use_status_error(self):
        bad = self.client.post(
            "/decode?carrier=both",
            files={"image": ("a.png", _PNG_1X1, "image/png")},
        )
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(bad.json()["status"], "error")
        self.assertIn("detail", bad.json())
        self.assertNotIn("accepted", bad.json())
        self.assertNotIn("candidate_digits", bad.json())

    def test_query_carrier_beats_form_and_aliases_work(self):
        response = self.client.post(
            "/decode?carrier=edge",
            data={"carrier": "inner"},
            files={"photo": ("a.png", _PNG_1X1, "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["carrier"], "edge")
        self.calls.clear()
        response = self.client.post(
            "/decode",
            data={"carrier": "inner"},
            files={"file": ("a.png", _PNG_1X1, "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.calls[0][0], "inner")

    def test_bad_layout_is_an_http_error(self):
        response = self._post("/decode?layout=9")
        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["status"], "error")
        self.assertIn("layout", body["detail"])
        self.assertEqual(self.calls, [])

    def test_bad_carrier_and_missing_image(self):
        bad = self.client.post(
            "/decode?carrier=both",
            files={"image": ("a.png", _PNG_1X1, "image/png")},
        )
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(self.calls, [])
        missing = self.client.post("/decode", files={"carrier": (None, "edge")})
        self.assertEqual(missing.status_code, 400)
        self.assertEqual(self.calls, [])
        empty = self.client.post(
            "/decode", files={"image": ("a.png", b"", "image/png")})
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(self.calls, [])
        plain = self.client.post("/decode", data={"carrier": "edge"})
        self.assertEqual(plain.status_code, 400)

    def test_garbage_bytes_are_unreadable_not_a_number(self):
        response = self.client.post(
            "/decode",
            files={"image": ("a.jpg", b"not-an-image", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertFalse(body["accepted"])
        self.assertEqual(body["status"], "rejected")
        self.assertIsNone(body["digits"])
        self.assertNotIn("candidate_digits", body)
        self.assertEqual(body["reason"], "unreadable")
        self.assertEqual(body["reason_text"], "读不到图像")
        self.assertEqual(self.calls, [("inner", 0, "2")])

    def test_token_guards_decode_but_not_health(self):
        os.environ["DECODE_TOKEN"] = "devsecret"
        self.assertEqual(self.client.get("/health").status_code, 200)
        denied = self._post(headers={"X-Decode-Token": "nope"})
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(denied.json()["detail"], "unauthorized")
        self.assertEqual(self.calls, [])
        allowed = self._post(headers={"X-Decode-Token": "devsecret"})
        self.assertEqual(allowed.status_code, 200, allowed.text)
        os.environ["DECODE_TOKEN"] = "   "
        self.assertEqual(self._post().status_code, 200)

    def test_oversize_helper(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            self.svc.ensure_upload_size(self.svc.MAX_IMAGE_BYTES + 1)
        self.assertEqual(ctx.exception.status_code, 413)

    def test_json_base64_uses_same_gate(self):
        payload = base64.b64encode(_PNG_1X1).decode("ascii")
        response = self.client.post(
            "/decode",
            json={"image_base64": payload, "carrier": "inner"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["accepted"])
        self.assertEqual(body["status"], "accepted")
        self.assertEqual(body["digits"], "13800138000")
        self.assertNotIn("candidate_digits", body)
        self.assertEqual(body["carrier"], "inner")
        self.assertEqual(self.calls, [("inner", 1, "2")])

    def test_json_query_carrier_beats_body_and_data_url(self):
        payload = "data:image/png;base64," + base64.b64encode(_PNG_1X1).decode("ascii")
        response = self.client.post(
            "/decode?carrier=edge",
            json={"image_base64": payload, "carrier": "inner"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["carrier"], "edge")
        self.assertEqual(self.calls[0][0], "edge")

    def test_json_errors_do_not_decode(self):
        missing = self.client.post("/decode", json={"carrier": "edge"})
        self.assertEqual(missing.status_code, 400)
        self.assertEqual(missing.json()["detail"], "missing image_base64")
        bad = self.client.post("/decode", json={"image_base64": "!!!!"})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(bad.json()["detail"], "invalid image_base64")
        both = self.client.post("/decode", json={"image_base64": "aaaa", "carrier": "both"})
        self.assertEqual(both.status_code, 400)
        broken = self.client.post(
            "/decode",
            content=b"{",
            headers={"content-type": "application/json"},
        )
        self.assertEqual(broken.status_code, 400)
        from fastapi import HTTPException
        previous = self.svc._BASE64_CHAR_CAP
        self.svc._BASE64_CHAR_CAP = 8
        try:
            with self.assertRaises(HTTPException) as ctx:
                self.svc.decode_base64_image("ABCDEFGHIJKL")
        finally:
            self.svc._BASE64_CHAR_CAP = previous
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertEqual(self.calls, [])

    def test_raw_image_body_and_token(self):
        response = self.client.post(
            "/decode?carrier=inner",
            content=_PNG_1X1,
            headers={"content-type": "image/png"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.calls, [("inner", 1, "2")])
        empty = self.client.post(
            "/decode",
            content=b"",
            headers={"content-type": "application/octet-stream"},
        )
        self.assertEqual(empty.status_code, 400)
        os.environ["DECODE_TOKEN"] = "devsecret"
        denied = self.client.post(
            "/decode",
            json={"image_base64": base64.b64encode(_PNG_1X1).decode("ascii")},
            headers={"X-Decode-Token": "nope"},
        )
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(self.client.get("/health").status_code, 200)


class CloudRunPackagingTests(unittest.TestCase):
    """Repo-root and wxcloudrun/ build contexts ship the same image inputs."""

    # 单一来源：tools/sync_vendor.py 既定义清单，也能一键同步。
    from tools.sync_vendor import VENDOR_FILES as _VENDOR

    def _root(self):
        return os.path.dirname(os.path.abspath(__file__))

    def test_dockerfiles_are_identical_and_listen_on_8501(self):
        root = self._root()
        paths = [
            os.path.join(root, "Dockerfile"),
            os.path.join(root, "services", "decode_api", "Dockerfile"),
            os.path.join(root, "wxcloudrun", "Dockerfile"),
        ]
        bodies = []
        for path in paths:
            with open(path, "rb") as handle:
                bodies.append(handle.read())
        self.assertEqual(bodies[0], bodies[1])
        self.assertEqual(bodies[0], bodies[2])
        text = bodies[0].decode("utf-8")
        self.assertIn("EXPOSE 8501", text)
        self.assertIn("--host 0.0.0.0", text)
        self.assertIn("--port ${PORT:-8501}", text)
        self.assertIn("COPY services/decode_api/requirements.txt", text)
        self.assertIn("COPY arrow_decode.py arrow_wm.py arrow_inner.py ARROW_clean_pro.png", text)

    def test_dockerfiles_use_lf_endings(self):
        """CRLF turns ``FROM ...\\r`` into an unknown instruction on a local build."""
        root = self._root()
        for rel in ("Dockerfile", "services/decode_api/Dockerfile",
                    "wxcloudrun/Dockerfile"):
            with open(os.path.join(root, rel), "rb") as handle:
                body = handle.read()
            self.assertNotIn(b"\r\n", body, "%s has CRLF line endings" % rel)

    def test_wxcloudrun_vendor_matches_repo(self):
        root = self._root()
        for rel in self._VENDOR:
            src = os.path.join(root, rel)
            dst = os.path.join(root, "wxcloudrun", rel)
            with open(src, "rb") as src_file, open(dst, "rb") as dst_file:
                same = src_file.read() == dst_file.read()
            self.assertTrue(
                same,
                "wxcloudrun/%s 与 %s 不一致；运行 `python tools/sync_vendor.py` 同步"
                % (rel, rel),
            )


class LayoutVersionTests(unittest.TestCase):
    """v1 stays on the frozen plate. v2 is the default depth-6 layout."""

    @classmethod
    def setUpClass(cls):
        import arrow_wm as A
        cls.A = A
        cls.v1 = A.ArrowDesign(layout_version=1, depth=A.V1_DEPTH, verbose=False)
        cls.v1_deep = A.ArrowDesign(layout_version=1, depth=A.MOD_DEPTH, verbose=False)
        cls.v2 = A.ArrowDesign(verbose=False)

    def test_v1_geometry_matches_golden_sha1(self):
        A = self.A
        self.assertEqual(A.layout_geometry_sha1(self.v1.cells), A.V1_GEOMETRY_SHA1)
        self.assertEqual(
            A.layout_geometry_sha1(self.v1.cells, with_window=False),
            A.layout_geometry_sha1(self.v1_deep.cells, with_window=False))
        self.assertEqual(self.v1.pair_perm, tuple(range(A.N_BITS)))
        self.assertEqual(self.v1.pair_perm_id, A.PAIR_PERM_ID_V1)
        self.assertNotEqual(self.v1.cells[0].half_win, self.v1_deep.cells[0].half_win)

    def test_v2_cells_are_same_interval_and_deep_enough(self):
        A = self.A
        cells = self.v2.cells
        self.assertEqual(len(cells), 140)
        self.assertEqual(self.v2.layout_version, 2)
        self.assertEqual(self.v2.depth, A.MOD_DEPTH)
        self.assertEqual(self.v2.pair_perm, A.PAIR_PERM_V2)
        self.assertEqual(sorted(self.v2.pair_perm), list(range(70)))
        self.assertNotEqual(self.v2.pair_perm, tuple(range(70)))
        min_clear = A.min_clearance_for_depth(A.MOD_DEPTH)
        min_w = A.min_stroke_w_for_depth(A.MOD_DEPTH)
        self.assertEqual(min_clear, 24)
        self.assertEqual(min_w, 24.0)
        letters = set()
        for i in range(0, 140, 2):
            a, b = cells[i], cells[i + 1]
            self.assertEqual(a.track, b.track)
            self.assertEqual(a.interval, b.interval)
            h = min(a.height, b.height)
            self.assertGreaterEqual(h, A.MIN_CELL_H)
            self.assertLessEqual(abs(a.y_mid - b.y_mid), 1.2 * h + 1e-6)
            self.assertGreaterEqual(min(a.clearance, b.clearance), min_clear)
            self.assertGreaterEqual(min(a.width, b.width), min_w)
            letters.add(a.letter)
        self.assertGreaterEqual(len(letters), 4)
        for g in range(10):
            group = self.v2.pair_perm[7 * g:7 * g + 7]
            tracks = {cells[2 * p].track for p in group}
            names = {cells[2 * p].letter for p in group}
            self.assertGreaterEqual(len(tracks), 4)
            self.assertGreaterEqual(len(names), 4)

    def test_v1_clean_roundtrip_and_inner_still_builds(self):
        import arrow_decode as AD
        import arrow_inner as I
        digits = "13800138000"
        plate = self.A.WatermarkEmbedder(self.v1, verbose=False).embed(digits)
        self.assertEqual(plate["layout_version"], 1)
        self.assertEqual(plate["depth"], self.A.V1_DEPTH)
        self.assertEqual(plate["pair_perm_id"], self.A.PAIR_PERM_ID_V1)
        session = AD.Decoder(carrier="edge", design=self.v1, use_ecc_refine=False)
        result = session.decode(plate["plate"])
        self.assertTrue(result.accepted, result.public_dict())
        self.assertEqual(result.digits, digits)
        # v2 must not be what decodes a v1 plate bit-exact; the smoke is that
        # constructing the inner carrier on a v2 design does not raise.
        inner = I.InnerCarrier(self.v2, verbose=False)
        inner_plate = inner.embed(digits)
        self.assertEqual(inner_plate["digits"], digits)
        self.assertEqual(len(inner.cells), I.N_CELLS_V3)
        self.assertEqual(inner.layout_version, I.LAYOUT_V3)

    def test_clean_soft_n_is_order_one(self):
        digits = "13800138000"
        emb = self.A.WatermarkEmbedder(self.v2, verbose=False)
        plate = emb.embed(digits)
        dec = self.A.WatermarkDecoder(self.v2, use_ecc_refine=False, verbose=False)
        res = dec.decode(plate["plate"])
        self.assertEqual(res["digits"], digits)
        soft = res["soft_n"]
        self.assertGreater(float(abs(soft).mean()), 0.5)
        self.assertLess(float(abs(soft).mean()), 1.6)


class InnerV3Tests(unittest.TestCase):
    """v3: 154 码元、11 组交织、CRC-7 取代固定补零。"""

    @classmethod
    def setUpClass(cls):
        import arrow_inner as I
        import arrow_wm as A
        cls.I = I
        cls.A = A
        cls.design = A.ArrowDesign(verbose=False)

    def test_codec_roundtrip_corrects_one_bit(self):
        I = self.I
        bits = I.encode_payload_v3("17701234567")
        self.assertEqual(len(bits), I.N_BITS_V3)
        soft = [1.0 if b else -1.0 for b in bits]
        digits, crc_ok, changed = I.decode_payload_v3_soft(soft)
        self.assertEqual(digits, "17701234567")
        self.assertTrue(crc_ok)
        self.assertEqual(changed, 0)
        soft[3] *= -1.0            # 每组可纠 1 位，CRC 仍应通过
        digits, crc_ok, _ = I.decode_payload_v3_soft(soft)
        self.assertEqual(digits, "17701234567")
        self.assertTrue(crc_ok)

    def test_each_hamming_group_spans_three_holes(self):
        I = self.I
        carrier = I.InnerCarrier(self.design, verbose=False,
                                 layout_version=I.LAYOUT_V3)
        flat_to_pair = {}
        off = 0
        for hi, h in enumerate(carrier.holes):
            for k in range(h["n_cells"] // 2):
                flat_to_pair[off + k] = (hi, k)
            off += h["n_cells"] // 2
        for g in range(I.N_BITS_V3 // 7):
            holes = {flat_to_pair[carrier.pair_order[b]][0]
                     for b in range(7 * g, 7 * g + 7)}
            self.assertGreaterEqual(len(holes), 3, (g, holes))

    def test_auto_reads_both_plate_layouts(self):
        I = self.I
        session = AD.Decoder(carrier="inner", design=self.design)
        for layout in (I.LAYOUT_V2, I.LAYOUT_V3):
            plate = I.InnerCarrier(self.design, verbose=False,
                                   layout_version=layout).embed("13912345678")["plate"]
            result = session.decode(plate)
            self.assertTrue(result.accepted, (layout, result.public_dict()))
            self.assertEqual(result.digits, "13912345678")
            self.assertEqual(result.raw.get("layout_version"), layout)

    def test_crc_mismatch_is_rejected(self):
        raw = {
            "ok": True, "digits": "13912345678", "payload_valid": True,
            "margin": 3.2, "margin_min": 1.2, "cc": 0.97, "crc_ok": False,
            "logo_scale": 0.5, "mean_conf": 0.05,
        }
        result = AD.apply_gate(raw, carrier="inner")
        self.assertFalse(result.accepted, result.public_dict())
        self.assertIn("crc_mismatch", result.reason)
        self.assertIsNone(result.digits)


class SampleAssetTests(unittest.TestCase):
    """The committed plates are fixtures, not just docs.

    Synthetic roundtrips use the same code for embed and decode, so they can
    stay green while the layout quietly drifts away from already-printed
    plates. Decoding the checked-in edge/inner samples is the anchor that
    catches that: these files were produced by older builds.
    """

    _SAMPLES = (
        ("edge", "samples/edge/ARROW_13800138000.png", "13800138000"),
        ("inner", "samples/inner/ARROW_13800138000.png", "13800138000"),
    )

    def test_committed_plates_still_decode(self):
        import arrow_decode as AD
        root = os.path.dirname(os.path.abspath(__file__))
        for carrier, rel, digits in self._SAMPLES:
            path = os.path.join(root, rel)
            self.assertTrue(os.path.exists(path), path)
            result = AD.decode(path, carrier=carrier)
            self.assertTrue(result.accepted, (carrier, result.public_dict()))
            self.assertEqual(result.digits, digits, carrier)


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
