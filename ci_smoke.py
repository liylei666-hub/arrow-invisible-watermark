"""Fast production-decode smoke. Used by GitHub Actions and local checks.

Covers the accept/reject gate without a robustness bench, plus one clean
embed/decode roundtrip on the edge carrier (the production default) and one
on the inner carrier. Inner extreme-tier rates are intentionally not measured.
"""
from __future__ import annotations

import base64
import inspect
import os
import subprocess
import sys
import tempfile
import unittest

import arrow_decode as AD


class GateTests(unittest.TestCase):
    def test_default_carrier_is_edge(self):
        self.assertEqual(AD.DEFAULT_CARRIER, "edge")
        self.assertEqual(
            inspect.signature(AD.decode).parameters["carrier"].default, "edge")
        self.assertEqual(
            inspect.signature(AD.Decoder).parameters["carrier"].default, "edge")

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

    def test_clear_read_is_accepted(self):
        raw = {
            "ok": True, "digits": "13800138000", "payload_valid": True,
            "margin": 4.2, "cc": 0.96, "mean_conf": 0.05, "logo_scale": 1.0,
        }
        result = AD.apply_gate(raw)
        self.assertTrue(result.accepted)
        self.assertEqual(result.digits, "13800138000")
        self.assertEqual(result.reason, "")
        self.assertGreaterEqual(result.margin, AD.MARGIN_THRESHOLD)
        self.assertGreaterEqual(result.cc, AD.CC_THRESHOLD)


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
        self.assertEqual(result.digits, self.digits)
        self.assertEqual(result.carrier, "edge")
        self.assertGreaterEqual(result.margin, AD.MARGIN_THRESHOLD)
        self.assertGreaterEqual(result.cc, AD.CC_THRESHOLD)
        self.assertEqual(result.cc_threshold, AD.CC_THRESHOLD)
        self.assertTrue(self.edge.use_ecc_refine)
        self.assertIsNotNone(result.logo_scale)

        # Same plate via the function entry and via a file path.
        via_fn = AD.decode(plate, design=self.edge.design)
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
        self.assertEqual(result.digits, self.digits)
        self.assertGreaterEqual(result.margin, AD.MARGIN_THRESHOLD)
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
            plate, design=self.edge.design, use_ecc_refine=False)
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
        import services.decode_api.app as svc
        from fastapi.testclient import TestClient

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

        def fake(carrier, gray):
            self.calls.append((carrier, int(gray.size)))
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
        self.assertEqual(self.calls[0][0], "edge")

    def test_default_carrier_and_public_dict_shape(self):
        response = self._post()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("接受".encode("utf-8"), response.content)
        for key in (
            "accepted", "digits", "candidate_digits", "carrier", "margin", "cc",
            "logo_scale", "mean_conf", "payload_valid", "ok", "reason",
            "reason_text", "margin_threshold", "cc_threshold",
        ):
            self.assertIn(key, body)
        self.assertNotIn("raw", body)
        self.assertTrue(body["accepted"])
        self.assertEqual(body["digits"], "13800138000")
        self.assertEqual(body["carrier"], "edge")
        self.assertEqual(body["reason"], "")
        self.assertEqual(body["reason_text"], "接受")
        self.assertEqual(self.calls, [("edge", 1)])

    def test_reject_hides_candidate_digits(self):
        self.raw = {
            "ok": True, "digits": "13800138000", "payload_valid": True,
            "margin": 1.2, "cc": 0.97, "mean_conf": 0.04, "logo_scale": 1.0,
        }
        body = self._post().json()
        self.assertFalse(body["accepted"])
        self.assertIsNone(body["digits"])
        self.assertEqual(body["candidate_digits"], "13800138000")
        self.assertNotEqual(body["digits"], body["candidate_digits"])
        self.assertEqual(body["reason"], "low_margin")
        self.assertEqual(body["reason_text"], "码字裕度不足")

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
        self.assertIsNone(body["digits"])
        self.assertEqual(body["reason"], "unreadable")
        self.assertEqual(body["reason_text"], "读不到图像")
        self.assertEqual(self.calls, [("edge", 0)])

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


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
