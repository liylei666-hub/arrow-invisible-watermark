"""Fast production-decode smoke. Used by GitHub Actions and local checks.

Covers the accept/reject gate without a robustness bench, plus one clean
embed/decode roundtrip on the edge carrier (the production default) and one
on the inner carrier. Inner extreme-tier rates are intentionally not measured.
"""
from __future__ import annotations

import inspect
import os
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


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
