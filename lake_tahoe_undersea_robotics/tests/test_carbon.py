import base64
import copy
import json
import math
import unittest

from app import app
from backend.carbon_economics import Assumptions, evaluate, parse_assumptions
from backend.pqc_reports import ReportSigner, canonical_bytes


class EconomicsTests(unittest.TestCase):
    def test_default_accounting(self):
        r = evaluate(Assumptions())
        self.assertEqual(r["capital_cad"], 1100000)
        self.assertEqual(r["annual_kwh"], 9360)
        self.assertAlmostEqual(r["annual_operational_kg"], 374.4)
        self.assertEqual(r["embodied_kg"], 53000)
        self.assertAlmostEqual(r["net_avoided_kg_10yr"], 193256)
        self.assertAlmostEqual(r["financial_npv_cad"], 591900.4159903412)
        self.assertEqual(r["trajectory"][0]["carbon_valued_npv_cad"], -1105300)
        self.assertEqual(r["trajectory"][0]["npv_cad"], -1100000)
        self.assertEqual(r["trajectory"][0]["project_kg"], 53000)
        self.assertTrue(r["viable_under_assumptions"])
        self.assertFalse(r["qml"]["used_for_financial_forecast"])

    def test_zero_discount_matches_cash_sum(self):
        r = evaluate(parse_assumptions({"discount": 0}))
        cash = 250000 + 100000 - r["annual_cost_cad"]
        self.assertEqual(r["financial_npv_cad"], -1100000 + 10 * cash)

    def test_qml_overhead_and_savings(self):
        base = evaluate(Assumptions())
        overhead = evaluate(parse_assumptions({"qml_kw": 10}))
        savings = evaluate(parse_assumptions({"qml_reduction": 0.5}))
        self.assertGreater(overhead["annual_kwh"], base["annual_kwh"])
        self.assertLess(overhead["financial_npv_cad"], base["financial_npv_cad"])
        self.assertLess(savings["annual_kwh"], base["annual_kwh"])
        self.assertEqual(savings["annual_kwh"], 1800 * (4 * 0.8 * 0.5 + 2))

    def test_no_carbon_break_even(self):
        r = evaluate(parse_assumptions({"baseline_kg": 0}))
        self.assertIsNone(r["carbon_break_even_years"])
        self.assertFalse(r["viable_under_assumptions"])
        self.assertLess(r["net_avoided_kg_10yr"], 0)

    def test_sensitivity_and_prime_groups(self):
        r = evaluate(parse_assumptions({"fleet": 7, "prime": 3}))
        self.assertEqual([x["residue"] for x in r["number_theory"]["station_groups"]],
                         [1, 2, 0, 1, 2, 0, 1])
        self.assertEqual(r["capex_sensitivity"][2]["npv_cad"], r["financial_npv_cad"])
        self.assertAlmostEqual(r["capex_sensitivity"][0]["npv_cad"],
                               r["financial_npv_cad"] + r["capital_cad"] * 0.5)

    def test_nonfinite_and_invalid_inputs(self):
        invalid = [
            [], {"fleet": True}, {"fleet": 2.5}, {"prime": 9}, {"hours": -1},
            {"hours": "1800"}, {"grid_kg_kwh": math.inf},
            {"discount": math.nan}, {"qml_reduction": 0.9}, {"unknown": 1},
            {"robot_capex": 10 ** 1000},
        ]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_assumptions(data)


class CryptoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signer = ReportSigner()
        cls.report = evaluate(Assumptions())

    def test_signature_tamper_and_wrong_key(self):
        envelope = self.signer.sign(self.report)
        self.assertTrue(self.signer.verify(envelope))
        altered = copy.deepcopy(envelope)
        altered["report"]["capital_cad"] += 1
        self.assertFalse(self.signer.verify(altered))
        with self.assertRaises(ValueError):
            ReportSigner().verify(envelope)

    def test_order_independent_canonicalization(self):
        self.assertEqual(canonical_bytes({"b": 2, "a": 1}),
                         canonical_bytes({"a": 1, "b": 2}))
        self.assertTrue(canonical_bytes({}).startswith(b"ONTARIO-CARBON-REPORT-V1\x00"))
        with self.assertRaises(ValueError):
            canonical_bytes({"nan": math.nan})

    def test_corrupt_signature(self):
        envelope = self.signer.sign(self.report)
        sig = bytearray(base64.b64decode(envelope["signature_b64"]))
        sig[20] ^= 1
        envelope["signature_b64"] = base64.b64encode(sig).decode()
        self.assertFalse(self.signer.verify(envelope))

    def test_browser_numeric_round_trip(self):
        envelope = self.signer.sign(self.report)
        def browser_number(value):
            numeric = float(value)
            return int(numeric) if numeric.is_integer() else numeric
        restored = json.loads(json.dumps(envelope), parse_float=browser_number)
        self.assertTrue(self.signer.verify(restored))
        self.assertEqual(canonical_bytes({"x": 0.0}), canonical_bytes({"x": 0}))
        self.assertEqual(canonical_bytes({"x": 1e-7}).split(b"\x00")[1],
                         b'{"x":1e-7}')

    def test_self_test_and_cloak_helper(self):
        result = self.signer.self_test(self.report)
        for key in ("signature_valid", "altered_report_rejected",
                    "kem_shared_secret_agreed", "altered_ciphertext_secret_mismatch"):
            self.assertTrue(result[key])
        self.assertEqual(result["cloak_integration"]["ramanujan_c5_1"], -1)
        self.assertEqual(result["public_key_bytes"], 1952)
        self.assertEqual(result["signature_bytes"], 3309)


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_evaluate_verify_and_alter(self):
        response = self.client.post("/api/carbon/evaluate", json={"fleet": 6})
        self.assertEqual(response.status_code, 200)
        envelope = response.json["envelope"]
        self.assertTrue(self.client.post("/api/carbon/pqc/verify", json=envelope).json["signature_valid"])
        envelope["report"]["assumptions"]["fleet"] = 7
        self.assertFalse(self.client.post("/api/carbon/pqc/verify", json=envelope).json["signature_valid"])

    def test_input_errors_explicit(self):
        for value in ([], {"prime": 4}, {"hours": -1}, {"fleet": False}):
            response = self.client.post("/api/carbon/evaluate", json=value)
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json["status"], "error")
        response = self.client.post("/api/carbon/evaluate", data="not json")
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/api/carbon/evaluate", data='{"fleet":',
                                    content_type="application/json")
        self.assertEqual(response.status_code, 400)

    def test_limits_and_invalid_base64(self):
        response = self.client.post("/api/carbon/pqc/verify", data="x" * 140000,
                                    content_type="application/json")
        self.assertEqual(response.status_code, 413)
        envelope = self.client.post("/api/carbon/evaluate", json={}).json["envelope"]
        envelope["signature_b64"] = "not base64!"
        self.assertEqual(self.client.post("/api/carbon/pqc/verify", json=envelope).status_code, 400)

    def test_pdf_and_default_api(self):
        self.assertEqual(self.client.get("/api/carbon/defaults").json["assumptions"]["fleet"], 4)
        response = self.client.post("/api/carbon/preprint", json={"fleet": 2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/pdf")
        self.assertTrue(response.data.startswith(b"%PDF-"))
        self.assertIn("Ontario_Carbon_Economics", response.headers["Content-Disposition"])

    def test_existing_routes_preserved(self):
        for path in ("/", "/api/ontario/stations", "/api/auv/stations",
                     "/api/ontario/recovery/10yr", "/api/ontario/preprint/view"):
            with self.client.get(path) as response:
                self.assertEqual(response.status_code, 200, path)


if __name__ == "__main__":
    unittest.main()
