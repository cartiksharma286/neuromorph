import json
import unittest
from dataclasses import asdict

import numpy as np

from app import app
from backend.ecoli_analysis import Inputs, analyze, kernel, parse_inputs, slam
from backend.ecoli_preprint import render_preprint


class SpectroscopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = analyze(Inputs())

    def test_reproducibility_and_finite_output(self):
        self.assertEqual(self.report, analyze(Inputs()))
        json.dumps(self.report, allow_nan=False)
        self.assertNotEqual(self.report["spectra"], analyze(Inputs(seed=43))["spectra"])

    def test_spectral_axes_and_shapes(self):
        s = self.report["spectra"]
        self.assertEqual(len(s["chemical_shift_ppm"]), 401)
        self.assertEqual(len(s["nmr_intensity_au"]), 401)
        self.assertEqual(len(s["cpmg_time_ms"]), 21)
        self.assertEqual(len(s["cpmg_intensity_au"]), 21)
        self.assertEqual(len(s["dosy_gradient_t_m"]), 21)
        self.assertEqual([s["chemical_shift_ppm"][0], s["chemical_shift_ppm"][-1]], [0.0, 10.0])
        self.assertEqual([s["cpmg_time_ms"][0], s["cpmg_time_ms"][-1]], [0.0, 400.0])

    def test_kernel_is_valid_fidelity(self):
        x = np.random.default_rng(4).normal(size=(20, 8))
        gram = kernel(x, x)
        np.testing.assert_allclose(np.diag(gram), 1)
        np.testing.assert_allclose(gram, gram.T)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(gram).min()), -1e-10)
        self.assertTrue(np.all((gram >= 0) & (gram <= 1)))

    def test_held_out_metrics_match_confusion_counts(self):
        ch = self.report["chemometrics"]
        c = ch["confusion"]
        self.assertEqual(sum(c.values()), 60)
        self.assertEqual(ch["training_samples"], 180)
        self.assertAlmostEqual(ch["accuracy"], (c["tp"] + c["tn"]) / 60)
        self.assertAlmostEqual(ch["sensitivity"], c["tp"] / (c["tp"] + c["fn"]))
        self.assertAlmostEqual(ch["specificity"], c["tn"] / (c["tn"] + c["fp"]))
        self.assertIn(ch["synthetic_class"], ("higher-burden", "lower-burden"))
        self.assertIn("acetate_1_92ppm", ch["deconvolved_metabolites"])
        self.assertIn("lactate_1_33ppm", ch["deconvolved_metabolites"])

    def test_sample_and_recovery_equations(self):
        trajectory = self.report["sample_trajectory"]
        self.assertEqual(len(trajectory), 13)
        self.assertEqual(trajectory[0]["ecoli_cfu_100ml"], 1200)
        self.assertAlmostEqual(trajectory[-1]["ecoli_cfu_100ml"], 1200 * np.exp(-0.035 * 72))
        self.assertEqual(len(self.report["recovery"]), 11)
        self.assertAlmostEqual(self.report["recovery"][-1]["intervention_cfu_100ml"],
                               1200 * np.exp(-0.18 * 10))
        for row in trajectory:
            self.assertLessEqual(row["low"], row["ecoli_cfu_100ml"])
            self.assertGreaterEqual(row["high"], row["ecoli_cfu_100ml"])

    def test_zero_concentration_and_zero_rates(self):
        r = analyze(Inputs(ecoli_cfu_100ml=0, nitrate_mg_l=0, phosphate_mg_l=0,
                           turbidity_ntu=0, organic_mg_l=0,
                           decay_per_hour=0, intervention_per_year=0))
        json.dumps(r, allow_nan=False)
        self.assertTrue(all(v["ecoli_cfu_100ml"] == 0 for v in r["sample_trajectory"]))
        self.assertTrue(render_preprint(r).startswith(b"%PDF-"))
        r = analyze(Inputs(decay_per_hour=0, intervention_per_year=0))
        self.assertTrue(all(v["ecoli_cfu_100ml"] == 1200 for v in r["sample_trajectory"]))
        self.assertTrue(all(v["intervention_cfu_100ml"] == 1200 for v in r["recovery"]))

    def test_slam_covariance_and_rmse(self):
        s = self.report["slam"]
        covariance = np.array(s["final_covariance"])
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-9)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(covariance).min()), -1e-9)
        self.assertEqual(len(s["estimated_xy_m"]), 61)
        self.assertEqual(len(s["position_sigma_m"]), 61)
        self.assertEqual(len(s["estimated_landmarks_xy_m"]), 3)
        error = np.array(s["estimated_xy_m"]) - np.array(s["truth_xy_m"])
        self.assertAlmostEqual(s["rmse_m"], np.sqrt(np.mean(np.sum(error ** 2, axis=1))))
        self.assertLess(s["rmse_m"], s["dead_reckoning_rmse_m"])
        self.assertLess(s["rmse_m"], 1.5)
        self.assertNotEqual(s["estimated_xy_m"], slam(43, 0.025)["estimated_xy_m"])

    def test_bounded_inputs(self):
        invalid = [[], None, {"bad": 1}, {"seed": 1.5}, {"seed": True},
                   {"sensor_noise": 0}, {"ecoli_cfu_100ml": -1},
                   {"nitrate_mg_l": float("nan")}, {"organic_mg_l": float("inf")},
                   {"seed": "42"}, {"ecoli_cfu_100ml": 10**1000}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_inputs(value)
        self.assertEqual(parse_inputs({}), Inputs())
        self.assertEqual(parse_inputs({"seed": 3.0}).seed, 3)

    def test_pdf_and_endpoints(self):
        client = app.test_client()
        defaults = client.get("/api/ontario/ecoli/defaults")
        self.assertEqual(defaults.status_code, 200)
        self.assertEqual(defaults.json["inputs"], asdict(Inputs()))
        response = client.post("/api/ontario/ecoli/analyze", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["report"], self.report)
        for payload in ([], {"seed": -1}, {"unexpected": 1}):
            self.assertEqual(client.post("/api/ontario/ecoli/analyze", json=payload).status_code, 400)
            self.assertEqual(client.post("/api/ontario/ecoli/preprint", json=payload).status_code, 400)
        self.assertEqual(client.post("/api/ontario/ecoli/analyze", data="no").status_code, 400)
        self.assertEqual(client.post("/api/ontario/ecoli/analyze",
                                    data="{" + " " * 17000, content_type="application/json").status_code, 413)
        pdf = client.post("/api/ontario/ecoli/preprint", json={"ecoli_cfu_100ml": 321})
        self.assertEqual(pdf.status_code, 200)
        self.assertEqual(pdf.mimetype, "application/pdf")
        self.assertTrue(pdf.data.startswith(b"%PDF-"))
        self.assertGreater(len(pdf.data), 15000)
        html = client.get("/").data
        self.assertIn(b'id="tab-ecoli"', html)
        self.assertIn(b'/static/ecoli.js', html)


if __name__ == "__main__":
    unittest.main()
