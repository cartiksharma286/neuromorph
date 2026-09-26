import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import ROOT, app, simulate


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_simulation_response_contains_finite_series(self):
        response = self.client.post("/api/simulate", json={"spin": 0.5, "modulus": 0.7})
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(payload["ramanujan_delta_harmonic"]), 96)
        self.assertTrue(0 < payload["elliptic_nome"] < 1)
        self.assertGreater(payload["hawking_temperature_k"], 0)
        # Test ergodic game theory and statistical invariance payloads
        self.assertIn("ergodic_game", payload)
        self.assertIn("statistical_invariance", payload)
        self.assertEqual(len(payload["ergodic_game"]["pareto_frontier"]), 36)
        self.assertIn("nash_equilibrium", payload["ergodic_game"])
        self.assertEqual(len(payload["statistical_invariance"]["flux_bins"]), 48)
        self.assertGreater(payload["statistical_invariance"]["lyapunov_gamma"], 0)
        self.assertTrue(0 < payload["statistical_invariance"]["subring_scale_decay"] < 1)
        self.assertGreaterEqual(payload["statistical_invariance"]["wasserstein_w1"], 0)

    def test_parameters_are_clamped(self):
        payload = simulate({"mass": 0, "spin": 4, "inclination": 120, "modulus": -2})
        self.assertEqual(payload["parameters"]["mass_solar"], 1)
        self.assertEqual(payload["parameters"]["spin"], 0.98)
        self.assertEqual(payload["parameters"]["inclination_deg"], 85)
        self.assertEqual(payload["parameters"]["elliptic_modulus"], 0.05)

    def test_preprint_route_matches_artifact_state(self):
        response = self.client.get("/preprint.pdf")
        expected = 200 if (Path(ROOT) / "Nature_Preprint_Cosmological_Modular_Signatures.pdf").exists() else 404
        self.assertEqual(response.status_code, expected)
        response.close()

    def test_reconstruction_and_hifu_payloads(self):
        response = self.client.post("/api/simulate", json={"hifu_freq": 1.5, "hifu_pressure": 0.9})
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        # Image reconstruction assertions
        self.assertIn("image_reconstruction", payload)
        self.assertTrue(len(payload["image_reconstruction"]["cf_gamma_terms"]) >= 2)
        self.assertEqual(len(payload["image_reconstruction"]["cross_section"]), 64)
        self.assertGreater(payload["image_reconstruction"]["metrics"]["snr_gain_db"], 10.0)
        self.assertGreater(payload["image_reconstruction"]["metrics"]["ssim"], 0.70)

        # HIFU dementia care assertions
        self.assertIn("hifu_dementia", payload)
        self.assertGreater(payload["hifu_dementia"]["mechanical_index"], 0.0)
        self.assertGreater(payload["hifu_dementia"]["bbb_permeability_ratio"], 1.0)
        self.assertEqual(len(payload["hifu_dementia"]["session_trajectory"]), 8)
        self.assertGreater(payload["hifu_dementia"]["stable_cavitation"], 0.0)

    def test_hifu_preprint_route(self):
        response = self.client.get("/hifu-preprint.pdf")
        expected = 200 if (Path(ROOT) / "Nature_Preprint_HIFU_Cosmological_Reconstruction.pdf").exists() else 404
        self.assertEqual(response.status_code, expected)
        response.close()


if __name__ == "__main__":
    unittest.main()