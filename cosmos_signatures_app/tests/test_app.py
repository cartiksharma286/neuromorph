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

    def test_special_relativity_and_conebeam_payloads(self):
        response = self.client.post("/api/simulate", json={
            "sr_energy": 60.0,
            "sr_fork_gev": 50.0,
            "sr_liv_coupling": 0.45,
            "cbct_kvp": 120.0,
            "cbct_cone_angle": 14.0,
            "cf_order": 6
        })
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("special_relativity_conebeam", payload)
        sr = payload["special_relativity_conebeam"]

        # Kinematics assertions
        self.assertIn("kinematics", sr)
        self.assertGreater(sr["kinematics"]["lorentz_gamma"], 1.0)
        self.assertGreater(sr["kinematics"]["beta"], 0.99)
        self.assertGreater(sr["kinematics"]["speed_deficit_ms"], 0.0)
        self.assertLess(sr["kinematics"]["speed_deficit_ms"], 300000000.0)

        # Fork in the road in terms of GeV
        self.assertIn("fork_in_the_road", sr)
        fork = sr["fork_in_the_road"]
        self.assertEqual(fork["fork_threshold_gev"], 50.0)
        self.assertTrue(fork["bifurcation_active"])
        self.assertGreater(fork["bifurcation_split_ms"], 0.0)
        self.assertEqual(len(fork["spectrum_points"]), 60)
        for pt in fork["spectrum_points"]:
            self.assertGreaterEqual(pt["beta_superluminal"], pt["beta_subluminal"])

        # Conebeam CT assertions
        self.assertIn("conebeam_ct", sr)
        cbct = sr["conebeam_ct"]
        self.assertEqual(len(cbct["spatial_line_profile"]), 50)
        self.assertGreater(cbct["max_cupping_artifact_hu"], 50.0)
        self.assertLess(cbct["corrected_residual_hu"], cbct["max_cupping_artifact_hu"] * 0.15)

        # Statistical distributions assertions
        self.assertIn("statistical_distribution", sr)
        stat = sr["statistical_distribution"]
        self.assertEqual(len(stat["maxwell_juttner_pdf"]), 48)
        self.assertGreater(stat["mean_gamma"], 1.0)
        self.assertGreater(stat["std_dev_gamma"], 0.0)
        self.assertEqual(len(stat["cf_convergence"]), 10)

    def test_relativity_preprint_route(self):
        response = self.client.get("/relativity-preprint.pdf")
        expected = 200 if (Path(ROOT) / "Nature_Preprint_Special_Relativity_CT_Conebeam_Stieltjes.pdf").exists() else 404
        self.assertEqual(response.status_code, expected)
        response.close()

    def test_big_bang_and_quantum_payloads(self):
        response = self.client.post("/api/simulate", json={
            "bb_prime_modulus": 17,
            "bb_inflation_efolds": 60.0,
            "bb_pbh_mass_log": 11.0,
            "bb_pbh_fraction": 0.04,
            "bb_quantum_coherence": 0.65
        })
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("big_bang", payload)
        bb = payload["big_bang"]

        # Timeline assertions
        self.assertIn("cosmic_timeline", bb)
        self.assertEqual(len(bb["cosmic_timeline"]), 60)
        self.assertLess(bb["cosmic_timeline"][0]["log_t_s"], -40.0)
        self.assertGreater(bb["cosmic_timeline"][-1]["log_t_s"], 10.0)

        # Prime moduli lattice assertions
        self.assertIn("prime_moduli", bb)
        self.assertEqual(bb["prime_moduli"]["prime_p"], 17)
        self.assertEqual(len(bb["prime_moduli"]["lattice"]), 17)
        for pt in bb["prime_moduli"]["lattice"]:
            self.assertTrue(0 <= pt["curvature_mod_p"] < 17)
            self.assertIn(pt["legendre"], (-1, 0, 1))

        # PBH Evaporation assertions
        self.assertIn("pbh_evaporation", bb)
        self.assertGreater(bb["pbh_evaporation"]["lifetime_s"], 0)
        self.assertGreater(bb["pbh_evaporation"]["peak_hawking_gev"], 1.0)

        # Quantum signatures assertions
        self.assertIn("quantum_signatures", bb)
        self.assertEqual(len(bb["quantum_signatures"]["wavefunction"]), 48)
        self.assertEqual(len(bb["quantum_signatures"]["hawking_spectrum"]), 48)
        self.assertGreater(bb["quantum_signatures"]["bogoliubov_beta_peak"], 0)

    def test_big_bang_preprint_route(self):
        response = self.client.get("/big-bang-preprint.pdf")
        expected = 200 if (Path(ROOT) / "Nature_Preprint_Big_Bang_Prime_Moduli_Quantum_Radiation.pdf").exists() else 404
        self.assertEqual(response.status_code, expected)
        response.close()


if __name__ == "__main__":
    unittest.main()