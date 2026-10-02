"""Unit tests for Parkinson's Syndrome Combinatorial Optimal Staging & BEM rTMS."""

import unittest
from logic.parkinsons_rtms import simulate_parkinsons_rtms, compute_stieltjes_convergents
from app import app


class TestParkinsonsRTMS(unittest.TestCase):

    def test_stieltjes_convergents(self):
        convergents = compute_stieltjes_convergents(1.61803398875, depth=5)
        self.assertEqual(len(convergents), 5)
        self.assertEqual(convergents[0]["term"], 1)
        self.assertEqual(convergents[0]["fraction"], "1/1")
        self.assertEqual(convergents[1]["fraction"], "2/1")

    def test_simulate_parkinsons_rtms_defaults(self):
        res = simulate_parkinsons_rtms()
        self.assertEqual(res["status"], "success")
        self.assertIn("cortical_lobes", res)
        self.assertEqual(len(res["cortical_lobes"]), 7)

        # Verify BEM simulation data
        bem = res["bem_simulation"]
        self.assertEqual(len(bem["compartments"]), 4)
        self.assertGreater(bem["total_triangles"], 20000)
        self.assertIn("surface_potential_grid", bem)
        self.assertEqual(len(bem["surface_potential_grid"]["potential_mv"]), 25)

        # Verify RF strength characteristics
        rf = res["rf_characteristics"]
        self.assertEqual(rf["carrier_frequency_khz"], 3.2)
        self.assertGreater(rf["peak_bfield_tesla"], 2.0)
        self.assertGreater(rf["peak_efield_v_m"], 140.0)
        self.assertLess(rf["sar_10g_w_kg"], 2.0) # ICNIRP limit
        self.assertGreater(rf["skin_depth_cm"], 10.0)

        # Verify Combinatorial Optimal Staging
        comb = res["combinatorial_optimal_staging"]
        self.assertEqual(len(comb["stages"]), 4)
        for stage in comb["stages"]:
            self.assertIn("bellman_value_v_star", stage)
            self.assertGreater(stage["bellman_value_v_star"], 70.0)

        # Verify Long-Term Horizons
        traj = res["clinical_trajectories"]
        self.assertGreater(len(traj["weeks"]), 50)
        self.assertLess(traj["updrs_qml"][-1], 15.0) # Motor remission
        self.assertEqual(len(traj["milestones_table"]), 4)

        # Verify Finite Math Equations
        eqs = res["finite_math_equations"]
        self.assertGreaterEqual(len(eqs), 6)

    def test_simulate_parkinsons_rtms_parameter_clamping(self):
        res = simulate_parkinsons_rtms(
            baseline_updrs=100.0, # should clamp to 80
            intensity_rmt=200.0,  # should clamp to 130
            rf_carrier_khz=10.0,  # should clamp to 6.0
            pulse_budget_daily=10000, # should clamp to 5000
        )
        self.assertEqual(res["parameters"]["baseline_updrs"], 80.0)
        self.assertEqual(res["parameters"]["stimulation_intensity_rmt"], 130.0)
        self.assertEqual(res["parameters"]["rf_carrier_khz"], 6.0)
        self.assertEqual(res["parameters"]["pulse_budget_daily"], 5000)

    def test_flask_endpoints(self):
        client = app.test_client()
        res = client.get('/api/parkinsons-rtms?baseline_updrs=42&intensity_rmt=110')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")

        # Test PDF endpoint
        res_pdf = client.get('/api/parkinsons-nature-preprint')
        self.assertEqual(res_pdf.status_code, 200)
        self.assertEqual(res_pdf.headers.get("Content-Type"), "application/pdf")
        self.assertGreater(len(res_pdf.data), 100000)


if __name__ == '__main__':
    unittest.main()

