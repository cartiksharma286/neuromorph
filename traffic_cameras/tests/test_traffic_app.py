"""Unit tests for Quantum-Statistical Traffic Signal Intelligence & Red Light Camera App."""

import unittest
from backend.data_loader import load_cameras_dataset
from backend.quantum_observables import QuantumTrafficObservables, get_camera_quantum_state
from backend.statistical_ml import StatisticalViolationModel
from backend.mitigation_engine import evaluate_mitigation_policy
from backend.app import app


class TestTrafficApp(unittest.TestCase):

    def setUp(self):
        self.cameras = load_cameras_dataset()
        self.client = app.test_client()

    def test_data_loader(self):
        self.assertEqual(len(self.cameras), 301)
        first = self.cameras[0]
        self.assertIn("id", first)
        self.assertIn("latitude", first)
        self.assertIn("longitude", first)
        self.assertIn("speed_85th_kmh", first)
        self.assertIn("dilemma_zone_gap_m", first)
        self.assertIn("monthly_violations", first)
        self.assertIn("risk_tier", first)
        self.assertGreater(first["latitude"], 43.5)
        self.assertLess(first["latitude"], 44.0)

    def test_quantum_observables(self):
        cam = self.cameras[0]
        q = get_camera_quantum_state(cam)
        self.assertIn("expected_violation_operator", q)
        self.assertIn("von_neumann_entropy", q)
        self.assertIn("state_purity", q)
        self.assertIn("quantum_fisher_information", q)
        self.assertIn("heisenberg_satisfied", q)
        self.assertTrue(q["heisenberg_satisfied"])
        self.assertGreater(q["uncertainty_product"], q["heisenberg_bound"])

        qo = QuantumTrafficObservables()
        vqe = qo.simulate_vqe_convergence(epochs=15)
        self.assertEqual(len(vqe["epochs"]), 15)
        self.assertLess(vqe["loss_history"][-1], vqe["loss_history"][0])

    def test_statistical_glm_and_clustering(self):
        model = StatisticalViolationModel(self.cameras)

        # GLM
        glm = model.fit_negative_binomial_glm()
        self.assertIn("coefficients", glm)
        self.assertGreater(glm["pseudo_r2"], 0.70)
        self.assertGreater(glm["iterations"], 0)

        # EVT
        evt = model.fit_extreme_value_tail()
        self.assertIn("gpd_shape_parameter_xi", evt)
        self.assertEqual(len(evt["return_levels_breach_seconds"]), 7)

        # GMM
        gmm = model.fit_spatial_gmm_clusters(n_clusters=4)
        self.assertEqual(len(gmm["clusters"]), 4)
        total_clustered = sum(cl["camera_count"] for cl in gmm["clusters"])
        self.assertEqual(total_clustered, 301)

        # MCMC
        mcmc = model.simulate_mcmc_speed_posterior(n_samples=200)
        self.assertLess(mcmc["gelman_rubin_r_hat"], 1.10)

    def test_mitigation_engine(self):
        policy = evaluate_mitigation_policy(self.cameras, yellow_extension_s=0.6, all_red_extension_s=0.4)
        self.assertGreater(policy["overall_reduction_pct"], 30.0)
        self.assertGreater(policy["annual_crashes_prevented"], 500)
        self.assertGreater(policy["annual_economic_savings_cad"], 50000000)

    def test_flask_endpoints(self):
        # GET /
        res_home = self.client.get("/")
        self.assertEqual(res_home.status_code, 200)
        self.assertIn(b"Quantum-Statistical Traffic Signal Intelligence", res_home.data)

        # GET /api/cameras
        res_cams = self.client.get("/api/cameras")
        self.assertEqual(res_cams.status_code, 200)
        data = res_cams.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["total_count"], 301)

        # Filtered /api/cameras
        res_filt = self.client.get("/api/cameras?district=North%20York")
        self.assertEqual(res_filt.status_code, 200)
        self.assertGreater(res_filt.get_json()["total_count"], 0)

        # GET /api/convergence
        res_conv = self.client.get("/api/convergence")
        self.assertEqual(res_conv.status_code, 200)

        # GET /api/extreme-values
        res_evt = self.client.get("/api/extreme-values")
        self.assertEqual(res_evt.status_code, 200)

        # GET /api/clusters
        res_clust = self.client.get("/api/clusters")
        self.assertEqual(res_clust.status_code, 200)

        # POST /api/simulate
        res_sim = self.client.post("/api/simulate", json={
            "yellow_extension_s": 0.8,
            "all_red_extension_s": 0.5,
            "flow_multiplier": 1.1,
            "weather_condition": "rain"
        })
        self.assertEqual(res_sim.status_code, 200)
        sim_data = res_sim.get_json()
        self.assertEqual(sim_data["status"], "success")


if __name__ == "__main__":
    unittest.main()
