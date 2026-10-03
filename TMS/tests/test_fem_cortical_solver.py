"""Unit Tests for Finite Element Cortical Surface Simulation & Stieltjes Continued Fractions."""

import os
import unittest
from pathlib import Path

from logic.fem_cortical_solver import (
    compute_stieltjes_continued_fraction,
    generate_parkinsonian_cortical_mesh,
    solve_quasistatic_fem_field,
    simulate_gcp_cloud_hyperpartitioning,
    run_full_cortical_fem_simulation,
)
from app import app
from generate_nature_fem_cortical_preprint import build_pdf
import numpy as np


class TestFemCorticalSolver(unittest.TestCase):
    def test_stieltjes_continued_fraction(self):
        x = np.linspace(-30.0, 30.0, 30)
        y = 5.0 * np.cos(0.15 * x) - 10.0 * np.exp(-0.02 * (x**2))
        res = compute_stieltjes_continued_fraction(x, y, depth=6)

        self.assertIn("c_coeffs", res)
        self.assertIn("convergents", res)
        self.assertGreater(len(res["c_coeffs"]), 0)
        self.assertTrue(all(c > 0 for c in res["c_coeffs"]))
        self.assertGreater(res["r2_score"], 0.85)
        self.assertLess(res["stieltjes_mse"], 5.0)

    def test_parkinsonian_cortical_mesh(self):
        # Stage 1 (Mild) vs Stage 5 (Severe)
        mesh_mild = generate_parkinsonian_cortical_mesh(hoehn_yahr_stage=1, baseline_updrs=20.0, atrophy_level="mild")
        mesh_severe = generate_parkinsonian_cortical_mesh(hoehn_yahr_stage=5, baseline_updrs=65.0, atrophy_level="severe")

        self.assertGreater(mesh_mild["cortical_thickness_mm"], mesh_severe["cortical_thickness_mm"])
        self.assertLess(mesh_mild["sulcal_widening_factor"], mesh_severe["sulcal_widening_factor"])
        self.assertEqual(len(mesh_mild["x"]), mesh_mild["num_vertices"])
        self.assertGreater(mesh_mild["num_elements"], 0)
        self.assertEqual(len(mesh_mild["faces_i"]), mesh_mild["num_elements"])

    def test_quasistatic_fem_field(self):
        mesh = generate_parkinsonian_cortical_mesh(hoehn_yahr_stage=3, baseline_updrs=42.0)
        fem = solve_quasistatic_fem_field(mesh, intensity_mso_pct=110.0, coil_orientation_deg=45.0)

        self.assertGreater(fem["e_max_V_per_m"], 80.0)
        self.assertGreater(fem["m1_hand_knob_e_V_per_m"], 50.0)
        self.assertGreater(fem["volume_activated_pct"], 0.0)
        self.assertEqual(len(fem["e_field"]), mesh["num_vertices"])
        self.assertEqual(len(fem["current_density"]), mesh["num_vertices"])

    def test_gcp_cloud_hyperpartitioning(self):
        # Test across node counts and reduction mechanisms
        for nodes in [4, 16]:
            for mech in ["aca_low_rank", "krylov_pod", "guyan_schur"]:
                cloud = simulate_gcp_cloud_hyperpartitioning(
                    num_nodes=nodes,
                    num_elements=2450,
                    matrix_reduction_type=mech,
                    reduction_tolerance=1e-4
                )
                summary = cloud["gcp_cluster_summary"]
                self.assertEqual(summary["active_nodes"], nodes)
                self.assertGreater(summary["cloud_speedup_factor"], 1.0)
                self.assertGreater(cloud["matrix_reduction"]["compression_ratio_pct"], 50.0)
                self.assertGreater(len(cloud["sample_nodes"]), 0)
                self.assertGreater(len(cloud["solver_convergence"]), 0)

    def test_full_simulation_pipeline(self):
        res = run_full_cortical_fem_simulation(
            hoehn_yahr_stage=3,
            baseline_updrs=42.0,
            atrophy_level="moderate",
            intensity_mso_pct=110.0,
            num_gcp_nodes=16,
            matrix_reduction_type="aca_low_rank"
        )
        self.assertEqual(res["status"], "success")
        self.assertIn("patient_pathology", res)
        self.assertIn("mesh", res)
        self.assertIn("stieltjes_interpolation", res)
        self.assertIn("fem_field", res)
        self.assertIn("gcp_cloud_computation", res)

    def test_flask_api_endpoints(self):
        client = app.test_client()

        # Test POST /api/fem-cortical-simulation
        post_resp = client.post(
            "/api/fem-cortical-simulation",
            json={
                "hoehn_yahr_stage": 3,
                "baseline_updrs": 42.0,
                "atrophy_level": "moderate",
                "intensity_mso_pct": 110.0,
                "num_gcp_nodes": 16
            }
        )
        self.assertEqual(post_resp.status_code, 200)
        post_data = post_resp.get_json()
        self.assertEqual(post_data["status"], "success")
        self.assertIn("fem_field", post_data)

        # Test GET /api/fem-cortical-preprint
        pdf_resp = client.get("/api/fem-cortical-preprint")
        self.assertEqual(pdf_resp.status_code, 200)
        self.assertEqual(pdf_resp.mimetype, "application/pdf")
        self.assertGreater(len(pdf_resp.data), 50000)

    def test_preprint_pdf_generation(self):
        test_pdf = Path(__file__).parent / "test_fem_cortical_preprint.pdf"
        try:
            out = build_pdf(str(test_pdf))
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 50000)
        finally:
            if test_pdf.exists():
                test_pdf.unlink()


if __name__ == "__main__":
    unittest.main()
