#!/usr/bin/env python3
"""Unit test suite for Heart and Stroke Long-Term Clinical Trajectories Suite.
Verifies Ramanujan Modular Nome States, Stem Cell Ring Algebra, Game-Theoretic Nash Gap,
Preprint PDF generation, and Flask API endpoints.
"""

import os
import sys
import unittest
from pathlib import Path

# Add TMS directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from logic.heart_stroke_trajectories import (
    compute_ramanujan_modular_nome,
    StemCellRingAlgebra,
    HeartStrokeGameTheoreticEngine,
    HeartStrokeClinicalTrajectoryOptimizer
)
from generate_nature_heart_stroke_preprint import build_pdf
from app import app


class TestHeartStrokeSuite(unittest.TestCase):

    def setUp(self):
        self.optimizer = HeartStrokeClinicalTrajectoryOptimizer()
        self.client = app.test_client()

    def test_ramanujan_modular_nome(self):
        res = compute_ramanujan_modular_nome(tau_real=0.25, tau_imag=0.85)
        self.assertIn("nome_abs", res)
        self.assertIn("dedekind_eta", res)
        self.assertIn("theta_3", res)
        self.assertIn("eisenstein_e4", res)
        self.assertIn("eisenstein_e6", res)
        self.assertIn("ramanujan_delta", res)
        self.assertIn("poincare_potential", res)
        self.assertGreater(res["nome_abs"], 0.0)
        self.assertLess(res["nome_abs"], 1.0)

    def test_stem_cell_ring_algebra(self):
        ring = StemCellRingAlgebra(lambda_0=1.45, lambda_1=2.10)
        # e0 is identity: e0 * e1 = e1
        e0 = [1.0, 0.0, 0.0, 0.0]
        e1 = [0.0, 1.0, 0.0, 0.0]
        prod01 = ring.multiply(e0, e1)
        self.assertAlmostEqual(prod01[1], 1.0)
        self.assertAlmostEqual(prod01[0], 0.0)

        # e2 * e2 = e4 = lambda_0 * e0 + lambda_1 * e2
        e2 = [0.0, 0.0, 1.0, 0.0]
        prod22 = ring.multiply(e2, e2)
        self.assertAlmostEqual(prod22[0], 1.45)
        self.assertAlmostEqual(prod22[2], 2.10)

        # Homomorphism
        homo = ring.compute_regeneration_homomorphism([0.4, 0.3, 0.2, 0.1], scar_factor=0.35)
        self.assertIn("tissue_homomorphism_yield_pct", homo)
        self.assertGreater(homo["tissue_homomorphism_yield_pct"], 50.0)

    def test_nash_game_verification(self):
        game = HeartStrokeGameTheoreticEngine()
        res = game.compute_nash_equilibrium_and_gap(intervention_intensity=1.15)
        self.assertIn("terminal_nash_gap", res)
        self.assertIn("nash_gap_convergence_trace", res)
        self.assertEqual(len(res["nash_gap_convergence_trace"]), 12)
        # Check monotonic or asymptotic vanishing gap
        self.assertLess(res["terminal_nash_gap"], 0.05)
        self.assertTrue(res["equilibrium_stability_certified"])

    def test_5year_clinical_trajectories(self):
        res = self.optimizer.simulate_longterm_trajectories(
            baseline_nihss=16.5,
            baseline_lvef=35.0,
            tau_imag_y=0.85,
            tau_real_x=0.25,
            horizon_months=60
        )
        self.assertIn("summary", res)
        self.assertIn("trajectory_milestones", res)
        self.assertIn("stem_cell_ring_algebra", res)
        self.assertIn("governing_formulations", res)

        s = res["summary"]
        self.assertEqual(s["baseline_nihss"], 16.5)
        self.assertLess(s["final_5yr_nihss"], 5.0) # sub-5 remitted deficit
        self.assertGreater(s["final_5yr_lvef_pct"], 50.0) # normal LVEF
        self.assertGreater(s["penumbra_tissue_salvage_pct"], 80.0)
        self.assertLess(s["vanishing_nash_gap"], 0.05)

        milestones = res["trajectory_milestones"]
        self.assertEqual(milestones[0]["month"], 0)
        self.assertEqual(milestones[-1]["month"], 60)

    def test_nature_preprint_generation(self):
        test_pdf = os.path.join(BASE_DIR, "tests", "test_heart_stroke_preprint.pdf")
        if os.path.exists(test_pdf):
            os.remove(test_pdf)
        pdf_path = build_pdf(test_pdf)
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 50000) # substantial ReportLab + Matplotlib PDF
        if os.path.exists(test_pdf):
            os.remove(test_pdf)

    def test_flask_endpoints(self):
        # GET /api/heart-stroke-trajectories
        res_get = self.client.get("/api/heart-stroke-trajectories?baseline_nihss=18.0&baseline_lvef=32.0")
        self.assertEqual(res_get.status_code, 200)
        data_get = res_get.get_json()
        self.assertEqual(data_get["status"], "success")
        self.assertEqual(data_get["data"]["summary"]["baseline_nihss"], 18.0)

        # POST /api/heart-stroke-trajectories
        res_post = self.client.post("/api/heart-stroke-trajectories", json={
            "baseline_nihss": 14.0,
            "baseline_lvef": 40.0,
            "tau_imag_y": 0.90,
            "tau_real_x": 0.20,
            "horizon_months": 60
        })
        self.assertEqual(res_post.status_code, 200)
        data_post = res_post.get_json()
        self.assertEqual(data_post["status"], "success")
        self.assertEqual(data_post["data"]["summary"]["baseline_nihss"], 14.0)

        # GET /api/heart-stroke-nature-preprint
        res_pdf = self.client.get("/api/heart-stroke-nature-preprint")
        self.assertEqual(res_pdf.status_code, 200)
        self.assertEqual(res_pdf.content_type, "application/pdf")


if __name__ == "__main__":
    unittest.main()
