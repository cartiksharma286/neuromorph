#!/usr/bin/env python3
"""Comprehensive test suite for City of Toronto Community Grants Management Platform.
Validates Quantum Selective Excitation, Pigeonhole Combinatorics, Municipal Projections,
Preprint PDF generation, and Flask API endpoints.
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from quantum_pigeonhole_engine import (
    QuantumSelectiveExcitationEngine,
    PigeonholeCombinatoricsEngine,
    TorontoCommunityGrantOptimizer,
    TORONTO_WARDS,
    GRANT_SECTORS
)
from generate_nature_preprint import generate_nature_preprint_pdf
from app import app

class TestTorontoGrantsPlatform(unittest.TestCase):

    def setUp(self):
        self.optimizer = TorontoCommunityGrantOptimizer()
        self.client = app.test_client()

    def test_quantum_selective_excitation(self):
        engine = QuantumSelectiveExcitationEngine(time_steps=128, duration_ms=6.0)
        pulse = engine.generate_sinc_hermite_pulse(flip_angle_deg=90.0, bandwidth_khz=2.0)
        self.assertEqual(len(pulse), 128)

        profile = engine.compute_excitation_profile(pulse, freq_range_khz=(-4.0, 4.0), n_freqs=50)
        self.assertIn("frequencies", profile)
        self.assertIn("excitation_probability", profile)
        self.assertIn("transverse_magnetization", profile)
        self.assertEqual(len(profile["frequencies"]), 50)

        # Check bounds
        for prob in profile["excitation_probability"]:
            self.assertTrue(0.0 <= prob <= 1.0)
        for mxy in profile["transverse_magnetization"]:
            self.assertTrue(0.0 <= mxy <= 1.0)

    def test_pigeonhole_combinatorics_invariants(self):
        engine = PigeonholeCombinatoricsEngine(n_applicants=1000, k_pigeonholes=6)
        invariants = engine.compute_dirichlet_bounds()
        self.assertEqual(invariants["total_applicants_n"], 1000)
        self.assertEqual(invariants["impact_bins_k"], 6)
        self.assertEqual(invariants["pigeonhole_minimum_peak"], 167) # ceil(1000 / 6) = 167
        self.assertEqual(invariants["guaranteed_collisions"], 994) # 1000 - 6

        weights = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
        caps = [500] * 6
        partition = engine.partition_with_resonance(weights, caps)
        self.assertEqual(len(partition), 6)
        self.assertEqual(sum(partition), 1000)
        # Every bin has at least 1 applicant
        for count in partition:
            self.assertGreaterEqual(count, 1)

    def test_toronto_allocation_engine(self):
        res = self.optimizer.run_full_allocation(
            total_budget_cad=100_000_000.0,
            flip_angle_deg=90.0,
            bandwidth_khz=2.5,
            total_applicants=1200,
            projection_years=5
        )

        self.assertIn("summary", res)
        self.assertIn("sector_allocations", res)
        self.assertIn("ward_allocations", res)
        self.assertIn("yearly_projections", res)

        summary = res["summary"]
        self.assertEqual(summary["total_budget_allocated_cad"], 100_000_000.0)
        self.assertEqual(summary["total_grants_awarded"], 1200)
        self.assertGreater(summary["5yr_cumulative_municipal_yield_cad"], 100_000_000.0)
        self.assertGreater(summary["5yr_net_municipal_roi_pct"], 0.0)
        self.assertEqual(len(res["ward_allocations"]), 25)
        self.assertEqual(len(res["sector_allocations"]), 6)
        self.assertEqual(len(res["yearly_projections"]), 5)

        # Check job margin fields
        self.assertIn("5yr_total_citywide_jobs_created", summary)
        self.assertGreater(summary["5yr_total_citywide_jobs_created"], 5000)
        self.assertIn("total_new_jobs_created", res["yearly_projections"][0])
        self.assertIn("cumulative_total_jobs", res["yearly_projections"][0])

        # Check biotech corridor
        self.assertIn("biotech_corridor", res)
        self.assertEqual(len(res["biotech_corridor"]["hubs"]), 6)
        self.assertGreater(res["biotech_corridor"]["summary"]["five_year_total_stem_jobs"], 5000)

    def test_nature_preprint_pdf_generation(self):
        test_pdf = os.path.join(BASE_DIR, "tests", "test_preprint.pdf")
        if os.path.exists(test_pdf):
            os.remove(test_pdf)
        pdf_path = generate_nature_preprint_pdf(test_pdf)
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 5000)
        if os.path.exists(test_pdf):
            os.remove(test_pdf)

    def test_flask_endpoints(self):
        # GET /
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"City of Toronto", resp.data)

        # GET /api/wards
        resp_wards = self.client.get("/api/wards")
        self.assertEqual(resp_wards.status_code, 200)
        wards_data = resp_wards.get_json()
        self.assertEqual(len(wards_data), 25)

        # GET /api/sectors
        resp_sec = self.client.get("/api/sectors")
        self.assertEqual(resp_sec.status_code, 200)
        sec_data = resp_sec.get_json()
        self.assertEqual(len(sec_data), 6)

        # POST /api/allocate
        resp_alloc = self.client.post("/api/allocate", json={
            "budget": 150000000,
            "flip_angle": 90,
            "bandwidth": 2.4,
            "applicants": 1600,
            "years": 5
        })
        self.assertEqual(resp_alloc.status_code, 200)
        alloc_data = resp_alloc.get_json()
        self.assertEqual(alloc_data["summary"]["total_budget_allocated_cad"], 150000000)

        # POST /api/auto_optimize
        resp_auto = self.client.post("/api/auto_optimize", json={
            "budget": 150000000,
            "applicants": 1750
        })
        self.assertEqual(resp_auto.status_code, 200)
        auto_data = resp_auto.get_json()
        self.assertIn("optimal_parameters", auto_data)

        # GET /download_preprint
        resp_pdf = self.client.get("/download_preprint")
        self.assertEqual(resp_pdf.status_code, 200)
        self.assertEqual(resp_pdf.content_type, "application/pdf")

if __name__ == "__main__":
    unittest.main()
