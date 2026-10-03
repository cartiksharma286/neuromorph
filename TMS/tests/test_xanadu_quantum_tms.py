"""Unit Tests for Xanadu Photonic Quantum TMS, Number-Theoretic Fractional States & Neural Repair."""

import os
import unittest
from pathlib import Path

from logic.xanadu_quantum_tms import (
    compute_farey_fractional_states,
    simulate_xanadu_photonic_wigner,
    simulate_quantum_qml_speedup,
    simulate_longterm_clinical_trajectories_xanadu,
    simulate_xanadu_qaas_commercial_economics,
    run_full_xanadu_quantum_tms_simulation,
    XANADU_CHIPSETS
)
from app import app
from generate_nature_xanadu_qml_tms import build_pdf


class TestXanaduQuantumTms(unittest.TestCase):
    def test_chipsets_present(self):
        for key in ["borealis", "x8", "x12"]:
            self.assertIn(key, XANADU_CHIPSETS)
            chip = XANADU_CHIPSETS[key]
            self.assertGreater(chip["qumodes"], 0)
            self.assertGreater(chip["max_squeezing_db"], 5.0)
            self.assertGreater(chip["clock_rate_mhz"], 1.0)
            self.assertIn("cloud_endpoint", chip)

    def test_farey_fractional_states(self):
        states = compute_farey_fractional_states(modulus_q=17, depth=6)
        self.assertEqual(len(states), 6)
        for s in states:
            self.assertIn("fraction", s)
            self.assertIn("quantum_phase_rad", s)
            self.assertIn("dedekind_sum", s)
            self.assertGreaterEqual(s["value"], 0.0)
            self.assertLessEqual(s["value"], 1.0)

    def test_photonic_wigner_simulation(self):
        wig = simulate_xanadu_photonic_wigner(r_squeeze=1.25, theta_rad=0.35, alpha_disp=0.65, grid_res=25)
        self.assertEqual(len(wig["x_coords"]), 25)
        self.assertEqual(len(wig["p_coords"]), 25)
        self.assertEqual(len(wig["wigner_matrix"]), 25)
        self.assertGreater(wig["squeezing_db"], 10.0)
        self.assertGreater(len(wig["fock_photon_probabilities"]), 0)
        self.assertAlmostEqual(sum(wig["fock_photon_probabilities"]), 1.0, places=2)

    def test_qml_speedup_simulation(self):
        speed = simulate_quantum_qml_speedup(iterations=25, n_qumodes=216, squeezing_r=1.25)
        self.assertEqual(len(speed["iterations"]), 25)
        self.assertEqual(len(speed["qml_loss"]), 25)
        self.assertGreater(speed["peak_speedup_factor"], 1.0)
        self.assertGreater(speed["time_reduction_pct"], 50.0)
        self.assertLess(speed["wallclock_quantum_ms"], speed["wallclock_classical_ms"])

    def test_clinical_trajectories_60m(self):
        clin = simulate_longterm_clinical_trajectories_xanadu(horizon_months=60)
        self.assertIn("stroke", clin)
        self.assertIn("dementia", clin)
        self.assertIn("parkinsons", clin)
        self.assertIn("schizophrenia", clin)
        self.assertIn("alexander", clin)
        self.assertEqual(clin["months"][-1], 60)

        # In stroke FMA, higher is better
        self.assertGreater(clin["stroke"]["xanadu_qml_rtms"][-1], clin["stroke"]["soc"][-1])
        # In Parkinson's UPDRS, lower is better
        self.assertLess(clin["parkinsons"]["xanadu_qml_rtms"][-1], clin["parkinsons"]["soc"][-1])

    def test_qaas_commercial_economics(self):
        comm = simulate_xanadu_qaas_commercial_economics(horizon_months=36)
        self.assertEqual(len(comm["monthly"]["months"]), 36)
        self.assertGreater(comm["summary_3yr"]["total_3yr_revenue"], 5000000.0)
        self.assertGreater(comm["summary_3yr"]["forward_arr_m36"], 2000000.0)
        self.assertTrue(comm["health_economics"]["is_dominant_strategy"])
        self.assertLess(comm["health_economics"]["icer_per_qaly"], 0.0)

    def test_full_simulation_pipeline(self):
        res = run_full_xanadu_quantum_tms_simulation(chipset="borealis", squeezing_r=1.25)
        self.assertEqual(res["status"], "success")
        self.assertIn("chipset_specification", res)
        self.assertIn("number_theoretic_states", res)
        self.assertIn("photonic_wigner", res)
        self.assertIn("qml_optimization", res)
        self.assertIn("clinical_trajectories", res)
        self.assertIn("commercial_economics", res)

    def test_flask_endpoints(self):
        client = app.test_client()

        # POST /api/xanadu-quantum-tms
        post_resp = client.post(
            "/api/xanadu-quantum-tms",
            json={
                "chipset": "borealis",
                "squeezing_r": 1.25,
                "displacement_alpha": 0.65,
                "modulus_q": 17,
                "horizon_months": 60
            }
        )
        self.assertEqual(post_resp.status_code, 200)
        post_data = post_resp.get_json()
        self.assertEqual(post_data["status"], "success")
        self.assertIn("qml_optimization", post_data)

        # GET /api/xanadu-quantum-tms
        get_resp = client.get("/api/xanadu-quantum-tms?chipset=borealis&squeezing_r=1.25")
        self.assertEqual(get_resp.status_code, 200)

        # GET /api/xanadu-quantum-tms-pdf
        pdf_resp = client.get("/api/xanadu-quantum-tms-pdf")
        self.assertEqual(pdf_resp.status_code, 200)
        self.assertEqual(pdf_resp.mimetype, "application/pdf")
        self.assertGreater(len(pdf_resp.data), 40000)

    def test_preprint_pdf_generation(self):
        test_pdf = Path(__file__).parent / "test_nature_xanadu_qml.pdf"
        try:
            out = build_pdf(str(test_pdf))
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 40000)
        finally:
            if test_pdf.exists():
                test_pdf.unlink()


if __name__ == "__main__":
    unittest.main()
