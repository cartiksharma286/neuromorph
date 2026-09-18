import unittest
from logic.clinical_trajectories_benchmark import (
    simulate_clinical_trajectories_benchmark,
    get_cf_expansion
)


class TestClinicalTrajectoriesBenchmark(unittest.TestCase):
    def test_get_cf_expansion(self):
        coeffs, convergents = get_cf_expansion(1.6180339887, depth=5)
        self.assertGreaterEqual(len(coeffs), 1)
        self.assertGreaterEqual(len(convergents), 1)
        self.assertIn("fraction", convergents[0])

    def test_simulate_clinical_trajectories_benchmark(self):
        res = simulate_clinical_trajectories_benchmark(
            horizon_variant="neurodegenerative",
            horizon_weeks=52,
            galois_modulus=17,
            closed_loop_gain=1.85,
            ramanujan_depth=6
        )
        self.assertIn("trajectories", res)
        self.assertIn("neuromorph_dbs", res["trajectories"])
        self.assertIn("medtronic_percept", res["trajectories"])
        self.assertIn("boston_sci_vercise", res["trajectories"])
        self.assertIn("brainsway_dtms", res["trajectories"])
        self.assertIn("benchmark_table", res)
        self.assertGreaterEqual(len(res["benchmark_table"]), 7)
        self.assertIn("kpis", res)
        self.assertGreater(res["kpis"]["neuromorph_remission_rate_pct"], 85.0)
        self.assertIn("finite_states", res)
        self.assertEqual(len(res["finite_states"]["galois_states"]), 53)


if __name__ == "__main__":
    unittest.main()
