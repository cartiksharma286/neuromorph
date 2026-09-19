import unittest
from logic.nih_grant_budget_optimizer import optimize_nih_grant_budget, get_neuromorph_dbs_specs


class TestNIHGrantBudgetOptimizer(unittest.TestCase):
    def test_optimize_nih_grant_budget(self):
        res = optimize_nih_grant_budget(
            total_direct_budget=640000.0,
            fa_rate=0.545,
            n_years=5,
            contingency_reserve_pct=4.0
        )
        self.assertEqual(res["total_direct_budget"], 640000.0)
        self.assertEqual(res["total_indirect"], 348800.0)
        self.assertEqual(res["total_budget"], 988800.0)
        self.assertTrue(res["is_under_1m"])
        self.assertLess(res["total_budget"], 1000000.0)
        self.assertEqual(len(res["categories"]), 7)
        self.assertEqual(len(res["annual_summary"]), 5)
        self.assertEqual(len(res["quarters"]), 20)
        self.assertEqual(len(res["planned_cum_spend"]), 20)
        self.assertGreater(len(res["math_equations"]), 2)
        self.assertIn("dbs_specs", res)
        self.assertGreaterEqual(res["dbs_specs"]["vta_specificity_pct"], 95.0)

    def test_neuromorph_dbs_specs(self):
        specs = get_neuromorph_dbs_specs()
        self.assertEqual(specs["vta_specificity_pct"], 96.8)
        self.assertEqual(specs["battery_longevity_years"], 28.5)
        self.assertLess(specs["closed_loop_latency_ms"], 2.0)
        self.assertLess(specs["power_dissipation_uW"], 15.0)


if __name__ == "__main__":
    unittest.main()
