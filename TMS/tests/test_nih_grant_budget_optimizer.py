import unittest
from logic.nih_grant_budget_optimizer import optimize_nih_grant_budget


class TestNIHGrantBudgetOptimizer(unittest.TestCase):
    def test_optimize_nih_grant_budget(self):
        res = optimize_nih_grant_budget(
            total_direct_budget=2500000.0,
            fa_rate=0.545,
            n_years=5,
            contingency_reserve_pct=5.0
        )
        self.assertEqual(res["total_direct_budget"], 2500000.0)
        self.assertEqual(res["total_indirect"], 2500000.0 * 0.545)
        self.assertEqual(res["total_budget"], 3862500.0)
        self.assertEqual(len(res["categories"]), 7)
        self.assertEqual(len(res["annual_summary"]), 5)
        self.assertEqual(len(res["quarters"]), 20)
        self.assertEqual(len(res["planned_cum_spend"]), 20)
        self.assertGreater(len(res["math_equations"]), 2)


if __name__ == "__main__":
    unittest.main()
