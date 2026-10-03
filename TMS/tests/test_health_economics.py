"""Unit Tests for Health Economics, Neural Repair Outcomes, and Corporate Valuation."""

import os
import unittest
from pathlib import Path

from logic.health_economics_valuation import (
    simulate_markov_cohort,
    compute_ceac_curve,
    generate_neuromorph_2year_revenue_projections,
    compute_neuromorph_valuation,
    run_full_health_economics_simulation,
    INDICATIONS
)
from app import app
from generate_economist_health_economics import build_pdf


class TestHealthEconomicsValuation(unittest.TestCase):
    def test_all_indications_present(self):
        for ind in ["dementia", "schizophrenia", "parkinsons", "alexander"]:
            self.assertIn(ind, INDICATIONS)
            spec = INDICATIONS[ind]
            self.assertGreater(len(spec["states"]), 2)
            self.assertEqual(len(spec["states"]), len(spec["utilities"]))
            self.assertEqual(len(spec["states"]), len(spec["annual_cost_soc"]))
            self.assertEqual(len(spec["states"]), len(spec["annual_cost_rtms"]))

    def test_markov_cohort_simulation(self):
        for ind in ["dementia", "schizophrenia", "parkinsons", "alexander"]:
            res = simulate_markov_cohort(ind, months=24)
            self.assertEqual(len(res["history_soc"]), 24)
            self.assertEqual(len(res["history_rtms"]), 24)
            self.assertGreater(res["delta_qaly"], 0.0)
            self.assertIn("icer_per_qaly", res)
            self.assertIn("net_monetary_benefit_100k", res)

            # Check that probabilities sum to 1 at each month
            for pt in res["history_rtms"]:
                self.assertAlmostEqual(sum(pt["distribution"]), 1.0, places=2)

    def test_ceac_curve_computation(self):
        ceac = compute_ceac_curve(delta_cost_mean=5000.0, delta_qaly_mean=0.35, wtp_max=200000.0, steps=25)
        self.assertEqual(len(ceac["wtp_thresholds"]), 25)
        self.assertEqual(len(ceac["probabilities"]), 25)
        # Probability of cost-effectiveness should increase with willingness-to-pay
        self.assertGreaterEqual(ceac["probabilities"][-1], ceac["probabilities"][0])
        self.assertTrue(all(0.0 <= p <= 1.0 for p in ceac["probabilities"]))

    def test_financial_revenue_projections(self):
        fin = generate_neuromorph_2year_revenue_projections(
            system_unit_price=65000.0,
            session_consumable_fee=75.0
        )
        self.assertEqual(len(fin["monthly"]["months"]), 24)
        self.assertEqual(len(fin["quarters"]), 8)
        self.assertGreater(fin["summary"]["total_2year_revenue"], 3000000.0)
        self.assertGreater(fin["summary"]["run_rate_arr_month24"], 1000000.0)
        self.assertGreater(fin["summary"]["final_installed_systems"], 20)

    def test_corporate_valuation(self):
        fin = generate_neuromorph_2year_revenue_projections()
        val = compute_neuromorph_valuation(fin, wacc=0.115)
        self.assertIn("dcf_valuation", val)
        self.assertIn("rnpv_valuation", val)
        self.assertIn("market_multiple_valuation", val)
        self.assertIn("blended_valuation", val)

        self.assertGreater(val["dcf_valuation"]["enterprise_value"], 0.0)
        self.assertGreater(val["rnpv_valuation"]["pipeline_rnpv"], 0.0)
        self.assertGreater(val["blended_valuation"]["enterprise_value"], 20000000.0)
        self.assertGreater(val["blended_valuation"]["price_per_share"], 0.50)

    def test_full_simulation_pipeline(self):
        res = run_full_health_economics_simulation(
            discount_rate=0.035,
            wacc=0.115,
            rtms_price_per_session=280.0
        )
        self.assertEqual(res["status"], "success")
        self.assertIn("markov_cohorts", res)
        self.assertIn("pooled_outcomes", res)
        self.assertIn("ceac_curve", res)
        self.assertIn("financials_2year", res)
        self.assertIn("corporate_valuation", res)

    def test_flask_endpoints(self):
        client = app.test_client()

        # POST /api/health-economics-simulation
        post_resp = client.post(
            "/api/health-economics-simulation",
            json={
                "discount_rate": 0.035,
                "wacc": 0.115,
                "rtms_price_per_session": 280.0,
                "system_capital_price": 65000.0,
                "wtp_threshold": 100000.0
            }
        )
        self.assertEqual(post_resp.status_code, 200)
        post_data = post_resp.get_json()
        self.assertEqual(post_data["status"], "success")
        self.assertIn("corporate_valuation", post_data)

        # GET /api/health-economics-simulation
        get_resp = client.get("/api/health-economics-simulation?discount_rate=0.035&wacc=0.115")
        self.assertEqual(get_resp.status_code, 200)

        # GET /api/health-economics-pdf
        pdf_resp = client.get("/api/health-economics-pdf")
        self.assertEqual(pdf_resp.status_code, 200)
        self.assertEqual(pdf_resp.mimetype, "application/pdf")
        self.assertGreater(len(pdf_resp.data), 40000)

    def test_preprint_pdf_generation(self):
        test_pdf = Path(__file__).parent / "test_economist_health_economics.pdf"
        try:
            out = build_pdf(str(test_pdf))
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 40000)
        finally:
            if test_pdf.exists():
                test_pdf.unlink()


if __name__ == "__main__":
    unittest.main()
