import unittest
from app import app
from logic.us_health_economics import (
    compute_us_reimbursement_schedule,
    compute_us_cost_effectiveness,
    simulate_us_clinic_revenue,
    run_full_us_health_economics_simulation,
    US_REGIONS,
    CLINICAL_INDICATIONS
)
from generate_nature_us_health_economics import build_us_health_economics_pdf


class TestUSHealthEconomics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_01_cpt_reimbursement_gpci(self):
        nat = compute_us_reimbursement_schedule(region_key="national_avg")
        cal = compute_us_reimbursement_schedule(region_key="california_west")

        # California has higher PE GPCI, so commercial and medicare allowable should be higher
        self.assertGreater(cal["commercial_course_revenue_usd"], nat["commercial_course_revenue_usd"])
        self.assertGreater(cal["blended_course_revenue_usd"], nat["blended_course_revenue_usd"])
        self.assertIn("90867", nat["cpt_codes"])
        self.assertIn("90868", nat["cpt_codes"])
        self.assertIn("90869", nat["cpt_codes"])
        self.assertGreater(nat["blended_course_revenue_usd"], 5000.0)
        print(f"[Pass] CPT Reimbursement GPCI Verified: Nat=${nat['blended_course_revenue_usd']:.2f}, Cal=${cal['blended_course_revenue_usd']:.2f}")

    def test_02_cost_effectiveness_icer(self):
        res = compute_us_cost_effectiveness(indication_key="depression_trd")
        self.assertGreater(res["incremental_qaly_gained"], 0.20)
        self.assertGreater(res["annual_medical_savings_per_patient_usd"], 5000.0)
        self.assertLess(res["icer_usd_per_qaly"], 0.0)  # Dominant strategy
        self.assertEqual(len(res["ceac_wtp_axis"]), len(res["ceac_probabilities"]))
        self.assertGreater(res["ceac_probabilities"][-1], 0.95)
        print(f"[Pass] Cost-Effectiveness Verified: Delta Q={res['incremental_qaly_gained']:.2f} QALYs, Savings=${res['annual_medical_savings_per_patient_usd']:.0f}/yr, ICER={res['icer_usd_per_qaly']:.0f}/QALY")

    def test_03_monte_carlo_revenue_quantiles(self):
        res = simulate_us_clinic_revenue(num_tms_chairs=2, region_key="national_avg", annual_referrals_mean=240.0)
        dist = res["statistical_revenue_distribution"]
        fin = res["financial_metrics"]

        # Quantile ordering
        self.assertLess(dist["p10_annual_revenue_usd_m"], dist["p50_annual_revenue_usd_m"])
        self.assertLess(dist["p50_annual_revenue_usd_m"], dist["p90_annual_revenue_usd_m"])

        # Financial metrics
        self.assertGreater(fin["ten_year_npv_p50_usd_m"], 0.5)
        self.assertGreater(fin["benefit_cost_ratio"], 1.5)
        self.assertGreater(fin["internal_rate_of_return_irr"], 0.10)
        self.assertEqual(len(res["dcf_schedule"]), 11)  # Years 0 to 10
        print(f"[Pass] Monte Carlo Revenue Verified: P10=${dist['p10_annual_revenue_usd_m']}M, P50=${dist['p50_annual_revenue_usd_m']}M, P90=${dist['p90_annual_revenue_usd_m']}M, 10-Yr NPV=${fin['ten_year_npv_p50_usd_m']}M")

    def test_04_api_simulation_and_defaults(self):
        # Defaults endpoint
        res_def = self.client.get("/api/us-health-economics-defaults")
        self.assertEqual(res_def.status_code, 200)
        self.assertIn("regions", res_def.json)
        self.assertIn("indications", res_def.json)

        # Simulation endpoint
        payload = {
            "num_tms_chairs": 3,
            "region_key": "northeast_ny",
            "indication_key": "addiction_sud"
        }
        res_sim = self.client.post("/api/us-health-economics-simulation", json=payload)
        self.assertEqual(res_sim.status_code, 200)
        data = res_sim.json
        self.assertEqual(data["status"], "success")
        self.assertIn("summary_kpis", data)
        self.assertIn("revenue_simulation", data)
        self.assertIn("cost_effectiveness", data)
        print(f"[Pass] API Simulation Verified: Status={data['status']}, Indication={data['indication_key']}")

    def test_05_pdf_compilation(self):
        output_path = "/tmp/test_us_health_economics.pdf"
        built_path = build_us_health_economics_pdf(output_path, {})
        self.assertTrue(built_path.endswith(".pdf"))
        with open(built_path, 'rb') as f:
            header = f.read(5)
            self.assertEqual(header, b"%PDF-")
        print(f"[Pass] Nature PDF Compilation Verified: {built_path}")


if __name__ == '__main__':
    unittest.main()
