import unittest
from app import app
from logic.xanadu_quantum_tms import (
    compute_maximum_dispersion_staging,
    simulate_10yr_xanadu_recovery_revenue,
    run_full_xanadu_quantum_tms_simulation
)
from generate_nature_xanadu_qml_tms import build_pdf


class TestXanaduMaximumDispersion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_01_maximum_dispersion_staging(self):
        disp = compute_maximum_dispersion_staging(num_stages=4, squeezing_r=1.25)
        self.assertEqual(disp["num_stages"], 4)
        th = disp["staging_thresholds"]
        self.assertEqual(len(th), 3)
        self.assertTrue(0.0 < th[0] < th[1] < th[2] < 1.0)
        self.assertGreater(disp["between_stage_dispersion"], 0.05)
        self.assertGreater(disp["dispersion_enhancement_ratio"], 1.0)

        # Wasserstein distance decays over weeks
        w2 = disp["wasserstein_2_distance"]
        self.assertGreater(w2[0], w2[-1])
        self.assertEqual(len(disp["stages"]), 4)
        print(f"[Pass] Maximum Dispersion Staging Verified: Thresholds={th}, D_between={disp['between_stage_dispersion']:.4f}")

    def test_02_ten_year_revenue_projections(self):
        rev = simulate_10yr_xanadu_recovery_revenue(real_discount_rate=0.075)
        self.assertEqual(len(rev["years"]), 10)
        p10 = rev["total_revenue_p10_usd_m"]
        p50 = rev["total_revenue_p50_usd_m"]
        p90 = rev["total_revenue_p90_usd_m"]

        # Quantile ordering
        self.assertLess(p10[-1], p50[-1])
        self.assertLess(p50[-1], p90[-1])

        # Financial metrics
        self.assertGreater(rev["ten_year_npv_usd_m"], 10.0)
        self.assertGreater(rev["ten_year_ebitda_usd_m"], 20.0)
        self.assertLessEqual(rev["payback_year"], 4)
        print(f"[Pass] 10-Yr Revenue Projections Verified: P50 Year 10=${p50[-1]}M, 10-Yr NPV=${rev['ten_year_npv_usd_m']:.2f}M, Payback={rev['payback_year']} yrs")

    def test_03_full_pipeline_simulation(self):
        res = run_full_xanadu_quantum_tms_simulation(chipset="borealis")
        self.assertEqual(res["status"], "success")
        self.assertIn("maximum_dispersion_staging", res)
        self.assertIn("ten_year_revenue_projections", res)
        self.assertIn("clinical_trajectories", res)
        print("[Pass] Full Master Pipeline with Maximum Dispersion Verified")

    def test_04_api_and_pdf(self):
        # API endpoint
        res = self.client.post("/api/xanadu-quantum-tms", json={"chipset": "borealis"})
        self.assertEqual(res.status_code, 200)
        data = res.json
        self.assertIn("maximum_dispersion_staging", data)
        self.assertIn("ten_year_revenue_projections", data)

        # PDF Compilation
        output_path = "/tmp/test_xanadu_preprint.pdf"
        built_path = build_pdf(output_path, {})
        self.assertTrue(built_path.endswith(".pdf"))
        with open(built_path, "rb") as f:
            header = f.read(5)
            self.assertEqual(header, b"%PDF-")
        print(f"[Pass] Nature Preprint PDF Compilation Verified: {built_path}")


if __name__ == '__main__':
    unittest.main()
