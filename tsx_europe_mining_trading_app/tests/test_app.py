import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import ROOT, app, simulate


class TransatlanticTradingAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_simulation_api_response_structure(self):
        response = self.client.post("/api/simulate", json={"ore_momentum": 0.5, "options_overlay": 0.9})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        # Check top-level keys
        self.assertIn("parameters", data)
        self.assertIn("stieltjes_fractions", data)
        self.assertIn("leibniz_recurrent_primes", data)
        self.assertIn("neural_ode", data)
        self.assertIn("dividends_and_trades", data)

        # Check Stieltjes Continued Fractions
        stieltjes = data["stieltjes_fractions"]
        self.assertTrue(len(stieltjes["coefficients"]) >= 6)
        self.assertTrue(len(stieltjes["convergents"]) >= 4)
        self.assertEqual(len(stieltjes["depth_curve"]), 61)
        self.assertGreater(stieltjes["liquidity_gain_pct"], 30.0)

        # Check Leibniz Recurrent Primes
        primes = data["leibniz_recurrent_primes"]
        self.assertTrue(len(primes["selected_primes"]) >= 4)
        self.assertGreater(len(primes["leibniz_series"]), 10)
        self.assertAlmostEqual(primes["leibniz_limit"], 3.14159, delta=0.25)

        # Check Neural ODE Phase Space
        ode = data["neural_ode"]
        self.assertEqual(len(ode["trajectory"]), 60)
        self.assertEqual(len(ode["phase_space"]), 60)
        self.assertGreater(ode["damping_ratio"], 0.0)

        # Check Dividend Yield > 40%
        divs = data["dividends_and_trades"]
        self.assertGreaterEqual(divs["annual_dividend_yield_pct"], 40.0)
        self.assertTrue(divs["yield_target_met"])
        self.assertGreater(divs["sharpe_ratio"], 2.0)
        self.assertEqual(len(divs["monthly_trajectory"]), 12)
        self.assertTrue(len(divs["trades"]) >= 4)

    def test_parameters_bounding_and_stability(self):
        # Pass out-of-bounds parameters
        payload = simulate({
            "tsx_weight": 5.0,
            "ger_weight": -2.0,
            "ore_momentum": 10.0,
            "volatility": 4.0,
            "gamma_damping": -1.0,
            "prime_order": 100
        })
        p = payload["parameters"]
        self.assertEqual(p["tsx_weight"], 0.9)
        self.assertEqual(p["ger_weight"], 0.1)
        self.assertEqual(p["ore_momentum"], 1.0)
        self.assertEqual(p["volatility"], 0.95)
        self.assertEqual(p["gamma_damping"], 0.05)
        self.assertEqual(p["prime_order"], 20)

    def test_preprint_pdf_endpoint(self):
        response = self.client.get("/preprint.pdf")
        pdf_exists = (Path(ROOT) / "Nature_Preprint_TSX_Frankfurt_Mining_Forex.pdf").exists()
        expected_status = 200 if pdf_exists else 404
        self.assertEqual(response.status_code, expected_status)
        if pdf_exists:
            self.assertEqual(response.mimetype, "application/pdf")
        response.close()

    def test_live_tick_endpoint(self):
        response = self.client.get("/api/live_tick")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("timestamp", data)
        self.assertIn("new_trade", data)
        self.assertIn("tickers", data)
        self.assertIn("copper_usd_lb", data["tickers"])
        self.assertIn("eur_cad_fx", data["tickers"])
        self.assertEqual(data["new_trade"]["status"], "LIVE FILL")


if __name__ == "__main__":
    unittest.main()