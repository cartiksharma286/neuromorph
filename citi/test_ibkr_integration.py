import math
import unittest
from unittest.mock import patch

import numpy as np

from app import app
from ibkr_bridge import IBKRBridge, IBKRUnavailableError, calculate_volatility_signal
from market_engine import MarketSimulationEngine, _feynman_path_integral_yield, optimize_dividend_portfolio


class FakeIBKRBridge:
    trading_mode = "paper"

    def status(self):
        return {
            "connected": True,
            "host": "127.0.0.1",
            "port": 7497,
            "client_id": 17,
            "trading_mode": "paper",
            "market_data_type": 3,
            "accounts": ["DU123456"],
        }

    def connect(self):
        return self.status()

    def stock_feed(self, symbols):
        return [{
            "ticker": symbols[0],
            "price": 201.25,
            "bid": 201.2,
            "ask": 201.3,
            "change_pct": 0.8,
            "currency": "USD",
            "exchange": "NASDAQ",
            "signal": "BUY",
            "signal_score": 74.5,
            "realized_volatility": 0.22,
            "realized_volatility_pct": 22.0,
            "sma_20": 198.0,
            "sma_50": 192.0,
            "momentum_20_pct": 4.1,
            "method": "test fixture",
        }]

    def place_market_order(self, symbol, action, quantity, confirmed):
        if confirmed is not True:
            raise ValueError("confirmed=true is required to transmit an order")
        return {
            "order_id": 42,
            "perm_id": 84,
            "status": "Submitted",
            "ticker": symbol,
            "action": action,
            "quantity": quantity,
            "filled": 0.0,
            "remaining": float(quantity),
            "average_fill_price": 0.0,
            "trading_mode": "paper",
            "message": "Order transmitted to fake broker.",
        }


class IBKRIntegrationTests(unittest.TestCase):
    def test_connect_initializes_event_loop_before_import(self):
        bridge = IBKRBridge()
        calls = []

        with patch.object(bridge, "_ensure_event_loop", side_effect=lambda: calls.append("loop")), \
                patch.object(bridge, "_imports", side_effect=IBKRUnavailableError("stop")):
            with self.assertRaisesRegex(IBKRUnavailableError, "stop"):
                bridge.connect()

        self.assertEqual(calls, ["loop"])

    def test_market_snapshot_has_complete_cached_series(self):
        engine = MarketSimulationEngine(seed=7)
        first = engine.snapshot()
        second = engine.snapshot()
        for name in ("sp500", "uranium", "lithium", "cobalt", "vix"):
            self.assertEqual(len(first[name]), 60)
            self.assertTrue(all({"price", "upper", "lower"} <= point.keys() for point in first[name]))
        self.assertTrue(second["cache_hit"])

    def test_dividend_optimizer_normalizes_allocations(self):
        portfolio, portfolio_yield = optimize_dividend_portfolio([
            {"ticker": "RIO", "type": "Stock", "yield": 7.2, "vol": 0.25},
            {"ticker": "BHP", "type": "Stock", "yield": 6.8, "vol": 0.22},
        ])
        allocation = sum(float(item["allocation"].rstrip("%")) for item in portfolio)
        self.assertAlmostEqual(allocation, 100.0, delta=0.2)
        self.assertGreater(portfolio_yield, 0.0)
        self.assertIn("path_integral_yield", portfolio[0])

    def test_feynman_path_integral_yield_matches_classical_limit(self):
        # With zero drift, E[S_t] = 1 for all t regardless of volatility, so the
        # path-integral estimator must converge to the base yield exactly.
        rng = np.random.default_rng(42)
        result = _feynman_path_integral_yield(7.2, volatility=0.25, drift=0.0,
                                               n_paths=20000, rng=rng)
        self.assertAlmostEqual(result, 7.2, delta=0.05)

    def test_feynman_path_integral_yield_matches_closed_form_drift(self):
        # E[(1/T) integral S_t dt] has the closed form (exp(mu*T)-1)/(mu*T) for GBM.
        rng = np.random.default_rng(7)
        drift, horizon = 0.10, 0.25
        result = _feynman_path_integral_yield(10.0, volatility=0.2, drift=drift,
                                               horizon_years=horizon, n_paths=20000, rng=rng)
        closed_form = 10.0 * (math.exp(drift * horizon) - 1.0) / (drift * horizon)
        self.assertAlmostEqual(result, closed_form, delta=0.05)

    def test_signal_uses_price_history(self):
        closes = 100.0 * np.exp(np.linspace(0.0, 0.12, 60))
        result = calculate_volatility_signal(closes)
        self.assertEqual(result["signal"], "BUY")
        self.assertGreater(result["sma_20"], result["sma_50"])
        self.assertGreater(result["momentum_20_pct"], 0.0)

    def test_signal_rejects_short_history(self):
        with self.assertRaisesRegex(ValueError, "at least 21"):
            calculate_volatility_signal([100.0] * 20)

    @patch("app.ibkr", new_callable=FakeIBKRBridge)
    def test_feed_and_order_routes_use_broker_results(self, _fake_bridge):
        with app.test_client() as client:
            feed_response = client.get("/api/ibkr/feed?symbols=AAPL")
            self.assertEqual(feed_response.status_code, 200)
            self.assertEqual(feed_response.get_json()["feed"][0]["price"], 201.25)

            rejected = client.post("/api/trade/place", json={
                "ticker": "AAPL", "action": "BUY", "quantity": 1,
            })
            self.assertEqual(rejected.status_code, 400)

            submitted = client.post("/api/trade/place", json={
                "ticker": "AAPL", "action": "BUY", "quantity": 1, "confirmed": True,
            })
            payload = submitted.get_json()
            self.assertEqual(submitted.status_code, 200)
            self.assertEqual(payload["status"], "Submitted")
            self.assertEqual(payload["filled"], 0.0)
            self.assertEqual(payload["trading_mode"], "paper")


if __name__ == "__main__":
    unittest.main()