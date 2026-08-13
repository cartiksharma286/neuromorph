import copy
import math
import threading
import time
from datetime import datetime, timezone

import numpy as np


class MarketSimulationEngine:
    def __init__(self, seed=20260813, cache_seconds=2.0):
        self._rng = np.random.default_rng(seed)
        self._cache_seconds = cache_seconds
        self._lock = threading.RLock()
        self._cached_at = 0.0
        self._snapshot = None
        self._state = {
            "sp500": 5400.0,
            "uranium": 86.0,
            "lithium": 14250.0,
            "cobalt": 28600.0,
            "vix": 16.5,
        }
        self._drift = np.array([0.08, 0.12, 0.09, 0.07, 0.0])
        self._volatility = np.array([0.16, 0.31, 0.38, 0.34, 0.42])
        self._correlation = np.array([
            [1.00, 0.32, 0.28, 0.25, -0.62],
            [0.32, 1.00, 0.44, 0.39, -0.10],
            [0.28, 0.44, 1.00, 0.52, -0.08],
            [0.25, 0.39, 0.52, 1.00, -0.06],
            [-0.62, -0.10, -0.08, -0.06, 1.00],
        ])
        self._cholesky = np.linalg.cholesky(self._correlation)

    def snapshot(self, force=False):
        with self._lock:
            now = time.monotonic()
            if not force and self._snapshot and now - self._cached_at < self._cache_seconds:
                cached = copy.deepcopy(self._snapshot)
                cached["cache_hit"] = True
                return cached

            self._advance_state()
            self._snapshot = self._build_snapshot()
            self._cached_at = now
            return copy.deepcopy(self._snapshot)

    def _advance_state(self):
        names = tuple(self._state)
        values = np.array([self._state[name] for name in names], dtype=float)
        dt = 1.0 / (252.0 * 6.5 * 60.0)
        shocks = self._cholesky @ self._rng.standard_normal(len(names))
        values *= np.exp((self._drift - 0.5 * self._volatility ** 2) * dt + self._volatility * math.sqrt(dt) * shocks)
        values[-1] += 0.08 * (16.5 - values[-1])
        values[-1] = np.clip(values[-1], 8.0, 80.0)
        self._state.update(dict(zip(names, values)))

    def _series(self, name, points=60):
        current = self._state[name]
        index = tuple(self._state).index(name)
        annual_vol = self._volatility[index]
        dt = 1.0 / (252.0 * 6.5 * 60.0)
        minute_shocks = self._rng.standard_normal(points - 1)
        returns = (self._drift[index] - 0.5 * annual_vol ** 2) * dt + annual_vol * math.sqrt(dt) * minute_shocks
        historical = current / math.exp(float(np.sum(returns)))
        prices = np.concatenate(([historical], historical * np.exp(np.cumsum(returns))))
        band_width = np.maximum(prices * annual_vol / math.sqrt(252.0), prices * 0.002)
        return [
            {
                "time": f"T-{points - 1 - position:02d}m" if position < points - 1 else "Now",
                "price": round(float(price), 2),
                "upper": round(float(price + 1.96 * width), 2),
                "lower": round(float(max(0.01, price - 1.96 * width)), 2),
            }
            for position, (price, width) in enumerate(zip(prices, band_width))
        ]

    def _build_snapshot(self):
        series = {name: self._series(name) for name in self._state}
        ten_year = 4.15 + self._rng.normal(0.0, 0.015)
        maturities = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
        treasury_offsets = [0.82, 0.74, 0.60, 0.35, 0.12, -0.05, 0.0, 0.18]
        yield_curve = []
        for maturity, offset in zip(maturities, treasury_offsets):
            treasury = ten_year + offset + self._rng.normal(0.0, 0.012)
            credit_spread = 0.82 + 0.05 * maturities.index(maturity)
            yield_curve.append({
                "maturity": maturity,
                "treasury": round(float(treasury), 3),
                "corporate": round(float(treasury + credit_spread), 3),
            })

        return {
            **series,
            "yield_curve": yield_curve,
            "characteristics": {
                "pe_ratio": 24.6,
                "earnings_yield": 4.07,
                "10y_treasury": round(float(ten_year), 3),
                "equity_risk_premium": round(float(4.07 - ten_year), 3),
                "credit_spread": round(yield_curve[6]["corporate"] - yield_curve[6]["treasury"], 3),
                "vix_current": series["vix"][-1]["price"],
            },
            "meta": {
                "sp_target": series["sp500"][-1]["price"],
                "uranium_target": series["uranium"][-1]["price"],
                "lithium_target": series["lithium"][-1]["price"],
                "cobalt_target": series["cobalt"][-1]["price"],
                "vix_target": series["vix"][-1]["price"],
            },
            "source": "correlated geometric-Brownian live simulation",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "refresh_interval_seconds": self._cache_seconds,
            "cache_hit": False,
        }


def optimize_dividend_portfolio(assets, capital=100000.0):
    stocks = [asset.copy() for asset in assets if asset["type"] == "Stock" and asset["yield"] > 0]
    if not stocks:
        return [], 0.0
    scores = np.array([
        max(asset["yield"], 0.0) / max(asset["vol"], 0.05) * (1.0 - min(asset["vol"], 0.8) * 0.35)
        for asset in stocks
    ])
    weights = scores / scores.sum()
    portfolio_yield = 0.0
    result = []
    for asset, weight, score in zip(stocks, weights, scores):
        income = capital * weight * asset["yield"] / 100.0
        portfolio_yield += weight * asset["yield"]
        result.append({
            "ticker": asset["ticker"],
            "dividend_yield": f"{asset['yield']:.1f}% cash dividend",
            "allocation": f"{weight * 100.0:.1f}%",
            "projected_income": f"${income:,.2f} /yr",
            "risk_adjusted_score": round(float(score), 2),
        })
    return result, float(portfolio_yield)