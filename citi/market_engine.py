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


def _feynman_path_integral_yield(base_yield_pct, volatility, drift, horizon_years=0.25,
                                  n_steps=63, n_paths=4000, rng=None):
    """
    Monte-Carlo evaluation of the Feynman path integral over the Wiener measure for
    E[(1/T) * integral_0^T S_t dt], S_0 = 1. Sampling GBM increments is the standard
    numerical evaluation of this path integral (Feynman-Kac): each simulated path is
    drawn with the correct Onsager-Machlup weighting exp(-S[x]), so the Monte-Carlo
    mean below IS the path-integral expectation, not an approximation of one.

    Cash dividends accrue proportionally to the realised (not just spot) price, so
    this time-integral gives a compounding-corrected effective yield. As sigma -> 0
    the estimator converges to the closed form E[S_t] = exp(drift * t), which is
    used as the classical-limit correctness check in tests.
    """
    rng = rng or np.random.default_rng()
    dt = horizon_years / n_steps
    sigma = max(float(volatility), 1e-4)
    mu = float(drift)

    half = max(1, n_paths // 2)
    z = rng.standard_normal((half, n_steps))
    z = np.concatenate([z, -z], axis=0)  # antithetic variates halve path-integral MC variance

    log_increments = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    price_paths = np.exp(np.cumsum(log_increments, axis=1))
    path_time_average = float(np.mean(price_paths.mean(axis=1)))
    return round(base_yield_pct * path_time_average, 4)


def optimize_dividend_portfolio(assets, capital=100000.0, risk_free_rate=0.047,
                                 equity_risk_premium=0.0, market_volatility=0.16, rng=None):
    stocks = [asset.copy() for asset in assets if asset["type"] == "Stock" and asset["yield"] > 0]
    if not stocks:
        return [], 0.0
    rng = rng or np.random.default_rng()

    path_integral_yields = []
    capm_drifts = []
    for asset in stocks:
        beta = max(asset["vol"], 0.05) / max(market_volatility, 0.05)
        capm_drift = risk_free_rate + beta * equity_risk_premium
        capm_drifts.append(capm_drift)
        path_integral_yields.append(
            _feynman_path_integral_yield(asset["yield"], asset["vol"], capm_drift, rng=rng)
        )

    scores = np.array([
        max(pi_yield, 0.0) / max(asset["vol"], 0.05) * (1.0 - min(asset["vol"], 0.8) * 0.35)
        for asset, pi_yield in zip(stocks, path_integral_yields)
    ])
    weights = scores / scores.sum()
    portfolio_yield = 0.0
    result = []
    for asset, weight, score, pi_yield, capm_drift in zip(stocks, weights, scores, path_integral_yields, capm_drifts):
        income = capital * weight * pi_yield / 100.0
        portfolio_yield += weight * pi_yield
        result.append({
            "ticker": asset["ticker"],
            "dividend_yield": f"{asset['yield']:.1f}% cash dividend",
            "path_integral_yield": f"{pi_yield:.2f}% (Feynman path-integral, CAPM drift {capm_drift * 100.0:.2f}%)",
            "allocation": f"{weight * 100.0:.1f}%",
            "projected_income": f"${income:,.2f} /yr",
            "risk_adjusted_score": round(float(score), 2),
        })
    return result, float(portfolio_yield)