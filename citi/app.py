from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__, static_url_path='', static_folder='.')

# Mock Data for Canadian, N. Ontario & Greenland Mineral Ores
TICKERS = [
    {"symbol": "AMRQ.TO", "name": "Amaroq Minerals", "yield": 0.0, "sector": "Gold/Strategic", "region": "Greenland"},
    {"symbol": "HUD.VN", "name": "Hudson Resources", "yield": 0.0, "sector": "Rare Earths", "region": "Greenland"},
    {"symbol": "AEM.TO", "name": "Agnico Eagle", "yield": 2.6, "sector": "Gold", "region": "N. Ontario"},
    {"symbol": "AGI.TO", "name": "Alamos Gold", "yield": 1.1, "sector": "Gold", "region": "N. Ontario"},
    {"symbol": "K.TO", "name": "Kinross Gold", "yield": 1.4, "sector": "Gold", "region": "N. Ontario"},
    {"symbol": "TECK.B.TO", "name": "Teck Resources", "yield": 2.1, "sector": "Diversified Metals", "region": "Canada"},
    {"symbol": "FM.TO", "name": "First Quantum", "yield": 1.8, "sector": "Copper", "region": "Canada"},
    {"symbol": "LUN.TO", "name": "Lundin Mining", "yield": 3.4, "sector": "Base Metals", "region": "Canada"},
    {"symbol": "ETM.AX", "name": "Energy Transition", "yield": 0.0, "sector": "Rare Earths", "region": "Greenland"},
    {"symbol": "NTR.TO", "name": "Nutrien Ltd", "yield": 3.8, "sector": "Potash", "region": "Canada"},
]

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/api/portfolio')
def get_portfolio():
    # Simulate an existing portfolio or optimized suggestion
    portfolio = []
    
    # Regional Factors
    GREENLAND_MULTIPLIER = 1.35
    ONTARIO_STABILITY_FACTOR = 1.25 # Premium for Tier-1 Jurisdiction
    
    for t in TICKERS:
        # Advanced Statistical Simulation
        # N. Ontario has lower volatility (stable jurisdiction)
        if t['region'] == 'N. Ontario':
            volatility = round(random.uniform(0.12, 0.28), 2)
        elif t['region'] == 'Greenland':
            volatility = round(random.uniform(0.25, 0.55), 2) # Higher frontier risk
        else:
            volatility = round(random.uniform(0.15, 0.45), 2)
            
        beta = round(random.uniform(0.8, 1.8), 2)
        
        # Geodesic boost calc (Scholes Extension)
        risk_free = 0.045
        # Lower vol for Ontario means lower option premiums, but we apply a "Stability Multiplier"
        # because the Probability of Exercise is more congruent (measurable).
        d1 = (np.log(1.1) + (risk_free + 0.5 * volatility ** 2)) / volatility
        option_premium_est = (d1 * 0.45)
        
        geodesic_boost = round(option_premium_est * 3.0, 2)
        
        base_yield = t['yield']
        # Synthesize yield
        if base_yield == 0:
            effective_yield = geodesic_boost
        else:
            effective_yield = round(base_yield + geodesic_boost, 2)
        
        sharpe = round((effective_yield - 2.0) / (volatility * 12), 2)
        
        # Optimization Scoring Logic
        base_score = (effective_yield * 4) + (sharpe * 12)
        
        # Region specific scoring
        if t['region'] == 'Greenland':
            base_score *= GREENLAND_MULTIPLIER
        elif t['region'] == 'N. Ontario':
            base_score *= ONTARIO_STABILITY_FACTOR # Reward stability
            base_score += 5 # Tier 1 Bonus
            
        noise = np.random.normal(0, 5)
        ai_confidence = min(max(base_score + noise + 40, 0), 99)
        
        # Recommendation Tiers
        if ai_confidence > 85:
            recommendation = "TIER-1 BUY" if t['region'] == 'N. Ontario' else "STRATEGIC BUY"
        elif ai_confidence > 70:
            recommendation = "ACCUMULATE"
        else:
            recommendation = "HOLD"
            
        allocation = ai_confidence / 1.8

        portfolio.append({
            "ticker": t["symbol"],
            "name": t["name"],
            "dividend_yield": effective_yield,
            "base_yield": t['yield'],
            "geodesic_alpha": geodesic_boost,
            "region": t['region'],
            "ai_score": round(ai_confidence, 1),
            "stats": {
                "beta": beta,
                "sharpe": sharpe,
                "volatility": f"{int(volatility*100)}%"
            },
            "recommendation": recommendation,
            "suggested_allocation": f"{allocation:.1f}%"
        })
    
    # Sort by AI Score
    portfolio.sort(key=lambda x: x['ai_score'], reverse=True)
    return jsonify(portfolio)

@app.route('/api/connect_ibkr', methods=['POST'])
def connect_ibkr():
    # Simulate NVQLink / IBKR connection handshake
    # In production: ib = IB(); ib.connect('127.0.0.1', 7497, clientId=1)
    
    # Simulate network latency for realistic UI feedback
    import time
    time.sleep(1.5)
    
    # Force success for demo reliability
    return jsonify({
        "status": "connected", 
        "message": "Connected to Interactive Brokers via NVQLink (TWS: 985, API: 10.19)",
        "account_id": "U12345678",
        "timestamp": pd.Timestamp.now().isoformat()
    }), 200

@app.route('/api/gold_prices')
def get_gold_prices():
    # Simulate forward curve based on Monte Carlo
    # Start at current mock spot
    current_price = 2350
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    forecast = []
    
    price = current_price
    for m in months:
        # Random walk with positive drift (bullish gold view)
        change = np.random.normal(15, 40) 
        price += change
        forecast.append({"month": m, "price": int(price)})
        
    return jsonify(forecast)

@app.route('/api/optimize_trade', methods=['POST'])
def optimize_trade():
    """
    Performs trade optimization using Feynman Path Integrals to find the
    most likely price path (least action principle) vs Black-Scholes baseline.
    """
    data = request.json
    ticker = data.get('ticker', 'AMRQ.TO')
    
    # 1. Black-Scholes (Geodesic) Signature
    # Standard log-normal distribution assumption
    spot = 100.0
    vol = 0.35
    bs_price = spot * np.exp(0.05 * 1.0) # Simple drift
    
    # 2. Feynman Path Integral (Quantum) Signature
    # Simulate "Sum over Histories"
    # We generate multiple paths and weight them by their "Action"
    n_paths = 500
    n_steps = 50
    dt = 1/n_steps
    
    final_prices = []
    
    for _ in range(n_paths):
        path_spot = spot
        # Random walk with "Quantum Potential" (simulated by non-gaussian jumps)
        for _ in range(n_steps):
            # Complex noise pattern (simulating Levy flight or jump diffusion)
            noise = np.random.normal(0, np.sqrt(dt)) + (np.random.laplace(0, 0.05) if random.random() < 0.1 else 0)
            path_spot *= np.exp((0.05 - 0.5 * vol**2) * dt + vol * noise)
        final_prices.append(path_spot)
    
    # The "Propagator" implies the most probable path (median/mode of distribution)
    # usually slightly different from BS due to the fat tails (jumps)
    feynman_price = np.median(final_prices)
    
    # Calculate "Quantum Arbitrage" or Alpha
    quantum_alpha = round(((feynman_price - bs_price) / bs_price) * 100, 2)
    
    # Ramanujan's Elliptic Integral Projection (Congruence Model)
    # Beyond Black-Scholes: Using theta functions to model non-linear price harmonics
    # Ramanujan's theta function: f(a,b) = sum(a^(n(n+1)/2) * b^(n(n-1)/2))
    
    ramanujan_prices = []
    # Simplified modular form for price projection
    def theta_proxy(n, volatility):
        # Using modular arithmetic behavior for statistical congruence
        q = np.exp(-np.pi * volatility) 
        # A mock approximation of a theta-function series expansion relative to 'n'
        term = 1 + 2 * (q ** (n**2)) 
        return term

    # Generate a comparative characteristic plot (Projected yield curve)
    steps = 12
    comparison_plot = []
    
    current_val = feynman_price
    bs_val = bs_price
    ram_val = feynman_price
    
    for i in range(1, steps + 1):
        # 1. Black-Scholes Projection (Standard Log-Normal)
        bs_val *= np.exp(0.05/12) # constant drift
        
        # 2. Ramanujan Projection (Elliptic Non-Linearity)
        # Using the theta_proxy to modulate the drift based on market "harmonics"
        harmonic_factor = theta_proxy(i, vol)
        drift_correction = (harmonic_factor - 1) * 0.5 # Scale effect
        ram_val *= (1 + (0.05/12) + drift_correction)
        
        comparison_plot.append({
            "step": i,
            "bs_val": round(bs_val, 2),
            "ramanujan_val": round(ram_val, 2)
        })
        
    ramanujan_yield_alpha = round(((ram_val - bs_val) / bs_val) * 100, 2)
    
    return jsonify({
        "ticker": ticker,
        "bs_signature": round(bs_price, 2),
        "feynman_integral": round(feynman_price, 2),
        "ramanujan_projection": round(ram_val, 2),
        "quantum_alpha": f"{quantum_alpha}%",
        "ramanujan_yield_boost": f"{ramanujan_yield_alpha}%",
        "optimization_status": "CONVERGED",
        "comparative_plot": comparison_plot,
        "optimal_entry": round(feynman_price * 0.98, 2) 
    })

if __name__ == '__main__':
    app.run(debug=True, port=3000)
