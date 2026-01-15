from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__, static_url_path='', static_folder='.')

# Mock Data for Canadian & Greenland Mineral Ores (Greenland Strategic Optimization)
TICKERS = [
    {"symbol": "AMRQ.TO", "name": "Amaroq Minerals", "yield": 0.0, "sector": "Gold/Strategic", "region": "Greenland"},
    {"symbol": "HUD.VN", "name": "Hudson Resources", "yield": 0.0, "sector": "Rare Earths", "region": "Greenland"},
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
    
    # Greenland Strategic Factor (Simulating heightened geopolitical value)
    GREENLAND_MULTIPLIER = 1.35
    
    for t in TICKERS:
        # Advanced Statistical Simulation
        volatility = round(random.uniform(0.15, 0.55), 2)
        if t['region'] == 'Greenland':
            volatility += 0.1 # Emerging frontier risk
            
        beta = round(random.uniform(0.8, 1.8), 2)
        
        # Geodesic boost calc
        risk_free = 0.045
        d1 = (np.log(1.1) + (risk_free + 0.5 * volatility ** 2)) / volatility
        option_premium_est = (d1 * 0.45)
        
        geodesic_boost = round(option_premium_est * 3.0, 2) # Higher vol = higher optionality yield
        
        # Greenland assets often don't pay div, so we synthesize purely from geodesic
        base_yield = t['yield']
        if base_yield == 0:
            effective_yield = geodesic_boost # Pure alpha yield
        else:
            effective_yield = round(base_yield + geodesic_boost, 2)
        
        sharpe = round((effective_yield - 2.0) / (volatility * 12), 2)
        
        # Optimization Logic
        base_score = (effective_yield * 4) + (sharpe * 12)
        
        # Apply Greenland Optimization
        if t['region'] == 'Greenland':
            base_score *= GREENLAND_MULTIPLIER
            
        noise = np.random.normal(0, 5)
        ai_confidence = min(max(base_score + noise + 40, 0), 99)
        
        if ai_confidence > 80:
            recommendation = "STRATEGIC BUY"
        elif ai_confidence > 65:
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

if __name__ == '__main__':
    app.run(debug=True, port=3000)
