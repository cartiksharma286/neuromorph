from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__, static_url_path='', static_folder='.')

# Mock Data for Canadian, N. Ontario & Greenland Mineral Ores
# Mock Data for NASDAQ Stocks and Forex Pairs
ASSETS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "type": "Stock", "sector": "Technology", "currency": "USD"},
    {"symbol": "MSFT", "name": "Microsoft Corp.", "type": "Stock", "sector": "Technology", "currency": "USD"},
    {"symbol": "NVDA", "name": "NVIDIA Corp.", "type": "Stock", "sector": "Semiconductors", "currency": "USD"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "type": "Stock", "sector": "Consumer Cyclical", "currency": "USD"},
    {"symbol": "AMZN", "name": "Amazon.com", "type": "Stock", "sector": "Consumer Cyclical", "currency": "USD"},
    {"symbol": "EUR/USD", "name": "Euro / US Dollar", "type": "Forex", "sector": "Currency", "currency": "USD"},
    {"symbol": "GBP/USD", "name": "British Pound / USD", "type": "Forex", "sector": "Currency", "currency": "USD"},
    {"symbol": "USD/JPY", "name": "US Dollar / Japanese Yen", "type": "Forex", "sector": "Currency", "currency": "JPY"},
    {"symbol": "USD/CAD", "name": "US Dollar / Canadian Dollar", "type": "Forex", "sector": "Currency", "currency": "CAD"},
    {"symbol": "AUD/USD", "name": "Aust. Dollar / USD", "type": "Forex", "sector": "Currency", "currency": "USD"},
    {"symbol": "LITH", "name": "Lithium Carbonate", "type": "Commodity", "sector": "Minerals", "currency": "USD"},
    {"symbol": "COBT", "name": "Cobalt Futures", "type": "Commodity", "sector": "Minerals", "currency": "USD"},
    {"symbol": "NICK", "name": "Nickel Spot", "type": "Commodity", "sector": "Minerals", "currency": "USD"},
    {"symbol": "VALE", "name": "Vale S.A.", "type": "Stock", "sector": "Mining", "currency": "USD"},
    {"symbol": "RIO", "name": "Rio Tinto", "type": "Stock", "sector": "Mining", "currency": "USD"},
]

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/api/portfolio')
def get_portfolio():
    # Simulate a mixed asset portfolio
    portfolio = []
    
    for asset in ASSETS:
        # Generate realistic price movement simulation
        base_price = 100.0
        if asset['type'] == 'Forex':
            base_price = 1.05 if 'EUR' in asset['symbol'] else 145.0 if 'JPY' in asset['symbol'] else 0.75
        elif asset['type'] == 'Commodity':
            base_price = 25000.0 if 'LITH' in asset['symbol'] else 30000.0 if 'COBT' in asset['symbol'] else 18000.0
        else:
            base_price = random.uniform(50, 800)
            
        volatility = 0.02 if asset['type'] == 'Forex' else (0.08 if asset['type'] == 'Commodity' else 0.05)
        
        # Simulate current stats
        change_pct = np.random.normal(0, 1.5 if asset['type'] != 'Commodity' else 3.0)
        current_price = base_price * (1 + change_pct/100)
        
        # Beta calculation
        if asset['type'] == 'Stock':
             beta = round(random.uniform(0.8, 1.5), 2)
        elif asset['type'] == 'Commodity':
             beta = round(random.uniform(0.5, 1.2), 2)
        else:
             beta = round(random.uniform(-0.2, 0.2), 2)
        
        # Simple signal generation
        ma_50 = base_price * (1 + np.random.normal(0, 0.02))
        
        # AI Logic
        score = 50 + change_pct * 5 + (10 if current_price > ma_50 else -10)
        ai_confidence = min(max(score + np.random.normal(0, 5), 0), 99)
        
        if ai_confidence > 80:
            recommendation = "STRONG BUY"
        elif ai_confidence > 60:
            recommendation = "BUY"
        elif ai_confidence < 30:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
            
        allocation = ai_confidence / 2.5 # Simple heuristic

        portfolio.append({
            "ticker": asset["symbol"],
            "name": asset["name"],
            "type": asset["type"],
            "price": round(current_price, 4 if asset['type'] == 'Forex' else 2),
            "change": round(change_pct, 2),
            "stats": {
                "beta": beta,
                "volatility": f"{int(volatility*100 * (10 if asset['type']=='Forex' else 1))}%" 
            },
            "ai_score": round(ai_confidence, 1),
            "recommendation": recommendation,
            "suggested_allocation": f"{allocation:.1f}%"
        })
    
    # Sort by AI Score
    portfolio.sort(key=lambda x: x['ai_score'], reverse=True)
    return jsonify(portfolio)

@app.route('/api/minerals/strategy', methods=['POST'])
def generate_mineral_strategy():
    """
    Generates a specialized trading strategy for Mineral Ores.
    Includes Dividend Portfolio Optimization.
    """
    # 1. Select High-Yield Mining Stocks
    mining_stocks = [
        {"ticker": "RIO", "yield": 6.5, "price": 68.50, "volatility": 0.25},
        {"ticker": "BHP", "yield": 5.8, "price": 59.20, "volatility": 0.22},
        {"ticker": "VALE", "yield": 7.2, "price": 14.10, "volatility": 0.35},
        {"ticker": "GLNCY", "yield": 4.5, "price": 10.80, "volatility": 0.28},
        {"ticker": "SCCO", "yield": 3.2, "price": 75.40, "volatility": 0.20}
    ]
    
    # 2. Simulate Dividend Optimization (Kevin's Method / Mean-Variance)
    # Objective: Maximize Yield / Risk
    optimized_portfolio = []
    total_score = 0
    
    for stock in mining_stocks:
        # Score = Yield / Volatility (Sharpe-like metric for income)
        score = stock['yield'] / stock['volatility']
        stock['score'] = score
        total_score += score
        
    # Allocate based on score
    for stock in mining_stocks:
        weight = (stock['score'] / total_score) * 100
        optimized_portfolio.append({
            "ticker": stock['ticker'],
            "dividend_yield": f"{stock['yield']}%",
            "allocation": f"{weight:.1f}%",
            "projected_income": f"${(100000 * (weight/100) * (stock['yield']/100)):,.2f} /yr",
            "risk_adjusted_score": round(stock['score'], 2)
        })
        
    return jsonify({
        "strategy_name": "High-Yield Mineral Dividend Maximizer",
        "description": "Optimizes allocation in major mining conglomerates to maximize dividend income while minimizing commodity cycle volatility.",
        "target_yield": "5.9%",
        "optimized_portfolio": optimized_portfolio,
        "market_context": "Commodity Supercycle: Demand for battery metals (Lithium, Nickel) supports long-term valuations for diversified miners."
    })

@app.route('/api/connect_ibkr', methods=['POST'])
def connect_ibkr():
    # Simulate NVQLink / IBKR connection handshake
    # In production: ib = IB(); ib.connect('127.0.0.1', 7497, clientId=1)
    
    # Simulate network latency for realistic UI feedback
    import time
    time.sleep(1.0)
    
    return jsonify({
        "status": "connected", 
        "message": "Connected to Interactive Brokers (TWS API 10.19) - NASDAQ/FOREX Streams Active",
        "account_id": "U12345678",
        "timestamp": pd.Timestamp.now().isoformat()
    }), 200

@app.route('/api/market_data')
def get_market_data():
    # Generalized market curve (replacing gold)
    # Simulate S&P 500 futures
    # Generalized market curve 
    # Simulate S&P 500 futures, Lithium Carbonate, and VIX
    
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # 1. S&P 500
    sp_price = 5200
    sp_forecast = []
    
    # 2. Lithium Carbonate ($/tonne)
    lithium_price = 14500
    lithium_forecast = []
    
    # 3. VIX
    vix = 14.5
    vix_forecast = []

    for m in months:
        # S&P Trend (Bullish)
        sp_change = np.random.normal(35, 40)
        sp_price += sp_change
        sp_forecast.append({"month": m, "price": int(sp_price)})
        
        # Lithium Trend (Volatile Bullish)
        lith_change = np.random.normal(200, 350)
        lithium_price += lith_change
        lithium_forecast.append({"month": m, "price": int(lithium_price)})
        
        # VIX Trend (Mean Reverting)
        vix = vix * 0.9 + 14.0 * 0.1 + np.random.normal(0, 0.8)
        vix_forecast.append({"month": m, "price": round(vix, 2)})
        
    return jsonify({
        "sp500": sp_forecast,
        "lithium": lithium_forecast,
        "vix": vix_forecast,
        "meta": {
             "sp_target": int(sp_price),
             "lithium_target": int(lithium_price),
             "vix_target": round(vix, 2)
        }
    })

@app.route('/api/optimize_trade', methods=['POST'])
def optimize_trade():
    """
    Performs trade optimization using Feynman Path Integrals.
    """
    data = request.json
    ticker = data.get('ticker', 'NVDA')
    
    # 1. Black-Scholes (Geodesic) Signature
    spot = 450.0
    vol = 0.35
    bs_price = spot * np.exp(0.05 * 1.0) 
    
    # 2. Feynman Path Integral (Quantum) Signature
    n_paths = 500
    n_steps = 50
    dt = 1/n_steps
    
    final_prices = []
    
    for _ in range(n_paths):
        path_spot = spot
        for _ in range(n_steps):
            noise = np.random.normal(0, np.sqrt(dt)) + (np.random.laplace(0, 0.05) if random.random() < 0.1 else 0)
            path_spot *= np.exp((0.05 - 0.5 * vol**2) * dt + vol * noise)
        final_prices.append(path_spot)
    
    feynman_price = np.median(final_prices)
    quantum_alpha = round(((feynman_price - bs_price) / bs_price) * 100, 2)
    
    # Ramanujan's Elliptic Integral Projection
    ram_val = feynman_price
    steps = 12
    comparison_plot = []
    
    bs_val = bs_price
    
    # Simplified modular form
    def theta_proxy(n, volatility):
        q = np.exp(-np.pi * volatility) 
        term = 1 + 2 * (q ** (n**2)) 
        return term

    for i in range(1, steps + 1):
        bs_val *= np.exp(0.05/12) 
        harmonic_factor = theta_proxy(i, vol)
        drift_correction = (harmonic_factor - 1) * 0.5
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
        "optimal_entry": round(feynman_price * 0.99, 2) 
    })

@app.route('/api/gemini/optimize', methods=['POST'])
def gemini_optimize():
    # Simulate Gemini 3.0 Advanced Reasoning
    data = request.json
    user_context = data.get('context', 'General Market')
    
    # 1. Simulate "Chain of Thought" data processing
    reasoning_steps = [
        "Analyzing real-time order books (L2 Data)...",
        "Correlating interest rate differentials (Fed vs ECB)...",
        "Scanning 10-K filings for hidden liabilities...",
        "Projecting volatility surfaces for options chaining..."
    ]
    
    # 2. Generate "Deep Thinking" Analysis
    market_sentiment = random.choice(["Bullish", "Bearish", "Range-bound", "High Volatility"])
    
    if "forex" in user_context.lower() or "fx" in user_context.lower():
        focus = "G10 FX Markets"
        picks = ["USD/JPY", "EUR/USD", "GBP/USD", "AUD/NZD"]
        rationale = "Interest rate divergence between the Fed and BOJ creates a strong carry trade opportunity, while the Euro remains range-bound."
    elif "tech" in user_context.lower():
        focus = "NASDAQ Growth"
        picks = ["NVDA", "AMD", "MSFT", "PLTR"]
        rationale = "AI Capex cycle is accelerating. Infrastructure plays are offering better risk-adjusted returns than pure software plays."
    else:
        focus = "Global Macro"
        picks = ["SPY", "TLT", "GLD", "UUP"]
        rationale = "Defensive positioning is recommended as yield curve inversion signals potential recessionary pressures in Q3."
        
    response_data = {
        "model_version": "Gemini 3.0 Pro (Financial Tuned)",
        "processing_time": "0.42s",
        "reasoning_trace": reasoning_steps,
        "market_regime": market_sentiment,
        "analysis": {
            "focus_sector": focus,
            "macro_thesis": rationale,
            "suggested_allocation": [
                {"ticker": picks[0], "weight": "35%", "reason": "High Conviction"},
                {"ticker": picks[1], "weight": "25%", "reason": "Diversification"},
                {"ticker": picks[2], "weight": "20%", "reason": "Hedge"},
                {"ticker": picks[3], "weight": "20%", "reason": "Yield"},
            ]
        },
        "gemini_alpha_score": round(random.uniform(92.0, 99.9), 1)
    }
    
    import time
    time.sleep(2) 
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
