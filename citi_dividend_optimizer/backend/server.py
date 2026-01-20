"""
Citi Dividend Portfolio Server
Backend Logic for Stats Learning, NVQLink, and Optimization.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import threading
import time

from market_data import MarketData
from ml_engine import StatisticalLearner
from nvq_link import NVQLink

app = Flask(__name__)
CORS(app)

# --- Service Initialization ---
market = MarketData()
learner = StatisticalLearner()
nvq = NVQLink()

# In-memory store for generated data
cache = {
    'history': {},
    'forecasts': {},
    'nvq_data': []
}

def bg_update_models():
    """Background task to update ML models and NVQ data."""
    while True:
        # 1. Update Market Data
        cache['history'] = market.generate_history()
        
        # 2. Run Stat Learning
        cache['forecasts'] = learner.forecast_returns_ridge(cache['history'])
        
        # 3. Update NVQ Link
        cache['nvq_data'] = nvq.get_live_ore_prices()
        
        print(f"[System] Models Updated. Forecasts: {len(cache['forecasts'])} | NVQ: {len(cache['nvq_data'])}")
        time.sleep(60)

# Start Background Thread
t = threading.Thread(target=bg_update_models, daemon=True)
t.start()

# Let the thread init first
time.sleep(1)

@app.route('/api/init', methods=['GET'])
def get_init_data():
    """Initial load data."""
    return jsonify({
        'stocks': market.get_universe(),
        'market_context': market.get_market_context()
    })

@app.route('/api/nvq/minerals', methods=['GET'])
def get_minerals():
    """Get NVQLink Mineral Data."""
    return jsonify(cache['nvq_data'])

@app.route('/api/optimize', methods=['POST'])
def optimize():
    """
    Optimize Portfolio using Stats Learning + NVQ Adjustments.
    """
    data = request.json
    risk_profile = data.get('risk', 'moderate') # moderate, aggressive, conservative
    
    stocks = market.get_universe()
    allocations = []
    
    total_score = 0
    scores = []
    
    # NVQ Adjustment Factors
    nvq_map = {item['symbol']: item['nvq_signal'] for item in cache['nvq_data']}
    
    for stock in stocks:
        symbol = stock['symbol']
        
        # Base Score from ML Forecast
        forecast = cache['forecasts'].get(symbol, 0.05)
        score = forecast * 100 # Normalize roughly to 5-15 scale
        
        # Dividend Yield Boost
        score += stock['div_yield'] * 2.5 
        
        # NVQ Signal Integration (Mineral Sensitivity)
        # Financials/Energy sensitive to Commodities
        if stock['sector'] in ['Energy', 'Materials', 'Industrials']:
            if nvq_map.get('COPP') == 'BUY' or nvq_map.get('LITH') == 'BUY':
                score *= 1.2 # Boost cyclical sectors on commodity boom
        
        # Risk Profile Adjustment
        if risk_profile == 'conservative':
            if stock['sector'] in ['Utilities', 'Consumer', 'Healthcare']:
                score *= 1.3
        
        scores.append(score if score > 0 else 0)
        
    total_score = sum(scores)
    
    # Calculate Allocations
    for i, stock in enumerate(stocks):
        weight = scores[i] / total_score if total_score > 0 else 0
        allocations.append({
            'symbol': stock['symbol'],
            'name': stock['name'],
            'sector': stock['sector'],
            'weight': round(weight * 100, 2),
            'value': round(weight * 100000, 2), # Assuming 100k portfolio
            'ml_forecast': round(cache['forecasts'].get(stock['symbol'], 0) * 100, 1)
        })
        
    # Sort by weight
    allocations.sort(key=lambda x: x['weight'], reverse=True)
    
    return jsonify({
        'allocations': allocations,
        'status': 'Optimized via Ridge Regression & Elliptic NVQ'
    })

if __name__ == '__main__':
    print("Citi Dividend Optimizer | Statistical Learning Service")
    print("Server running on port 5002")
    app.run(port=5002)
