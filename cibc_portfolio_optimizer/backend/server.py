"""
CIBC Dividend Portfolio Server
Flask REST API for quantum-enhanced dividend portfolio optimization
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
from typing import Dict, List

from quantum_optimizer import DividendPortfolioOptimizer
from dividend_engine import DividendEngine
from market_data import MarketDataGenerator
from qiskit_optimizer import QiskitGeodesicOptimizer
from portfolio_analytics import PortfolioAnalytics
from ai_advisor import AIAdvisor
from risk_classifier import RiskClassifier
from spread_optimizer import SpreadOptimizer
from ibkr_client import IBKRClient


app = Flask(__name__)
CORS(app)

# Initialize components
market_data = MarketDataGenerator()
# quantum_optimizer = DividendPortfolioOptimizer(num_assets=len(market_data.get_all_stocks()))
quantum_optimizer = QiskitGeodesicOptimizer(num_assets=len(market_data.get_all_stocks()))
dividend_engine = DividendEngine()
portfolio_analytics = PortfolioAnalytics()
ai_advisor = AIAdvisor()
risk_classifier = RiskClassifier()
spread_optimizer = SpreadOptimizer()
ibkr_client = IBKRClient()
ibkr_client.connect()


@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    """Get all available stocks"""
    stocks = market_data.get_all_stocks()
    return jsonify({
        'stocks': stocks,
        'count': len(stocks)
    })


@app.route('/api/stocks/<symbol>', methods=['GET'])
def get_stock(symbol):
    """Get specific stock details"""
    stock = market_data.get_stock_by_symbol(symbol)
    
    if not stock:
        return jsonify({'error': 'Stock not found'}), 404
    
    # Add dividend analysis
    analysis = dividend_engine.analyze_dividend_stock(stock)
    
    return jsonify({
        'stock': stock,
        'analysis': analysis
    })


@app.route('/api/market/summary', methods=['GET'])
def get_market_summary():
    """Get market summary statistics"""
    summary = market_data.get_market_summary()
    return jsonify(summary)


@app.route('/api/optimize', methods=['POST'])
def optimize_portfolio():
    """
    Optimize portfolio using quantum VQE
    
    Request body:
    {
        "portfolio_value": 100000,
        "risk_tolerance": "moderate",
        "target_dividend_yield": 5.0,
        "target_dividend_yield": 5.0,
        "target_return": 0.30,
        "sector_constraints": {"Financials": 0.35}
    }
    """
    data = request.json
    
    portfolio_value = data.get('portfolio_value', 100000)
    risk_tolerance = data.get('risk_tolerance', 'moderate')
    target_dividend_yield = data.get('target_dividend_yield')
    target_return = data.get('target_return', 0.30)  # Default 30% per user request
    sector_constraints = data.get('sector_constraints')
    
    # Get market data
    stocks = market_data.get_all_stocks()
    expected_returns = market_data.generate_expected_returns()
    covariance_matrix = market_data.generate_covariance_matrix()
    dividend_yields = market_data.get_dividend_yields()
    
    # Run quantum optimization
    try:
        # Use Qiskit Geodesic Optimizer
        optimization_result = quantum_optimizer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            dividend_yields=dividend_yields,
            risk_tolerance=risk_tolerance,
            target_return=target_return,
            sector_constraints=sector_constraints,
            stock_list=stocks
        )
        
        # ML-Enhanced Optimization (Comparison)
        # Use shrunk covariance for better stability
        shrunk_cov = portfolio_analytics.calculate_shrunk_covariance(market_data.generate_historical_returns())
        ml_result = portfolio_analytics.optimize_sharpe_ratio_ml(
            expected_returns, shrunk_cov
        )
        
        # Calculate portfolio analytics
        weights = np.array(optimization_result['weights'])
        portfolio_metrics = portfolio_analytics.calculate_portfolio_metrics(
            weights, expected_returns, covariance_matrix, dividend_yields
        )
        
        # Calculate VaR and CVaR
        risk_metrics = portfolio_analytics.calculate_var_cvar(
            weights, expected_returns, covariance_matrix
        )
        
        # Sector allocation
        sector_allocation = portfolio_analytics.calculate_sector_allocation(weights, stocks)
        
        # Get quantum metrics
        quantum_metrics = quantum_optimizer.get_quantum_metrics()
        
        # Build holdings list
        holdings = []
        for i, stock in enumerate(stocks):
            if weights[i] > 0.001:  # Only include meaningful positions
                holdings.append({
                    'symbol': stock['symbol'],
                    'name': stock['name'],
                    'sector': stock['sector'],
                    'weight': float(weights[i]),
                    'value': float(weights[i] * portfolio_value),
                    'shares': int(weights[i] * portfolio_value / stock['price']),
                    'price': stock['price'],
                    'dividend_yield': stock['dividend_yield'],
                    'annual_dividend': stock['annual_dividend']
                })
        
        # Sort by weight
        holdings.sort(key=lambda x: x['weight'], reverse=True)
        
        return jsonify({
            'success': True,
            'optimization': optimization_result,
            'ml_optimization': {
                'sharpe_ratio': float(ml_result['sharpe_ratio']),
                'weights': ml_result['weights'].tolist()
            },
            'portfolio_metrics': portfolio_metrics,
            'risk_metrics': risk_metrics,
            'sector_allocation': sector_allocation,
            'quantum_metrics': quantum_metrics,
            'holdings': holdings,
            'total_value': portfolio_value
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analytics/efficient-frontier', methods=['POST'])
def get_efficient_frontier():
    """Generate efficient frontier"""
    data = request.json
    num_portfolios = data.get('num_portfolios', 100)
    
    expected_returns = market_data.generate_expected_returns()
    covariance_matrix = market_data.generate_covariance_matrix()
    dividend_yields = market_data.get_dividend_yields()
    
    frontier = portfolio_analytics.generate_efficient_frontier(
        expected_returns, covariance_matrix, dividend_yields, num_portfolios
    )
    
    return jsonify(frontier)


@app.route('/api/dividend/calendar', methods=['POST'])
def get_dividend_calendar():
    """
    Generate dividend payment calendar
    
    Request body:
    {
        "holdings": [...],
        "months_ahead": 12
    }
    """
    data = request.json
    holdings = data.get('holdings', [])
    months_ahead = data.get('months_ahead', 12)
    
    calendar = dividend_engine.generate_dividend_calendar(holdings, months_ahead)
    
    # Calculate monthly totals
    monthly_totals = {}
    for payment in calendar:
        month = payment['payment_date'][:7]  # YYYY-MM
        if month not in monthly_totals:
            monthly_totals[month] = 0
        monthly_totals[month] += payment['total_payment']
    
    return jsonify({
        'calendar': calendar,
        'monthly_totals': monthly_totals
    })


@app.route('/api/dividend/forecast', methods=['POST'])
def get_dividend_forecast():
    """
    Forecast future dividend income
    
    Request body:
    {
        "portfolio_value": 100000,
        "weights": [...],
        "years": 10
    }
    """
    data = request.json
    portfolio_value = data.get('portfolio_value', 100000)
    weights = np.array(data.get('weights', []))
    years = data.get('years', 10)
    
    dividend_yields = market_data.get_dividend_yields()
    
    # Calculate growth rates from historical data
    stocks = market_data.get_all_stocks()
    growth_rates = []
    for stock in stocks:
        history = stock.get('dividend_history', [])
        if len(history) >= 5:
            growth_rate = ((history[-1] / history[-5]) ** 0.2 - 1) * 100
        else:
            growth_rate = 5.0  # Default 5%
        growth_rates.append(growth_rate)
    
    growth_rates = np.array(growth_rates)
    
    forecast = dividend_engine.forecast_dividend_income(
        portfolio_value, weights, dividend_yields, growth_rates, years
    )
    
    return jsonify(forecast)


@app.route('/api/dividend/tax-analysis', methods=['POST'])
def get_tax_analysis():
    """
    Calculate Canadian dividend tax analysis
    
    Request body:
    {
        "dividend_amount": 5000,
        "is_eligible": true,
        "marginal_tax_rate": 0.50
    }
    """
    data = request.json
    dividend_amount = data.get('dividend_amount', 0)
    is_eligible = data.get('is_eligible', True)
    marginal_tax_rate = data.get('marginal_tax_rate', 0.50)
    
    tax_analysis = dividend_engine.calculate_canadian_tax_credit(
        dividend_amount, is_eligible, marginal_tax_rate
    )
    
    return jsonify(tax_analysis)


@app.route('/api/ai/analyze', methods=['POST'])
def ai_analyze_portfolio():
    """
    Get AI analysis of portfolio
    
    Request body:
    {
        "portfolio_data": {...},
        "market_conditions": {...},
        "user_profile": {...}
    }
    """
    data = request.json
    portfolio_data = data.get('portfolio_data', {})
    market_conditions = data.get('market_conditions', {})
    user_profile = data.get('user_profile', {})
    
    analysis = ai_advisor.analyze_portfolio(
        portfolio_data, market_conditions, user_profile
    )
    
    return jsonify({
        'analysis': analysis
    })


@app.route('/api/ai/recommendations', methods=['POST'])
def ai_get_recommendations():
    """
    Get AI-powered recommendations
    
    Request body:
    {
        "portfolio_data": {...},
        "user_goals": {...}
    }
    """
    data = request.json
    portfolio_data = data.get('portfolio_data', {})
    user_goals = data.get('user_goals', {})
    
    stocks = market_data.get_all_stocks()
    
    recommendations = ai_advisor.generate_recommendations(
        portfolio_data, stocks, user_goals
    )
    
    return jsonify({
        'recommendations': recommendations
    })


@app.route('/api/ai/ask', methods=['POST'])
def ai_answer_question():
    """
    Ask AI advisor a question
    
    Request body:
    {
        "question": "What are the best dividend stocks?",
        "context": {...}
    }
    """
    data = request.json
    question = data.get('question', '')
    context = data.get('context', {})
    
    # Add stocks to context
    context['stocks'] = market_data.get_all_stocks()
    
    answer = ai_advisor.answer_question(question, context)
    
    return jsonify({
        'question': question,
        'answer': answer
    })


@app.route('/api/ai/generate-code', methods=['POST'])
def ai_generate_code():
    """
    Generate Python code for portfolio analysis
    """
    data = request.json
    query = data.get('query', '')
    context = data.get('context', {})
    
    code = ai_advisor.generate_python_code(query, context)
    
    return jsonify({
        'query': query,
        'code': code
    })


@app.route('/api/analytics/performance-attribution', methods=['POST'])
def get_performance_attribution():
    """Get performance attribution analysis"""
    data = request.json
    weights = np.array(data.get('weights', []))
    
    stocks = market_data.get_all_stocks()
    expected_returns = market_data.generate_expected_returns()
    dividend_yields = market_data.get_dividend_yields()
    
    attribution = portfolio_analytics.performance_attribution(
        weights, expected_returns, dividend_yields, stocks
    )
    
    return jsonify(attribution)


@app.route('/api/risk/classify', methods=['POST'])
def classify_portfolio_risk():
    """
    Classify portfolio risk using statistical methods
    
    Request body:
    {
        "returns": [...],
        "volatility": 0.15,
        "beta": 1.2,
        "var_95": 0.05
    }
    """
    data = request.json
    returns = np.array(data.get('returns', []))
    volatility = data.get('volatility', 0.15)
    beta = data.get('beta', 1.0)
    var_95 = data.get('var_95', 0.05)
    
    classification = risk_classifier.classify_asset_risk(
        returns, volatility, beta, var_95
    )
    
    return jsonify(classification)


@app.route('/api/risk/parametric', methods=['POST'])
def get_parametric_risk_metrics():
    """
    Get comprehensive parametric risk metrics
    
    Request body:
    {
        "returns": [...],
        "portfolio_value": 100000
    }
    """
    data = request.json
    returns = np.array(data.get('returns', []))
    portfolio_value = data.get('portfolio_value', 100000)
    
    metrics = risk_classifier.calculate_parametric_risk_metrics(
        returns, portfolio_value
    )
    
    return jsonify(metrics)


@app.route('/api/risk/distribution-fit', methods=['POST'])
def fit_return_distribution():
    """
    Fit parametric distributions to return data
    
    Request body:
    {
        "returns": [...]
    }
    """
    data = request.json
    returns = np.array(data.get('returns', []))
    
    distribution_fit = risk_classifier.fit_parametric_distribution(returns)
    
    return jsonify(distribution_fit)


@app.route('/api/risk/regime-detection', methods=['POST'])
def detect_market_regimes():
    """
    Detect market regime changes
    
    Request body:
    {
        "returns": [...],
        "n_regimes": 3
    }
    """
    data = request.json
    returns = np.array(data.get('returns', []))
    n_regimes = data.get('n_regimes', 3)
    
    regimes = risk_classifier.detect_regime_changes(returns, n_regimes)
    
    return jsonify(regimes)


@app.route('/api/risk/cvar-estimates', methods=['POST'])
def estimate_cvar():
    """
    Estimate CVaR using multiple methods
    
    Request body:
    {
        "returns": [...],
        "confidence_level": 0.95,
        "method": "parametric"
    }
    """
    data = request.json
    returns = np.array(data.get('returns', []))
    confidence_level = data.get('confidence_level', 0.95)
    method = data.get('method', 'parametric')
    
    cvar_estimates = risk_classifier.estimate_conditional_var(
        returns, confidence_level, method
    )
    
    return jsonify(cvar_estimates)


@app.route('/api/spreads', methods=['GET'])
def get_spreads():
    """Get optimal spread projections for Equities and Forex"""
    stocks = market_data.get_all_stocks()
    spreads = spread_optimizer.generate_spreads(stocks)
    return jsonify(spreads)


@app.route('/api/trade/execute', methods=['POST'])
def execute_trade():
    """Execute portfolio rebalancing via Interactive Brokers"""
    data = request.json
    holdings = data.get('holdings', [])
    
    if not holdings:
        return jsonify({'error': 'No holdings provided'}), 400
        
    execution_result = ibkr_client.execute_portfolio_rebalance(holdings)
    return jsonify(execution_result)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'CIBC Dividend Portfolio Optimizer',
        'quantum_enabled': True
    })


if __name__ == '__main__':
    print("=" * 60)
    print("CIBC Dividend Portfolio Optimization System")
    print("Quantum-Enhanced Portfolio Management")
    print("=" * 60)
    print(f"\nLoaded {len(market_data.get_all_stocks())} dividend stocks")
    print(f"Quantum optimizer initialized with {quantum_optimizer.num_qubits} qubits")
    print("\nServer starting on http://localhost:5001")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
