

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


