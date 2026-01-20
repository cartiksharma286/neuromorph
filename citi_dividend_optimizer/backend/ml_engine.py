"""
Statistical Learning Engine for Portfolio Optimization.
Uses Ridge Regression and Probabilistic Momentum to forecast returns.
"""

import numpy as np
from scipy import stats

class StatisticalLearner:
    def __init__(self):
        self.alpha = 0.5 # Learning rate / Smoothing factor
        
    def calculate_probabilistic_momentum(self, prices, window=20):
        """
        Calculate momentum using a probabilistic approach (t-distribution of returns).
        """
        if len(prices) < window:
            return 0.0
            
        recent_prices = prices[-window:]
        returns = np.diff(np.log(recent_prices))
        
        # t-statistic for positive return hypothesis
        t_stat, p_val = stats.ttest_1samp(returns, 0.0)
        
        # Momentum score: sign(t) * (1 - p/2)
        # High confidence positive trend -> score close to 1
        score = np.sign(t_stat) * (1 - p_val/2) if not np.isnan(t_stat) else 0.0
        return score

    def forecast_returns_ridge(self, history_data):
        """
        Forecast expected returns using a simplified Ridge Regression on lagged returns.
        y_t = w * y_{t-1} + b
        """
        forecasts = {}
        for symbol, prices in history_data.items():
            if len(prices) < 30:
                forecasts[symbol] = 0.05 # Default
                continue
                
            # Create lag features
            # X = returns[t-1], y = returns[t]
            returns = np.diff(np.log(prices))
            X = returns[:-1]
            y = returns[1:]
            
            # Ridge Regression (Closed Form): w = (X^T X + lambda I)^-1 X^T y
            lambda_reg = 0.1
            X_centered = X - np.mean(X)
            y_centered = y - np.mean(y)
            
            numerator = np.dot(X_centered, y_centered)
            denominator = np.dot(X_centered, X_centered) + lambda_reg
            
            beta = numerator / denominator if denominator != 0 else 0
            alpha = np.mean(y) - beta * np.mean(X)
            
            # Forecast next period return
            next_return = alpha + beta * returns[-1]
            
            # Annualize
            annualized_return = next_return * 252
            forecasts[symbol] = annualized_return
            
        return forecasts

    def calculate_risk_parity_weights(self, cov_matrix):
        """
        Calculate weights inversely proportional to volatility (Risk Parity approximation).
        """
        volatilities = np.sqrt(np.diag(cov_matrix))
        inv_vols = 1.0 / volatilities
        weights = inv_vols / np.sum(inv_vols)
        return weights
