"""
CIBC Portfolio Analytics Engine
Comprehensive risk and performance analytics for dividend portfolios
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


class PortfolioAnalytics:
    """Advanced portfolio analytics and risk metrics"""
    
    def __init__(self):
        self.risk_free_rate = 0.04  # 4% risk-free rate (Canadian T-bills)
        
    def calculate_portfolio_metrics(self,
                                    weights: np.ndarray,
                                    expected_returns: np.ndarray,
                                    covariance_matrix: np.ndarray,
                                    dividend_yields: np.ndarray) -> Dict:
        """
        Calculate comprehensive portfolio metrics
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            dividend_yields: Dividend yields
        
        Returns:
            Dictionary with all portfolio metrics
        """
        # Basic metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_dividend_yield = np.dot(weights, dividend_yields)
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Risk-adjusted metrics
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = expected_returns[expected_returns < self.risk_free_rate] - self.risk_free_rate
        downside_variance = np.mean(downside_returns ** 2) if len(downside_returns) > 0 else portfolio_variance
        downside_deviation = np.sqrt(downside_variance)
        sortino_ratio = (portfolio_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Information ratio (assuming benchmark return of 6%)
        benchmark_return = 0.06
        tracking_error = portfolio_volatility * 0.5  # Simplified
        information_ratio = (portfolio_return - benchmark_return) / tracking_error if tracking_error > 0 else 0
        
        return {
            'expected_return': float(portfolio_return * 100),
            'dividend_yield': float(portfolio_dividend_yield),
            'volatility': float(portfolio_volatility * 100),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'information_ratio': float(information_ratio),
            'variance': float(portfolio_variance)
        }
    
    def calculate_var_cvar(self,
                          weights: np.ndarray,
                          expected_returns: np.ndarray,
                          covariance_matrix: np.ndarray,
                          confidence_level: float = 0.95,
                          time_horizon: int = 1) -> Dict:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
        
        Returns:
            Dictionary with VaR and CVaR
        """
        # Portfolio statistics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_std = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
        
        # Scale to time horizon
        scaled_return = portfolio_return * time_horizon
        scaled_std = portfolio_std * np.sqrt(time_horizon)
        
        # VaR using parametric method (assumes normal distribution)
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -(scaled_return + z_score * scaled_std)
        
        # CVaR (expected shortfall)
        cvar = -(scaled_return + scaled_std * stats.norm.pdf(z_score) / (1 - confidence_level))
        
        return {
            'var_95': float(var * 100),
            'cvar_95': float(cvar * 100),
            'confidence_level': confidence_level,
            'time_horizon_days': time_horizon
        }
    
    def calculate_beta(self,
                      asset_returns: np.ndarray,
                      market_returns: np.ndarray) -> float:
        """Calculate beta relative to market"""
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        beta = covariance / market_variance if market_variance > 0 else 1.0
        return float(beta)
    
    def calculate_alpha(self,
                       portfolio_return: float,
                       beta: float,
                       market_return: float) -> float:
        """Calculate Jensen's alpha"""
        expected_return = self.risk_free_rate + beta * (market_return - self.risk_free_rate)
        alpha = portfolio_return - expected_return
        return float(alpha)
    
    def generate_efficient_frontier(self,
                                    expected_returns: np.ndarray,
                                    covariance_matrix: np.ndarray,
                                    dividend_yields: np.ndarray,
                                    num_portfolios: int = 100) -> Dict:
        """
        Generate efficient frontier for dividend portfolios
        
        Args:
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            dividend_yields: Dividend yields
            num_portfolios: Number of portfolios to generate
        
        Returns:
            Dictionary with frontier data
        """
        n_assets = len(expected_returns)
        results = {
            'returns': [],
            'volatilities': [],
            'sharpe_ratios': [],
            'dividend_yields': [],
            'weights': []
        }
        
        # Generate random portfolios
        for _ in range(num_portfolios):
            # Random weights
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            
            # Calculate metrics
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            portfolio_dividend_yield = np.dot(weights, dividend_yields)
            
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            results['returns'].append(float(portfolio_return * 100))
            results['volatilities'].append(float(portfolio_volatility * 100))
            results['sharpe_ratios'].append(float(sharpe))
            results['dividend_yields'].append(float(portfolio_dividend_yield))
            results['weights'].append(weights.tolist())
        
        # Find max Sharpe ratio portfolio
        max_sharpe_idx = np.argmax(results['sharpe_ratios'])
        
        # Find min volatility portfolio
        min_vol_idx = np.argmin(results['volatilities'])
        
        return {
            'portfolios': results,
            'max_sharpe_portfolio': {
                'return': results['returns'][max_sharpe_idx],
                'volatility': results['volatilities'][max_sharpe_idx],
                'sharpe_ratio': results['sharpe_ratios'][max_sharpe_idx],
                'dividend_yield': results['dividend_yields'][max_sharpe_idx],
                'weights': results['weights'][max_sharpe_idx]
            },
            'min_volatility_portfolio': {
                'return': results['returns'][min_vol_idx],
                'volatility': results['volatilities'][min_vol_idx],
                'sharpe_ratio': results['sharpe_ratios'][min_vol_idx],
                'dividend_yield': results['dividend_yields'][min_vol_idx],
                'weights': results['weights'][min_vol_idx]
            }
        }
    
    def calculate_correlation_matrix(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix"""
        std_devs = np.sqrt(np.diag(covariance_matrix))
        correlation = covariance_matrix / np.outer(std_devs, std_devs)
        return correlation
    
    def calculate_diversification_ratio(self,
                                        weights: np.ndarray,
                                        volatilities: np.ndarray,
                                        portfolio_volatility: float) -> float:
        """
        Calculate diversification ratio
        Higher ratio indicates better diversification
        """
        weighted_avg_volatility = np.dot(weights, volatilities)
        diversification_ratio = weighted_avg_volatility / portfolio_volatility if portfolio_volatility > 0 else 1.0
        return float(diversification_ratio)
    
    def calculate_sector_allocation(self,
                                    weights: np.ndarray,
                                    stocks: List[Dict]) -> Dict:
        """Calculate allocation by sector"""
        sector_weights = {}
        
        for i, stock in enumerate(stocks):
            sector = stock['sector']
            if sector not in sector_weights:
                sector_weights[sector] = 0.0
            sector_weights[sector] += float(weights[i])
        
        return sector_weights
    
    def calculate_tax_efficiency_score(self,
                                       weights: np.ndarray,
                                       dividend_yields: np.ndarray,
                                       capital_gains_yields: np.ndarray,
                                       marginal_tax_rate: float = 0.50) -> Dict:
        """
        Calculate tax efficiency score for Canadian investors
        Eligible dividends are tax-advantaged vs interest/capital gains
        """
        # Canadian dividend tax credit makes dividends more efficient
        # Eligible dividend effective tax rate ~30% vs 50% on interest
        
        portfolio_dividend_income = np.dot(weights, dividend_yields)
        portfolio_capital_gains = np.dot(weights, capital_gains_yields)
        
        # Effective tax rates
        dividend_tax_rate = 0.30  # After dividend tax credit
        capital_gains_tax_rate = marginal_tax_rate * 0.5  # Only 50% taxable
        
        # After-tax income
        after_tax_dividend = portfolio_dividend_income * (1 - dividend_tax_rate)
        after_tax_capital_gains = portfolio_capital_gains * (1 - capital_gains_tax_rate)
        
        total_after_tax = after_tax_dividend + after_tax_capital_gains
        total_before_tax = portfolio_dividend_income + portfolio_capital_gains
        
        tax_efficiency = (total_after_tax / total_before_tax * 100) if total_before_tax > 0 else 100
        
        return {
            'tax_efficiency_score': float(tax_efficiency),
            'dividend_income_pct': float(portfolio_dividend_income / total_before_tax * 100) if total_before_tax > 0 else 0,
            'capital_gains_pct': float(portfolio_capital_gains / total_before_tax * 100) if total_before_tax > 0 else 0,
            'estimated_tax_savings': float((total_before_tax - total_after_tax) * 100)
        }
    
    def performance_attribution(self,
                               weights: np.ndarray,
                               returns: np.ndarray,
                               dividend_yields: np.ndarray,
                               stocks: List[Dict]) -> Dict:
        """
        Perform performance attribution analysis
        Break down returns by sector and security
        """
        total_return = np.dot(weights, returns)
        total_dividend = np.dot(weights, dividend_yields)
        
        # Sector attribution
        sector_contribution = {}
        for i, stock in enumerate(stocks):
            sector = stock['sector']
            if sector not in sector_contribution:
                sector_contribution[sector] = {
                    'return': 0.0,
                    'dividend': 0.0,
                    'weight': 0.0
                }
            
            sector_contribution[sector]['return'] += float(weights[i] * returns[i])
            sector_contribution[sector]['dividend'] += float(weights[i] * dividend_yields[i])
            sector_contribution[sector]['weight'] += float(weights[i])
        
        # Top contributors
        security_contributions = []
        for i, stock in enumerate(stocks):
            security_contributions.append({
                'symbol': stock['symbol'],
                'name': stock['name'],
                'weight': float(weights[i]),
                'return_contribution': float(weights[i] * returns[i]),
                'dividend_contribution': float(weights[i] * dividend_yields[i])
            })
        
        # Sort by total contribution
        security_contributions.sort(
            key=lambda x: x['return_contribution'] + x['dividend_contribution'],
            reverse=True
        )
        
        return {
            'total_return': float(total_return * 100),
            'total_dividend_yield': float(total_dividend),
            'sector_attribution': sector_contribution,
            'top_contributors': security_contributions[:10]
        }

    def calculate_shrunk_covariance(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate robust covariance matrix using Ledoit-Wolf shrinkage
        This reduces estimation error and improves out-of-sample performance
        """
        lw = LedoitWolf()
        shrunk_cov = lw.fit(returns).covariance_
        return shrunk_cov

    def optimize_sharpe_ratio_ml(self,
                                expected_returns: np.ndarray,
                                covariance_matrix: np.ndarray,
                                risk_free_rate: float = 0.04) -> Dict:
        """
        Optimize portfolio for maximum Sharpe ratio using ML-enhanced covariance
        """
        num_assets = len(expected_returns)
        args = (expected_returns, covariance_matrix, risk_free_rate)
        
        def neg_sharpe_ratio(weights, expected_returns, cov_matrix, rf_rate):
            p_ret = np.dot(weights, expected_returns)
            p_var = np.dot(weights, np.dot(cov_matrix, weights))
            p_vol = np.sqrt(p_var)
            return -(p_ret - rf_rate) / p_vol

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        initial_guess = num_assets * [1. / num_assets,]
        
        result = minimize(neg_sharpe_ratio, initial_guess, args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
        
        return {
            'weights': result.x,
            'sharpe_ratio': -result.fun,
            'success': result.success
        }
