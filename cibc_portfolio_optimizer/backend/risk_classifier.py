"""
Risk Statistical Classifiers and Parametric Estimates
Advanced risk modeling for portfolio optimization
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class RiskClassifier:
    """Statistical risk classification and parametric estimation"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.risk_classes = ['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']
        
    def classify_asset_risk(self, returns: np.ndarray, volatility: float, 
                           beta: float, var_95: float) -> Dict:
        """
        Classify asset risk using multiple statistical measures
        
        Args:
            returns: Historical returns
            volatility: Annualized volatility
            beta: Market beta
            var_95: Value at Risk at 95% confidence
        
        Returns:
            Risk classification with confidence scores
        """
        # Feature vector for classification
        features = np.array([
            volatility,
            abs(beta),
            var_95,
            self._calculate_downside_deviation(returns),
            self._calculate_tail_risk(returns)
        ])
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features.reshape(1, -1))[0]
        
        # Calculate composite risk score (0-100)
        risk_score = self._calculate_composite_risk_score(features_normalized)
        
        # Classify into risk buckets
        if risk_score < 25:
            risk_class = 'Low Risk'
            color = '#10B981'
        elif risk_score < 50:
            risk_class = 'Moderate Risk'
            color = '#F59E0B'
        elif risk_score < 75:
            risk_class = 'High Risk'
            color = '#EF4444'
        else:
            risk_class = 'Very High Risk'
            color = '#DC2626'
        
        return {
            'risk_class': risk_class,
            'risk_score': float(risk_score),
            'confidence': self._calculate_classification_confidence(features),
            'color': color,
            'features': {
                'volatility': float(volatility),
                'beta': float(beta),
                'var_95': float(var_95),
                'downside_deviation': float(self._calculate_downside_deviation(returns)),
                'tail_risk': float(self._calculate_tail_risk(returns))
            }
        }
    
    def fit_parametric_distribution(self, returns: np.ndarray) -> Dict:
        """
        Fit parametric distributions to return data
        
        Tests: Normal, Student's t, Skewed Normal, Generalized Extreme Value
        
        Returns:
            Best-fit distribution with parameters
        """
        distributions = {}
        
        # 1. Normal Distribution
        mu, sigma = stats.norm.fit(returns)
        distributions['normal'] = {
            'name': 'Normal',
            'params': {'mu': mu, 'sigma': sigma},
            'aic': self._calculate_aic(returns, stats.norm, (mu, sigma)),
            'bic': self._calculate_bic(returns, stats.norm, (mu, sigma))
        }
        
        # 2. Student's t Distribution (heavy tails)
        df, loc, scale = stats.t.fit(returns)
        distributions['t'] = {
            'name': "Student's t",
            'params': {'df': df, 'loc': loc, 'scale': scale},
            'aic': self._calculate_aic(returns, stats.t, (df, loc, scale)),
            'bic': self._calculate_bic(returns, stats.t, (df, loc, scale))
        }
        
        # 3. Skewed Normal Distribution
        a, loc, scale = stats.skewnorm.fit(returns)
        distributions['skewnorm'] = {
            'name': 'Skewed Normal',
            'params': {'skewness': a, 'loc': loc, 'scale': scale},
            'aic': self._calculate_aic(returns, stats.skewnorm, (a, loc, scale)),
            'bic': self._calculate_bic(returns, stats.skewnorm, (a, loc, scale))
        }
        
        # 4. Generalized Extreme Value (for tail events)
        c, loc, scale = stats.genextreme.fit(returns)
        distributions['gev'] = {
            'name': 'Generalized Extreme Value',
            'params': {'shape': c, 'loc': loc, 'scale': scale},
            'aic': self._calculate_aic(returns, stats.genextreme, (c, loc, scale)),
            'bic': self._calculate_bic(returns, stats.genextreme, (c, loc, scale))
        }
        
        # Select best distribution (lowest BIC)
        best_dist = min(distributions.items(), key=lambda x: x[1]['bic'])
        
        return {
            'best_distribution': best_dist[0],
            'best_params': best_dist[1]['params'],
            'all_distributions': distributions,
            'goodness_of_fit': self._kolmogorov_smirnov_test(returns, best_dist[0], best_dist[1]['params']),
            'statistical_congruence': float(np.sum(np.power(returns, 2)) % 1.0)
        }
    
    def estimate_conditional_var(self, returns: np.ndarray, 
                                 confidence_level: float = 0.95,
                                 method: str = 'parametric') -> Dict:
        """
        Estimate Conditional Value at Risk (CVaR) using various methods
        
        Methods: parametric, historical, cornish-fisher
        
        Returns:
            CVaR estimates with confidence intervals
        """
        results = {}
        
        if method == 'parametric':
            # Parametric CVaR (assumes normal distribution)
            mu = np.mean(returns)
            sigma = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mu + z_score * sigma
            cvar = mu - sigma * stats.norm.pdf(z_score) / (1 - confidence_level)
            
            results['method'] = 'Parametric (Normal)'
            results['var'] = float(var)
            results['cvar'] = float(cvar)
            
        elif method == 'historical':
            # Historical CVaR
            sorted_returns = np.sort(returns)
            cutoff_index = int(len(returns) * (1 - confidence_level))
            var = sorted_returns[cutoff_index]
            cvar = np.mean(sorted_returns[:cutoff_index])
            
            results['method'] = 'Historical Simulation'
            results['var'] = float(var)
            results['cvar'] = float(cvar)
            
        elif method == 'cornish-fisher':
            # Cornish-Fisher expansion (accounts for skewness and kurtosis)
            mu = np.mean(returns)
            sigma = np.std(returns)
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)
            
            z = stats.norm.ppf(1 - confidence_level)
            z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24 - (2*z**3 - 5*z) * skew**2 / 36
            
            var = mu + z_cf * sigma
            cvar = mu - sigma * stats.norm.pdf(z_cf) / (1 - confidence_level)
            
            results['method'] = 'Cornish-Fisher'
            results['var'] = float(var)
            results['cvar'] = float(cvar)
            results['skewness'] = float(skew)
            results['kurtosis'] = float(kurt)
        
        # Add confidence intervals
        results['confidence_interval'] = self._bootstrap_cvar_ci(returns, confidence_level)
        
        return results
    
    def detect_regime_changes(self, returns: np.ndarray, n_regimes: int = 3) -> Dict:
        """
        Detect market regime changes using Gaussian Mixture Models
        
        Args:
            returns: Historical returns
            n_regimes: Number of market regimes to detect
        
        Returns:
            Regime classification and transition probabilities
        """
        # Reshape for GMM
        X = returns.reshape(-1, 1)
        
        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        gmm.fit(X)
        
        # Predict regimes
        regimes = gmm.predict(X)
        regime_probs = gmm.predict_proba(X)
        
        # Calculate regime statistics
        regime_stats = []
        for i in range(n_regimes):
            regime_returns = returns[regimes == i]
            regime_stats.append({
                'regime': i,
                'mean_return': float(np.mean(regime_returns)),
                'volatility': float(np.std(regime_returns)),
                'frequency': float(np.sum(regimes == i) / len(regimes)),
                'label': self._label_regime(np.mean(regime_returns), np.std(regime_returns))
            })
        
        # Sort by mean return
        regime_stats.sort(key=lambda x: x['mean_return'])
        
        # Calculate transition matrix
        transition_matrix = self._calculate_transition_matrix(regimes, n_regimes)
        
        return {
            'n_regimes': n_regimes,
            'current_regime': int(regimes[-1]),
            'regime_probability': float(regime_probs[-1][regimes[-1]]),
            'regime_stats': regime_stats,
            'transition_matrix': transition_matrix.tolist(),
            'bic': float(gmm.bic(X)),
            'aic': float(gmm.aic(X))
        }
    
    def calculate_parametric_risk_metrics(self, returns: np.ndarray, 
                                         portfolio_value: float = 100000) -> Dict:
        """
        Calculate comprehensive parametric risk metrics
        
        Returns:
            Dictionary with all parametric risk estimates
        """
        mu = np.mean(returns)
        sigma = np.std(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        # Fit best distribution
        dist_fit = self.fit_parametric_distribution(returns)
        
        # Calculate various VaR/CVaR estimates
        var_cvar_parametric = self.estimate_conditional_var(returns, 0.95, 'parametric')
        var_cvar_historical = self.estimate_conditional_var(returns, 0.95, 'historical')
        var_cvar_cf = self.estimate_conditional_var(returns, 0.95, 'cornish-fisher')
        
        # Regime detection
        regimes = self.detect_regime_changes(returns, n_regimes=3)
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar Ratio
        calmar_ratio = mu / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Omega Ratio (probability-weighted ratio of gains vs losses)
        threshold = 0
        omega_ratio = self._calculate_omega_ratio(returns, threshold)
        
        return {
            'moments': {
                'mean': float(mu),
                'std_dev': float(sigma),
                'skewness': float(skew),
                'kurtosis': float(kurt),
                'excess_kurtosis': float(kurt)
            },
            'distribution_fit': dist_fit,
            'var_cvar': {
                'parametric': var_cvar_parametric,
                'historical': var_cvar_historical,
                'cornish_fisher': var_cvar_cf
            },
            'regime_analysis': regimes,
            'drawdown': {
                'max_drawdown': float(max_drawdown),
                'max_drawdown_dollars': float(max_drawdown * portfolio_value),
                'calmar_ratio': float(calmar_ratio)
            },
            'omega_ratio': float(omega_ratio),
            'tail_risk': {
                'left_tail_index': self._estimate_tail_index(returns[returns < 0]),
                'right_tail_index': self._estimate_tail_index(-returns[returns > 0])
            }
        }
    
    # Helper methods
    
    def _calculate_downside_deviation(self, returns: np.ndarray, 
                                      target_return: float = 0.0) -> float:
        """Calculate downside deviation (semi-deviation)"""
        downside_returns = returns[returns < target_return] - target_return
        return np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0.0
    
    def _calculate_tail_risk(self, returns: np.ndarray, percentile: float = 5) -> float:
        """Calculate tail risk (average of worst percentile returns)"""
        cutoff = np.percentile(returns, percentile)
        tail_returns = returns[returns <= cutoff]
        return abs(np.mean(tail_returns)) if len(tail_returns) > 0 else 0.0
    
    def _calculate_composite_risk_score(self, features: np.ndarray) -> float:
        """Calculate composite risk score from normalized features"""
        # Weighted average of risk features
        weights = np.array([0.3, 0.2, 0.25, 0.15, 0.1])  # Volatility, Beta, VaR, Downside, Tail
        score = np.dot(features, weights)
        # Normalize to 0-100
        return float(np.clip((score + 3) / 6 * 100, 0, 100))
    
    def _calculate_classification_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence in risk classification"""
        # Based on feature consistency
        feature_std = np.std(features)
        confidence = 1.0 / (1.0 + feature_std)
        return float(np.clip(confidence, 0, 1))
    
    def _calculate_aic(self, data: np.ndarray, dist, params: tuple) -> float:
        """Calculate Akaike Information Criterion"""
        log_likelihood = np.sum(np.log(dist.pdf(data, *params) + 1e-10))
        k = len(params)
        return 2 * k - 2 * log_likelihood
    
    def _calculate_bic(self, data: np.ndarray, dist, params: tuple) -> float:
        """Calculate Bayesian Information Criterion"""
        log_likelihood = np.sum(np.log(dist.pdf(data, *params) + 1e-10))
        k = len(params)
        n = len(data)
        return k * np.log(n) - 2 * log_likelihood
    
    def _kolmogorov_smirnov_test(self, data: np.ndarray, dist_name: str, 
                                 params: Dict) -> Dict:
        """Perform Kolmogorov-Smirnov goodness-of-fit test"""
        if dist_name == 'normal':
            ks_stat, p_value = stats.kstest(data, 'norm', args=(params['mu'], params['sigma']))
        elif dist_name == 't':
            ks_stat, p_value = stats.kstest(data, 't', args=(params['df'], params['loc'], params['scale']))
        elif dist_name == 'skewnorm':
            ks_stat, p_value = stats.kstest(data, 'skewnorm', args=(params['skewness'], params['loc'], params['scale']))
        else:
            ks_stat, p_value = 0.0, 1.0
        
        return {
            'ks_statistic': float(ks_stat),
            'p_value': float(p_value),
            'reject_null': bool(p_value < 0.05)
        }
    
    def _bootstrap_cvar_ci(self, returns: np.ndarray, confidence_level: float, 
                          n_bootstrap: int = 1000) -> Dict:
        """Bootstrap confidence intervals for CVaR"""
        cvar_samples = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            sorted_sample = np.sort(sample)
            cutoff_index = int(len(sample) * (1 - confidence_level))
            cvar = np.mean(sorted_sample[:cutoff_index])
            cvar_samples.append(cvar)
        
        cvar_samples = np.array(cvar_samples)
        
        return {
            'lower_95': float(np.percentile(cvar_samples, 2.5)),
            'upper_95': float(np.percentile(cvar_samples, 97.5)),
            'mean': float(np.mean(cvar_samples)),
            'std': float(np.std(cvar_samples))
        }
    
    def _calculate_transition_matrix(self, regimes: np.ndarray, n_regimes: int) -> np.ndarray:
        """Calculate regime transition probability matrix"""
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regimes) - 1):
            current_regime = regimes[i]
            next_regime = regimes[i + 1]
            transition_matrix[current_regime, next_regime] += 1
        
        # Normalize rows to get probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                     where=row_sums != 0, out=np.zeros_like(transition_matrix))
        
        return transition_matrix
    
    def _label_regime(self, mean_return: float, volatility: float) -> str:
        """Label market regime based on characteristics"""
        if mean_return > 0.001 and volatility < 0.015:
            return "Bull Market (Low Vol)"
        elif mean_return > 0.001 and volatility >= 0.015:
            return "Bull Market (High Vol)"
        elif mean_return < -0.001 and volatility >= 0.015:
            return "Bear Market"
        else:
            return "Sideways Market"
    
    def _calculate_omega_ratio(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if len(losses) == 0 or np.sum(losses) == 0:
            return float('inf')
        
        return np.sum(gains) / np.sum(losses) if len(gains) > 0 else 0.0
    
    def _estimate_tail_index(self, tail_data: np.ndarray) -> float:
        """Estimate tail index using Hill estimator"""
        if len(tail_data) < 10:
            return 0.0
        
        sorted_data = np.sort(np.abs(tail_data))[::-1]
        k = min(int(len(sorted_data) * 0.1), 50)  # Use top 10% or 50 observations
        
        if k < 2:
            return 0.0
        
        hill_estimator = np.mean(np.log(sorted_data[:k])) - np.log(sorted_data[k])
        tail_index = 1.0 / hill_estimator if hill_estimator > 0 else 0.0
        
        return float(tail_index)
