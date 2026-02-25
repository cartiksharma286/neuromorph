"""
Statistical Learning through Distributions Optimizing SNR (Signal-to-Noise Ratio)

This module implements statistical learning algorithms that optimize SNR
by learning optimal probability distributions for signal representation.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict
import warnings


class SNROptimizer:
    """
    Optimizes signal-to-noise ratio through distribution learning.
    
    The optimizer learns the optimal parameters of a probability distribution
    that maximizes the SNR of a signal in the presence of noise.
    """
    
    def __init__(self, distribution_type: str = 'gaussian'):
        """
        Initialize the SNR optimizer.
        
        Args:
            distribution_type: Type of distribution ('gaussian', 'laplace', 'student_t')
        """
        self.distribution_type = distribution_type
        self.params = None
        self.snr_history = []
        
    def compute_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """
        Compute Signal-to-Noise Ratio.
        
        SNR = 10 * log10(P_signal / P_noise)
        where P is power (variance for zero-mean signals)
        
        Args:
            signal: Clean signal
            noise: Noise component
            
        Returns:
            SNR in dB
        """
        signal_power = np.var(signal)
        noise_power = np.var(noise)
        
        if noise_power == 0:
            return np.inf
            
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        return snr_db
    
    def fit_distribution(self, data: np.ndarray) -> Dict:
        """
        Fit a probability distribution to the data.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary containing distribution parameters
        """
        if self.distribution_type == 'gaussian':
            mu = np.mean(data)
            sigma = np.std(data)
            params = {'mu': mu, 'sigma': sigma}
            
        elif self.distribution_type == 'laplace':
            # MLE for Laplace distribution
            mu = np.median(data)
            b = np.mean(np.abs(data - mu))
            params = {'mu': mu, 'b': b}
            
        elif self.distribution_type == 'student_t':
            # Fit Student's t-distribution
            df, loc, scale = stats.t.fit(data)
            params = {'df': df, 'loc': loc, 'scale': scale}
            
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")
            
        return params
    
    def sample_from_distribution(self, params: Dict, size: int) -> np.ndarray:
        """
        Sample from the learned distribution.
        
        Args:
            params: Distribution parameters
            size: Number of samples to generate
            
        Returns:
            Array of samples
        """
        if self.distribution_type == 'gaussian':
            return np.random.normal(params['mu'], params['sigma'], size)
            
        elif self.distribution_type == 'laplace':
            return np.random.laplace(params['mu'], params['b'], size)
            
        elif self.distribution_type == 'student_t':
            return stats.t.rvs(params['df'], loc=params['loc'], 
                              scale=params['scale'], size=size)
    
    def optimize_snr_params(self, noisy_signal: np.ndarray, 
                           true_signal: Optional[np.ndarray] = None) -> Dict:
        """
        Optimize distribution parameters to maximize SNR.
        
        Args:
            noisy_signal: Observed noisy signal
            true_signal: Optional ground truth signal for supervised learning
            
        Returns:
            Optimized parameters
        """
        # Initial parameter estimation
        initial_params = self.fit_distribution(noisy_signal)
        
        if self.distribution_type == 'gaussian':
            x0 = [initial_params['mu'], initial_params['sigma']]
            
            def objective(x):
                mu, sigma = x
                # Generate denoised estimate based on distribution
                denoised = self._denoise_signal(noisy_signal, {'mu': mu, 'sigma': sigma})
                
                if true_signal is not None:
                    # Supervised: maximize SNR relative to true signal
                    noise = noisy_signal - true_signal
                    signal = true_signal
                else:
                    # Unsupervised: estimate noise as deviation from distribution mean
                    noise = noisy_signal - denoised
                    signal = denoised
                
                snr = self.compute_snr(signal, noise)
                return -snr  # Negative because we minimize
            
            # Optimization with constraints
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 1000})
            
            optimized_params = {'mu': result.x[0], 'sigma': abs(result.x[1])}
            
        else:
            # For other distributions, use initial MLE estimates
            optimized_params = initial_params
            
        self.params = optimized_params
        return optimized_params
    
    def _denoise_signal(self, noisy_signal: np.ndarray, params: Dict) -> np.ndarray:
        """
        Denoise signal using Wiener-like filtering based on distribution.
        
        Args:
            noisy_signal: Noisy input signal
            params: Distribution parameters
            
        Returns:
            Denoised signal estimate
        """
        if self.distribution_type == 'gaussian':
            # Simple Wiener-like filter: shrink towards mean
            mu = params['mu']
            sigma = params['sigma']
            
            # Estimate noise variance (robust estimator)
            noise_var = np.median(np.abs(noisy_signal - np.median(noisy_signal))) / 0.6745
            noise_var = noise_var ** 2
            
            # Wiener gain
            signal_var = sigma ** 2
            gain = signal_var / (signal_var + noise_var)
            
            # Apply filter
            denoised = mu + gain * (noisy_signal - mu)
            return denoised
        
        return noisy_signal
    
    def learn_optimal_distribution(self, signal: np.ndarray, 
                                   noise: np.ndarray,
                                   iterations: int = 10) -> Dict:
        """
        Iteratively learn the optimal distribution to maximize SNR.
        
        Args:
            signal: Clean signal (for training/validation)
            noise: Noise to be added
            iterations: Number of learning iterations
            
        Returns:
            Learned optimal parameters
        """
        noisy_signal = signal + noise
        self.snr_history = []
        
        for i in range(iterations):
            # Optimize parameters
            params = self.optimize_snr_params(noisy_signal, true_signal=signal)
            
            # Compute achieved SNR
            denoised = self._denoise_signal(noisy_signal, params)
            estimated_noise = noisy_signal - denoised
            snr = self.compute_snr(signal, estimated_noise)
            
            self.snr_history.append(snr)
            
        return params


class AdaptiveSNRLearner:
    """
    Adaptive statistical learning for SNR optimization across multiple distributions.
    
    This class compares multiple distribution types and selects the best one
    for maximizing SNR.
    """
    
    def __init__(self):
        """Initialize the adaptive learner."""
        self.distribution_types = ['gaussian', 'laplace', 'student_t']
        self.optimizers = {dt: SNROptimizer(dt) for dt in self.distribution_types}
        self.best_distribution = None
        self.best_params = None
        self.best_snr = -np.inf
        
    def fit(self, signal: np.ndarray, noise: np.ndarray) -> Dict:
        """
        Fit all distribution types and select the best.
        
        Args:
            signal: Clean signal
            noise: Noise component
            
        Returns:
            Dictionary with best distribution type and parameters
        """
        results = {}
        
        for dist_type, optimizer in self.optimizers.items():
            params = optimizer.learn_optimal_distribution(signal, noise, iterations=5)
            
            # Evaluate final SNR
            noisy = signal + noise
            denoised = optimizer._denoise_signal(noisy, params)
            estimated_noise = noisy - denoised
            snr = optimizer.compute_snr(signal, estimated_noise)
            
            results[dist_type] = {
                'params': params,
                'snr': snr,
                'optimizer': optimizer
            }
            
            if snr > self.best_snr:
                self.best_snr = snr
                self.best_distribution = dist_type
                self.best_params = params
        
        return {
            'best_distribution': self.best_distribution,
            'best_params': self.best_params,
            'best_snr': self.best_snr,
            'all_results': results
        }
    
    def denoise(self, noisy_signal: np.ndarray) -> np.ndarray:
        """
        Denoise a signal using the best learned distribution.
        
        Args:
            noisy_signal: Noisy input signal
            
        Returns:
            Denoised signal
        """
        if self.best_distribution is None:
            raise ValueError("Must call fit() before denoise()")
            
        optimizer = self.optimizers[self.best_distribution]
        return optimizer._denoise_signal(noisy_signal, self.best_params)
