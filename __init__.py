"""
Statistical Learning through Distributions - SNR Optimization

This package implements statistical learning algorithms for optimizing
signal-to-noise ratio (SNR) through probability distribution modeling.

Key Features:
- SNR-optimized distribution learning (Gaussian, Laplace, Student's t)
- Adaptive distribution selection
- Wiener-like filtering for denoising
- Comprehensive visualization and demos

Author: Generated for statistical learning research
"""

__version__ = "1.0.0"

from .snr_optimizer import SNROptimizer, AdaptiveSNRLearner

__all__ = ['SNROptimizer', 'AdaptiveSNRLearner']
