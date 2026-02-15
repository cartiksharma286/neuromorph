"""
Unit tests for SNR Optimizer module
"""

import numpy as np
import pytest
from snr_optimizer import SNROptimizer, AdaptiveSNRLearner


class TestSNROptimizer:
    """Test cases for SNROptimizer class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        np.random.seed(42)
        self.n_samples = 100
        self.signal = np.sin(np.linspace(0, 4*np.pi, self.n_samples))
        self.noise = np.random.normal(0, 0.3, self.n_samples)
        
    def test_compute_snr(self):
        """Test SNR computation."""
        optimizer = SNROptimizer()
        snr = optimizer.compute_snr(self.signal, self.noise)
        
        # SNR should be positive for our test case
        assert snr > 0, "SNR should be positive"
        assert isinstance(snr, (float, np.floating)), "SNR should be a float"
        
    def test_fit_gaussian_distribution(self):
        """Test Gaussian distribution fitting."""
        optimizer = SNROptimizer(distribution_type='gaussian')
        params = optimizer.fit_distribution(self.signal)
        
        assert 'mu' in params, "Should have mu parameter"
        assert 'sigma' in params, "Should have sigma parameter"
        assert params['sigma'] > 0, "Sigma should be positive"
        
    def test_fit_laplace_distribution(self):
        """Test Laplace distribution fitting."""
        optimizer = SNROptimizer(distribution_type='laplace')
        params = optimizer.fit_distribution(self.signal)
        
        assert 'mu' in params, "Should have mu parameter"
        assert 'b' in params, "Should have b parameter"
        assert params['b'] > 0, "b should be positive"
        
    def test_fit_student_t_distribution(self):
        """Test Student's t distribution fitting."""
        optimizer = SNROptimizer(distribution_type='student_t')
        params = optimizer.fit_distribution(self.signal)
        
        assert 'df' in params, "Should have df parameter"
        assert 'loc' in params, "Should have loc parameter"
        assert 'scale' in params, "Should have scale parameter"
        assert params['df'] > 0, "df should be positive"
        
    def test_sample_from_distribution(self):
        """Test sampling from learned distribution."""
        optimizer = SNROptimizer(distribution_type='gaussian')
        params = optimizer.fit_distribution(self.signal)
        samples = optimizer.sample_from_distribution(params, 50)
        
        assert len(samples) == 50, "Should generate requested number of samples"
        assert isinstance(samples, np.ndarray), "Should return numpy array"
        
    def test_learn_optimal_distribution(self):
        """Test optimal distribution learning."""
        optimizer = SNROptimizer(distribution_type='gaussian')
        params = optimizer.learn_optimal_distribution(self.signal, self.noise, iterations=5)
        
        assert params is not None, "Should return parameters"
        assert len(optimizer.snr_history) == 5, "Should track SNR history"
        
    def test_denoise_signal(self):
        """Test signal denoising."""
        optimizer = SNROptimizer(distribution_type='gaussian')
        params = optimizer.fit_distribution(self.signal)
        
        noisy = self.signal + self.noise
        denoised = optimizer._denoise_signal(noisy, params)
        
        assert len(denoised) == len(noisy), "Should preserve signal length"
        
        # Denoised should be closer to original than noisy
        mse_noisy = np.mean((noisy - self.signal)**2)
        mse_denoised = np.mean((denoised - self.signal)**2)
        assert mse_denoised < mse_noisy, "Denoising should reduce MSE"


class TestAdaptiveSNRLearner:
    """Test cases for AdaptiveSNRLearner class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        np.random.seed(42)
        self.n_samples = 100
        self.signal = np.sin(np.linspace(0, 4*np.pi, self.n_samples))
        self.noise = np.random.normal(0, 0.3, self.n_samples)
        
    def test_initialization(self):
        """Test learner initialization."""
        learner = AdaptiveSNRLearner()
        
        assert len(learner.distribution_types) > 0, "Should have distribution types"
        assert len(learner.optimizers) > 0, "Should have optimizers"
        assert learner.best_distribution is None, "Should not have best before fitting"
        
    def test_fit(self):
        """Test adaptive fitting."""
        learner = AdaptiveSNRLearner()
        results = learner.fit(self.signal, self.noise)
        
        assert 'best_distribution' in results, "Should return best distribution"
        assert 'best_params' in results, "Should return best parameters"
        assert 'best_snr' in results, "Should return best SNR"
        assert 'all_results' in results, "Should return all results"
        
        assert learner.best_distribution is not None, "Should select a distribution"
        assert learner.best_snr > -np.inf, "Should have valid SNR"
        
    def test_denoise(self):
        """Test denoising with best model."""
        learner = AdaptiveSNRLearner()
        learner.fit(self.signal, self.noise)
        
        noisy = self.signal + self.noise
        denoised = learner.denoise(noisy)
        
        assert len(denoised) == len(noisy), "Should preserve length"
        
        # Denoised should be closer to original
        mse_noisy = np.mean((noisy - self.signal)**2)
        mse_denoised = np.mean((denoised - self.signal)**2)
        assert mse_denoised < mse_noisy, "Should improve signal quality"
        
    def test_denoise_before_fit_raises_error(self):
        """Test that denoising before fitting raises error."""
        learner = AdaptiveSNRLearner()
        noisy = self.signal + self.noise
        
        with pytest.raises(ValueError):
            learner.denoise(noisy)


def test_snr_improvement():
    """Integration test: verify SNR improvement."""
    np.random.seed(42)
    
    # Create test signal
    t = np.linspace(0, 4*np.pi, 500)
    signal = np.sin(t) + 0.5*np.sin(3*t)
    noise = np.random.normal(0, 0.4, len(signal))
    
    # Compute initial SNR
    initial_snr = 10 * np.log10(np.var(signal) / np.var(noise))
    
    # Optimize
    optimizer = SNROptimizer(distribution_type='gaussian')
    params = optimizer.learn_optimal_distribution(signal, noise, iterations=5)
    
    # Check improvement
    noisy = signal + noise
    denoised = optimizer._denoise_signal(noisy, params)
    estimated_noise = noisy - denoised
    final_snr = optimizer.compute_snr(signal, estimated_noise)
    
    print(f"\nIntegration test - Initial SNR: {initial_snr:.2f} dB, Final SNR: {final_snr:.2f} dB")
    assert final_snr > initial_snr, "SNR should improve after optimization"


if __name__ == "__main__":
    # Run tests without pytest
    print("Running SNR Optimizer Tests\n")
    
    test_optimizer = TestSNROptimizer()
    test_optimizer.setup_method()
    
    print("Testing SNR computation...")
    test_optimizer.test_compute_snr()
    print("✓ SNR computation test passed")
    
    print("Testing Gaussian distribution fitting...")
    test_optimizer.test_fit_gaussian_distribution()
    print("✓ Gaussian fitting test passed")
    
    print("Testing denoising...")
    test_optimizer.test_denoise_signal()
    print("✓ Denoising test passed")
    
    print("\nTesting Adaptive Learner...")
    test_adaptive = TestAdaptiveSNRLearner()
    test_adaptive.setup_method()
    
    print("Testing adaptive fitting...")
    test_adaptive.test_fit()
    print("✓ Adaptive fitting test passed")
    
    print("Testing adaptive denoising...")
    test_adaptive.test_denoise()
    print("✓ Adaptive denoising test passed")
    
    print("\nRunning integration test...")
    test_snr_improvement()
    print("✓ Integration test passed")
    
    print("\n" + "="*50)
    print("All tests passed successfully!")
    print("="*50)
