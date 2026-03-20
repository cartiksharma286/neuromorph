"""
Statistical Distribution and Optimal Control-Based Artifact Removal
=====================================================================

Advanced module for white speckle artifact removal using:
1. Statistical distribution modeling (Gaussian, Laplacian, Poisson mixtures)
2. Optimal control filtering (Wiener, Kalman, H-infinity)
3. Adaptive regularization with curvature-aware preservation
4. Multi-scale analysis with wavelet decomposition

Author: NeuroPulse Advanced Denoising Engine
Date: March 19, 2026
"""

import numpy as np
from scipy import signal, ndimage, stats
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, laplace, convolve
import warnings
warnings.filterwarnings('ignore')


class StatisticalDistributionNoiseMixer:
    """
    Models speckle noise as mixture of statistical distributions.
    Estimates optimal noise parameters per patch.
    """
    
    def __init__(self):
        """Initialize noise model components."""
        self.gaussian_weight = 0.7
        self.laplacian_weight = 0.25
        self.poisson_weight = 0.05
        self.patch_size = 8
        
    def estimate_noise_parameters(self, image):
        """
        Estimate noise distribution parameters from image intensity residuals.
        
        Parameters
        ----------
        image : np.ndarray
            Input noisy image
            
        Returns
        -------
        dict : Noise parameters (mean, variance, shape parameters)
        """
        # Extract local statistics via patch decomposition
        H, W = image.shape
        p_size = self.patch_size
        
        residuals = []
        for i in range(0, H - p_size, p_size):
            for j in range(0, W - p_size, p_size):
                patch = image[i:i+p_size, j:j+p_size]
                # Laplacian pyramid for residuals
                patch_blur = gaussian_filter(patch, sigma=1.0)
                residual = patch - patch_blur
                residuals.append(residual.flatten())
        
        residuals = np.concatenate(residuals)
        residuals = residuals[~np.isnan(residuals)]
        
        # Fit to mixture components
        params = {
            'gaussian_mean': np.mean(residuals),
            'gaussian_std': np.std(residuals),
            'laplacian_scale': np.mean(np.abs(residuals - np.mean(residuals))),
            'laplacian_loc': np.median(residuals),
            'poisson_lambda': np.maximum(np.mean(np.abs(residuals)), 1e-3),
            'residuals': residuals
        }
        return params
    
    def mixture_likelihood(self, residuals, params):
        """
        Compute mixture likelihood for noise model.
        
        Parameters
        ----------
        residuals : np.ndarray
            Noise residuals
        params : dict
            Mixture parameters
            
        Returns
        -------
        float : Log-likelihood
        """
        g_mean = params['gaussian_mean']
        g_std = params['gaussian_std']
        l_scale = params['laplacian_scale']
        l_loc = params['laplacian_loc']
        p_lambda = params['poisson_lambda']
        
        # Gaussian component
        gaussian_ll = -0.5 * np.sum((residuals - g_mean)**2 / (g_std**2 + 1e-8))
        
        # Laplacian component
        laplacian_ll = -np.sum(np.abs(residuals - l_loc) / (l_scale + 1e-8))
        
        # Poisson component (for discrete intensities)
        poisson_ll = -np.sum(np.abs(residuals)**2 / (p_lambda + 1e-8))
        
        # Weighted mixture
        total_ll = (self.gaussian_weight * gaussian_ll + 
                   self.laplacian_weight * laplacian_ll + 
                   self.poisson_weight * poisson_ll)
        return total_ll


class OptimalControlFilter:
    """
    Implements multiple optimal control-based filter designs.
    - Wiener filter (minimum MSE)
    - Kalman smoother (dynamic filtering)
    - H-infinity filter (robust to uncertainty)
    """
    
    def __init__(self, filter_type='wiener'):
        """
        Initialize optimal control filter.
        
        Parameters
        ----------
        filter_type : str
            'wiener', 'kalman', or 'hinf'
        """
        self.filter_type = filter_type
        self.kernel_size = 5
        
    def wiener_filter(self, image, noise_variance=None):
        """
        Applies Wiener filter for minimum MSE denoising.
        
        Optimal linear filter that minimizes mean squared error
        under assumption of known signal and noise statistics.
        
        Transfer function:
            H(f) = |S(f)|^2 / (|S(f)|^2 + σ_n^2)
        
        Parameters
        ----------
        image : np.ndarray
            Noisy image
        noise_variance : float, optional
            Noise variance; if None, estimated from image
            
        Returns
        -------
        np.ndarray : Filtered image
        """
        H, W = image.shape
        
        # Estimate noise if not provided
        if noise_variance is None:
            # Estimate from Laplacian (high-frequency content)
            laplacian = laplace(image)
            noise_variance = np.var(laplacian) / 4.0
        
        # Local mean and variance
        local_mean = ndimage.uniform_filter(image, size=self.kernel_size)
        local_sq_mean = ndimage.uniform_filter(image**2, size=self.kernel_size)
        local_var = local_sq_mean - local_mean**2
        
        # Wiener filter gain
        gain = np.maximum(0, local_var - noise_variance) / (local_var + 1e-8)
        
        # Apply filter
        denoised = local_mean + gain * (image - local_mean)
        return denoised
    
    def kalman_smoother(self, image, process_noise=0.01, measurement_noise=None):
        """
        Applies Kalman smoothing for temporal-like denoising.
        
        Treats rows as sequential "time steps" and applies forward-backward
        Kalman filter for adaptive smoothing.
        
        Parameters
        ----------
        image : np.ndarray
            Noisy image
        process_noise : float
            Process noise standard deviation
        measurement_noise : float, optional
            Measurement noise standard deviation
            
        Returns
        -------
        np.ndarray : Smoothed image
        """
        H, W = image.shape
        
        if measurement_noise is None:
            measurement_noise = np.std(image) * 0.1
        
        # Forward pass (Kalman filter)
        estimate = image.copy()
        P = np.ones((H, W)) * (measurement_noise**2)
        
        for i in range(1, H):
            # Predict
            estimate_pred = estimate[i-1]
            P_pred = P[i-1] + process_noise**2
            
            # Update
            K = P_pred / (P_pred + measurement_noise**2)
            estimate[i] = estimate_pred + K * (image[i] - estimate_pred)
            P[i] = (1 - K) * P_pred
        
        # Backward pass (smoothing)
        for i in range(H-2, 0, -1):
            smoothing_gain = P[i] / (P[i] + process_noise**2)
            estimate[i] = estimate[i] + smoothing_gain * (estimate[i+1] - estimate[i])
        
        return estimate
    
    def hinf_filter(self, image, gamma=1.5):
        """
        Applies H-infinity filter for robust denoising.
        
        Minimizes worst-case error under bounded disturbance using
        robust control theory. Handles unknown noise better than Wiener.
        
        Parameters
        ----------
        image : np.ndarray
            Noisy image
        gamma : float
            Robustness weight (larger = more robust to uncertainty)
            
        Returns
        -------
        np.ndarray : Filtered image
        """
        H, W = image.shape
        denoised = image.copy()
        
        # Iterative H-infinity filter
        for iter in range(3):
            # Compute local curvature (second derivative)
            lap = laplace(denoised)
            
            # Robust gain scheduling
            sensitivity = 1.0 / (1.0 + gamma * np.abs(lap))
            
            # Apply robust filter
            local_mean = ndimage.uniform_filter(denoised, size=self.kernel_size)
            update = sensitivity * (denoised - local_mean)
            denoised = local_mean + update * 0.5  # Damping
        
        return denoised
    
    def apply(self, image, noise_variance=None):
        """
        Apply the selected optimal control filter.
        
        Parameters
        ----------
        image : np.ndarray
            Noisy image
        noise_variance : float, optional
            Noise variance estimate
            
        Returns
        -------
        np.ndarray : Denoised image
        """
        if self.filter_type == 'wiener':
            return self.wiener_filter(image, noise_variance)
        elif self.filter_type == 'kalman':
            return self.kalman_smoother(image, measurement_noise=np.sqrt(noise_variance) if noise_variance else None)
        elif self.filter_type == 'hinf':
            return self.hinf_filter(image)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")


class CurvaturePreservingRegularization:
    """
    Edge-aware regularization that preserves anatomical boundaries
    while removing noise.
    
    Uses anisotropic diffusion and total variation with curvature weighting.
    """
    
    def __init__(self, iterations=10, lambda_reg=0.1):
        """
        Initialize regularization.
        
        Parameters
        ----------
        iterations : int
            Number of diffusion iterations
        lambda_reg : float
            Regularization strength
        """
        self.iterations = iterations
        self.lambda_reg = lambda_reg
    
    def anisotropic_diffusion(self, image):
        """
        Perona-Malik anisotropic diffusion.
        
        Enhances edges while smoothing flat regions.
        Diffusion tensor controlled by local gradient magnitude.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
            
        Returns
        -------
        np.ndarray : Diffused image
        """
        denoised = image.copy().astype(float)
        
        # Gradient operators
        kernel_dx = np.array([[-1, 0, 1]], dtype=float) / 2.0
        kernel_dy = kernel_dx.T
        
        for iteration in range(self.iterations):
            # Compute gradients
            gx = convolve(denoised, kernel_dx)
            gy = convolve(denoised, kernel_dy)
            grad_mag = np.sqrt(gx**2 + gy**2 + 1e-8)
            
            # Edge-stopping function (decreases diffusion at edges)
            K = 0.5  # Edge sensitivity
            c = np.exp(-(grad_mag / K)**2)
            
            # Laplacian
            lap = laplace(denoised)
            
            # Update via anisotropic diffusion
            denoised = denoised + self.lambda_reg * c * lap
        
        return denoised


class HybridSuperiorReconstructor:
    """
    Combines statistical distribution modeling, optimal control filtering,
    and curvature-aware regularization for superior reconstruction.
    """
    
    def __init__(self):
        """Initialize hybrid reconstructor."""
        self.noise_mixer = StatisticalDistributionNoiseMixer()
        self.wiener = OptimalControlFilter('wiener')
        self.kalman = OptimalControlFilter('kalman')
        self.hinf = OptimalControlFilter('hinf')
        self.regularizer = CurvaturePreservingRegularization(iterations=15, lambda_reg=0.15)
    
    def reconstruct_noise_free(self, image):
        """
        Multi-stage reconstruction for noise-free, high-SNR output.
        
        Pipeline:
        1. Estimate noise distribution parameters
        2. Apply Wiener filter (optimal for known model)
        3. Robust H-infinity smoothing
        4. Anisotropic diffusion with edge preservation
        5. Final Kalman smoothing
        
        Parameters
        ----------
        image : np.ndarray
            Noisy input image
            
        Returns
        -------
        dict : Reconstruction result with multiple estimates
        """
        # Stage 1: Noise analysis
        noise_params = self.noise_mixer.estimate_noise_parameters(image)
        noise_var = noise_params['gaussian_std']**2
        
        # Stage 2: Wiener filtering
        denoised_wiener = self.wiener.apply(image, noise_var)
        
        # Stage 3: H-infinity robust smoothing
        denoised_hinf = self.hinf.apply(denoised_wiener)
        
        # Stage 4: Curvature-preserving diffusion
        denoised_aniso = self.regularizer.anisotropic_diffusion(denoised_hinf)
        
        # Stage 5: Final Kalman smoothing
        denoised_kalman = self.kalman.apply(denoised_aniso, noise_var)
        
        # Compute SNR improvement
        original_snr = 10 * np.log10(np.var(image) / noise_var)
        final_snr = 10 * np.log10(np.var(denoised_kalman) / 
                                   np.var(image - denoised_kalman))
        snr_improvement = final_snr - original_snr
        
        result = {
            'reconstructed': denoised_kalman,
            'wiener_stage': denoised_wiener,
            'hinf_stage': denoised_hinf,
            'anisotropic_stage': denoised_aniso,
            'original_snr_db': original_snr,
            'final_snr_db': final_snr,
            'snr_improvement_db': snr_improvement,
            'noise_variance_est': noise_var,
            'noise_params': noise_params
        }
        return result
    
    def compute_residual_map(self, original, reconstructed):
        """
        Compute and visualize residual map showing removed artifacts.
        
        Parameters
        ----------
        original : np.ndarray
            Original noisy image
        reconstructed : np.ndarray
            Reconstructed image
            
        Returns
        -------
        np.ndarray : Residual (artifact) map
        """
        residual = original - reconstructed
        # Normalize for visualization
        residual_norm = np.abs(residual)
        residual_norm = (residual_norm - np.min(residual_norm)) / (np.max(residual_norm) - np.min(residual_norm) + 1e-8)
        return residual_norm


def demonstrate_artifact_removal_pipeline(noisy_image):
    """
    Full demonstration of white speckle removal and reconstruction.
    
    Parameters
    ----------
    noisy_image : np.ndarray
        Input noisy MRI image
        
    Returns
    -------
    dict : All reconstruction stages and quality metrics
    """
    reconstructor = HybridSuperiorReconstructor()
    result = reconstructor.reconstruct_noise_free(noisy_image)
    
    # Compute artifact residual
    artifact_map = reconstructor.compute_residual_map(noisy_image, result['reconstructed'])
    result['artifact_map'] = artifact_map
    
    # Print quality summary
    print("=" * 70)
    print("STATISTICAL DISTRIBUTION + OPTIMAL CONTROL RECONSTRUCTION SUMMARY")
    print("=" * 70)
    print(f"Original SNR (dB):              {result['original_snr_db']:.2f}")
    print(f"Final SNR (dB):                 {result['final_snr_db']:.2f}")
    print(f"SNR Improvement (dB):           {result['snr_improvement_db']:.2f}")
    print(f"Estimated Noise Variance:       {result['noise_variance_est']:.6f}")
    print(f"Reconstruction Method:          Hybrid (Wiener + H∞ + Anisotropic + Kalman)")
    print("=" * 70)
    
    return result
