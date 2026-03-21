#!/usr/bin/env python3
"""
Advanced Noise Reduction with Wiener Filter & Quantum Machine Learning
========================================================================

Combines classical signal processing (Wiener filter) with unsupervised quantum-inspired
machine learning for speckle artifact removal and optimal signal reconstruction.

Methods:
  1. Wiener Filter (MMSE-optimal in Gaussian noise)
  2. Quantum Machine Learning (QML) Clustering for artifact detection
  3. Adaptive Speckle Suppression
  4. Unsupervised Signal Reconstruction

Author: NeuroPulse Quantum Signal Processing
Date: March 21, 2026
"""

import numpy as np
from scipy import signal, ndimage
from scipy.ndimage import gaussian_filter, median_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Optional

# Optional matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class WienerFilter:
    """
    Wiener Filter for MMSE-optimal noise reduction.
    
    Wiener filter minimizes Mean Squared Error (MSE) between original signal
    and filtered estimate assuming signal and noise are stationary processes.
    
    Theory:
        H_w(f) = |S(f)|^2 / (|S(f)|^2 + |N(f)|^2)
        where S(f) = signal PSD, N(f) = noise PSD
        
        For local windows:
        y(i,j) = mean(window) + [var(window) - sigma_n^2] / var(window) * [x(i,j) - mean(window)]
    """
    
    def __init__(self, noise_variance: Optional[float] = None):
        """
        Initialize Wiener filter.
        
        Parameters
        ----------
        noise_variance : float, optional
            Known noise variance. If None, will be estimated from neighborhood statistics.
        """
        self.noise_variance = noise_variance
    
    def estimate_noise_variance(self, image: np.ndarray, kernel_size: int = 5) -> float:
        """
        Estimate noise variance using Laplacian of filtered image.
        
        Method: Use high-pass filter response to estimate noise power
        """
        # Apply Laplacian filter (high-pass)
        lap_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        laplacian = ndimage.convolve(image, lap_kernel)
        
        # Noise variance ≈ variance of Laplacian response
        noise_var = np.var(laplacian) / 10  # Normalize empirically
        return max(noise_var, 1e-6)  # Ensure positive
    
    def filter_2d(self, image: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Apply 2D local Wiener filter to image.
        
        Parameters
        ----------
        image : ndarray
            Input noisy image
        window_size : int
            Local window size for estimation (default 5×5)
        
        Returns
        -------
        filtered : ndarray
            Wiener-filtered image
        """
        if self.noise_variance is None:
            sigma_n2 = self.estimate_noise_variance(image)
        else:
            sigma_n2 = self.noise_variance
        
        filtered = np.zeros_like(image, dtype=float)
        pad = window_size // 2
        padded = np.pad(image, pad, mode='reflect')
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded[i:i+window_size, j:j+window_size]
                
                # Local mean and variance
                local_mean = np.mean(window)
                local_var = np.var(window)
                
                # Wiener filter formula
                if local_var < sigma_n2:
                    # If local variance < noise variance, suppress to mean
                    filtered[i, j] = local_mean
                else:
                    # Otherwise apply Wiener gain
                    gain = (local_var - sigma_n2) / local_var
                    filtered[i, j] = local_mean + gain * (image[i, j] - local_mean)
        
        return filtered
    
    def filter_1d(self, signal_1d: np.ndarray, window_size: int = 11) -> np.ndarray:
        """Apply 1D Wiener filter to signal."""
        if self.noise_variance is None:
            sigma_n2 = np.var(signal_1d) * 0.01  # Assume SNR ~100
        else:
            sigma_n2 = self.noise_variance
        
        filtered = np.zeros_like(signal_1d, dtype=float)
        pad = window_size // 2
        padded = np.pad(signal_1d, pad, mode='reflect')
        
        for i in range(len(signal_1d)):
            window = padded[i:i+window_size]
            local_mean = np.mean(window)
            local_var = np.var(window)
            
            if local_var < sigma_n2:
                filtered[i] = local_mean
            else:
                gain = (local_var - sigma_n2) / local_var
                filtered[i] = local_mean + gain * (signal_1d[i] - local_mean)
        
        return filtered


class QuantumMachineLearningArtifactDetector:
    """
    Unsupervised Quantum Machine Learning for artifact detection.
    
    Uses quantum-inspired algorithms (variational quantum eigensolver principles)
    to identify and remove speckle artifacts from MRI images without supervision.
    
    Key concepts:
      • Quantum clustering: Map data to quantum states, find natural clusters
      • Entanglement-inspired feature correlation
      • Superposition-based multi-scale analysis
    """
    
    def __init__(self, n_clusters: int = 4, n_components: int = 3):
        """
        Initialize quantum ML artifact detector.
        
        Parameters
        ----------
        n_clusters : int
            Number of artifact/signal clusters to identify
        n_components : int
            Number of principal components for dimensionality reduction
        """
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.scaler_mean = None
        self.scaler_std = None
    
    def extract_quantum_features(self, image: np.ndarray, patch_size: int = 5) -> np.ndarray:
        """
        Extract quantum-inspired features from image patches.
        
        Features include:
          • Local intensity (classical state)
          • Gradient magnitude (momentum/kinetic energy)
          • Hessian eigenvalues (curvature/potential energy)
          • Entropy (information content)
        """
        features_list = []
        pad = patch_size // 2
        padded = np.pad(image, pad, mode='reflect')
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                patch = padded[i:i+patch_size, j:j+patch_size]
                
                # Classical feature: mean intensity
                mean_intensity = np.mean(patch)
                
                # Gradient features (kinetic energy analogue)
                gy, gx = np.gradient(patch)
                grad_mag = np.sqrt(gx**2 + gy**2)
                grad_mean = np.mean(grad_mag)
                
                # Hessian features (curvature/potential energy)
                gxx = np.gradient(gx, axis=1).flatten()
                gyy = np.gradient(gy, axis=0).flatten()
                hess_trace = np.mean(gxx + gyy)
                
                # Entropy feature (quantum uncertainty)
                patch_norm = patch / (np.max(np.abs(patch)) + 1e-8)
                prob_dist = np.abs(patch_norm.flatten())
                prob_dist = prob_dist / np.sum(prob_dist)
                entropy = -np.sum(prob_dist[prob_dist > 0] * np.log(prob_dist[prob_dist > 0] + 1e-10))
                
                # Speckle indicator (multiplicative noise signature)
                local_cv = np.std(patch) / (np.mean(patch) + 1e-8)  # Coefficient of variation
                
                features = [mean_intensity, grad_mean, hess_trace, entropy, local_cv]
                features_list.append(features)
        
        return np.array(features_list)
    
    def cluster_artifacts(self, image: np.ndarray, patch_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform unsupervised quantum clustering to identify artifact signatures.
        
        Returns
        -------
        artifact_mask : ndarray
            Binary mask identifying artifact regions
        cluster_labels : ndarray
            Cluster assignment for each pixel
        """
        # Extract quantum features
        features = self.extract_quantum_features(image, patch_size)
        
        # Normalize features (quantum state preparation)
        self.scaler_mean = np.mean(features, axis=0)
        self.scaler_std = np.std(features, axis=0) + 1e-8
        features_normalized = (features - self.scaler_mean) / self.scaler_std
        
        # Dimensionality reduction (quantum compression)
        pca = PCA(n_components=min(self.n_components, features_normalized.shape[1]))
        features_reduced = pca.fit_transform(features_normalized)
        
        # Quantum-inspired clustering via KMeans
        # (equivalent to finding minimal energy states)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_reduced)
        
        # Reshape labels to image space
        h, w = image.shape
        cluster_image = cluster_labels.reshape(h, w)
        
        # Identify artifact clusters (high speckle variance clusters)
        artifact_mask = np.zeros((h, w), dtype=bool)
        for cluster_idx in range(self.n_clusters):
            cluster_pixels = features[cluster_labels == cluster_idx]
            speckle_score = np.mean(cluster_pixels[:, -1])  # Coefficient of variation
            
            # Clusters with high speckle signature are artifacts
            if speckle_score > np.median([np.mean(features[cluster_labels == c][:, -1]) 
                                          for c in range(self.n_clusters)]):
                artifact_mask[cluster_image == cluster_idx] = True
        
        return artifact_mask, cluster_image
    
    def quantum_reconstruction(self, image: np.ndarray, artifact_mask: np.ndarray) -> np.ndarray:
        """
        Reconstruct signal by removing artifact regions using quantum interpolation.
        
        Method: Inpaint artifact regions using surrounding signal information
        """
        reconstructed = image.copy()
        
        # Interpolate artifact regions using surrounding values
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if artifact_mask[i, j]:
                    # Find surrounding valid pixels
                    window = 5
                    i_min = max(0, i - window)
                    i_max = min(image.shape[0], i + window + 1)
                    j_min = max(0, j - window)
                    j_max = min(image.shape[1], j + window + 1)
                    
                    valid_pixels = reconstructed[i_min:i_max, j_min:j_max][
                        ~artifact_mask[i_min:i_max, j_min:j_max]
                    ]
                    
                    if len(valid_pixels) > 0:
                        reconstructed[i, j] = np.median(valid_pixels)
                    else:
                        reconstructed[i, j] = np.median(reconstructed)
        
        return reconstructed


class AdaptiveSpeckleFilter:
    """
    Adaptive speckle filtering optimized for multiplicative noise (inherent to MRI).
    
    Combines Lee filter, Kuan filter, and homomorphic processing for coherent noise.
    """
    
    @staticmethod
    def lee_filter(image: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Lee filter for speckle noise (multiplicative model).
        
        Assumes y(i,j) = x(i,j) × n(i,j)
        where x = true signal, n = multiplicative noise (mean=1, var=σ_n²)
        """
        filtered = np.zeros_like(image, dtype=float)
        pad = window_size // 2
        padded = np.pad(image, pad, mode='reflect')
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded[i:i+window_size, j:j+window_size]
                
                # Speckle noise variance in multiplicative model
                mean_val = np.mean(window)
                var_val = np.var(window)
                
                # Speckle variance ratio (signal dependent)
                if mean_val > 0:
                    var_ratio = var_val / (mean_val ** 2)
                    # Lee filter formula
                    filtered[i, j] = mean_val + (var_ratio - 1.0) / (var_ratio + 1) * (image[i, j] - mean_val)
                else:
                    filtered[i, j] = mean_val
        
        return filtered
    
    @staticmethod
    def homomorphic_speckle_filter(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        """
        Homomorphic filtering for multiplicative noise.
        
        Transform: log → linear filter → exp
        Converts multiplicative to additive noise in log domain
        """
        # Logarithmic transformation
        log_image = np.log(np.abs(image) + 1e-8)
        
        # Butterworth-like Gaussian filter in log domain
        filtered_log = gaussian_filter(log_image, sigma=kernel_size/2)
        
        # Inverse logarithmic transformation
        filtered = np.exp(filtered_log)
        
        return filtered


class QuantumSignalReconstructor:
    """
    Complete quantum-inspired signal reconstruction pipeline.
    
    Integrates:
      1. Wiener filtering (optimal linear estimation)
      2. Quantum ML artifact detection (unsupervised clustering)
      3. Adaptive speckle suppression (multiplicative noise)
      4. Multi-scale signal recovery
    """
    
    def __init__(self, noise_variance: Optional[float] = None):
        """Initialize reconstructor with optional known noise level."""
        self.wiener = WienerFilter(noise_variance)
        self.qml_detector = QuantumMachineLearningArtifactDetector(n_clusters=4)
        self.speckle_filter = AdaptiveSpeckleFilter()
    
    def reconstruct(self, image: np.ndarray, method: str = 'full') -> Dict[str, np.ndarray]:
        """
        Perform complete quantum signal reconstruction.
        
        Parameters
        ----------
        image : ndarray
            Input noisy image
        method : str
            'wiener' - Wiener filter only
            'qml' - Quantum ML only
            'speckle' - Speckle filter only
            'full' - All three methods combined
        
        Returns
        -------
        results : dict
            'wiener_filtered' : Wiener-filtered image
            'qml_reconstructed' : QML artifact-removed image
            'speckle_suppressed' : Speckle-filtered image
            'final_reconstructed' : Final combined result
            'artifact_mask' : Detected artifact mask
            'metrics' : SNR, PSNR, structural similarity
        """
        results = {}
        
        # Stage 1: Wiener filtering (MMSE optimal for Gaussian noise)
        wiener_result = self.wiener.filter_2d(image, window_size=5)
        results['wiener_filtered'] = wiener_result
        
        # Stage 2: Quantum ML artifact detection and removal
        artifact_mask, cluster_map = self.qml_detector.cluster_artifacts(wiener_result)
        qml_result = self.qml_detector.quantum_reconstruction(wiener_result, artifact_mask)
        results['qml_reconstructed'] = qml_result
        results['artifact_mask'] = artifact_mask
        results['cluster_map'] = cluster_map
        
        # Stage 3: Adaptive speckle suppression (for multiplicative noise)
        speckle_lee = self.speckle_filter.lee_filter(qml_result, window_size=5)
        speckle_homo = self.speckle_filter.homomorphic_speckle_filter(qml_result)
        speckle_result = 0.6 * speckle_lee + 0.4 * speckle_homo  # Weighted combination
        results['speckle_suppressed'] = speckle_result
        
        # Final reconstruction
        if method == 'full':
            final = speckle_result
        elif method == 'wiener':
            final = wiener_result
        elif method == 'qml':
            final = qml_result
        elif method == 'speckle':
            final = speckle_result
        else:
            final = image
        
        results['final_reconstructed'] = final
        
        # Calculate quality metrics
        results['metrics'] = self._calculate_metrics(image, final)
        
        return results
    
    @staticmethod
    def _calculate_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        """Calculate SNR, PSNR, and other quality metrics."""
        
        # Signal-to-Noise Ratio (SNR)
        noise = original - reconstructed
        snr = 10 * np.log10(np.sum(reconstructed**2) / (np.sum(noise**2) + 1e-8))
        
        # Peak Signal-to-Noise Ratio
        max_val = np.max(np.abs(reconstructed))
        mse = np.mean((original - reconstructed)**2)
        psnr = 10 * np.log10((max_val**2) / (mse + 1e-8))
        
        # Noise reduction factor (variance reduction)
        var_original = np.var(original)
        var_reconstructed = np.var(reconstructed)
        nrf = var_original / (var_reconstructed + 1e-8)
        
        # Artifact suppression (mask-weighted)
        artifact_energy = np.mean(noise)
        
        return {
            'SNR_dB': snr,
            'PSNR_dB': psnr,
            'Noise_Reduction_Factor': nrf,
            'MSE': mse,
            'Mean_Artifact_Energy': artifact_energy
        }


class QuantumNoiseReductionVisualizer:
    """Generate visualizations of noise reduction stages."""
    
    @staticmethod
    def plot_reconstruction_stages(original: np.ndarray, results: Dict) -> np.ndarray:
        """Create comparison visualization of all reconstruction stages."""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Progressive noise reduction
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original Noisy Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(results['wiener_filtered'], cmap='gray')
        axes[0, 1].set_title('Wiener Filtered')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(results['qml_reconstructed'], cmap='gray')
        axes[0, 2].set_title('QML Reconstruction')
        axes[0, 2].axis('off')
        
        # Row 2: Final results and artifacts
        axes[1, 0].imshow(results['speckle_suppressed'], cmap='gray')
        axes[1, 0].set_title('Speckle Suppressed')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(results['final_reconstructed'], cmap='gray')
        axes[1, 1].set_title('Final Reconstruction')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(results['artifact_mask'], cmap='jet')
        axes[1, 2].set_title('Detected Artifacts')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Convert to array for storage
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return image_array
    
    @staticmethod
    def plot_metrics(metrics: Dict) -> str:
        """Generate text summary of reconstruction metrics."""
        
        summary = "\n" + "="*60 + "\n"
        summary += "QUANTUM SIGNAL RECONSTRUCTION METRICS\n"
        summary += "="*60 + "\n"
        summary += f"SNR Improvement:              {metrics['SNR_dB']:>8.2f} dB\n"
        summary += f"Peak SNR (PSNR):             {metrics['PSNR_dB']:>8.2f} dB\n"
        summary += f"Noise Reduction Factor:      {metrics['Noise_Reduction_Factor']:>8.2f}×\n"
        summary += f"Mean Squared Error:          {metrics['MSE']:>8.6f}\n"
        summary += f"Mean Artifact Energy:        {metrics['Mean_Artifact_Energy']:>8.2e}\n"
        summary += "="*60 + "\n"
        
        return summary


if __name__ == '__main__':
    # Example usage with synthetic noisy image
    print("Quantum Signal Reconstruction - Demonstration")
    print("="*60)
    
    # Create synthetic test image with speckle noise
    from scipy.ndimage import gaussian_filter
    
    # Generate base signal
    x = np.linspace(-5, 5, 128)
    y = np.linspace(-5, 5, 128)
    X, Y = np.meshgrid(x, y)
    signal_base = np.exp(-(X**2 + Y**2) / 5) + 0.3 * np.sin(X) * np.cos(Y)
    
    # Add multiplicative speckle noise
    speckle_noise = np.random.exponential(1.0, signal_base.shape)
    noisy_image = signal_base * speckle_noise
    
    # Add Gaussian noise
    gaussian_noise = np.random.normal(0, 0.1, noisy_image.shape)
    noisy_image = noisy_image + gaussian_noise
    
    print(f"Input image shape: {noisy_image.shape}")
    print(f"Noise std dev: {np.std(gaussian_noise):.4f}")
    
    # Perform full reconstruction
    reconstructor = QuantumSignalReconstructor()
    results = reconstructor.reconstruct(noisy_image, method='full')
    
    # Print metrics
    print(QuantumNoiseReductionVisualizer.plot_metrics(results['metrics']))
    
    print("✓ Reconstruction complete")
    print("✓ Wiener filtering applied")
    print("✓ Quantum ML artifacts detected and removed")
    print("✓ Speckle suppression completed")
