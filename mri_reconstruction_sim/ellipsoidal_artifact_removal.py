"""
Ellipsoidal Bounding Box for MRI White Speckle Noise Artifact Removal
====================================================================

Implements adaptive ellipsoidal masking with intelligent white speckle
noise detection and removal for MRI reconstruction.

Key Features:
  • Adaptive ellipsoid fitting based on image intensity distribution
  • Multi-scale white speckle detection via Laplacian of Gaussian
  • Morphological analysis for artifact classification
  • Anatomical structure preservation
  • Support for multi-modal reconstruction (brain, cardiac, knee)

Author: NeuroPulse Artifact Removal Engine
Date: March 12, 2026
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import minimize
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


class EllipsoidalArtifactRemover:
    """
    Advanced ellipsoidal bounding box detector and white speckle artifact remover.
    
    Uses adaptive fitting and multi-scale analysis to identify and suppress
    white speckle noise while preserving anatomical structures.
    """
    
    def __init__(self, phantom_type='brain', resolution=128):
        """
        Initialize the artifact remover.
        
        Parameters
        ----------
        phantom_type : str
            Type of anatomical structure: 'brain', 'cardiac', 'knee'
        resolution : int
            Image resolution (assumed square)
        """
        self.phantom_type = phantom_type
        self.resolution = resolution
        self.ellipse_params = None
        self.speckle_mask = None
        
    def adaptive_ellipsoid_fit(self, image):
        """
        Adaptively fits an ellipsoid to the image based on intensity distribution.
        
        Uses the center of mass and principal component analysis to find
        optimal ellipsoid parameters.
        
        Parameters
        ----------
        image : np.ndarray
            Input MRI image
            
        Returns
        -------
        dict : Parameters including center, semi-axes a, b, angle
        """
        N = self.resolution
        
        # Normalize image to [0, 1]
        img_norm = np.nan_to_num(image)
        img_norm = (img_norm - np.min(img_norm)) / (np.max(img_norm) - np.min(img_norm) + 1e-9)
        
        # Find center of mass (weighted by intensity)
        y, x = np.ogrid[:N, :N]
        total_intensity = np.sum(img_norm)
        
        if total_intensity > 0:
            cy = np.sum(y * img_norm) / total_intensity
            cx = np.sum(x * img_norm) / total_intensity
        else:
            cy, cx = N // 2, N // 2
        
        # Compute second moments for principal axes
        y_centered = y - cy
        x_centered = x - cx
        
        Ixx = np.sum(img_norm * x_centered**2)
        Iyy = np.sum(img_norm * y_centered**2)
        Ixy = np.sum(img_norm * x_centered * y_centered)
        
        # Eigenvalues (second moments)
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy**2
        
        # Eigenvalues via quadratic formula - ensure discriminant is non-negative
        disc = max(0, trace**2 - 4*det)
        lambda1 = (trace + np.sqrt(disc)) / 2.0
        lambda2 = (trace - np.sqrt(disc)) / 2.0
        
        # Safeguard: lambda values should be positive
        lambda1 = max(lambda1, 1e-9)
        lambda2 = max(lambda2, 1e-9)
        
        # Semi-axes from second moments
        # For ellipsoid: a² ≈ 5*lambda_max, b² ≈ 5*lambda_min (empirical)
        a_squared = 5 * lambda1 / (total_intensity + 1e-9)
        b_squared = 5 * lambda2 / (total_intensity + 1e-9)
        
        a = np.sqrt(max(a_squared, 1e-6))
        b = np.sqrt(max(b_squared, 1e-6))
        
        # Compute rotation angle from eigenvector
        if abs(Ixy) > 1e-9 or abs(Ixx - lambda1) > 1e-9:
            angle = 0.5 * np.arctan2(2*Ixy, Ixx - Iyy)
        else:
            angle = 0.0
        
        # Scale axes based on phantom type (heuristics)
        if self.phantom_type == 'brain':
            # Brain: slightly elongated, covers most of image
            # Neurovascular: head and neck vasculature typically occupies 40-45% of FOV
            a_scale = 0.42
            b_scale = 0.45
        elif self.phantom_type == 'cardiac':
            # Cardiac: more circular, smaller footprint (heart ~35-38% of FOV)
            a_scale = 0.35
            b_scale = 0.38
        else:  # knee
            # Knee: elongated along one axis (cartilage ~38-48% of FOV)
            a_scale = 0.38
            b_scale = 0.48
        
        # Blend fitted parameters with type-specific defaults
        # Use proportions of image size for consistent scaling
        a_final = 0.5 * a + 0.5 * (a_scale * N)
        b_final = 0.5 * b + 0.5 * (b_scale * N)
        
        # Ensure minimum size
        a_final = max(a_final, 5.0)
        b_final = max(b_final, 5.0)
        
        self.ellipse_params = {
            'center_x': float(cx),
            'center_y': float(cy),
            'a': float(a_final),
            'b': float(b_final),
            'angle': float(angle)
        }
        
        return self.ellipse_params
    
    def create_ellipsoidal_mask(self, params=None):
        """
        Creates binary ellipsoidal mask.
        
        Parameters
        ----------
        params : dict, optional
            Ellipsoid parameters. If None, uses self.ellipse_params
            
        Returns
        -------
        np.ndarray : Binary mask (1 inside, 0 outside)
        """
        if params is None:
            params = self.ellipse_params or self.adaptive_ellipsoid_fit(np.zeros((self.resolution, self.resolution)))
        
        N = self.resolution
        y, x = np.ogrid[:N, :N]
        
        cx = params['center_x']
        cy = params['center_y']
        a = params['a']
        b = params['b']
        angle = params['angle']
        
        # Rotate coordinates
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        x_rot = (x - cx) * cos_a + (y - cy) * sin_a
        y_rot = -(x - cx) * sin_a + (y - cy) * cos_a
        
        # Ellipse equation: (x²/a²) + (y²/b²) ≤ 1
        mask = (x_rot**2 / (a**2 + 1e-9) + y_rot**2 / (b**2 + 1e-9)) <= 1.0
        
        return mask.astype(np.float32)
    
    def detect_white_speckles(self, image, scale_range=None):
        """
        Detects white speckle noise using multi-scale Laplacian of Gaussian.
        
        White speckles are small, bright, isolated features characteristic
        of impulse noise and salt-and-pepper artifacts. This method is neurovascular-aware
        and uses multiple scale analysis to avoid removing small blood vessel details.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        scale_range : tuple, optional
            (min_scale, max_scale) in pixels. Default: (1, 5)
            
        Returns
        -------
        np.ndarray : Speckle confidence map (0-1)
        """
        if scale_range is None:
            # For neurovascular images, use wider scale range to detect noise at multiple sizes
            scale_range = (1, 6) if self.phantom_type == 'brain' else (1, 5)
        
        img = np.nan_to_num(image).astype(np.float64)
        
        # Normalize to [0, 1]
        if np.max(img) > 0:
            img = img / np.max(img)
        
        speckle_scores = np.zeros_like(img)
        n_scales = 5 if self.phantom_type == 'brain' else 4
        
        for scale_idx in range(n_scales):
            # Multi-scale analysis
            sigma = scale_range[0] + (scale_range[1] - scale_range[0]) * scale_idx / (n_scales - 1)
            
            # Laplacian of Gaussian
            log_kernel = ndi.gaussian_laplace(img, sigma=sigma)
            
            # Bright speckles have high positive LoG response
            speckle_response = np.maximum(log_kernel, 0)
            
            # Normalize and accumulate
            if np.max(speckle_response) > 0:
                speckle_response = speckle_response / np.max(speckle_response)
            
            speckle_scores += speckle_response / n_scales
        
        # Further refine: combine with local intensity contrast
        local_max = ndi.maximum_filter(img, size=5)
        local_min = ndi.minimum_filter(img, size=5)
        local_contrast = np.maximum(local_max - local_min, 1e-9)
        
        # Normalized local intensity
        intensity_map = (img - local_min) / (local_contrast + 1e-9)
        
        # High intensity speckles in low-contrast regions are artifacts
        artifact_likelihood = speckle_scores * intensity_map
        
        # Neurovascular-aware thresholding: higher threshold for brain to preserve vessel details
        if self.phantom_type == 'brain':
            percentile_threshold = 88  # More conservative for preserving neurovascular details
        elif self.phantom_type == 'cardiac':
            percentile_threshold = 85
        else:  # knee
            percentile_threshold = 85
        
        threshold = np.percentile(artifact_likelihood, percentile_threshold)
        speckle_mask = artifact_likelihood > threshold
        
        self.speckle_mask = speckle_mask
        
        return artifact_likelihood
    
    def remove_artifacts(self, image, use_ellipsoid=True, speckle_threshold=0.75):
        """
        Main artifact removal pipeline.
        
        Parameters
        ----------
        image : np.ndarray
            Input MRI image
        use_ellipsoid : bool
            Whether to apply ellipsoidal mask
        speckle_threshold : float
            Threshold for speckle removal confidence
            
        Returns
        -------
        np.ndarray : Cleaned image
        dict : Artifact removal statistics
        """
        img = np.nan_to_num(image).astype(np.float64)
        
        stats = {
            'original_max': float(np.max(img)),
            'original_mean': float(np.mean(img)),
            'artifacts_removed_count': 0,
            'ellipsoid_applied': use_ellipsoid
        }
        
        # Step 1: Adaptive ellipsoid fitting
        self.adaptive_ellipsoid_fit(img)
        ellipsoid_mask = self.create_ellipsoidal_mask()
        
        # Step 2: Detect white speckles
        speckle_likelihood = self.detect_white_speckles(img)
        speckle_artifacts = speckle_likelihood > speckle_threshold
        
        # Step 3: Remove identified speckles
        cleaned = img.copy()
        
        if np.any(speckle_artifacts):
            # Label connected components
            labeled, num_features = ndi.label(speckle_artifacts)
            
            stats['artifacts_removed_count'] = num_features
            
            for label_id in range(1, num_features + 1):
                component = (labeled == label_id)
                
                # Get local neighborhood
                dilated = ndi.binary_dilation(component, iterations=2)
                neighbor_mask = dilated & ~component
                
                if np.any(neighbor_mask):
                    # Replace with median of neighbors
                    neighbor_vals = cleaned[neighbor_mask]
                    replacement_val = np.median(neighbor_vals)
                else:
                    replacement_val = np.median(cleaned)
                
                cleaned[component] = replacement_val
        
        # Step 4: Apply ellipsoidal mask if requested
        if use_ellipsoid:
            cleaned = cleaned * ellipsoid_mask
        
        # Step 5: Final smoothing to blend repairs
        cleaned = ndi.gaussian_filter(cleaned, sigma=0.8)
        
        # Step 6: Restore valid intensity range
        if stats['original_max'] > 0:
            cleaned = cleaned * (stats['original_max'] / np.max(cleaned)) if np.max(cleaned) > 0 else cleaned
        
        stats['final_max'] = float(np.max(cleaned))
        stats['final_mean'] = float(np.mean(cleaned))
        stats['intensity_preserved'] = float(1.0 - abs(stats['original_mean'] - stats['final_mean']) / (stats['original_mean'] + 1e-9))
        
        return cleaned, stats


def reconstruct_with_ellipsoidal_removal(simulator, kspace_data, phantom_type='brain', **recon_kwargs):
    """
    Convenient wrapper to perform reconstruction with ellipsoidal artifact removal.
    
    Parameters
    ----------
    simulator : MRIReconstructionSimulator
        The simulator instance
    kspace_data : list[np.ndarray]
        K-space data from each coil
    phantom_type : str
        Type of phantom: 'brain', 'cardiac', 'knee'
    **recon_kwargs : dict
        Additional arguments for reconstruct_image
        
    Returns
    -------
    cleaned_image : np.ndarray
        Artifact-removed reconstruction
    stats : dict
        Removal statistics
    """
    # Standard reconstruction
    recon_img, coil_imgs = simulator.reconstruct_image(kspace_data, **recon_kwargs)
    
    # Apply ellipsoidal artifact removal
    remover = EllipsoidalArtifactRemover(phantom_type=phantom_type, resolution=simulator.resolution)
    cleaned_img, stats = remover.remove_artifacts(recon_img, use_ellipsoid=True)
    
    return cleaned_img, stats, coil_imgs
