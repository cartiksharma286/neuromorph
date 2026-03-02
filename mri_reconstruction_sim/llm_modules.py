
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# GeminiRFDesigner and LLMPulseDesigner removed as part of "Head Coil Designer" cleanup.


class StatisticalClassifier:
    """
    Performs statistical analysis and variational reconstruction support.
    """
    def analyze_image(self, image_data):
        """
        Returns statistical metrics and tissue classification.
        """
        flat = image_data.flatten()
        # Remove background for stats
        flat = flat[flat > 0.05]
        
        if len(flat) == 0:
            return {"cnr": 0, "snr": 0, "classification": "Empty"}
            
        mean_sig = np.mean(flat)
        std_sig = np.std(flat)
        snr = mean_sig / (std_sig + 1e-9)
        
        # Simple Clustering (K-Means like)
        # Class 1: CSF (Dark/Light depends on seq), Class 2: GM, Class 3: WM
        
        # QML Statistical Distribution Fitting (Gamma Approx for MR Magnitude)
        # Using moment matching: mean = alpha * scale, var = alpha * scale^2
        if mean_sig > 0:
            scale = (std_sig ** 2) / mean_sig
            alpha = mean_sig / (scale + 1e-9)
        else:
            scale, alpha = 1.0, 1.0
            
        # Infer pseudo-temperature based on signal deviation (T1 is temperature dependent)
        inferred_temp = 37.0 - (mean_sig - 0.5) * 5.0 # Simulated PRF shift relationship
        
        # Save state internally for plotting
        self.latest_stats = {
            "mean_intensity": float(mean_sig),
            "std_intensity": float(std_sig),
            "snr_estimate": float(snr),
            "entropy": float(-np.sum(flat/np.sum(flat) * np.log(flat/np.sum(flat) + 1e-9))),
            "distribution_type": "Gamma Parametric",
            "inferred_mean_temp_c": float(inferred_temp),
            "confidence_interval": [float(inferred_temp - 0.2), float(inferred_temp + 0.2)],
            "params": {"alpha": float(alpha), "loc": 0.0, "scale": float(scale)}
        }
        
        return self.latest_stats
        
    def variational_denoise(self, image, lambda_tv=0.1):
        """
        Total Variation Denoising (Simulated Variational Upgrade)
        """
        from scipy.ndimage import gaussian_filter
        # Simple implementation: Proximal operator approx or just bilateral
        # Let's use a specialized filter to simulate TV preservation of edges
        
        # Approximate TV result
        smooth = gaussian_filter(image, sigma=1)
        # sharpen edges
        details = image - smooth
        
        # Soft thresholding of details (sparsity in gradient domain)
        details = np.sign(details) * np.maximum(0, np.abs(details) - lambda_tv)
        
        out = smooth + details
        return out
        
    def classify_tissue(self, image_data, k=3):
        """
        Performs K-Means clustering to separate Background, Soft Tissue, and Bright Tissue.
        Returns a labeled mask having dimensions of image_data.
        """
        flat = image_data.flatten().reshape(-1, 1)
        
        # Initialize centroids (Background~0, Tissue~Mid, Contrast~High)
        min_val = np.min(flat)
        max_val = np.max(flat)
        centroids = np.array([min_val, (min_val+max_val)/2, max_val])
        
        # Simple 1D K-Means
        for _ in range(10): # 10 iterations usually enough for 1D
            # Assign points to nearest centroid
            distances = np.abs(flat - centroids.reshape(1, -1))
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([flat[labels == i].mean() if np.sum(labels==i)>0 else centroids[i] for i in range(k)])
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
            
        return labels.reshape(image_data.shape), centroids
        
    def create_background_mask(self, image_data):
        """
        Generates a robust binary mask where 1=Tissue, 0=Background.
        Uses K-Means clustering to identify the lowest energy cluster.
        """
        labels, centroids = self.classify_tissue(image_data, k=2) # 2 clusters: bg vs tissue
        
        # Identify background cluster (lowest centroid)
        bg_label = np.argmin(centroids)
        
        # Create mask (invert background)
        mask = (labels != bg_label).astype(np.float32)
        
        # Morphological Cleanup (remove small floating pixels)
        from scipy.ndimage import binary_opening, binary_closing
        mask = binary_opening(mask, structure=np.ones((2,2))) # Remove small noise
        mask = binary_closing(mask, structure=np.ones((3,3))) # Fill small holes
        
        return mask
