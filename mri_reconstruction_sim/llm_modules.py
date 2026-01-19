
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

class GeminiRFDesigner:
    """
    Simulates a Generative AI module for RF Coil Design.
    Constructs novel coil geometries and specifications based on prompts.
    """
    def generate_design(self, prompt, target_field_strength="3T"):
        # Simulate LLM processing
        design_specs = self._process_prompt(prompt, target_field_strength)
        schematic_b64 = self._generate_generative_schematic(design_specs)
        
        return {
            "specs": design_specs,
            "schematic": schematic_b64,
            "explanation": f"Generated design based on prompt '{prompt}' optimizing for {target_field_strength} homogeneity using {design_specs['topology']} topology."
        }

    def _process_prompt(self, prompt, field):
        prompt = prompt.lower()
        specs = {
            "topology": "Birdcage",
            "channels": 8,
            "material": "Copper",
            "tuning_freq": "128 MHz" if field == "3T" else "298 MHz" # 7T
        }
        
        if "high density" in prompt or "dense" in prompt:
            specs["topology"] = "Geodesic Array"
            specs["channels"] = 64
        elif "surface" in prompt:
            specs["topology"] = "Flexible Surface Loop"
            specs["channels"] = 4
        elif "traveling wave" in prompt:
            specs["topology"] = "Helical Antenna"
            specs["channels"] = 2
            
        if "7t" in prompt or field == "7T":
            specs["tuning_freq"] = "298 MHz"
        
        return specs

    def _generate_generative_schematic(self, specs):
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')
        ax.axis('off')
        
        # Generative Art style schematic
        t = np.linspace(0, 10 * np.pi, 500)
        
        if specs['topology'] == "Geodesic Array":
            # Draw complex lattice
            for i in range(10):
                r = 1 + 0.1 * np.sin(5 * t + i)
                x = r * np.cos(t)
                y = r * np.sin(t)
                ax.plot(x, y, alpha=0.6, color='#38bdf8', lw=1)
                
            ax.set_title("Geodesic 64-Ch Layout", color='white')
            
        elif specs['topology'] == "Helical Antenna":
            z = np.linspace(0, 5, 500)
            r = 1
            x = r * np.cos(5*z)
            y = z 
            # Project to 2D
            ax.plot(x, y, color='#38bdf8', lw=2)
            ax.set_title("Traveling Wave Helix", color='white')
            
        else:
            # Default Abstract Design
            ax.plot(np.cos(t), np.sin(t), color='#38bdf8', lw=2)
            ax.scatter(np.cos(t[::20]), np.sin(t[::20]), color='#f472b6', s=20)
            ax.set_title(f"{specs['topology']} Schematic", color='white')
            
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')


class LLMPulseDesigner:
    """
    Simulates an LLM-based Pulse Sequence Designer.
    Translates natural language descriptions into simulator parameters.
    """
    def interpret_request(self, text_prompt):
        text_prompt = text_prompt.lower()
        
        # Default
        params = {
            "sequence": "SE",
            "tr": 2000,
            "te": 100,
            "description": "Standard Spin Echo"
        }
        
        # Logic Extraction
        if "fast" in text_prompt or "speed" in text_prompt:
            params["sequence"] = "GRE"
            params["tr"] = 100
            params["te"] = 5
            params["description"] = "Fast Gradient Echo"
            
        elif "fluid" in text_prompt or "suppress csf" in text_prompt:
            params["sequence"] = "FLAIR"
            params["tr"] = 9000
            params["te"] = 140
            params["ti"] = 2500
            params["description"] = "Fluid Attenuated Inversion Recovery"
            
        elif "contrast" in text_prompt and "gray matter" in text_prompt:
            params["sequence"] = "InversionRecovery"
            params["tr"] = 4000
            params["te"] = 30
            params["ti"] = 400
            params["description"] = "T1-Weighted Inversion Recovery for GM/WM Contrast"
            
        elif "quantum" in text_prompt:
            params["sequence"] = "QuantumStatisticalCongruence"
            params["tr"] = 3000
            params["te"] = 80
            params["description"] = "Quantum Statistical Optimization"
            
        elif "cortical" in text_prompt or "brain surface" in text_prompt:
            params["sequence"] = "GRE"
            params["tr"] = 150
            params["te"] = 10
            params["description"] = "High-Res GRE for Cortical Mapping"
            
        elif "deep learning" in text_prompt or "ai recon" in text_prompt:
            params["recon_method"] = "DeepLearning"
            params["description"] = "AI-Driven Reconstruction Protocol"
        
        return params

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
        
        return {
            "mean_intensity": float(mean_sig),
            "std_intensity": float(std_sig),
            "snr_estimate": float(snr),
            "entropy": float(-np.sum(flat/np.sum(flat) * np.log(flat/np.sum(flat) + 1e-9))) # Shannon entropy
        }
        
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
