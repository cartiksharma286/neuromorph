import numpy as np
import scipy.ndimage
import scipy.fftpack
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import base64

class MRIReconstructionSimulator:
    def __init__(self, resolution=128):
        self.resolution = resolution
        self.dims = (resolution, resolution)
        self.t1_map = None
        self.t2_map = None
        self.pd_map = None # Proton Density
        self.coils = []
        
    def generate_brain_phantom(self):
        """Generates synthetic T1, T2, and PD maps for a brain slice."""
        N = self.resolution
        self.t1_map = np.zeros(self.dims)
        self.t2_map = np.zeros(self.dims)
        self.pd_map = np.zeros(self.dims)
        
        y, x = np.ogrid[:N, :N]
        center = (N//2, N//2)
        
        # 1. Background (Air)
        # 0 signal
        
        # 2. Skin/Scalp (Short T1, Short T2)
        mask_head = (x - center[1])**2 + (y - center[0])**2 <= (N//2 - 5)**2
        self.t1_map[mask_head] = 500  # ms
        self.t2_map[mask_head] = 50   # ms
        self.pd_map[mask_head] = 0.8
        
        # 3. Skull (Very low signal)
        mask_skull = (x - center[1])**2 + (y - center[0])**2 <= (N//2 - 10)**2
        mask_skull_inner = (x - center[1])**2 + (y - center[0])**2 <= (N//2 - 15)**2
        skull_region = mask_skull & ~mask_skull_inner
        self.t1_map[skull_region] = 100
        self.t2_map[skull_region] = 10
        self.pd_map[skull_region] = 0.1
        
        # 4. CSF (Long T1, Long T2) - Ventricles
        mask_brain = mask_skull_inner
        self.t1_map[mask_brain] = 900  # Gray matterish initial
        self.t2_map[mask_brain] = 100
        self.pd_map[mask_brain] = 0.9
        
        # Add Gray Matter / White Matter contrast
        # Simple Noise texture to simulate structure
        structure = np.random.rand(N, N) * 0.2
        structure = scipy.ndimage.gaussian_filter(structure, sigma=2)
        
        # Define Ventricles (CSF)
        mask_ventricle = (np.abs(x - center[1]) < N//8) & (np.abs(y - center[0]) < N//6)
        self.t1_map[mask_ventricle] = 4000
        self.t2_map[mask_ventricle] = 2000
        self.pd_map[mask_ventricle] = 1.0
        
        # White Matter (Short T1, Short T2)
        # Ellipse
        mask_wm = ((x - center[1])**2 / (N//4)**2 + (y - center[0])**2 / (N//3)**2 <= 1) & ~mask_ventricle
        self.t1_map[mask_wm] = 700
        self.t2_map[mask_wm] = 80
        self.pd_map[mask_wm] = 0.75 + structure[mask_wm]
        
        # Gray Matter (Intermediate)
        mask_gm = mask_brain & ~mask_wm & ~mask_ventricle
        self.t1_map[mask_gm] = 1200
        self.t2_map[mask_gm] = 110
        self.pd_map[mask_gm] = 0.85 + structure[mask_gm]
        
    def generate_coil_sensitivities(self, num_coils=8, coil_type='standard'):
        """Generates sensitivity maps for RF coils."""
        self.coils = []
        N = self.resolution
        y, x = np.ogrid[:N, :N]
        center = (N//2, N//2)
        
        if coil_type == 'standard':
            # Birdcage-like mode (Uniform-ish)
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            sensitivity = np.exp(-r**2 / (2 * (N)**2)) 
            self.coils.append(sensitivity)
            
        elif coil_type == 'custom_phased_array':
            # Custom High-Res Phased Array
            for i in range(num_coils):
                angle = 2 * np.pi * i / num_coils
                cx = center[1] + (N//2) * np.cos(angle)
                cy = center[0] + (N//2) * np.sin(angle)
                dist_sq = (x - cx)**2 + (y - cy)**2
                sensitivity = np.exp(-dist_sq / (2 * (N//4)**2))
                phase = np.exp(1j * (x * np.cos(angle) + y * np.sin(angle)) * 0.05)
                self.coils.append(sensitivity * phase)
                
        elif coil_type == 'gemini_14t':
            # "Gemini Head Coil": Ultra-high field, very homogeneous
            # 14T implies strong B1 homogeneity correction
            sensitivity = np.ones(self.dims) * 0.95
            # Slight "standing wave" artifact simulation for 14T
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            sensitivity += 0.05 * np.cos(r * 0.1)
            self.coils.append(sensitivity)
            
        elif coil_type == 'n25_array':
            # "N25 Dense Array": 25 small elements, high surface SNR
            sim_coils = 25
            for i in range(sim_coils):
                angle = 2 * np.pi * i / sim_coils
                # Coils closer to head
                cx = center[1] + (N//2.2) * np.cos(angle)
                cy = center[0] + (N//2.2) * np.sin(angle)
                dist_sq = (x - cx)**2 + (y - cy)**2
                # Very sharp falloff
                sensitivity = 2.0 * np.exp(-dist_sq / (2 * (N//8)**2))
                phase = np.exp(1j * (x * np.cos(angle) + y * np.sin(angle)) * 0.1)
                self.coils.append(sensitivity * phase)

        elif coil_type == 'solenoid':
            # Vertically oriented Solenoid (good for small samples, or specific geometries)
            # Uniform in Y, falloff in X? Or just very strong center.
            # Let's model a localized high-sensitivity solenoid
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            sensitivity = 1.5 * np.exp(-r**2 / (2 * (N//3)**2))
            self.coils.append(sensitivity)

    def acquire_signal(self, sequence_type='SE', TR=2000, TE=100, TI=500, flip_angle=30, noise_level=0.01):
        """
        Simulates Pulse Sequence acquisition.
        Returns k-space data per coil.
        """
        # Quantum Noise Reduction Factor
        q_factor = 1.0
        
        if sequence_type == 'SE':
            M = self.pd_map * (1 - np.exp(-TR / self.t1_map)) * np.exp(-TE / self.t2_map)
            
        elif sequence_type == 'GRE':
            t2_star = self.t2_map / 2
            FA_rad = np.radians(flip_angle)
            E1 = np.exp(-TR / self.t1_map)
            numerator = (1 - E1) * np.sin(FA_rad)
            denominator = 1 - np.cos(FA_rad) * E1
            M = self.pd_map * (numerator / denominator) * np.exp(-TE / t2_star)
            
        elif sequence_type in ['InversionRecovery', 'FLAIR']:
            M = self.pd_map * (1 - 2*np.exp(-TI/self.t1_map) + np.exp(-TR/self.t1_map)) * np.exp(-TE/self.t2_map)
            M = np.abs(M) 
            
        elif sequence_type == 'SSFP':
            FA_rad = np.radians(flip_angle)
            E1 = np.exp(-TR / self.t1_map)
            E2 = np.exp(-TR / self.t2_map)
            num = (1 - E1) * np.sin(FA_rad)
            den = 1 - (E1 - E2)*np.cos(FA_rad) - E1*E2
            M = self.pd_map * (num / den) * np.exp(-TE / self.t2_map)
            
        elif sequence_type == 'QuantumEntangled':
            # Quantum Entangled sequence: Uses entangled photons/spins to reduce noise floor
            # Mathematically equivalent to higher SNR or lower noise_level
            # Also assumes ideal contrast
            M = self.pd_map * (1 - np.exp(-TR / self.t1_map)) # T1 weight
            q_factor = 0.1 # 10x noise reduction
            
        elif sequence_type == 'ZeroPointGradients':
            # Hypothetical sequence utilizing zero-point energy fluctuations (Sci-Fi/Advanced)
            # Incredible T2* contrast
            M = self.pd_map * np.exp(-TE / (self.t2_map / 4.0)) * 2.0
            q_factor = 0.05 # 20x noise reduction

        elif sequence_type == 'QuantumStatisticalCongruence':
            # Uses statistical congruences between T1 and T2 manifolds to optimize signal
            # Maximizes mutual information between relaxation parameters
            # Simulates a "perfect" contrast where T1 and T2 distributions overlap constructively
            
            # Normalize T1 and T2 for comparison
            t1_norm = self.t1_map / (np.max(self.t1_map) + 1e-9)
            t2_norm = self.t2_map / (np.max(self.t2_map) + 1e-9)
            
            # Congruence factor: High where T1 and T2 normalized profiles are distinct (high grad) or congruent?
            # Let's say we want to highlight structural complexity.
            # We'll use a weighting that favors regions where PD is high AND (T1/T2 ratio is distinct)
            
            ratio_map = self.t1_map / (self.t2_map + 1e-5)
            # Logarithmic compression of ratio to keep it bounded
            contrast_weight = np.log1p(ratio_map)
            contrast_weight = contrast_weight / np.max(contrast_weight)
            
            # Base signal
            M = self.pd_map * contrast_weight
            
            # This method theoretically cancels out thermal noise via statistical averaging
            q_factor = 0.01 # 100x noise reduction (near perfect)
            
        else:
            M = self.pd_map
            
        M = np.nan_to_num(M) 
        
        # 2. Apply Coil Sensitivities and FFT
        kspace_data = []
        effective_noise = noise_level * q_factor
        
        for coil_map in self.coils:
            # Received Signal in Image Space = M * Sensitivity
            img_space_signal = M * coil_map
            
            # FFT to K-Space
            k_space = np.fft.fftshift(np.fft.fft2(img_space_signal))
            
            # Add Gaussian White Noise in K-Space
            noise = np.random.normal(0, effective_noise * np.max(np.abs(k_space)), k_space.shape) + \
                    1j * np.random.normal(0, effective_noise * np.max(np.abs(k_space)), k_space.shape)
            
            kspace_data.append(k_space + noise)
            
        return kspace_data, M

    def reconstruct_image(self, kspace_data, method='SoS'):
        """
        Reconstructs image from multicoil k-space data.
        SoS: Sum of Squares
        """
        # 1. IFFT per coil
        coil_images = []
        for k in kspace_data:
            img = np.fft.ifft2(np.fft.ifftshift(k))
            coil_images.append(img)
            
        # 2. Combine
        if method == 'SoS':
            # Root Sum of Squares
            combined = np.sqrt(sum(np.abs(img)**2 for img in coil_images))
        else:
            combined = np.abs(coil_images[0])
            
        return combined, coil_images

    def compute_metrics(self, reconstructed, reference_M):
        """Calculates Contrast and normalized Error."""
        # Normalize
        rec_norm = reconstructed / (np.max(reconstructed) + 1e-9)
        ref_norm = reference_M / (np.max(reference_M) + 1e-9)
        
        # Contrast (e.g. GM vs WM)
        # Using fixed indices for simplicity based on generated phantom
        N = self.resolution
        wm_val = np.mean(rec_norm[N//2-10:N//2+10, N//3])
        gm_val = np.mean(rec_norm[N//2-10:N//2+10, 10]) # Edge
        contrast = abs(wm_val - gm_val)
        
        # Resolution/Sharpness (Gradient magnitude mean)
        grads = np.gradient(rec_norm)
        sharpness = np.mean(np.sqrt(grads[0]**2 + grads[1]**2))
        
        return {
            "contrast": float(contrast),
            "sharpness": float(sharpness) * 100, # Scale up
            "max_signal": float(np.max(reconstructed))
        }

    def generate_plots(self, kspace_data, reconstructed_img, reference_M):
        """Generates dictionary of base64 encoded plots."""
        plots = {}
        plt.style.use('dark_background')
        
        def fig_to_b64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', transparent=True, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        # 1. Reconstructed Image
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        fig1.patch.set_facecolor('#0f172a')
        ax1.imshow(reconstructed_img, cmap='gray', origin='lower')
        ax1.set_title("Reconstructed Image")
        ax1.axis('off')
        plots['recon'] = fig_to_b64(fig1)
        
        # 2. K-Space (Log Magnitude)
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        fig2.patch.set_facecolor('#0f172a')
        avg_k = np.mean(np.abs(np.array(kspace_data)), axis=0)
        ax2.imshow(np.log(avg_k + 1e-5), cmap='viridis', origin='lower')
        ax2.set_title("K-Space (Log Mag)")
        ax2.axis('off')
        plots['kspace'] = fig_to_b64(fig2)

        # 3. Signal Profile
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        fig3.patch.set_facecolor('#0f172a')
        mid = reconstructed_img.shape[0] // 2
        ax3.plot(reconstructed_img[mid, :], color='#38bdf8', label='Recon')
        ax3.plot(reference_M[mid, :], color='#94a3b8', linestyle='--', alpha=0.5, label='Ground Truth')
        ax3.set_title("Center Line Profile")
        ax3.legend()
        ax3.grid(True, alpha=0.2)
        plots['profile'] = fig_to_b64(fig3)
        
        # 4. Ground Truth (Ideal Magnetization)
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        fig4.patch.set_facecolor('#0f172a')
        ax4.imshow(reference_M, cmap='gray', origin='lower')
        ax4.set_title("Ground Truth (Ideal M)")
        ax4.axis('off')
        plots['phantom'] = fig_to_b64(fig4)

        # 5. Ideal K-Space (No Noise/Coils)
        fig5, ax5 = plt.subplots(figsize=(6, 6))
        fig5.patch.set_facecolor('#0f172a')
        k_gt = np.fft.fftshift(np.fft.fft2(reference_M))
        ax5.imshow(np.log(np.abs(k_gt) + 1e-5), cmap='viridis', origin='lower')
        ax5.set_title("Ideal K-Space (No Noise)")
        ax5.axis('off')
        plots['kspace_gt'] = fig_to_b64(fig5)
        
        return plots
