import numpy as np
import scipy.ndimage
import scipy.fftpack
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import nibabel as nib
from nibabel.testing import data_path
import os
from skimage.transform import resize
from llm_modules import StatisticalClassifier

class MRIReconstructionSimulator:
    def __init__(self, resolution=128):
        self.resolution = resolution
        self.dims = (resolution, resolution)
        self.t1_map = None
        self.t2_map = None
        self.pd_map = None 
        self.coils = []
        self.classifier = StatisticalClassifier()
        

    def generate_brain_phantom(self):
        # Legacy support
        self.setup_phantom(use_real_data=False)

    def setup_phantom(self, use_real_data=True):
        """Generates T1, T2, PD maps. Tries to load real data first."""
        if use_real_data:
            if self.load_real_data():
                return
        
        # Fallback to synthetic
        self._generate_synthetic_phantom()

    def load_real_data(self):
        try:
            # Look for a standard example file or a specific one
            # Try finding *any* .nii file in current dir or Data dir
            target_file = None
            
            common_paths = [
                os.path.join(os.getcwd(), 'example_nifti.nii.gz'),
                os.path.join(data_path, 'example_nifti.nii.gz'),
                os.path.join(data_path, 'example4d.nii.gz'),
                os.path.join(os.getcwd(), 'Data', 'structure.nii') 
            ]
            
            for p in common_paths:
                if os.path.exists(p):
                    target_file = p
                    break
            
            if not target_file:
                # Try finding any nii in subdirs (shallow search)
                for root, dirs, files in os.walk(os.getcwd()):
                    for f in files:
                        if f.endswith('.nii') or f.endswith('.nii.gz'):
                            target_file = os.path.join(root, f)
                            break
                    if target_file: break
            
            if not target_file:
                print("No NIfTI file found.")
                return False
                
            print(f"Loading real data from {target_file}")
            img = nib.load(target_file)
            data = img.get_fdata()
            
            # Extract slice
            if data.ndim == 4:
                slc = data[:, :, data.shape[2]//2, 0]
            elif data.ndim == 3:
                slc = data[:, :, data.shape[2]//2]
            else:
                slc = data
            
            # Normalize orientation (simple rot90 if needed to match view)
            slc = np.rot90(slc)
            
            # Resize
            slc_resized = resize(slc, self.dims, mode='reflect', anti_aliasing=True)
            
            # Normalize 0-1
            slc_resized = (slc_resized - slc_resized.min()) / (slc_resized.max() - slc_resized.min() + 1e-9)
            
            # Assign Maps based on intensity (Simple approximate segmentation)
            # Assume T1w input: WM is bright, GM is gray, CSF/Air is dark
            
            self.pd_map = slc_resized
            
            # Model: 
            # Air < 0.1
            # CSF < 0.3 (in T1w? actually CSF is Dark in T1w)
            # GM ~ 0.5-0.7
            # WM > 0.8
            
            # Synthesize T1/T2 from this "Structure"
            # T1: WM(700), GM(1200), CSF(4000)
            # T2: WM(80), GM(110), CSF(2000)
            
            # Continuous mapping for simulation visual fun
            # Invert for T1? No, T1 is short for WM (Bright signal in T1w).
            # So Bright Input -> Short T1 (Wait, T1w signal ~ 1/T1... actually Signal ~ (1-exp(-TR/T1)). Short T1 = Bright.)
            # So High Intensity -> Short T1.
            
            # Let's map Intensity (I) [0-1] to T1.
            # I=1 (WM) -> T1=700
            # I=0.6 (GM) -> T1=1200
            # I=0.1 (CSF) -> T1=4000
            # I=0 (Air) -> T1=0 (or 5000)
            
            # Linear interp model won't capture the CSF jump well, but code is robust.
            # Using piecewise construction or just functional
            self.t1_map = 4000 - 3300 * slc_resized # 0->4000, 1->700. Crude but effective for contrast.
            self.t2_map = 2000 - 1900 * slc_resized # 0->2000, 1->100.
            
            # Mask Air firmly
            mask_air = slc_resized < 0.05
            self.t1_map[mask_air] = 0
            self.t2_map[mask_air] = 0
            self.pd_map[mask_air] = 0
            
            # Add Pathology (Plaque Burden)
            self._add_pathology_plaques()
            
            return True
            
        except Exception as e:
            print(f"Error loading neuroimage: {e}")
            return False

    def _add_pathology_plaques(self):
        """Simulates plaque burden (Amyloid/Vascular)."""
        # Plaques are often small, focal lesions.
        # Amyloid: Short T2* (Iron), sometimes iso-intense on T1.
        # Let's model them as small circular regions with specific properties.
        
        N = self.resolution
        num_plaques = np.random.randint(5, 15)
        
        for _ in range(num_plaques):
            # Random location in "Brain" (avoiding air/edges roughly)
            cx = np.random.randint(N//4, 3*N//4)
            cy = np.random.randint(N//4, 3*N//4)
            radius = np.random.randint(1, 4)
            
            y, x = np.ogrid[:N, :N]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            
            # Check if we are in tissue (PD > 0.2)
            if np.mean(self.pd_map[mask]) > 0.2:
                # Plaque Signatures
                # T1: Slightly Hypointense (Darker) or Iso. Let's drop T1 a bit.
                self.t1_map[mask] *= 0.8 
                
                # T2: Hypointense (Dark on T2/T2* due to iron/paramagnetic)
                self.t2_map[mask] = 40 # Very Short T2
                
                # PD: Can be lower (atrophy/space occupying)
                self.pd_map[mask] *= 0.9 

    def _generate_synthetic_phantom(self):
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
        
        self.active_coil_type = coil_type
        
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

        elif coil_type == 'quantum_surface_lattice':
            # Quantum Surface Integral Approach
            # Models a continuous current probability density on a spherical surface (Helmet)
            # Uses discretized surface integral for B1+ (Transmit) and B1- (Receive)
            
            # 1. Define Surface Points (approximate sphere/helmet)
            theta = np.linspace(0, np.pi, 20)
            phi = np.linspace(0, 2*np.pi, 40)
            Theta, Phi = np.meshgrid(theta, phi)
            R_coil = N // 1.8
            
            # Surface currents (Quantum Wavefunction like distribution)
            # J ~ |Psi|^2
            Psi = np.sin(3*Theta) * np.exp(1j * 2 * Phi)
            Current_Density = np.abs(Psi)**2
            
            # 2. Integrate Biot-Savart (Simplified 2D projection for speed)
            # B(r) = int (J(r') x (r-r')) / |r-r'|^3 dA
            
            # We'll create 4 "modes" or "channels" from this surface integral
            # representing different quadrature modes of the surface state
            
            for mode in range(4):
                phase_shift = np.pi/2 * mode
                sensitivity = np.zeros(self.dims, dtype=complex)
                
                # Vectorized approximation of the surface integral
                # Summing contributions from "sources" on the circle R_coil
                # weighted by the "Quantum Surface" distribution
                
                num_sources = 64
                for i in range(num_sources):
                    ang = 2*np.pi*i/num_sources
                    sx = center[1] + R_coil * np.cos(ang)
                    sy = center[0] + R_coil * np.sin(ang)
                    
                    # Source strength modulates with the "Quantum Lattice" freq
                    source_str = np.sin(3*ang + phase_shift)
                    
                    dist_sq = (x - sx)**2 + (y - sy)**2 + 100 # +z^2 regularization
                    
                    # Field ~ 1/r^2
                    # Phase ~ geometric phase
                    field_amp = source_str / (dist_sq)
                    field_phase = np.exp(1j * np.arctan2(y-sy, x-sx))
                    
                    sensitivity += field_amp * field_phase
                
                # Normalize and bound
                sensitivity = sensitivity / (np.max(np.abs(sensitivity)) + 1e-9)
                self.coils.append(sensitivity * 2.5) # Gain factor

        elif coil_type == 'geodesic_chassis':
            # Geodesic Head Coil: Geometric distribution based on Golden Angle
            num_elements = 64 
            phi = (1 + np.sqrt(5)) / 2
            
            for i in range(num_elements):
                # Golden Angle distribution
                idx = i + 0.5
                phi_ang = np.arccos(1 - 2*idx/num_elements)
                theta_ang = 2 * np.pi * idx / phi * phi
                
                # Project sphere to 2D slice (Simulating a helmet cross-section)
                R = N // 2.2
                cx = center[1] + R * np.sin(phi_ang) * np.cos(theta_ang)
                cy = center[0] + R * np.sin(phi_ang) * np.sin(theta_ang)
                cz = R * np.cos(phi_ang)
                
                # Depth attenuation (3D effect)
                dist_sq = (x - cx)**2 + (y - cy)**2 + (cz**2 / 5.0)
                
                # Sensitivity Profile
                sensitivity = 2.0 * np.exp(-dist_sq / (2 * (N//9)**2))
                phase = np.exp(1j * (x * np.cos(theta_ang) + y * np.sin(theta_ang)) * 0.15)
                
                self.coils.append(sensitivity * phase)

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

        elif sequence_type == 'QuantumDualIntegral':
            # "Dual Sense" Simulation:
            # 1. Accounts for Transmit Inhomogeneity (B1+) derived from Surface Integral
            # 2. Accounts for Receive Sensitivity (B1-) derived from Surface Integral
            # 3. Incorporates Geometric Phase (Berry Phase) into the signal
            
            # We assume the first coil map approximates the B1+ transmit profile (Reciprocity)
            if len(self.coils) > 0:
                B1_plus_mag = np.abs(self.coils[0])
                B1_plus_mag = B1_plus_mag / (np.max(B1_plus_mag) + 1e-9) # Normalize
            else:
                B1_plus_mag = np.ones(self.dims)

            # Actual Flip Angle viewed by spins is spatially varying
            local_FA = np.radians(flip_angle) * (0.5 + 0.5 * B1_plus_mag) 
            
            # GRE-like base with spatially varying FA
            E1 = np.exp(-TR / self.t1_map)
            t2_star = self.t2_map / 1.5 # Surface effects broaden line
            
            # Signal Equation with Local FA
            numerator = (1 - E1) * np.sin(local_FA)
            denominator = 1 - np.cos(local_FA) * E1
            
            # Geometric Phase Factor (Berry Phase from adiabatic surface transport)
            # Simulated as a gradient dependent phase
            grads = np.gradient(self.pd_map)
            berry_phase = np.exp(1j * 5 * (grads[0] + grads[1]))
            
            M = self.pd_map * (numerator / denominator) * np.exp(-TE / t2_star) * berry_phase
            M = np.abs(M) # Take magnitude for M matrix gen (phase handled in k-space loop typically but here we bake it in)
            
            q_factor = 0.02
            
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
        elif method == 'Variational':
            # Variational Theory Denoising
            # First SoS
            combined_raw = np.sqrt(sum(np.abs(img)**2 for img in coil_images))
            # Apply Variational Denoising
            combined = self.classifier.variational_denoise(combined_raw, lambda_tv=0.05)
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

        # 6. Circuit Diagram (New)
        plots['circuit'] = self.generate_circuit_diagram()
        
        return plots

    def generate_circuit_diagram(self):
        """Generates a circuit schematic for the active coil."""
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')
        ax.axis('off')
        
        # Determine coil type from self.coils generation context 
        # (Since we don't store coil_type in self, we'll infer or just default to a generic/geodesic one for now if not stored.
        # Ideally, we should store self.coil_type in __init__ or generate_coil_sensitivities, but let's assume Geodesic/Phased Array logic 
        # as that matches the user request. Realistically, we'll create a generic 'Phased Array Element' schematic.)
        
        # Determine coil type from self.active_coil_type
        coil_type = getattr(self, 'active_coil_type', 'standard')
        
        # Common Style
        color_wire = '#94a3b8'
        color_comp = '#38bdf8'
        
        if coil_type == 'standard':
            # Birdcage: High-Pass Ladder Network
            # Legs with Caps, Rungs with Inductors (or vice versa depending on mode)
            ax.set_title("Birdcage Coil (High-Pass Ladder)", color='white', fontsize=12)
            
            # Draw Rings
            t = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(t), np.sin(t), color=color_wire, lw=2) # Top Ring
            ax.plot(0.7*np.cos(t), 0.7*np.sin(t), color=color_wire, lw=2) # Bottom Ring
            
            # Legs (Capacitors)
            for i in range(8):
                ang = 2*np.pi * i / 8
                x1, y1 = np.cos(ang), np.sin(ang)
                x2, y2 = 0.7*np.cos(ang), 0.7*np.sin(ang)
                
                # Leg Wire with Gap for Cap
                # Simple line for now
                ax.plot([x1, x2], [y1, y2], color=color_comp, lw=2)
                # Capacitor Symbol perpendicular
                mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
                # Tiny cross per leg
                ax.text(mid_x, mid_y, "||", color='white', ha='center', va='center', rotation=np.degrees(ang)+90, fontsize=8)
                
            ax.text(0, 0, "8-Rung\nBirdcage", color='white', ha='center', va='center')
            
        elif coil_type == 'solenoid':
             # Solenoid Schematic
            ax.set_title("Solenoid Coil (High-Q)", color='white', fontsize=12)
            
            # Spiral
            t = np.linspace(0, 6*np.pi, 200)
            z = np.linspace(-1, 1, 200)
            x = np.cos(t)
            y = z 
            # 2D projection...
            # Just draw a zig zag inductor
            x_ind = np.linspace(1, 5, 100)
            y_ind = 2 + 0.5 * np.sin(10*x_ind)
            ax.plot(x_ind, y_ind, color=color_wire, lw=3)
            ax.text(3, 2.7, "L_solenoid", color=color_comp, ha='center')
            
            # Tuning Cap parallel
            ax.plot([1, 1], [2, 1], color=color_wire, lw=2)
            ax.plot([5, 5], [2, 1], color=color_wire, lw=2)
            ax.plot([1, 2.8], [1, 1], color=color_wire, lw=2)
            ax.plot([3.2, 5], [1, 1], color=color_wire, lw=2)
            # Cap
            ax.plot([2.8, 2.8], [0.8, 1.2], color=color_comp, lw=2)
            ax.plot([3.2, 3.2], [0.8, 1.2], color=color_comp, lw=2)
            ax.text(3, 0.5, "C_tune (10pF)", color=color_comp, ha='center')

        elif coil_type in ['geodesic_chassis', 'n25_array']:
            # Geodesic Element Circuit
            # Tuning, Matching, Decoupling
            ax.set_title(f"Geodesic Element (Golden Angle Node)", color='white', fontsize=12)
            
            # Coordinates
            y_rail = 2
            x_step = 2
            
            # Loop
            ax.plot([0, 1], [y_rail, y_rail], color=color_wire, lw=2)
            # Box for Coil
            ax.plot([1, 1.2], [y_rail+0.2, y_rail-0.2], color=color_wire, lw=2)
            ax.plot([1.2, 1.4], [y_rail-0.2, y_rail+0.2], color=color_wire, lw=2)
            ax.plot([1.4, 1.6], [y_rail+0.2, y_rail-0.2], color=color_wire, lw=2)
            ax.plot([1.6, 1.8], [y_rail-0.2, y_rail], color=color_wire, lw=2)
            ax.text(1.4, y_rail+0.5, "L_geo", color=color_comp, ha='center')
            
            # Match Network
            ax.plot([1.8, 3], [y_rail, y_rail], color=color_wire, lw=2)
            
            # Series Cap (Tune)
            ax.plot([3, 3], [y_rail+0.2, y_rail-0.2], color=color_comp, lw=2) # Plate 1 - Wrong implementation of gap
            # Draw properly: Gap
            ax.plot([3, 3.2], [y_rail, y_rail], color='#000000', lw=4) # Erase? No, drawing bg usually works but let's just shift
            
            # Let's do a simple node graph
            # Node 0 -> L -> Node 1 -> C_tune -> Node 2 -> C_match_gnd -> Node 3 (Out)
            nodes = [0, 2, 4, 6]
            h = 3
            
            # L
            ax.plot([0, 2], [h, h], color=color_wire, lw=2)
            ax.text(1, h+0.2, "L_loop", color=color_comp, ha='center')
            
            # C_tune
            ax.plot([2, 3], [h, h], color=color_wire, lw=2)
            ax.text(2.5, h+0.2, "C_tune", color=color_comp, ha='center')
            # Symbol
            ax.plot([2.4, 2.4], [h-0.2, h+0.2], color=color_comp, lw=2)
            ax.plot([2.6, 2.6], [h-0.2, h+0.2], color=color_comp, lw=2)
            
            # C_match
            ax.plot([4, 4], [h, 1], color=color_wire, lw=2)
            ax.text(4.2, 2, "C_match", color=color_comp)
            # Symbol
            ax.plot([3.8, 4.2], [2.1, 2.1], color=color_comp, lw=2)
            ax.plot([3.8, 4.2], [1.9, 1.9], color=color_comp, lw=2)
            
            # Ground
            ax.plot([0, 6], [1, 1], color=color_wire, lw=2)
            ax.text(3, 0.5, "Common Ground / Shield", color='gray', ha='center')
            
            # Output
            ax.plot([4, 6], [h, h], color=color_wire, lw=2)
            ax.text(6, h, "> Preamp", color=color_wire, va='center')
            
        else:
             # Generic
            ax.set_title(f"Generic RF Front-End ({coil_type})", color='white', fontsize=12)
            ax.text(0.5, 0.5, "Standard Matching Network\nL-C-C Topology", color='white', ha='center', va='center', transform=ax.transAxes)
        
        # Render
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
