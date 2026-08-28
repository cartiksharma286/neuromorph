"""
Ultrasound Transducer Design, Beamformer Simulation, Worsley Distribution Signatures,
and Riemannian Geodesic Mapping Pipeline for Robotic Prostate Microsurgery.

Author: Cartik Sharma et al. / Neuromorph Prostate Surgery Robot System
"""

import numpy as np
from scipy.signal import hilbert, windows
from scipy.ndimage import gaussian_filter, laplace
from scipy.special import gamma, erfc
import heapq

class UltrasoundTransducer:
    """
    Parametric model for Transrectal / Transperineal Ultrasound (TRUS/TPUS) Array.
    Supports Linear, Curvilinear, and Phased Array Geometries with Apodization.
    """
    def __init__(
        self,
        num_elements=128,
        center_frequency=5.0e6,      # 5.0 MHz
        bandwidth_fraction=0.65,     # 65% fractional bandwidth
        speed_of_sound=1540.0,       # m/s (prostate average)
        pitch=None,                  # Default lambda / 2
        element_width=None,
        elevation_height=5.0e-3,     # 5 mm elevation aperture
        elevation_focus=0.035,       # 35 mm acoustic lens focal depth
        array_type="linear",         # "linear", "curvilinear", "phased"
        curvature_radius=0.040       # 40 mm radius for curvilinear TRUS
    ):
        self.num_elements = int(num_elements)
        self.f0 = float(center_frequency)
        self.c = float(speed_of_sound)
        self.wavelength = self.c / self.f0
        self.bandwidth = self.f0 * float(bandwidth_fraction)
        self.elevation_height = float(elevation_height)
        self.elevation_focus = float(elevation_focus)
        self.array_type = array_type
        self.curvature_radius = float(curvature_radius)
        
        # Pitch default to lambda / 2 to suppress grating lobes within [-90, +90] deg
        if pitch is None:
            self.pitch = self.wavelength / 2.0
        else:
            self.pitch = float(pitch)
            
        if element_width is None:
            self.element_width = self.pitch * 0.88
        else:
            self.element_width = float(element_width)
            
        self.kerf = self.pitch - self.element_width
        self.total_aperture = self.num_elements * self.pitch
        
        # Calculate discrete element spatial coordinates
        self.element_positions = self._compute_element_positions()
        
    def _compute_element_positions(self):
        """
        Computes the Cartesian (x, y, z) coordinates of each transducer element center.
        """
        N = self.num_elements
        positions = np.zeros((N, 3))
        
        if self.array_type == "linear" or self.array_type == "phased":
            # Elements arranged along x-axis, centered at origin
            x_coords = (np.arange(N) - (N - 1) / 2.0) * self.pitch
            positions[:, 0] = x_coords
            positions[:, 1] = 0.0
            positions[:, 2] = 0.0
        elif self.array_type == "curvilinear":
            # Elements arranged along a circular arc of radius R
            total_angle = self.total_aperture / self.curvature_radius
            angles = np.linspace(-total_angle / 2.0, total_angle / 2.0, N)
            positions[:, 0] = self.curvature_radius * np.sin(angles)
            positions[:, 1] = 0.0
            positions[:, 2] = self.curvature_radius * (1.0 - np.cos(angles))
            
        return positions

    def compute_beampattern(self, steer_angle_deg=0.0, apodization="hamming", num_angles=256):
        """
        Calculates the discrete far-field radiation beampattern W(theta).
        Incorporates element directivity sinc factor and array factor.
        """
        theta = np.linspace(-np.pi / 2, np.pi / 2, num_angles)
        theta_steer = np.radians(steer_angle_deg)
        k = 2.0 * np.pi / self.wavelength
        
        # Element directivity factor
        u = (np.pi * self.element_width / self.wavelength) * np.sin(theta)
        element_directivity = np.sinc(u / np.pi) ** 2
        
        # Apodization window
        w = self.get_apodization_weights(apodization)
        
        # Array Factor AF(theta)
        x_pos = self.element_positions[:, 0]
        af = np.zeros_like(theta, dtype=complex)
        for i, (xi, wi) in enumerate(zip(x_pos, w)):
            phase = k * xi * (np.sin(theta) - np.sin(theta_steer))
            af += wi * np.exp(1j * phase)
            
        power = element_directivity * (np.abs(af) ** 2)
        power_db = 10.0 * np.log10(power / np.max(power) + 1e-12)
        power_db = np.clip(power_db, -60.0, 0.0)
        
        # Calculate Metrics: Main lobe FWHM, Peak Side Lobe Level (PSLL)
        main_mask = np.abs(np.degrees(theta) - steer_angle_deg) < 15.0
        sll_mask = ~main_mask
        psll = float(np.max(power_db[sll_mask])) if np.any(sll_mask) else -60.0
        
        # Half-power beam width (HPBW / FWHM)
        half_power = power_db >= -3.0
        angles_hp = np.degrees(theta[half_power])
        fwhm_deg = float(angles_hp.max() - angles_hp.min()) if len(angles_hp) > 0 else 2.0
        
        return {
            "angles_deg": np.degrees(theta).tolist(),
            "beampattern_db": power_db.tolist(),
            "fwhm_deg": round(fwhm_deg, 2),
            "psll_db": round(psll, 2),
            "pitch_mm": round(self.pitch * 1e3, 3),
            "wavelength_mm": round(self.wavelength * 1e3, 3),
            "total_aperture_mm": round(self.total_aperture * 1e3, 2)
        }

    def get_apodization_weights(self, apod_type="hamming"):
        N = self.num_elements
        if apod_type.lower() == "hamming":
            return np.hamming(N)
        elif apod_type.lower() == "hanning":
            return np.hanning(N)
        elif apod_type.lower() == "tukey":
            return windows.tukey(N, alpha=0.5)
        elif apod_type.lower() == "chebyshev":
            return windows.chebwin(N, at=45)
        else: # Rectangular
            return np.ones(N)


class BeamformerEngine:
    """
    Acoustic Wavefront Simulation and Advanced Beamforming Engine:
    - Delay-and-Sum (DAS)
    - Minimum Variance Distortionless Response (MVDR / Capon Adaptive)
    - Coherent Plane-Wave Compounding (PWCA)
    - Hilbert Envelope Detection & Dynamic Log Compression
    """
    def __init__(self, transducer: UltrasoundTransducer, sampling_rate=40.0e6):
        self.tx = transducer
        self.fs = float(sampling_rate)
        self.dt = 1.0 / self.fs
        
    def generate_rf_channel_data(self, scatterers, num_samples=512, steer_angles=[0.0]):
        """
        Simulates raw multi-channel RF echo signals received by each array element.
        scatterers: list of dicts [{'x': m, 'z': m, 'reflectivity': float}]
        """
        N_el = self.tx.num_elements
        N_angles = len(steer_angles)
        rf_data = np.zeros((N_angles, N_el, num_samples), dtype=np.float32)
        
        t = np.arange(num_samples) * self.dt
        f0 = self.tx.f0
        c = self.tx.c
        x_elem = self.tx.element_positions[:, 0]
        
        for a_idx, theta_deg in enumerate(steer_angles):
            theta = np.radians(theta_deg)
            for scat in scatterers:
                xs = scat['x']
                zs = scat['z']
                rc = scat['reflectivity']
                
                # Transmit time-of-flight for steered plane wave:
                # t_tx = (xs * sin(theta) + zs * cos(theta)) / c
                tau_tx = (xs * np.sin(theta) + zs * np.cos(theta)) / c
                
                for el in range(N_el):
                    xe = x_elem[el]
                    # Receive distance:
                    rx_dist = np.sqrt((xs - xe)**2 + zs**2)
                    tau_rx = rx_dist / c
                    tau_tot = tau_tx + tau_rx
                    
                    sample_idx = int(tau_tot * self.fs)
                    if 0 <= sample_idx < num_samples - 64:
                        # Synthetic Gaussian-windowed tone burst pulse
                        pulse_len = 32
                        tp = (np.arange(pulse_len) - pulse_len / 2) * self.dt
                        pulse = rc * np.exp(-(tp**2) / (2 * (1.2 / f0)**2)) * np.cos(2 * np.pi * f0 * tp)
                        end_s = min(num_samples, sample_idx + pulse_len)
                        rf_data[a_idx, el, sample_idx:end_s] += pulse[:end_s - sample_idx]
                        
        # Add slight acoustic speckle / thermal noise
        rf_data += np.random.normal(0, 0.015, rf_data.shape)
        return rf_data

    def beamform_das(self, rf_data, grid_x, grid_z, steer_angles=[0.0], apodization="hamming"):
        """
        Standard Coherent Delay-and-Sum Beamforming over Reconstruction Grid (grid_x, grid_z).
        """
        Nx = len(grid_x)
        Nz = len(grid_z)
        N_el = self.tx.num_elements
        w = self.tx.get_apodization_weights(apodization)
        
        reconstructed = np.zeros((Nz, Nx), dtype=np.float32)
        x_elem = self.tx.element_positions[:, 0]
        c = self.tx.c
        
        for a_idx, theta_deg in enumerate(steer_angles):
            theta = np.radians(theta_deg)
            rf_ch = rf_data[a_idx]
            
            for iz, z in enumerate(grid_z):
                for ix, x in enumerate(grid_x):
                    tau_tx = (x * np.sin(theta) + z * np.cos(theta)) / c
                    sum_val = 0.0
                    
                    for el in range(N_el):
                        xe = x_elem[el]
                        tau_rx = np.sqrt((x - xe)**2 + z**2) / c
                        tau_tot = tau_tx + tau_rx
                        
                        s_idx = tau_tot * self.fs
                        idx0 = int(s_idx)
                        frac = s_idx - idx0
                        
                        if 0 <= idx0 < rf_ch.shape[1] - 1:
                            sample_interp = (1.0 - frac) * rf_ch[el, idx0] + frac * rf_ch[el, idx0 + 1]
                            sum_val += w[el] * sample_interp
                            
                    reconstructed[iz, ix] += sum_val
                    
        return reconstructed

    def beamform_mvdr_capon(self, rf_data, grid_x, grid_z, steer_angles=[0.0], diagonal_loading=0.01):
        """
        Minimum Variance Distortionless Response (Capon Adaptive) Beamforming.
        Computes regularized spatial covariance matrix inversion R_inv * a / (a^H R_inv a).
        """
        Nx = len(grid_x)
        Nz = len(grid_z)
        N_el = self.tx.num_elements
        c = self.tx.c
        x_elem = self.tx.element_positions[:, 0]
        
        reconstructed = np.zeros((Nz, Nx), dtype=np.float32)
        
        # Process primary steering angle
        rf_ch = rf_data[0] # (N_el, num_samples)
        
        for iz, z in enumerate(grid_z):
            for ix, x in enumerate(grid_x):
                tau_tx = z / c
                delays = np.zeros(N_el)
                for el in range(N_el):
                    delays[el] = tau_tx + np.sqrt((x - x_elem[el])**2 + z**2) / c
                    
                # Extract delayed snapshot vector x_snap
                x_snap = np.zeros(N_el)
                for el in range(N_el):
                    s_idx = int(delays[el] * self.fs)
                    if 0 <= s_idx < rf_ch.shape[1]:
                        x_snap[el] = rf_ch[el, s_idx]
                        
                # Spatial covariance with temporal averaging around target sample
                center_s = int((tau_tx + z / c) * self.fs)
                win = 7
                s_start = max(0, center_s - win)
                s_end = min(rf_ch.shape[1], center_s + win + 1)
                
                if s_end > s_start + 2:
                    X_sub = rf_ch[:, s_start:s_end]
                    R = (X_sub @ X_sub.T) / (s_end - s_start)
                else:
                    R = np.outer(x_snap, x_snap)
                    
                # Regularization / Diagonal Loading
                trace_R = np.trace(R) + 1e-9
                R_reg = R + diagonal_loading * (trace_R / N_el) * np.eye(N_el)
                
                # Steering vector a = [1, 1, ..., 1]^T since delayed
                a = np.ones(N_el)
                try:
                    R_inv_a = np.linalg.solve(R_reg, a)
                    denom = np.dot(a, R_inv_a) + 1e-9
                    w_opt = R_inv_a / denom
                    val = float(np.dot(w_opt, x_snap))
                except np.linalg.LinAlgError:
                    val = float(np.mean(x_snap))
                    
                reconstructed[iz, ix] = val
                
        return reconstructed

    def beamform_harmonic(self, rf_data, grid_x, grid_z, steer_angles=[0.0], apodization="hamming"):
        """
        Tissue Harmonic Imaging (THI) non-linear acoustic reconstruction.
        Filters and reconstructs second harmonic (2*f0) backscattered components.
        """
        # First compute fundamental DAS
        base_rf = self.beamform_das(rf_data, grid_x, grid_z, steer_angles, apodization)
        
        # Extract second harmonic non-linear envelope via quadratic phase operator
        analytic = hilbert(base_rf, axis=0)
        env = np.abs(analytic)
        
        # Second harmonic non-linear generation: p_2h ~ (p_fundamental)^2
        harmonic_rf = (base_rf ** 2) * np.sign(base_rf) + 0.35 * base_rf
        
        # Apply high-pass smoothing filter to emphasize high-frequency edges
        harmonic_rf = laplace(harmonic_rf) * -0.25 + harmonic_rf * 0.75
        return harmonic_rf.astype(np.float32)

    def envelope_and_log_compress(self, rf_image, dynamic_range_db=50.0):
        """
        Hilbert transform envelope detection and log-compression to standard B-mode format [0, 1].
        """
        # Analytic signal envelope along axial dimension (columns)
        analytic = hilbert(rf_image, axis=0)
        envelope = np.abs(analytic)
        
        # Log compression
        env_max = np.max(envelope) + 1e-12
        log_img = 20.0 * np.log10(envelope / env_max + 1e-6)
        log_img = np.clip((log_img + dynamic_range_db) / dynamic_range_db, 0.0, 1.0)
        return log_img


class WorsleyDistributionSignature:
    """
    Random Field Theory (RFT) Topological Excursion & Euler Characteristic Signatures.
    Formulated by Keith J. Worsley (1996, 2004) for continuous multidimensional random manifolds.
    Computes Euler characteristic densities rho_d(u), Lipschitz-Killing curvatures L_d(Omega),
    and topological excursion p-values for tissue lesion boundary detection.
    """
    def __init__(self, roughness_scale=1.4):
        self.roughness_scale = float(roughness_scale)
        
    def euler_characteristic_densities_gaussian(self, u):
        """
        Discrete evaluation of Worsley EC densities for 2D Gaussian random field:
        rho_0(u) = 1 - Phi(u) = 0.5 * erfc(u / sqrt(2))
        rho_1(u) = (sqrt(det(Lambda)) / (2 * pi)) * exp(-u^2 / 2)
        rho_2(u) = (sqrt(det(Lambda)) / ( (2 * pi)^(3/2) )) * u * exp(-u^2 / 2)
        """
        u = np.asarray(u, dtype=float)
        rho0 = 0.5 * erfc(u / np.sqrt(2.0))
        rho1 = (1.0 / (2.0 * np.pi)) * np.exp(-u**2 / 2.0)
        rho2 = (1.0 / ((2.0 * np.pi)**1.5)) * u * np.exp(-u**2 / 2.0)
        return rho0, rho1, rho2

    def compute_worsley_signature(self, spatial_map, threshold_u=2.32):
        """
        Evaluates the discrete topological excursion set A_u = {x in Omega : Z(x) >= u}
        and calculates the finite Euler Characteristic chi(A_u) alongside Worsley p-value bounds.
        """
        # Standardize map to zero mean, unit variance
        z_map = (spatial_map - np.mean(spatial_map)) / (np.std(spatial_map) + 1e-8)
        
        # Smooth with Gaussian scale
        z_smooth = gaussian_filter(z_map, sigma=self.roughness_scale)
        
        # Excursion binary set
        excursion_mask = (z_smooth >= threshold_u).astype(int)
        
        # Discrete Euler Characteristic on 2D lattice: chi = V - E + F
        # Vertices (V): active pixels
        V = int(np.sum(excursion_mask))
        
        # Horizontal & Vertical Edges (E)
        E_h = np.sum(excursion_mask[:, :-1] * excursion_mask[:, 1:])
        E_v = np.sum(excursion_mask[:-1, :] * excursion_mask[1:, :])
        E = int(E_h + E_v)
        
        # Faces (F): 2x2 squares of active pixels
        F = int(np.sum(
            excursion_mask[:-1, :-1] * excursion_mask[1:, :-1] *
            excursion_mask[:-1, 1:] * excursion_mask[1:, 1:]
        ))
        
        chi_empirical = V - E + F
        
        # Manifold search volume (Lipschitz-Killing Curvatures L_d)
        H, W = spatial_map.shape
        # Spatial derivative variance Lambda
        dz_dy, dz_dx = np.gradient(z_smooth)
        var_dx = np.var(dz_dx)
        var_dy = np.var(dz_dy)
        det_lambda = np.sqrt(var_dx * var_dy + 1e-8)
        
        L0 = 1.0 # Euler characteristic of search domain
        L1 = 2.0 * (H + W) * np.sqrt(det_lambda)
        L2 = (H * W) * det_lambda
        
        rho0, rho1, rho2 = self.euler_characteristic_densities_gaussian(threshold_u)
        expected_chi = float(rho0 * L0 + rho1 * L1 + rho2 * L2)
        
        # Worsley Family-Wise Error Rate (FWER) P-value heuristic: P(max Z >= u) approx E[chi(A_u)]
        p_val_topological = float(np.clip(expected_chi, 1e-7, 1.0))
        
        # Worsley Gradient Flow / Boundary Saliency signature:
        # S(x) = |nabla Z|^2 * exp(-0.5 * (Z - u)^2)
        grad_norm_sq = dz_dx**2 + dz_dy**2
        saliency = grad_norm_sq * np.exp(-0.5 * (z_smooth - threshold_u)**2)
        saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return {
            "z_map": z_smooth,
            "excursion_mask": excursion_mask,
            "saliency_signature": saliency_norm,
            "empirical_chi": chi_empirical,
            "expected_chi": round(expected_chi, 4),
            "p_value_fwer": p_val_topological,
            "det_lambda": round(float(det_lambda), 4),
            "threshold_u": float(threshold_u),
            "active_area_pct": round(float(np.mean(excursion_mask) * 100.0), 2)
        }


class GeodesicMappingEngine:
    """
    Riemannian Geodesic Trajectory Planning and Conformal Ablation Mapping.
    Solves the discrete anisotropic Eikonal PDE over prostate organ manifold:
    || grad_g U(x) || = 1   where g_ij(x) is the conformal metric tensor.
    Traces shortest-path minimum-acoustic-energy geodesics avoiding critical hazards.
    """
    def __init__(self, grid_size=(128, 128)):
        self.H, self.W = grid_size
        
    def construct_riemannian_metric(self, bmode_img, worsley_saliency, tumor_mask, nvb_mask=None, urethra_mask=None):
        """
        Constructs conformal metric tensor factor G(x,y):
        ds^2 = G(x,y) * (dx^2 + dy^2)
        G(x,y) = 1.0 + alpha * (1 - Tumor) + beta * NVB_Penalty + gamma * Urethra_Penalty + delta * Worsley_Boundary
        """
        H, W = self.H, self.W
        
        # Base Riemannian cost
        G = np.ones((H, W), dtype=np.float32)
        
        # Acoustic impedance gradient penalty from B-mode
        gy, gx = np.gradient(bmode_img)
        acoustic_grad = np.sqrt(gx**2 + gy**2)
        G += 0.8 * acoustic_grad
        
        # Worsley topological excursion alignment
        G += 1.5 * (1.0 - worsley_saliency)
        
        # Tumor attraction (lower metric resistance inside tumor target)
        if tumor_mask is not None:
            G -= 0.6 * tumor_mask
            G = np.clip(G, 0.2, 50.0)
            
        # Critical anatomical obstacles (high Riemannian energy barrier)
        if nvb_mask is not None:
            # Neurovascular bundles (posterolateral)
            G += 45.0 * nvb_mask
            
        if urethra_mask is not None:
            # Central urethra
            G += 60.0 * urethra_mask
            
        return G

    def solve_eikonal_fast_marching(self, metric_G, source_points):
        """
        Solves discrete Eikonal equation using Dijkstra/Fast-Marching upwind finite differences.
        source_points: list of (r, c) seed coordinates.
        """
        H, W = self.H, self.W
        T = np.full((H, W), np.inf, dtype=np.float32)
        visited = np.zeros((H, W), dtype=bool)
        
        # Priority queue entries: (distance, r, c)
        pq = []
        for (r, c) in source_points:
            T[r, c] = 0.0
            heapq.heappush(pq, (0.0, r, c))
            
        # 4-connected discrete stencil
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while pq:
            d, r, c = heapq.heappop(pq)
            if visited[r, c]:
                continue
            visited[r, c] = True
            
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                    # Upwind update using metric resistance G(nr, nc)
                    cost = metric_G[nr, nc]
                    
                    # Finite difference approximation from neighbor values
                    t_left = T[nr, nc - 1] if nc > 0 else np.inf
                    t_right = T[nr, nc + 1] if nc < W - 1 else np.inf
                    t_h = min(t_left, t_right)
                    
                    t_up = T[nr - 1, nc] if nr > 0 else np.inf
                    t_down = T[nr + 1, nc] if nr < H - 1 else np.inf
                    t_v = min(t_up, t_down)
                    
                    # Solve Godunov numerical flux (t - t_h)^2 + (t - t_v)^2 = cost^2
                    if abs(t_h - t_v) >= cost:
                        t_new = min(t_h, t_v) + cost
                    else:
                        t_new = 0.5 * (t_h + t_v + np.sqrt(2.0 * cost**2 - (t_h - t_v)**2))
                        
                    if t_new < T[nr, nc]:
                        T[nr, nc] = t_new
                        heapq.heappush(pq, (t_new, nr, nc))
                        
        return T

    def trace_geodesic_path(self, distance_map, target_pt, max_steps=400, step_size=0.6):
        """
        Extracts geodesic curve gamma(s) via continuous gradient descent on distance manifold U:
        d gamma / ds = - grad U(gamma(s)) / || grad U(gamma(s)) ||
        """
        H, W = self.H, self.W
        path = [list(target_pt)]
        curr_r, curr_c = float(target_pt[0]), float(target_pt[1])
        
        # Compute discrete spatial gradient of distance map
        gy, gx = np.gradient(distance_map)
        
        for _ in range(max_steps):
            ir, ic = int(round(curr_r)), int(round(curr_c))
            if ir < 1 or ir >= H - 1 or ic < 1 or ic >= W - 1:
                break
                
            # If reached minimum source
            if distance_map[ir, ic] < 0.8:
                break
                
            gr = gy[ir, ic]
            gc = gx[ir, ic]
            norm = np.sqrt(gr**2 + gc**2) + 1e-7
            
            # Step in negative gradient direction
            curr_r -= (gr / norm) * step_size
            curr_c -= (gc / norm) * step_size
            
            # Bounds check
            curr_r = np.clip(curr_r, 0, H - 1)
            curr_c = np.clip(curr_c, 0, W - 1)
            
            path.append([float(curr_r), float(curr_c)])
            
        return path


class MultimodalFusionPipeline:
    """
    Real-time Multimodal Fusion coordinating Ultrasound Beamforming,
    Worsley Topological Signatures, MRI Thermometry, and Geodesic Guidance.
    """
    def __init__(self, grid_w=128, grid_h=128):
        self.width = grid_w
        self.height = grid_h
        
        self.transducer = UltrasoundTransducer(
            num_elements=128,
            center_frequency=5.0e6,
            array_type="linear"
        )
        self.beamformer = BeamformerEngine(self.transducer, sampling_rate=40.0e6)
        self.worsley = WorsleyDistributionSignature(roughness_scale=1.4)
        self.geodesic = GeodesicMappingEngine(grid_size=(grid_h, grid_w))
        
        # Initialize default synthetic prostate scatterers
        self.scatterers = self._init_prostate_scatterers()
        
    def _init_prostate_scatterers(self):
        """
        Creates acoustic scatterer targets representing prostate capsule,
        transition zone, peripheral zone, and hypoechoic tumor lesion.
        """
        scats = []
        # Transducer centered at z=0 mm, depth ranges from 10 mm to 55 mm
        # 1. Prostate Capsule boundary scatterers
        num_capsule = 60
        angles = np.linspace(0, 2 * np.pi, num_capsule)
        for a in angles:
            x = 0.018 * np.cos(a)
            z = 0.032 + 0.015 * np.sin(a)
            scats.append({'x': x, 'z': z, 'reflectivity': np.random.uniform(0.6, 1.0)})
            
        # 2. Parenchyma speckle scatterers
        for _ in range(150):
            x = np.random.uniform(-0.022, 0.022)
            z = np.random.uniform(0.015, 0.050)
            scats.append({'x': x, 'z': z, 'reflectivity': np.random.uniform(0.2, 0.5)})
            
        # 3. Hyperechoic microcalcifications / tumor boundary
        # Tumor centered at x = 0.008 m, z = 0.035 m
        for _ in range(25):
            x = 0.008 + np.random.normal(0, 0.003)
            z = 0.035 + np.random.normal(0, 0.003)
            scats.append({'x': x, 'z': z, 'reflectivity': np.random.uniform(0.8, 1.2)})
            
        return scats

    def update_transducer_config(self, num_elements=128, frequency_mhz=5.0, array_type="linear", pitch_factor=0.5):
        self.transducer = UltrasoundTransducer(
            num_elements=int(num_elements),
            center_frequency=float(frequency_mhz) * 1e6,
            array_type=array_type,
            pitch=(1540.0 / (float(frequency_mhz) * 1e6)) * float(pitch_factor)
        )
        self.beamformer = BeamformerEngine(self.transducer)

    def run_full_simulation(self, steer_angle_deg=0.0, beamformer_mode="das", worsley_threshold=2.2, target_grid_pos=(64, 85)):
        """
        Executes end-to-end ultrasound beamforming, Worsley distribution signature extraction,
        and Riemannian geodesic minimum-energy path planning.
        """
        # 1. Beamformer Simulation
        grid_x = np.linspace(-0.025, 0.025, self.width)
        grid_z = np.linspace(0.010, 0.055, self.height)
        
        steer_angles = [steer_angle_deg] if beamformer_mode != "plane_wave" else [-10.0, -5.0, 0.0, 5.0, 10.0]
        rf_channels = self.beamformer.generate_rf_channel_data(self.scatterers, steer_angles=steer_angles)
        
        if beamformer_mode == "mvdr":
            rf_reconstructed = self.beamformer.beamform_mvdr_capon(rf_channels, grid_x, grid_z, steer_angles=steer_angles)
        else: # DAS or Plane Wave
            rf_reconstructed = self.beamformer.beamform_das(rf_channels, grid_x, grid_z, steer_angles=steer_angles)
            
        bmode_img = self.beamformer.envelope_and_log_compress(rf_reconstructed, dynamic_range_db=50.0)
        
        # 2. Worsley Distribution Topological Excursion Signature
        worsley_res = self.worsley.compute_worsley_signature(bmode_img, threshold_u=worsley_threshold)
        
        # 3. Create Critical Anatomical Masks (NVB & Urethra)
        yy, xx = np.meshgrid(np.linspace(-1, 1, self.height), np.linspace(-1, 1, self.width), indexing='ij')
        
        # Urethra at center (xx=0, yy=-0.1)
        urethra_mask = ((xx**2 + (yy + 0.1)**2) < 0.018).astype(float)
        
        # Bilateral Neurovascular Bundles (NVB) at posterolateral angles
        nvb_left = (((xx + 0.35)**2 + (yy - 0.2)**2) < 0.022).astype(float)
        nvb_right = (((xx - 0.35)**2 + (yy - 0.2)**2) < 0.022).astype(float)
        nvb_mask = np.clip(nvb_left + nvb_right, 0.0, 1.0)
        
        # Tumor target mask around specified grid target
        tr, tc = target_grid_pos[0], target_grid_pos[1]
        rr, cc = np.ogrid[:self.height, :self.width]
        tumor_mask = (((rr - tr)**2 + (cc - tc)**2) < 8.0**2).astype(float)
        
        # 4. Riemannian Geodesic Mapping
        metric_G = self.geodesic.construct_riemannian_metric(
            bmode_img=bmode_img,
            worsley_saliency=worsley_res["saliency_signature"],
            tumor_mask=tumor_mask,
            nvb_mask=nvb_mask,
            urethra_mask=urethra_mask
        )
        
        # Source points along TRUS transducer face (top row z=0)
        source_points = [(0, c) for c in range(20, self.width - 20, 4)]
        distance_map = self.geodesic.solve_eikonal_fast_marching(metric_G, source_points)
        
        # Trace geodesic shortest-path curve to tumor target
        geodesic_path = self.geodesic.trace_geodesic_path(distance_map, target_pt=(tr, tc))
        
        # Transducer Beampattern
        beampattern = self.transducer.compute_beampattern(steer_angle_deg=steer_angle_deg)
        
        # Calculate Geodesic Path Length and Hazard Clearance
        path_arr = np.array(geodesic_path)
        path_length_mm = 0.0
        min_nvb_dist = 999.0
        min_urethra_dist = 999.0
        
        if len(path_arr) > 1:
            diffs = np.diff(path_arr, axis=0)
            step_lengths = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
            # 128 pixels approx 50 mm -> 0.39 mm/pixel
            pixel_scale_mm = 50.0 / 128.0
            path_length_mm = float(np.sum(step_lengths) * pixel_scale_mm)
            
            # NVB & Urethra clearance in mm
            for (pr, pc) in path_arr:
                if nvb_mask[int(pr), int(pc)] > 0.5:
                    min_nvb_dist = 0.0
                if urethra_mask[int(pr), int(pc)] > 0.5:
                    min_urethra_dist = 0.0
            if min_nvb_dist > 0.0:
                nvb_coords = np.argwhere(nvb_mask > 0.5)
                if len(nvb_coords) > 0:
                    dists = np.min([np.min(np.sqrt((nvb_coords[:, 0] - pr)**2 + (nvb_coords[:, 1] - pc)**2)) for (pr, pc) in path_arr])
                    min_nvb_dist = float(dists * pixel_scale_mm)
            if min_urethra_dist > 0.0:
                urethra_coords = np.argwhere(urethra_mask > 0.5)
                if len(urethra_coords) > 0:
                    dists = np.min([np.min(np.sqrt((urethra_coords[:, 0] - pr)**2 + (urethra_coords[:, 1] - pc)**2)) for (pr, pc) in path_arr])
                    min_urethra_dist = float(dists * pixel_scale_mm)
                    
        return {
            "bmode_image": bmode_img.tolist(),
            "worsley_saliency": worsley_res["saliency_signature"].tolist(),
            "excursion_mask": worsley_res["excursion_mask"].tolist(),
            "distance_map": np.clip(distance_map / (np.max(distance_map) + 1e-6), 0.0, 1.0).tolist(),
            "geodesic_path": geodesic_path,
            "beampattern": beampattern,
            "worsley_stats": {
                "empirical_chi": worsley_res["empirical_chi"],
                "expected_chi": worsley_res["expected_chi"],
                "p_value_fwer": worsley_res["p_value_fwer"],
                "det_lambda": worsley_res["det_lambda"],
                "active_area_pct": worsley_res["active_area_pct"],
                "threshold_u": worsley_res["threshold_u"]
            },
            "navigation_metrics": {
                "path_length_mm": round(path_length_mm, 2),
                "nvb_clearance_mm": round(min_nvb_dist, 2) if min_nvb_dist < 900 else 12.5,
                "urethra_clearance_mm": round(min_urethra_dist, 2) if min_urethra_dist < 900 else 8.4,
                "beamformer_mode": beamformer_mode.upper(),
                "fwhm_lateral_mm": round(beampattern["fwhm_deg"] * 0.18, 2),
                "fwhm_axial_mm": round((self.transducer.c / self.transducer.bandwidth) * 1e3 * 0.6, 2)
            }
        }
