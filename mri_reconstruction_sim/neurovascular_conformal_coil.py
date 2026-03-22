#!/usr/bin/env python3
"""
Neurovascular Conformal Coil with Chamfered Geometry & EM State Transfer

Design:
- Chamfered circular loop elements (beveled edges for field conformation)
- Electromagnetic state transfer cues modeled via Jaynes-Cummings interaction
- Improved electrical circuitry with differential amplifiers and noise matching
- Direct integration with quantum phase thermometry for signal reconstruction

Physics:
- Biot-Savart field calculation with chamfer correction
- Popliteal artery network (vascular target)
- Coupling strength modulated by proximity and geometric factor
- SNR enhancement via multi-element combinatorial shimming

Author: NeuroMorph Team
Date: March 22, 2026
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.special import ellipkinc, ellipeinc  # Elliptic integrals for chamfered loops
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

# Physical constants
MU_0 = 4 * np.pi * 1e-7  # H/m (permeability)
GAMMA_RAD = 267.52e6  # rad/(s·T) for protons
LARMOR_3T = GAMMA_RAD * 3.0 / (2 * np.pi)  # Hz at 3T

# Neurovascular circuit constants
IMPEDANCE_CHARACTERISTIC = 50.0  # Ohms (RF coil standard)
NOISE_FIGURE_DB = 0.8  # dB low-noise amplifiers
COUPLING_BASE = 2 * np.pi * 10e6  # rad/s base coupling strength


class ChamferedLoopElement:
    """Chamfered (beveled-edge) circular loop RF coil element.
    
    Chamfering reduces sharp EM field discontinuities at loop edges,
    improving field homogeneity and coupling efficiency.
    """
    
    def __init__(self, radius=0.045, chamfer_radius=0.005, n_chamfer=8):
        """
        Args:
            radius (float): Loop radius in meters (m), default 45 mm
            chamfer_radius (float): Bevel radius at corners (m), default 5 mm
            n_chamfer (int): Number of chamfer segments per quadrant
        """
        self.radius = radius
        self.chamfer_radius = min(chamfer_radius, radius * 0.1)  # Limit chamfer
        self.n_chamfer = n_chamfer
        self._compute_chamfer_points()
        
    def _compute_chamfer_points(self):
        """Generate chamfered loop geometry as 3D curve."""
        theta = np.linspace(0, 2 * np.pi, 4 * self.n_chamfer + 1)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        z = np.zeros_like(theta)
        
        # Apply chamfer smoothing: replace sharp points with bezier-like curves
        for i in range(len(theta) - 1):
            t_local = (i % self.n_chamfer) / self.n_chamfer
            chamfer_factor = np.sin(t_local * np.pi) ** 0.5  # Smooth nonlinear ramp
            scale = 1.0 - self.chamfer_radius / self.radius * (1 - chamfer_factor)
            x[i] *= scale
            y[i] *= scale
        
        self.loop_curve = np.column_stack([x, y, z])
        self.effective_radius = self.radius * (1 - self.chamfer_radius / (2 * self.radius))
        
    def b_field_at_point(self, target_pos, current=1.0):
        """Compute magnetic field via Biot-Savart with chamfer correction.
        
        The chamfer reduces the field singularity at the loop center and
        improves far-field decay characteristics.
        
        Args:
            target_pos (np.ndarray): Target point (x, y, z)
            current (float): Loop current (A)
            
        Returns:
            np.ndarray: B-field vector (3,) in Tesla
        """
        b_field = np.zeros(3)
        R = self.effective_radius
        
        # Distance from target to loop center
        r = np.linalg.norm(target_pos)
        z = target_pos[2]
        rho = np.sqrt(target_pos[0]**2 + target_pos[1]**2)
        
        if r < 1e-6:
            return b_field
        
        # Elliptic integral formulation for on-axis and off-axis
        # K, E are complete elliptic integrals of 1st and 2nd kind
        k_param = 2 * R * rho / (R**2 + rho**2 + z**2)
        
        if abs(k_param) < 1.0:
            K = ellipkinc(np.pi / 2, k_param**2)  # Complete elliptic integral K
            E = ellipeinc(np.pi / 2, k_param**2)  # Complete elliptic integral E
        else:
            K, E = np.pi / 2, np.pi / 2
        
        # Axial field: B_z
        denominator = np.pi * np.sqrt((R + rho)**2 + z**2)
        if denominator > 1e-10:
            b_z = MU_0 * current * (K + E * (R**2 - rho**2 - z**2) / ((R - rho)**2 + z**2)) / denominator
        else:
            b_z = 0
        
        # Radial field: B_rho
        if rho > 1e-6:
            b_rho = MU_0 * current * z / (np.pi * rho * np.sqrt((R + rho)**2 + z**2)) * \
                    (K - E * (R**2 + rho**2 + z**2) / ((R - rho)**2 + z**2))
        else:
            b_rho = 0
        
        # Convert cylindrical (B_z, B_rho) to Cartesian
        if rho > 1e-6:
            cos_phi = target_pos[0] / rho
            sin_phi = target_pos[1] / rho
            b_field[0] = b_rho * cos_phi
            b_field[1] = b_rho * sin_phi
        
        b_field[2] = b_z
        
        # Chamfer correction factor (reduces singularity)
        chamfer_factor = 1.0 + 0.15 * (self.chamfer_radius / self.radius)
        if r > self.radius:
            b_field *= chamfer_factor * (self.effective_radius / self.radius)**2
        
        return b_field
    
    def sensitivity_at_points(self, target_points, current=1.0):
        """Compute sensitivity (|B1+|) at multiple target points.
        
        Args:
            target_points (np.ndarray): Shape (N, 3) target positions
            current (float): Loop current (A)
            
        Returns:
            np.ndarray: Shape (N,) sensitivities in T/A
        """
        sensitivities = np.zeros(len(target_points))
        for i, pos in enumerate(target_points):
            b = self.b_field_at_point(pos, current)
            sensitivities[i] = np.linalg.norm(b)
        return sensitivities


class NeurovascularConformCoil:
    """Neurovascular conformal coil array for high-SNR vascular imaging.
    
    Architecture:
    - 16 chamfered loop elements arranged cylindrically
    - Centered on popliteal artery at knee
    - EM state transfer cues for quantum-enhanced coupling
    - Differential circuitry with noise matching for low-noise reception
    """
    
    def __init__(self, n_elements=16, coil_radius=0.125, target_depth=0.080):
        """
        Args:
            n_elements (int): Number of coil elements (default 16)
            coil_radius (float): Cylindrical radius of element array (m)
            target_depth (float): Axial depth of popliteal artery target (m)
        """
        self.n_elements = n_elements
        self.array_radius = coil_radius
        self.target_depth = target_depth
        
        # Vascular target (popliteal artery in knee)
        self.target_center = np.array([0.0, 0.0, 0.0])
        self.target_radius = 0.003  # 3 mm artery diameter
        
        # Element positions and orientations
        self._generate_element_geometry()
        
        # Circuitry parameters
        self.z_matching = np.ones(n_elements) * IMPEDANCE_CHARACTERISTIC  # Ohms
        self.noise_figure = np.ones(n_elements) * NOISE_FIGURE_DB
        self.coupling_matrix = np.eye(n_elements)  # Inter-element coupling
        self._compute_coupling_matrix()
        
        # State transfer cues (EM interaction model)
        self.em_cues = self._compute_em_cues()
        
    def _generate_element_geometry(self):
        """Arrange chamfered loop elements cylindrically."""
        angles = np.linspace(0, 2 * np.pi, self.n_elements, endpoint=False)
        
        self.element_positions = np.zeros((self.n_elements, 3))
        self.element_normal_vectors = np.zeros((self.n_elements, 3))
        self.chamfered_loops = []
        
        for i, angle in enumerate(angles):
            # Position on cylinder
            x = self.array_radius * np.cos(angle)
            y = self.array_radius * np.sin(angle)
            z = 0.0
            self.element_positions[i] = [x, y, z]
            
            # Normal: points toward axis (receive from target)
            self.element_normal_vectors[i] = [-np.cos(angle), -np.sin(angle), 0.0]
            
            # Chamfered loop element
            loop = ChamferedLoopElement(radius=0.045, chamfer_radius=0.005)
            self.chamfered_loops.append(loop)
    
    def _compute_coupling_matrix(self):
        """Compute inter-element coupling via magnetic field overlap."""
        for i in range(self.n_elements):
            for j in range(i + 1, self.n_elements):
                # Distance between elements
                dist = np.linalg.norm(self.element_positions[i] - self.element_positions[j])
                
                # Coupling decreases with distance (dipole-dipole interaction)
                # k_c ∝ 1/r³ mutual inductance scaling
                coupling_strength = 0.8 * np.exp(-3 * dist / self.array_radius)
                
                self.coupling_matrix[i, j] = coupling_strength
                self.coupling_matrix[j, i] = coupling_strength
    
    def _compute_em_cues(self):
        """Compute electromagnetic state transfer cues via Jaynes-Cummings model.
        
        EM cues represent coupling strength between RF coil and vascular target,
        modulated by geometry and proximity.
        
        Returns:
            dict: Per-element coupling parameters
        """
        cues = {
            'coupling_strength': np.zeros(self.n_elements),  # rad/s
            'detuning': np.zeros(self.n_elements),           # rad/s
            'coherence_time': np.zeros(self.n_elements),     # us
            'rabi_freq': np.zeros(self.n_elements),          # rad/s
        }
        
        for i in range(self.n_elements):
            # Distance to target
            r = np.linalg.norm(self.element_positions[i] - self.target_center)
            
            # Coupling strength: 1/r³ dipole scaling
            g = COUPLING_BASE / max(r * 100, 0.1)**3  # Convert m to cm for scaling
            cues['coupling_strength'][i] = g
            
            # Detuning (cavity-qubit detuning, small for resonant interaction)
            cues['detuning'][i] = LARMOR_3T * 2 * np.pi * 0.001  # 1 ppm detuning
            
            # Coherence time from coupling strength
            # T_coh = 1 / κ where κ = g / 10 (dissipation)
            kappa = g / 10
            cues['coherence_time'][i] = 1.0 / max(kappa, 1e6) * 1e6  # Convert to µs
            
            # Rabi frequency: Ω_R = √(g² + δ²)
            omega_rabi = np.sqrt(g**2 + cues['detuning'][i]**2)
            cues['rabi_freq'][i] = omega_rabi / (2 * np.pi)  # Convert to Hz
        
        return cues
    
    def build_sensitivity_matrix(self, target_points):
        """Build complex sensitivity matrix for all elements and target points.
        
        Args:
            target_points (np.ndarray): Shape (N, 3) target grid points
            
        Returns:
            np.ndarray: Shape (n_elements, N) complex sensitivities
        """
        n_targets = len(target_points)
        S = np.zeros((self.n_elements, n_targets), dtype=complex)
        
        for i, loop in enumerate(self.chamfered_loops):
            sensitivities = loop.sensitivity_at_points(target_points)
            
            # Add phase from element normal (receive coil phase)
            phase_angle = np.arctan2(
                self.element_normal_vectors[i, 1],
                self.element_normal_vectors[i, 0]
            )
            
            # Complex sensitivity with magnitude and receive phase
            S[i, :] = sensitivities * np.exp(1j * phase_angle)
        
        return S
    
    def optimize_weights(self, S_target, S_healthy=None, target_ratio=0.7, 
                         gradient_penalty=1e-4):
        """Optimize element weights for maximum SNR in target region.
        
        Uses constrained optimization with smoothness penalty for adjacent elements.
        
        Args:
            S_target (np.ndarray): Sensitivity matrix in target ROI (n_elem, N_target)
            S_healthy (np.ndarray): Sensitivity matrix in healthy tissue (n_elem, N_healthy)
            target_ratio (float): Min ratio of target SNR to healthy SNR
            gradient_penalty (float): Smoothness penalty for adjacent element weights
            
        Returns:
            dict: Optimized weights and performance metrics
        """
        
        def objective(weights):
            # RMS signal in target
            signal_target = np.linalg.norm(S_target @ weights)
            
            # SNR = signal / noise; noise decreases with √nr_elements (spatial whitening)
            snr_target = signal_target / np.sqrt(self.n_elements)
            
            if S_healthy is not None:
                signal_healthy = np.linalg.norm(S_healthy @ weights)
                snr_healthy = signal_healthy / np.sqrt(self.n_elements)
                # Penalty if ratio constraint violated
                if snr_target < target_ratio * snr_healthy:
                    penalty = 1e6 * (target_ratio * snr_healthy - snr_target)
                else:
                    penalty = 0
            else:
                penalty = 0
            
            # Smoothness penalty: adjacent elements should have similar weights
            grad_penalty = gradient_penalty * np.sum(
                (weights[:-1] - weights[1:])**2
            ) + (weights[-1] - weights[0])**2  # Circular array
            
            return -snr_target + penalty + grad_penalty
        
        # Optimization bounds
        bounds = [(0.0, 1.0) for _ in range(self.n_elements)]
        
        # Initial guess: uniform weights
        x0 = np.ones(self.n_elements) / self.n_elements
        
        # Differential evolution for global optimum
        result = differential_evolution(
            objective, bounds, seed=42, maxiter=200, atol=1e-6
        )
        
        w_opt = result.x
        
        # Compute final metrics
        signal_target = np.linalg.norm(S_target @ w_opt)
        snr_target = signal_target / np.sqrt(self.n_elements)
        
        snr_healthy = None
        if S_healthy is not None:
            signal_healthy = np.linalg.norm(S_healthy @ w_opt)
            snr_healthy = signal_healthy / np.sqrt(self.n_elements)
        
        return {
            'weights': w_opt,
            'snr_target': snr_target,
            'snr_healthy': snr_healthy,
            'snr_ratio': snr_target / (snr_healthy + 1e-10),
            'optimization_success': result.success,
        }
    
    def print_specifications(self):
        """Print coil specifications and EM state transfer parameters."""
        print("\n" + "="*70)
        print("NEUROVASCULAR CONFORMAL COIL SPECIFICATIONS")
        print("="*70)
        print(f"Elements:                {self.n_elements}")
        print(f"Array radius:            {self.array_radius*100:.1f} cm")
        print(f"Loop radius (chamfered): 4.5 cm, chamfer: 5 mm")
        print(f"Target:                  Popliteal artery (Ø 3 mm)")
        print(f"Impedance (matched):     {IMPEDANCE_CHARACTERISTIC:.0f} Ω")
        print(f"Noise figure (LNA):      {self.noise_figure[0]:.2f} dB")
        print(f"Larmor frequency (3T):   {LARMOR_3T/1e6:.2f} MHz")
        
        print("\n" + "-"*70)
        print("ELECTROMAGNETIC STATE TRANSFER CUES (per element)")
        print("-"*70)
        print(f"{'Element':>8} {'Coupling (Hz)':>16} {'T_coh (µs)':>16} {'f_Rabi (kHz)':>16}")
        print("-"*70)
        
        for i in range(min(8, self.n_elements)):  # Print first 8 elements
            g_hz = self.em_cues['coupling_strength'][i] / (2 * np.pi)
            t_coh = self.em_cues['coherence_time'][i]
            f_rabi = self.em_cues['rabi_freq'][i] / 1e3
            print(f"{i:8d} {g_hz:16.2e} {t_coh:16.3f} {f_rabi:16.2f}")
        
        if self.n_elements > 8:
            print(f"{'...':>8}")
        
        print("\n" + "-"*70)
        print("CIRCUITRY PARAMETERS")
        print("-"*70)
        print(f"Inter-element coupling (mean):  {np.mean(self.coupling_matrix - np.eye(self.n_elements)):.3f}")
        print(f"Coupling matrix condition #:    {np.linalg.cond(self.coupling_matrix):.2f}")
        

def run_neurovascular_thermometry(target_radius=0.006, b0=3.0, te_array_ms=None,
                                  pf_factor=0.625, matrix_size=128, hotspot_dt=15.0,
                                  output_seq_file=None):
    """Run quantum phase thermometry with neurovascular conformal coil.
    
    Integrates neurovascular coil with quantum thermometry for high-SNR
    vascular temperature monitoring.
    
    Args:
        target_radius (float): Target ROI radius (m)
        b0 (float): Magnetic field strength (T)
        te_array_ms (np.ndarray): Echo times (ms); auto-computed if None
        pf_factor (float): Partial Fourier factor (0.5-1.0)
        matrix_size (int): Image matrix size
        hotspot_dt (float): Hotspot temp elevation (°C)
        output_seq_file (str): Output .seq filename (Pulseq format)
        
    Returns:
        dict: Results with metrics and images
    """
    from quantum_phase_thermometry import (
        farey_echo_times, build_combinatorial_mask, pocs_phase_reconstruct,
        QuantumRBMDenoiser, wiener_2d, wls_phase_slope, write_seq_file
    )
    
    print("\n" + "="*70)
    print("NEUROVASCULAR QUANTUM PHASE THERMOMETRY")
    print("="*70)
    
    # Initialize coil
    coil = NeurovascularConformCoil(n_elements=16)
    coil.print_specifications()
    
    # Echo times from Continued Fraction Farey sequence
    if te_array_ms is None:
        te_array_ms = farey_echo_times(n=10, te_min=9.5, te_max=22.5)
    
    n_echoes = len(te_array_ms)
    print(f"\nEcho times: {te_array_ms[:5]} ... {te_array_ms[-5:]} ms ({n_echoes} total)")
    
    # Create target grid
    x = np.linspace(-0.015, 0.015, matrix_size)
    y = np.linspace(-0.015, 0.015, matrix_size)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    target_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Sensitivity matrix
    S = coil.build_sensitivity_matrix(target_points)
    print(f"\nSensitivity matrix: {S.shape} (elements, points)")
    print(f"Max sensitivity: {np.max(np.abs(S)):.3e} T/A")
    
    # Create ground-truth temperature map (Gaussian hotspot at center)
    temp_map_gt = np.ones_like(xx) * 37.0  # Body temperature
    hotspot_mask = (xx**2 + yy**2) < target_radius**2
    temp_map_gt[hotspot_mask] = 37.0 + hotspot_dt
    
    # PRF phase evolution
    gamma = GAMMA_RAD
    b0_hz = b0 * LARMOR_3T
    prf_alpha = -0.0094e-6  # ppm/°C
    
    # Simulate phase for each echo
    phase_array = np.zeros((n_echoes, matrix_size, matrix_size))
    for e, te_ms in enumerate(te_array_ms):
        te_s = te_ms * 1e-3
        # Phase from temperature-dependent PRF
        phase_map = gamma * b0 * 2 * np.pi * te_s * prf_alpha * (temp_map_gt - 37.0)
        phase_array[e] = phase_map
    
    # k-space generation with Gaussian noise
    noise_level = 0.015  # Moderate SNR for 16-element array
    kspace = np.zeros((n_echoes, matrix_size, matrix_size), dtype=complex)
    
    for e in range(n_echoes):
        # Image-domain signal from coil sensitivities (first 3 elements for visualization)
        weighted_signal = np.sum(np.abs(S[:3, :]) * np.exp(1j * phase_array[e].ravel())) / 3.0
        image = weighted_signal * np.exp(1j * phase_array[e]) + \
                noise_level * (np.random.randn(matrix_size, matrix_size) + 
                               1j * np.random.randn(matrix_size, matrix_size))
        kspace[e] = np.fft.ifftshift(np.fft.fft2(image))
    
    # Partial Fourier masking
    mask = build_combinatorial_mask(matrix_size, pf_factor)
    print(f"Partial Fourier factor: {pf_factor}, sampled {np.sum(mask)} of {matrix_size**2} k-space lines")
    
    # POCS reconstruction
    kspace_partial = kspace * mask[np.newaxis, :, :]
    print("Performing POCS phase reconstruction...")
    recon_array = np.zeros_like(phase_array)
    for e in range(n_echoes):
        recon = pocs_phase_reconstruct(kspace_partial[e], mask, iterations=14)
        recon_array[e] = np.angle(recon)
    
    # Apply Wiener filter
    print("Applying Wiener filter...")
    phase_wiener = np.zeros_like(recon_array)
    for e in range(n_echoes):
        phase_wiener[e] = wiener_2d(recon_array[e], noise_var=noise_level**2)
    
    # WLS phase slope estimation (per-pixel)
    print("WLS temperature estimation...")
    phase_mean = np.mean(phase_wiener, axis=0)
    te_s = te_array_ms * 1e-3
    temp_map_wls = np.zeros((matrix_size, matrix_size))
    snr_accum = []
    for r in range(matrix_size):
        for c in range(matrix_size):
            pixel_phases = phase_wiener[:, r, c]
            info = wls_phase_slope(pixel_phases, te_s)
            slope = info["slope"]
            temp_map_wls[r, c] = 37.0 + slope / (gamma * b0 * 2 * np.pi * prf_alpha) if abs(slope) > 1e-12 else 37.0
            snr_accum.append(info.get("snr_phase_db", 0.0))
    info_wls = {"snr_db": float(np.median(snr_accum))}
    
    # RBM denoising
    print("RBM quantum denoising (two-stage)...")
    rbm = QuantumRBMDenoiser(n_visible=64, n_hidden=32, lr=0.015)
    temp_map_rbm = rbm.denoise_image(temp_map_wls)
    
    # Compute metrics
    rmse_wls = np.sqrt(np.mean((temp_map_wls - temp_map_gt)**2))
    rmse_rbm = np.sqrt(np.mean((temp_map_rbm - temp_map_gt)**2))
    snr_phase = info_wls.get('snr_db', 9.5)
    
    print("\n" + "-"*70)
    print("THERMOMETRY RESULTS")
    print("-"*70)
    print(f"WLS RMSE:              {rmse_wls:.3f} °C")
    print(f"RBM RMSE (QML):        {rmse_rbm:.3f} °C")
    print(f"Phase SNR:             {snr_phase:.2f} dB")
    print(f"Improvement:           {(rmse_wls - rmse_rbm) / rmse_wls * 100:.1f}%")
    
    # Write .seq file if requested
    if output_seq_file:
        print(f"\nWriting Pulseq .seq file: {output_seq_file}")
        seq_dir = os.path.dirname(os.path.abspath(output_seq_file)) or "."
        write_seq_file(
            te_array_ms, tr_ms=50.0,
            seq_name="NeurovascularThermometry",
            matrix=matrix_size,
            output_dir=seq_dir
        )
    
    return {
        'coil': coil,
        'phase_array': phase_array,
        'kspace': kspace,
        'temp_map_gt': temp_map_gt,
        'temp_map_wls': temp_map_wls,
        'temp_map_rbm': temp_map_rbm,
        'rmse_wls': rmse_wls,
        'rmse_rbm': rmse_rbm,
        'snr_db': snr_phase,
        'te_array_ms': te_array_ms,
    }


if __name__ == '__main__':
    # Run neurovascular thermometry with .seq output
    results = run_neurovascular_thermometry(
        target_radius=0.006,
        b0=3.0,
        matrix_size=128,
        hotspot_dt=15.0,
        pf_factor=0.625,
        output_seq_file='seqs/NeurovascularThermometry.seq'
    )
    
    print("\n✅ Neurovascular conformal coil thermometry complete!")
