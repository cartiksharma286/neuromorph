"""
Quantum Geometry & Continued Fraction Pulse Sequences
=====================================================

Advanced fMRI sequences utilizing the Fubini-Study metric tensor of the 
spin Hilbert space and continued fraction expansions for pulse timing.
"""

import numpy as np
from statistical_adaptive_pulse import StatisticalAdaptivePulseSequence

class QuantumGeometryContinuedFractionSequence(StatisticalAdaptivePulseSequence):
    """
    Pulse sequence that optimizes TR/TE using Continued Fraction expansions
    of the Golden Ratio (φ) and modulates signal using the Quantum Metric Tensor.
    """
    
    def __init__(self, nvqlink_enabled=False):
        super().__init__(nvqlink_enabled)
        self.sequence_name = "Quantum Geometry (Continued Fraction)"
        
    def continued_fraction_tr(self, depth=5):
        """
        Calculates optimal TR using a continued fraction expansion 
        related to the Golden Ratio for minimized periodicity artifacts.
        
        TR = TR_base * [1; 1, 1, 1, ...] truncated at depth
        """
        def golden_cf(d):
            if d == 0: return 1
            return 1 + 1.0 / golden_cf(d - 1)
        
        phi_approx = golden_cf(depth)
        # Scale to realistic TR range (e.g., 2000ms base)
        return 1000 * phi_approx

    def quantum_metric_modulation(self, kspace_loc):
        """
        Modulates signal based on the Fubini-Study metric tensor g_μν.
        ds² = g_μν dξ^μ dξ^ν
        
        For a spin-1/2 system, g is proportional to the variance of the 
        generators of the transformation.
        """
        # Simulated metric-based modulation
        # In actual physics, this would depend on the adiabaticity of the pulse
        k_norm = np.linalg.norm(kspace_loc)
        g_factor = 1.0 / (1.0 + 0.1 * k_norm**2) # Metric-induced suppression
        return g_factor

    def generate_sequence(self, tissue_stats):
        """Generates the quantum-optimized sequence parameters."""
        # Use statistical results to bias the continued fraction depth
        noise_floor = tissue_stats.get('std_intensity', 0.1)
        cf_depth = int(np.clip(5 + 10 * noise_floor, 3, 10))
        
        opt_tr = self.continued_fraction_tr(depth=cf_depth)
        
        # Optimize TE based on T2* decay modulated by quantum geometry and inflection point analysis
        # Find the inflection point of the signal decay curve: S(t) = S0 * exp(-t/T2*)
        # The first derivative is S'(t) = -S0/T2* * exp(-t/T2*)
        # The second derivative is S''(t) = S0/(T2*)^2 * exp(-t/T2*)
        # To maximize SNR efficiency (SNR / sqrt(TE)), the optimal TE is frequently 
        # around the T2* value. We apply an inflection-point inspired offset to the standard TE.
        
        t2_gm = 110 # ms
        geometric_phase = 0.15 # Simulated Berry Phase contribution
        
        # Inflection point reasoning: 
        # For a decaying exponential, SNR efficiency peaks at TE = T2*.
        # We modulate this based on the variance (noise_floor) to push the TE 
        # slightly earlier (more signal) if noise is high, or later (more contrast) if noise is low.
        inflection_offset = (0.5 - noise_floor) * 20 # +/- 10ms based on noise
        
        opt_te = (t2_gm + inflection_offset) / (1.0 + geometric_phase)
        
        # Ensure TE remains within reasonable bounds
        opt_te = np.clip(opt_te, 10, 150)
        
        return {
            'sequence': 'QuantumGeometry',
            'tr': float(opt_tr),
            'te': float(opt_te),
            'cf_depth': cf_depth,
            'geometric_modulation': True,
            'inflection_optimized': True,
            'description': f"Quantum Geometry (CF={cf_depth}, Inflection-Opt TE)",
            'nvqlink_accelerated': self.nvqlink_enabled
        }

    def compute_geometric_analytics(self, recon_image):
        """
        Computes the 'Curvature Map' of the reconstructed image 
        based on the quantum geometric tensor.
        """
        # Finite difference gradient for curvature approximation
        dy, dx = np.gradient(recon_image)
        d2y, dxy = np.gradient(dy)
        dyx, d2x = np.gradient(dx)
        
        # Scalar curvature approximation K
        curvature = np.abs(d2x * d2y - dxy * dyx) / (1 + dx**2 + dy**2)**(1.5)
        
        return {
            'mean_curvature': float(np.mean(curvature)),
            'max_curvature': float(np.max(curvature)),
            'metric_tensor_norm': float(np.linalg.norm([dx, dy])),
            'manifold_status': 'Statistical Continuable'
        }
