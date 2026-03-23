#!/usr/bin/env python3
"""Quick smoke-test for quantum_thermometry_enhanced module."""

import numpy as np
from quantum_thermometry_enhanced import (
    lookup_prf_sensitivity, TISSUE_LUT, B0_LUT,
    fit_signal_distributions, compute_fisher_information_temperature,
    MultimodalThermometryReasoner, fft2c_enhanced, ifft2c_enhanced,
    generate_thermometry_pulse_sequence, _SCIPY_FFT
)

# 1. LUT
print("=== LUT Test ===")
lut = lookup_prf_sensitivity("gray_matter", 3.0)
print(f"  Gray matter @ 3T: dPhi/dT = {lut['dPhi_per_dT_deg']:.4f} deg/C, opt TE = {lut['optimal_TE_ms']}ms")

# 2. Distributions
print("\n=== Distribution Test ===")
signal = np.random.rayleigh(scale=0.5, size=(64, 64))
dist = fit_signal_distributions(signal)
print(f"  Best model: {dist['best_model']}, KS = {dist['best_ks_stat']:.4f}")

# 3. Fisher information
print("\n=== Fisher Test ===")
te_s = np.array([8, 12, 16, 20, 24]) * 1e-3
fisher = compute_fisher_information_temperature(te_s, 3.0, 30.0)
print(f"  Fisher I = {fisher['fisher_information']:.6e}, min dT = {fisher['min_detectable_dT_C']:.4f} C")

# 4. FFT round-trip
backend = "scipy.fft" if _SCIPY_FFT else "numpy.fft"
print(f"\n=== FFTW Backend: {backend} ===")
test = np.random.randn(64, 64) + 1j * np.random.randn(64, 64)
result = ifft2c_enhanced(fft2c_enhanced(test, apodise="none"))
err = np.mean(np.abs(result - test))
print(f"  Round-trip error: {err:.2e}")

# 5. Sequence generation
print("\n=== Sequence Generation ===")
seq = generate_thermometry_pulse_sequence("epi_thermometry", 7.0)
print(f"  Generated: {seq['seq_name']} at {seq['seq_path']}")

# 6. Multimodal reasoner
print("\n=== Multimodal Reasoner ===")
reasoner = MultimodalThermometryReasoner(b0=3.0)
phase_map = np.random.randn(64, 64) * 0.3
magnitude = np.abs(np.random.randn(64, 64)) * 100
result_mm = reasoner.run_multimodal_reasoning(magnitude, phase_map, te_s=0.012)
print(f"  Temperature range: {result_mm['fused_temperature'].min():.2f} to {result_mm['fused_temperature'].max():.2f} C")
print(f"  Tissue classes: {np.unique(result_mm['tissue_label']).tolist()}")
print(f"  Uncertainty mean: {np.nanmean(result_mm['uncertainty_map']):.4f}")

print("\n*** All tests passed! ***")
