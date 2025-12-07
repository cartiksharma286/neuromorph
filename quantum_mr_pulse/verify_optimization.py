
import numpy as np
import sys
import os

# Ensure we can import modules
sys.path.append(os.getcwd())

from pulse_generator import PulseGenerator
from quantum_integrals import QuantumIntegrals
from reconstruction import ReconstructionSimulator

def verify_optimization():
    print("==================================================")
    print("   QUANTUM MR PULSE OPTIMIZATION VERIFICATION")
    print("==================================================")
    print("Verifying against MRI Simulator & Quantum Metrics...")
    
    pg = PulseGenerator()
    qi = QuantumIntegrals()
    recon = ReconstructionSimulator()
    
    # ---------------------------------------------------------
    # TEST 1: Spin Echo Optimization (Coherence/Signal)
    # ---------------------------------------------------------
    print("\n[TEST 1] Spin Echo Optimization (Target TE=40ms, TR=500ms)")
    
    # Unoptimized
    print("  -> Generating Standard Sequence (TE=40ms)...")
    seq_std = pg.generate_se(te_ms=40, tr_ms=500, fov_mm=200, matrix_size=128, optimize=False)
    metrics_std = qi.calculate_surface_integral(seq_std)
    
    # Optimized
    print("  -> Running Quantum Optimization Loop...")
    seq_opt = pg.generate_se(te_ms=40, tr_ms=500, fov_mm=200, matrix_size=128, optimize=True)
    metrics_opt = qi.calculate_surface_integral(seq_opt)
    
    # Extract metadata
    opt_meta = seq_opt.get('optimization_metadata', {})
    opt_te = opt_meta.get('optimal_value', 40)
    
    print(f"  -> RESULTS:")
    print(f"     Standard TE: 40.0 ms | Metric (Coherence): {metrics_std['coherence_metric']:.4f}")
    print(f"     Optimized TE: {opt_te:.1f} ms | Metric (Coherence): {metrics_opt['coherence_metric']:.4f}")
    
    if metrics_opt['coherence_metric'] >= metrics_std['coherence_metric']:
        print("     [PASS] Optimization maintained or improved signal coherence.")
    else:
        print("     [WARN] Optimization yielded lower metric (Check constraints).")

    # ---------------------------------------------------------
    # TEST 2: GRE Optimization (Surface Integral/Berry Phase)
    # ---------------------------------------------------------
    print("\n[TEST 2] GRE Optimization (Target FA=10deg)")
    
    # Unoptimized
    seq_gre_std = pg.generate_gre(te_ms=5, tr_ms=50, flip_angle_deg=10, fov_mm=200, matrix_size=128, optimize=False)
    m_gre_std = qi.calculate_surface_integral(seq_gre_std)
    
    # Optimized
    seq_gre_opt = pg.generate_gre(te_ms=5, tr_ms=50, flip_angle_deg=10, fov_mm=200, matrix_size=128, optimize=True)
    m_gre_opt = qi.calculate_surface_integral(seq_gre_opt)
    
    opt_meta_gre = seq_gre_opt.get('optimization_metadata', {})
    opt_fa = opt_meta_gre.get('optimal_value', 10)
    
    print(f"  -> RESULTS:")
    print(f"     Standard FA: 10.0 deg | Metric (Berry): {m_gre_std['surface_integral']:.4f}")
    print(f"     Optimized FA: {opt_fa:.1f} deg | Metric (Berry): {m_gre_opt['surface_integral']:.4f}")

    if m_gre_opt['surface_integral'] > m_gre_std['surface_integral']:
        print("     [PASS] Optimization improved Quantum Surface Integral.")
    else:
        print("     [WARN] Optimization did not strictly improve metric.")

    # ---------------------------------------------------------
    # TEST 3: Image Reconstruction Verification
    # ---------------------------------------------------------
    print("\n[TEST 3] Simulator Reconstruction (Signal Verification)")
    
    print("  -> Reconstructing Standard SE...")
    res_std = recon.reconstruct(seq_std)
    img_std = np.array(res_std['image'])
    sig_std = np.mean(img_std)
    
    print("  -> Reconstructing Optimized SE...")
    res_opt = recon.reconstruct(seq_opt)
    img_opt = np.array(res_opt['image'])
    sig_opt = np.mean(img_opt)
    
    print(f"     Standard Mean Signal: {sig_std:.4f}")
    print(f"     Optimized Mean Signal: {sig_opt:.4f}")
    
    if sig_opt > sig_std:
         print("     [PASS] Optimized sequence yields higher image signal (SNR).")
    elif sig_opt == sig_std:
         print("     [INFO] Signal equivalent (Constraints likely limited optimization range).")
    else:
         print("     [WARN] Optimized signal is lower.")

    print("\n==================================================")
    print("VERIFICATION COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    verify_optimization()
