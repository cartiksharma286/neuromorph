"""
Example: Hybrid Quantum-Adaptive Optimization

Demonstrates the complete hybrid optimization workflow combining:
- Quantum ML parameter optimization (CUDA-Q VQE)
- Adaptive learning (PPO reinforcement learning)
- MRI pulse sequence generation (GRE)

This example shows how quantum computing and classical RL work together
to find optimal sequence parameters.
"""

import numpy as np
import sys
sys.path.append('..')

from quantum_optimizer import CUDAQOptimizer
from pulse_sequences import GradientEcho
from adaptive_learning import SequenceOptimizer


def main():
    print("=" * 70)
    print("Hybrid Quantum-Adaptive Sequence Optimization")
    print("=" * 70)
    print()
    
    # Define initial GRE sequence parameters
    initial_params = {
        'TE': 10.0,  # ms
        'TR': 500.0,  # ms
        'FA': 30.0,  # degrees
        'slice_thickness': 3.0,  # mm
        'fov': 256.0,  # mm
        'matrix_size': 256
    }
    
    # Define optimization targets
    target_metrics = {
        'SNR': 100.0,  # Target SNR
        'scan_time': 120.0,  # Maximum scan time (s)
        'resolution': 1.0  # Target resolution (mm)
    }
    
    # Define valid parameter ranges
    parameter_ranges = {
        'TE': (5.0, 50.0),  # ms
        'TR': (100.0, 2000.0),  # ms
        'FA': (10.0, 90.0),  # degrees
        'slice_thickness': (1.0, 10.0),  # mm
        'fov': (128.0, 512.0),  # mm
        'matrix_size': (64, 512)
    }
    
    print("Step 1: Initial Sequence Configuration")
    print("-" * 70)
    print("Initial parameters:")
    for key, val in initial_params.items():
        print(f"  {key}: {val}")
    print()
    print("Target metrics:")
    for key, val in target_metrics.items():
        print(f"  {key}: {val}")
    print()
    
    # Create initial sequence
    print("Step 2: Generate Baseline Sequence")
    print("-" * 70)
    gre_baseline = GradientEcho()
    gre_baseline.generate(initial_params)
    k_space_baseline = gre_baseline.calculate_k_space()
    print(f"  Baseline k-space points: {k_space_baseline.shape[0]}")
    print()
    
    # Hybrid optimization
    print("Step 3: Hybrid Quantum-Adaptive Optimization")
    print("-" * 70)
    optimizer = SequenceOptimizer(
        use_quantum=True,  # Enable quantum ML
        use_adaptive=True   # Enable adaptive learning
    )
    
    optimized_params = optimizer.optimize(
        initial_params=initial_params,
        target_metrics=target_metrics,
        parameter_ranges=parameter_ranges,
        optimization_steps=100
    )
    print()
    
    # Generate optimized sequence
    print("Step 4: Generate Optimized Sequence")
    print("-" * 70)
    gre_optimized = GradientEcho()
    gre_optimized.generate(optimized_params)
    k_space_optimized = gre_optimized.calculate_k_space()
    print(f"  Optimized k-space points: {k_space_optimized.shape[0]}")
    print()
    
    # Compare results
    print("Step 5: Optimization Results")
    print("-" * 70)
    print("Parameter changes:")
    for key in initial_params.keys():
        if key in optimized_params:
            initial = initial_params[key]
            optimized = optimized_params[key]
            change = ((optimized - initial) / initial) * 100
            print(f"  {key:15s}: {initial:8.2f} → {optimized:8.2f} ({change:+6.1f}%)")
    print()
    
    # Calculate metrics improvement
    def calculate_snr(params):
        te = params['TE']
        tr = params['TR']
        fa = params['FA']
        return (1000 / te) * np.sin(np.radians(fa)) * (tr / 1000)
    
    snr_baseline = calculate_snr(initial_params)
    snr_optimized = calculate_snr(optimized_params)
    snr_improvement = ((snr_optimized - snr_baseline) / snr_baseline) * 100
    
    print("Performance Improvement:")
    print(f"  Baseline SNR:  {snr_baseline:.2f}")
    print(f"  Optimized SNR: {snr_optimized:.2f}")
    print(f"  Improvement:   {snr_improvement:+.1f}%")
    print()
    
    # Export sequences
    print("Step 6: Export Sequences")
    print("-" * 70)
    gre_baseline.export_pypulseq("gre_baseline.seq")
    gre_optimized.export_pypulseq("gre_optimized.seq")
    print()
    
    print("=" * 70)
    print("Hybrid Optimization Complete!")
    print("=" * 70)
    print()
    print("Optimization Methods Used:")
    print("  ✓ Quantum ML (CUDA-Q VQE) - Parameter space exploration")
    print("  ✓ Adaptive Learning (PPO) - Policy-based refinement")
    print("  ✓ Hybrid Approach - Best of quantum and classical")
    print()
    print("Key Benefits:")
    print(f"  • SNR improved by {snr_improvement:+.1f}%")
    print("  • Automated parameter optimization")
    print("  • Quantum-enhanced parameter space search")
    print("  • Real-time adaptive learning capability")
    print()


if __name__ == "__main__":
    main()
