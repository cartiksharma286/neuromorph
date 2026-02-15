import argparse
import sys
import os
import numpy as np
from pulse_generator import PulseGenerator
from quantum_integrals import QuantumIntegrals

def main():
    print("==================================================")
    print("   QUANTUM PRIME DISTRIBUTION SEQ GENERATOR       ")
    print("==================================================")
    
    parser = argparse.ArgumentParser(description="Generate Optimal Prime Distribution Pulse Sequences")
    parser.add_argument("--duration", type=float, default=50.0, help="Duration of sequence in ms")
    parser.add_argument("--min_scale", type=float, default=10.0, help="Min time scaling factor")
    parser.add_argument("--max_scale", type=float, default=100.0, help="Max time scaling factor")
    parser.add_argument("--steps", type=int, default=20, help="Optimization steps")
    parser.add_argument("--output", type=str, default="prime_optimal.seq", help="Output .seq filename")
    
    args = parser.parse_args()
    
    generator = PulseGenerator()
    integrator = QuantumIntegrals()
    
    print(f"Initializing Optimization Loop...")
    print(f"Range: [{args.min_scale}, {args.max_scale}] with {args.steps} steps.")
    
    best_metric = -np.inf
    best_scale = None
    best_seq = None
    
    scales = np.linspace(args.min_scale, args.max_scale, args.steps)
    
    for i, scale in enumerate(scales):
        # Generate candidate
        seq = generator.generate_prime_weighted_sequence(args.duration, scale)
        
        # Evaluate
        metrics = integrator.calculate_surface_integral(seq)
        metric_val = metrics['surface_integral']
        
        # Display progress bar style
        bar = "#" * int((i+1)/args.steps * 20)
        print(f"\r[{bar:<20}] Scale: {scale:.1f} | Metric: {metric_val:.4f}", end="")
        
        if metric_val > best_metric:
            best_metric = metric_val
            best_scale = scale
            best_seq = seq
            
    print("\n\nOptimization Complete!")
    print(f"Optimal Scaling Factor: {best_scale:.4f}")
    print(f"Maximal Quantum Surface Integral: {best_metric:.4f}")
    
    if best_seq:
        # Add Optimization Metadata
        best_seq['optimization_metadata'] = {
            'optimized_parameter': 'prime_scaling_factor',
            'optimal_value': best_scale,
            'metric_value': best_metric,
            'method': 'Prime Distribution Surface Integral'
        }
        
        # Export
        output_path = os.path.abspath(args.output)
        content = generator.export_to_seq(best_seq, filename=args.output)
        with open(output_path, "w") as f:
            f.write(content)
            
        print(f"\nSequence file written to: {output_path}")
        print("This file contains the optimal pulse train derived from the Prime Geodesic map.")

if __name__ == "__main__":
    main()
