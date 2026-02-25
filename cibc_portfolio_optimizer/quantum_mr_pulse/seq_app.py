import argparse
import sys
import os
import json
from pulse_generator import PulseGenerator

def main():
    parser = argparse.ArgumentParser(description="Quantum MR Pulse Sequence Generator")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: generate
    gen_parser = subparsers.add_parser("generate", help="Generate a pulse sequence file (.seq)")
    gen_parser.add_argument("--type", type=str, required=True, choices=["GRE", "SE"], help="Sequence type")
    gen_parser.add_argument("--te", type=float, default=20.0, help="Echo Time (ms)")
    gen_parser.add_argument("--tr", type=float, default=100.0, help="Repetition Time (ms)")
    gen_parser.add_argument("--flip_angle", type=float, default=90.0, help="Flip Angle (degrees) - for GRE")
    gen_parser.add_argument("--optimize", action="store_true", help="Enable Quantum Optimization")
    gen_parser.add_argument("--output", type=str, default="sequence.seq", help="Output filename")

    # Command: list
    list_parser = subparsers.add_parser("list", help="List available sequence templates")

    args = parser.parse_args()

    if args.command == "list":
        print("Available Sequence Templates:")
        print("  - GRE (Gradient Recalled Echo) : Fast imaging, sensitive to field inhomogeneities.")
        print("  - SE  (Spin Echo)             : T2-weighted imaging, robust against inhomogeneities.")
        return

    if args.command == "generate":
        print(f"Initializing Quantum Pulse Generator...")
        try:
            generator = PulseGenerator()
            
            # Common params map
            params = {
                "te_ms": args.te,
                "tr_ms": args.tr,
                "fov_mm": 200,   # Default scalar for CLI
                "matrix_size": 128, # Default scalar
                "optimize": args.optimize
            }

            print(f"Generating {args.type} sequence with TE={args.te}ms, TR={args.tr}ms...")
            
            sequence_data = None
            if args.type == "GRE":
                sequence_data = generator.generate_gre(flip_angle_deg=args.flip_angle, **params)
            elif args.type == "SE":
                sequence_data = generator.generate_se(**params)
            
            if sequence_data:
                # Generate .seq content
                seq_content = generator.export_to_seq(sequence_data, filename=args.output)
                
                # Write to file
                output_path = os.path.abspath(args.output)
                with open(output_path, "w") as f:
                    f.write(seq_content)
                
                print(f"Success! Sequence saved to: {output_path}")
                
                if args.optimize and 'optimization_metadata' in sequence_data:
                    opt = sequence_data['optimization_metadata']
                    print("\nOptimization Results:")
                    print(f"  Method: {opt['method']}")
                    print(f"  Best {opt['optimized_parameter']}: {opt['optimal_value']:.2f}")
                    print(f"  Metric: {opt['metric_value']:.4f}")

            else:
                print("Error: Failed to generate sequence data.")

        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
        return

    # If no arguments, print help
    parser.print_help()

if __name__ == "__main__":
    main()
