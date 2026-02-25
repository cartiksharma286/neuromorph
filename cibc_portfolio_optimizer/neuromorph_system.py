import argparse
import sys
import os

# Add sub-modules to path
sys.path.append(os.path.join(os.getcwd(), 'pulse_system'))
sys.path.append(os.path.join(os.getcwd(), 'cbt_analysis'))
sys.path.append(os.path.join(os.getcwd(), 'entanglement_analysis'))

# Import components
try:
    from ricci_fermat_pulse import RicciFermatPulseGenerator
    from quantum_traits import TwoPhotonSimulator, QuantumSurfaceIntegrator, CBTTraitGenerator
    from entanglement_signatures import MultiRegionSimulator, EntanglementAnalyzer
    import dhf_validator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you are running this from the root directory.")
    sys.exit(1)

def run_pulse_gen(args):
    print("Generating Ricci-Fermat Pulse...")
    gen = RicciFermatPulseGenerator()
    t, rf = gen.generate_pulse(duration=args.duration*1e-3, flip_angle=args.flip_angle)
    if args.optimize:
        t, rf = gen.optimize_pulse((t, rf))
    
    gen.export_to_seq(t, rf, args.output)
    print(f"Pulse saved to {args.output}")

def run_cbt_analysis(args):
    print("Running CBT Quantum Analysis...")
    sim = TwoPhotonSimulator()
    data = sim.generate_data()
    
    integrator = QuantumSurfaceIntegrator()
    integral = integrator.surface_integral(data)
    print(f"Quantum Surface Integral: {integral}")
    
    gen = CBTTraitGenerator()
    traits = gen.generate_traits(integral)
    print("Derived CBT Traits:", traits)

def run_entanglement_analysis(args):
    print("Running Entanglement Analysis...")
    sim = MultiRegionSimulator()
    data = sim.generate_data(correlation_strength=args.correlation)
    
    analyzer = EntanglementAnalyzer()
    rho = analyzer.compute_density_matrix(data)
    entropy = analyzer.von_neumann_entropy(rho)
    pr = analyzer.participation_ratio(rho)
    
    print(f"Von Neumann Entropy: {entropy}")
    print(f"Participation Ratio: {pr}")

def check_compliance(args):
    print("Checking DHF Compliance...")
    # Load config manually since dhf_validator expects it
    config_path = "iec_13485_compliance/structure.json"
    if not os.path.exists(config_path):
        print("Config not found.")
        return
        
    config = dhf_validator.load_structure(config_path)
    dhf_validator.validate_dhf(config)

def main():
    parser = argparse.ArgumentParser(description="Neuromorph System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Pulse Gen
    pulse_parser = subparsers.add_parser("generate-pulse", help="Generate MRI Pulse")
    pulse_parser.add_argument("--duration", type=float, default=1.0, help="Duration in ms")
    pulse_parser.add_argument("--flip-angle", type=float, default=90.0, help="Flip angle in degrees")
    pulse_parser.add_argument("--optimize", action="store_true", help="Optimize curvature")
    pulse_parser.add_argument("--output", type=str, default="pulse.seq", help="Output file")
    
    # CBT Analysis
    cbt_parser = subparsers.add_parser("analyze-cbt", help="Run CBT Quantum Analysis")
    
    # Entanglement Analysis
    ent_parser = subparsers.add_parser("analyze-entanglement", help="Run Entanglement Analysis")
    ent_parser.add_argument("--correlation", type=float, default=0.8, help="Simulated correlation strength")
    
    # Compliance
    comp_parser = subparsers.add_parser("check-compliance", help="Check IEC 13485 Compliance")
    
    args = parser.parse_args()
    
    if args.command == "generate-pulse":
        run_pulse_gen(args)
    elif args.command == "analyze-cbt":
        run_cbt_analysis(args)
    elif args.command == "analyze-entanglement":
        run_entanglement_analysis(args)
    elif args.command == "check-compliance":
        check_compliance(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
