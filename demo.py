"""
Demo script for RF Coil Designer with Generative AI

Demonstrates:
1. Parametric coil design
2. Generative design for target frequency
3. Evolutionary optimization
4. Visualization and schematics
"""

import numpy as np
import matplotlib.pyplot as plt
from coil_designer import (RFCoilDesigner, GenerativeCoilDesigner, 
                           CoilParameters)
from coil_visualizer import CoilVisualizer


def demo_parametric_design():
    """Demonstrate parametric coil design and analysis."""
    print("="*70)
    print("Demo 1: Parametric RF Coil Design")
    print("="*70)
    
    designer = RFCoilDesigner()
    
    # Create a solenoid coil
    params = CoilParameters(
        coil_type='solenoid',
        wire_diameter=1.0,  # mm
        turns=15,
        diameter=30,  # mm
        length=40  # mm
    )
    
    # Calculate properties
    L = designer.calculate_inductance(params)
    C = 100e-12  # 100 pF
    f_res = designer.calculate_resonant_frequency(L, C)
    Q = designer.calculate_quality_factor(params, f_res)
    
    print(f"\nSolenoid Coil Design:")
    print(f"  Turns: {params.turns}")
    print(f"  Diameter: {params.diameter} mm")
    print(f"  Length: {params.length} mm")
    print(f"  Wire diameter: {params.wire_diameter} mm")
    print(f"\nElectrical Properties:")
    print(f"  Inductance: {L*1e6:.2f} µH")
    print(f"  Resonant frequency (with {C*1e12:.0f} pF): {f_res/1e6:.2f} MHz")
    print(f"  Quality factor: {Q:.1f}")
    
    # Visualize
    visualizer = CoilVisualizer(designer)
    visualizer.plot_coil_geometry(params, 
        save_path='C:/Users/User/.gemini/antigravity/scratch/rf_coil_designer/solenoid_geometry.png')
    print("\nVisualization saved: solenoid_geometry.png")
    
    # Generate schematic
    visualizer.generate_schematic(params, capacitance=C,
        save_path='C:/Users/User/.gemini/antigravity/scratch/rf_coil_designer/solenoid_schematic.png')
    print("Schematic saved: solenoid_schematic.png")


def demo_design_for_frequency():
    """Demonstrate generative design for target frequency."""
    print("\n" + "="*70)
    print("Demo 2: Generative Design for Target Frequency")
    print("="*70)
    
    designer = RFCoilDesigner()
    
    # Target: 13.56 MHz (ISM band)
    target_freq = 13.56e6
    
    print(f"\nTarget frequency: {target_freq/1e6:.2f} MHz (ISM band)")
    print("Generating optimized coil design...")
    
    constraints = {
        'max_diameter': 50,  # mm
        'min_diameter': 20,
        'max_turns': 30,
        'min_turns': 5,
        'wire_diameter': 1.0,
        'capacitance': 150e-12  # 150 pF
    }
    
    params = designer.design_for_frequency(target_freq, 
                                          coil_type='solenoid',
                                          constraints=constraints)
    
    # Verify design
    L = designer.calculate_inductance(params)
    C = constraints['capacitance']
    f_achieved = designer.calculate_resonant_frequency(L, C)
    Q = designer.calculate_quality_factor(params, f_achieved)
    
    print(f"\nOptimized Design:")
    print(f"  Turns: {params.turns}")
    print(f"  Diameter: {params.diameter:.2f} mm")
    print(f"  Length: {params.length:.2f} mm")
    print(f"\nPerformance:")
    print(f"  Inductance: {L*1e6:.2f} µH")
    print(f"  Achieved frequency: {f_achieved/1e6:.3f} MHz")
    print(f"  Error: {abs(f_achieved - target_freq)/target_freq * 100:.3f}%")
    print(f"  Quality factor: {Q:.1f}")
    
    # Visualize
    visualizer = CoilVisualizer(designer)
    visualizer.generate_schematic(params, capacitance=C,
        save_path='C:/Users/User/.gemini/antigravity/scratch/rf_coil_designer/optimized_13MHz_schematic.png')
    print("\nSchematic saved: optimized_13MHz_schematic.png")


def demo_evolutionary_design():
    """Demonstrate evolutionary coil design."""
    print("\n" + "="*70)
    print("Demo 3: Evolutionary RF Coil Design")
    print("="*70)
    
    designer = RFCoilDesigner()
    gen_designer = GenerativeCoilDesigner(designer)
    
    # Target: 27.12 MHz (another ISM band)
    target_freq = 27.12e6
    
    print(f"\nTarget frequency: {target_freq/1e6:.2f} MHz")
    print("Evolving design using genetic algorithm...")
    print("Population size: 50, Generations: 20")
    
    constraints = {
        'max_diameter': 40,
        'min_diameter': 15,
        'max_turns': 25,
        'min_turns': 5,
        'max_length': 30,
        'wire_diameter': 0.8,
        'capacitance': 100e-12
    }
    
    # Evolve design
    evolved_params = gen_designer.evolve_design(
        target_freq,
        coil_type='solenoid',
        constraints=constraints,
        population_size=50,
        generations=20
    )
    
    # Evaluate
    L = designer.calculate_inductance(evolved_params)
    C = constraints['capacitance']
    f_achieved = designer.calculate_resonant_frequency(L, C)
    Q = designer.calculate_quality_factor(evolved_params, f_achieved)
    
    print(f"\nEvolved Design:")
    print(f"  Turns: {evolved_params.turns}")
    print(f"  Diameter: {evolved_params.diameter:.2f} mm")
    print(f"  Length: {evolved_params.length:.2f} mm")
    print(f"  Wire: {evolved_params.wire_diameter:.2f} mm")
    print(f"\nPerformance:")
    print(f"  Inductance: {L*1e6:.2f} µH")
    print(f"  Achieved frequency: {f_achieved/1e6:.3f} MHz")
    print(f"  Error: {abs(f_achieved - target_freq)/target_freq * 100:.3f}%")
    print(f"  Quality factor: {Q:.1f}")
    
    # Generate variations
    print("\nGenerating 3 design variations...")
    variations = gen_designer.generate_design_variations(evolved_params, 3)
    
    for i, var in enumerate(variations, 1):
        L_var = designer.calculate_inductance(var)
        f_var = designer.calculate_resonant_frequency(L_var, C)
        print(f"  Variation {i}: {var.turns} turns, "
              f"{var.diameter:.1f} mm dia, {f_var/1e6:.2f} MHz")


def demo_planar_spiral():
    """Demonstrate planar spiral coil design."""
    print("\n" + "="*70)
    print("Demo 4: Planar Spiral Coil for PCB")
    print("="*70)
    
    designer = RFCoilDesigner()
    visualizer = CoilVisualizer(designer)
    
    # Planar spiral for NFC-like application
    params = CoilParameters(
        coil_type='planar_spiral',
        wire_diameter=0.3,  # PCB trace width
        turns=8,
        diameter=40,  # mm
        length=1.5,  # mm spacing between turns
        substrate_thickness=1.6  # Standard PCB thickness
    )
    
    L = designer.calculate_inductance(params)
    C = 220e-12  # 220 pF
    f_res = designer.calculate_resonant_frequency(L, C)
    
    print(f"\nPlanar Spiral Coil (PCB):")
    print(f"  Turns: {params.turns}")
    print(f"  Outer diameter: {params.diameter} mm")
    print(f"  Track spacing: {params.length} mm")
    print(f"  Track width: {params.wire_diameter} mm")
    print(f"\nElectrical Properties:")
    print(f"  Inductance: {L*1e6:.2f} µH")
    print(f"  Resonant frequency (with {C*1e12:.0f} pF): {f_res/1e6:.2f} MHz")
    
    # Visualize
    visualizer.plot_coil_geometry(params,
        save_path='C:/Users/User/.gemini/antigravity/scratch/rf_coil_designer/planar_spiral_geometry.png')
    print("\nVisualization saved: planar_spiral_geometry.png")
    
    visualizer.generate_schematic(params, capacitance=C,
        save_path='C:/Users/User/.gemini/antigravity/scratch/rf_coil_designer/planar_spiral_schematic.png')
    print("Schematic saved: planar_spiral_schematic.png")


def demo_comparison():
    """Compare different coil designs for same target frequency."""
    print("\n" + "="*70)
    print("Demo 5: Comparing Coil Types")
    print("="*70)
    
    designer = RFCoilDesigner()
    target_freq = 13.56e6
    C = 100e-12
    
    print(f"\nTarget: {target_freq/1e6:.2f} MHz with {C*1e12:.0f} pF capacitor")
    print("\nComparing different coil designs:\n")
    
    # Solenoid
    solenoid = CoilParameters('solenoid', 1.0, 12, 25, 30)
    L_sol = designer.calculate_inductance(solenoid)
    f_sol = designer.calculate_resonant_frequency(L_sol, C)
    Q_sol = designer.calculate_quality_factor(solenoid, f_sol)
    
    print(f"1. Solenoid:")
    print(f"   {solenoid.turns} turns, {solenoid.diameter}mm dia, {solenoid.length}mm length")
    print(f"   L={L_sol*1e6:.2f}µH, f={f_sol/1e6:.2f}MHz, Q={Q_sol:.1f}")
    
    # Planar spiral
    spiral = CoilParameters('planar_spiral', 0.5, 10, 35, 1.2)
    L_spiral = designer.calculate_inductance(spiral)
    f_spiral = designer.calculate_resonant_frequency(L_spiral, C)
    Q_spiral = designer.calculate_quality_factor(spiral, f_spiral)
    
    print(f"\n2. Planar Spiral:")
    print(f"   {spiral.turns} turns, {spiral.diameter}mm dia, {spiral.length}mm spacing")
    print(f"   L={L_spiral*1e6:.2f}µH, f={f_spiral/1e6:.2f}MHz, Q={Q_spiral:.1f}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    visualizer = CoilVisualizer(designer)
    visualizer.plot_2d_solenoid(solenoid, axes[0])
    axes[0].set_title(f'Solenoid: {f_sol/1e6:.2f} MHz, Q={Q_sol:.1f}', 
                     fontsize=12, fontweight='bold')
    
    visualizer.plot_2d_planar_spiral(spiral, axes[1])
    axes[1].set_title(f'Planar: {f_spiral/1e6:.2f} MHz, Q={Q_spiral:.1f}',
                     fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('C:/Users/User/.gemini/antigravity/scratch/rf_coil_designer/coil_comparison.png',
                dpi=300, bbox_inches='tight')
    print("\nComparison plot saved: coil_comparison.png")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("RF COIL DESIGNER WITH GENERATIVE AI")
    print("="*70)
    
    np.random.seed(42)
    
    # Run demos
    demo_parametric_design()
    demo_design_for_frequency()
    demo_evolutionary_design()
    demo_planar_spiral()
    demo_comparison()
    
    print("\n" + "="*70)
    print("All demonstrations completed successfully!")
    print("Generated visualizations and schematics saved to:")
    print("  C:/Users/User/.gemini/antigravity/scratch/rf_coil_designer/")
    print("="*70)


if __name__ == "__main__":
    main()
