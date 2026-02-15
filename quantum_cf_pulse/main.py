import numpy as np
import json
import os
import sys

# Add the parent directory to sys.path to allow imports from quantum_cf_pulse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_cf_pulse.optimizer import VariationalPulseOptimizer, FermatPulseOptimizer
from quantum_cf_pulse.solver import RiccatiSolver

def write_seq_file(filename, time, omega_x, omega_y):
    """
    Writes a basic Pulseq .seq file.
    This is a simplified writer that treats the pulse as a series of RF blocks.
    Real Pulseq files are more complex, but this provides a compatible structure.
    """
    with open(filename, "w") as f:
        f.write("# Pulseq Sequence File\n")
        f.write("# Created by Quantum CF Pulse Generator\n")
        f.write("# Format: Pulseq\n\n")
        
        f.write("[VERSION]\n")
        f.write("major 1\n")
        f.write("minor 4\n\n")
        
        f.write("[DEFINITIONS]\n")
        f.write("FOV 256 256 1\n\n")
        
        # We will define arbitrary shapes for now as we are just exporting the RF waveform
        f.write("[SHAPES]\n")
        f.write(f"shape_id 1\n")
        f.write(f"num_samples {len(omega_x)}\n")
        # Interleave real (x) and imag (y) for complex RF
        # Normalize to max 1 for shape definition
        max_amp = np.max(np.sqrt(np.array(omega_x)**2 + np.array(omega_y)**2))
        if max_amp == 0: max_amp = 1.0
        
        for ox, oy in zip(omega_x, omega_y):
            # Pulseq shapes are typically normalized
            val_x = ox / max_amp
            val_y = oy / max_amp
            # Compressed format is usually used, but we write uncompressed for simplicity
            f.write(f"{val_x:.6f} {val_y:.6f}\n")
        f.write("\n")

def main():
    # Parameters
    T_pulse = 1.0  # Total pulse duration
    n_steps = 100
    dt = T_pulse / n_steps
    
    # Target state: |1> (spin down, pi pulse)
    # |0> = [1, 0], |1> = [0, 1]
    target_state = np.array([0, 1], dtype=complex)
    
    print("Initializing Fermat Pulse Optimizer (Ricci States)...")
    optimizer = FermatPulseOptimizer(n_steps, dt, target_state)
    
    print("Starting optimization with Fermat Primes and Ricci Curvature Regularization...")
    omega_x, omega_y, final_cost = optimizer.optimize()
    
    print(f"Optimization complete. Final Cost (Infidelity + Ricci): {final_cost:.6f}")
    
    # Compute final trajectory for visualization
    solver = RiccatiSolver(dt)
    initial_state = np.array([1, 0], dtype=complex)
    trajectory = solver.evolve(initial_state, omega_x, omega_y, delta=0.0)
    final_state = trajectory[-1]
    fidelity = np.abs(np.vdot(target_state, final_state))**2
    print(f"Final Fidelity: {fidelity:.6f}")
    
    bloch_trajectory = [solver.state_to_bloch(state).tolist() for state in trajectory]
    
    # Export data
    output_data = {
        "time": np.linspace(0, T_pulse, n_steps + 1).tolist(),
        "omega_x": omega_x.tolist(),
        "omega_y": omega_y.tolist(),
        "bloch_trajectory": bloch_trajectory,
        "fidelity": fidelity,
        "fermat_primes": optimizer.fermat_primes
    }
    
    os.makedirs("quantum_cf_pulse/viz", exist_ok=True)
    with open("quantum_cf_pulse/viz/pulse_data.json", "w") as f:
        json.dump(output_data, f, indent=2)
        
    print("Data exported to quantum_cf_pulse/viz/pulse_data.json")

    # Export to CSV for external use
    csv_file = "quantum_cf_pulse/pulse_sequence.csv"
    with open(csv_file, "w") as f:
        f.write("time,omega_x,omega_y\n")
        for i in range(n_steps):
            f.write(f"{output_data['time'][i]},{omega_x[i]},{omega_y[i]}\n")
    print(f"Pulse sequence exported to {csv_file}")
    
    # Export to .seq file
    seq_file = "quantum_cf_pulse/pulse_sequence.seq"
    write_seq_file(seq_file, output_data['time'], omega_x, omega_y)
    print(f"Pulse sequence exported to {seq_file}")

if __name__ == "__main__":
    main()
