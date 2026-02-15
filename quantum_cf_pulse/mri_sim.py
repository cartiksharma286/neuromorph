import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to sys.path to allow imports from quantum_cf_pulse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_cf_pulse.solver import RiccatiSolver

def simulate_mri():
    # Load pulse sequence
    df = pd.read_csv("quantum_cf_pulse/pulse_sequence.csv")
    # time in CSV is the start time of each step (100 points)
    time_steps = df["time"].values
    omega_x = df["omega_x"].values
    omega_y = df["omega_y"].values
    dt = time_steps[1] - time_steps[0]
    
    # Simulation Parameters
    # Tissue properties (e.g., Grey Matter)
    T1 = 1.0  # seconds
    T2 = 0.1  # seconds
    
    # Off-resonance (B0 inhomogeneity)
    deltas = np.linspace(-10, 10, 21) # Hz * 2pi
    
    solver = RiccatiSolver(dt)
    
    results = []
    
    print(f"Simulating MRI response for {len(deltas)} isochromats...")
    
    for delta in deltas:
        # Initial state |0> (equilibrium) -> [0, 0, 1] in Bloch vector
        M = np.array([0.0, 0.0, 1.0])
        trajectory = [M.copy()]
        
        for ox, oy in zip(omega_x, omega_y):
            # Bloch Equation: dM/dt = M x (gamma B) - R M + R M0
            # B = [ox, oy, delta]
            
            # 1. Rotation
            omega = np.array([ox, oy, delta])
            norm_omega = np.linalg.norm(omega)
            
            if norm_omega > 1e-10:
                axis = omega / norm_omega
                angle = norm_omega * dt
                c = np.cos(angle)
                s = np.sin(angle)
                C = 1 - c
                x, y, z = axis
                
                R = np.array([
                    [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
                    [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
                    [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
                ])
                M = R @ M
            
            # 2. Relaxation
            E1 = np.exp(-dt/T1)
            E2 = np.exp(-dt/T2)
            
            M[0] = M[0] * E2
            M[1] = M[1] * E2
            M[2] = M[2] * E1 + (1 - E1) # M0=1
            
            trajectory.append(M.copy())
            
        results.append(trajectory)
    
    results = np.array(results)
    
    # Generate time array for plotting (results includes initial state, so N+1 points)
    plot_time = np.linspace(0, len(omega_x) * dt, len(omega_x) + 1)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for i, delta in enumerate(deltas):
        # Plot Mz for each isochromat
        plt.plot(plot_time, results[i, :, 2], label=f"Off-res={delta:.1f}" if i%5==0 else "")
        
    plt.xlabel("Time (s)")
    plt.ylabel("Mz (Longitudinal Magnetization)")
    plt.title("MRI Simulation: Inversion Recovery / Excitation Profile")
    plt.legend()
    plt.grid(True)
    plt.savefig("quantum_cf_pulse/viz/mri_simulation.png")
    print("Simulation complete. Plot saved to quantum_cf_pulse/viz/mri_simulation.png")

if __name__ == "__main__":
    simulate_mri()
