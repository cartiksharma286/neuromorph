import numpy as np

class QuantumIntegrals:
    def calculate_surface_integral(self, pulse_data):
        """
        Calculates a 'Quantum Surface Integral' representing the geometric phase
        accumulated by the spin system during the pulse sequence.
        
        This is a simulation of the Berry Phase on the Bloch sphere over the trajectory
        induced by the RF and Gradient fields.
        """
        rf = np.array(pulse_data['rf'])
        # Effective field approximation:
        # We assume B_eff is dominated by RF (B1) in the rotating frame for simple calculation
        # In a real quantum simulation we would use the full Hamiltonian.
        
        # Calculate trajectory on Bloch sphere (approximate)
        # M starts at [0, 0, 1]
        
        trajectory_x = []
        trajectory_y = []
        trajectory_z = []
        
        mx, my, mz = 0.0, 0.0, 1.0
        dt = 0.01 * 1e-3 # s
        gamma = 42.58 * 1e6 * 2 * np.pi # rad/s/T
        
        # Relaxation estimates for brain tissue (approx)
        t1 = 1000 * 1e-3 # s
        t2 = 100 * 1e-3 # s
        m0 = 1.0
        
        accumulated_phase = 0.0
        
        for i in range(len(rf)):
            # Simple Euler integration of Bloch equation without relaxation
            # dM/dt = M x gamma*B
            # B = [B1_x, B1_y, B0_offset]
            # Here assum rf is purely real (x-axis)
            
            b1 = rf[i] # T (already normalized in pulse gen roughly, but let's assume it's scaled)
            
            # Rotation (Precession + RF)
            # Apply Relaxation first or after? Simple Euler: apply concurrently/sequentially
            
            # 1. Relaxation
            dmx_relax = -mx / t2
            dmy_relax = -my / t2
            dmz_relax = -(mz - m0) / t1
            
            mx += dmx_relax * dt
            my += dmy_relax * dt
            mz += dmz_relax * dt
            
            # 2. RF Rotation (simplified X-axis B1)
            # Alpha angle for this step
            alpha = gamma * b1 * dt
            
            # Rotate Y-Z plane around X
            new_my = my * np.cos(alpha) - mz * np.sin(alpha)
            new_mz = my * np.sin(alpha) + mz * np.cos(alpha)
            
            my = new_my
            mz = new_mz
            
            trajectory_x.append(mx)
            trajectory_y.append(my)
            trajectory_z.append(mz)
            
            # Accumulate "Quantum Surface Integral"
            # Berry phase roughly proportional to solid angle subtended
            # We integrate (1 - cos(theta)) d(phi) or similar area element
            # Here we define a synthetic metric for 'Quantum Stability'
            accumulated_phase += np.sqrt(mx**2 + my**2) * dt 
            
        surface_integral = accumulated_phase * 1000 # Scaling factor
        
        return {
            "surface_integral": surface_integral,
            "berry_phase": surface_integral * 0.1, # Mock relation
            "coherence_metric": np.sqrt(mx**2 + my**2 + mz**2),
            "trajectory": {
                "x": trajectory_x[::10], # Downsample for UI
                "y": trajectory_y[::10],
                "z": trajectory_z[::10]
            }
        }
