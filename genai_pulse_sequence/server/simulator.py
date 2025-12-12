import numpy as np

class MRISimulator:
    def __init__(self):
        self.gamma = 42.58e6  # Hz/T
        self.dt = 1e-6 # 1 microsecond dwell time
        
    def bloch_simulation(self, rf_pulse, b0_inhomogeneity=0.0):
        """
        Simulate the effect of the RF pulse on a spin isochromat.
        rf_pulse: complex array (B1 field in Tesla).
        """
        # Initial magnetization [Mx, My, Mz]
        M = np.array([0.0, 0.0, 1.0])
        
        trajectory = [M.copy()]
        
        # Effective field calculation
        # Beff = [B1_x, B1_y, B0_z]
        # Rotation per step
        
        for rf in rf_pulse:
            # Assume on-resonance for simplicity initially, or add random offset
            # B1 in Tesla (derived from pulse amplitude which is arbitrary in generator)
            # Let's verify scaling.
            # Convert generic generator amplitude to Teslas (microTeslas range)
            b1 = rf * 1e-6 # Scale factor
            
            # Rotation vector
            w_x = -self.gamma * np.real(b1) * 2 * np.pi
            w_y = -self.gamma * np.imag(b1) * 2 * np.pi
            w_z = -self.gamma * b0_inhomogeneity * 2 * np.pi # Off resonance
            
            # Small angle approximation or Rodrigues formula for rotation
            w = np.array([w_x, w_y, w_z])
            angle = np.linalg.norm(w) * self.dt
            if angle > 0:
                axis = w / np.linalg.norm(w)
                M = self.rotate_vec(M, axis, angle)
            
            # Simple Relaxation (neglected for short pulse duration but can add T1/T2)
            # M[0] *= np.exp(-dt/T2) ...
            
            trajectory.append(M.copy())
            
        trajectory = np.array(trajectory)
        
        # Signal is transverse magnetization
        final_signal = complex(M[0], M[1])
        signal_mag = np.abs(final_signal)
        
        # Add noise to simulate real acquisition
        noise_floor = 0.05
        measured_signal = signal_mag + np.random.normal(0, noise_floor)
        
        snr = measured_signal / noise_floor
        
        # Calculate final flip angle from Mz
        # Mz = cos(alpha) -> alpha = arccos(Mz)
        final_mz = np.clip(M[2], -1.0, 1.0)
        flip_angle_deg = np.arccos(final_mz) * (180.0 / np.pi)
        
        return {
            "final_magnetization": M.tolist(),
            "trajectory": trajectory[:,:3].tolist(), # Nx3
            "signal": measured_signal,
            "snr": snr,
            "flip_angle": flip_angle_deg
        }

    def simulate_slice_profile(self, rf_pulse, n_positions=100):
        """
        Simulates the RF pulse across a range of off-resonance frequencies (positions).
        Returns the transverse magnetization profile (Slice Profile).
        """
        positions = np.linspace(-1000, 1000, n_positions) # Hz off-resonance
        profile = []
        
        # Re-use simplified bloch core, but for many offsets
        # Ideally vectorised, but loop is fine for demo
        for f_off in positions:
            # We only track Mxy magnitude for the profile
            M = np.array([0.0, 0.0, 1.0])
            for rf in rf_pulse:
                b1 = rf * 1e-6
                w_x = -self.gamma * np.real(b1) * 2 * np.pi
                w_y = -self.gamma * np.imag(b1) * 2 * np.pi
                w_z = -self.gamma * (f_off / self.gamma) * 2 * np.pi # Off resonance B0 term
                
                # Update B0 rotation directly from freq offset
                # w_z is actually just -2*pi*f_off
                w_z = -2 * np.pi * f_off
                
                w = np.array([w_x, w_y, w_z])
                angle = np.linalg.norm(w) * self.dt
                if angle > 0:
                    axis = w / np.linalg.norm(w)
                    M = self.rotate_vec(M, axis, angle)
            
            # Transverse mag
            mxy = np.sqrt(M[0]**2 + M[1]**2)
            profile.append(mxy)
            
        return positions, np.array(profile)

    def generate_reconstruction(self, rf_pulse):
        """
        Generates a mock phantom image multiplied by the slice profile.
        """
        import matplotlib.pyplot as plt
        import io
        import base64
        
        # 1. Get Slice Profile
        freqs, profile = self.simulate_slice_profile(rf_pulse)
        
        # 2. Create Phantom (2D circle)
        N = 128
        image = np.zeros((N, N))
        y, x = np.ogrid[-N//2:N//2, -N//2:N//2]
        mask = x*x + y*y <= (N//3)**2
        image[mask] = 1.0
        
        # 3. Apply Slice Profile as a weighting along one axis (Gradient direction)
        # Map profile (1D) to image (2D)
        # Interpolate profile to length N
        profile_interp = np.interp(np.linspace(0, len(profile)-1, N), np.arange(len(profile)), profile)
        
        # Apply profile along X-axis (readout/gradient direction simulation)
        weighted_image = image * profile_interp[None, :] 
        
        # 4. Render to Image
        plt.figure(figsize=(4, 4), facecolor='black')
        plt.imshow(weighted_image, cmap='magma', vmin=0, vmax=1)
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_b64

    def export_pulseq(self, rf_pulse, bandwidth, amplitude):
        """
        Writes a basic .seq file content for the given pulse.
        """
        # Header
        seq_content = []
        seq_content.append("# Pulseq Sequence file")
        seq_content.append("# Created by NeuroMorph GenAI")
        seq_content.append("")
        seq_content.append("[VERSION]")
        seq_content.append("major 1")
        seq_content.append("minor 4")
        seq_content.append("revision 1")
        seq_content.append("")
        seq_content.append("[DEFINITIONS]")
        seq_content.append("Target_Bandwidth " + str(bandwidth))
        seq_content.append("Target_Amplitude " + str(amplitude))
        seq_content.append("")
        
        # RF Shapes section
        seq_content.append("[SHAPES]")
        seq_content.append("id 1")
        # Pulseq uses normalized shapes (0..1) typically, or Real/Imag pairs
        # We will just dump the waveform as compressed text (Run-length encoded usually, but bare floats works in uncompressed)
        # Actually Pulseq standard requires specific columns. 
        # For simplicity, we just list the samples.
        seq_content.append("num_samples " + str(len(rf_pulse)))
        for sample in rf_pulse:
            # Check for non-float (e.g. numpy scalar)
            val = float(sample)
            seq_content.append(f"{val:.6f}") 
            
        seq_content.append("")
        seq_content.append("# End of sequence")
        
        return "\n".join(seq_content)

    def rotate_vec(self, v, axis, theta):
        """
        Rodrigues' rotation formula.
        """
        k = axis
        return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))

