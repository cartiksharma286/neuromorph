import numpy as np

class ReconstructionSimulator:
    def __init__(self):
        self.size = 128
        self.phantom = self._generate_phantom(self.size)
        
    def _generate_phantom(self, size):
        """Generates a simple 2D phantom (circles/squares)."""
        phantom = np.zeros((size, size))
        cx, cy = size // 2, size // 2
        
        # Outer circle (Head)
        y, x = np.ogrid[-cx:size-cx, -cy:size-cy]
        mask1 = x*x + y*y <= (size//2 - 5)**2
        phantom[mask1] = 1.0
        
        # Inner structures (Ventricles - mock)
        mask2 = (x-15)**2 + (y-10)**2 <= 10**2
        phantom[mask2] = 0.5
        mask3 = (x+15)**2 + (y-10)**2 <= 10**2
        phantom[mask3] = 0.5
        
        # Tumor or lesion (Hyperintense)
        mask4 = (x)**2 + (y+20)**2 <= 5**2
        phantom[mask4] = 2.0
        
        return phantom

    def reconstruct(self, pulse_data):
        """
        Simulates k-space acquisition based on the pulse sequence.
        Note: This is a simplified educational simulation.
        Real MRI fills k-space based on the integral of gradients.
        """
        # 1. Analyze Sequence to determine k-space coverage
        # For a standard GRE/SE, we expect Gy to step and Gx to be constant during readout.
        
        gx = np.array(pulse_data['gx'])
        gy = np.array(pulse_data['gy'])
        adc = np.array(pulse_data['adc'])
        
        # In a real sim, we would integrate G(t) to get k(t).
        # k(t) = gamma * integral(G(tau) dtau)
        # S(k) = Integral( Rho(r) * exp(-i 2pi k*r) dr ) => FT of object
        
        # Shortcut for demo: We just take the ground truth phantom FFT 
        # and apply some degradation based on the sequence quality to simulate "scanning".
        
        k_space_ideal = np.fft.fftshift(np.fft.fft2(self.phantom))
        
        # Check if sequence is "valid enough" to form an image
        # We need phase encoding steps (varying Gy) and frequency encoding (Gx)
        
        has_readout = np.max(np.abs(gx * adc)) > 0
        has_phase_enc = np.max(np.abs(gy)) > 0
        
        reconstructed = np.zeros_like(self.phantom)
        
        if has_readout and has_phase_enc:
            # Simulate sampling
            # If TR is too short, T1 effects (ignore for now or scale intensity)
            # If TE is long, T2 decay (apply scalar decay)
            
            # Simulate "Noise"
            noise = np.random.normal(0, 0.05, k_space_ideal.shape) + 1j * np.random.normal(0, 0.05, k_space_ideal.shape)
            k_space_acquired = k_space_ideal + noise
            
            # Reconstruction (Inverse FFT)
            img_cpx = np.fft.ifft2(np.fft.ifftshift(k_space_acquired))
            reconstructed = np.abs(img_cpx)
        else:
            # Return noise or blank if sequence is broken
            reconstructed = np.random.rand(self.size, self.size) * 0.1
            
        return {
            "image": reconstructed.tolist(),
            "k_space": np.abs(k_space_ideal).tolist(), # Send magnitude for viz
            "width": self.size,
            "height": self.size
        }
