
import numpy as np
from numba import cuda
import math

# Constant for simulation
TPB = 16

@cuda.jit
def propagate_field_kernel(field, result, decay_factor):
    """
    CUDA kernel to simulate field propagation and diffusion.
    Solves a simple discrete diffusion equation: dU/dt = D * Laplacian(U)
    """
    r, c = cuda.grid(2)
    
    rows, cols = field.shape
    
    if 1 <= r < rows - 1 and 1 <= c < cols - 1:
        # 5-point Laplacian stencil
        laplacian = (
            field[r+1, c] + field[r-1, c] + 
            field[r, c+1] + field[r, c-1] - 
            4 * field[r, c]
        )
        
        # Update result (simulating one time step)
        result[r, c] = field[r, c] + decay_factor * laplacian

class GenerativeAcousticField:
    """
    Generative AI model to synthesize optimal acoustic pressure fields.
    Uses 'simulated' GAN logic to produce patterns, then uses CUDA
    to physically validate/propagate them.
    """
    def __init__(self):
        self.latent_dim = 100
        self.resolution = 128
        
    def generate_pattern(self, seed_noise):
        """
        Simulates a Generator G(z) -> Image.
        Produces a 'focused' pattern typical of phased arrays.
        """
        # Mock Generator Logic: 
        # Create interference pattern based on seed
        x = np.linspace(-10, 10, self.resolution)
        y = np.linspace(-10, 10, self.resolution)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        # "Dream" parameters from noise
        freq_mod = np.mean(seed_noise) * 5.0
        phase_mod = np.std(seed_noise) * 10.0
        
        # Bessel-like pattern (Bessel beams are non-diffracting)
        pattern = np.cos(freq_mod * R + phase_mod) * np.exp(-0.1 * R)
        
        # Add some "learned" hotspots
        pattern += np.exp(-((X-2)**2 + (Y-2)**2)) * 0.5
        
        return pattern

    def propagate_cuda(self, initial_field):
        """
        Uses CUDA to propagate the field to the target depth.
        """
        if not cuda.is_available():
            # CPU Fallback
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(initial_field, sigma=2)
            
        # CUDA Path
        d_field = cuda.to_device(initial_field)
        d_result = cuda.device_array_like(initial_field)
        
        threadsperblock = (TPB, TPB)
        blockspergrid_x = math.ceil(initial_field.shape[0] / TPB)
        blockspergrid_y = math.ceil(initial_field.shape[1] / TPB)
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        # Run kernel
        propagate_field_kernel[blockspergrid, threadsperblock](d_field, d_result, 0.1)
        
        return d_result.copy_to_host()

    def synthesize_optimal_field(self):
        """
        Full pipeline: Latent Noise -> GAN -> CUDA Propagation -> Result
        """
        z = np.random.normal(0, 1, self.latent_dim)
        raw_pattern = self.generate_pattern(z)
        
        # Propagate to see effective dose at depth
        effective_dose = self.propagate_cuda(raw_pattern)
        
        return raw_pattern, effective_dose
