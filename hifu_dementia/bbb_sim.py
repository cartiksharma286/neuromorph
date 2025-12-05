
import numpy as np
import time

class BBBSimulation:
    """
    Simulates the Blood-Brain Barrier (BBB) response to HIFU.
    """
    def __init__(self):
        self.permeability = 0.0  # 0 to 1
        self.temp = 37.0         # Celsius
        self.pressure_map = np.zeros((10, 10)) # Simple grid
        
    def step(self, freq_mhz, intensity_wcm2, duration_ms):
        """
        Advance simulation by one time step.
        """
        # Physics simplifications:
        # Temperature rise ~ Intensity * duration
        # Permeability ~ Pressure (from Intensity) * Frequency resonance
        
        # 1. Update Temperature (Heat diffusion model simplified)
        delta_temp = (intensity_wcm2 * 0.1) - ((self.temp - 37.0) * 0.05)
        self.temp += delta_temp
        
        # 2. Permeability (Sigmoid activation based on acoustic pressure)
        # Pressure ~ sqrt(Intensity)
        pressure = np.sqrt(intensity_wcm2) * 1.5
        
        # Effective cavitation threshold logic
        if pressure > 1.2 and freq_mhz < 0.8:
            target_perm = 0.8 # Open
        else:
            target_perm = 0.0 # Closed
            
        # Smooth transition
        self.permeability = self.permeability * 0.9 + target_perm * 0.1
        
        return {
            "permeability": self.permeability,
            "temperature": self.temp,
            "pressure": pressure,
            "cavitation_stable": 1.0 if pressure < 1.8 else 0.0
        }
