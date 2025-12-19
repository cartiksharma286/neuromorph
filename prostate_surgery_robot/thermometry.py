
import numpy as np
from scipy.ndimage import gaussian_filter

class ThermalModel:
    def __init__(self, w, h, tissue_mask):
        self.w = w
        self.h = h
        self.temp_map = np.full((w, h), 37.0)
        self.damage_map = np.zeros((w, h))
        self.tissue_mask = tissue_mask # 1 where tissue exists
        
        # Physics
        self.diffusivity = 0.5 # mm^2/s approx
        self.perfusion = 0.005 # cooling rate
        
    def step(self):
        # 1. Diffusion (Gaussian filter approximation or Laplace)
        # Using a small gaussian blur matches heat equation step
        self.temp_map = gaussian_filter(self.temp_map, sigma=0.8)
        
        # 2. Perfusion (Return to 37C)
        # cool down to 37
        diff = 37.0 - self.temp_map
        self.temp_map += diff * self.perfusion
        
        # 3. Damage Calc (CEM43)
        # If T > 43
        high_temp = self.temp_map > 43.0
        # R = 0.5 for T>43
        # Add dose. Simple accumulation
        dose_rate = 2.0 ** (self.temp_map[high_temp] - 43.0)
        self.damage_map[high_temp] += dose_rate * 0.05 # dt assumption
        
    def apply_heat(self, x, y, power):
        # Add heat at x,y
        # Simple point source injection
        # Spread over 3px radius
        if 0 <= x < self.w and 0 <= y < self.h:
            self.temp_map[x, y] += power * 0.5
            # Neighbors
            self.temp_map[max(0, x-1):min(self.w, x+2), max(0, y-1):min(self.h, y+2)] += power * 0.2

    def apply_cooling(self, x, y, intensity):
        # Remove heat
        if 0 <= x < self.w and 0 <= y < self.h:
            current = self.temp_map[x, y]
            # Limit cooling to -100C standard cryo
            if current > -100.0:
                self.temp_map[x, y] -= intensity * 0.5
                self.temp_map[max(0, x-1):min(self.w, x+2), max(0, y-1):min(self.h, y+2)] -= intensity * 0.2
                
    def get_temp_map_compressed(self):
        # Return as list
        return self.temp_map.tolist()
