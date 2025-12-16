import numpy as np
from scipy.ndimage import laplace

class CryoModule:
    """
    Cryo-Ablation Simulation Module.
    Simulates Joule-Thomson cooling and ice ball formation.
    """
    def __init__(self, width=64, height=64):
        self.width = width
        self.height = height
        self.temp_map = np.full((width, height), 37.0) # start at body temp
        # Parameters for Argon/Helium Cryo
        self.diffusion_rate = 0.15 # Ice conducts better than tissue?
        self.thawing_rate = 0.05
        self.dt = 0.1
        self.anatomy_map = self.generate_anatomy()

    def generate_anatomy(self):
        """Generates a synthetic MR Grayscale slice (Brain phantom)"""
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height))
        # Skull/Head shape
        mask = (xx**2 + yy**2) < 0.8
        anatomy = np.zeros_like(xx)
        anatomy[mask] = 0.3 # Background tissue
        
        # 'Brain' structure differences (Gray vs White matter)
        brain_mask = (xx**2 + 0.5*yy**2) < 0.6
        anatomy[brain_mask] = 0.6 + np.random.normal(0, 0.05, np.sum(brain_mask))
        
        # 'Ventricles' (Darker fluid)
        ventricle = (np.abs(xx) < 0.15) & (np.abs(yy) < 0.3)
        anatomy[ventricle] = 0.1
        
        # Tumor target (Bright in T2)
        tumor = ((xx - 0.2)**2 + (yy - 0.2)**2) < 0.05
        anatomy[tumor] = 0.9
        
        return anatomy

    def apply_cryo(self, x, y, pressure_level, enabled):
        """
        x, y: Coordinates
        pressure_level: PSI (affects cooling power)
        """
        if not enabled:
            return

        # Ice ball formation is spherical (circular in 2D)
        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        
        # Joule-Thomson coefficient approx -> cooling power proportional to pressure drop
        # 3000 PSI -> very cold
        cooling_power = -8.0 * (pressure_level / 1000.0) 
        
        # Radius expands with time/power
        radius = 4.0 
        
        dist = np.sqrt((xx - x)**2 + (yy - y)**2)
        
        # Cooling source (sharp gradient) using Gaussian falloff for smoother simulation
        source = cooling_power * np.exp(-(dist**2) / (2 * radius**2))
        
        # Apply cooling source
        self.temp_map += source * self.dt
        
    def update(self):
        # 1. Thermal diffusion (Cold spreads)
        lap = laplace(self.temp_map)
        
        # 2. Thaw / Blood Perfusion (Warm up to 37.0)
        # Perfusion is lower in frozen tissue (vascular stasis) but high at margin
        warming = (37.0 - self.temp_map) * self.thawing_rate
        
        # 3. Update State
        delta = (self.diffusion_rate * lap) + warming
        self.temp_map += delta * self.dt
        
        # Clamp: Liquid Nitrogen/Argon limit (-180 C)
        self.temp_map = np.maximum(self.temp_map, -180.0)
        self.temp_map = np.minimum(self.temp_map, 37.0) 
        
    def get_map(self):
        return self.temp_map
