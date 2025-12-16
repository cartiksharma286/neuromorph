import numpy as np
from scipy.ndimage import laplace

class MRIThermometry:
    def __init__(self, width=64, height=64):
        self.width = width
        self.height = height
        # 37.0 degrees C baseline body temp
        self.temperature_map = np.full((width, height), 37.0) 
        self.diffusion_rate = 0.2
        self.cooling_rate = 0.01 # Blood perfusion
        self.dt = 0.1

        self.max_temp_history = []
        self.time_step_count = 0

    def apply_laser(self, x, y, power, enabled):
        """
        x, y: Coordinates in the grid (0..width, 0..height)
        power: Laser power intensity (heat source)
        """
        if not enabled:
            return

        # Simple Gaussian distribution for laser heat source
        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        # Center the gaussian on x,y
        sigma = 2.0
        # Cortical tissue absorbs differently? Let's just assume homogenous for now but higher power density
        source = power * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        
        # Add heat
        self.temperature_map += source * self.dt

    def update(self):
        """
        Run one time step of the heat equation.
        """
        # Diffusion: Laplacian
        lap = laplace(self.temperature_map)
        
        # Cooling toward body temp (37.0)
        # Cortical surface cooling might be slightly higher due to CSF buffering, 
        # let's simulate a variable cooling map later if needed.
        cooling = (self.temperature_map - 37.0) * self.cooling_rate
        
        # Update
        delta = (self.diffusion_rate * lap) - cooling
        self.temperature_map += delta * self.dt
        
        # Clip strictly to realistic physics (e.g. can't drop below 37 in this simple model if no cooler)
        self.temperature_map = np.maximum(self.temperature_map, 37.0)
        
        # Track history
        self.time_step_count += 1
        if self.time_step_count % 5 == 0: # Record every 5th step
            self.max_temp_history.append(float(np.max(self.temperature_map)))
            if len(self.max_temp_history) > 100:
                self.max_temp_history.pop(0)

    def get_map(self):
        return self.temperature_map
        
    def get_history(self):
        return self.max_temp_history
