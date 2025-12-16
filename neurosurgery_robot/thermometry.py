import numpy as np
from scipy.ndimage import laplace

class MRIThermometry:
    def __init__(self, width=64, height=64):
        self.width = width
        self.height = height
        # 37.0 degrees C baseline body temp
        self.temperature_map = np.full((width, height), 37.0) 
        self.damage_map = np.zeros((width, height)) # CEM43 Accumulated Damage
        
        self.diffusion_rate = 0.2
        self.base_cooling_rate = 0.01 # Blood perfusion
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
        Run one time step of the heat equation with Pennes Bioheat terms.
        """
        # Diffusion: Laplacian
        lap = laplace(self.temperature_map)
        
        # Bioheat Perfusion Term
        # Perfusion stops if tissue is necrotic (Coagulation > 240 CEM43 is a standard threshold)
        necrotic_mask = self.damage_map > 240.0
        
        # Perfusion cooling is proportional to (T - T_arterial), but effectively 0 in necrotic tissue
        cooling_factor = np.where(necrotic_mask, 0.0, self.base_cooling_rate)
        cooling = (self.temperature_map - 37.0) * cooling_factor
        
        # Update
        delta = (self.diffusion_rate * lap) - cooling
        self.temperature_map += delta * self.dt
        
        # Clip strictly to realistic physics (e.g., can't drop below 37 without cryo)
        self.temperature_map = np.maximum(self.temperature_map, 37.0)
        
        # Update Thermal Dose (CEM43)
        # R = 0.5 for T >= 43, R = 0.25 for T < 43
        # dCEM = dt * R^(43 - T) ... Wait, standard is R^(T-43)
        # Dose = Sum( R^(T-43) * dt )
        
        T = self.temperature_map
        R = np.where(T >= 43.0, 0.5, 0.25)
        d_dose = (R**(43.0 - T)) * (self.dt / 60.0) # Convert dt to minutes if dt is seconds? 
        # Let's assume dt is dimensionless simulation steps, treating as "seconds" for scale
        d_dose = (2.0**(T - 43.0)) * self.dt # Standard approximation for T>43 often uses 2^Delta
        # Actually standard: CEM43 = sum( R^(43-T) * dt )
        # If T > 43, R=0.5. So 0.5^(43-T) = 2^(T-43). Correct.
        # If T < 43, R=0.25. 0.25^(43-T) = 4^(T-43) -> very small.
        
        # Let's use the explicit piecewise for clarity:
        r_factor = np.zeros_like(T)
        mask_high = T >= 43.0
        r_factor[mask_high] = 0.5 ** (43.0 - T[mask_high]) # Equivalent to 2^(T-43)
        r_factor[~mask_high] = 0.25 ** (43.0 - T[~mask_high]) 
        
        self.damage_map += r_factor * (self.dt / 60.0) # accumulation in "equivalent minutes"
        
        # Track history
        self.time_step_count += 1
        if self.time_step_count % 5 == 0: # Record every 5th step
            self.max_temp_history.append(float(np.max(self.temperature_map)))
            if len(self.max_temp_history) > 100:
                self.max_temp_history.pop(0)

    def get_map(self):
        return self.temperature_map
        
    def get_damage_map(self):
        return self.damage_map

    def get_history(self):
        return self.max_temp_history
