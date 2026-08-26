"""
Enhanced Thermometry with Accurate Bioheat Transfer Equation (BHTE)
Implements Pennes' bioheat equation with tissue perfusion, blood flow, and metabolism
"""

import numpy as np
from scipy.ndimage import laplace, gaussian_filter, convolve
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

class EnhancedThermometry:
    """High-fidelity thermal simulation with bioheat physics"""
    
    def __init__(self, width=128, height=128):
        self.width = width
        self.height = height
        
        # Temperature field (°C)
        self.T = np.full((height, width), 37.0)
        self.T_prev = self.T.copy()
        
        # Tissue properties
        self.rho = 1000.0  # density kg/m³
        self.c = 3600.0    # specific heat J/(kg·°C)
        self.k = 0.5       # thermal conductivity W/(m·°C)
        
        # Perfusion parameters
        self.omega_b = 0.005  # blood perfusion rate (1/s) - varies with tissue type
        self.T_b = 37.0       # arterial blood temperature (°C)
        self.Q_m = 100.0      # metabolic heat generation (W/m³)
        
        # Tissue types map
        self.tissue_type = np.zeros((height, width))  # 0=white matter, 1=gray matter, 2=tumor
        self.setup_tissue_properties()
        
        # Damage/necrosis tracking
        self.damage_map = np.zeros((height, width))  # CEM43 thermal dose
        self.necrotic_map = np.zeros((height, width), dtype=bool)
        
        # History tracking
        self.temp_history = []
        self.dt = 0.01  # time step (s)
        self.accumulated_time = 0.0
        
    def setup_tissue_properties(self):
        """Set up tissue-type dependent properties"""
        # Create synthetic brain anatomy
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height))
        
        # White matter (background) - Low contrast (0.3)
        self.tissue_type.fill(0.3)
        
        # Gray matter (central region) - Mid contrast (0.6)
        gray_matter_mask = (xx**2 + 0.5*yy**2) < 0.5
        self.tissue_type[gray_matter_mask] = 0.6
        
        # Tumor (high intensity region) - High contrast (1.0)
        tumor_mask = ((xx - 0.2)**2 + (yy - 0.2)**2) < 0.05
        self.tissue_type[tumor_mask] = 1.0
    
    def get_tissue_parameters(self, value):
        """Get tissue-specific thermal parameters based on map value"""
        # Map values back to types
        if abs(value - 0.6) < 0.1:
            return {'rho': 1050, 'c': 3400, 'k': 0.51, 'omega_b': 0.008, 'Q_m': 700}  # Gray matter
        elif abs(value - 1.0) < 0.1:
            return {'rho': 1050, 'c': 3520, 'k': 0.52, 'omega_b': 0.012, 'Q_m': 1500} # Tumor
        else:
            return {'rho': 1040, 'c': 3300, 'k': 0.48, 'omega_b': 0.003, 'Q_m': 300}   # White matter
    
    def apply_heat_source(self, x, y, power_watts, radius_mm=2.0):
        """Apply laser heat source at position (x, y) normalized to [0,1]"""
        # Convert to pixel coordinates
        px = int(x * self.width)
        py = int(y * self.height)
        
        # Radius in pixels
        radius_pix = max(1, int(radius_mm * self.width / 100.0))
        
        # Create Gaussian heat distribution
        yy, xx = np.ogrid[-radius_pix:radius_pix+1, -radius_pix:radius_pix+1]
        gaussian = np.exp(-(xx**2 + yy**2) / (2 * radius_pix**2))
        gaussian = gaussian / np.sum(gaussian)  # Normalize
        
        # Apply heat with power scaling
        heat_field = gaussian * power_watts / (self.rho * self.c)
        
        # Place on temperature field with bounds checking
        y_start = max(0, py - radius_pix)
        y_end = min(self.height, py + radius_pix + 1)
        x_start = max(0, px - radius_pix)
        x_end = min(self.width, px + radius_pix + 1)
        
        h_start_y = max(0, radius_pix - py)
        h_end_y = h_start_y + (y_end - y_start)
        h_start_x = max(0, radius_pix - px)
        h_end_x = h_start_x + (x_end - x_start)
        
        self.T[y_start:y_end, x_start:x_end] += heat_field[h_start_y:h_end_y, h_start_x:h_end_x]
    
    def apply_cooling_source(self, x, y, cooling_power_watts, radius_mm=3.0):
        """Apply cryogenic cooling source"""
        px = int(x * self.width)
        py = int(y * self.height)
        radius_pix = max(1, int(radius_mm * self.width / 100.0))
        
        # Gaussian cooling distribution
        yy, xx = np.ogrid[-radius_pix:radius_pix+1, -radius_pix:radius_pix+1]
        gaussian = np.exp(-(xx**2 + yy**2) / (2 * radius_pix**2))
        gaussian = gaussian / np.sum(gaussian)
        
        # Apply cooling (negative heat)
        cooling_field = gaussian * cooling_power_watts / (self.rho * self.c)
        
        y_start = max(0, py - radius_pix)
        y_end = min(self.height, py + radius_pix + 1)
        x_start = max(0, px - radius_pix)
        x_end = min(self.width, px + radius_pix + 1)
        
        h_start_y = max(0, radius_pix - py)
        h_end_y = h_start_y + (y_end - y_start)
        h_start_x = max(0, radius_pix - px)
        h_end_x = h_start_x + (x_end - x_start)
        
        self.T[y_start:y_end, x_start:x_end] -= cooling_field[h_start_y:h_end_y, h_start_x:h_end_x]
    
    def update(self):
        """Update temperature using Pennes' bioheat equation"""
        self.T_prev = self.T.copy()
        
        # Pennes' equation: ρc ∂T/∂t = ∇·(k∇T) + ρ_b ω_b c_b (T_b - T) + Q_m + Q_ext
        
        # Diffusion term (∇·(k∇T))
        laplacian = laplace(self.T)
        diffusion = self.k * laplacian / (self.width ** 2 * 0.01)  # 0.01 = mm² per pixel²
        
        # Perfusion cooling term (ρ_b ω_b c_b (T_b - T))
        perfusion = np.zeros_like(self.T)
        for tissue_id in np.unique(self.tissue_type):
            params = self.get_tissue_parameters(tissue_id)
            tissue_mask = (self.tissue_type == tissue_id) & (~self.necrotic_map)
            
            # Perfusion stops in necrotic regions
            if np.any(tissue_mask):
                perfusion[tissue_mask] = (
                    params['rho'] * params['omega_b'] * params['c'] * 
                    (self.T_b - self.T[tissue_mask]) / (self.rho * self.c)
                )
        
        # Metabolic heat term
        metabolism = np.zeros_like(self.T)
        for tissue_id in np.unique(self.tissue_type):
            params = self.get_tissue_parameters(tissue_id)
            tissue_mask = (self.tissue_type == tissue_id) & (~self.necrotic_map)
            if np.any(tissue_mask):
                metabolism[tissue_mask] = params['Q_m'] / (self.rho * self.c)
        
        # Combined update
        dT_dt = diffusion + perfusion + metabolism
        self.T = self.T + dT_dt * self.dt
        
        # Clamp to reasonable temperature range
        self.T = np.clip(self.T, 20.0, 100.0)
        
        # Update thermal damage
        self._update_damage()
        
        # Track history
        self.accumulated_time += self.dt
        if int(self.accumulated_time * 10) % 2 == 0:  # Log every 0.2s
            self.temp_history.append(np.max(self.T))
    
    def _update_damage(self):
        """Update CEM43 thermal dose (Cumulative Equivalent Minutes at 43°C)"""
        # CEM43 = Σ R^(43-T) * Δt
        # where R = 0.5 for T ≥ 43°C, R = 0.25 for T < 43°C
        
        dt_minutes = self.dt / 60.0
        
        # Apply damage calculation
        R = np.where(self.T >= 43.0, 0.5, 0.25)
        damage_rate = R ** (43.0 - self.T)
        self.damage_map += damage_rate * dt_minutes
        
        # Mark necrotic tissue (damage > 240 CEM43 ≈ irreversible damage)
        self.necrotic_map = self.damage_map > 240.0
    
    def get_map(self):
        """Get current temperature map"""
        return self.T.copy()
    
    def get_damage_map(self):
        """Get thermal damage map"""
        return self.damage_map.copy()
    
    def get_necrotic_map(self):
        """Get necrotic tissue map"""
        return self.necrotic_map.astype(float)
    
    def get_history(self):
        """Get temperature history"""
        return self.temp_history.copy()
    
    def get_performance_metrics(self):
        """Get thermal simulation performance metrics"""
        return {
            'max_temperature': float(np.max(self.T)),
            'mean_temperature': float(np.mean(self.T)),
            'peak_damage': float(np.max(self.damage_map)),
            'necrotic_volume': int(np.sum(self.necrotic_map)),
            'temperature_gradient': float(np.max(np.gradient(self.T))),
            'simulation_time_s': self.accumulated_time,
        }
