"""
Enhanced Cryo-Ablation Module with Accurate Ice Ball Dynamics
Models Joule-Thomson cooling, ice crystal formation, and thawing
"""

import numpy as np
from scipy.ndimage import gaussian_filter, laplace, distance_transform_edt

class GenerativeCryoPredictor:
    """Generative Gaussian Mixture model for targeted ice-ball geometry"""
    def __init__(self, latent_dim=8):
        self.latent_dim = latent_dim
        self.latent_state = np.random.normal(0, 0.1, latent_dim)
        
    def predict_perturbation(self, theta):
        """Generates geometric perturbations based on latent state"""
        # Periodic generative harmonics
        perturbation = sum(
            self.latent_state[i] * np.sin((i + 1) * theta)
            for i in range(self.latent_dim)
        )
        return perturbation

class EnhancedCryoModule:
    """Advanced cryogenic ablation simulation with Generative AI targeting"""
    
    def __init__(self, width=128, height=128):
        self.width = width
        self.height = height
        
        # Temperature map (°C)
        self.temp = np.full((height, width), 37.0)
        
        # Ice phase map (0=liquid, 1=solid ice)
        self.ice_map = np.zeros((height, width))
        
        # Cryoprobe position and state
        self.probe_active = False
        self.probe_pos = np.array([0.5, 0.5])
        self.probe_temp = -150.0  # Argon cryo temperature
        
        # Physical parameters
        self.cooling_power = 150.0  # Watts (Increased for faster simulation feedback)
        self.thermal_conductivity = 0.5  # W/(m·K) tissue
        self.ice_conductivity = 5.2     # W/(m·K) ice (Enhanced conductivity)
        self.freeze_point = 0.0  # °C
        self.max_freeze_radius = 0.25  # Normalized radius (Larger ice ball)
        
        # Tissue properties
        self.latent_heat_fusion = 334000.0  # J/kg for ice formation
        self.tissue_density = 1000.0  # kg/m³
        
        # Simulation state
        self.dt = 0.05  # time step (s)
        self.ice_volume_map = np.zeros((height, width))  # Volume fraction of ice
        self.damage_history = []
        self.accumulated_time = 0.0
        
        # Generative AI components
        self.predictor = GenerativeCryoPredictor()
        self.necrotic_mask = np.zeros((height, width))
        self.targeting_bias = 0.0 # Influence of necrotic core on geometry
        
    def activate_cryoprobe(self, x, y, power_pct=100.0):
        """Activate cryoprobe at position (x, y)"""
        self.probe_active = True
        self.probe_pos = np.array([x, y])
        self.cooling_power = 50.0 * (power_pct / 100.0)
    
    def deactivate_cryoprobe(self):
        """Deactivate cryoprobe and start thawing"""
        self.probe_active = False
    
    def update(self):
        """Update cryo-ablation state"""
        if self.probe_active:
            self._apply_cooling()
        else:
            self._apply_passive_thawing()
        
        # Update ice formation based on temperature
        self._update_ice_formation()
        
        self.accumulated_time += self.dt
    
    def _apply_cooling(self):
        """Apply active cooling from cryoprobe"""
        px = int(self.probe_pos[0] * self.width)
        py = int(self.probe_pos[1] * self.height)
        
        # Joule-Thomson cooling distribution (Gaussian with exponential falloff)
        yy, xx = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
        
        max_dist = max(self.width, self.height) * self.max_freeze_radius
        
        # Calculate angles for generative perturbation
        theta = np.arctan2(yy - py, xx - px)
        g_perturb = self.predictor.predict_perturbation(theta)
        
        # Necrotic Attraction: Pull ice ball towards necrotic core
        if np.any(self.necrotic_mask > 0.5):
            ny, nx = np.where(self.necrotic_mask > 0.5)
            nc_y, nc_x = np.mean(ny), np.mean(nx)
            # Vector from probe to necrotic center
            vec_to_nc = np.array([nc_x - px, nc_y - py])
            nc_dist = np.linalg.norm(vec_to_nc)
            if nc_dist > 1.0:
                dist_norm = vec_to_nc / nc_dist
                # Attract the distance metric asymmetrically
                nc_bias = (dist_norm[0] * (xx - px) + dist_norm[1] * (yy - py)) / (max_dist + 1e-6)
                egg_skew -= 0.3 * np.clip(nc_bias, -1, 1) * self.targeting_bias
        
        # Combined geometry: Ovoid + Generative Perturbation + Necrotic Bias
        dist = np.sqrt((xx - px)**2 + (yy - py)**2) * (egg_skew + g_perturb * 0.1)
        
        # Cooling power distribution: rapidly falls off with distance
        cooling_factor = np.exp(-(dist / max_dist) ** 2)
        
        # Temperature reduction rate (°C/s)
        # Follows exponential approach to probe temperature
        cooling_rate = cooling_factor * self.cooling_power / (self.tissue_density * 3600.0)
        
        # Apply cooling with thermal contact resistance
        contact_resistance = 1.0 - cooling_factor  # Higher resistance away from probe
        effective_cooling = cooling_rate / (1.0 + contact_resistance * 3.0) # More efficient cooling
        
        # Preferential cooling at probe tip
        probe_region = dist < (max_dist * 1.5)
        self.temp[probe_region] = np.minimum(
            self.temp[probe_region],
            self.probe_temp + (self.probe_temp - 37.0) * (1.0 - cooling_factor[probe_region])
        )
        
        # Regional cooling away from probe
        away_region = ~probe_region & (dist < max_dist * 3.0)
        self.temp[away_region] -= effective_cooling[away_region] * self.dt * 100.0
        self.temp[away_region] = np.maximum(self.temp[away_region], -50.0)
    
    def _apply_passive_thawing(self):
        """Apply passive warming (thawing) after probe deactivation"""
        # Passive thaw rate - depends on ice volume and perfusion
        
        # Conduction from surrounding tissue
        laplacian = laplace(self.temp)
        conduction = self.ice_conductivity * laplacian / (self.width ** 2 * 0.01)
        
        # Self-warming of frozen tissue towards arterial temperature (37°C)
        arterial_temp = 37.0
        perfusion_thaw = 0.002 * (arterial_temp - self.temp)  # Perfusion-mediated thawing
        
        # Rate-limiting: thawing is slower than freezing
        thaw_rate = 0.15  # °C/s baseline
        perfusion_assist = 0.05  # °C/s from perfusion
        
        # Activate thawing where ice exists
        thawing = np.where(
            self.ice_map > 0.5,
            thaw_rate + perfusion_assist,
            0.0
        )
        
        self.temp += thawing * self.dt
        self.temp = np.clip(self.temp, -50.0, 45.0)
    
    def _update_ice_formation(self):
        """Update ice crystal formation based on temperature"""
        # Phase diagram: ice forms at or below 0°C
        # Temperature-dependent ice fraction
        
        # Below freeze point: more ice
        below_freeze = self.temp <= self.freeze_point
        ice_fraction = np.where(
            below_freeze,
            np.clip((self.freeze_point - self.temp) / 30.0, 0.0, 1.0),  # 100% ice at -30°C
            0.0
        )
        
        # Hysteresis: thawed tissue doesn't refreeze easily until back below -5°C
        refreeze_hysteresis = self.temp <= -5.0
        self.ice_map = np.where(
            refreeze_hysteresis | below_freeze,
            ice_fraction,
            0.0
        )
        
        self.ice_volume_map = self.ice_map.copy()
    
    def get_ice_ball_boundary(self):
        """Get the boundary of ice ball formation"""
        # Find contour where ice fraction > 0.5
        boundary = (self.ice_map > 0.3) & (self.ice_map < 0.7)
        return boundary.astype(float)
    
    def get_ice_ball_center(self):
        """Get center of ice ball"""
        if np.max(self.ice_map) > 0.5:
            # Find largest connected region of ice
            from scipy.ndimage import label
            labeled_ice, num_features = label(self.ice_map > 0.5)
            
            if num_features > 0:
                largest_component = np.argmax(np.bincount(labeled_ice.flat)[1:]) + 1
                ice_region = labeled_ice == largest_component
                
                if np.any(ice_region):
                    coords = np.argwhere(ice_region)
                    center_y, center_x = coords.mean(axis=0)
                    return np.array([center_x / self.width, center_y / self.height])
        
        return self.probe_pos.copy()
    
    def get_ice_ball_radius_mm(self):
        """Estimate ice ball radius in mm"""
        if np.max(self.ice_map) > 0.5:
            ice_volume = np.sum(self.ice_map > 0.5)
            equiv_radius_pixels = np.sqrt(ice_volume / np.pi)
            equiv_radius_mm = equiv_radius_pixels * 100.0 / self.width  # Assuming 128 pixels = 128mm
            return equiv_radius_mm
        return 0.0
    
    def get_frozen_region(self):
        """Get region completely frozen (ice > 0.8)"""
        return (self.ice_map > 0.8).astype(float)
    
    def get_transition_zone(self):
        """Get transition zone (0.3 < ice < 0.7) - partial freezing"""
        return ((self.ice_map > 0.3) & (self.ice_map < 0.7)).astype(float)
    
    def get_map(self):
        """Get current temperature map"""
        return self.temp.copy()
    
    def get_ice_map(self):
        """Get ice volume fraction map"""
        return self.ice_map.copy()
    
    def get_damage_metrics(self):
        """Get cryo-ablation damage metrics"""
        fully_frozen = np.sum(self.ice_map > 0.8)
        partial_freeze = np.sum((self.ice_map > 0.3) & (self.ice_map <= 0.8))
        
        return {
            'total_ice_pixels': int(np.sum(self.ice_map > 0.0)),
            'fully_frozen_pixels': int(fully_frozen),
            'transition_zone_pixels': int(partial_freeze),
            'ice_ball_radius_mm': self.get_ice_ball_radius_mm(),
            'max_penetration_mm': float(np.max(distance_transform_edt(self.ice_map < 0.3) if np.any(self.ice_map < 0.3) else np.zeros_like(self.ice_map)) * 100.0 / self.width),
            'probe_position': self.probe_pos.tolist(),
            'probe_active': self.probe_active,
            'min_temperature': float(np.min(self.temp[self.ice_map > 0.3])) if np.any(self.ice_map > 0.3) else 37.0,
            'necrotic_coverage': float(np.sum(self.ice_map[self.necrotic_mask > 0.5] > 0.5) / np.sum(self.necrotic_mask > 0.5)) if np.sum(self.necrotic_mask > 0.5) > 0 else 0.0,
            'generative_complexity': float(np.std(self.predictor.latent_state))
        }
    
    def set_necrotic_mask(self, mask):
        """Update necrotic mask from segmenter"""
        self.necrotic_mask = mask.astype(float)
        self.targeting_bias = 0.8  # Enable targeted cooling bias
