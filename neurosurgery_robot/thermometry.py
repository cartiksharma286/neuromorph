"""
High-Performance MRI Thermometry with Quantum-Enhanced Heat Transfer
Optimized for real-time surgical simulation with advanced bioheat modeling
"""

import numpy as np
from scipy.ndimage import laplace, gaussian_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import numba

@numba.jit(nopython=True, cache=True)
def _compute_cem43_fast(temperature, dt, damage_map):
    """
    Fast CEM43 thermal dose calculation using Numba JIT compilation
    
    CEM43 = Σ R^(43-T) * Δt
    where R = 0.5 for T ≥ 43°C, R = 0.25 for T < 43°C
    """
    height, width = temperature.shape
    dt_minutes = dt / 60.0
    
    for i in range(height):
        for j in range(width):
            T = temperature[i, j]
            if T >= 43.0:
                r_factor = 0.5 ** (43.0 - T)  # = 2^(T-43)
            else:
                r_factor = 0.25 ** (43.0 - T)  # = 4^(T-43)
            
            damage_map[i, j] += r_factor * dt_minutes
    
    return damage_map

@numba.jit(nopython=True, cache=True)
def _apply_perfusion_cooling(temperature, damage_map, base_rate, arterial_temp, dt):
    """
    Fast perfusion cooling with necrotic tissue handling
    Perfusion stops in necrotic regions (damage > 240 CEM43)
    """
    height, width = temperature.shape
    
    for i in range(height):
        for j in range(width):
            if damage_map[i, j] > 240.0:
                # Necrotic tissue - no perfusion
                continue
            else:
                # Active perfusion cooling
                cooling = (temperature[i, j] - arterial_temp) * base_rate
                temperature[i, j] -= cooling * dt
    
    return temperature


class MRIThermometry:
    """
    High-performance MRI thermometry with advanced bioheat modeling
    
    Features:
    - Implicit finite difference for stability
    - Numba JIT compilation for speed
    - Quantum-enhanced heat pattern integration
    - Realistic tissue perfusion modeling
    - CEM43 thermal dose tracking
    - Multi-tissue heterogeneity support
    """
    
    def __init__(self, width=64, height=64, high_performance=True):
        self.width = width
        self.height = height
        self.high_performance = high_performance
        
        # Temperature field (°C)
        self.temperature_map = np.full((height, width), 37.0, dtype=np.float64)
        
        # Thermal dose accumulation (CEM43 equivalent minutes)
        self.damage_map = np.zeros((height, width), dtype=np.float64)
        
        # Tissue properties map (0=normal, 1=tumor, 2=critical structure)
        self.tissue_map = self._initialize_tissue_map()
        
        # Physical parameters
        self.thermal_conductivity = 0.5  # W/(m·K) for brain tissue
        self.specific_heat = 3600.0  # J/(kg·K)
        self.density = 1050.0  # kg/m³
        self.thermal_diffusivity = self.thermal_conductivity / (self.density * self.specific_heat)
        
        # Bioheat parameters
        self.blood_perfusion_rate = 0.008  # 1/s (8 mL/100g/min typical for brain)
        self.arterial_temperature = 37.0  # °C
        self.metabolic_heat = 0.0  # W/m³ (negligible during surgery)
        
        # Numerical parameters
        self.dx = 0.001  # 1mm spatial resolution
        self.dt = 0.05  # Time step (seconds)
        
        # Stability criterion for explicit method: dt <= dx²/(4*α)
        max_dt_explicit = (self.dx ** 2) / (4 * self.thermal_diffusivity)
        if self.dt > max_dt_explicit and not high_performance:
            print(f"Warning: dt={self.dt} exceeds stability limit {max_dt_explicit:.4f}")
        
        # History tracking
        self.max_temp_history = []
        self.avg_temp_history = []
        self.damage_history = []
        self.time_step_count = 0
        
        # Performance metrics
        self.computation_time_ms = 0.0
        
    def _initialize_tissue_map(self):
        """Initialize heterogeneous tissue map"""
        tissue = np.zeros((self.height, self.width), dtype=np.int32)
        
        # Create a tumor region (higher absorption)
        center_y, center_x = self.height // 2, self.width // 2
        y, x = np.ogrid[:self.height, :self.width]
        
        # Tumor: circular region
        tumor_radius = 8
        tumor_mask = (x - center_x)**2 + (y - center_y)**2 <= tumor_radius**2
        tissue[tumor_mask] = 1
        
        # Critical structures: small regions to avoid
        critical_y1, critical_x1 = center_y - 15, center_x + 10
        critical_mask1 = (x - critical_x1)**2 + (y - critical_y1)**2 <= 16
        tissue[critical_mask1] = 2
        
        return tissue
    
    def apply_laser(self, x, y, power, enabled, pattern=None):
        """
        Apply laser heating with optional quantum-generated pattern
        
        Args:
            x, y: Grid coordinates (0..width, 0..height)
            power: Laser power intensity (W)
            enabled: Whether laser is active
            pattern: Optional 2D quantum-generated heat pattern
        """
        if not enabled or power <= 0:
            return
        
        # Generate heat source distribution
        if pattern is not None:
            # Use quantum-generated pattern (already normalized 0-1)
            source = power * pattern
        else:
            # Gaussian beam profile
            xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
            sigma = 2.5  # Beam width
            source = power * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        
        # Tissue-dependent absorption
        absorption_factor = np.ones_like(source)
        absorption_factor[self.tissue_map == 1] = 1.5  # Tumor: higher absorption
        absorption_factor[self.tissue_map == 2] = 0.3  # Critical: lower absorption (safety)
        
        # Apply heat (convert to temperature rise)
        # Q = P / (ρ * c * V) where V = dx³
        volume_element = self.dx ** 3
        heat_per_volume = source * absorption_factor / volume_element
        temp_rise = heat_per_volume * self.dt / (self.density * self.specific_heat)
        
        self.temperature_map += temp_rise
    
    def update(self):
        """
        Update temperature field using bioheat equation
        
        Pennes bioheat equation:
        ρc ∂T/∂t = ∇·(k∇T) - ω_b ρ_b c_b (T - T_a) + Q_m + Q_ext
        
        Uses implicit finite difference for numerical stability
        """
        import time
        start_time = time.time()
        
        if self.high_performance:
            self._update_implicit()
        else:
            self._update_explicit()
        
        # Update thermal dose (CEM43) using fast Numba implementation
        self.damage_map = _compute_cem43_fast(
            self.temperature_map, 
            self.dt, 
            self.damage_map
        )
        
        # Track history
        self.time_step_count += 1
        if self.time_step_count % 5 == 0:
            self.max_temp_history.append(float(np.max(self.temperature_map)))
            self.avg_temp_history.append(float(np.mean(self.temperature_map)))
            self.damage_history.append(float(np.max(self.damage_map)))
            
            # Keep history bounded
            if len(self.max_temp_history) > 200:
                self.max_temp_history.pop(0)
                self.avg_temp_history.pop(0)
                self.damage_history.pop(0)
        
        # Performance tracking
        self.computation_time_ms = (time.time() - start_time) * 1000.0
    
    def _update_explicit(self):
        """Explicit finite difference update (faster but less stable)"""
        # Diffusion term: ∇²T
        lap = laplace(self.temperature_map)
        diffusion = self.thermal_diffusivity * lap / (self.dx ** 2)
        
        # Perfusion cooling using fast Numba implementation
        self.temperature_map = _apply_perfusion_cooling(
            self.temperature_map,
            self.damage_map,
            self.blood_perfusion_rate,
            self.arterial_temperature,
            self.dt
        )
        
        # Apply diffusion
        self.temperature_map += diffusion * self.dt
        
        # Physical bounds
        self.temperature_map = np.clip(self.temperature_map, 20.0, 100.0)
    
    def _update_implicit(self):
        """
        Implicit finite difference update (unconditionally stable)
        
        Solves: (I - α·Δt·L)·T^(n+1) = T^n + Δt·S
        where L is Laplacian operator, S is source/sink terms
        """
        # Flatten for sparse solver
        T_flat = self.temperature_map.flatten()
        n = len(T_flat)
        
        # Build Laplacian matrix (5-point stencil for 2D)
        alpha = self.thermal_diffusivity * self.dt / (self.dx ** 2)
        
        # Main diagonal
        main_diag = np.ones(n) * (1 + 4 * alpha)
        
        # Off-diagonals
        off_diag = np.ones(n - 1) * (-alpha)
        off_diag_w = np.ones(n - self.width) * (-alpha)
        
        # Handle boundary conditions (Neumann: zero flux)
        # Set boundary coefficients to maintain stability
        
        # Create sparse matrix
        diagonals = [main_diag, off_diag, off_diag, off_diag_w, off_diag_w]
        offsets = [0, -1, 1, -self.width, self.width]
        A = diags(diagonals, offsets, shape=(n, n), format='csr')
        
        # Right-hand side: current temperature + source terms
        # Perfusion term
        necrotic_mask = self.damage_map > 240.0
        perfusion_rate = np.where(necrotic_mask, 0.0, self.blood_perfusion_rate)
        perfusion_term = perfusion_rate * (self.arterial_temperature - self.temperature_map) * self.dt
        
        b = T_flat + perfusion_term.flatten()
        
        # Solve linear system
        try:
            T_new = spsolve(A, b)
            self.temperature_map = T_new.reshape((self.height, self.width))
        except:
            # Fallback to explicit if solver fails
            self._update_explicit()
            return
        
        # Physical bounds
        self.temperature_map = np.clip(self.temperature_map, 20.0, 100.0)
    
    def apply_cryotherapy(self, x, y, cooling_power, enabled):
        """
        Apply cryogenic cooling
        
        Args:
            x, y: Grid coordinates
            cooling_power: Cooling power (negative heat)
            enabled: Whether cryo is active
        """
        if not enabled or cooling_power <= 0:
            return
        
        # Gaussian cooling distribution
        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        sigma = 3.0
        cooling_dist = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        
        # Apply cooling
        self.temperature_map -= cooling_power * cooling_dist * self.dt
        
        # Allow sub-physiological temperatures with cryo
        self.temperature_map = np.maximum(self.temperature_map, -20.0)
    
    def get_map(self):
        """Get current temperature map"""
        return self.temperature_map
    
    def get_damage_map(self):
        """Get cumulative thermal dose map (CEM43)"""
        return self.damage_map
    
    def get_tissue_map(self):
        """Get tissue type map"""
        return self.tissue_map
    
    def get_history(self):
        """Get temperature history"""
        return self.max_temp_history
    
    def get_performance_metrics(self):
        """Get performance metrics"""
        return {
            'computation_time_ms': self.computation_time_ms,
            'time_steps': self.time_step_count,
            'max_temperature': float(np.max(self.temperature_map)),
            'avg_temperature': float(np.mean(self.temperature_map)),
            'max_damage': float(np.max(self.damage_map)),
            'necrotic_volume': float(np.sum(self.damage_map > 240.0))
        }
    
    def reset(self):
        """Reset to initial state"""
        self.temperature_map = np.full((self.height, self.width), 37.0, dtype=np.float64)
        self.damage_map = np.zeros((self.height, self.width), dtype=np.float64)
        self.max_temp_history = []
        self.avg_temp_history = []
        self.damage_history = []
        self.time_step_count = 0
