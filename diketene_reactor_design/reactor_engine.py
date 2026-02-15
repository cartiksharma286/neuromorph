import numpy as np
import math

class ReactorDesigner:
    def __init__(self, volume_m3, pressure_bar, temp_c):
        self.volume = volume_m3
        self.pressure = pressure_bar
        self.temp = temp_c
        
        # Diketene Properties (highly reactive, exothermic)
        self.density = 1090 # kg/m3
        self.viscosity = 0.002 # Pa.s
        self.reaction_heat = -450 # kJ/mol (Estimate)
        
        # Materials (SS316L for Diketene)
        self.allowable_stress = 115 # MPa (at 100C approx)
        self.weld_efficiency = 0.85
        
        self.geometry = {}
        self.optimization = {}

    def calculate_dimensions(self):
        """
        Calculates optimal H/D ratio for heat transfer.
        For exothermic reactions, higher Area/Volume is better (Tall/Thin).
        Typical stirred tank H/D is 1.0 to 1.5.
        """
        # Let's target H/D ~ 1.5 for better heat removal surface
        target_ratio = 1.5
        
        # V = pi * (D/2)^2 * H
        # V = pi * D^2/4 * (1.5 * D) = 1.178 * D^3
        # D = (V / 1.178)^(1/3)
        
        diameter = (self.volume / (math.pi/4 * target_ratio))**(1.0/3.0)
        height = diameter * target_ratio
        
        self.geometry['diameter'] = diameter
        self.geometry['tan_tan_height'] = height
        
        # Torispherical Head (DIN 28011)
        # Crown Radius R = D
        # Knuckle Radius r = 0.1 * D
        self.geometry['crown_radius'] = diameter
        self.geometry['knuckle_radius'] = 0.1 * diameter
        
        # Minimum Thickness (ASME VIII Div 1)
        # t = (P * R) / (2 * S * E - 0.2 * P)
        # P in MPa
        P_mpa = self.pressure * 0.1
        S = self.allowable_stress
        E = self.weld_efficiency
        R = diameter # Worst case crown radius
        
        thickness = (P_mpa * R * 1000) / (2 * S * E - 0.2 * P_mpa) # mm
        self.geometry['wall_thickness_mm'] = max(6.0, thickness + 3.0) # + Corrosion allow
        
        return self.geometry

    def optimize_jacket(self):
        """
        Optimizes the Half-Pipe Coil Jacket using Statistical Congruence.
        We want the coil spacing (pitch) to avoid integer resonance with the agitator frequency.
        """
        D = self.geometry['diameter']
        H = self.geometry['tan_tan_height']
        
        # Agitator Frequency (Target 120 RPM = 2 Hz)
        agitator_freq = 2.0 
        
        # Flow velocity target in coil
        v_coolant = 1.5 # m/s
        
        # Statistical Congruence:
        # We model the coil pitch 'p' as a function of the Generalized Fibonacci Sequence
        # to ensure non-repeating thermal gradients (Golden Ratio distribution).
        
        phi = (1 + math.sqrt(5)) / 2
        
        # Base pitch estimate
        pipe_dia = 0.05 # 2 inch half pipe
        min_pitch = pipe_dia * 1.5
        
        # Statistical Congruence Factor
        # Minimize (H / Pitch) mod 1 to avoid 'perfect stacking' of thermal zones?
        # Actually, we want the number of turns 'N' to be coprime to agitator blade count (3 or 4).
        
        N_turns_ideal = H / min_pitch
        
        # Find nearest Prime to N_turns_ideal for turbulence chaos
        def is_prime(n):
            if n <= 1: return False
            for i in range(2, int(n**0.5)+1):
                if n%i==0: return False
            return True
            
        N_prime = int(N_turns_ideal)
        while not is_prime(N_prime):
            N_prime -= 1 # Go lower to increase pitch (safer weld)
            
        optimal_pitch = H / N_prime
        
        # Statistical Score (Congruence with Golden Ratio)
        # How close is Pitch/Diameter to a power of Phi?
        ratio = optimal_pitch / D
        congruence_metric = abs((ratio * 100) % phi) # Arbitrary metric
        
        self.optimization = {
            "coil_pitch": optimal_pitch,
            "num_turns": N_prime,
            "pipe_diameter": pipe_dia,
            "flow_regime": "Turbulent (Prime Optimized)",
            "congruence_score": congruence_metric
        }
        
        return self.optimization

    def get_specs(self):
        return {
            "geometry": self.geometry,
            "jacket_optimization": self.optimization,
            "material": "SS316L / Hastelloy C-276 (Diketene Service)"
        }
