"""
RF Coil Designer with Generative AI
====================================

This module implements parametric RF coil design and generative algorithms
for creating optimized coil geometries.

Features:
- Parametric coil models (solenoid, planar spiral, Helmholtz)
- Generative design using optimization algorithms
- Automatic parameter tuning for target frequency/inductance
- 2D/3D visualization
- Schematic generation
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import json


@dataclass
class CoilParameters:
    """Parameters defining an RF coil design."""
    coil_type: str  # 'solenoid', 'planar_spiral', 'helmholtz'
    wire_diameter: float  # mm
    turns: int
    diameter: float  # mm (outer diameter)
    length: float  # mm (for solenoid) or spacing (for planar)
    substrate_thickness: float = 0.0  # mm (for planar coils)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'coil_type': self.coil_type,
            'wire_diameter': self.wire_diameter,
            'turns': self.turns,
            'diameter': self.diameter,
            'length': self.length,
            'substrate_thickness': self.substrate_thickness
        }


class RFCoilDesigner:
    """
    Main class for RF coil design and analysis.
    
    Provides methods to calculate electrical properties and generate
    optimized coil designs.
    """
    
    def __init__(self):
        """Initialize the designer."""
        # Physical constants
        self.mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
        self.epsilon_0 = 8.854e-12    # Permittivity of free space (F/m)
        
    def calculate_solenoid_inductance(self, params: CoilParameters) -> float:
        """
        Calculate inductance of a solenoid coil using Wheeler's formula.
        
        L = (N^2 * r^2) / (9*r + 10*l)  (in microhenries)
        
        Args:
            params: Coil parameters
            
        Returns:
            Inductance in henries (H)
        """
        N = params.turns
        r = params.diameter / 2 / 1000  # Convert mm to meters
        l = params.length / 1000  # Convert mm to meters
        
        if l == 0:
            # Single-layer formula
            L_uH = (N**2 * (r * 1000)**2) / (9 * (r * 1000) + 10 * 0.1)
        else:
            # Wheeler's formula (result in microhenries)
            L_uH = (N**2 * (r * 1000)**2) / (9 * (r * 1000) + 10 * (l * 1000))
        
        return L_uH * 1e-6  # Convert to henries
    
    def calculate_planar_spiral_inductance(self, params: CoilParameters) -> float:
        """
        Calculate inductance of a planar spiral coil.
        
        Uses modified Wheeler formula for planar spirals.
        
        Args:
            params: Coil parameters
            
        Returns:
            Inductance in henries (H)
        """
        N = params.turns
        d_outer = params.diameter / 1000  # Convert to meters
        spacing = params.length / 1000  # Spacing between turns
        
        # Estimate inner diameter
        d_inner = d_outer - 2 * N * spacing
        if d_inner < 0:
            d_inner = d_outer * 0.3  # Minimum inner diameter
        
        # Average diameter
        d_avg = (d_outer + d_inner) / 2
        
        # Modified Wheeler formula for planar spirals (result in henries)
        L = (self.mu_0 * N**2 * d_avg**2) / (8 * d_avg + 11 * (d_outer - d_inner))
        
        return L
    
    def calculate_inductance(self, params: CoilParameters) -> float:
        """
        Calculate inductance based on coil type.
        
        Args:
            params: Coil parameters
            
        Returns:
            Inductance in henries (H)
        """
        if params.coil_type == 'solenoid':
            return self.calculate_solenoid_inductance(params)
        elif params.coil_type == 'planar_spiral':
            return self.calculate_planar_spiral_inductance(params)
        elif params.coil_type == 'helmholtz':
            # Helmholtz is two solenoids
            return 2 * self.calculate_solenoid_inductance(params)
        else:
            raise ValueError(f"Unknown coil type: {params.coil_type}")
    
    def calculate_resonant_frequency(self, L: float, C: float) -> float:
        """
        Calculate resonant frequency of LC circuit.
        
        f = 1 / (2π√(LC))
        
        Args:
            L: Inductance in henries
            C: Capacitance in farads
            
        Returns:
            Frequency in Hz
        """
        return 1 / (2 * np.pi * np.sqrt(L * C))
    
    def calculate_required_capacitance(self, L: float, f: float) -> float:
        """
        Calculate required capacitance for target frequency.
        
        C = 1 / (4π²f²L)
        
        Args:
            L: Inductance in henries
            f: Target frequency in Hz
            
        Returns:
            Capacitance in farads
        """
        return 1 / (4 * np.pi**2 * f**2 * L)
    
    def calculate_quality_factor(self, params: CoilParameters, 
                                 frequency: float,
                                 wire_resistivity: float = 1.68e-8) -> float:
        """
        Estimate quality factor (Q) of the coil.
        
        Q = ωL / R
        
        Args:
            params: Coil parameters
            frequency: Operating frequency in Hz
            wire_resistivity: Resistivity of wire material (Ω·m), default for copper
            
        Returns:
            Quality factor (dimensionless)
        """
        L = self.calculate_inductance(params)
        
        # Calculate wire length
        if params.coil_type == 'solenoid':
            length_per_turn = np.pi * params.diameter / 1000
            total_length = length_per_turn * params.turns
        elif params.coil_type == 'planar_spiral':
            avg_diameter = params.diameter / 1000
            total_length = np.pi * avg_diameter * params.turns
        else:
            total_length = np.pi * params.diameter / 1000 * params.turns
        
        # Calculate resistance
        wire_area = np.pi * (params.wire_diameter / 2000)**2
        R_dc = wire_resistivity * total_length / wire_area
        
        # Account for skin effect at high frequency
        skin_depth = np.sqrt(wire_resistivity / (np.pi * frequency * self.mu_0))
        if params.wire_diameter / 2000 > 2 * skin_depth:
            # Approximate AC resistance increase
            R_ac = R_dc * (params.wire_diameter / 2000) / (2 * skin_depth)
        else:
            R_ac = R_dc
        
        # Quality factor
        omega = 2 * np.pi * frequency
        Q = omega * L / R_ac
        
        return Q
    
    def design_for_frequency(self, target_freq: float,
                            coil_type: str = 'solenoid',
                            constraints: Optional[Dict] = None) -> CoilParameters:
        """
        Generate coil design for a target frequency using optimization.
        
        Args:
            target_freq: Target resonant frequency in Hz
            coil_type: Type of coil to design
            constraints: Optional design constraints (max_diameter, max_turns, etc.)
            
        Returns:
            Optimized coil parameters
        """
        from scipy.optimize import differential_evolution
        
        if constraints is None:
            constraints = {}
        
        max_diameter = constraints.get('max_diameter', 100)  # mm
        min_diameter = constraints.get('min_diameter', 10)
        max_turns = constraints.get('max_turns', 50)
        min_turns = constraints.get('min_turns', 3)
        max_length = constraints.get('max_length', 100)  # mm
        wire_diameter = constraints.get('wire_diameter', 1.0)  # mm
        target_capacitance = constraints.get('capacitance', 100e-12)  # F
        
        def objective(x):
            """Objective function to minimize."""
            diameter, turns, length = x
            
            params = CoilParameters(
                coil_type=coil_type,
                wire_diameter=wire_diameter,
                turns=int(turns),
                diameter=diameter,
                length=length
            )
            
            L = self.calculate_inductance(params)
            f_resonant = self.calculate_resonant_frequency(L, target_capacitance)
            
            # Minimize frequency error
            freq_error = abs(f_resonant - target_freq) / target_freq
            
            # Prefer higher Q factor
            Q = self.calculate_quality_factor(params, target_freq)
            Q_penalty = -0.01 * Q if Q > 0 else 1000
            
            return freq_error + Q_penalty
        
        # Parameter bounds: [diameter, turns, length]
        bounds = [
            (min_diameter, max_diameter),
            (min_turns, max_turns),
            (1, max_length)
        ]
        
        # Optimize
        result = differential_evolution(objective, bounds, seed=42, maxiter=100)
        
        # Create final parameters
        diameter, turns, length = result.x
        params = CoilParameters(
            coil_type=coil_type,
            wire_diameter=wire_diameter,
            turns=int(turns),
            diameter=diameter,
            length=length
        )
        
        return params


class GenerativeCoilDesigner:
    """
    Generative design system for RF coils.
    
    Uses evolutionary algorithms and parametric generation to create
    novel coil designs optimized for specific applications.
    """
    
    def __init__(self, designer: RFCoilDesigner):
        """
        Initialize generative designer.
        
        Args:
            designer: Base RF coil designer instance
        """
        self.designer = designer
        self.population = []
        
    def generate_random_design(self, coil_type: str,
                              constraints: Dict) -> CoilParameters:
        """
        Generate a random coil design within constraints.
        
        Args:
            coil_type: Type of coil
            constraints: Design constraints
            
        Returns:
            Random coil parameters
        """
        params = CoilParameters(
            coil_type=coil_type,
            wire_diameter=constraints.get('wire_diameter', 
                                         np.random.uniform(0.5, 2.0)),
            turns=np.random.randint(constraints.get('min_turns', 5),
                                   constraints.get('max_turns', 30)),
            diameter=np.random.uniform(constraints.get('min_diameter', 10),
                                      constraints.get('max_diameter', 100)),
            length=np.random.uniform(1, constraints.get('max_length', 50))
        )
        return params
    
    def evaluate_fitness(self, params: CoilParameters,
                        target_freq: float,
                        target_inductance: Optional[float] = None,
                        capacitance: float = 100e-12) -> float:
        """
        Evaluate fitness of a coil design.
        
        Args:
            params: Coil parameters
            target_freq: Target frequency in Hz
            target_inductance: Optional target inductance
            capacitance: Tuning capacitance
            
        Returns:
            Fitness score (lower is better)
        """
        L = self.designer.calculate_inductance(params)
        f_resonant = self.designer.calculate_resonant_frequency(L, capacitance)
        Q = self.designer.calculate_quality_factor(params, target_freq)
        
        # Frequency accuracy (most important)
        freq_error = abs(f_resonant - target_freq) / target_freq
        
        # Inductance target (if specified)
        if target_inductance is not None:
            inductance_error = abs(L - target_inductance) / target_inductance
        else:
            inductance_error = 0
        
        # Prefer higher Q
        Q_score = 1.0 / (1.0 + Q)
        
        # Weighted fitness
        fitness = (10 * freq_error + 
                  2 * inductance_error + 
                  0.5 * Q_score)
        
        return fitness
    
    def evolve_design(self, target_freq: float,
                     coil_type: str = 'solenoid',
                     constraints: Optional[Dict] = None,
                     population_size: int = 50,
                     generations: int = 20) -> CoilParameters:
        """
        Evolve optimal coil design using genetic algorithm.
        
        Args:
            target_freq: Target frequency in Hz
            coil_type: Type of coil
            constraints: Design constraints
            population_size: Number of designs in population
            generations: Number of evolution iterations
            
        Returns:
            Best evolved design
        """
        if constraints is None:
            constraints = {}
        
        # Initialize population
        population = [self.generate_random_design(coil_type, constraints)
                     for _ in range(population_size)]
        
        capacitance = constraints.get('capacitance', 100e-12)
        
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(p, target_freq, 
                                                   capacitance=capacitance)
                            for p in population]
            
            # Sort by fitness
            sorted_pop = [p for _, p in sorted(zip(fitness_scores, population))]
            
            # Keep top 50%
            survivors = sorted_pop[:population_size // 2]
            
            # Generate offspring through crossover and mutation
            offspring = []
            while len(offspring) < population_size // 2:
                # Select two parents
                parent1 = survivors[np.random.randint(len(survivors))]
                parent2 = survivors[np.random.randint(len(survivors))]
                
                # Crossover
                child = CoilParameters(
                    coil_type=coil_type,
                    wire_diameter=parent1.wire_diameter if np.random.rand() > 0.5 
                                 else parent2.wire_diameter,
                    turns=parent1.turns if np.random.rand() > 0.5 else parent2.turns,
                    diameter=parent1.diameter if np.random.rand() > 0.5 
                            else parent2.diameter,
                    length=parent1.length if np.random.rand() > 0.5 else parent2.length
                )
                
                # Mutation (10% chance)
                if np.random.rand() < 0.1:
                    child.diameter *= np.random.uniform(0.9, 1.1)
                    child.diameter = np.clip(child.diameter,
                                            constraints.get('min_diameter', 10),
                                            constraints.get('max_diameter', 100))
                
                if np.random.rand() < 0.1:
                    child.turns = int(child.turns * np.random.uniform(0.9, 1.1))
                    child.turns = np.clip(child.turns,
                                         constraints.get('min_turns', 3),
                                         constraints.get('max_turns', 50))
                
                offspring.append(child)
            
            # New population
            population = survivors + offspring
        
        # Return best design
        final_fitness = [self.evaluate_fitness(p, target_freq, 
                                               capacitance=capacitance)
                        for p in population]
        best_idx = np.argmin(final_fitness)
        
        return population[best_idx]
    
    def generate_design_variations(self, base_params: CoilParameters,
                                   n_variations: int = 5) -> List[CoilParameters]:
        """
        Generate variations of a base design.
        
        Args:
            base_params: Base coil design
            n_variations: Number of variations to generate
            
        Returns:
            List of varied designs
        """
        variations = []
        
        for i in range(n_variations):
            # Apply random variations
            variation = CoilParameters(
                coil_type=base_params.coil_type,
                wire_diameter=base_params.wire_diameter * np.random.uniform(0.8, 1.2),
                turns=int(base_params.turns * np.random.uniform(0.8, 1.2)),
                diameter=base_params.diameter * np.random.uniform(0.9, 1.1),
                length=base_params.length * np.random.uniform(0.9, 1.1)
            )
            variations.append(variation)
        
        return variations
