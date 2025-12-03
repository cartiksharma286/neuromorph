"""
Material Properties and Selection for 3D Printed Catheters

Database of biocompatible materials with mechanical analysis
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class MaterialType(Enum):
    """Available biocompatible materials"""
    POLYURETHANE = "polyurethane"
    SILICONE = "silicone"
    PTFE = "ptfe"
    PEBAX = "pebax"
    NYLON = "nylon"


@dataclass
class MaterialProperties:
    """Physical and mechanical properties of materials"""
    name: str
    density: float  # kg/m³
    youngs_modulus: float  # GPa
    yield_strength: float  # MPa
    elongation_at_break: float  # %
    shore_hardness: str  # e.g., "80A", "55D"
    biocompatibility: float  # 0-1 score
    cost_per_gram: float  # USD
    friction_coefficient: float
    
    
class BiomaterialDatabase:
    """Database of medical-grade polymers for catheters"""
    
    MATERIALS = {
        MaterialType.POLYURETHANE: MaterialProperties(
            name="Medical-grade Polyurethane",
            density=1200,
            youngs_modulus=0.05,  # 50 MPa
            yield_strength=35,
            elongation_at_break=500,
            shore_hardness="80A",
            biocompatibility=0.95,
            cost_per_gram=0.15,
            friction_coefficient=0.25
        ),
        MaterialType.SILICONE: MaterialProperties(
            name="Medical Silicone (Platinum-cured)",
            density=1100,
            youngs_modulus=0.003,  # 3 MPa (very soft)
            yield_strength=7,
            elongation_at_break=600,
            shore_hardness="40A",
            biocompatibility=0.98,
            cost_per_gram=0.35,
            friction_coefficient=0.40
        ),
        MaterialType.PTFE: MaterialProperties(
            name="PTFE (Teflon)",
            density=2200,
            youngs_modulus=0.5,  # 500 MPa
            yield_strength=25,
            elongation_at_break=300,
            shore_hardness="55D",
            biocompatibility=0.99,
            cost_per_gram=0.50,
            friction_coefficient=0.04  # Very low friction
        ),
        MaterialType.PEBAX: MaterialProperties(
            name="Pebax (Polyether Block Amide)",
            density=1010,
            youngs_modulus=0.15,  # 150 MPa
            yield_strength=45,
            elongation_at_break=400,
            shore_hardness="55D",
            biocompatibility=0.96,
            cost_per_gram=0.40,
            friction_coefficient=0.20
        ),
        MaterialType.NYLON: MaterialProperties(
            name="Nylon 12 (Medical-grade)",
            density=1010,
            youngs_modulus=1.6,  # 1600 MPa
            yield_strength=50,
            elongation_at_break=300,
            shore_hardness="75D",
            biocompatibility=0.90,
            cost_per_gram=0.10,
            friction_coefficient=0.30
        )
    }
    
    @classmethod
    def get_material(cls, material_type: MaterialType) -> MaterialProperties:
        """Retrieve material properties"""
        return cls.MATERIALS[material_type]
    
    @classmethod
    def list_materials(cls) -> List[str]:
        """List all available materials"""
        return [m.value for m in MaterialType]


class MechanicalAnalyzer:
    """Analyzes mechanical properties of catheter designs"""
    
    @staticmethod
    def compute_bending_stiffness(
        outer_diameter: float,  # mm
        inner_diameter: float,  # mm
        youngs_modulus: float  # GPa
    ) -> float:
        """
        Compute bending stiffness EI
        
        I = π/64 * (D_o^4 - D_i^4) for hollow cylinder
        """
        # Convert to meters
        D_o = outer_diameter / 1000
        D_i = inner_diameter / 1000
        
        # Second moment of area
        I = (np.pi / 64) * (D_o ** 4 - D_i ** 4)
        
        # Bending stiffness
        E = youngs_modulus * 1e9  # GPa to Pa
        EI = E * I
        
        return EI  # N·m²
    
    @staticmethod
    def compute_kink_resistance(
        outer_diameter: float,  # mm
        wall_thickness: float,  # mm
        youngs_modulus: float  # GPa
    ) -> float:
        """
        Estimate kink resistance (critical curvature)
        
        Higher value = more resistant to kinking
        """
        # Simplified model: ratio of thickness to diameter
        thickness_ratio = wall_thickness / outer_diameter
        
        # Material stiffness factor
        stiffness_factor = youngs_modulus / 0.1  # Normalized to typical value
        
        kink_resistance = thickness_ratio * stiffness_factor
        
        return kink_resistance
    
    @staticmethod
    def compute_flexibility_index(
        bending_stiffness: float,
        length: float  # mm
    ) -> float:
        """
        Compute flexibility index (0-1, higher = more flexible)
        
        Based on inverse of bending stiffness normalized by length
        """
        # Normalize by typical values
        typical_EI = 1e-6  # N·m²
        length_m = length / 1000
        
        flexibility = 1 / (1 + (bending_stiffness / typical_EI) * length_m)
        
        return np.clip(flexibility, 0, 1)
    
    @staticmethod
    def estimate_pushability(
        wall_thickness: float,  # mm
        youngs_modulus: float  # GPa
    ) -> float:
        """
        Estimate pushability (0-1, higher = easier to advance)
        
        Balance between stiffness and thickness
        """
        # Thicker wall and stiffer material = better pushability
        pushability = (wall_thickness / 0.5) * (youngs_modulus / 0.5)
        
        return np.clip(pushability / 2, 0, 1)


class CompositeDesigner:
    """Design multi-material composite catheters"""
    
    @staticmethod
    def create_layered_design(
        inner_diameter: float,  # mm
        total_wall_thickness: float,  # mm
        layer_materials: List[MaterialType],
        layer_thickness_ratios: List[float]
    ) -> Dict:
        """
        Design multi-layer catheter wall
        
        Example: Inner PTFE liner + Pebax reinforcement + Polyurethane outer
        """
        if len(layer_materials) != len(layer_thickness_ratios):
            raise ValueError("Materials and thickness ratios must match")
        
        if not np.isclose(sum(layer_thickness_ratios), 1.0):
            raise ValueError("Thickness ratios must sum to 1.0")
        
        layers = []
        current_radius = inner_diameter / 2
        
        for material_type, thickness_ratio in zip(layer_materials, layer_thickness_ratios):
            layer_thickness = total_wall_thickness * thickness_ratio
            outer_radius = current_radius + layer_thickness
            
            material = BiomaterialDatabase.get_material(material_type)
            
            layers.append({
                'material': material.name,
                'material_type': material_type,
                'inner_radius': current_radius,
                'outer_radius': outer_radius,
                'thickness': layer_thickness,
                'properties': material
            })
            
            current_radius = outer_radius
        
        # Compute composite properties
        composite_modulus = CompositeDesigner._compute_effective_modulus(layers)
        composite_density = CompositeDesigner._compute_effective_density(layers, total_wall_thickness)
        
        return {
            'layers': layers,
            'inner_diameter': inner_diameter,
            'outer_diameter': current_radius * 2,
            'total_wall_thickness': total_wall_thickness,
            'effective_youngs_modulus': composite_modulus,
            'effective_density': composite_density
        }
    
    @staticmethod
    def _compute_effective_modulus(layers: List[Dict]) -> float:
        """Compute effective Young's modulus for composite"""
        # Volume-weighted average (simplified)
        total_volume = sum(
            np.pi * (l['outer_radius']**2 - l['inner_radius']**2)
            for l in layers
        )
        
        weighted_modulus = sum(
            l['properties'].youngs_modulus * 
            np.pi * (l['outer_radius']**2 - l['inner_radius']**2)
            for l in layers
        )
        
        return weighted_modulus / total_volume
    
    @staticmethod
    def _compute_effective_density(layers: List[Dict], total_thickness: float) -> float:
        """Compute effective density for composite"""
        # Thickness-weighted average
        weighted_density = sum(
            l['properties'].density * (l['thickness'] / total_thickness)
            for l in layers
        )
        
        return weighted_density


class MaterialOptimizer:
    """Quantum-inspired material selection optimizer"""
    
    @staticmethod
    def optimize_composition(
        target_flexibility: float,
        target_pushability: float,
        target_biocompatibility: float,
        cost_weight: float = 0.1
    ) -> Dict[MaterialType, float]:
        """
        Optimize material composition for target properties
        
        Returns: Dictionary of material types to mixing ratios
        """
        # Define optimization objective
        materials = list(MaterialType)
        n_materials = len(materials)
        
        # Initialize with uniform distribution
        weights = np.ones(n_materials) / n_materials
        
        # Simple gradient-free optimization (placeholder for quantum)
        best_weights = weights.copy()
        best_score = -np.inf
        
        for iteration in range(100):
            # Evaluate current composition
            score = MaterialOptimizer._evaluate_composition(
                weights,
                materials,
                target_flexibility,
                target_pushability,
                target_biocompatibility,
                cost_weight
            )
            
            if score > best_score:
                best_score = score
                best_weights = weights.copy()
            
            # Perturb weights
            perturbation = np.random.randn(n_materials) * 0.1
            weights = weights + perturbation
            
            # Normalize and clip
            weights = np.clip(weights, 0, None)
            weights = weights / weights.sum()
        
        # Convert to dictionary
        composition = {
            material: weight
            for material, weight in zip(materials, best_weights)
            if weight > 0.01  # Filter out negligible components
        }
        
        return composition
    
    @staticmethod
    def _evaluate_composition(
        weights: np.ndarray,
        materials: List[MaterialType],
        target_flex: float,
        target_push: float,
        target_bio: float,
        cost_weight: float
    ) -> float:
        """Evaluate fitness of material composition"""
        
        # Compute weighted average properties
        avg_modulus = 0
        avg_bio = 0
        avg_cost = 0
        
        for weight, material_type in zip(weights, materials):
            props = BiomaterialDatabase.get_material(material_type)
            avg_modulus += weight * props.youngs_modulus
            avg_bio += weight * props.biocompatibility
            avg_cost += weight * props.cost_per_gram
        
        # Estimate flexibility and pushability from modulus
        # Lower modulus = higher flexibility
        flexibility = 1 / (1 + avg_modulus)
        pushability = avg_modulus / (1 + avg_modulus)
        
        # Compute fitness
        flex_error = (flexibility - target_flex) ** 2
        push_error = (pushability - target_push) ** 2
        bio_error = (avg_bio - target_bio) ** 2
        
        fitness = -(flex_error + push_error + bio_error + cost_weight * avg_cost)
        
        return fitness


if __name__ == '__main__':
    print("Material Properties and Optimization")
    print("=" * 60)
    
    # Show available materials
    print("\nAvailable Biocompatible Materials:")
    for material_type in MaterialType:
        props = BiomaterialDatabase.get_material(material_type)
        print(f"\n{props.name}:")
        print(f"  Density: {props.density} kg/m³")
        print(f"  Young's Modulus: {props.youngs_modulus*1000:.1f} MPa")
        print(f"  Yield Strength: {props.yield_strength} MPa")
        print(f"  Elongation: {props.elongation_at_break}%")
        print(f"  Shore Hardness: {props.shore_hardness}")
        print(f"  Biocompatibility: {props.biocompatibility*100:.1f}%")
        print(f"  Cost: ${props.cost_per_gram:.2f}/g")
    
    # Mechanical analysis
    print("\n" + "=" * 60)
    print("\nMechanical Analysis Example:")
    
    outer_dia = 4.5  # mm
    inner_dia = 3.8  # mm
    wall_thickness = 0.35  # mm
    length = 1000  # mm
    
    material = BiomaterialDatabase.get_material(MaterialType.POLYURETHANE)
    
    analyzer = MechanicalAnalyzer()
    
    EI = analyzer.compute_bending_stiffness(outer_dia, inner_dia, material.youngs_modulus)
    kink = analyzer.compute_kink_resistance(outer_dia, wall_thickness, material.youngs_modulus)
    flexibility = analyzer.compute_flexibility_index(EI, length)
    pushability = analyzer.estimate_pushability(wall_thickness, material.youngs_modulus)
    
    print(f"\nFor {material.name} catheter:")
    print(f"  Outer diameter: {outer_dia} mm")
    print(f"  Inner diameter: {inner_dia} mm")
    print(f"  Wall thickness: {wall_thickness} mm")
    print(f"  Bending stiffness: {EI:.2e} N·m²")
    print(f"  Kink resistance: {kink:.3f}")
    print(f"  Flexibility index: {flexibility:.3f}")
    print(f"  Pushability: {pushability:.3f}")
    
    # Composite design
    print("\n" + "=" * 60)
    print("\nComposite Catheter Design:")
    
    composite = CompositeDesigner.create_layered_design(
        inner_diameter=3.8,
        total_wall_thickness=0.7,
        layer_materials=[
            MaterialType.PTFE,      # Inner liner (low friction)
            MaterialType.PEBAX,     # Middle (strength)
            MaterialType.POLYURETHANE  # Outer (flexibility)
        ],
        layer_thickness_ratios=[0.2, 0.5, 0.3]
    )
    
    print(f"\nComposite Structure:")
    for i, layer in enumerate(composite['layers']):
        print(f"\nLayer {i+1}: {layer['material']}")
        print(f"  Thickness: {layer['thickness']:.3f} mm")
        print(f"  Inner radius: {layer['inner_radius']:.3f} mm")
        print(f"  Outer radius: {layer['outer_radius']:.3f} mm")
    
    print(f"\nComposite Properties:")
    print(f"  Effective Young's Modulus: {composite['effective_youngs_modulus']*1000:.1f} MPa")
    print(f"  Effective Density: {composite['effective_density']:.1f} kg/m³")
    
    # Material optimization
    print("\n" + "=" * 60)
    print("\nOptimized Material Composition:")
    
    optimal_composition = MaterialOptimizer.optimize_composition(
        target_flexibility=0.7,
        target_pushability=0.6,
        target_biocompatibility=0.95,
        cost_weight=0.1
    )
    
    print(f"\nOptimal composition (by weight):")
    for material_type, ratio in optimal_composition.items():
        print(f"  {material_type.value}: {ratio*100:.1f}%")
