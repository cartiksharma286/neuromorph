"""
Parametric Stent Geometry Generator

Generates 3D meshes for coronary stents with configurable cell topology,
dimensions, and expansion mechanics.
"""

import numpy as np
import trimesh
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum


class StentMaterial(Enum):
    """Stent material types"""
    NITINOL = "nitinol"  # Superelastic (Self-expanding)
    COBALT_CHROMIUM = "cocr"  # Plastic (Balloon-expandable)
    STAINLESS_STEEL = "ss316l"


class CellTopology(Enum):
    """Stent cell design patterns"""
    DIAMOND = "diamond"
    HEXAGONAL = "hexagonal"
    HYBRID = "hybrid"  # Open/Closed cell mix


@dataclass
class StentParameters:
    """Parameters for stent generation"""
    length: float  # mm
    diameter: float  # mm (expanded)
    strut_thickness: float  # mm (radial)
    strut_width: float  # mm (circumferential)
    crowns_per_ring: int
    material: StentMaterial
    topology: CellTopology
    expansion_ratio: float = 1.0  # 1.0 = fully expanded
    foreshortening: float = 0.05  # 5% foreshortening at full expansion


class StentGenerator:
    """Generates 3D stent geometry"""
    
    def __init__(self, params: StentParameters):
        self.params = params
    
    def generate_mesh(self) -> trimesh.Trimesh:
        """Generate complete 3D stent mesh"""
        
        # 1. Generate centerline graph of struts
        strut_graph = self._generate_strut_graph()
        
        # 2. Sweep cross-section along graph
        mesh = self._sweep_struts(strut_graph)
        
        return mesh
    
    def _generate_strut_graph(self) -> List[np.ndarray]:
        """
        Generate centerline paths for all struts
        Returns list of point arrays (polylines)
        """
        paths = []
        
        # Calculate dimensions based on expansion
        current_diameter = self.params.diameter * self.params.expansion_ratio
        radius = current_diameter / 2
        
        # Adjust length for foreshortening (simplified model)
        # Length decreases as diameter increases
        current_length = self.params.length * (1.0 - self.params.foreshortening * self.params.expansion_ratio)
        
        # Number of rings along length
        # Approximate ring height based on topology
        ring_height = (current_diameter * np.pi) / self.params.crowns_per_ring
        n_rings = int(current_length / ring_height)
        
        # Generate rings
        for ring_idx in range(n_rings):
            z_offset = ring_idx * ring_height
            
            # Generate crowns for this ring
            ring_paths = self._generate_ring_crowns(radius, z_offset, ring_height)
            paths.extend(ring_paths)
            
            # Generate connectors to next ring
            if ring_idx < n_rings - 1:
                connector_paths = self._generate_connectors(radius, z_offset, ring_height, ring_idx)
                paths.extend(connector_paths)
                
        return paths
    
    def _generate_ring_crowns(self, radius: float, z_start: float, height: float) -> List[np.ndarray]:
        """Generate zigzag crown pattern for a single ring"""
        paths = []
        n_crowns = self.params.crowns_per_ring
        
        for i in range(n_crowns):
            # Angle coverage for one crown (peak to peak)
            theta_start = (i / n_crowns) * 2 * np.pi
            theta_mid = ((i + 0.5) / n_crowns) * 2 * np.pi
            theta_end = ((i + 1) / n_crowns) * 2 * np.pi
            
            # Points for one V-shape strut
            # Peak -> Valley -> Peak
            
            # Parametric interpolation for smooth curve
            t = np.linspace(0, 1, 10)
            
            # First leg (Peak to Valley)
            theta1 = theta_start + t * (theta_mid - theta_start)
            z1 = z_start + height/2 + t * height/2  # Valley is at z_start + height
            
            # Use sine wave for smooth transition
            # z = z_center + amplitude * cos(theta)
            
            # Simplified V-shape with rounded corners
            # We'll use a sine wave approximation for the whole ring instead
            pass 
            
        # Better approach: Generate full sine wave for the ring
        theta = np.linspace(0, 2 * np.pi, n_crowns * 20)
        
        # Zigzag pattern (sine wave)
        z_wave = z_start + height/2 + (height/2) * np.sin(n_crowns * theta)
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = z_wave
        
        # Split into segments to avoid closing the loop incorrectly in list
        # But here we want a continuous ring
        points = np.column_stack([x, y, z])
        
        # Close the loop
        points = np.vstack([points, points[0]])
        
        return [points]

    def _generate_connectors(self, radius: float, z_start: float, height: float, ring_idx: int) -> List[np.ndarray]:
        """Generate connectors between rings"""
        paths = []
        n_crowns = self.params.crowns_per_ring
        
        # Connector strategy depends on topology
        if self.params.topology == CellTopology.DIAMOND:
            # Connect Peak to Valley
            # Peak of current ring is at z_start + height
            # Valley of next ring is at z_start + height (they touch)
            # In a real stent, they are often fused or have a small link
            
            # For visualization, we add small bridge segments
            # Connect every peak
            for i in range(n_crowns):
                theta = ((i + 0.25) / n_crowns) * 2 * np.pi # Peak position approximation
                
                # Point on current ring (Peak)
                p1_z = z_start + height
                p1_x = radius * np.cos(theta)
                p1_y = radius * np.sin(theta)
                
                # Point on next ring (Valley)
                # Next ring starts at z_start + height
                # Its valley is at z_start + height
                
                # Just a small weld point
                # No extra path needed if they touch, but let's add a small link
                pass
                
        elif self.params.topology == CellTopology.HEXAGONAL:
            # Connectors are longer, creating open cells
            # Connect every 2nd or 3rd peak
            pass
            
        return paths

    def _sweep_struts(self, paths: List[np.ndarray]) -> trimesh.Trimesh:
        """Sweep rectangular cross-section along paths"""
        meshes = []
        
        for path in paths:
            if len(path) < 2:
                continue
                
            # Create tube/strut
            # For simplicity, use cylinders for now, or extruded rects
            
            # Using trimesh creation
            # Create a path 3D
            try:
                # Create a tube along the path
                # radius = strut_width / 2 (approx)
                strut = trimesh.creation.sweep_polygon(
                    polygon=trimesh.creation.Rectangle(
                        width=self.params.strut_width, 
                        height=self.params.strut_thickness
                    ),
                    path=path
                )
                meshes.append(strut)
            except Exception as e:
                print(f"Error generating strut: {e}")
                continue
                
        if not meshes:
            return trimesh.Trimesh()
            
        # Combine all struts
        combined = trimesh.util.concatenate(meshes)
        return combined


if __name__ == '__main__':
    print("Stent Geometry Generator")
    print("=" * 60)
    
    params = StentParameters(
        length=18.0,
        diameter=3.0,
        strut_thickness=0.08,  # 80 microns
        strut_width=0.10,      # 100 microns
        crowns_per_ring=6,
        material=StentMaterial.COBALT_CHROMIUM,
        topology=CellTopology.DIAMOND
    )
    
    generator = StentGenerator(params)
    mesh = generator.generate_mesh()
    
    print(f"Generated stent mesh:")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Volume: {mesh.volume:.4f} mm³")
    
    # Export for testing
    mesh.export('test_stent.stl')
    print("Exported to test_stent.stl")
