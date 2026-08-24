"""
Visualization Engine - Generate stunning 3D models and visualizations
"""

import numpy as np
from typing import Dict, List
import json
import uuid

class VisualizationEngine:
    """Generate 3D models and visualizations for surgical planning"""
    
    def __init__(self):
        self.color_scheme = {
            'implant': '#FF6B9D',  # Rose pink
            'chamber_anchor': '#2E86AB',  # Deep blue
            'chamber_support': '#A23B72',  # Purple
            'chamber_load': '#F18F01',  # Orange
            'chamber_hydro': '#C73E1D',  # Red
            'tissue': '#E8F4F8',  # Light blue
            'defect': '#8B0000'  # Dark red
        }
    
    def generate_3d_model(self, design: Dict) -> Dict:
        """Generate 3D model data for implant design"""
        
        dimensions = design['dimensions']
        length = dimensions['length_mm']
        width = dimensions['width_mm']
        thickness = dimensions['thickness_mm']
        
        # Generate vertices for implant (rectangular box with detail)
        vertices = self._generate_implant_geometry(length, width, thickness)
        
        # Generate faces
        faces = self._generate_implant_faces(len(vertices))
        
        # Add surface features based on pore size
        features = self._generate_surface_features(
            design['pore_size_microns'],
            length, width
        )
        
        model = {
            'model_id': f"model_{uuid.uuid4().hex[:8]}",
            'design_id': design['id'],
            'material': design['material'],
            'shape_profile': design['shape_profile'],
            'vertices': vertices,
            'faces': faces,
            'features': features,
            'color': self.color_scheme['implant'],
            'properties': {
                'wireframe': False,
                'opacity': 0.9,
                'shininess': 30,
                'metalness': 0.3
            },
            'metadata': {
                'total_vertices': len(vertices),
                'total_faces': len(faces),
                'complexity': 'medium'
            }
        }
        
        return model
    
    def _generate_implant_geometry(self, length: float, width: float, 
                                   thickness: float) -> List[List[float]]:
        """Generate 3D vertices for implant geometry"""
        vertices = []
        
        # Base rectangle
        x_positions = [0, length, length, 0, 0, length, length, 0]
        y_positions = [0, 0, width, width, 0, 0, width, width]
        z_positions = [0, 0, 0, 0, thickness, thickness, thickness, thickness]
        
        for x, y, z in zip(x_positions, y_positions, z_positions):
            vertices.append([x, y, z])
        
        # Add central feature vertices for anatomical profile
        vertices.extend([
            [length/2, width/2, -0.5],  # Central anchor point
            [length*0.25, width*0.5, thickness/2],  # Support points
            [length*0.75, width*0.5, thickness/2],
            [length*0.5, width*0.25, thickness/2],
            [length*0.5, width*0.75, thickness/2]
        ])
        
        return vertices
    
    def _generate_implant_faces(self, num_vertices: int) -> List[List[int]]:
        """Generate face indices for implant"""
        faces = []
        
        # Base faces (rectangular box)
        faces.extend([
            [0, 1, 2, 3],  # Bottom
            [4, 5, 6, 7],  # Top
            [0, 1, 5, 4],  # Side 1
            [1, 2, 6, 5],  # Side 2
            [2, 3, 7, 6],  # Side 3
            [3, 0, 4, 7]   # Side 4
        ])
        
        # Central feature faces
        if num_vertices > 8:
            faces.extend([
                [8, 9, 10],
                [8, 10, 11],
                [8, 11, 12],
                [8, 12, 9]
            ])
        
        return faces
    
    def _generate_surface_features(self, pore_size: float, 
                                   length: float, width: float) -> Dict:
        """Generate surface features based on porosity"""
        
        # Calculate pore density based on size
        pore_density = 1000 / (pore_size / 100)  # Higher density for smaller pores
        
        features = {
            'pore_pattern': 'distributed',
            'pore_size_microns': pore_size,
            'estimated_pore_count': int(pore_density * (length * width) / 1000),
            'surface_roughness': self._calculate_roughness(pore_size),
            'texture_map': 'procedural_generated'
        }
        
        return features
    
    def _calculate_roughness(self, pore_size: float) -> float:
        """Calculate surface roughness from pore size"""
        return min(3.0, pore_size / 50)  # Ra roughness in micrometers
    
    def generate_chamber_model(self, chamber: Dict, index: int) -> Dict:
        """Generate 3D model for support chamber"""
        
        chamber_type = chamber['type']
        color_key = f"chamber_{chamber_type}"
        color = self.color_scheme.get(color_key, '#999999')
        
        # Generate sphere/capsule for chamber
        position = chamber['position']
        diameter = chamber['dimensions']['diameter_mm']
        depth = chamber['dimensions']['depth_mm']
        
        # Generate sphere vertices
        vertices = self._generate_sphere_vertices(
            position['x_mm'], position['y_mm'], position['z_mm'],
            diameter/2
        )
        
        model = {
            'model_id': f"chamber_model_{index}_{uuid.uuid4().hex[:8]}",
            'chamber_id': chamber['chamber_id'],
            'chamber_type': chamber_type,
            'position': position,
            'vertices': vertices,
            'color': color,
            'diameter_mm': diameter,
            'depth_mm': depth,
            'fill_material': chamber['fill_material'],
            'properties': {
                'opacity': 0.7,
                'glossiness': 0.4,
                'shininess': 20
            },
            'pressure_kpa': chamber.get('pressure_optimal_kpa', 0),
            'load_capacity_n': chamber.get('load_capacity_n', 0)
        }
        
        return model
    
    def _generate_sphere_vertices(self, center_x: float, center_y: float, 
                                  center_z: float, radius: float,
                                  segments: int = 12) -> List[List[float]]:
        """Generate vertices for a sphere"""
        vertices = []
        
        for i in range(segments):
            for j in range(segments):
                theta = 2 * np.pi * i / segments
                phi = np.pi * j / segments
                
                x = center_x + radius * np.sin(phi) * np.cos(theta)
                y = center_y + radius * np.sin(phi) * np.sin(theta)
                z = center_z + radius * np.cos(phi)
                
                vertices.append([float(x), float(y), float(z)])
        
        return vertices
    
    def generate_anatomy_visualization(self, defect_location: Dict) -> Dict:
        """Generate background anatomy visualization"""
        
        anatomy = {
            'pelvic_floor': self._generate_pelvic_floor_mesh(),
            'ligament_structure': self._generate_ligament_network(),
            'nerve_pathways': self._generate_nerve_visualization(),
            'blood_vessels': self._generate_vascular_network()
        }
        
        return anatomy
    
    def _generate_pelvic_floor_mesh(self) -> Dict:
        """Generate pelvic floor muscle mesh"""
        return {
            'components': ['levator_ani', 'puborectalis', 'iliococcygeus'],
            'color': self.color_scheme['tissue'],
            'opacity': 0.4,
            'description': 'Main pelvic floor muscle complex'
        }
    
    def _generate_ligament_network(self) -> Dict:
        """Generate supporting ligament structure"""
        return {
            'ligaments': ['uterosacral', 'cardinal', 'round', 'broad'],
            'color': '#FFD700',  # Gold
            'opacity': 0.5,
            'description': 'Support ligament network'
        }
    
    def _generate_nerve_visualization(self) -> Dict:
        """Generate nerve pathway visualization"""
        return {
            'primary_nerve': 'pudendal_nerve',
            'plexus': 'sacral_plexus_s2_s4',
            'color': '#00FF00',  # Green
            'opacity': 0.6,
            'description': 'Neural pathways'
        }
    
    def _generate_vascular_network(self) -> Dict:
        """Generate blood vessel visualization"""
        return {
            'arteries': ['internal_pudendal', 'superior_rectal'],
            'veins': ['internal_pudendal_veins', 'rectal_venous_plexus'],
            'color': '#FF0000',  # Red
            'opacity': 0.5,
            'description': 'Vascular network'
        }
    
    def generate_comparison_view(self, designs: List[Dict]) -> Dict:
        """Generate side-by-side comparison visualization"""
        
        comparison = {
            'view_id': f"comparison_{uuid.uuid4().hex[:8]}",
            'num_designs': len(designs),
            'layouts': ['side_by_side', 'carousel', 'grid_2x2', 'grid_3x3'],
            'selected_layout': 'grid_2x2',
            'designs': []
        }
        
        for i, design in enumerate(designs[:4]):  # Show top 4
            comparison['designs'].append({
                'position': i,
                'design_id': design['id'],
                'material': design['material'],
                'score': design.get('rank_score', 0),
                'thumbnail': f"design_{i}_thumb.png"
            })
        
        return comparison
    
    def generate_animation_sequence(self, implant: Dict, chambers: List[Dict]) -> Dict:
        """Generate surgical animation sequence"""
        
        animation = {
            'animation_id': f"anim_{uuid.uuid4().hex[:8]}",
            'total_frames': 240,
            'fps': 30,
            'duration_seconds': 8,
            'stages': [
                {
                    'stage': 0,
                    'name': 'Anatomical View',
                    'frame_range': [0, 30],
                    'description': 'Show defect in context of pelvic anatomy'
                },
                {
                    'stage': 1,
                    'name': 'Implant Positioning',
                    'frame_range': [31, 80],
                    'description': 'Animate implant placement'
                },
                {
                    'stage': 2,
                    'name': 'Chamber Installation',
                    'frame_range': [81, 150],
                    'description': 'Show chamber distribution and anchoring'
                },
                {
                    'stage': 3,
                    'name': 'Final Integration',
                    'frame_range': [151, 240],
                    'description': 'Visualize tissue integration over time'
                }
            ],
            'camera_path': 'orbital_rotation',
            'lighting': 'professional_medical'
        }
        
        return animation
