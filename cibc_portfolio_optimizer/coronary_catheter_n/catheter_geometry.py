"""
Parametric Catheter Geometry Generation

Generates 3D meshes for catheters based on optimized design parameters
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import trimesh
from quantum_catheter_optimizer import DesignParameters, PatientConstraints


@dataclass
class CatheterMesh:
    """Container for catheter 3D mesh data"""
    vertices: np.ndarray
    faces: np.ndarray
    mesh: trimesh.Trimesh
    metadata: dict


class LumenProfileGenerator:
    """Generates optimized internal lumen geometry"""
    
    @staticmethod
    def generate_circular_profile(
        inner_diameter: float,
        n_points: int = 32
    ) -> np.ndarray:
        """Generate circular cross-section"""
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        radius = inner_diameter / 2
        
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        
        return np.column_stack([x, y])
    
    @staticmethod
    def generate_optimized_profile(
        inner_diameter: float,
        optimization_factor: float = 0.0,
        n_points: int = 32
    ) -> np.ndarray:
        """
        Generate flow-optimized cross-section
        
        optimization_factor: 0 = circular, 1 = maximally optimized
        """
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        base_radius = inner_diameter / 2
        
        # Apply optimization: slight elliptical deformation for laminar flow
        radial_variation = 1 + optimization_factor * 0.1 * np.sin(2 * angles)
        
        x = base_radius * radial_variation * np.cos(angles)
        y = base_radius * radial_variation * np.sin(angles)
        
        return np.column_stack([x, y])


class CatheterBodyGenerator:
    """Generates main catheter body geometry"""
    
    def __init__(self, design: DesignParameters, constraints: PatientConstraints):
        self.design = design
        self.constraints = constraints
        
    def generate_centerline(self, n_segments: int = 100) -> np.ndarray:
        """
        Generate catheter centerline path
        
        Incorporates vessel curvature for realistic geometry
        """
        length = self.constraints.required_length
        curvature = self.constraints.vessel_curvature
        
        # Create parametric curve
        t = np.linspace(0, 1, n_segments)
        
        # Base linear path
        z = t * length
        
        # Apply curvature (helical path)
        if curvature > 0:
            radius_of_curvature = 1 / curvature
            angle = t * length / radius_of_curvature
            
            x = radius_of_curvature * (1 - np.cos(angle))
            y = radius_of_curvature * np.sin(angle) * 0.5  # Reduced y variation
        else:
            x = np.zeros_like(t)
            y = np.zeros_like(t)
        
        return np.column_stack([x, y, z])
    
    def generate_body_mesh(
        self,
        n_segments: int = 100,
        n_radial: int = 32
    ) -> CatheterMesh:
        """Generate full catheter body mesh"""
        
        # Get centerline
        centerline = self.generate_centerline(n_segments)
        
        # Generate cross-sections along centerline
        vertices_list = []
        faces_list = []
        
        for i, point in enumerate(centerline):
            # Interpolate diameter along length (taper at tip)
            t = i / (len(centerline) - 1)
            taper_factor = 1.0 - 0.3 * (t ** 2)  # Reduce diameter toward tip
            
            outer_radius = (self.design.outer_diameter / 2) * taper_factor
            
            # Generate cross-section
            angles = np.linspace(0, 2 * np.pi, n_radial, endpoint=False)
            
            x_local = outer_radius * np.cos(angles)
            y_local = outer_radius * np.sin(angles)
            z_local = np.full(n_radial, point[2])
            
            # Apply global position
            x_global = x_local + point[0]
            y_global = y_local + point[1]
            
            section_vertices = np.column_stack([x_global, y_global, z_local])
            vertices_list.append(section_vertices)
            
            # Generate faces (connect to previous section)
            if i > 0:
                for j in range(n_radial):
                    j_next = (j + 1) % n_radial
                    
                    # Current section indices
                    v0 = i * n_radial + j
                    v1 = i * n_radial + j_next
                    
                    # Previous section indices
                    v2 = (i - 1) * n_radial + j
                    v3 = (i - 1) * n_radial + j_next
                    
                    # Two triangles per quad
                    faces_list.append([v0, v2, v1])
                    faces_list.append([v1, v2, v3])
        
        # Combine all vertices
        vertices = np.vstack(vertices_list)
        faces = np.array(faces_list)
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Add end caps
        mesh = self._add_end_caps(mesh, n_radial, n_segments)
        
        metadata = {
            'type': 'catheter_body',
            'length': self.constraints.required_length,
            'outer_diameter': self.design.outer_diameter,
            'segments': n_segments
        }
        
        return CatheterMesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            mesh=mesh,
            metadata=metadata
        )
    
    def _add_end_caps(
        self,
        mesh: trimesh.Trimesh,
        n_radial: int,
        n_segments: int
    ) -> trimesh.Trimesh:
        """Add closed caps to catheter ends"""
        
        vertices = mesh.vertices.copy()
        faces = mesh.faces.tolist()
        
        # Front cap (proximal end)
        front_center_idx = len(vertices)
        front_center = vertices[:n_radial].mean(axis=0)
        vertices = np.vstack([vertices, front_center])
        
        for i in range(n_radial):
            i_next = (i + 1) % n_radial
            faces.append([front_center_idx, i_next, i])
        
        # Back cap (distal/tip end)
        back_start = (n_segments - 1) * n_radial
        back_center_idx = len(vertices)
        back_center = vertices[back_start:back_start + n_radial].mean(axis=0)
        vertices = np.vstack([vertices, back_center])
        
        for i in range(n_radial):
            i_next = (i + 1) % n_radial
            v0 = back_start + i
            v1 = back_start + i_next
            faces.append([back_center_idx, v0, v1])
        
        return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))


class CatheterTipGenerator:
    """Generates atraumatic tip geometry"""
    
    def __init__(self, design: DesignParameters):
        self.design = design
    
    def generate_tip_mesh(
        self,
        tip_length: float = 5.0,  # mm
        n_segments: int = 20,
        n_radial: int = 32
    ) -> CatheterMesh:
        """
        Generate smooth, atraumatic tip
        
        Uses Bezier curve for smooth taper
        """
        vertices_list = []
        faces_list = []
        
        # Tip taper profile (cubic bezier)
        t = np.linspace(0, 1, n_segments)
        
        # Radius profile: smooth taper from outer_diameter to point
        start_radius = self.design.outer_diameter / 2
        radius_profile = start_radius * (1 - t) ** 2
        
        # Z position
        z_profile = np.linspace(0, tip_length, n_segments)
        
        # Apply tip angle (bend)
        tip_angle_rad = np.radians(self.design.tip_angle)
        x_offset = t ** 2 * tip_length * np.tan(tip_angle_rad)
        
        for i in range(n_segments):
            if radius_profile[i] < 0.01:  # Near tip point
                # Single vertex at tip
                vertices_list.append(np.array([[x_offset[i], 0, z_profile[i]]]))
            else:
                # Full cross-section
                angles = np.linspace(0, 2 * np.pi, n_radial, endpoint=False)
                
                x_local = radius_profile[i] * np.cos(angles) + x_offset[i]
                y_local = radius_profile[i] * np.sin(angles)
                z_local = np.full(n_radial, z_profile[i])
                
                section_vertices = np.column_stack([x_local, y_local, z_local])
                vertices_list.append(section_vertices)
            
            # Generate faces
            if i > 0:
                prev_size = len(vertices_list[i-1])
                curr_size = len(vertices_list[i])
                
                if curr_size == 1:  # Tip point
                    # Connect to tip point
                    tip_idx = sum(len(v) for v in vertices_list[:i])
                    base_idx = tip_idx - prev_size
                    
                    for j in range(prev_size):
                        j_next = (j + 1) % prev_size
                        faces_list.append([tip_idx, base_idx + j, base_idx + j_next])
                else:
                    # Regular quad faces
                    curr_base = sum(len(v) for v in vertices_list[:i])
                    prev_base = curr_base - prev_size
                    
                    for j in range(min(prev_size, curr_size)):
                        j_next = (j + 1) % min(prev_size, curr_size)
                        
                        v0 = curr_base + j
                        v1 = curr_base + j_next
                        v2 = prev_base + j
                        v3 = prev_base + j_next
                        
                        faces_list.append([v0, v2, v1])
                        faces_list.append([v1, v2, v3])
        
        vertices = np.vstack(vertices_list)
        faces = np.array(faces_list)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        metadata = {
            'type': 'catheter_tip',
            'tip_angle': self.design.tip_angle,
            'tip_length': tip_length
        }
        
        return CatheterMesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            mesh=mesh,
            metadata=metadata
        )


class SideHoleGenerator:
    """Generates side drainage holes"""
    
    @staticmethod
    def add_side_holes(
        body_mesh: CatheterMesh,
        hole_positions: List[float],
        hole_diameter: float = 1.0  # mm
    ) -> CatheterMesh:
        """
        Add side holes to catheter body
        
        Note: This is a simplified implementation. Production version
        would use boolean operations for proper hole cutting.
        """
        # For now, mark hole positions in metadata
        # Full implementation would use trimesh.boolean operations
        
        metadata = body_mesh.metadata.copy()
        metadata['side_holes'] = {
            'positions': hole_positions,
            'diameter': hole_diameter,
            'count': len(hole_positions)
        }
        
        return CatheterMesh(
            vertices=body_mesh.vertices,
            faces=body_mesh.faces,
            mesh=body_mesh.mesh,
            metadata=metadata
        )


class CompleteCatheterGenerator:
    """Assembles complete catheter from components"""
    
    def __init__(self, design: DesignParameters, constraints: PatientConstraints):
        self.design = design
        self.constraints = constraints
    
    def generate(self) -> CatheterMesh:
        """Generate complete catheter assembly"""
        
        print("Generating catheter body...")
        body_gen = CatheterBodyGenerator(self.design, self.constraints)
        body_mesh = body_gen.generate_body_mesh(n_segments=100, n_radial=32)
        
        print("Generating catheter tip...")
        tip_gen = CatheterTipGenerator(self.design)
        tip_mesh = tip_gen.generate_tip_mesh()
        
        print("Adding side holes...")
        body_with_holes = SideHoleGenerator.add_side_holes(
            body_mesh,
            self.design.side_holes,
            hole_diameter=self.design.inner_diameter * 0.3
        )
        
        # Combine meshes (tip positioned at distal end)
        tip_offset = np.array([0, 0, self.constraints.required_length])
        tip_vertices = tip_mesh.vertices + tip_offset
        
        combined_vertices = np.vstack([body_with_holes.vertices, tip_vertices])
        combined_faces = np.vstack([
            body_with_holes.faces,
            tip_mesh.faces + len(body_with_holes.vertices)
        ])
        
        combined_mesh = trimesh.Trimesh(
            vertices=combined_vertices,
            faces=combined_faces
        )
        
        # Merge close vertices and fix normals
        combined_mesh.merge_vertices()
        combined_mesh.fix_normals()
        
        metadata = {
            'type': 'complete_catheter',
            'components': ['body', 'tip', 'side_holes'],
            'design_parameters': {
                'outer_diameter': self.design.outer_diameter,
                'inner_diameter': self.design.inner_diameter,
                'wall_thickness': self.design.wall_thickness,
                'tip_angle': self.design.tip_angle,
                'flexibility_index': self.design.flexibility_index,
                'side_holes': len(self.design.side_holes)
            },
            'patient_constraints': {
                'vessel_diameter': self.constraints.vessel_diameter,
                'required_length': self.constraints.required_length
            }
        }
        
        return CatheterMesh(
            vertices=combined_mesh.vertices,
            faces=combined_mesh.faces,
            mesh=combined_mesh,
            metadata=metadata
        )


if __name__ == '__main__':
    from quantum_catheter_optimizer import PatientConstraints, DesignParameters
    
    print("Catheter Geometry Generator")
    print("=" * 60)
    
    # Example design parameters
    design = DesignParameters(
        outer_diameter=4.5,
        inner_diameter=3.8,
        wall_thickness=0.35,
        tip_angle=30.0,
        flexibility_index=0.75,
        side_holes=[200, 400, 600, 800],
        material_composition={'polyurethane': 0.7, 'silicone': 0.2, 'ptfe': 0.1}
    )
    
    constraints = PatientConstraints(
        vessel_diameter=5.5,
        vessel_curvature=0.15,
        required_length=1000.0,
        flow_rate=8.0
    )
    
    print("\nGenerating complete catheter mesh...")
    generator = CompleteCatheterGenerator(design, constraints)
    catheter = generator.generate()
    
    print(f"\nMesh Statistics:")
    print(f"  Vertices: {len(catheter.vertices)}")
    print(f"  Faces: {len(catheter.faces)}")
    print(f"  Watertight: {catheter.mesh.is_watertight}")
    print(f"  Volume: {catheter.mesh.volume:.2f} mm³")
    print(f"  Surface area: {catheter.mesh.area:.2f} mm²")
    
    print(f"\nMetadata:")
    for key, value in catheter.metadata.items():
        print(f"  {key}: {value}")
