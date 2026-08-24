"""
Optimized Visualization Engine - High-Performance 3D Model Generation
Generates production-quality 3D meshes for medical implants and chambers
"""

import numpy as np
from typing import Dict, List, Tuple
import uuid
from dataclasses import dataclass

@dataclass
class Mesh:
    """Optimized mesh data structure"""
    vertices: np.ndarray  # N x 3 array
    faces: np.ndarray     # M x 3 array of vertex indices
    normals: np.ndarray   # N x 3 array
    metadata: Dict
    
    def to_dict(self):
        """Convert to JSON-serializable format"""
        return {
            'vertices': self.vertices.tolist(),
            'faces': self.faces.tolist(),
            'normals': self.normals.tolist(),
            'metadata': self.metadata
        }

class VisualizationEngineOptimized:
    """High-performance 3D visualization and mesh generation"""
    
    def __init__(self):
        self.color_scheme = {
            'implant': '#FF6B9D',
            'chamber_anchor': '#2E86AB',
            'chamber_support': '#A23B72',
            'chamber_load': '#F18F01',
            'chamber_hydro': '#C73E1D',
            'tissue': '#E8F4F8',
            'defect': '#8B0000'
        }
    
    def generate_implant_mesh_detailed(self, design: Dict) -> Dict:
        """Generate detailed 3D mesh for implant"""
        dims = design['dimensions']
        L = dims['length_mm']
        W = dims['width_mm']
        T = dims['thickness_mm']
        
        # Generate vertices for anatomically-shaped implant
        vertices = self._generate_implant_vertices_optimized(L, W, T, design)
        
        # Generate optimized face topology
        faces = self._generate_implant_faces_optimized(len(vertices))
        
        # Calculate vertex normals for smooth shading
        normals = self._calculate_normals(vertices, faces)
        
        # Add surface features based on material properties
        vertices = self._apply_material_surface_features(
            vertices, design.get('pore_size_microns', 100), L, W
        )
        
        mesh = Mesh(vertices, faces, normals, {
            'design_id': design['id'],
            'material': design['material'],
            'shape_profile': design['shape_profile'],
            'dimensions': dims,
            'biocompatibility_score': design.get('biocompatibility_score', 0.85),
            'color': self.color_scheme['implant'],
            'opacity': 0.9,
            'shininess': 30
        })
        
        return {
            'mesh': mesh.to_dict(),
            'metadata': mesh.metadata,
            'model_id': f"implant_{uuid.uuid4().hex[:8]}"
        }
    
    def _generate_implant_vertices_optimized(self, L: float, W: float, 
                                            T: float, design: Dict) -> np.ndarray:
        """Generate vertices for implant with smooth anatomical shape"""
        
        # Use parametric approach for smooth surfaces
        nu, nv = 20, 16  # Resolution parameters
        u = np.linspace(0, 1, nu)
        v = np.linspace(0, 1, nv)
        
        vertices = []
        
        # Bottom surface (primary contact)
        for vi in v:
            for ui in u:
                x = ui * L
                y = vi * W
                z = -T * 0.3 * np.sin(ui * np.pi) * np.sin(vi * np.pi)
                vertices.append([x, y, z])
        
        # Middle surface (structural)
        for vi in v:
            for ui in u:
                x = ui * L
                y = vi * W
                z = T * 0.5 * np.cos(ui * np.pi * 0.5) * np.cos(vi * np.pi * 0.5)
                vertices.append([x, y, z])
        
        # Top surface (contact)
        for vi in v:
            for ui in u:
                x = ui * L
                y = vi * W
                z = T * (1.0 + 0.2 * np.sin(ui * np.pi * 2) * np.sin(vi * np.pi))
                vertices.append([x, y, z])
        
        # Edge reinforcement vertices
        for i in range(nu):
            t = i / (nu - 1)
            # Reinforced edges
            vertices.append([t * L, 0, T * 0.6])
            vertices.append([t * L, W, T * 0.6])
            vertices.append([0, t * W, T * 0.6])
            vertices.append([L, t * W, T * 0.6])
        
        return np.array(vertices, dtype=np.float32)
    
    def _generate_implant_faces_optimized(self, num_vertices: int) -> np.ndarray:
        """Generate optimized face connectivity with proper topology"""
        faces = []
        nu, nv = 20, 16
        
        # Generate faces for parametric surfaces
        for layer in range(2):  # Bottom, middle, top
            offset = layer * nu * nv
            for j in range(nv - 1):
                for i in range(nu - 1):
                    v0 = offset + j * nu + i
                    v1 = offset + j * nu + i + 1
                    v2 = offset + (j + 1) * nu + i + 1
                    v3 = offset + (j + 1) * nu + i
                    
                    faces.append([v0, v1, v2])
                    faces.append([v0, v2, v3])
        
        # Add edge reinforcement faces
        edge_start = 3 * nu * nv
        for i in range(nu - 1):
            # Reinforce edges with additional triangles
            v_edge = edge_start + i * 4
            faces.append([v_edge, v_edge + 1, v_edge + 2])
            faces.append([v_edge + 1, v_edge + 3, v_edge + 2])
        
        return np.array(faces, dtype=np.uint32)
    
    def _calculate_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Calculate smooth vertex normals for shading"""
        normals = np.zeros_like(vertices)
        
        for face in faces:
            v0, v1, v2 = vertices[face]
            
            # Calculate face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            
            # Accumulate to vertex normals
            for vertex_idx in face:
                normals[vertex_idx] += face_normal
        
        # Normalize
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0] = 1  # Avoid division by zero
        normals = normals / lengths
        
        return normals
    
    def _apply_material_surface_features(self, vertices: np.ndarray, 
                                        pore_size: float, L: float, W: float) -> np.ndarray:
        """Apply surface features based on material properties"""
        # Add subtle micro-texturing based on pore size
        pore_amplitude = min(0.1, pore_size / 1000)
        
        # Generate pseudo-random surface features
        np.random.seed(42)  # Reproducible
        for i in range(len(vertices)):
            if vertices[i, 2] > 0:  # Top surface
                # Add pore-based surface variation
                freq = 2 * np.pi * pore_size / max(L, W)
                vertices[i, 2] += pore_amplitude * np.sin(
                    freq * vertices[i, 0] + np.random.rand() * 0.1
                ) * np.sin(freq * vertices[i, 1] + np.random.rand() * 0.1)
        
        return vertices
    
    def generate_chamber_mesh_detailed(self, chamber: Dict, index: int) -> Dict:
        """Generate detailed 3D mesh for support chamber"""
        
        chamber_type = chamber['type']
        color_key = f"chamber_{chamber_type}"
        color = self.color_scheme.get(color_key, '#999999')
        
        position = chamber['position']
        diameter = chamber['dimensions']['diameter_mm']
        depth = chamber['dimensions']['depth_mm']
        
        # Generate icosphere for chamber (optimized sphere)
        vertices, faces = self._generate_icosphere(
            center=np.array([position['x_mm'], position['y_mm'], position['z_mm']]),
            radius=diameter / 2,
            subdivisions=3
        )
        
        normals = self._calculate_normals(vertices, faces)
        
        mesh = Mesh(vertices, faces, normals, {
            'chamber_id': chamber['chamber_id'],
            'chamber_type': chamber_type,
            'position': position,
            'diameter_mm': diameter,
            'depth_mm': depth,
            'fill_material': chamber['fill_material'],
            'pressure_kpa': chamber.get('pressure_optimal_kpa', 0),
            'load_capacity_n': chamber.get('load_capacity_n', 0),
            'color': color,
            'opacity': 0.7
        })
        
        return {
            'mesh': mesh.to_dict(),
            'metadata': mesh.metadata,
            'model_id': f"chamber_{index}_{uuid.uuid4().hex[:8]}"
        }
    
    def _generate_icosphere(self, center: np.ndarray, radius: float, 
                           subdivisions: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Generate icosphere using golden ratio approach (fast, efficient)"""
        
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2
        
        # Initial icosahedron vertices
        initial_vertices = np.array([
            [-1,  phi, -1],
            [1,   phi, -1],
            [-1,  phi,  1],
            [1,   phi,  1],
            [-phi, -1, -1],
            [phi,  -1, -1],
            [-phi, -1,  1],
            [phi,  -1,  1],
            [-1, -1, -phi],
            [1,  -1, -phi],
            [-1, -1,  phi],
            [1,  -1,  phi],
        ], dtype=np.float32)
        
        # Normalize
        initial_vertices /= np.linalg.norm(initial_vertices, axis=1, keepdims=True)
        initial_vertices *= radius
        initial_vertices += center
        
        # Initial faces
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ], dtype=np.uint32)
        
        vertices = initial_vertices.copy()
        
        # Subdivide
        for _ in range(subdivisions):
            new_faces = []
            for face in faces:
                v0, v1, v2 = vertices[face]
                
                # Calculate midpoints
                mid01 = (v0 + v1) / 2
                mid12 = (v1 + v2) / 2
                mid20 = (v2 + v0) / 2
                
                # Normalize and scale to sphere
                for mid in [mid01, mid12, mid20]:
                    norm = np.linalg.norm(mid - center)
                    if norm > 0:
                        mid[:] = (mid - center) / norm * radius + center
                
                # Add new vertices
                idx0, idx1, idx2 = len(vertices), len(vertices) + 1, len(vertices) + 2
                vertices = np.vstack([vertices, [mid01, mid12, mid20]])
                
                # Create 4 new faces
                new_faces.extend([
                    [face[0], idx0, idx2],
                    [idx0, face[1], idx1],
                    [idx2, idx1, face[2]],
                    [idx0, idx1, idx2]
                ])
            
            faces = np.array(new_faces, dtype=np.uint32)
        
        return vertices, faces
    
    def mesh_to_stl(self, mesh_data: Dict) -> str:
        """Convert mesh to ASCII STL format string"""
        vertices = np.array(mesh_data['vertices'])
        faces = np.array(mesh_data['faces'])
        
        stl_str = "solid implant\n"
        
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            
            # Calculate normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm
            
            stl_str += f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n"
            stl_str += "    outer loop\n"
            for v in [v0, v1, v2]:
                stl_str += f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n"
            stl_str += "    endloop\n"
            stl_str += "  endfacet\n"
        
        stl_str += "endsolid implant"
        return stl_str
    
    def generate_comparison_view(self, designs: List[Dict]) -> Dict:
        """Generate optimized comparison visualization"""
        return {
            'view_id': f"comparison_{uuid.uuid4().hex[:8]}",
            'num_designs': len(designs),
            'layouts': ['side_by_side', 'carousel', 'grid_2x2'],
            'selected_layout': 'grid_2x2',
            'designs': [{
                'position': i,
                'design_id': d['id'],
                'material': d['material'],
                'score': d.get('rank_score', 0),
                'model_id': f"design_{i}"
            } for i, d in enumerate(designs[:4])]
        }
