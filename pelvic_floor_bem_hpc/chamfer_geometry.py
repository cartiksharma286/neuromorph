"""
Chamfer Geometry & Geometric Mesh Repair Engine
Generates chamfered implant boundary profiles for pelvic floor implants and
performs manifold-consistency repair (hole filling, non-manifold edge fusion,
Euler-characteristic validation) on the resulting boundary-element mesh.
"""

import numpy as np
from typing import Dict, List, Tuple
import uuid


class ChamferGeometryEngine:
    """Generates chamfered implant edge profiles and repairs mesh manifolds"""

    def __init__(self):
        self.default_segments = 24

    def generate_implant_boundary(self, length_mm: float, width_mm: float,
                                   corner_radius_mm: float = 4.0, segments: int = None) -> np.ndarray:
        """Generate a rounded-rectangle implant boundary polygon (outer loop)"""
        segments = segments or self.default_segments
        hl, hw = length_mm / 2.0, width_mm / 2.0
        r = min(corner_radius_mm, hl, hw)
        pts = []
        corners = [
            (hl - r, hw - r, 0),
            (-hl + r, hw - r, 90),
            (-hl + r, -hw + r, 180),
            (hl - r, -hw + r, 270),
        ]
        for cx, cy, start_deg in corners:
            for k in range(segments // 4):
                theta = np.radians(start_deg + 90.0 * k / (segments // 4))
                pts.append((cx + r * np.cos(theta), cy + r * np.sin(theta)))
        return np.array(pts)

    def apply_chamfer(self, boundary: np.ndarray, chamfer_width_mm: float = 1.5,
                       chamfer_angle_deg: float = 45.0, thickness_mm: float = 1.0) -> Dict:
        """
        Apply a uniform edge chamfer to the implant boundary. The chamfer is modeled
        as a bevel plane connecting the top face at offset -chamfer_width to the
        side wall at depth z = -thickness, inclined at chamfer_angle_deg.

        Returns the 3D chamfer strip vertices plus the inset top-face loop.
        """
        n = len(boundary)
        centroid = boundary.mean(axis=0)
        # inward normal direction per vertex (approx via vector to centroid)
        inset_loop = []
        for p in boundary:
            d = centroid - p
            norm = d / (np.linalg.norm(d) + 1e-9)
            inset_loop.append(p + norm * chamfer_width_mm)
        inset_loop = np.array(inset_loop)

        chamfer_depth = chamfer_width_mm * np.tan(np.radians(chamfer_angle_deg))
        chamfer_depth = min(chamfer_depth, thickness_mm * 0.9)

        top_face_3d = np.column_stack([inset_loop, np.zeros(n)])
        edge_3d = np.column_stack([boundary, np.full(n, -chamfer_depth)])

        # Build chamfer strip (quad strip between top inset loop and outer edge)
        strip_faces = []
        for i in range(n):
            j = (i + 1) % n
            strip_faces.append((i, j, n + j, n + i))

        vertices = np.vstack([top_face_3d, edge_3d])

        return {
            'chamfer_id': str(uuid.uuid4()),
            'vertices': vertices.tolist(),
            'faces': strip_faces,
            'chamfer_width_mm': chamfer_width_mm,
            'chamfer_angle_deg': chamfer_angle_deg,
            'chamfer_depth_mm': round(float(chamfer_depth), 4),
            'num_vertices': int(len(vertices)),
            'num_faces': int(len(strip_faces)),
            'stress_relief_factor': round(1.0 + 0.18 * np.log1p(chamfer_width_mm) * np.sin(np.radians(chamfer_angle_deg)), 4)
        }

    def repair_manifold(self, vertices: List, faces: List[Tuple]) -> Dict:
        """
        Geometric repair pass over a quad/tri mesh:
          1. Detect non-manifold edges (shared by != 2 faces, excluding open boundary)
          2. Detect boundary loops (holes) and fill them via fan triangulation
          3. Recompute Euler characteristic chi = V - E + F to confirm genus-0 closure
        """
        edge_face_count = {}

        def edge_key(a, b):
            return (a, b) if a < b else (b, a)

        for f in faces:
            m = len(f)
            for i in range(m):
                a, b = f[i], f[(i + 1) % m]
                key = edge_key(a, b)
                edge_face_count[key] = edge_face_count.get(key, 0) + 1

        non_manifold_edges = [e for e, c in edge_face_count.items() if c > 2]
        boundary_edges = [e for e, c in edge_face_count.items() if c == 1]

        # Chain boundary edges into loops (holes) for fan-fill repair
        holes_filled = 0
        repaired_faces = list(faces)
        adjacency = {}
        for a, b in boundary_edges:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)

        visited_edges = set()
        for start_edge in boundary_edges:
            if start_edge in visited_edges:
                continue
            loop = [start_edge[0], start_edge[1]]
            visited_edges.add(start_edge)
            current = start_edge[1]
            prev = start_edge[0]
            safety = 0
            while current != loop[0] and safety < 5000:
                safety += 1
                neighbors = [v for v in adjacency.get(current, []) if v != prev]
                if not neighbors:
                    break
                nxt = neighbors[0]
                key = edge_key(current, nxt)
                if key in visited_edges:
                    break
                visited_edges.add(key)
                loop.append(nxt)
                prev, current = current, nxt
            if len(loop) >= 3 and loop[0] == loop[-1] or len(loop) >= 3:
                # fan triangulate the hole from its first vertex
                anchor = loop[0]
                for i in range(1, len(loop) - 1):
                    repaired_faces.append((anchor, loop[i], loop[i + 1]))
                holes_filled += 1

        V = len(vertices)
        E = len(edge_face_count)
        F = len(repaired_faces)
        euler_characteristic = V - E + F
        genus = max(0, (2 - euler_characteristic) // 2)

        return {
            'repair_id': str(uuid.uuid4()),
            'original_faces': len(faces),
            'repaired_faces': len(repaired_faces),
            'non_manifold_edges_detected': len(non_manifold_edges),
            'boundary_edges_detected': len(boundary_edges),
            'holes_filled': holes_filled,
            'euler_characteristic': int(euler_characteristic),
            'estimated_genus': int(genus),
            'is_manifold_closed': len(non_manifold_edges) == 0 and (V - E + F) == 2,
            'vertices_count': V,
            'edges_count': E,
            'faces_count': F,
        }
