import numpy as np
import json

class StentGenerator:
    def __init__(self, length, diameter, strut_thickness, num_rings=10, points_per_ring=100):
        self.length = length
        self.diameter = diameter
        self.radius = diameter / 2.0
        self.strut_thickness = strut_thickness
        self.num_rings = num_rings
        self.points_per_ring = points_per_ring
        self.geometry = []

    def generate_sine_rings(self, amplitude=0.5):
        """Generates a series of sine wave rings."""
        rings = []
        z_step = self.length / self.num_rings
        
        for i in range(self.num_rings):
            z_center = i * z_step
            theta = np.linspace(0, 2 * np.pi, self.points_per_ring)
            
            # Sine wave pattern on the cylinder surface
            # Unrolled: x = z, y = r * theta
            # Rolled: x = r * cos(theta), y = r * sin(theta), z = z_center + amp * sin(freq * theta)
            
            # Let's define the pattern in unrolled coordinates (z, theta) first
            # z = z_center + amplitude * sin(6 * theta) # 6 crowns per ring
            
            z_coords = z_center + amplitude * np.sin(6 * theta)
            
            ring_points = []
            for t, z in zip(theta, z_coords):
                ring_points.append({
                    "theta": t,
                    "z": z,
                    "x": self.radius * np.cos(t),
                    "y": self.radius * np.sin(t)
                })
            rings.append(ring_points)
        
        self.geometry = rings
        return rings

    def add_connectors(self):
        """Adds simple straight connectors between rings."""
        # For simplicity, connect peaks to peaks or valleys to valleys
        # This is a placeholder for more complex logic
        pass

    def export_to_json(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                "parameters": {
                    "length": self.length,
                    "diameter": self.diameter,
                    "strut_thickness": self.strut_thickness
                },
                "geometry": self.geometry
            }, f, indent=4)

    def get_unrolled_points(self):
        """Returns points in (z, arc_length) format for 2D plotting."""
        unrolled_data = []
        for ring in self.geometry:
            ring_pts = []
            for pt in ring:
                ring_pts.append((pt['z'], pt['theta'] * self.radius))
            unrolled_data.append(ring_pts)
        return unrolled_data
