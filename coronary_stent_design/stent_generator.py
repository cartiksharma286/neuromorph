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

    def generate_sine_rings(self, amplitude=0.5, crowns=6):
        """Generates a series of sine wave rings with alternating phases."""
        self.crowns = crowns
        self.amplitude = amplitude
        rings = []
        z_step = self.length / self.num_rings
        
        for i in range(self.num_rings):
            z_center = i * z_step
            # Alternate phase for every other ring to allow peak-to-valley connections
            phase_shift = 0 if i % 2 == 0 else np.pi / crowns
            
            theta = np.linspace(0, 2 * np.pi, self.points_per_ring)
            # z = z_center + A * sin(N * theta + phi)
            z_coords = z_center + amplitude * np.sin(crowns * theta + phase_shift)
            
            ring_points = []
            for t, z in zip(theta, z_coords):
                ring_points.append({
                    "theta": t,
                    "z": z,
                    "x": self.radius * np.cos(t),
                    "y": self.radius * np.sin(t)
                })
            rings.append(ring_points)
        
        self.geometry = {"rings": rings, "connectors": []}
        return rings

    def add_connectors(self):
        """Adds longitudinal connectors between adjacent rings."""
        # Connect peaks of ring i to valleys (or matching peaks) of ring i+1
        # With the phase shift, peaks of ring i align with valleys of ring i+1?
        # Let's check: 
        # Ring 0: sin(N*t) -> Peak at pi/(2N)
        # Ring 1: sin(N*t + pi/N) = sin(N*(t + pi/N^2)) ... actually sin(A+B).
        # Shift of pi/crowns shifts the wave by half a period (peak to valley).
        # So peaks align with peaks of the next ring spatially? No, peak is max z.
        # If we shift by pi (half wave), a peak becomes a valley.
        # The goal is often to connect Peak of Ring I to Valley of Ring I+1 for a "closed cell" design 
        # or Peak to Peak for "open cell".
        # Let's perform simple Peak-to-Peak connections (requiring alignment) via a straight strut.
        
        rings = self.geometry["rings"]
        connectors = []
        
        for i in range(len(rings) - 1):
            ring_curr = rings[i]
            ring_next = rings[i+1]
            
            # Find peaks indices
            # sin(kx) has peaks at roughly equal intervals
            # Just take points where z is max locally? 
            # Or use analytical known angles: 
            # Theta_peak = (pi/2 + 2*pi*k) / crowns - phase/crowns
            
            # Simplified: Connect every 'k' points
            # Ideally we connect specific points (e.g. 0, pi/3, ...)
            step = self.points_per_ring // self.crowns
            for j in range(0, self.points_per_ring, step):
                p1 = ring_curr[j]
                p2 = ring_next[j] # Assuming alignment due to phase design or lack thereof
                
                # Connector segment
                connectors.append([p1, p2])
                
        self.geometry["connectors"] = connectors
        return connectors

    def export_to_json(self, filename):
        # Convert numpy types to native python for JSON serialization
        def default(obj):
            if isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            raise TypeError

        with open(filename, 'w') as f:
            json.dump({
                "parameters": {
                    "length": self.length,
                    "diameter": self.diameter,
                    "strut_thickness": self.strut_thickness,
                    "crowns": getattr(self, 'crowns', 6)
                },
                "geometry": self.geometry
            }, f, indent=4, default=default)

    def get_unrolled_points(self):
        """Returns points in (z, arc_length) format for 2D plotting."""
        unrolled_rings = []
        for ring in self.geometry["rings"]:
            ring_pts = []
            for pt in ring:
                ring_pts.append((pt['z'], pt['theta'] * self.radius))
            unrolled_rings.append(ring_pts)
            
        unrolled_connectors = []
        for conn in self.geometry["connectors"]:
            p1, p2 = conn
            c_pts = [
                (p1['z'], p1['theta'] * self.radius),
                (p2['z'], p2['theta'] * self.radius)
            ]
            unrolled_connectors.append(c_pts)
            
        return unrolled_rings, unrolled_connectors
