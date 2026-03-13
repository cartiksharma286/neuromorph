"""
5G Neural Pathway Guidance System for Surgical Robotics
Implements neural pathway mapping and 5G-based real-time trajectory guidance
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import heapq


class FiveGNeuralPathway:
    """
    Enhanced surgical guidance using 5G neural pathway mapping
    """
    def __init__(self, anatomy_map, tumor_mask, width=128, height=128):
        self.width = width
        self.height = height
        self.anatomy = anatomy_map
        self.tumor_mask = tumor_mask
        self.neural_pathways = None
        self.safe_corridors = None
        self.optimal_trajectory = None
        self.current_idx = 0
        self.completed = False
        self.active = False
        
        self._compute_neural_pathways()
        self._compute_safe_corridors()
        self._plan_optimal_trajectory()
        
    def _compute_neural_pathways(self):
        """
        Compute neural pathway probability map using distance-based weighting
        Neural pathways are regions to avoid during ablation
        """
        # Detect high-intensity regions (neural tissue)
        neural_density = self.anatomy.copy()
        
        # Gaussian convolution for smoothness
        from scipy.ndimage import gaussian_filter
        neural_pathways = gaussian_filter(neural_density, sigma=2.0)
        
        # Normalize
        if np.max(neural_pathways) > 0:
            neural_pathways = neural_pathways / np.max(neural_pathways)
        
        self.neural_pathways = neural_pathways
        
    def _compute_safe_corridors(self):
        """
        Compute safe ablation corridors that minimize neural pathway impact
        Uses distance transform to find regions furthest from critical structures
        """
        # Create danger map (neural pathways + tumor edges)
        danger_map = np.zeros_like(self.anatomy)
        
        # Add neural pathway threat
        danger_map += self.neural_pathways * 0.7
        
        # Add tumor boundaries
        tumor_edges = ndimage.sobel(self.tumor_mask.astype(float))
        danger_map += tumor_edges * 0.3
        
        # Compute distance from danger
        distance_map = distance_transform_edt(danger_map < 0.3)
        
        # Safe corridors are regions with good distance from danger
        safe_corridors = np.zeros_like(self.anatomy)
        safe_corridors[distance_map > 5] = 1.0
        
        # But only within the tumor region + margin
        tumor_margin = ndimage.binary_dilation(self.tumor_mask, iterations=3)
        safe_corridors = safe_corridors * tumor_margin
        
        self.safe_corridors = safe_corridors
        
    def _plan_optimal_trajectory(self):
        """
        Plan an optimal trajectory through safe corridors for complete tumor coverage
        """
        # Get points in safe corridors within tumor
        safe_points = np.argwhere((self.safe_corridors > 0.5) & (self.tumor_mask > 0.5))
        
        if len(safe_points) == 0:
            # Fallback: use tumor region itself
            safe_points = np.argwhere(self.tumor_mask > 0.5)
        
        if len(safe_points) == 0:
            self.optimal_trajectory = []
            return
        
        # Sort for continuous path (snake scan pattern)
        safe_points = safe_points[np.lexsort((safe_points[:, 1], safe_points[:, 0]))]
        
        # Create smooth path by interpolating between scan lines
        trajectory = self._generate_smooth_path(safe_points)
        self.optimal_trajectory = trajectory
        
    def _generate_smooth_path(self, points):
        """Generate smooth trajectory from waypoints using spline interpolation"""
        if len(points) < 2:
            return [tuple(p) for p in points]
        
        # Group by X coordinate (rows)
        rows = {}
        for i, (x, y) in enumerate(points):
            if x not in rows:
                rows[x] = []
            rows[x].append((i, x, y))
        
        trajectory = []
        sorted_x = sorted(rows.keys())
        
        for idx, x in enumerate(sorted_x):
            row_points = sorted(rows[x], key=lambda p: p[2])  # Sort by Y
            
            if idx % 2 == 1:  # Alternate direction for snake pattern
                row_points.reverse()
            
            for _, px, py in row_points:
                trajectory.append((px, py))
        
        return trajectory
    
    def get_next_waypoint(self, current_pos, approach_threshold=0.01):
        """
        Get next waypoint for the surgical robot
        
        Args:
            current_pos: Current robot position (x, y, z)
            approach_threshold: Distance threshold to target for firing laser
            
        Returns:
            (target_x, target_z), laser_should_fire: Next target and laser state
        """
        if not self.active or self.completed:
            return None, False
        
        if len(self.optimal_trajectory) == 0:
            self.completed = True
            return None, False
        
        # Get current target
        tx, tz = self.optimal_trajectory[self.current_idx]
        
        # Convert grid coordinates to robot space
        robot_target_x = (tx / self.width) - 0.5
        robot_target_z = tz / self.height
        
        # Compute distance to target
        dx = current_pos[0] - robot_target_x
        dz = current_pos[2] - robot_target_z
        dist = np.sqrt(dx*dx + dz*dz)
        
        # Check if we've reached the target
        if dist < approach_threshold:
            self.current_idx += 1
            if self.current_idx >= len(self.optimal_trajectory):
                self.completed = True
                return None, False
            else:
                # Move to next target
                return self.get_next_waypoint(current_pos, approach_threshold)
        
        # Determine if laser should fire
        # Fire laser when close enough to target (dwell zone)
        laser_should_fire = dist < 0.02 and dist > 0.001
        
        return (robot_target_x, robot_target_z), laser_should_fire
    
    def get_visualization_data(self):
        """Return visualization data for UI"""
        return {
            'neural_pathways': self.neural_pathways,
            'safe_corridors': self.safe_corridors,
            'trajectory': np.array(self.optimal_trajectory) if self.optimal_trajectory else np.array([]),
            'progress': self.current_idx / max(len(self.optimal_trajectory), 1),
        }


class LaserDeliveryOptimizer:
    """
    Optimizes laser delivery parameters for maximum ablation efficacy
    """
    def __init__(self):
        self.baseline_power = 60.0  # watts
        self.dwell_time = 0.1  # seconds per waypoint
        self.beam_radius = 2.5  # mm
        
    def compute_optimal_power(self, tissue_temperature, target_temperature=65.0):
        """
        Compute optimal laser power based on current tissue temperature
        Adaptive power scaling for consistent ablation
        """
        if tissue_temperature < 40:
            return self.baseline_power * 1.2  # Full power when cold
        elif tissue_temperature < 50:
            return self.baseline_power
        elif tissue_temperature < 60:
            return self.baseline_power * 0.8  # Back off a bit
        else:
            return self.baseline_power * 0.5  # Minimal power when hot
    
    def should_continuously_ablate(self, distance_to_target, velocity=0.05):
        """
        Determine if laser should fire continuously during motion
        vs only at waypoint dwell
        """
        if distance_to_target > 0.03:  # Far from target
            return False
        if distance_to_target < 0.001:  # At target
            return True
        # In approach zone - fire if moving slowly
        return velocity < 0.02
    
    def compute_dwell_time(self, tissue_density, desired_necrosis_volume=1.0):
        """
        Compute optimal dwell time at each waypoint
        Based on tissue properties and ablation goals
        """
        # Higher density tissue needs more dwell time
        base_dwell = self.dwell_time
        density_factor = 1.0 + (tissue_density - 0.5) * 0.4
        
        # Scale based on necrosis target
        necrosis_factor = desired_necrosis_volume / 1.0
        
        return base_dwell * density_factor * necrosis_factor
