
import numpy as np
import heapq

class AutomatedGuidanceSystem:
    """
    Automated Tumor Segmentation and Trajectory Planning.
    Identifies tumor regions from MRI anatomy and generates a cover-path.
    """
    def __init__(self, anatomy_map, width=64, height=64):
        self.width = width
        self.height = height
        self.anatomy = anatomy_map
        self.targets = []
        self.current_idx = 0
        self.active = False
        self.completed = False
        
        self.plan_path()
        
    def plan_path(self):
        """
        Segment tumor and create a path.
        """
        # 1. Segment Tumor (Intensity > 0.8)
        # anatomy_map is (width, height)
        # We need indices
        tumor_indices = np.argwhere(self.anatomy > 0.85)
        
        if len(tumor_indices) == 0:
            print("Guidance: No tumor detected.")
            return

        # 2. Sort indices to minimize travel distance (Nearest Neighbor heuristic)
        # Start from center of mass or first point
        ordered_path = []
        current = tumor_indices[0]
        remaining = [tuple(p) for p in tumor_indices]
        
        # Simple Greedy TSP / Nearest Neighbor
        # To speed up, we can just use a spatial sort (e.g. by Y then X, raster scan)
        # Raster scan is often best for systematic ablation.
        # Let's sort by Y, then X.
        
        # Actually, alternating raster (snake) is better to reduce jerky movements
        # Sort by X primary, Y secondary? No, usually scan lines (Y).
        
        # Let's do a KDTree or just simple distance sort if small number of points.
        # Given 64x64, tumor is small. Nearest neighbor is fine.
        
        current_pos = np.array([0,0]) # relative
        
        # Let's stick to a snake-raster for stability and coverage
        # Sort by Y
        # For each unique Y, sort X. If Y is odd, reverse X.
        
        # Group by Y
        rows = {}
        for r, c in remaining: # r is x, c is y in numpy? 
            # Wait, numpy is row-major (y, x) usually, but app.py used (width, height) in init.
            # cryo.py: xx, yy meshgrid. anatomy[xx, yy].
            # anatomy is [x][y] based on cryo.py logic `self.temp_map = np.full((width, height)...)`
            
            # Let's assume indices are (x, y).
            row_idx = c # Y
            col_idx = r # X
            if row_idx not in rows: rows[row_idx] = []
            rows[row_idx].append(col_idx)
            
        sorted_y = sorted(rows.keys())
        for i, y in enumerate(sorted_y):
            xs = sorted(rows[y])
            if i % 2 == 1:
                xs.reverse() # Snake
            for x in xs:
                ordered_path.append((x, y))
                
        self.targets = ordered_path
        print(f"Guidance: Path planned with {len(self.targets)} targets.")

    def start(self):
        self.active = True
        self.current_idx = 0
        self.completed = False
        
    def stop(self):
        self.active = False
        
    def get_next_target(self, current_robot_pos):
        """
        Returns (x, z) for the robot and boolean 'laser_on'
        """
        if not self.active or self.completed:
            return None, False
            
        if self.current_idx >= len(self.targets):
            self.completed = True
            self.active = False
            return None, False
            
        # Get grid target
        gx, gy = self.targets[self.current_idx]
        
        # Convert to Robot Space
        # grid_x = int((end_effector[0] + 0.5) * 64)
        # -> robot_x = (gx / 64.0) - 0.5
        rx = (gx / 64.0) - 0.5
        rz = (gy / 64.0)
        
        # Check if we are close enough to "ablate" and move to next
        # Grid resolution is 1/64 ~ 0.015
        # Robot pos is [x, y, z] (y is up/down, z is depth in MRI bore/plane)
        # app.py maps robot[2] to grid_y. robot[0] to grid_x.
        
        dx = current_robot_pos[0] - rx
        dz = current_robot_pos[2] - rz
        dist = np.sqrt(dx*dx + dz*dz)
        
        # Threshold: 1 pixel size approx 0.015
        if dist < 0.02:
            # We are at target. Hold for a moment?
            # For this loop, we just move to next. The update loop speed determines dwell time.
            # Ablation happens every step.
            self.current_idx += 1
            return (rx, rz), True # Laser ON
        else:
            # Move towards target
            return (rx, rz), False # Laser OFF while moving? Or ON? 
            # Usually OFF to preserve healthy tissue while traversing.
            # But if next target is neighbor, maybe ON?
            # Let's say OFF if dist > 0.02 (moving to start), ON if close.
            # Actually, standard is to track path.
            return (rx, rz), False

