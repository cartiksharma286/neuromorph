
import numpy as np

class RobotKinematics:
    """
    Simulates a 6-DOF medical robot for transperineal access.
    Using a standard RCM (Remote Center of Motion) mechanic or generic arm.
    """
    def __init__(self):
        # 6 Joints: [BaseRot, Shoulder, Elbow, Wrist1, Wrist2, NeedleInsertion]
        self.current_joints = np.zeros(6)
        self.target_pos = np.zeros(3)
        self.velocity_limit = 0.05
        
        # Dimensions (meters)
        self.L1 = 0.2
        self.L2 = 0.2
        self.L3 = 0.15
        
    def fk(self, joints):
        # Simplified FK for visuals and logic
        # J1: Rotate Grid (Azimuth)
        # J2/J3: Positioning in plane
        # J6: Depth
        
        # Let's use a simple Cartesian mapping for the "Target" control
        # In real life this is complex, but for the app we map:
        # End Effector = Forward Kinematics
        
        # For this demo, let's treat it as a positioning gantry + needle
        # X, Y controlled by arm, Z by insertion
        
        x = joints[0] * 0.5 # Scale factors to meters
        y = joints[1] * 0.5 
        z = joints[2] * 0.5
        
        # Just return the stored "Cartesian" joints for simplicity in this specific requested flow
        # unless we want full trig. Let's do a mock trig to look "robotic".
        
        t1, t2, t3, t4, t5, t6 = joints
        
        # Simple SCARA-like logic for medical robot
        px = self.L1 * np.cos(t1) + self.L2 * np.cos(t1 + t2)
        py = self.L1 * np.sin(t1) + self.L2 * np.sin(t1 + t2)
        pz = t3 # Prismatic Z
        
        # But we need to target specific prostate coordinates.
        # Let's revert to a "Perfect IK" simulation where the robot just tracks the target smoothly.
        return np.array([px, py, pz])

    def get_end_effector(self):
        # Return current cartesian position approximation
        # We tracked the target directly for the "simulation" aspect
        # so we return the smoothed position
        return self.target_pos
        
    def update(self, target_dict):
        # Move internal state towards target
        tx = float(target_dict.get('x', 0))
        ty = float(target_dict.get('y', 0))
        tz = float(target_dict.get('z', 0))
        
        target = np.array([tx, ty, tz])
        
        # Interpolate
        diff = target - self.target_pos
        dist = np.linalg.norm(diff)
        
        if dist > 0.001:
            move = (diff / dist) * min(dist, self.velocity_limit)
            self.target_pos += move
            
        # Update "joints" to match this position (Inverse Kinematics Mock)
        # Just filling arrays to animate visualizations
        self.current_joints[0] = np.sin(self.target_pos[0] * 5.0) # Wobble visuals
        self.current_joints[1] = self.target_pos[0] * 2.0
        self.current_joints[2] = self.target_pos[1] * 2.0
        self.current_joints[5] = self.target_pos[2] # Depth
