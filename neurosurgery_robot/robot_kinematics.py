import numpy as np

class Robot6DOF:
    def __init__(self):
        # DH Parameters for a generic 6DOF arm (e.g., similar to PUMA or UR)
        # [theta, d, a, alpha]
        self.links = [
            {'d': 0.333, 'a': 0.000, 'alpha': -np.pi/2}, # Joint 1
            {'d': 0.000, 'a': -0.425, 'alpha': 0.000},   # Joint 2
            {'d': 0.000, 'a': -0.392, 'alpha': 0.000},   # Joint 3
            {'d': 0.109, 'a': 0.000, 'alpha': np.pi/2},  # Joint 4
            {'d': 0.095, 'a': 0.000, 'alpha': -np.pi/2}, # Joint 5
            {'d': 0.082, 'a': 0.000, 'alpha': 0.000}     # Joint 6
        ]
        self.joints = np.zeros(6) # Current joint angles in radians
        self.target_pos = np.array([0.5, 0.0, 0.5]) # Initial target

    def fk(self, joints):
        """Forward Kinematics to get end effector position"""
        T = np.eye(4)
        for i, link in enumerate(self.links):
            theta = joints[i]
            d = link['d']
            a = link['a']
            alpha = link['alpha']
            
            # DH Transformation Matrix
            Ti = np.array([
                [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [0,             np.sin(alpha),                np.cos(alpha),               d],
                [0,             0,                            0,                           1]
            ])
            T = np.dot(T, Ti)
        return T[:3, 3]

    def ik_step(self, target_pos, step_size=0.1):
        """
        Simple Jacobian Transpose or CCD step to move towards target.
        Using a simplified approach for robustness: coordinate descent or random optimization 
        might be too slow, so let's do small random perturbations or a pseudo-inverse Jacobian if time permits.
        
        For reliability in this demo, let's use a simple heuristic or just 'move joints towards valid range' 
        if we were doing full path planning. 
        
        Actually, for the simulation visual, we might simply update joint angles 
        to interpolate towards a "pre-calculated" posture or use a library.
        
        Let's implement a basic Jacobian-based IK for the 3D position (ignoring orientation for now).
        """
        current_pos = self.fk(self.joints)
        error = target_pos - current_pos
        
        if np.linalg.norm(error) < 0.001:
            return # Reached
            
        # Numerical Jacobian
        J = np.zeros((3, 6))
        epsilon = 1e-4
        original_joints = self.joints.copy()
        
        for i in range(6):
            self.joints[i] += epsilon
            pos_plus = self.fk(self.joints)
            self.joints[i] = original_joints[i]
            
            col = (pos_plus - current_pos) / epsilon
            J[:, i] = col
            
        # Damped Least Squares: J_pinv = J.T * (J * J.T + lambda^2 * I)^-1
        lambda_sq = 0.01
        inv = np.linalg.inv(np.dot(J, J.T) + lambda_sq * np.eye(3))
        J_pinv = np.dot(J.T, inv)
        
        d_theta = np.dot(J_pinv, error)
        
        # Apply and clamp
        self.joints += d_theta * step_size
        self.joints = np.clip(self.joints, -np.pi, np.pi) # Simple limits

    def set_target(self, x, y, z):
        self.target_pos = np.array([x, y, z])

    def update(self):
        # Perform one IK step towards target
        self.ik_step(self.target_pos)
        return self.joints.tolist(), self.fk(self.joints).tolist()
