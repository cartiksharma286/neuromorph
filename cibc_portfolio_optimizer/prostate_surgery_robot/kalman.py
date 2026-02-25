
import numpy as np

class KalmanFilter:
    """
    Simple 1D/3D Kalman Filter for position estimation.
    Used to stabilize laser targeting on tumor surfaces.
    """
    def __init__(self, process_noise=1e-5, measurement_noise=1e-2):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # State: [x, y, z]
        self.x = np.zeros(3)
        self.P = np.eye(3) # Covariance
        self.F = np.eye(3) # State transition (Static model for pinpointing)
        self.H = np.eye(3) # Measurement function
        self.initialized = False

    def update(self, measurement):
        """
        Update state with new measurement [x, y, z]
        """
        z = np.array([measurement['x'], measurement['y'], measurement['z']])
        
        if not self.initialized:
            self.x = z
            self.initialized = True
            return { 'x': self.x[0], 'y': self.x[1], 'z': self.x[2] }

        # Prediction
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.process_noise

        # Update
        y = z - (self.H @ x_pred) # Residual
        S = self.H @ P_pred @ self.H.T + self.measurement_noise
        
        # Robust Inversion via Eigen Decomposition
        eigvals, eigvecs = np.linalg.eigh(S)
        # Invert eigenvalues (threshold for singular values)
        eigvals_inv = np.array([1.0/e if e > 1e-8 else 0.0 for e in eigvals])
        S_inv = eigvecs @ np.diag(eigvals_inv) @ eigvecs.T
        
        K = P_pred @ self.H.T @ S_inv # Kalman Gain
        
        self.x = x_pred + K @ y
        self.P = (np.eye(3) - K @ self.H) @ P_pred
        
        return { 'x': self.x[0], 'y': self.x[1], 'z': self.x[2] }
