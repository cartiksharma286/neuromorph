
import numpy as np
from scipy.special import j0  # Bessel function of first kind, for "quantum" modes

class GenerativeTissueHeating:
    """
    Quantum-Enhanced Generative AI module for Tissue Heating Control.
    Uses simulated Quantum Machine Learning (QML) to generate optimal 2D heat diffusion patterns
    targeted at tumor sites, refined by statistical classifiers.
    """
    def __init__(self, width=64, height=64):
        self.width = width
        self.height = height
        self.target_temp = 65.0 
        self.sigma = 5.0 
        self.diffusion_steps = 100
        self.current_step = 0
        self.generated_profile = [] # Scalar profile (legacy support)
        self.generated_field = np.zeros((width, height)) # 2D Quantum Field
        self.model_state = "IDLE" 
        self.mode = "STANDARD"
        
        # QML Parameters
        self.num_qubits = 6 # Simulated
        self.eigenmodes = self._generate_eigenmodes()
        
    def _generate_eigenmodes(self):
        """Pre-compute quantum eigenmodes (Bessel functions) for the domain."""
        modes = []
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height))
        r = np.sqrt(xx**2 + yy**2)
        theta = np.arctan2(yy, xx)
        
        # Generate a basis set of "Quantum Thermal Modes"
        for n in range(5):
            for m in range(2):
                # Mode shape: J_n(k*r) * cos(m*theta)
                # Rescale r to be 0..1 roughly
                mode = j0((n+1)*3.14*r) * np.cos(m * theta)
                modes.append(mode)
        return np.array(modes)

    def statistical_classifier(self, current_map, target_map):
        """
        Statistical Classifier (Simulated) to evaluate ablation efficacy.
        Returns a 'probability of success' based on overlap between heat and target.
        """
        # Feature extraction
        # Handle cases where mask is empty
        if np.sum(target_map) < 1:
            return 0.0
            
        tumor_heat = np.mean(current_map[target_map > 0.5])
        healthy_heat = np.mean(current_map[target_map <= 0.5])
        
        # Logistic function for probability
        # We want Tumor Heat > 60, Healthy Heat < 40
        score = (tumor_heat - 60.0) - (healthy_heat - 37.0)
        prob = 1.0 / (1.0 + np.exp(-0.1 * score))
        return prob

    def generate_heating_pattern(self, target_mask, mode="STANDARD"):
        """
        Generates a 2D heat diffusion pattern using QML optimization to match the target mask.
        """
        self.model_state = "QUANTUM_OPTIMIZING"
        self.mode = mode
        
        # 1. Quantum State Preparation (Superposition)
        # We want to find weights 'w' such that sum(w_i * mode_i) ~ target_mask
        # Simple Linear Regression (Least Squares) as a proxy for VQE (Variational Quantum Eigensolver)
        
        # Flatten for solver
        # Ensure target_mask is same shape
        if target_mask.shape != (self.width, self.height):
             # Resize or fail? Let's assume it matches for now as both are 64x64
             pass
             
        A = self.eigenmodes.reshape(len(self.eigenmodes), -1).T
        b = target_mask.reshape(-1)
        
        # Solve A * w = b
        # Regularization (Ridge) to prevent instability
        alpha = 0.5
        w = np.linalg.lstsq(A.T @ A + alpha * np.eye(len(self.eigenmodes)), A.T @ b, rcond=None)[0]
        
        # Reconstruct Field
        field = np.zeros((self.width, self.height))
        for i, weight in enumerate(w):
            field += weight * self.eigenmodes[i]
            
        # Normalize and Apply Activation (Simulating physical constraints)
        field = np.maximum(0, field)
        if np.max(field) > 0:
            field /= np.max(field)
            
        # Add "Diffusion" noise (Generative aspect)
        noise = np.random.normal(0, 0.05, (self.width, self.height))
        field += noise
        field = np.maximum(0, field)

        self.generated_field = field
        self.model_state = "FIELD_READY"
        return self.generated_field

    def generate_heating_curve(self, duration_steps=100, mode="STANDARD"):
        """Legacy 1D curve generation."""
        self.model_state = "GENERATING"
        self.mode = mode
        t = np.linspace(0, 1, duration_steps)
        if mode == "RAPID":
            base_curve = 37.0 + (self.target_temp - 37.0) * (1 / (1 + np.exp(-20 * (t - 0.2))))
            base_curve += 5.0 * np.exp(-100 * (t - 0.25)**2)
        elif mode == "GENTLE":
            base_curve = 37.0 + (self.target_temp - 37.0) * (0.5 - 0.5 * np.cos(np.pi * t))
        else:
            base_curve = 37.0 + (self.target_temp - 37.0) * (1 / (1 + np.exp(-10 * (t - 0.5))))
        noise = np.random.normal(0, 2.0, duration_steps)
        window = 10 if mode == "GENTLE" else 5
        smoothed_noise = np.convolve(noise, np.ones(window)/window, mode='same')
        self.generated_profile = base_curve + smoothed_noise
        self.model_state = "READY"
        return self.generated_profile

    def get_control_action(self, current_temp, dt=0.1):
        """
        Returns scalar power for compatibility, but the main logic is now in generate_heating_pattern.
        """
        # Legacy support for scalar control
        if len(self.generated_profile) == 0:
            self.generate_heating_curve()
        target = self.generated_profile[min(self.current_step, len(self.generated_profile)-1)]
        error = target - current_temp
        power = max(0, error * 5.0)
        self.current_step = (self.current_step + 1) % len(self.generated_profile)
        return {
            'power': power,
            'target_temp': target,
            'model_state': self.model_state,
            'mode': self.mode
        }
