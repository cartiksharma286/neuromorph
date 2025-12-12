import numpy as np


class PulseOntology:
    """
    Semantic Ontological Reasoner.
    Classifies the current state of the pulse generation system into semantic categories
    and prescribes actions based on domain knowledge (MRI Physics).
    """
    def __init__(self):
        self.state = "INITIAL"
        self.convergence_history = []

    def reason(self, snr, angle_error):
        """
        Derives the semantic state and prescribed strategy.
        """
        # Ontology Rules
        abs_error = abs(angle_error)
        
        if snr < 5.0:
            self.state = "LOW_SNR_CRITICAL"
            return {"strategy": "EXPLORE_AGGRESSIVE", "amp_gain": 2.0, "bw_noise": 0.1}
        
        if abs_error > 45.0:
            self.state = "FLIP_ANGLE_MISMATCH_SEVERE"
            return {"strategy": "FIX_PHYSICS", "amp_gain": 1.5, "bw_noise": 0.0}
            
        if abs_error > 5.0:
            self.state = "FLIP_ANGLE_TUNING"
            return {"strategy": "FINE_TUNE_AMP", "amp_gain": 0.5, "bw_noise": 0.01}
            
        if snr > 20.0 and abs_error < 2.0:
            self.state = "OPTIMAL_CONVERGENCE"
            return {"strategy": "STABILIZE", "amp_gain": 0.1, "bw_noise": 0.001}
            
        self.state = "SEARCHING"
        return {"strategy": "EXPLORE_LOCAL", "amp_gain": 0.8, "bw_noise": 0.05}

class PulseDiffusionModel:
    def __init__(self, sequence_length=128):
        self.sequence_length = sequence_length
        self.timesteps = 50
        self.betas = np.linspace(0.0001, 0.02, self.timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        # Semantic Adaptive parameters
        self.target_bandwidth = 0.5
        self.target_amplitude = 40.0  # Physics-informed initialization
        self.momentum_amp = 0.0
        self.momentum_bw = 0.0
        
        # Knowledge Base
        self.ontology = PulseOntology()

    def noise_scheduler(self, x_start, t, noise=None):
        if noise is None:
            noise = np.random.randn(*x_start.shape)
        
        sqrt_alphas_cumprod_t = np.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod_t = np.sqrt(1.0 - self.alphas_cumprod[t])
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def sample(self, bandwidth=None, amplitude=None):
        """
        Simulates the reverse diffusion process.
        """
        if bandwidth is None: bandwidth = self.target_bandwidth
        if amplitude is None: amplitude = self.target_amplitude

        # Target shape (The "Ideal" manifold the model learned)
        # Sinc shape is standard for slice selection
        t_axis = np.linspace(-4, 4, self.sequence_length)
        ideal_pulse = amplitude * np.sinc(t_axis * bandwidth)
        
        # Start from pure noise
        x = np.random.randn(self.sequence_length)
        
        history = []
        
        for t in reversed(range(self.timesteps)):
            z = np.random.randn(self.sequence_length) if t > 0 else 0
            
            alpha = self.alphas[t]
            alpha_hat = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            # Semantic Guidance:
            # We enforce the "Ideal" structure more strongly in early steps
            # to ensure the pulse actually flips spins.
            
            predicted_noise = (x - np.sqrt(alpha_hat) * ideal_pulse) / np.sqrt(1 - alpha_hat + 1e-8)
             
            # Standard DDPM Update
            noise_factor = (1 - alpha) / (np.sqrt(1 - alpha_hat) + 1e-8)
            mean = (1 / np.sqrt(alpha)) * (x - noise_factor * predicted_noise)
            
            sigma = np.sqrt(beta)
            x = mean + sigma * z
            
            if t % 5 == 0 or t == 0:
                history.append(x.tolist())
                
        return x, history

    def optimize_step(self, current_snr, current_angle):
        """
        Semantic Adaptive Learning Step.
        Uses physics constraints (Target Flip Angle = 90 deg) 
        to drive the diffusion parameters.
        """
        target_angle = 90.0
        error = target_angle - current_angle
        
        # Consult Ontology
        reasoning = self.ontology.reason(current_snr, error)
        strategy = reasoning["strategy"]
        amp_gain = reasoning["amp_gain"]
        bw_noise = reasoning["bw_noise"]
        
        # Update Amplitude based on Strategy
        delta_amp = error * amp_gain
        self.momentum_amp = 0.8 * self.momentum_amp + 0.2 * delta_amp
        self.target_amplitude += self.momentum_amp
        self.target_amplitude = np.clip(self.target_amplitude, 0.0, 300.0)
        
        # Update Bandwidth based on Strategy
        if strategy == "STABILIZE":
             self.target_bandwidth += np.random.normal(0, bw_noise)
        else:
             # Gradient ascent for SNR if not stabilizing
             # Simple momentum heuristic for bandwidth too
             if current_snr < 15.0:
                 self.target_bandwidth += np.random.uniform(-bw_noise, bw_noise)
        
        self.target_bandwidth = np.clip(self.target_bandwidth, 0.1, 2.0)
        
        return {
            "new_bandwidth": self.target_bandwidth,
            "new_amplitude": self.target_amplitude,
            "angle_error": error,
            "semantic_state": self.ontology.state,
            "strategy": strategy
        }
