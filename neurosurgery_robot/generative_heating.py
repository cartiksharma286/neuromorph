
import numpy as np

class GenerativeTissueHeating:
    """
    Generative AI module for Tissue Heating Control.
    Simulates a Diffusion-based Generative Model to optimize heating profiles
    for robot actuation during thermal ablation.
    """
    def __init__(self):
        self.target_temp = 65.0 # Target ablation temperature
        self.sigma = 5.0 # Noise level for diffusion process
        self.diffusion_steps = 100
        self.current_step = 0
        self.generated_profile = []
        self.model_state = "IDLE" 
        self.mode = "STANDARD"
        
    def generate_heating_curve(self, duration_steps=100, mode="STANDARD"):
        """
        Generates an optimal heating curve using a simulated stochastic process.
        Modes: STANDARD, RAPID, GENTLE
        """
        self.model_state = "GENERATING"
        self.mode = mode
        t = np.linspace(0, 1, duration_steps)
        
        # Base curve shape depends on mode
        if mode == "RAPID":
            # Steep rise, potential overshoot allowed
            # Sigmoid with k=20
            base_curve = 37.0 + (self.target_temp - 37.0) * (1 / (1 + np.exp(-20 * (t - 0.2))))
            # Add overshoot bump
            overshoot = 5.0 * np.exp(-100 * (t - 0.25)**2)
            base_curve += overshoot
        elif mode == "GENTLE":
            # Linear/Slow ramp, very stable
            base_curve = np.linspace(37.0, self.target_temp, duration_steps)
            # Smooth start/end
            base_curve = 37.0 + (self.target_temp - 37.0) * (0.5 - 0.5 * np.cos(np.pi * t))
        else:
            # STANDARD
            # Standard sigmoid
            base_curve = 37.0 + (self.target_temp - 37.0) * (1 / (1 + np.exp(-10 * (t - 0.5))))
        
        # Add "Generative" noise (Simulating the diffusion generation process)
        noise = np.random.normal(0, 2.0, duration_steps)
        # Smoothing depends on mode (Gentle is smoother)
        window = 10 if mode == "GENTLE" else 5
        smoothed_noise = np.convolve(noise, np.ones(window)/window, mode='same')
        
        self.generated_profile = base_curve + smoothed_noise
        self.model_state = "READY"
        return self.generated_profile

    def get_control_action(self, current_temp, dt=0.1):
        """
        Returns the recommended power level to track the generated profile.
        uses a PID-like control steered by the Generative Model's prediction.
        """
        # If no profile, generate one
        if len(self.generated_profile) == 0:
            self.generate_heating_curve()
            
        # Get target for current time step
        target = self.generated_profile[min(self.current_step, len(self.generated_profile)-1)]
        
        error = target - current_temp
        
        # Generative AI "Correction" term
        generative_gain = 5.0
        if self.mode == "RAPID": generative_gain = 8.0
        if self.mode == "GENTLE": generative_gain = 3.0
        
        power = max(0, error * generative_gain)
        
        simulation_noise = np.random.normal(0, 0.5) 
        power += simulation_noise 
        
        self.current_step = (self.current_step + 1) % len(self.generated_profile)
        
        return {
            'power': power,
            'target_temp': target,
            'model_state': self.model_state,
            'mode': self.mode
        }
