import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Any

class QuantumPulseOptimizer:
    def __init__(self, target_flip_angle: float, duration_ms: float = 5.0, time_steps: int = 50):
        self.target_flip_angle = target_flip_angle
        self.duration_ms = duration_ms
        self.dt = duration_ms / time_steps
        self.time_steps = time_steps
        
        # Constraints
        self.max_amplitude = 25.0 # uT (microTesla) - simplified RF amplitude
        self.sar_limit = 4.0

    def _simulate_quantum_evolution(self, rf_waveform: np.ndarray) -> float:
        """
        Simulates the spin evolution (Bloch sphere) under the RF waveform.
        Returns the final flip angle.
        
        Ideally this would run on a QPU using parameterized gates (RX, RY).
        Here we use a simplified classical integration of the Bloch equations.
        """
        # Start state |0> (z-axis)
        state_vector = np.array([0, 0, 1.0]) 
        
        current_angle = 0.0
        
        # Simple integration: each step rotates the spin by (gamma * B1 * dt)
        # Gyromagnetic ratio for Protons ~ 42.58 MHz/T. 
        # Simplified: Angle += Amplitude * constant
        
        for amplitude in rf_waveform:
            # Rotation increment proportional to amplitude
            # Scaling factor tuned for "realistic" simulation ranges
            rotation_inc = amplitude * self.dt * 1.5 
            current_angle += rotation_inc
            
        return current_angle

    def _calculate_sar(self, rf_waveform: np.ndarray) -> float:
        """
        Estimates SAR based on integral of B1^2.
        """
        # SAR is proportional to integral of |B1(t)|^2
        energy = np.sum(np.square(rf_waveform)) * self.dt
        # Normalize to some "W/kg" scale for mock purposes
        sar = energy * 0.05 
        return sar

    def cost_function(self, parameters: np.ndarray) -> float:
        """
        VQE Cost Function = |Target - Actual|^2 + lambda * SAR
        """
        current_flip = self._simulate_quantum_evolution(parameters)
        sar = self._calculate_sar(parameters)
        
        # Penalties
        error_term = (self.target_flip_angle - current_flip)**2
        sar_penalty = 0
        if sar > self.sar_limit:
            sar_penalty = (sar - self.sar_limit) * 100 # Steep penalty
            
        # We also want to minimize SAR generally, not just keep it under limit
        regularization = sar * 0.1 
        
        return error_term + sar_penalty + regularization

    def run_optimization(self) -> Dict[str, Any]:
        """
        Runs the classical optimizer loop (mimicking the QPU-CPU loop).
        """
        # Initial guess: Gaussian-ish pulses randomly perturbed
        t = np.linspace(0, self.duration_ms, self.time_steps)
        # Gaussian envelope centered
        initial_guess = 10.0 * np.exp(-(t - self.duration_ms/2)**2 / 2) 
        
        # Bounds for Amplitude
        bounds = [(-self.max_amplitude, self.max_amplitude) for _ in range(self.time_steps)]
        
        # Run optimization
        # methods like 'COBYLA' or 'L-BFGS-B' are commonly used in VQE
        result = minimize(
            self.cost_function, 
            initial_guess, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        optimized_waveform = result.x
        final_flip = self._simulate_quantum_evolution(optimized_waveform)
        final_sar = self._calculate_sar(optimized_waveform)
        
        return {
            "success": result.success,
            "optimized_waveform": optimized_waveform.tolist(),
            "time_points": t.tolist(),
            "final_flip_angle": final_flip,
            "final_sar": final_sar,
            "iterations": result.nit,
            "cost": result.fun
        }
