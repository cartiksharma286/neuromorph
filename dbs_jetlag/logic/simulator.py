
import numpy as np

class JetLagSimulator:
    def __init__(self, num_neurons=100):
        self.num_neurons = num_neurons
        # Initial phases of SCN neurons (scattered for jet lag)
        self.phases = np.random.uniform(0, 2 * np.pi, num_neurons)
        # Internal frequencies (slightly varied)
        self.freqs = np.random.normal(1.0, 0.05, num_neurons)
        # Synaptic coupling matrix
        self.weights = np.full((num_neurons, num_neurons), 0.01)
        self.time = 0
        
    def step(self, dbs_intensity=0.5, target_phase=0.0, hebbian_rate=0.01, prime_mod=17):
        """
        Single simulation step.
        dbs_intensity: Strength of representational energy transfer.
        target_phase: The "correct" phase of the target time zone.
        hebbian_rate: Learning rate for synaptic amplification.
        prime_mod: Modulus for congruential prime repair.
        """
        self.time += 0.1
        
        # 1. Internal Dynamics + DBS Representational Energy Transfer
        # Each neuron attracted to its frequency and the target phase
        phase_velocity = self.freqs + dbs_intensity * np.sin(target_phase - self.phases)
        
        # 2. Coupling (Mean Field influence)
        mean_phase = np.arctan2(np.mean(np.sin(self.phases)), np.mean(np.cos(self.phases)))
        phase_velocity += 0.2 * np.sin(mean_phase - self.phases)
        
        self.phases += phase_velocity * 0.1
        self.phases = np.mod(self.phases, 2 * np.pi)
        
        # 3. Hebbian Amplification
        # Strengthen weights between neurons that are in phase
        for i in range(self.num_neurons):
            correlation = np.cos(self.phases[i] - self.phases)
            self.weights[i] += hebbian_rate * correlation * 0.01
            # Weight normalization to prevent runaway excitation
            self.weights[i] = np.clip(self.weights[i], 0, 0.5)
            
        # 4. Congruential Prime Repair (Pruning of noisy cells)
        # Using a PRNG-like check for pruning
        pruning_indices = []
        for i in range(self.num_neurons):
            # Prime-based hash for pruning logic
            prime_hash = (i * 13 + 7) % prime_mod
            if prime_hash == 0:
                # If "noisy" or selected for repair, snap to mean phase
                self.phases[i] = mean_phase
                pruning_indices.append(i)
                
        return {
            "phases": self.phases.tolist(),
            "mean_phase": float(mean_phase),
            "order_parameter": float(np.abs(np.mean(np.exp(1j * self.phases)))),
            "pruned_count": len(pruning_indices)
        }
