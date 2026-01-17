
"""
OCD Neural Model
Simulates the CSTC (Cortico-Striato-Thalamo-Cortical) loop dynamics for OCD.
Focuses on the Orbitofrontal Cortex (OFC) -> Caudate -> Thalamus -> OFC feedback cycle.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy import stats

@dataclass
class OCDState:
    ofc_activity: float
    caudate_activity: float
    thalamus_activity: float
    cycle_gain: float
    ybocs_score: float

class OCDNeuralModel:
    def __init__(self):
        # Baseline connectivity strengths (Hyperactive loop in OCD)
        self.connectivity = {
            'ofc_to_caudate': 1.5,     # Excitatory (Glutamate) - Hyperconnected in OCD
            'caudate_to_thalamus': 0.8, # Inhibitory (GABA) - but dysregulated
            'thalamus_to_ofc': 1.2      # Excitatory (Glutamate)
        }
        
        # Baseline neural activity (0-1 scale)
        self.state = {
            'ofc': 0.8,      # Hyperactive worry/error detection
            'caudate': 0.7,  # Gating failure
            'thalamus': 0.6  # Relay
        }
        
        self.time_step = 0.01

    def calculate_cycle_gain(self) -> float:
        """
        Calculates the loop gain of the CSTC circuit.
        Gain > 1.0 implies 'runaway' cyclic reasoning (Obsession).
        """
        # Simplified loop gain approximation
        gain = (self.connectivity['ofc_to_caudate'] * 
                self.connectivity['caudate_to_thalamus'] * 
                self.connectivity['thalamus_to_ofc'])
        return gain

    def calculate_ybocs(self) -> float:
        """
        Yale-Brown Obsessive Compulsive Scale (0-40).
        Correlates with OFC hyperactivity and loop gain.
        """
        gain = self.calculate_cycle_gain()
        ofc_act = self.state['ofc']
        
        # Base score + gain factor + activity factor
        # Typical severe OCD is > 24
        
        score = 10 + (gain * 10) + (ofc_act * 15)
        return min(40, max(0, score))

    def apply_dbs(self, target: str, frequency: float, amplitude: float):
        """
        Simulate DBS treatment.
        High frequency stimulation (>100Hz) effectively creates a functional lesion
        or regularizes the target, reducing its gain/activity.
        """
        
        # Efficacy curve based on frequency (tuned for High Freq usually)
        # Optimal ~ 130Hz
        freq_efficacy = np.exp(-((frequency - 130)**2) / (2 * 30**2)) 
        
        # Amplitude intensity
        intensity = np.tanh(amplitude / 3.0) # Saturation at higher amps
        
        suppression_factor = freq_efficacy * intensity * 0.6 # Max 60% suppression
        
        if target == 'caudate':
            # DBS to Caudate reduces its output to Thalamus, or regularizes it
            # In our simplified model, we reduce connectivity
            self.connectivity['caudate_to_thalamus'] *= (1.0 - suppression_factor)
            self.state['caudate'] *= (1.0 - suppression_factor * 0.5)
            
        elif target == 'stn':
            # STN (Subthalamic Nucleus) is another target. 
            # It drives the Globus Pallidus, which inhibits Thalamus.
            # Simplified: STN-DBS reduces Thalamic output driving the OFC
            self.connectivity['thalamus_to_ofc'] *= (1.0 - suppression_factor)
            
        elif target == 'vc_vs':
            # Ventral Capsule / Ventral Striatum
            # Major highway connecting OFC and Thalamus
            self.connectivity['ofc_to_caudate'] *= (1.0 - suppression_factor)
            
        # Re-settle state after parameter change (simple relaxation)
        for _ in range(10): 
            self._step_dynamics()
            
    def _step_dynamics(self):
        """Single step of circuit dynamics"""
        # OFC drives Caudate
        input_caudate = self.state['ofc'] * self.connectivity['ofc_to_caudate']
        self.state['caudate'] += (np.tanh(input_caudate) - self.state['caudate']) * 0.1
        
        # Caudate (via GPI/SNr chain) gates Thalamus (Simplified)
        # In OCD, the "Direct Pathway" (Go) is often overactive relative to Indirect
        # We model this as Caudate transmission allowing Thalamus to fire
        input_thalamus = self.state['caudate'] * self.connectivity['caudate_to_thalamus']
        self.state['thalamus'] += (np.tanh(input_thalamus) - self.state['thalamus']) * 0.1
        
        # Thalamus drives OFC (The Loop)
        input_ofc = self.state['thalamus'] * self.connectivity['thalamus_to_ofc']
        self.state['ofc'] += (np.tanh(input_ofc) - self.state['ofc']) * 0.1


    def run_clinical_trial(self, n_subjects: int, dbs_target: str, freq: float, amp: float):
        """
        Simulate a statistical clinical trial.
        Returns Pre and Post YBOCS scores for a population.
        """
        pre_scores = []
        post_scores = []
        
        np.random.seed(42) # Reproducibility
        
        for i in range(n_subjects):
            # Create a subject with some variance
            subject = OCDNeuralModel()
            
            # Add random variance to baseline connectivity (Heterogeneity)
            subject.connectivity['ofc_to_caudate'] *= np.random.normal(1.0, 0.1)
            subject.connectivity['thalamus_to_ofc'] *= np.random.normal(1.0, 0.1)
            subject.state['ofc'] *= np.random.normal(1.0, 0.1)
            
            # Initial (Pre) Score
            # Run dynamics for a bit to stabilize
            for _ in range(20): subject._step_dynamics()
            pre = subject.calculate_ybocs()
            pre_scores.append(pre)
            
            # Treatment
            subject.apply_dbs(dbs_target, freq, amp)
            
            # Post Score
            post = subject.calculate_ybocs()
            post_scores.append(post)
            
        # Statistical Analysis
        t_stat, p_val = stats.ttest_rel(pre_scores, post_scores)
        
        return {
            'pre_scores': [float(x) for x in pre_scores],
            'post_scores': [float(x) for x in post_scores],
            'mean_pre': float(np.mean(pre_scores)),
            'mean_post': float(np.mean(post_scores)),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'significant': bool(p_val < 0.05)
        }

if __name__ == "__main__":
    # Test
    model = OCDNeuralModel()
    print(f"Baseline Gain: {model.calculate_cycle_gain()}")
    print(f"Baseline YBOCS: {model.calculate_ybocs()}")
    
    model.apply_dbs('caudate', 130, 3.0)
    print(f"Post-DBS Gain: {model.calculate_cycle_gain()}")
    print(f"Post-DBS YBOCS: {model.calculate_ybocs()}")
    
    trial = model.run_clinical_trial(20, 'caudate', 130, 3.0)
    print(f"Trial P-Value: {trial['p_value']}")
