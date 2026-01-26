
"""
Depression Neural Model
Simulates Serotonin-Dopamine regulation and Executive Function recovery via DBS using Finite Element Analysis (FEA) based cortical profiles.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from fea_simulator import DBSFEASimulator

@dataclass
class NeurotransmitterState:
    serotonin: float  # 5-HT: Mood stability, inhibition of negative affect
    dopamine: float   # DA: Motivation, Reward, Executive gating in PFC
    glutamate: float  # Excitatory drive (often overactive in depression circuits like sgACC)
    
@dataclass
class ExecutiveMetrics:
    decision_speed: float
    working_memory: float
    cognitive_flexibility: float

class DepressionNeuralModel:
    def __init__(self):
        # Baseline: Depression typically characterized by low 5-HT, low/dysregulated DA, high Glutamate in local circuitry
        self.state = NeurotransmitterState(
            serotonin=0.3,
            dopamine=0.4,
            glutamate=0.8
        )
        
        self.executive_function = ExecutiveMetrics(
            decision_speed=0.4,
            working_memory=0.5,
            cognitive_flexibility=0.3
        )
        
        # FEA Simulator for Cortical Profiles
        self.fea = DBSFEASimulator(resolution=64)
        
        # Target specific coordinates in the FEA grid
        self.targets = {
            'sgACC': (32, 45),     # Subgenual Anterior Cingulate (Area 25)
            'NAc': (32, 20),       # Nucleus Accumbens
            'dlPFC': (20, 10),     # Dorsolateral Prefrontal Cortex (indirect)
            'mFB': (32, 32)        # Medial Forebrain Bundle
        }
        
        # Simulation History
        self.history = []

    def simulate_treatment_step(self, target: str, frequency: float, amplitude: float):
        """
        Simulate one treatment step:
        1. Run FEA to determine activation volume and profile.
        2. Adjust Neurotransmitters based on target activation.
        3. Update Executive Function based on transmitters.
        """
        
        # 1. Run FEA
        target_coords = self.targets.get(target, (32, 32))
        self.fea.generate_tissue_model(target_center=target_coords)
        
        # Configure electrode (simplified mapping of amplitude to voltage)
        # Higher amplitude = higher voltage spread
        voltage = -1.0 * amplitude
        config = {'voltages': {'c1': voltage}}
        
        phi, e_field, vta_mask = self.fea.solve_electric_field(config)
        
        # Calculate Activation metrics
        activation_volume = np.sum(vta_mask)
        max_field = np.max(e_field)
        
        # 2. Neurotransmitter Dynamics
        # Mechanism: High Freq DBS interacts with reuptake and release
        
        # Frequency efficacy (similar to OCD model, tuned for 130Hz)
        freq_factor = np.exp(-((frequency - 130)**2) / (2 * 40**2))
        
        # Intensity factor from FEA
        intensity_factor = min(1.0, activation_volume / 200.0) * (abs(voltage)/5.0)
        
        treatment_efficacy = freq_factor * intensity_factor
        
        # Regulatory changes based on Target
        if target == 'sgACC':
            # Area 25 suppression leads to 5-HT normalization and Glutamate reduction
            self.state.glutamate -= 0.1 * treatment_efficacy * self.state.glutamate
            self.state.serotonin += 0.05 * treatment_efficacy * (1.0 - self.state.serotonin)
            # Indirectly helps Dopamine due to reduced inhibition/stress
            self.state.dopamine += 0.02 * treatment_efficacy * (1.0 - self.state.dopamine)
            
        elif target == 'NAc':
            # Reward pathway -> Dopamine boost
            self.state.dopamine += 0.08 * treatment_efficacy * (1.0 - self.state.dopamine)
            self.state.serotonin += 0.02 * treatment_efficacy
            
        elif target == 'mFB':
            # Major highway -> global boost
            self.state.dopamine += 0.06 * treatment_efficacy
            self.state.serotonin += 0.06 * treatment_efficacy
            
        # Homestasis / Decay (tendency to return to baseline without persistent treatment)
        # Modeled as a slow leak back to pathology if treatment efficacy is low
        
        # 3. Update Executive Function (Derived from Neurotransmitters)
        # DA drives flexibility and memory; 5-HT drives decision stability/speed (less hesitation)
        
        self.executive_function.working_memory = 0.3 + 0.6 * self.state.dopamine
        self.executive_function.cognitive_flexibility = 0.2 + 0.8 * self.state.dopamine
        self.executive_function.decision_speed = 0.3 + 0.4 * self.state.serotonin + 0.3 * self.state.dopamine
        
        # Generate Visualization
        fea_heatmap = self.fea.generate_heatmap_plot(e_field, title=f"Cortical Profile: {target} (E-Field)")
        
        # Post Treatment Paradigms (Suggestions based on state)
        paradigms = self._generate_paradigms()
        
        result = {
            'neurotransmitters': self.state.__dict__,
            'executive': self.executive_function.__dict__,
            'fea_heatmap': fea_heatmap,
            'activation_stats': {
                'volume_mm3': float(activation_volume * 0.5), # Approx scaling
                'max_field_v_mm': float(max_field)
            },
            'paradigms': paradigms
        }
        
        return result

    def _generate_paradigms(self) -> List[str]:
        """Generate post-treatment behavioral paradigms based on current state."""
        paradigms = []
        
        if self.state.dopamine > 0.6:
            paradigms.append("Reward Learning Integration: Engage in goal-directed tasks immediately post-stimulation.")
        else:
            paradigms.append("Motivation Priming: Low-effort reward tasks recommended.")
            
        if self.state.serotonin > 0.6:
            paradigms.append("Affective Resilience: Cognitive Behavioral Therapy (CBT) consolidation window open.")
        
        if self.executive_function.cognitive_flexibility > 0.6:
            paradigms.append("Cognitive Reappraisal: Active reframing of negative thought patterns.")
            
        if self.state.glutamate < 0.6:
            paradigms.append("Neuroplasticity Maintenance: Sleep hygiene critical for consolidating synaptic LTD.");
            
        return paradigms

    def get_cortical_profile_data(self):
        """Return raw data for frontend custom rendering if needed."""
        # For now, we rely on the server-side heatmap generation
        pass
