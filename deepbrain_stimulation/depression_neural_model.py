
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
    serotonin: float  # 5-HT: Mood stability
    dopamine: float   # DA: Motivation & Reward
    glutamate: float  # Excitatory drive (target for reduction)
    hippocampal_activity: float # New: Memory encoding & Neurogenesis marker

@dataclass
class ExecutiveMetrics:
    decision_speed: float
    working_memory: float
    cognitive_flexibility: float
    emotional_regulation: float # New metric

class DepressionNeuralModel:
    def __init__(self):
        # Baseline: High Glutamate, Low 5-HT/DA, Low Hippocampal Activity (atrophy risk)
        self.state = NeurotransmitterState(
            serotonin=0.3,
            dopamine=0.4,
            glutamate=0.85, 
            hippocampal_activity=0.3
        )
        
        self.executive_function = ExecutiveMetrics(
            decision_speed=0.4,
            working_memory=0.5,
            cognitive_flexibility=0.3,
            emotional_regulation=0.2
        )
        
        # FEA Simulator
        self.fea = DBSFEASimulator(resolution=64)
        
        # Expanded Targets including Frontal Lobe regions
        self.targets = {
            'sgACC': (32, 45),     # Subgenual Cingulate (Area 25)
            'dlPFC': (20, 10),     # Dorsolateral Prefrontal Cortex (Executive)
            'vmPFC': (32, 55),     # Ventromedial PFC (Emotion Regulation)
            'NAc': (32, 20),       # Nucleus Accumbens (Reward)
            'HP': (32, 32)         # Hippocampus (Direct - theoretical/modeling)
        }
        
        self.history = []

    def simulate_treatment_step(self, target: str, frequency: float, amplitude: float):
        """
        Simulate treatment step with focus on Frontal Lobe & Hippocampal regulation.
        """
        
        # 1. Run FEA
        target_coords = self.targets.get(target, (32, 32))
        self.fea.generate_tissue_model(target_center=target_coords)
        
        voltage = -1.0 * amplitude
        config = {'voltages': {'c1': voltage}}
        
        phi, e_field, vta_mask = self.fea.solve_electric_field(config)
        
        # Activation metrics
        activation_volume = np.sum(vta_mask)
        max_field = np.max(e_field)
        
        # 2. Dynamics
        # Frequency tuning (130Hz optimal for depression)
        freq_factor = np.exp(-((frequency - 130)**2) / (2 * 40**2))
        intensity_factor = min(1.0, activation_volume / 250.0) * (abs(voltage)/4.0)
        efficacy = freq_factor * intensity_factor

        # Target-Specific Logic
        if target == 'dlPFC' or target == 'Frontal Lobe': # Frontal Lobe Acute DBS
            # Top-down regulation: Increases Hippocampal activity via control pathways
            # Reduces excess Glutamate in limbic loops
            self.state.hippocampal_activity += 0.15 * efficacy * (1.0 - self.state.hippocampal_activity)
            self.state.glutamate -= 0.12 * efficacy * self.state.glutamate
            self.state.dopamine += 0.05 * efficacy
            self.executive_function.working_memory += 0.1 * efficacy
            self.executive_function.emotional_regulation += 0.08 * efficacy

        elif target == 'vmPFC':
            # Strong emotional regulation, glutamatergic dampening
            self.state.glutamate -= 0.15 * efficacy * self.state.glutamate
            self.state.serotonin += 0.08 * efficacy * (1.0 - self.state.serotonin)
            self.executive_function.emotional_regulation += 0.15 * efficacy
            self.state.hippocampal_activity += 0.05 * efficacy # Mild boost

        elif target == 'sgACC':
            # Classic target: Reduces sadness, normalizes metabolic activity
            self.state.glutamate -= 0.1 * efficacy * self.state.glutamate
            self.state.serotonin += 0.1 * efficacy * (1.0 - self.state.serotonin)
            self.executive_function.decision_speed += 0.05 * efficacy

        elif target == 'NAc':
            # Reward boost
            self.state.dopamine += 0.2 * efficacy * (1.0 - self.state.dopamine)
            self.executive_function.cognitive_flexibility += 0.1 * efficacy

        # General Hippocampal decay if untreated
        if efficacy < 0.1:
            self.state.hippocampal_activity *= 0.99

        # 3. Update Metrics based on new state
        self.executive_function.decision_speed = 0.3 + 0.5 * self.state.serotonin
        self.executive_function.cognitive_flexibility = 0.2 + 0.6 * self.state.dopamine + 0.2 * self.state.hippocampal_activity
        
        # Generate Visualization
        fea_heatmap = self.fea.generate_heatmap_plot(e_field, title=f"Cortical: {target} | Glu: {self.state.glutamate:.2f}")
        
        paradigms = self._generate_paradigms()
        
        return {
            'neurotransmitters': self.state.__dict__,
            'executive': self.executive_function.__dict__,
            'fea_heatmap': fea_heatmap,
            'activation_stats': {
                'volume_mm3': float(activation_volume * 0.5),
                'max_field_v_mm': float(max_field)
            },
            'paradigms': paradigms
        }

    def _generate_paradigms(self) -> List[str]:
        paradigms = []
        if self.state.hippocampal_activity > 0.6:
            paradigms.append("Neurogenesis Protocol: Engage in spatial navigation tasks to boost hippocampal volume.")
        if self.state.glutamate < 0.6:
            paradigms.append("Excitotoxicity Reduced: Maintenance phase - prioritize sleep and stress reduction.")
        if self.executive_function.emotional_regulation > 0.6:
            paradigms.append("Affective Control: Practice mindfulness to consolidate vmPFC-Amygdala connectivity.")
        if self.state.dopamine > 0.6:
            paradigms.append("Reward Integration: Goal-directed behavioral activation.")
        return paradigms

    def get_cortical_profile_data(self):
        pass
