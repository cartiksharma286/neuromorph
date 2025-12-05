"""
Dementia Neural Model
Simulates memory circuits, Alzheimer's pathology, and cognitive decline for DBS optimization
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json


@dataclass
class BrainRegion:
    """Brain region configuration for dementia model"""
    name: str
    baseline_activity: float  # 0-1
    connectivity: Dict[str, float]
    atrophy_rate: float  # Rate of volume loss in Alzheimer's
    cholinergic_sensitivity: float  # Response to acetylcholine


@dataclass
class CognitiveScores:
    """Cognitive assessment scores"""
    mmse: float = 24.0  # Mini-Mental State Examination (0-30)
    moca: float = 22.0  # Montreal Cognitive Assessment (0-30)
    memory_encoding: float = 0.5  # 0-1
    memory_retrieval: float = 0.5  # 0-1
    executive_function: float = 0.6  # 0-1
    attention: float = 0.6  # 0-1
    
    def get_disease_stage(self) -> str:
        """Determine disease stage based on MMSE"""
        if self.mmse >= 24:
            return "Mild Cognitive Impairment (MCI)"
        elif self.mmse >= 20:
            return "Mild Dementia"
        elif self.mmse >= 10:
            return "Moderate Dementia"
        else:
            return "Severe Dementia"


class DementiaNeuralModel:
    """
    Computational model of dementia pathology and memory circuits
    Simulates Alzheimer's disease effects and DBS treatment
    """
    
    def __init__(self, disease_duration_years: float = 2.0):
        self.disease_duration = disease_duration_years
        
        # Initialize brain regions for memory circuits
        self.regions = {
            'hippocampus': BrainRegion(
                name='Hippocampus',
                baseline_activity=0.4,  # Reduced in Alzheimer's
                connectivity={
                    'entorhinal_cortex': 0.8,
                    'nucleus_basalis': 0.5,
                    'prefrontal_cortex': 0.6
                },
                atrophy_rate=0.05,  # 5% per year
                cholinergic_sensitivity=0.9
            ),
            'entorhinal_cortex': BrainRegion(
                name='Entorhinal Cortex',
                baseline_activity=0.3,  # Early damage in Alzheimer's
                connectivity={
                    'hippocampus': 0.8,
                    'prefrontal_cortex': 0.5
                },
                atrophy_rate=0.08,  # 8% per year (earliest affected)
                cholinergic_sensitivity=0.7
            ),
            'nucleus_basalis': BrainRegion(
                name='Nucleus Basalis of Meynert',
                baseline_activity=0.3,  # Cholinergic neurons lost
                connectivity={
                    'hippocampus': 0.7,
                    'prefrontal_cortex': 0.6,
                    'entorhinal_cortex': 0.5
                },
                atrophy_rate=0.06,
                cholinergic_sensitivity=1.0  # Source of acetylcholine
            ),
            'prefrontal_cortex': BrainRegion(
                name='Prefrontal Cortex',
                baseline_activity=0.5,
                connectivity={
                    'hippocampus': 0.6,
                    'nucleus_basalis': 0.5
                },
                atrophy_rate=0.03,
                cholinergic_sensitivity=0.6
            )
        }
        
        # Current neural activity
        self.activity = {
            region: config.baseline_activity 
            for region, config in self.regions.items()
        }
        
        # Alzheimer's pathology
        self.amyloid_beta = 0.7  # Plaque burden (0-1)
        self.tau_tangles = 0.6  # Neurofibrillary tangles (0-1)
        self.acetylcholine_level = 0.4  # Reduced in Alzheimer's (0-1)
        
        # Cognitive scores
        self.cognitive_scores = CognitiveScores()
        
        # Treatment history
        self.treatment_history = []
        
        # Apply disease progression
        self._apply_disease_progression()
    
    def _apply_disease_progression(self):
        """Apply Alzheimer's disease progression effects"""
        # Increase pathology with disease duration
        self.amyloid_beta = min(1.0, 0.3 + self.disease_duration * 0.15)
        self.tau_tangles = min(1.0, 0.2 + self.disease_duration * 0.12)
        
        # Reduce acetylcholine
        self.acetylcholine_level = max(0.1, 0.8 - self.disease_duration * 0.15)
        
        # Apply atrophy to regions
        for region, config in self.regions.items():
            atrophy_factor = 1.0 - (config.atrophy_rate * self.disease_duration)
            self.activity[region] = config.baseline_activity * max(0.1, atrophy_factor)
        
        # Update cognitive scores based on pathology
        self._update_cognitive_scores()
    
    def _update_cognitive_scores(self):
        """Update cognitive scores based on neural activity and pathology"""
        # MMSE correlates with hippocampal volume and overall cognition
        hippocampal_factor = self.activity['hippocampus'] / 0.7
        pathology_factor = 1.0 - (self.amyloid_beta * 0.3 + self.tau_tangles * 0.3)
        self.cognitive_scores.mmse = 30 * hippocampal_factor * pathology_factor
        
        # MoCA is more sensitive to early changes
        entorhinal_factor = self.activity['entorhinal_cortex'] / 0.5
        self.cognitive_scores.moca = 30 * entorhinal_factor * pathology_factor * 0.95
        
        # Memory functions
        self.cognitive_scores.memory_encoding = self.activity['entorhinal_cortex'] * 1.5
        self.cognitive_scores.memory_retrieval = self.activity['hippocampus'] * 1.5
        
        # Executive function and attention
        self.cognitive_scores.executive_function = self.activity['prefrontal_cortex'] * 1.2
        self.cognitive_scores.attention = self.acetylcholine_level * 0.8
        
        # Clip all scores to valid ranges
        self.cognitive_scores.mmse = np.clip(self.cognitive_scores.mmse, 0, 30)
        self.cognitive_scores.moca = np.clip(self.cognitive_scores.moca, 0, 30)
        self.cognitive_scores.memory_encoding = np.clip(self.cognitive_scores.memory_encoding, 0, 1)
        self.cognitive_scores.memory_retrieval = np.clip(self.cognitive_scores.memory_retrieval, 0, 1)
        self.cognitive_scores.executive_function = np.clip(self.cognitive_scores.executive_function, 0, 1)
        self.cognitive_scores.attention = np.clip(self.cognitive_scores.attention, 0, 1)
    
    def simulate_neural_dynamics(self, time_steps: int = 100, dt: float = 0.01):
        """Simulate memory circuit dynamics"""
        activity_history = {region: [] for region in self.regions}
        
        for t in range(time_steps):
            new_activity = {}
            
            for region, config in self.regions.items():
                # Network input from connected regions
                network_input = 0
                for connected_region, weight in config.connectivity.items():
                    if connected_region in self.activity:
                        network_input += weight * self.activity[connected_region]
                
                # Cholinergic modulation
                cholinergic_boost = config.cholinergic_sensitivity * self.acetylcholine_level * 0.3
                
                # Pathology effects (amyloid and tau reduce activity)
                pathology_suppression = 1.0 - (self.amyloid_beta * 0.2 + self.tau_tangles * 0.15)
                
                # Neural activation
                activation = self._sigmoid(network_input + cholinergic_boost) * pathology_suppression
                
                # Update with time constant
                tau = 10.0
                dA = (-self.activity[region] + activation) / tau
                new_activity[region] = self.activity[region] + dt * dA
                new_activity[region] = np.clip(new_activity[region], 0, 1)
                
                activity_history[region].append(new_activity[region])
            
            self.activity = new_activity
        
        return activity_history
    
    def apply_dbs_stimulation(self, target_region: str, amplitude_ma: float,
                             frequency_hz: float, pulse_width_us: float,
                             duration_s: float = 1.0):
        """
        Apply DBS stimulation to target region
        Different frequencies have different effects on memory
        """
        if target_region not in self.regions:
            raise ValueError(f"Unknown region: {target_region}")
        
        # Calculate stimulation effect
        stim_effect = self._calculate_stimulation_effect(
            amplitude_ma, frequency_hz, pulse_width_us
        )
        
        # Frequency-specific effects on memory
        if 4 <= frequency_hz <= 8:  # Theta band - memory encoding
            memory_boost = stim_effect * 0.4
            self.cognitive_scores.memory_encoding += memory_boost
        elif 30 <= frequency_hz <= 100:  # Gamma band - memory retrieval
            memory_boost = stim_effect * 0.3
            self.cognitive_scores.memory_retrieval += memory_boost
        elif frequency_hz >= 100:  # High frequency - general network modulation
            # Modulate target region activity
            sensitivity = self.regions[target_region].cholinergic_sensitivity
            self.activity[target_region] *= (1 + stim_effect * sensitivity * 0.2)
        
        # Cholinergic enhancement (especially for nucleus basalis)
        if target_region == 'nucleus_basalis':
            self.acetylcholine_level = min(1.0, self.acetylcholine_level + stim_effect * 0.15)
        
        # Simulate network dynamics
        self.simulate_neural_dynamics(time_steps=int(duration_s * 100))
        
        # Update cognitive scores
        self._update_cognitive_scores()
        
        # Record treatment
        self.treatment_history.append({
            'target': target_region,
            'amplitude_ma': amplitude_ma,
            'frequency_hz': frequency_hz,
            'pulse_width_us': pulse_width_us,
            'duration_s': duration_s,
            'mmse': self.cognitive_scores.mmse,
            'moca': self.cognitive_scores.moca,
            'activity_state': self.activity.copy()
        })
        
        return {
            'activity': self.activity.copy(),
            'cognitive_scores': self.cognitive_scores,
            'efficacy': self._calculate_efficacy()
        }
    
    def _calculate_stimulation_effect(self, amplitude_ma: float,
                                     frequency_hz: float,
                                     pulse_width_us: float) -> float:
        """Calculate effective stimulation strength"""
        charge_per_phase = amplitude_ma * (pulse_width_us / 1000)
        freq_factor = np.log(frequency_hz + 1) / np.log(200)
        effect = (charge_per_phase / 10.0) * freq_factor
        return np.clip(effect, 0, 1)
    
    def _calculate_efficacy(self) -> float:
        """Calculate treatment efficacy (cognitive improvement)"""
        baseline_mmse = 18.0  # Typical moderate Alzheimer's
        current_mmse = self.cognitive_scores.mmse
        efficacy = (current_mmse - baseline_mmse) / (30 - baseline_mmse)
        return max(0, efficacy)
    
    def _sigmoid(self, x: float, gain: float = 1.0) -> float:
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-gain * x))
    
    def predict_treatment_response(self, target_region: str,
                                  amplitude_ma: float,
                                  frequency_hz: float,
                                  pulse_width_us: float,
                                  treatment_months: int = 6) -> Dict:
        """Predict long-term treatment response"""
        monthly_results = []
        
        # Reset to current state
        initial_mmse = self.cognitive_scores.mmse
        
        for month in range(treatment_months):
            # Apply monthly stimulation
            result = self.apply_dbs_stimulation(
                target_region, amplitude_ma, frequency_hz,
                pulse_width_us, duration_s=1.0
            )
            
            # Neuroplasticity effects (gradual improvement)
            plasticity_factor = 1 - np.exp(-month / 3.0)
            
            # Slow disease progression continues
            disease_factor = 1 - (month * 0.01)
            
            # Net effect
            self.cognitive_scores.mmse *= (1 + plasticity_factor * 0.05) * disease_factor
            self.cognitive_scores.moca *= (1 + plasticity_factor * 0.05) * disease_factor
            
            # Clip scores
            self.cognitive_scores.mmse = np.clip(self.cognitive_scores.mmse, 0, 30)
            self.cognitive_scores.moca = np.clip(self.cognitive_scores.moca, 0, 30)
            
            monthly_results.append({
                'month': month + 1,
                'mmse': self.cognitive_scores.mmse,
                'moca': self.cognitive_scores.moca,
                'memory_encoding': self.cognitive_scores.memory_encoding,
                'memory_retrieval': self.cognitive_scores.memory_retrieval,
                'executive_function': self.cognitive_scores.executive_function,
                'disease_stage': self.cognitive_scores.get_disease_stage(),
                'activity': self.activity.copy()
            })
        
        # Calculate overall response
        final_mmse = monthly_results[-1]['mmse']
        response_rate = (final_mmse - initial_mmse) / initial_mmse if initial_mmse > 0 else 0
        
        return {
            'monthly_progression': monthly_results,
            'response_rate': response_rate,
            'responder': response_rate > 0.1,  # >10% improvement = responder
            'final_cognitive_scores': self.cognitive_scores
        }
    
    def get_biomarkers(self) -> Dict:
        """Get dementia-specific biomarkers"""
        return {
            'mmse_score': round(self.cognitive_scores.mmse, 1),
            'moca_score': round(self.cognitive_scores.moca, 1),
            'disease_stage': self.cognitive_scores.get_disease_stage(),
            'acetylcholine_level_percent': round(self.acetylcholine_level * 100, 1),
            'amyloid_beta_burden': round(self.amyloid_beta, 2),
            'tau_tangles_burden': round(self.tau_tangles, 2),
            'hippocampal_activity': round(self.activity['hippocampus'], 2),
            'entorhinal_activity': round(self.activity['entorhinal_cortex'], 2),
            'memory_encoding_score': round(self.cognitive_scores.memory_encoding, 2),
            'memory_retrieval_score': round(self.cognitive_scores.memory_retrieval, 2)
        }
    
    def export_state(self) -> Dict:
        """Export current model state"""
        return {
            'activity': self.activity,
            'cognitive_scores': {
                'mmse': self.cognitive_scores.mmse,
                'moca': self.cognitive_scores.moca,
                'memory_encoding': self.cognitive_scores.memory_encoding,
                'memory_retrieval': self.cognitive_scores.memory_retrieval,
                'executive_function': self.cognitive_scores.executive_function,
                'attention': self.cognitive_scores.attention,
                'disease_stage': self.cognitive_scores.get_disease_stage()
            },
            'pathology': {
                'amyloid_beta': self.amyloid_beta,
                'tau_tangles': self.tau_tangles,
                'acetylcholine_level': self.acetylcholine_level
            },
            'biomarkers': self.get_biomarkers(),
            'treatment_history': self.treatment_history
        }


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Dementia Neural Model - Alzheimer's Disease Simulation")
    print("="*60)
    
    # Create model with 3 years disease duration
    model = DementiaNeuralModel(disease_duration_years=3.0)
    
    print("\nInitial State:")
    print(f"Disease Stage: {model.cognitive_scores.get_disease_stage()}")
    print(f"MMSE: {model.cognitive_scores.mmse:.1f}/30")
    print(f"MoCA: {model.cognitive_scores.moca:.1f}/30")
    print(f"Biomarkers: {json.dumps(model.get_biomarkers(), indent=2)}")
    
    print("\n" + "="*60)
    print("Applying DBS to Nucleus Basalis (20 Hz, 3 mA, 90 us)...")
    print("="*60)
    
    result = model.apply_dbs_stimulation(
        target_region='nucleus_basalis',
        amplitude_ma=3.0,
        frequency_hz=20,  # Low frequency for cholinergic enhancement
        pulse_width_us=90,
        duration_s=1.0
    )
    
    print(f"\nPost-stimulation:")
    print(f"MMSE: {result['cognitive_scores'].mmse:.1f}/30")
    print(f"MoCA: {result['cognitive_scores'].moca:.1f}/30")
    print(f"Efficacy: {result['efficacy']:.2%}")
    print(f"Biomarkers: {json.dumps(model.get_biomarkers(), indent=2)}")
    
    print("\n" + "="*60)
    print("Predicting 6-month treatment response...")
    print("="*60)
    
    prediction = model.predict_treatment_response(
        target_region='nucleus_basalis',
        amplitude_ma=3.0,
        frequency_hz=20,
        pulse_width_us=90,
        treatment_months=6
    )
    
    print(f"\nResponse Rate: {prediction['response_rate']:.2%}")
    print(f"Responder: {prediction['responder']}")
    print(f"Final MMSE: {prediction['final_cognitive_scores'].mmse:.1f}/30")
    print(f"Final Disease Stage: {prediction['final_cognitive_scores'].get_disease_stage()}")
    
    print("\nMonthly Progression:")
    for month_data in prediction['monthly_progression']:
        print(f"  Month {month_data['month']}: MMSE={month_data['mmse']:.1f}, "
              f"MoCA={month_data['moca']:.1f}, Stage={month_data['disease_stage']}")
