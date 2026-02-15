"""
PTSD Neural Pathway Model
Simulates neural dynamics and treatment response for DBS optimization
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json


@dataclass
class BrainRegion:
    """Brain region configuration"""
    name: str
    baseline_activity: float  # 0-1
    connectivity: Dict[str, float]  # Connections to other regions
    stimulation_sensitivity: float  # Response to DBS


@dataclass
class PTSDSymptoms:
    """PTSD symptom severity scores (0-1)"""
    hyperarousal: float = 0.7
    re_experiencing: float = 0.6
    avoidance: float = 0.5
    negative_cognition: float = 0.6
    
    def total_severity(self) -> float:
        """Calculate total symptom severity"""
        return (self.hyperarousal + self.re_experiencing + 
                self.avoidance + self.negative_cognition) / 4


class PTSDNeuralModel:
    """
    Computational model of PTSD neural pathways and DBS effects
    Based on fear conditioning and extinction circuits
    """
    
    def __init__(self):
        # Initialize brain regions involved in PTSD
        self.regions = {
            'amygdala': BrainRegion(
                name='Basolateral Amygdala (BLA)',
                baseline_activity=0.8,  # Hyperactive in PTSD
                connectivity={
                    'hippocampus': 0.6,
                    'vmPFC': -0.4,  # Negative = inhibitory
                    'hypothalamus': 0.7
                },
                stimulation_sensitivity=0.9
            ),
            'hippocampus': BrainRegion(
                name='Hippocampus (CA1)',
                baseline_activity=0.5,
                connectivity={
                    'amygdala': 0.5,
                    'vmPFC': 0.6,
                    'cortex': 0.4
                },
                stimulation_sensitivity=0.7
            ),
            'vmPFC': BrainRegion(
                name='Ventromedial Prefrontal Cortex',
                baseline_activity=0.3,  # Hypoactive in PTSD
                connectivity={
                    'amygdala': -0.8,  # Should inhibit amygdala
                    'hippocampus': 0.5,
                    'dlPFC': 0.6
                },
                stimulation_sensitivity=0.8
            ),
            'hypothalamus': BrainRegion(
                name='Hypothalamus (HPA axis)',
                baseline_activity=0.7,
                connectivity={
                    'amygdala': 0.6,
                    'vmPFC': -0.3
                },
                stimulation_sensitivity=0.5
            )
        }
        
        # Current neural activity levels
        self.activity = {
            region: config.baseline_activity 
            for region, config in self.regions.items()
        }
        
        # Symptom state
        self.symptoms = PTSDSymptoms()
        
        # Treatment history
        self.treatment_history = []
        
    def simulate_neural_dynamics(self, time_steps: int = 100, dt: float = 0.01):
        """
        Simulate neural dynamics using coupled differential equations
        dA/dt = -A + f(sum(w_ij * A_j) + I_ext)
        """
        activity_history = {region: [] for region in self.regions}
        
        for t in range(time_steps):
            new_activity = {}
            
            for region, config in self.regions.items():
                # Calculate input from connected regions
                network_input = 0
                for connected_region, weight in config.connectivity.items():
                    if connected_region in self.activity:
                        network_input += weight * self.activity[connected_region]
                
                # Neural dynamics with sigmoid activation
                activation = self._sigmoid(network_input + config.baseline_activity)
                
                # Update with time constant
                tau = 10.0  # Time constant
                dA = (-self.activity[region] + activation) / tau
                new_activity[region] = self.activity[region] + dt * dA
                
                # Clip to [0, 1]
                new_activity[region] = np.clip(new_activity[region], 0, 1)
                
                activity_history[region].append(new_activity[region])
            
            self.activity = new_activity
        
        return activity_history
    
    def apply_dbs_stimulation(self, target_region: str, amplitude_ma: float, 
                             frequency_hz: float, pulse_width_us: float, 
                             duration_s: float = 1.0):
        """
        Apply DBS stimulation to target region
        Models both direct effects and network propagation
        """
        if target_region not in self.regions:
            raise ValueError(f"Unknown region: {target_region}")
        
        # Calculate stimulation effect
        stim_effect = self._calculate_stimulation_effect(
            amplitude_ma, frequency_hz, pulse_width_us
        )
        
        # Apply to target region
        sensitivity = self.regions[target_region].stimulation_sensitivity
        direct_effect = stim_effect * sensitivity
        
        # Modulate activity based on stimulation
        # High-frequency DBS typically reduces activity
        if frequency_hz > 100:
            self.activity[target_region] *= (1 - direct_effect * 0.3)
        else:
            # Low-frequency can increase activity
            self.activity[target_region] *= (1 + direct_effect * 0.2)
        
        # Clip to valid range
        self.activity[target_region] = np.clip(self.activity[target_region], 0, 1)
        
        # Simulate network effects
        self.simulate_neural_dynamics(time_steps=int(duration_s * 100))
        
        # Update symptoms based on new activity levels
        self._update_symptoms()
        
        # Record treatment
        self.treatment_history.append({
            'target': target_region,
            'amplitude_ma': amplitude_ma,
            'frequency_hz': frequency_hz,
            'pulse_width_us': pulse_width_us,
            'duration_s': duration_s,
            'symptom_severity': self.symptoms.total_severity(),
            'activity_state': self.activity.copy()
        })
        
        return {
            'activity': self.activity.copy(),
            'symptoms': self.symptoms,
            'efficacy': self._calculate_efficacy()
        }
    
    def _calculate_stimulation_effect(self, amplitude_ma: float, 
                                     frequency_hz: float, 
                                     pulse_width_us: float) -> float:
        """
        Calculate effective stimulation strength
        Based on charge per phase: Q = I * PW
        """
        charge_per_phase = amplitude_ma * (pulse_width_us / 1000)  # Convert to ms
        
        # Frequency modulation (higher freq = more pulses)
        freq_factor = np.log(frequency_hz + 1) / np.log(200)
        
        # Combined effect (normalized to 0-1)
        effect = (charge_per_phase / 10.0) * freq_factor
        return np.clip(effect, 0, 1)
    
    def _update_symptoms(self):
        """Update PTSD symptoms based on neural activity"""
        # Hyperarousal correlates with amygdala and hypothalamus activity
        self.symptoms.hyperarousal = (
            0.6 * self.activity['amygdala'] + 
            0.4 * self.activity['hypothalamus']
        )
        
        # Re-experiencing correlates with amygdala and hippocampus
        self.symptoms.re_experiencing = (
            0.7 * self.activity['amygdala'] + 
            0.3 * self.activity['hippocampus']
        )
        
        # Avoidance inversely correlates with vmPFC
        self.symptoms.avoidance = 1.0 - (
            0.8 * self.activity['vmPFC'] + 
            0.2 * (1 - self.activity['amygdala'])
        )
        
        # Negative cognition inversely correlates with vmPFC and hippocampus
        self.symptoms.negative_cognition = 1.0 - (
            0.5 * self.activity['vmPFC'] + 
            0.5 * self.activity['hippocampus']
        )
        
        # Clip all to [0, 1]
        self.symptoms.hyperarousal = np.clip(self.symptoms.hyperarousal, 0, 1)
        self.symptoms.re_experiencing = np.clip(self.symptoms.re_experiencing, 0, 1)
        self.symptoms.avoidance = np.clip(self.symptoms.avoidance, 0, 1)
        self.symptoms.negative_cognition = np.clip(self.symptoms.negative_cognition, 0, 1)
    
    def _calculate_efficacy(self) -> float:
        """Calculate treatment efficacy (symptom reduction)"""
        baseline_severity = 0.6  # Typical PTSD severity
        current_severity = self.symptoms.total_severity()
        efficacy = (baseline_severity - current_severity) / baseline_severity
        return max(0, efficacy)  # Ensure non-negative
    
    def _sigmoid(self, x: float, gain: float = 1.0) -> float:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-gain * x))
    
    def predict_treatment_response(self, target_region: str, 
                                  amplitude_ma: float, 
                                  frequency_hz: float, 
                                  pulse_width_us: float,
                                  treatment_weeks: int = 12) -> Dict:
        """
        Predict long-term treatment response
        Simulates weekly progression
        """
        weekly_results = []
        
        # Reset to baseline
        self._reset_to_baseline()
        
        for week in range(treatment_weeks):
            # Apply weekly stimulation (simplified)
            result = self.apply_dbs_stimulation(
                target_region, amplitude_ma, frequency_hz, 
                pulse_width_us, duration_s=1.0
            )
            
            # Add neuroplasticity effects (gradual improvement)
            plasticity_factor = 1 - np.exp(-week / 4.0)  # Exponential learning
            
            # Enhance vmPFC activity (extinction learning)
            self.activity['vmPFC'] = min(
                0.7, 
                self.activity['vmPFC'] + 0.05 * plasticity_factor
            )
            
            # Reduce amygdala hyperactivity
            self.activity['amygdala'] = max(
                0.4,
                self.activity['amygdala'] - 0.03 * plasticity_factor
            )
            
            self._update_symptoms()
            
            weekly_results.append({
                'week': week + 1,
                'symptom_severity': self.symptoms.total_severity(),
                'hyperarousal': self.symptoms.hyperarousal,
                're_experiencing': self.symptoms.re_experiencing,
                'avoidance': self.symptoms.avoidance,
                'negative_cognition': self.symptoms.negative_cognition,
                'activity': self.activity.copy()
            })
        
        # Calculate overall response
        initial_severity = weekly_results[0]['symptom_severity']
        final_severity = weekly_results[-1]['symptom_severity']
        response_rate = (initial_severity - final_severity) / initial_severity
        
        return {
            'weekly_progression': weekly_results,
            'response_rate': response_rate,
            'responder': response_rate > 0.5,  # >50% reduction = responder
            'final_symptoms': self.symptoms
        }
    
    def _reset_to_baseline(self):
        """Reset model to baseline PTSD state"""
        for region, config in self.regions.items():
            self.activity[region] = config.baseline_activity
        self.symptoms = PTSDSymptoms()
        self.treatment_history = []
    
    def get_biomarkers(self) -> Dict:
        """
        Get physiological biomarkers associated with PTSD
        """
        # Heart rate variability (lower in PTSD, correlates with vmPFC)
        hrv = 50 + 50 * self.activity['vmPFC']  # ms (RMSSD)
        
        # Cortisol (higher in PTSD, correlates with HPA axis)
        cortisol = 10 + 20 * self.activity['hypothalamus']  # μg/dL
        
        # Skin conductance (higher in PTSD, correlates with amygdala)
        skin_conductance = 5 + 15 * self.activity['amygdala']  # μS
        
        # Sleep quality (worse in PTSD, inverse of hyperarousal)
        sleep_quality = 1 - self.symptoms.hyperarousal
        
        return {
            'heart_rate_variability_ms': round(hrv, 1),
            'cortisol_ug_dl': round(cortisol, 1),
            'skin_conductance_us': round(skin_conductance, 1),
            'sleep_quality_0_1': round(sleep_quality, 2),
            'amygdala_activity': round(self.activity['amygdala'], 2),
            'vmPFC_activity': round(self.activity['vmPFC'], 2)
        }
    
    def optimize_parameters(self, target_region: str, 
                          num_trials: int = 50) -> Dict:
        """
        Simple parameter optimization using grid search
        """
        best_params = None
        best_efficacy = -1
        
        # Parameter ranges
        amplitudes = np.linspace(1.0, 6.0, 5)
        frequencies = [60, 130, 185]
        pulse_widths = [60, 90, 120, 150]
        
        results = []
        
        for amp in amplitudes:
            for freq in frequencies:
                for pw in pulse_widths:
                    # Reset and test
                    self._reset_to_baseline()
                    result = self.apply_dbs_stimulation(
                        target_region, amp, freq, pw, duration_s=1.0
                    )
                    
                    efficacy = result['efficacy']
                    
                    results.append({
                        'amplitude_ma': amp,
                        'frequency_hz': freq,
                        'pulse_width_us': pw,
                        'efficacy': efficacy,
                        'symptom_severity': result['symptoms'].total_severity()
                    })
                    
                    if efficacy > best_efficacy:
                        best_efficacy = efficacy
                        best_params = {
                            'amplitude_ma': amp,
                            'frequency_hz': freq,
                            'pulse_width_us': pw
                        }
        
        return {
            'best_parameters': best_params,
            'best_efficacy': best_efficacy,
            'all_results': sorted(results, key=lambda x: -x['efficacy'])[:10]
        }
    
    def export_state(self) -> Dict:
        """Export current model state"""
        return {
            'activity': self.activity,
            'symptoms': {
                'hyperarousal': self.symptoms.hyperarousal,
                're_experiencing': self.symptoms.re_experiencing,
                'avoidance': self.symptoms.avoidance,
                'negative_cognition': self.symptoms.negative_cognition,
                'total_severity': self.symptoms.total_severity()
            },
            'biomarkers': self.get_biomarkers(),
            'treatment_history': self.treatment_history
        }


if __name__ == "__main__":
    # Example usage
    model = PTSDNeuralModel()
    
    print("Initial State:")
    print(f"Symptoms: {model.symptoms}")
    print(f"Biomarkers: {json.dumps(model.get_biomarkers(), indent=2)}")
    
    print("\n" + "="*60)
    print("Applying DBS to Amygdala (130 Hz, 3 mA, 90 us)...")
    print("="*60)
    
    result = model.apply_dbs_stimulation(
        target_region='amygdala',
        amplitude_ma=3.0,
        frequency_hz=130,
        pulse_width_us=90,
        duration_s=1.0
    )
    
    print(f"\nPost-stimulation:")
    print(f"Symptoms: {result['symptoms']}")
    print(f"Efficacy: {result['efficacy']:.2%}")
    print(f"Biomarkers: {json.dumps(model.get_biomarkers(), indent=2)}")
    
    print("\n" + "="*60)
    print("Predicting 12-week treatment response...")
    print("="*60)
    
    prediction = model.predict_treatment_response(
        target_region='amygdala',
        amplitude_ma=3.0,
        frequency_hz=130,
        pulse_width_us=90,
        treatment_weeks=12
    )
    
    print(f"\nResponse Rate: {prediction['response_rate']:.2%}")
    print(f"Responder: {prediction['responder']}")
    print(f"Final Symptoms: {prediction['final_symptoms']}")
    
    print("\n" + "="*60)
    print("Optimizing parameters...")
    print("="*60)
    
    optimization = model.optimize_parameters(target_region='amygdala')
    print(f"\nBest Parameters: {optimization['best_parameters']}")
    print(f"Best Efficacy: {optimization['best_efficacy']:.2%}")
    print(f"\nTop 5 parameter combinations:")
    for i, result in enumerate(optimization['all_results'][:5], 1):
        print(f"  {i}. {result}")
