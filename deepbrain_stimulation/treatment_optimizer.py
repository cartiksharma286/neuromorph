import numpy as np
import random

class TreatmentProtocolOptimizer:
    def __init__(self, neural_model):
        self.neural_model = neural_model

    def generate_bem_surface_data(self, target_zones=['amygdala']):
        """
        Simulates Boundary Element Method (BEM) surface data for cortical stimulation.
        Returns vertices and field intensities proportional to target zones.
        Now supports 3D coordinates.
        """
        # grid for 2d projection (top-down) & 3d sphere section
        grid_size = 60
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1.2, 1.2, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Mask for brain shape (top-down ellipse)
        mask = (X**2/0.8 + Y**2/1.4) <= 1.0 # 1.4 for slightly longer z-axis in 2D or Y in top-down
        
        # Calculate Z for 3D hemisphere surface (z > 0)
        # ellipsoid equation: x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
        # z = c * sqrt(1 - x^2/a^2 - y^2/b^2)
        # let a=0.9 (width), b=1.2 (length), c=0.8 (height)
        Z = np.zeros_like(X)
        valid_mask = (X**2/0.81 + Y**2/1.44) <= 1.0
        
        # Safe sqrt
        Z[valid_mask] = 0.8 * np.sqrt(np.maximum(0, 1 - X[valid_mask]**2/0.81 - Y[valid_mask]**2/1.44))
        
        # Target Coordinate Mapping (Approximate Top-Down X, Y)
        # X: -1 (Left) to 1 (Right), Y: -1.2 (Post) to 1.2 (Ant)
        coord_map = {
            'amygdala': [(0.6, 0.1), (-0.6, 0.1)], # Bilateral Temporal
            'hippocampus': [(0.5, -0.2), (-0.5, -0.2)], # Bilateral Medial Temporal
            'vmPFC': [(0.0, 1.0)], # Midline Anterior
            'hypothalamus': [(0.1, -0.1), (-0.1, -0.1)], # Deep, center
            'nucleus_basalis': [(0.3, 0.2), (-0.3, 0.2)], # Basal Forebrain
            'dlpfc': [(0.7, 0.8), (-0.7, 0.8)], # Frontal Lateral
            'pcc': [(0.0, -0.6)], # Posterior Cingulate
            'm1': [(0.6, 0.0), (-0.6, 0.0)] # Motor Cortex
        }
        
        total_intensity = np.zeros_like(X)
        
        # Fallback if list empty
        if not target_zones: target_zones = ['amygdala']
        
        for zone in target_zones:
            coords = coord_map.get(zone, [(0,0)])
            for (tx, ty) in coords:
                # Gaussian blob for field projection
                # Broader spread for deep structures radiating to cortex
                sigma = 0.15 if zone in ['vmPFC', 'm1', 'dlpfc'] else 0.25
                blob = np.exp(-((X - tx)**2 + (Y - ty)**2) / sigma)
                total_intensity += (blob * 0.8) # 0.8 max contribution per source
                
        # Add baseline activity
        total_intensity += 0.1
        
        # Noise
        noise = np.random.normal(0, 0.03, total_intensity.shape)
        intensity = np.clip(total_intensity + noise, 0, 1) * valid_mask
        
        # Extract valid points
        valid_indices = valid_mask > 0
        
        return {
            'x': X[valid_indices].tolist(),
            'y': Y[valid_indices].tolist(),
            'z': Z[valid_indices].tolist(),
            'intensity': intensity[valid_indices].tolist(),
            'max_field_v_m': float(1.8 + np.max(intensity) * 1.5), 
            'vta_intersection': float(np.mean(intensity[valid_indices]) * 0.8 + 0.4)
        }

    def generate_protocol_sequence(self, target_zones, total_weeks=12):
        """
        Generates a multi-stage treatment protocol optimized for specific zones with multimodal reasoning.
        Includes BEM and Connectome constraints.
        """
        protocol = []
        
        # Define complex sub-stage structure with Multimodal Reasoning
        structure = [
            {
                'phase': 'Induction',
                'substages': [
                    {
                        'name': 'Connectome Mapping', 
                        'fraction': 0.1, 
                        'rationale': 'DTI-verified tractography alignment for pathway selectivity.'
                    },
                    {
                        'name': 'Desynchronization', 
                        'fraction': 0.3, 
                        'rationale': 'Phase-resetting burst to disrupt pathological oscillatory beta-band activity (13-30Hz).'
                    }
                ]
            },
            {
                'phase': 'Optimization',
                'substages': [
                    {
                        'name': 'Boundary Element Tuning', 
                        'fraction': 0.3, 
                        'rationale': 'FEA-constrained adjustment to minimize current leakage outside target VTA.'
                    },
                    {
                        'name': 'Plasticity Induction', 
                        'fraction': 0.1, 
                        'rationale': 'Hebbian-timed pulses to reinforce cortical-subcortical connectivity.'
                    }
                ]
            },
            {
                'phase': 'Maintenance',
                'substages': [
                    {
                        'name': 'Homeostatic Stabilization', 
                        'fraction': 0.2, 
                        'rationale': 'Low-frequency tonic stimulation to maintain new synaptic weights.'
                    }
                ]
            }
        ]
        
        paradigms = [
            {'name': 'High-Frequency Burst', 'freq': 185, 'width': 60, 'amp_mod': 1.2},
            {'name': 'Standard Continuous', 'freq': 130, 'width': 90, 'amp_mod': 1.0},
            {'name': 'Low-Frequency Cyclic', 'freq': 60, 'width': 120, 'amp_mod': 0.8},
            {'name': 'Theta-Burst Modulation', 'freq': 50, 'width': 200, 'amp_mod': 1.5},
            {'name': 'Adaptive Closed-Loop', 'freq': 'Variable', 'width': 90, 'amp_mod': 1.0}
        ]

        cumulative_efficacy = 0
        current_symptoms = 1.0
        
        # BEM Simulation for the whole protocol
        bem_simulation = self.generate_bem_surface_data(target_zones)
        
        for p_idx, phase_struct in enumerate(structure):
            for sub in phase_struct['substages']:
                # Calculate duration
                stage_weeks = max(1, int(total_weeks * sub['fraction']))
                
                # Logic for paradigm selection based on reasoning
                if 'Connectome' in sub['name']:
                     chosen_p = next(p for p in paradigms if p['name'] == 'Standard Continuous')
                elif 'Desynchronization' in sub['name']:
                     chosen_p = next(p for p in paradigms if p['name'] == 'High-Frequency Burst')
                elif 'Boundary' in sub['name']:
                    chosen_p = next(p for p in paradigms if p['name'] == 'Adaptive Closed-Loop')
                elif 'Plasticity' in sub['name']:
                     chosen_p = next(p for p in paradigms if p['name'] == 'Theta-Burst Modulation')
                else:
                    chosen_p = next(p for p in paradigms if p['name'] == 'Low-Frequency Cyclic')

                
                # Assign target (rotate)
                target = target_zones[p_idx % len(target_zones)]
                
                # Calculate predicted improvement
                base_imp = 0.05 * stage_weeks
                
                stage = {
                    'phase_name': phase_struct['phase'],
                    'substage_name': sub['name'],
                    'rationale': sub['rationale'], # Multimodal reasoning
                    'duration_weeks': stage_weeks,
                    'paradigm': chosen_p['name'],
                    'parameters': {
                        'frequency_hz': chosen_p['freq'],
                        'pulse_width_us': chosen_p['width'],
                        'amplitude_adj': chosen_p['amp_mod']
                    },
                    'target_focus': target,
                    'predicted_improvement': float(min(0.4, base_imp))
                }
                
                current_symptoms *= (1.0 - stage['predicted_improvement'])
                protocol.append(stage)

        return {
            'protocol': protocol,
            'final_projected_symptom_reduction': (1.0 - current_symptoms) * 100,
            'bem_data': bem_simulation,
            'connectome_data': {'pathway_integrity': 0.88, 'structural_connectivity': 0.92}
        }

    def simulate_long_term_outcome(self, protocol_data, condition='ptsd'):
        """
        Simulates statistical distributions (fan chart) of outcomes.
        Returns p5, p25, p50, p75, p95 percentiles.
        """
        weeks = []
        
        # Monte Carlo Simulation parameters
        num_simulations = 200
        
        # Initial State
        start_level = 75.0 if condition == 'dementia' else 100.0
        
        # We simulate multiple paths to generate statistics
        trajectories = np.zeros((num_simulations, sum(s['duration_weeks'] for s in protocol_data) + 1))
        trajectories[:, 0] = start_level
        
        total_time = 0
        current_idx = 0
        
        timeline_meta = [] # To store metadata per week
        
        for stage in protocol_data:
            duration = stage['duration_weeks']
            
            # Base rates
            if condition == 'dementia':
                # Cure rate vs Decline
                cure_rate = (stage['predicted_improvement'] / duration) * 2.0
                decline_rate = 0.05 # Per week baseline
            else:
                improve_rate = (stage['predicted_improvement'] / duration)
            
            for w in range(duration):
                current_idx += 1
                total_time += 1
                
                # Evolve all trajectories
                prev_vals = trajectories[:, current_idx - 1]
                
                if condition == 'dementia':
                    # Stochastic decline
                    natural_progression = 0.05 * (1 + total_time/20.0) + np.random.normal(0, 0.1, num_simulations)
                    # Stochastic treatment
                    treatment = prev_vals * cure_rate * (1 + np.random.normal(0, 0.2, num_simulations))
                    
                    # Net
                    new_vals = prev_vals - treatment + natural_progression
                else:
                    # PTSD
                    improvement = prev_vals * improve_rate * (1 + np.random.normal(0, 0.15, num_simulations))
                    # Occasionally bad weeks
                    stress_events = (np.random.random(num_simulations) < 0.1) * 5.0
                    new_vals = prev_vals - improvement + stress_events
                
                # Clamp
                new_vals = np.clip(new_vals, 0, 100)
                trajectories[:, current_idx] = new_vals
                
                timeline_meta.append({
                    'week': total_time,
                    'phase': stage['phase_name'],
                    'substage': stage['substage_name'],
                    'paradigm': stage['paradigm']
                })
        
        # Calculate percentiles for each week
        for t in range(1, trajectories.shape[1]): # Skip week 0
            vals = trajectories[:, t]
            meta = timeline_meta[t-1]
            weeks.append({
                'week': meta['week'],
                'phase': meta['phase'],
                'substage': meta['substage'],
                'active_paradigm': meta['paradigm'],
                'p05': np.percentile(vals, 5),
                'p25': np.percentile(vals, 25),
                'p50': np.percentile(vals, 50), # Median
                'p75': np.percentile(vals, 75),
                'p95': np.percentile(vals, 95)
            })
            
        return weeks
