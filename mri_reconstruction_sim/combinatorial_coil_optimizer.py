"""
Statistical Combinatorial Coil Optimizer
========================================

Uses statistical combinatorial reasoning to select the optimal subset 
of elements from the Quantum Vascular Coil Library for interventional 
neurovascular procedures.
"""

import numpy as np
import itertools
from quantum_vascular_coils import QUANTUM_VASCULAR_COIL_LIBRARY

class CombinatorialCoilOptimizer:
    """
    Optimizes the selection and weighting of RF coil elements 
    for specific surgical targets using combinatorial reasoning.
    """
    
    def __init__(self, target_region='Circle of Willis'):
        self.target_region = target_region
        self.library = QUANTUM_VASCULAR_COIL_LIBRARY
        
    def calculate_combinatorial_snr(self, subset_indices, pos_x=64, pos_y=64):
        """
        Calculates the effective SNR for a given combination of coil elements.
        
        SNR_eff = sqrt(Σ Sensitivity_i^2) / Noise_floor
        """
        total_sensitivity_sq = 0
        for idx in subset_indices:
            coil_cls = self.library.get(idx)
            if not coil_cls: continue
            
            coil = coil_cls()
            # Simulated sensitivity at target position
            # Higher weight for Conformal and Neurovascular designs
            base_sens = 1.0
            if 'Conformal' in coil.name or 'Neurovascular' in coil.name:
                base_sens = 1.8
            
            # Simple distance-based falloff simulation
            dist = np.sqrt((pos_x - 64)**2 + (pos_y - 64)**2)
            total_sensitivity_sq += (base_sens * np.exp(-dist/128))**2
            
        return np.sqrt(total_sensitivity_sq)

    def optimize_configuration(self, max_elements=4):
        """
        Performs combinatorial search for the optimal coil combination.
        """
        best_snr = 0
        best_subset = []
        
        # We don't want to check ALL combinations if the library is large (29 coils)
        # Instead, we filter by 'best candidates' for neurovascular first
        candidates = [idx for idx, cls in self.library.items() 
                      if any(kw in cls().name for kw in ['Vascular', 'Neuro', 'Conformal', 'Quantum'])]
        
        # Combinatorial reasoning: Check subsets of size 1 to max_elements
        for r in range(1, max_elements + 1):
            for subset in itertools.combinations(candidates[:10], r):
                current_snr = self.calculate_combinatorial_snr(subset)
                if current_snr > best_snr:
                    best_snr = current_snr
                    best_subset = subset
                    
        return {
            'target': self.target_region,
            'optimal_subset': [self.library[idx]().name for idx in best_subset],
            'estimated_snr_gain': f"{best_snr:.2f}",
            'config_ids': list(best_subset),
            'reasoning_path': "Combinatorial Subset Maximization"
        }

if __name__ == "__main__":
    optimizer = CombinatorialCoilOptimizer()
    result = optimizer.optimize_configuration()
    print("Optimization Result:", result)
