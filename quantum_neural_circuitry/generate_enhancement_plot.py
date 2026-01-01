
import matplotlib.pyplot as plt
import numpy as np
from server import QuantumCircuitModel
from generate_brain_comparison import create_dementia_model
from prime_math_core import PrimeVortexField

def generate_enhancement_comparison():
    print("Generating Cognitive Enhancement vs Healthy Comparison...")
    
    N = 25
    prime_field = PrimeVortexField(N)
    
    # 1. Healthy Baseline
    healthy = QuantumCircuitModel(num_qubits=N)
    # Ensure standard connectivity
    for e in healthy.topology.edges():
        if e not in healthy.entanglements:
            healthy.entanglements[e] = 0.6
    healthy_weights = list(healthy.entanglements.values())
    
    # 2. Enhanced State
    # Start from healthy and BOOST
    enhanced = QuantumCircuitModel(num_qubits=N)
    for e in enhanced.topology.edges():
         if e not in enhanced.entanglements:
            enhanced.entanglements[e] = 0.6
            
    # Apply Hyper-Criticality
    hyper_weights = prime_field.optimize_for_hyper_criticality(enhanced.entanglements)
    enhanced.entanglements.update(hyper_weights)
    enhanced_weights = list(enhanced.entanglements.values())
    
    # 3. Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(healthy_weights, bins=20, alpha=0.5, color='green', label='Healthy Baseline (Max ~1.0)', density=True)
    ax.hist(enhanced_weights, bins=20, alpha=0.5, color='gold', label='Cognitive Enhancement (Max >1.0)', density=True)
    
    ax.set_title("Synaptic Strength: Healthy vs Enhanced", fontsize=16, fontweight='bold')
    ax.set_xlabel("Entanglement Strength (J)", fontsize=12)
    ax.axvline(x=1.0, color='red', linestyle='--', label='Biological Limit (Standard)')
    
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.savefig("cognitive_enhancement_plot.png", dpi=300)
    print("Saved cognitive_enhancement_plot.png")

if __name__ == "__main__":
    generate_enhancement_comparison()
