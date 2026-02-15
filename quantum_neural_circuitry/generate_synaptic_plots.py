
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from generate_brain_comparison import create_dementia_model
from prime_math_core import PrimeVortexField

def generate_plots():
    print("Generating Post-Synaptic Treatment Plots...")
    
    # 1. Initialize Degraded Model
    model = create_dementia_model(num_qubits=20)
    prime_field = PrimeVortexField(20)
    
    # 0. Initialize Healthy Model (Baseline)
    from server import QuantumCircuitModel
    healthy_model = QuantumCircuitModel(num_qubits=20)
    # Ensure healthy model is fully connected/weighted
    for e in healthy_model.topology.edges():
        if e not in healthy_model.entanglements:
            healthy_model.entanglements[e] = 0.5
    healthy_weights = list(healthy_model.entanglements.values())

    # Capture Initial Degraded State
    initial_weights = list(model.entanglements.values())
    
    # 2. Apply Prime Resonance Repair
    print("Applying Prime Resonance Repair...")
    optimized_weights = prime_field.optimize_entanglement_distribution(model.entanglements)
    model.entanglements.update(optimized_weights)
    
    # Apply Hebbian
    for (u, v) in list(model.entanglements.keys()):
        w = model.entanglements[(u, v)]
        model.entanglements[(u, v)] = prime_field.calculate_hebbian_prime_factor(u, v, model.qubits, w)
        
    final_weights = list(model.entanglements.values())
    
    # 3. Generate Plot
    fig = plt.figure(figsize=(14, 8))
    
    # Subplot 1: Synaptic Weight Distribution (Pre vs Post)
    ax1 = fig.add_subplot(1, 2, 1)
    
    # Histogram
    # Histogram
    ax1.hist(healthy_weights, bins=15, alpha=0.3, label='Healthy Baseline', color='#00ff88', density=True, histtype='stepfilled')
    ax1.hist(initial_weights, bins=15, alpha=0.5, label='Degraded State (Dementia)', color='#ff4444', density=True)
    ax1.hist(final_weights, bins=15, alpha=0.5, label='Repaired State (Prime Resonance)', color='#00f3ff', density=True)
    
    ax1.set_title("Synaptic Weight Distribution", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Entanglement Strength (J)", fontsize=12)
    ax1.set_ylabel("Probability Density", fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.2)
    
    # Subplot 2: 3D Topology
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    pos = nx.spring_layout(model.topology, dim=3, seed=42)
    
    # Draw Nodes
    xs = [pos[n][0] for n in model.topology.nodes()]
    ys = [pos[n][1] for n in model.topology.nodes()]
    zs = [pos[n][2] for n in model.topology.nodes()]
    ax2.scatter(xs, ys, zs, c='#00f3ff', s=100, edgecolors='w')
    
    # Draw Edges (colored by strength)
    for u, v in model.topology.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        z = [pos[u][2], pos[v][2]]
        w = model.entanglements.get((u, v), 0.1)
        ax2.plot(x, y, z, c=plt.cm.cool(w), alpha=0.3 + w*0.5, linewidth=w*3)
        
    ax2.set_title("Reconstructed 3D Topology", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig("post_synaptic_treatment_plot.png", dpi=300, facecolor='#f4f4f4')
    print("Saved post_synaptic_treatment_plot.png")

if __name__ == "__main__":
    try:
        generate_plots()
    except Exception as e:
        print(f"Error: {e}")
