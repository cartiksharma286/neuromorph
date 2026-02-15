
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from server import QuantumCircuitModel
from prime_math_core import PrimeVortexField

def generate_enhancement_3d():
    print("Generating 3D Cognitive Enhancement Projection...")
    
    # 1. Initialize Healthy Model as base
    N = 30
    model = QuantumCircuitModel(num_qubits=N)
    prime_field = PrimeVortexField(N)
    
    # Ensure decent starting connectivity
    for i in range(N):
        u, v = i, (i+1)%N
        if not model.topology.has_edge(u, v):
            model.topology.add_edge(u, v)
        model.entanglements[(u, v)] = 0.6
        
    # 2. Apply Cognitive Enhancement (Hyper-Criticality)
    print("Activating Twin Prime Super-Hubs...")
    hyper_weights = prime_field.optimize_for_hyper_criticality(model.entanglements)
    model.entanglements.update(hyper_weights)
    
    # 3. 3D Visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dark background for "Cyberpunk/High-Tech" feel
    ax.set_facecolor('#050510') 
    fig.patch.set_facecolor('#050510')
    
    pos = nx.spring_layout(model.topology, dim=3, seed=99)
    
    # Draw Nodes (Neurons/Qubits)
    xs = [pos[n][0] for n in model.topology.nodes()]
    ys = [pos[n][1] for n in model.topology.nodes()]
    zs = [pos[n][2] for n in model.topology.nodes()]
    
    # Color nodes by prime nature? 
    # Gold for Twin Primes, Silver for others
    node_colors = []
    sizes = []
    
    primes = prime_field.primes
    for n in model.topology.nodes():
        p_val = primes[n] if n < len(primes) else 0
        is_twin = any(abs(p_val - p) == 2 for p in primes[:N])
        
        if is_twin:
            node_colors.append('#ffd700') # Gold
            sizes.append(200)
        else:
            node_colors.append('#a0a0ff') # Light Blue
            sizes.append(80)
            
    ax.scatter(xs, ys, zs, c=node_colors, s=sizes, edgecolors='white', alpha=0.9)
    
    # Draw Edges (Synapses)
    for u, v in model.topology.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        z = [pos[u][2], pos[v][2]]
        
        w = model.entanglements.get((u, v), 0.1)
        
        # Color Logic
        if w > 1.0:
            # Super-Critical Link
            c = '#ffd700' # Gold
            alpha = 0.9
            width = 3.5
        elif w > 0.7:
            # Strong Link
            c = '#00f3ff' # Cyan
            alpha = 0.6
            width = 2.0
        else:
            # Normal Link
            c = '#444455'
            alpha = 0.3
            width = 1.0
            
        ax.plot(x, y, z, c=c, alpha=alpha, linewidth=width)
        
    ax.set_title("Cognitive Enhancement: Twin Prime Hyper-Criticality", fontsize=16, color='white', fontweight='bold')
    ax.axis('off')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#ffd700', lw=3, label='Super-Critical (>100% Connectivity)'),
        Line2D([0], [0], color='#00f3ff', lw=2, label='Healthy Coherence'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffd700', markersize=10, label='Twin Prime Node')
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor='#111', edgecolor='white', labelcolor='white')
    
    outfile = "3d_projection_cognitive_enhancement.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, facecolor='#050510')
    print(f"Saved {outfile}")

if __name__ == "__main__":
    generate_enhancement_3d()
