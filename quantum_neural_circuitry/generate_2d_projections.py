
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from generate_brain_comparison import create_dementia_model
from prime_math_core import PrimeVortexField

def generate_projections():
    print("Generating 2D Projections with Hebbian Amplification...")
    
    # 1. Initialize and Repair Model
    model = create_dementia_model(num_qubits=24)
    prime_field = PrimeVortexField(24)
    
    # Apply Prime-Hebbian Repair
    optimized_weights = prime_field.optimize_entanglement_distribution(model.entanglements)
    model.entanglements.update(optimized_weights)
    
    for (u, v) in list(model.entanglements.keys()):
        w = model.entanglements[(u, v)]
        model.entanglements[(u, v)] = prime_field.calculate_hebbian_prime_factor(u, v, model.qubits, w)

    # 2. Compute 3D Layout
    pos_3d = nx.spring_layout(model.topology, dim=3, seed=137)
    
    # 3. Create Multi-View Plot
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("2D Tomographic Projections of Hebbian-Amplified Circuitry", fontsize=16, fontweight='bold', color='#333')
    
    planes = [
        ('XY Projection (Axial)', 0, 1),
        ('XZ Projection (Coronal)', 0, 2),
        ('YZ Projection (Sagittal)', 1, 2)
    ]
    
    for i, (title, dim1, dim2) in enumerate(planes):
        ax = fig.add_subplot(1, 3, i+1)
        ax.set_facecolor('#f8f9fa')
        
        # Draw Edges
        for u, v in model.topology.edges():
            # Project coords
            pt1 = (pos_3d[u][dim1], pos_3d[u][dim2])
            pt2 = (pos_3d[v][dim1], pos_3d[v][dim2])
            
            weight = model.entanglements.get((u, v), 0.1)
            
            # Color map: Red (low) -> Cyan/Blue (High Hebbian)
            color = plt.cm.cool(weight)
            alpha = 0.2 + (weight * 0.8)
            width = weight * 2.5
            
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, alpha=alpha, linewidth=width, zorder=1)
            
        # Draw Nodes
        xs = [pos_3d[n][dim1] for n in model.topology.nodes()]
        ys = [pos_3d[n][dim2] for n in model.topology.nodes()]
        
        # Color nodes by degree/centrality
        degrees = [val for (node, val) in model.topology.degree()]
        ax.scatter(xs, ys, s=[d*50 for d in degrees], c=degrees, cmap='viridis', edgecolors='white', zorder=2)
        
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("2d_projections_hebbian.png", dpi=300)
    print("Saved 2d_projections_hebbian.png")

if __name__ == "__main__":
    generate_projections()
