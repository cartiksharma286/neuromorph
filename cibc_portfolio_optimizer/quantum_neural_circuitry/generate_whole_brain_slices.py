
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from generate_brain_comparison import create_dementia_model
from server import QuantumCircuitModel
from prime_math_core import PrimeVortexField

def generate_whole_brain_slices():
    print("Generating Whole Brain Connectivity Slices (Dual Encoded)...")
    
    # 1. Initialize Models
    N = 30 # Higher res for slice detail
    dementia_model = create_dementia_model(num_qubits=N)
    prime_field = PrimeVortexField(N)
    
    # Healthy Baseline for comparison metrics
    healthy_model = QuantumCircuitModel(num_qubits=N) 
    
    # 2. Apply Dual Encoding Restoration
    print("Applying structural and functional dual encoding...")
    dual_weights = prime_field.apply_dual_prime_encoding(dementia_model.entanglements, dementia_model.qubits)
    dementia_model.entanglements.update(dual_weights)
    
    # Topological Repair (Gliders)
    new_edges = prime_field.calculate_repair_vector(dementia_model.topology, 0.9)
    for u, v in new_edges:
        if not dementia_model.topology.has_edge(u, v):
            dementia_model.topology.add_edge(u, v)
            dementia_model.entanglements[(u, v)] = 0.6
            
    # Relayout for 3D visualization
    pos = nx.spring_layout(dementia_model.topology, dim=3, seed=88)
    
    # 3. Visualization Construction
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Whole Brain Connectivity: Dual Encoded Restoration", fontsize=18, fontweight='bold', y=0.95)
    
    # Main 3D View (Left)
    ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Plot 3D Edges
    for u, v in dementia_model.topology.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        z = [pos[u][2], pos[v][2]]
        w = dementia_model.entanglements.get((u, v), 0.1)
        
        # Dual Color Map: Cyan for Class 1 mod 4, Magenta for Class 3 mod 4 (intra), White for Mixed
        p_u = prime_field.primes[u] % 4
        p_v = prime_field.primes[v] % 4
        
        if p_u == p_v:
             c = '#00f3ff' if p_u == 1 else '#ff00ff' # Cognitive vs Sensory channels
        else:
             c = '#ffffff' # Integration fibers
             
        ax_3d.plot(x, y, z, c=c, alpha=0.3 + w*0.6, linewidth=w*2)
        
    # Plot Nodes
    xs = [pos[n][0] for n in dementia_model.topology.nodes()]
    ys = [pos[n][1] for n in dementia_model.topology.nodes()]
    zs = [pos[n][2] for n in dementia_model.topology.nodes()]
    ax_3d.scatter(xs, ys, zs, s=100, c='white', edgecolors='black')
    
    ax_3d.set_title("Restored Dual-Encoded Manifold (3D)", fontsize=14)
    ax_3d.axis('off')
    
    # Slices Column (Right)
    # Filter nodes based on geometric planes to simulate MRI slices
    
    # Axial Slice (Z approx 0)
    ax_axial = fig.add_subplot(3, 2, 2)
    plot_slice(ax_axial, dementia_model, pos, 2, 0.2, "Axial Slice (Z=0)")
    
    # Coronal Slice (Y approx 0)
    ax_coronal = fig.add_subplot(3, 2, 4)
    plot_slice(ax_coronal, dementia_model, pos, 1, 0.2, "Coronal Slice (Y=0)")
    
    # Sagittal Slice (X approx 0)
    ax_sagittal = fig.add_subplot(3, 2, 6)
    plot_slice(ax_sagittal, dementia_model, pos, 0, 0.2, "Sagittal Slice (X=0)")
    
    plt.tight_layout()
    plt.savefig("whole_brain_dual_encoding.png", dpi=300, facecolor='#111111')
    print("Saved whole_brain_dual_encoding.png")

def plot_slice(ax, model, pos, axis_idx, thickness, title):
    """
    Plots a 2D slice of the 3D graph. Nodes are included if their coordinate
    in axis_idx is within +/- thickness of 0.
    """
    ax.set_facecolor('#111')
    
    nodes_in_slice = []
    for n, coords in pos.items():
        if abs(coords[axis_idx]) < thickness:
            nodes_in_slice.append(n)
            
    subgraph = model.topology.subgraph(nodes_in_slice)
    
    # Project to 2D (remove slice axis)
    pos_2d = {}
    dims = [0, 1, 2]
    dims.remove(axis_idx)
    
    for n in nodes_in_slice:
        pos_2d[n] = [pos[n][dims[0]], pos[n][dims[1]]]
        
    # Draw
    nx.draw_networkx_nodes(subgraph, pos_2d, ax=ax, node_size=80, node_color='#00f3ff')
    
    # Draw edges only if both nodes in slice
    for u, v in subgraph.edges():
        if u in nodes_in_slice and v in nodes_in_slice:
            w = model.entanglements.get((u, v), 0.1)
            nx.draw_networkx_edges(subgraph, pos_2d, ax=ax, edgelist=[(u, v)], width=w*3, edge_color='white', alpha=0.5)
            
    ax.set_title(title, color='white', fontsize=10)
    ax.axis('off')

if __name__ == "__main__":
    generate_whole_brain_slices()
