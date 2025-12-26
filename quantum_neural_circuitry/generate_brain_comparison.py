import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import json
from server import DementiaTreatmentModel, ethics_board
from server import QuantumCircuitModel

def create_dementia_model(num_qubits=20):
    """
    Simulates a brain with dementia:
    - Reduced connectivity (edges removed)
    - Lower entanglement strength (synaptic degradation)
    """
    model = DementiaTreatmentModel(num_qubits=num_qubits)
    
    # Degradation: Remove 40% of edges to simulate synaptic loss
    edges = list(model.topology.edges())
    num_to_remove = int(len(edges) * 0.4)
    edges_to_remove = random.sample(edges, num_to_remove)
    model.topology.remove_edges_from(edges_to_remove)
    
    # Update entanglements dictionary
    # Also weaken remaining connections
    new_entanglements = {}
    for u, v in model.topology.edges():
        # Original strength or default
        original_strength = model.entanglements.get((u, v), 0.5)
        # Weaken by 50% on average
        new_entanglements[(u, v)] = original_strength * random.uniform(0.3, 0.7)
        
    model.entanglements = new_entanglements
    return model

def plot_circuit(model, title, filename):
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(model.topology, seed=42) # Consistent layout strategy
    
    # Draw nodes
    nx.draw_networkx_nodes(model.topology, pos, 
                           node_size=600, 
                           node_color='#00f3ff' if "Normal" in title else '#ff4444', 
                           edgecolors='#fff',
                           linewidths=2)
    
    # Draw edges
    if model.entanglements:
        weights = [model.entanglements.get((u, v), 0.1) * 4 for u, v in model.topology.edges()]
        colors = [model.entanglements.get((u, v), 0.1) for u, v in model.topology.edges()]
    else:
        weights = 1
        colors = 'gray'

    nx.draw_networkx_edges(model.topology, pos, 
                           width=weights, 
                           edge_color=colors,
                           edge_cmap=plt.cm.cool if "Normal" in title else plt.cm.Reds,
                           alpha=0.7)
    
    # Labels
    nx.draw_networkx_labels(model.topology, pos, font_color='white', font_weight='bold')
    
    plt.title(title, fontsize=24, color='#333333')
    
    # Stats box
    connectivity = len(model.topology.edges()) / (model.num_qubits * (model.num_qubits-1) / 2)
    avg_strength = np.mean(list(model.entanglements.values())) if model.entanglements else 0
    
    stats = (f"Qubits: {model.num_qubits}\n"
             f"Connections: {len(model.topology.edges())}\n"
             f"Avg Entanglement: {avg_strength:.2f}")
             
    plt.text(0.95, 0.05, stats, 
             horizontalalignment='right', 
             verticalalignment='bottom', 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ccc', boxstyle='round'))

    plt.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

if __name__ == "__main__":
    print("Initializing Quantum Neural Circuitry Models...")
    
    # 1. Healthy Brain
    healthy_model = QuantumCircuitModel(num_qubits=20)
    print("Generating Healthy Brain Schematic...")
    plot_circuit(healthy_model, "Normal Neural Circuitry (Healthy)", "healthy_brain_schematic.png")
    
    # 2. Dementia Brain
    dementia_model = create_dementia_model(num_qubits=20)
    print("Generating Dementia Brain Schematic...")
    plot_circuit(dementia_model, "Degraded Neural Circuitry (Dementia)", "dementia_brain_schematic.png")
    
    print("Comparisons Generated.")

    # ------------------------------------------------------------
    # 3. Post‑treatment Brain Schematic
    # ------------------------------------------------------------
    print("Generating Post‑Treatment Brain Schematic...")
    # Ensure ethical consent is granted (required before treatment)
    ethics_board.grant_consent("STANDARD")
    # Apply a cognitive treatment with moderate intensity
    treatment_type = "cognitive"
    intensity = 0.6
    dementia_model.apply_treatment(treatment_type, intensity)
    # Let the system evolve a few steps to integrate the treatment effects
    for _ in range(5):
        dementia_model.step()
    # Plot the resulting post‑treatment connectivity
    plot_circuit(dementia_model, "Post‑Treatment Neural Circuitry", "post_treatment_brain_schematic.png")
    print("Post‑Treatment schematic saved.")
    # Export the post‑treatment connectivity data to JSON
    connectivity_data = dementia_model.get_state()
    with open("post_treatment_connectivity.json", "w") as f:
        json.dump(connectivity_data, f, indent=2)
    print("Post‑treatment connectivity data saved to post_treatment_connectivity.json.")

    # Apply treatment and generate post‑treatment schematic (single block)
    print("Generating Post‑Treatment Brain Schematic...")
    ethics_board.grant_consent("STANDARD")
    treatment_type = "cognitive"
    intensity = 0.6
    dementia_model.apply_treatment(treatment_type, intensity)
    for _ in range(5):
        dementia_model.step()
    plot_circuit(dementia_model, "Post‑Treatment Neural Circuitry", "post_treatment_brain_schematic.png")
    print("Post‑Treatment schematic saved.")
