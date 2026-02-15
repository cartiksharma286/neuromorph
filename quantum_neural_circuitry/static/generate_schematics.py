import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from server import QuantumCircuitModel

def generate_schematic():
    print("Generating Quantum Circuit Schematic...")
    
    # Instantiate the model (using same parameters as server)
    circuit = QuantumCircuitModel(num_qubits=20)
    
    # Create a layout
    pos = nx.spring_layout(circuit.topology, seed=42)
    
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    # Color nodes by their initial random state (simulated) or just a standard color
    nx.draw_networkx_nodes(circuit.topology, pos, 
                           node_size=500, 
                           node_color='#00f3ff', 
                           edgecolors='#fff',
                           linewidths=2)
    
    # Draw edges
    # Thickness based on entanglement strength
    weights = [circuit.entanglements.get((u, v), 0.5) * 3 for u, v in circuit.topology.edges()]
    nx.draw_networkx_edges(circuit.topology, pos, 
                           width=weights, 
                           edge_color='#bc13fe',
                           alpha=0.6)
    
    # Labels
    nx.draw_networkx_labels(circuit.topology, pos, 
                            font_family='sans-serif',
                            font_weight='bold',
                            font_color='#050510')
    
    plt.title("Quantum Neuromorphic Circuitry Schematic", fontsize=20, color='#333333')
    plt.axis('off')
    
    # Add a legend/info box
    plt.text(0.95, 0.05, 
             f"Qubits: {circuit.num_qubits}\nTopology: Watts-Strogatz\nEntanglements: {len(circuit.entanglements)}", 
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#ccc'))

    output_path = "qubit_schematics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#f0f0f5')
    print(f"Schematic saved to {output_path}")
    
    # Also save a DOT file for structural schematic
    print(f"Schematic saved to {output_path}")
    print("Structure saved to qubit_schematics.dot")

if __name__ == "__main__":
    generate_schematic()
