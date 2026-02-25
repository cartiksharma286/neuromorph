
import random
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO

class PipelineManager:
    """
    Manages Pipeline Networks (Jamnagar, AB BC)
    Routes, Flow Analysis, and Network Visualization
    """
    def __init__(self):
        # Create Jamnagar Pipeline Graph
        self.jamnagar_graph = nx.DiGraph()
        self.jamnagar_graph.add_edge("Crude Inlet", "Heater", weight=2.0)
        self.jamnagar_graph.add_edge("Heater", "Desalter", weight=1.5)
        self.jamnagar_graph.add_edge("Desalter", "Atmospheric Column", weight=3.0)
        self.jamnagar_graph.add_edge("Atmospheric Column", "Naphtha", weight=1.0)
        self.jamnagar_graph.add_edge("Atmospheric Column", "Kerosene", weight=1.0)
        self.jamnagar_graph.add_edge("Atmospheric Column", "Diesel", weight=1.0)
        self.jamnagar_graph.add_edge("Atmospheric Column", "Bottoms", weight=1.5)
        
        # AB BC Pipeline (Hypothetical long-distance)
        self.abbc_graph = nx.DiGraph()
        nodes = ["Pump Station 1", "Valve 1", "Compressor", "Pump Station 2", "Refinery Terminal"]
        for i in range(len(nodes)-1):
            self.abbc_graph.add_edge(nodes[i], nodes[i+1], weight=random.uniform(10, 50))
            
    def visualize_network(self, name="jamnagar"):
        """Visualize graph network"""
        if name == "jamnagar":
            G = self.jamnagar_graph
            title = "Jamnagar Refinery Flow Diagram"
        else:
            G = self.abbc_graph
            title = "AB-BC Trans-Regional Pipeline Route"
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pos = nx.spring_layout(G, k=1.5, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='#ffcc00', 
                node_size=2500, font_weight='bold', font_size=9,
                edge_color='#666', width=2, arrowsize=20, ax=ax)
                
        plt.title(title, fontsize=14, weight='bold')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        import base64
        return base64.b64encode(buf.getvalue()).decode('utf-8')
