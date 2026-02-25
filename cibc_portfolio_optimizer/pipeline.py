import matplotlib.pyplot as plt
import networkx as nx

def draw_pipeline_geo_flow():
    # Create a directed graph
    G = nx.DiGraph()

    # Define nodes (Locations and facilities)
    # Positions roughly estimated for visual flow (left=east/AB, right=west/BC)
    nodes = {
        'Edmonton Terminal (AB)': (0, 5),
        'Hinton Pump Station': (2, 5.5),
        'Jasper Pump Station (Rockies)': (4, 4.5),
        'Blue River Station': (6, 3),
        'Kamloops Terminal': (8, 2),
        'Hope Pressure Control': (10, 1),
        'Burnaby Terminal/Refinery (BC)': (12, 0.5),
        'Westridge Marine Terminal': (13, -1)
    }

    G.add_nodes_from(nodes.keys())

    # Define Edges (The Pipeline Segments)
    edges = [
        ('Edmonton Terminal (AB)', 'Hinton Pump Station'),
        ('Hinton Pump Station', 'Jasper Pump Station (Rockies)'),
        ('Jasper Pump Station (Rockies)', 'Blue River Station'),
        ('Blue River Station', 'Kamloops Terminal'),
        ('Kamloops Terminal', 'Hope Pressure Control'),
        ('Hope Pressure Control', 'Burnaby Terminal/Refinery (BC)'),
        ('Burnaby Terminal/Refinery (BC)', 'Westridge Marine Terminal'),
    ]

    G.add_edges_from(edges)

    # Canvas setup
    plt.figure(figsize=(14, 8))
    plt.title("Conceptual High-Level Oil Pipeline Flow (Alberta to BC Coast)", fontsize=16, pad=20)

    # Draw Nodes based on fixed positions
    pos = nodes
    
    # Define node colors based on type
    node_colors = []
    for node in G.nodes():
        if "Terminal" in node or "Refinery" in node:
            node_colors.append('#FF5733') # Orange for major terminals
        elif "Pump" in node or "Control" in node:
            node_colors.append('#3498DB') # Blue for mid-stream infrastructure
        else:
            node_colors.append('gray')

    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.9, edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=3, edge_color='#555555', arrowsize=30, arrowstyle='-|>')
    
    # Add labels with a box to make them readable
    label_options = {"ec": "k", "fc": "white", "alpha": 0.8}
    nx.draw_networkx_labels(G, pos, font_size=9, bbox=label_options)

    # Add visual context annotations
    plt.text(1, 7, "ALBERTA PLAINS", fontsize=14, fontweight='bold', color='gray')
    plt.text(4, 6, "ROCKY MOUNTAINS", fontsize=14, fontweight='bold', color='gray', rotation=-20)
    plt.text(10, 3, "BC INTERIOR / COAST", fontsize=14, fontweight='bold', color='gray')

    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_pipeline_geo_flow()
