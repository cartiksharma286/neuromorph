import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class StentVisualizer:
    def __init__(self, stent_generator):
        self.generator = stent_generator

    def plot_unrolled_pattern(self, output_path="stent_pattern.png"):
        unrolled_rings, unrolled_connectors = self.generator.get_unrolled_points()
        
        plt.figure(figsize=(10, 6))
        
        # Plot Rings
        for ring in unrolled_rings:
            z = [p[0] for p in ring]
            arc = [p[1] for p in ring]
            plt.plot(z, arc, 'b-', linewidth=2, label='Strut' if unrolled_rings.index(ring) == 0 else "")
            
        # Plot Connectors
        for conn in unrolled_connectors:
            z = [p[0] for p in conn]
            arc = [p[1] for p in conn]
            plt.plot(z, arc, 'r--', linewidth=2, alpha=0.7, label='Connector' if unrolled_connectors.index(conn) == 0 else "")
        
        plt.title("Unrolled Stent Pattern (Sine + Connectors)")
        plt.xlabel("Axial Position (mm)")
        plt.ylabel("Circumferential Position (mm)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        plt.savefig(output_path)
        plt.close()
        print(f"Visualization saved to {os.path.abspath(output_path)}")
