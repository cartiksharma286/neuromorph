import matplotlib.pyplot as plt
import os

class StentVisualizer:
    def __init__(self, stent_generator):
        self.generator = stent_generator

    def plot_unrolled_pattern(self, output_path="stent_pattern.png"):
        unrolled_data = self.generator.get_unrolled_points()
        
        plt.figure(figsize=(10, 6))
        for ring in unrolled_data:
            z = [p[0] for p in ring]
            arc = [p[1] for p in ring]
            plt.plot(z, arc, 'b-', linewidth=2)
        
        plt.title("Unrolled Stent Pattern")
        plt.xlabel("Axial Position (mm)")
        plt.ylabel("Circumferential Position (mm)")
        plt.grid(True)
        plt.axis('equal')
        
        plt.savefig(output_path)
        plt.close()
        print(f"Visualization saved to {os.path.abspath(output_path)}")
