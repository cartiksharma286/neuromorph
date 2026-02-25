
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
import base64

class CircuitSchematicGenerator:
    def __init__(self):
        self.fig_size = (12, 10)
        self.colors = {
            'conductor': '#38bdf8',  # Light blue
            'capacitor': '#f472b6',  # Pink
            'inductor': '#a78bfa',   # Purple
            'ground': '#94a3b8',     # Grey
            'background': '#0f172a', # Dark blue/slate
            'text': '#e2e8f0'        # White/slate
        }

    def _setup_plot(self, title):
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(self.colors['background'])
        ax.set_facecolor(self.colors['background'])
        ax.set_title(title, color=self.colors['text'], fontsize=14, pad=15)
        ax.axis('off')
        ax.set_aspect('equal')
        return fig, ax

    def draw_capacitor(self, ax, center, width=0.5, height=0.5, vertical=False):
        x, y = center
        color = self.colors['capacitor']
        gap = width * 0.2
        
        if vertical:
            # Top plate
            ax.plot([x - width/2, x + width/2], [y + gap/2, y + gap/2], color=color, linewidth=2)
            # Bottom plate
            ax.plot([x - width/2, x + width/2], [y - gap/2, y - gap/2], color=color, linewidth=2)
            # Leads
            ax.plot([x, x], [y + gap/2, y + height/2], color=self.colors['conductor'], linewidth=1.5)
            ax.plot([x, x], [y - gap/2, y - height/2], color=self.colors['conductor'], linewidth=1.5)
        else:
            # Left plate
            ax.plot([x - gap/2, x - gap/2], [y - height/2, y + height/2], color=color, linewidth=2)
            # Right plate
            ax.plot([x + gap/2, x + gap/2], [y - height/2, y + height/2], color=color, linewidth=2)
            # Leads
            ax.plot([x - gap/2, x - width/2], [y, y], color=self.colors['conductor'], linewidth=1.5)
            ax.plot([x + gap/2, x + width/2], [y, y], color=self.colors['conductor'], linewidth=1.5)

    def draw_inductor(self, ax, start, end, num_loops=4):
        x1, y1 = start
        x2, y2 = end
        color = self.colors['inductor']
        
        # Simple zig-zag as inductor
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        angle = np.arctan2(y2-y1, x2-x1)
        
        step_len = dist / (num_loops * 2)
        
        # Rotate coordinates
        current_x, current_y = 0, 0
        
        points = [(0,0)]
        for i in range(num_loops):
            # Up
            points.append((current_x + step_len, 0.3))
            # Down
            current_x += step_len * 2
            points.append((current_x, 0))
            
        # Rotate and translate points
        rot_points = []
        for px, py in points:
            rx = px * np.cos(angle) - py * np.sin(angle) + x1
            ry = px * np.sin(angle) + py * np.cos(angle) + y1
            rot_points.append((rx, ry))
            
        px, py = zip(*rot_points)
        ax.plot(px, py, color=color, linewidth=1.5)


    def generate_birdcage_schematic(self):
        fig, ax = self._setup_plot("Birdcage Coil (High-Pass)")
        
        # Simplified Birdcage: 2 horizontal rails (rings) connected by vertical rungs
        
        # Top Ring segments
        for i in range(5):
            start = (i * 2, 4)
            end = ((i * 2) + 2, 4)
            # Capacitor in ring for High Pass
            mid = (start[0] + 1, start[1])
            self.draw_capacitor(ax, mid, width=1.0, height=0.4, vertical=False)
            
        # Bottom Ring segments
        for i in range(5):
            start = (i * 2, 0)
            end = ((i * 2) + 2, 0)
            mid = (start[0] + 1, start[1])
            self.draw_capacitor(ax, mid, width=1.0, height=0.4, vertical=False)
            
        # Vertical Rungs (Inductors/Conductors)
        for i in range(6):
            x = i * 2
            # Connect top to bottom
            ax.plot([x, x], [0, 4], color=self.colors['conductor'], linewidth=1.5)
            
        # Labels
        ax.text(5, 4.5, "End Ring (Capacitive)", color=self.colors['text'], ha='center')
        ax.text(5, -0.5, "End Ring (Capacitive)", color=self.colors['text'], ha='center')
        ax.text(11, 2, "Rung", color=self.colors['text'], ha='left')
        ax.annotate("", xy=(10, 2), xytext=(11, 2), arrowprops=dict(arrowstyle="->", color=self.colors['text']))

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor=self.colors['background'])
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def generate_solenoid_schematic(self):
        fig, ax = self._setup_plot("Solenoid / Loop Coil")
        
        # Draw coiled wire
        t = np.linspace(0, 8*np.pi, 200)
        x = np.linspace(1, 9, 200)
        y = 2 + 1.0 * np.sin(t)
        
        # 3D effect: modulate line width or alpha? Simple plot first
        ax.plot(x, y, color=self.colors['conductor'], linewidth=2.5)
        
        # Matching circuit box
        rect = patches.Rectangle((3, -1.5), 4, 2, linewidth=1.5, edgecolor=self.colors['text'], facecolor='none')
        ax.add_patch(rect)
        ax.text(5, -2, "Matching Network", ha='center', color=self.colors['text'])
        
        # Lines to coil
        ax.plot([1, 3], [2, 0.5], color=self.colors['conductor'], linestyle='--')
        ax.plot([9, 7], [2, 0.5], color=self.colors['conductor'], linestyle='--')
        
        # Capacitors inside matching
        self.draw_capacitor(ax, (4, -0.5), width=0.8, height=0.4, vertical=True) # Tune
        self.draw_capacitor(ax, (6, -0.5), width=0.8, height=0.4, vertical=True) # Match
        
        ax.text(4, -1.1, "Ct", color=self.colors['text'], fontsize=8, ha='center')
        ax.text(6, -1.1, "Cm", color=self.colors['text'], fontsize=8, ha='center')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor=self.colors['background'])
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def generate_surface_array_schematic(self):
        fig, ax = self._setup_plot("Surface Phased Array (Overlap)")
        
        centers = [(2, 2), (4.5, 2), (7, 2), (9.5, 2)]
        colors = ['#f87171', '#fbbf24', '#4ade80', '#60a5fa'] # Red, Amber, Green, Blue
        
        for idx, (cx, cy) in enumerate(centers):
            # Draw Loop
            circle = patches.Circle((cx, cy), radius=1.5, fill=False, edgecolor=colors[idx], linewidth=2.5, alpha=0.8)
            ax.add_patch(circle)
            
            # Label
            ax.text(cx, cy, f"Ch {idx+1}", color=colors[idx], ha='center', va='center', fontsize=10, weight='bold')
            
            # Preamp
            rect = patches.Rectangle((cx-0.3, cy-2.5), 0.6, 0.6, edgecolor=colors[idx], facecolor='none')
            ax.add_patch(rect)
            ax.text(cx, cy-2.2, "LNA", color=colors[idx], fontsize=6, ha='center', va='center')
            
            # Connect
            ax.plot([cx, cx], [cy-1.5, cy-1.9], color=colors[idx])
            
        ax.text(5.5, 4, "Overlapping Loops minimize mutual inductance", color=self.colors['text'], ha='center', fontsize=10)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor=self.colors['background'])
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def generate_quantum_lattice_schematic(self):
        fig, ax = self._setup_plot("Quantum Lattice Coil (Non-Local)")
        
        # Grid
        grid_x = np.linspace(1, 9, 5)
        grid_y = np.linspace(1, 5, 3)
        
        for gy in grid_y:
            ax.plot([1, 9], [gy, gy], color=self.colors['ground'], alpha=0.3, linewidth=0.5)
        for gx in grid_x:
            ax.plot([gx, gx], [1, 5], color=self.colors['ground'], alpha=0.3, linewidth=0.5)
            
        # Psi nodes
        for gy in grid_y:
            for gx in grid_x:
                ax.text(gx, gy, "Ψ", color=self.colors['inductor'], ha='center', va='center', fontsize=12, weight='bold')
                
        # Entanglement lines (curved)
        t = np.linspace(0, np.pi, 50)
        for i in range(len(grid_x)-1):
            x = np.linspace(grid_x[i], grid_x[i+1], 50)
            y = 3 + 0.5 * np.sin(t)
            ax.plot(x, y, color=self.colors['capacitor'], alpha=0.6, linestyle=':')
            
        ax.text(5, 0.5, "Entangled Flux States via Josephson Junctions", color=self.colors['text'], ha='center', fontsize=10)
        
        # "JJ" box
        self.draw_capacitor(ax, (5, 3), width=0.4, height=0.4, vertical=True)
        ax.text(5, 3.5, "JJ", color=self.colors['capacitor'], fontsize=8)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor=self.colors['background'])
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def generate_all(self):
         return {
             'birdcage': self.generate_birdcage_schematic(),
             'solenoid': self.generate_solenoid_schematic(),
             'surface_array': self.generate_surface_array_schematic(),
             'quantum_lattice': self.generate_quantum_lattice_schematic()
         }

    def generate_base64(self, schematic_type="Standard MRI"):
        """Generates a base64 string for the requested schematic type."""
        try:
            if schematic_type == "Quantum Vascular":
                return self.generate_quantum_lattice_schematic()
            elif schematic_type == "Ultra-High Field":
                return self.generate_surface_array_schematic() # Use array for high field
            else:
                return self.generate_birdcage_schematic() # Default
        except Exception as e:
            print(f"Error generating schematic: {e}")
            # Return empty 1x1 pixel
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

if __name__ == "__main__":
    gen = CircuitSchematicGenerator()
    schematics = gen.generate_all()
    
    # Save to disk
    for name, b64_str in schematics.items():
        filename = f"{name}_schematic.png"
        with open(filename, "wb") as f:
            f.write(base64.b64decode(b64_str))
        print(f"Saved {filename}")
        
    print("Circuit Schematics Generated and Saved to Disk.")
