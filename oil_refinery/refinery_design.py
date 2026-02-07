
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import BytesIO

class RefineryDesigner:
    """
    Design module for Oil Refinery Components
    Generates drawings for Heat Exchangers, Reactors, Valves
    """
    
    @staticmethod
    def draw_heat_exchanger():
        """Draw Shell and Tube Heat Exchanger"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_axis_off()
        
        # Shell
        shell = patches.Rectangle((2, 3), 10, 4, facecolor='#ddd', edgecolor='black', linewidth=2)
        ax.add_patch(shell)
        
        # Determine baffles
        for i in range(1, 9):
            x = 2 + i
            if i % 2 == 0:
                baffle = patches.Rectangle((x, 3), 0.2, 3, facecolor='gray')
            else:
                baffle = patches.Rectangle((x, 4), 0.2, 3, facecolor='gray')
            ax.add_patch(baffle)
            
        # Tubes (simplified)
        ax.plot([1, 13], [4, 4], 'r-', linewidth=2)
        ax.plot([1, 13], [5, 5], 'r-', linewidth=2)
        ax.plot([1, 13], [6, 6], 'r-', linewidth=2)
        
        # Inlets/Outlets
        # Tube side
        ax.arrow(0, 4, 1, 0, head_width=0.3, head_length=0.3, fc='red', ec='red')
        ax.arrow(13, 6, 1, 0, head_width=0.3, head_length=0.3, fc='red', ec='red')
        
        # Shell side
        ax.arrow(3, 7.5, 0, -0.5, head_width=0.3, head_length=0.3, fc='blue', ec='blue')
        ax.arrow(11, 3, 0, -1, head_width=0.3, head_length=0.3, fc='blue', ec='blue')
        
        ax.text(7, 8, "Shell & Tube Heat Exchanger", fontsize=14, ha='center', weight='bold')
        ax.text(7, 2, "AB-BC Pipeline Interface", fontsize=10, ha='center', style='italic')
        
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 9)
        
        buf = BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight')
        plt.close()
        return buf.getvalue().decode('utf-8')

    @staticmethod
    def draw_refinery_reactor():
        """Draw Catalytic Cracking Reactor"""
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.set_axis_off()
        
        # Reactor Body
        reactor = patches.FancyBboxPatch((3, 2), 4, 12, boxstyle="round,pad=0.2", fc='#f0f0f0', ec='black', linewidth=2)
        ax.add_patch(reactor)
        
        # Catalyst Bed
        bed = patches.Rectangle((3.2, 5), 3.6, 6, facecolor='#aaa', alpha=0.5)
        ax.add_patch(bed)
        for _ in range(50):
            x = 3.2 + np.random.rand() * 3.6
            y = 5 + np.random.rand() * 6
            ax.plot(x, y, 'k.', markersize=1)

        # Feed Inlet
        ax.arrow(1, 4, 2, 0, head_width=0.5, head_length=0.5, fc='black', ec='black')
        ax.text(1, 3.5, "Crude Feed", ha='center')
        
        # Product Outlet
        ax.arrow(5, 14.5, 0, 1.5, head_width=0.5, head_length=0.5, fc='green', ec='green')
        ax.text(6.5, 15, "Refined Product", ha='left')
        
        ax.text(5, 1, "Jamnagar Refinery Unit: R-201", fontsize=12, ha='center', weight='bold')
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 17)
        
        buf = BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight')
        plt.close()
        return buf.getvalue().decode('utf-8')

    @staticmethod
    def draw_control_valve():
        """Draw Control Valve Schematic"""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_axis_off()
        
        # Valve Body (Bowtie symbol)
        p1 = patches.Polygon([[2, 4], [2, 2], [4, 3]], closed=True, fc='#444', ec='black')
        p2 = patches.Polygon([[6, 4], [6, 2], [4, 3]], closed=True, fc='#444', ec='black')
        ax.add_patch(p1)
        ax.add_patch(p2)
        
        # Stem
        ax.plot([4, 4], [3, 5], 'k-', linewidth=3)
        
        # Actuator (Semicircle)
        actuator = patches.Arc((4, 5), 2, 1.5, theta1=0, theta2=180, facecolor='#ddd', edgecolor='black', linewidth=2)
        ax.add_patch(actuator)
        ax.plot([3, 5], [5, 5], 'k-', linewidth=2)
        
        ax.text(4, 1, "Flow Control Valve\nFCV-101", fontsize=12, ha='center')
        
        ax.set_xlim(1, 7)
        ax.set_ylim(0, 7)
        
        buf = BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight')
        plt.close()
        return buf.getvalue().decode('utf-8')

    @staticmethod
    def draw_isometric_plant():
        """Draw 3D Isometric View of Refinery Unit"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Simplified representations
        # Tower 1
        z = np.linspace(0, 10, 20)
        theta = np.linspace(0, 2*np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = 2 + np.cos(theta_grid)
        y_grid = 2 + np.sin(theta_grid)
        ax.plot_surface(x_grid, y_grid, z_grid, color='silver', alpha=0.8)
        
        # Tank
        z = np.linspace(0, 4, 10)
        theta_grid, z_grid = np.meshgrid(theta, z)
         # (radius 2 at x=6, y=6)
        x_grid = 6 + 2*np.cos(theta_grid)
        y_grid = 6 + 2*np.sin(theta_grid)
        ax.plot_surface(x_grid, y_grid, z_grid, color='white', alpha=0.9)
        
        # Pipe
        ax.plot([2, 6], [2, 6], [8, 4], color='red', linewidth=3)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Jamnagar Unit 3D View')
        ax.view_init(elev=30, azim=45)
        
        buf = BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight')
        plt.close()
        return buf.getvalue().decode('utf-8')

    @staticmethod
    def draw_orthographic_plant():
        """Draw Orthographic Projections (Top, Front, Side)"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # TOP VIEW
        ax1.set_axis_off()
        ax1.set_title("TOP VIEW")
        ax1.add_patch(patches.Circle((3, 3), 1, fc='silver', ec='black')) # Tower
        ax1.add_patch(patches.Circle((7, 7), 2, fc='white', ec='black')) # Tank
        ax1.plot([3, 7], [3, 7], 'r-', linewidth=2) # Pipe
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        
        # FRONT VIEW
        ax2.set_axis_off()
        ax2.set_title("FRONT VIEW")
        ax2.add_patch(patches.Rectangle((2, 0), 2, 8, fc='silver', ec='black')) # Tower
        ax2.add_patch(patches.Rectangle((5, 0), 4, 3, fc='white', ec='black')) # Tank
        ax2.plot([3, 7], [7, 3], 'r-', linewidth=2) # Pipe
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        
        # SIDE VIEW
        ax3.set_axis_off()
        ax3.set_title("SIDE VIEW")
        ax3.add_patch(patches.Rectangle((2, 0), 2, 8, fc='silver', ec='black')) # Tower
        ax3.add_patch(patches.Rectangle((4, 0), 4, 3, fc='white', ec='black')) # Tank Side
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 10)

        buf = BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight')
        plt.close()
        return buf.getvalue().decode('utf-8')

    @staticmethod
    def draw_pipeline_assembly():
        """Draw Detailed Pipeline Assembly with Flanges and Valves"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_axis_off()
        ax.set_title("PIPELINE SEGMENT ASSEMBLY - PA-104", fontsize=14, weight='bold')
        
        # Main Pipe Sections
        ax.add_patch(patches.Rectangle((1, 4), 3, 1, fc='#ddd', ec='black', linewidth=1.5)) # Left Pipe
        ax.add_patch(patches.Rectangle((6, 4), 3, 1, fc='#ddd', ec='black', linewidth=1.5)) # Right Pipe
        
        # Valve Section
        ax.add_patch(patches.Rectangle((4, 3.8), 0.3, 1.4, fc='gray', ec='black')) # Left Flange
        ax.add_patch(patches.Rectangle((4.3, 3.5), 1.4, 2, fc='#555', ec='black')) # Valve Body
        ax.add_patch(patches.Rectangle((5.7, 3.8), 0.3, 1.4, fc='gray', ec='black')) # Right Flange
        
        # Valve Stem & Actuator
        ax.plot([5, 5], [5.5, 7], 'k-', linewidth=3)
        ax.add_patch(patches.Circle((5, 7.5), 0.8, fc='orange', ec='black'))
        
        # Bolts (Simulated)
        ax.plot([4.15, 4.15], [4.0, 4.9], 'k.', markersize=6)
        ax.plot([5.85, 5.85], [4.0, 4.9], 'k.', markersize=6)
            
        # Pipe Supports
        ax.add_patch(patches.Rectangle((2, 2), 0.2, 2, fc='brown', ec='black'))
        ax.add_patch(patches.Rectangle((1.5, 1.8), 1.2, 0.2, fc='brown', ec='black'))
        
        ax.add_patch(patches.Rectangle((7, 2), 0.2, 2, fc='brown', ec='black'))
        ax.add_patch(patches.Rectangle((6.5, 1.8), 1.2, 0.2, fc='brown', ec='black'))
        
        # Dimensions & Labels
        ax.annotate('1000mm', xy=(2.5, 3.5), xytext=(2.5, 3.0), arrowprops=dict(arrowstyle='<->'), ha='center')
        ax.annotate('1000mm', xy=(7.5, 3.5), xytext=(7.5, 3.0), arrowprops=dict(arrowstyle='<->'), ha='center')
                    
        ax.text(5, 8.5, "Motor Operated Valve (MOV)", ha='center', fontsize=10)
        ax.text(2.5, 4.5, "API 5L Gr.B", ha='center', fontsize=8, color='#555')
        ax.text(7.5, 4.5, "API 5L Gr.B", ha='center', fontsize=8, color='#555')
        ax.text(5, 1, "Scale: 1:20", ha='center', style='italic')

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 9)

        buf = BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight')
        plt.close()
        return buf.getvalue().decode('utf-8')
