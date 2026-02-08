
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
        """Draw 3D Isometric View of Refinery Unit with Enhanced Geometry"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Simplified representations
        # Tower 1 (Main Distillation)
        z = np.linspace(0, 10, 20)
        theta = np.linspace(0, 2*np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = 2 + np.cos(theta_grid)
        y_grid = 2 + np.sin(theta_grid)
        ax.plot_surface(x_grid, y_grid, z_grid, color='silver', alpha=0.9, edgecolor='k', linewidth=0.1)
        
        # Tower 2 (Stripper) - smaller and offset
        z2 = np.linspace(0, 8, 20)
        x_grid2 = 5 + 0.8 * np.cos(theta_grid)
        y_grid2 = 2 + 0.8 * np.sin(theta_grid)
        ax.plot_surface(x_grid2, y_grid2, np.tile(z2[:,np.newaxis], (1,20)), color='darkgrey', alpha=0.9, edgecolor='k', linewidth=0.1)

        # Storage Tank 1
        z_tank = np.linspace(0, 4, 10)
        theta_grid_t, z_grid_t = np.meshgrid(theta, z_tank)
        x_grid_t = 6 + 2*np.cos(theta_grid_t)
        y_grid_t = 6 + 2*np.sin(theta_grid_t)
        ax.plot_surface(x_grid_t, y_grid_t, z_grid_t, color='white', alpha=0.9, edgecolor='k', linewidth=0.1)
        
        # Horizontal Drum
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, 3, 10)
        U, V = np.meshgrid(u, v)
        X = 2 + 0.8 * np.cos(U)
        Z = 2 + 0.8 * np.sin(U) # Elevated
        Y = 6 + V
        ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.9, edgecolor='k', linewidth=0.1)

        # Pipe Racks / Connecting Lines
        # Tower 1 to Tank
        ax.plot([2, 2, 6, 6], [2, 4, 4, 6], [9, 9, 3.8, 3.8], color='red', linewidth=3)
        # Tower 1 to Tower 2
        ax.plot([2, 5], [2, 2], [5, 5], color='blue', linewidth=3)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height')
        ax.set_title('Jamnagar Unit 3D Process Isometric', fontsize=12, weight='bold')
        ax.view_init(elev=35, azim=-45) # Classic Isometric Angle
        
        # Make pane background white
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False)
        
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

    @staticmethod
    def draw_master_plan_layout():
        """Draw 100,000 bbl/day Refinery Master Plan (Orthographic)"""
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_axis_off()
        ax.set_title("100,000 BBL/DAY REFINERY MASTER PLAN - GENERAL ARRANGEMENT", fontsize=16, weight='bold')
        
        # Site boundary
        ax.add_patch(patches.Rectangle((0, 0), 20, 15, fc='#fafafa', ec='black', ls='--'))
        
        # 1. Tank Farm (Crude & Product Storage)
        ax.text(3, 13, "TANK FARM (ZONE A)", fontsize=10, weight='bold', ha='center')
        for i in range(3):
            for j in range(2):
                ax.add_patch(patches.Circle((1.5 + i*2.5, 10 + j*2.5), 1, fc='white', ec='black'))
                ax.text(1.5 + i*2.5, 10 + j*2.5, f"TK-{10+i*2+j}01", ha='center', va='center', fontsize=6)
        
        # 2. Process Units (CDU, VDU, Cracker)
        ax.text(10, 13, "PROCESS UNITS (ZONE B)", fontsize=10, weight='bold', ha='center')
        # CDU Tower
        ax.add_patch(patches.Rectangle((8, 8), 1, 4, fc='#ddd', ec='black'))
        ax.text(8.5, 10, "CDU", ha='center', rotation=90)
        
        # VDU Tower
        ax.add_patch(patches.Rectangle((10, 8), 1, 4, fc='#ddd', ec='black'))
        ax.text(10.5, 10, "VDU", ha='center', rotation=90)
        
        # FCC Unit
        ax.add_patch(patches.Rectangle((12, 7.5), 2, 4.5, fc='#ddd', ec='black'))
        ax.text(13, 9.75, "FCC UNIT", ha='center')
        
        # 3. Utilities & Power
        ax.text(17, 13, "UTILITIES", fontsize=10, weight='bold', ha='center')
        ax.add_patch(patches.Rectangle((16, 9), 3, 3, fc='#ccc', ec='black', hatch='///'))
        
        # 4. Connecting Pipelines (Rack)
        ax.add_patch(patches.Rectangle((1, 6), 18, 0.5, fc='#444', ec='black')) # Main East-West Rack
        ax.text(10, 6.25, "MAIN PIPE RACK (CRUDE/PRODUCT/STEAM)", color='white', ha='center', fontsize=8)
        
        # Feeder Lines
        for x in [3, 5.5, 8.5, 10.5, 13, 17]:
            ax.plot([x, x], [6.5, 9], 'k-', linewidth=1)
            
        # 5. Isometric Flare Stack (stylized orthographic)
        ax.add_patch(patches.Rectangle((18, 2), 0.5, 10, fc='gray', ec='black'))
        # Flame
        ax.add_patch(patches.Polygon([[18, 12], [18.5, 12], [18.25, 13]], fc='orange'))
        ax.text(18.25, 13.5, "FLARE", ha='center', fontsize=8)
        
        # 6. Loading Gantry / Export Terminal
        ax.text(10, 2, "EXPORT TERMINAL (ZONE C)", fontsize=10, weight='bold', ha='center')
        ax.add_patch(patches.Rectangle((5, 1), 10, 3, fc='#eee', ec='black', ls='-.'))
        # Truck/Rail spots
        for k in range(8):
            ax.add_patch(patches.Rectangle((6 + k*1.1, 1.5), 0.8, 2, fc='yellow', ec='black'))

        # Scale Bar
        ax.plot([1, 4], [0.5, 0.5], 'k|-')
        ax.text(2.5, 0.7, "100 METERS", ha='center', fontsize=8)

        ax.set_xlim(0, 20)
        ax.set_ylim(0, 15)
        
        buf = BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight')
        plt.close()
        return buf.getvalue().decode('utf-8')

    @staticmethod
    def draw_chemical_plant_schematic():
        """Draw Chemical Plant Process Flow Diagram (PFD)"""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_axis_off()
        ax.set_title("ETHYLENE CRACKER COMPLEX - PROCESS FLOW DIAGRAM", fontsize=15, weight='bold')
        
        # 1. Feed Preparation
        ax.add_patch(patches.Rectangle((1, 4), 2, 3, fc='#e0f7fa', ec='black')) # Feed Prep
        ax.text(2, 5.5, "FEED\nPREP", ha='center', va='center', fontsize=9)
        
        # 2. Pyrolysis Furnaces
        for i in range(3):
            ax.add_patch(patches.Rectangle((4, 2 + i*2.5), 1.5, 2, fc='#ffccbc', ec='black'))
            ax.text(4.75, 3 + i*2.5, f"F-{100+i}", ha='center', fontsize=8)
            # Lines from Feed
            ax.arrow(3, 5.5, 1, (3+i*2.5)-5.5, head_width=0.1, color='blue')
            
        # 3. Quench Tower
        ax.add_patch(patches.Rectangle((7, 3), 1.5, 6, fc='#b2dfdb', ec='black')) # Quench
        ax.text(7.75, 6, "QUENCH\nTOWER", ha='center', fontsize=9)
        # Lines from Furnaces
        for i in range(3):
            ax.arrow(5.5, 3 + i*2.5, 1.5, 0, head_width=0.1, color='red')
            
        # 4. Compressor Train
        ax.add_patch(patches.Polygon([[9.5, 5], [10.5, 6], [10.5, 4]], fc='#ffe0b2', ec='black')) # Compressor
        ax.text(10.2, 5, "C-201", ha='center', fontsize=8)
        ax.arrow(8.5, 6, 1, -1, head_width=0.1, color='green')
        
        # 5. Separation Train (Cold Section)
        # Demethanizer
        ax.add_patch(patches.Rectangle((12, 7), 1, 4, fc='#e1bee7', ec='black'))
        ax.text(12.5, 9, "C1", ha='center', fontsize=8)
        # Deethanizer
        ax.add_patch(patches.Rectangle((14, 6), 1, 4, fc='#e1bee7', ec='black'))
        ax.text(14.5, 8, "C2", ha='center', fontsize=8)
        # Depropanizer
        ax.add_patch(patches.Rectangle((16, 5), 1, 4, fc='#e1bee7', ec='black'))
        ax.text(16.5, 7, "C3", ha='center', fontsize=8)
        
        # Connecting Lines
        ax.arrow(10.5, 5, 1.5, 3, head_width=0.1, color='green') # To C1
        ax.arrow(13, 8, 1, -1, head_width=0.1, color='purple') # C1 to C2
        ax.arrow(15, 7, 1, -1, head_width=0.1, color='purple') # C2 to C3
        
        # Products
        ax.arrow(12.5, 11, 0, 1, head_width=0.2, color='cyan')
        ax.text(12.5, 12.2, "Methane", ha='center')
        
        ax.arrow(14.5, 10, 0, 1, head_width=0.2, color='cyan')
        ax.text(14.5, 11.2, "Ethylene", ha='center', weight='bold')
        
        ax.arrow(16.5, 9, 0, 1, head_width=0.2, color='cyan')
        ax.text(16.5, 10.2, "Propylene", ha='center', weight='bold')

        ax.set_xlim(0, 18)
        ax.set_ylim(0, 13)
        
        buf = BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight')
        plt.close()
        return buf.getvalue().decode('utf-8')
