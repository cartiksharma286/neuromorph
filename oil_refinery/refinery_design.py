
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
        """Draw 3D Isometric View of Eco-Friendly Refinery Unit"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. Main Distillation Tower (Recycled Steel)
        z = np.linspace(0, 10, 20)
        theta = np.linspace(0, 2*np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = 2 + np.cos(theta_grid)
        y_grid = 2 + np.sin(theta_grid)
        ax.plot_surface(x_grid, y_grid, z_grid, color='#CFD8DC', alpha=0.9, edgecolor='#455A64', linewidth=0.1) # Recycled Steel
        
        # 2. Stripper Tower (Bio-Catalyst Unit)
        z2 = np.linspace(0, 8, 20)
        x_grid2 = 5 + 0.8 * np.cos(theta_grid)
        y_grid2 = 2 + 0.8 * np.sin(theta_grid)
        ax.plot_surface(x_grid2, y_grid2, np.tile(z2[:,np.newaxis], (1,20)), color='#A5D6A7', alpha=0.9, edgecolor='#2E7D32', linewidth=0.1) # Green Body

        # 3. Storage Tank (Solar Powered)
        z_tank = np.linspace(0, 4, 10)
        theta_grid_t, z_grid_t = np.meshgrid(theta, z_tank)
        x_grid_t = 6 + 2*np.cos(theta_grid_t)
        y_grid_t = 6 + 2*np.sin(theta_grid_t)
        ax.plot_surface(x_grid_t, y_grid_t, z_grid_t, color='white', alpha=0.9, edgecolor='#1B5E20', linewidth=0.1)
        
        # Solar Panel on Tank Roof
        r_solar = np.linspace(0, 2, 5)
        R_solar, The_solar = np.meshgrid(r_solar, theta)
        X_solar = 6 + R_solar * np.cos(The_solar)
        Y_solar = 6 + R_solar * np.sin(The_solar)
        Z_solar = np.full_like(X_solar, 4.1)
        ax.plot_surface(X_solar, Y_solar, Z_solar, color='#0288D1', alpha=1.0) # Blue Solar Panel

        # 4. Carbon Capture Unit (New Addition)
        z_cc = np.linspace(0, 6, 10)
        x_grid_cc = 2 + 1 * np.cos(theta_grid)
        y_grid_cc = 7 + 1 * np.sin(theta_grid)
        ax.plot_surface(x_grid_cc, y_grid_cc, np.tile(z_cc[:,np.newaxis], (1,20)), color='#81C784', alpha=0.9, edgecolor='#1B5E20', linewidth=0.1)

        # 5. Eco-Piping (Zero-Leak Composite)
        # Tower 1 to Tank
        ax.plot([2, 2, 6, 6], [2, 4, 4, 6], [9, 9, 3.8, 3.8], color='#2E7D32', linewidth=3) # Green Pipe
        # Tower 1 to CC Unit
        ax.plot([2, 2], [2, 7], [8, 5], color='#43A047', linewidth=2, linestyle='--') # Capture Line

        # Annotations
        ax.text(2, 2, 11, "H2-Steel Tower", fontsize=8, ha='center')
        ax.text(6, 6, 5, "Solar Array", fontsize=8, ha='center', color='#0277BD')
        ax.text(2, 7, 7, "Carbon Capture\nModule", fontsize=8, ha='center', color='#1B5E20')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height')
        ax.set_title('Eco-Refinery 3D Isometric View\n(Net-Zero Design)', fontsize=12, weight='bold', color='#1B5E20')
        ax.view_init(elev=35, azim=-45)
        
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
        """Draw Orthographic Projections (Top, Front, Side) with Eco-Friendly Layout"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # TOP VIEW
        ax1.set_axis_off()
        ax1.set_title("TOP VIEW (Eco-Plan)")
        ax1.add_patch(patches.Circle((3, 3), 1, fc='#CFD8DC', ec='#455A64')) # Tower
        ax1.add_patch(patches.Circle((7, 7), 2, fc='white', ec='#1B5E20')) # Tank
        ax1.add_patch(patches.Rectangle((6, 6), 2, 2, fc='#0288D1', alpha=0.3)) # Solar Panel Outline
        ax1.add_patch(patches.Circle((3, 8), 1, fc='#81C784', ec='#1B5E20')) # Carbon Capture
        
        ax1.plot([3, 7], [3, 7], color='#2E7D32', linewidth=2) # Green Pipe
        ax1.plot([3, 3], [3, 7], color='#43A047', linewidth=1, linestyle='--') # CC Line
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        
        # FRONT VIEW
        ax2.set_axis_off()
        ax2.set_title("FRONT VIEW")
        ax2.add_patch(patches.Rectangle((2, 0), 2, 8, fc='#CFD8DC', ec='#455A64')) # Tower
        ax2.add_patch(patches.Rectangle((5, 0), 4, 3, fc='white', ec='#1B5E20')) # Tank
        ax2.add_patch(patches.Rectangle((5.5, 3), 3, 0.2, fc='#0288D1')) # Solar
        ax2.add_patch(patches.Rectangle((2, 8), 1, 1, fc='#81C784', ec='none')) # CC Vent
        
        ax2.plot([3, 7], [7, 3], color='#2E7D32', linewidth=2) # Pipe
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        
        # SIDE VIEW
        ax3.set_axis_off()
        ax3.set_title("SIDE VIEW")
        ax3.add_patch(patches.Rectangle((2, 0), 2, 8, fc='#CFD8DC', ec='#455A64')) # Tower
        ax3.add_patch(patches.Rectangle((4, 0), 4, 3, fc='white', ec='#1B5E20')) # Tank Side
        ax3.add_patch(patches.Rectangle((4.5, 3), 3, 0.2, fc='#0288D1')) # Solar
        ax3.add_patch(patches.Rectangle((1, 4), 1, 4, fc='#81C784', ec='#1B5E20')) # CCU Side
        
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

    @staticmethod
    def draw_abbc_pipeline_systematic():
        """
        Draw AB-BC Pipeline Systematic with Compressible Flow & Green Footprint
        """
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_axis_off()
        ax.set_title("AB-BC CORRIDOR: SYSTEMATIC PIPELINE & FLOW DYNAMICS", fontsize=16, weight='bold', color='#2E7D32')
        
        # 1. Green Canadian Footprint (Terrain Background)
        # Rocky Mountains stylized
        x_mtn = np.linspace(0, 16, 100)
        y_mtn = 2 + 1.5 * np.sin(x_mtn * 1.5) * np.exp(-0.1 * (x_mtn - 8)**2) + 0.5 * np.random.rand(100)
        ax.fill_between(x_mtn, 0, y_mtn, color='#C8E6C9', alpha=0.6) # Light Green Footer
        
        # Specific Peaks
        peaks = [(4, 6), (7, 8), (10, 7), (13, 5)]
        for px, py in peaks:
            ax.add_patch(patches.Polygon([[px-1.5, 2], [px, py], [px+1.5, 2]], fc='#81C784', ec='#2E7D32', alpha=0.8))

        # 2. Pipeline Path (Edmonton -> Burnaby)
        # Curve representing the route
        path_x = np.linspace(1, 15, 100)
        # A path that navigates 'through' the mountains
        path_y = 5 - 0.15 * (path_x - 1) - 0.5 * np.sin(path_x) 
        
        # 3. Compressible Fluid Flow Visualization
        # Use a scatter plot with changing color/size to represent pressure/density drop
        # High Pressure (Red/Dense) -> Low Pressure (Blue/Expanded)
        cmap = plt.get_cmap('RdYlBu')
        for i in range(len(path_x)-1):
            color = cmap(i / len(path_x))
            ax.plot(path_x[i:i+2], path_y[i:i+2], color=color, linewidth=6, solid_capstyle='round')
            
        # 4. Points of Interest (POIs)
        pois = [
            (1, 5, "Edmonton\nTerminal", "Inlet"),
            (5, 5.5, "Jasper\nPump Station", "Boost"),
            (10, 4.5, "Kamloops\nTerminal", "Distribution"),
            (15, 2.5, "Burnaby\nExport", "Outlet")
        ]
        
        for x, y, label, type_ in pois:
            # Station Marker
            ax.add_patch(patches.Circle((x, y), 0.3, fc='white', ec='black', zorder=10))
            ax.text(x, y + 0.5, label, ha='center', fontsize=9, weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            # Valve Dynamics Visualization at POIs
            if type_ in ["Boost", "Distribution"]:
                # Draw control valve symbol with 'dynamics' (e.g. percentage open)
                ax.add_patch(patches.Polygon([[x-0.4, y+0.8], [x+0.4, y+0.8], [x, y+0.4]], fc='#FF9800', ec='black'))
                ax.text(x, y+1.0, "CV-Active", fontsize=7, ha='center', color='#E65100')
                # Pressure Gauge
                ax.add_patch(patches.Circle((x+0.5, y+0.5), 0.2, fc='#ECEFF1', ec='black'))
                ax.plot([x+0.5, x+0.5+0.15], [y+0.5, y+0.5+0.15], 'r-', linewidth=1) 

        # 5. Flow Vectors & Annotations
        # Show flow direction
        for i in range(0, 100, 15):
            ax.arrow(path_x[i], path_y[i], path_x[i+1]-path_x[i], path_y[i+1]-path_y[i], 
                     head_width=0.2, color='white', alpha=0.8)

        # Environmental Note
        ax.text(8, 0.5, "Eco-Friendly Corridor Monitoring • Carbon Neutral Operations", 
                ha='center', fontsize=10, color='#1B5E20', style='italic')

        # Legend for Flow
        # Gradient Bar
        grad_x = np.linspace(12, 15, 50)
        for i in range(len(grad_x)):
            color = cmap(i / len(grad_x))
            ax.add_patch(patches.Rectangle((grad_x[i], 8.5), 0.1, 0.5, color=color, linewidth=0))
        ax.text(12, 9.1, "High Pressure (Liquid)", fontsize=8)
        ax.text(15, 9.1, "Low Pressure (Gas)", fontsize=8, ha='right')
        ax.text(13.5, 8.2, "Compressible Flow Gradient", ha='center', fontsize=8)


        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)

        buf = BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight')
        plt.close()
        return buf.getvalue().decode('utf-8')
