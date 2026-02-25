"""
RF Coil Visualization Module
=============================

Generates 2D and 3D visualizations of RF coil designs,
including schematics and physical geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional
from coil_designer import CoilParameters, RFCoilDesigner


class CoilVisualizer:
    """Visualize RF coil designs in 2D and 3D."""
    
    def __init__(self, designer: RFCoilDesigner):
        """
        Initialize visualizer.
        
        Args:
            designer: RF coil designer instance
        """
        self.designer = designer
        
    def plot_2d_solenoid(self, params: CoilParameters, 
                        ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot 2D side view of solenoid coil.
        
        Args:
            params: Coil parameters
            ax: Optional matplotlib axes
            
        Returns:
            Axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        diameter = params.diameter
        length = params.length
        turns = params.turns
        wire_d = params.wire_diameter
        
        # Calculate turn spacing
        turn_spacing = length / turns if turns > 1 else 0
        
        # Draw coil turns
        for i in range(turns):
            x_pos = i * turn_spacing
            
            # Top line
            ax.plot([x_pos, x_pos + wire_d], 
                   [diameter/2, diameter/2], 
                   'b-', linewidth=2)
            
            # Bottom line
            ax.plot([x_pos, x_pos + wire_d], 
                   [-diameter/2, -diameter/2], 
                   'b-', linewidth=2)
            
            # Vertical connectors
            if i < turns - 1:
                ax.plot([x_pos + wire_d, x_pos + turn_spacing],
                       [diameter/2, diameter/2],
                       'b--', alpha=0.3, linewidth=1)
                ax.plot([x_pos + wire_d, x_pos + turn_spacing],
                       [-diameter/2, -diameter/2],
                       'b--', alpha=0.3, linewidth=1)
        
        # Labels
        ax.set_xlabel('Length (mm)', fontsize=12)
        ax.set_ylabel('Diameter (mm)', fontsize=12)
        ax.set_title(f'Solenoid Coil - {turns} turns', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add annotations
        L = self.designer.calculate_inductance(params)
        ax.text(0.02, 0.98, f'Inductance: {L*1e6:.2f} µH\n'
                           f'Diameter: {diameter:.1f} mm\n'
                           f'Length: {length:.1f} mm\n'
                           f'Turns: {turns}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
        
        return ax
    
    def plot_2d_planar_spiral(self, params: CoilParameters,
                             ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot 2D top view of planar spiral coil.
        
        Args:
            params: Coil parameters
            ax: Optional matplotlib axes
            
        Returns:
            Axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        outer_diameter = params.diameter
        turns = params.turns
        spacing = params.length  # Track spacing
        
        # Generate spiral
        theta = np.linspace(0, turns * 2 * np.pi, 1000)
        radius = outer_diameter/2 - (theta / (2 * np.pi)) * spacing
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        ax.plot(x, y, 'b-', linewidth=2)
        
        # Mark start and end
        ax.plot(x[0], y[0], 'go', markersize=10, label='Start')
        ax.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
        
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_title(f'Planar Spiral Coil - {turns} turns', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend()
        
        # Add annotations
        L = self.designer.calculate_inductance(params)
        ax.text(0.02, 0.98, f'Inductance: {L*1e6:.2f} µH\n'
                           f'Outer Diameter: {outer_diameter:.1f} mm\n'
                           f'Track Spacing: {spacing:.2f} mm\n'
                           f'Turns: {turns}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                fontsize=10)
        
        return ax
    
    def plot_3d_solenoid(self, params: CoilParameters,
                        ax: Optional[Axes3D] = None) -> Axes3D:
        """
        Plot 3D visualization of solenoid coil.
        
        Args:
            params: Coil parameters
            ax: Optional 3D axes
            
        Returns:
            3D axes object
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        diameter = params.diameter
        length = params.length
        turns = params.turns
        
        # Generate helix
        t = np.linspace(0, turns * 2 * np.pi, 1000)
        z = (t / (2 * np.pi)) * (length / turns)
        x = (diameter / 2) * np.cos(t)
        y = (diameter / 2) * np.sin(t)
        
        ax.plot(x, y, z, 'b-', linewidth=2)
        
        # Mark start and end
        ax.scatter([x[0]], [y[0]], [z[0]], c='g', s=100, label='Start')
        ax.scatter([x[-1]], [y[-1]], [z[-1]], c='r', s=100, label='End')
        
        ax.set_xlabel('X (mm)', fontsize=11)
        ax.set_ylabel('Y (mm)', fontsize=11)
        ax.set_zlabel('Z (mm)', fontsize=11)
        ax.set_title(f'3D Solenoid Coil - {turns} turns', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        
        # Equal aspect ratio
        max_range = max(diameter, length) / 2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, length])
        
        return ax
    
    def generate_schematic(self, params: CoilParameters,
                          capacitance: float = 100e-12,
                          show_matching: bool = True,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate circuit schematic with coil and matching network.
        
        Args:
            params: Coil parameters
            capacitance: Tuning capacitance in F
            show_matching: Whether to show matching network
            save_path: Optional path to save figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')
        
        # Calculate electrical properties
        L = self.designer.calculate_inductance(params)
        f_res = self.designer.calculate_resonant_frequency(L, capacitance)
        Q = self.designer.calculate_quality_factor(params, f_res)
        
        # Draw inductor (coil)
        inductor_x = 2
        inductor_y = 3
        n_loops = 4
        loop_width = 0.3
        
        # Draw coil loops
        for i in range(n_loops):
            x_offset = i * loop_width
            circle = Circle((inductor_x + x_offset, inductor_y), 
                          loop_width/2, fill=False, 
                          edgecolor='blue', linewidth=2.5)
            ax.add_patch(circle)
        
        # Wire connections to inductor
        ax.plot([1, inductor_x], [inductor_y, inductor_y], 
               'k-', linewidth=2)
        ax.plot([inductor_x + n_loops*loop_width, 4.5], 
               [inductor_y, inductor_y], 'k-', linewidth=2)
        
        # Draw capacitor
        cap_x = 5
        cap_y = 3
        plate_height = 0.4
        plate_gap = 0.15
        
        ax.plot([cap_x - plate_gap/2, cap_x - plate_gap/2],
               [cap_y - plate_height/2, cap_y + plate_height/2],
               'k-', linewidth=3)
        ax.plot([cap_x + plate_gap/2, cap_x + plate_gap/2],
               [cap_y - plate_height/2, cap_y + plate_height/2],
               'k-', linewidth=3)
        
        # Wire connections to capacitor
        ax.plot([4.5, cap_x - plate_gap/2], [cap_y, cap_y], 
               'k-', linewidth=2)
        ax.plot([cap_x + plate_gap/2, 6], [cap_y, cap_y], 
               'k-', linewidth=2)
        
        # Ground symbol
        ground_x = 6
        ground_y = cap_y
        ax.plot([ground_x, ground_x], [ground_y, ground_y - 0.3], 
               'k-', linewidth=2)
        for i in range(3):
            width = 0.3 - i * 0.1
            ax.plot([ground_x - width/2, ground_x + width/2],
                   [ground_y - 0.3 - i*0.1, ground_y - 0.3 - i*0.1],
                   'k-', linewidth=2)
        
        # Input port
        ax.plot([0.5, 1], [inductor_y, inductor_y], 'k-', linewidth=2)
        ax.text(0.3, inductor_y, 'RF IN', fontsize=12, 
               verticalalignment='center', fontweight='bold')
        
        # Labels
        ax.text(inductor_x + n_loops*loop_width/2, inductor_y + 0.7,
               f'L = {L*1e6:.2f} µH', fontsize=11, 
               horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        ax.text(cap_x, cap_y - 0.7,
               f'C = {capacitance*1e12:.1f} pF', fontsize=11,
               horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Title and specifications
        title_text = f'RF Coil Circuit Schematic - {params.coil_type.replace("_", " ").title()}'
        ax.text(5, 5.5, title_text, fontsize=16, fontweight='bold',
               horizontalalignment='center')
        
        spec_text = (f'Resonant Frequency: {f_res/1e6:.2f} MHz\n'
                    f'Quality Factor: {Q:.1f}\n'
                    f'Turns: {params.turns} | Diameter: {params.diameter:.1f} mm')
        
        ax.text(8.5, 4.5, spec_text, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Matching network (optional)
        if show_matching:
            match_box = FancyBboxPatch((6.5, 2), 2, 2.5,
                                       boxstyle="round,pad=0.1",
                                       facecolor='lightgray',
                                       edgecolor='black',
                                       alpha=0.3,
                                       linewidth=1.5)
            ax.add_patch(match_box)
            ax.text(7.5, 3.5, 'Matching\nNetwork', fontsize=10,
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_coil_geometry(self, params: CoilParameters,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot complete coil geometry visualization.
        
        Args:
            params: Coil parameters
            save_path: Optional path to save
            
        Returns:
            Figure object
        """
        if params.coil_type == 'solenoid':
            fig = plt.figure(figsize=(14, 6))
            
            # 2D view
            ax1 = fig.add_subplot(121)
            self.plot_2d_solenoid(params, ax1)
            
            # 3D view
            ax2 = fig.add_subplot(122, projection='3d')
            self.plot_3d_solenoid(params, ax2)
            
        elif params.coil_type == 'planar_spiral':
            fig, ax = plt.subplots(figsize=(8, 8))
            self.plot_2d_planar_spiral(params, ax)
        
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, f'Visualization for {params.coil_type} not implemented',
                   ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
