import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class ReactorVisualizer:
    def __init__(self, designer_instance):
        self.designer = designer_instance
        self.geo = designer_instance.geometry
        self.opt = designer_instance.optimization

    def plot_vessel(self, filename):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Geometry
        D = self.geo['diameter']
        H = self.geo['tan_tan_height']
        R_crown = self.geo['crown_radius']
        r_knuckle = self.geo['knuckle_radius']
        N_turns = self.opt['num_turns']
        pitch = self.opt['coil_pitch']
        pipe_dia = self.opt['pipe_diameter']
        
        # 1. Cylinder Shell
        z = np.linspace(0, H, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = (D/2) * np.cos(theta_grid)
        y_grid = (D/2) * np.sin(theta_grid)
        
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color='grey')
        
        # 2. Torispherical Heads (Approximation)
        # Top Head
        # Z = H + sqrt(R^2 - r^2) ... simplified to spherical cap for visualization
        # Ideally, complex blend. Let's use simple hemisphere cap scaled by 0.2H to verify "head" presence.
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi / 2, 10) # cap
        x_head = (D/2) * np.outer(np.cos(u), np.sin(v))
        y_head = (D/2) * np.outer(np.sin(u), np.sin(v))
        z_head = H + (D/4) * np.outer(np.ones(np.size(u)), np.cos(v)) # Ellipsoidal approx
        ax.plot_surface(x_head, y_head, z_head, color='darkgrey', alpha=0.5)
        
        # Bottom Head
        z_bot = -(D/4) * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_head, y_head, z_bot, color='darkgrey', alpha=0.5)
        
        # 3. Prime Optimized Half-Pipe Coil
        # Parametric spiral: x = r cos(t), y = r sin(t), z = (H/N) * (t/2pi)
        # Total Angle = 2 * pi * N_turns
        total_angle = 2 * np.pi * N_turns
        t_coil = np.linspace(0, total_angle, int(N_turns * 20))
        z_coil = (pitch * t_coil) / (2 * np.pi) 
        
        # Clamp Z to 0..H (ensure coil stays on tan-tan)
        # Add offset for thickness
        r_coil = (D/2) + (pipe_dia/2)
        x_coil = r_coil * np.cos(t_coil)
        y_coil = r_coil * np.sin(t_coil)
        
        # Plot only valid Z range
        mask = (z_coil >= 0) & (z_coil <= H)
        ax.plot(x_coil[mask], y_coil[mask], z_coil[mask], color='red', linewidth=2, label=f'Prime Coil (N={N_turns})')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Height (m)')
        ax.set_title(f'Diketene Reactor: Statistical Congruence N={N_turns}')
        
        # Adjust Limits
        max_range = np.array([x_grid.max()-x_grid.min(), y_grid.max()-y_grid.min(), z_head.max()-z_bot.min()]).max() / 2.0
        mid_x = (x_grid.max()+x_grid.min()) * 0.5
        mid_y = (y_grid.max()+y_grid.min()) * 0.5
        mid_z = (z_head.max()+z_bot.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.savefig(filename)
        plt.close()
