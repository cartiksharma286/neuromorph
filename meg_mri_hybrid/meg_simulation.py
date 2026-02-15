import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class SQUIDSensor:
    """Represents a single SQUID sensor coil."""
    def __init__(self, position, orientation, sensor_id=0):
        self.position = np.array(position, dtype=float)
        self.orientation = np.array(orientation, dtype=float)
        self.orientation /= np.linalg.norm(self.orientation)
        self.id = sensor_id

class MEGBoreScanner:
    """Simulates a cylindrical MEG scanner arrangement."""
    def __init__(self, radius=0.12, length=0.3, n_circumference=12, n_length=10):
        self.sensors = []
        self.radius = radius
        self.length = length
        self.generate_bore_geometry(n_circumference, n_length)

    def generate_bore_geometry(self, nc, nl):
        """Generates sensors arranged in a cylindrical bore."""
        z_coords = np.linspace(-self.length/2, self.length/2, nl)
        angle_step = 2 * np.pi / nc
        
        count = 0
        for z in z_coords:
            for i in range(nc):
                theta = i * angle_step
                x = self.radius * np.cos(theta)
                y = self.radius * np.sin(theta)
                
                pos = np.array([x, y, z])
                # Orientation determines sensitivity direction (radial inwards)
                orient = -np.array([np.cos(theta), np.sin(theta), 0])
                
                self.sensors.append(SQUIDSensor(pos, orient, sensor_id=count))
                count += 1
                
    def get_sensor_positions(self):
        return np.array([s.position for s in self.sensors])
    
    def get_sensor_orientations(self):
        return np.array([s.orientation for s in self.sensors])

class NeuralSource:
    """Represents a current dipole source in the brain."""
    def __init__(self, position, moment, time_course=None):
        self.position = np.array(position, dtype=float)
        self.moment = np.array(moment, dtype=float) # Dipole moment vector (Q)
        self.time_course = time_course # 1D array of activation over time

class ForwardPhysics:
    """Physics engine for calculating magnetic fields from dipoles."""
    def __init__(self):
        self.mu0 = 4 * np.pi * 1e-7  # Vacuum permeability
        
    def compute_lead_field(self, scanner, sources):
        """
        Computes the lead field matrix (gain matrix) L.
        Vectorized implementation for performance.
        """
        sensor_pos = scanner.get_sensor_positions() # (N_sens, 3)
        sensor_orient = scanner.get_sensor_orientations() # (N_sens, 3)
        
        # Prepare sources
        src_pos = np.array([s.position for s in sources]) # (N_src, 3)
        src_mom = np.array([s.moment for s in sources])   # (N_src, 3)
        
        # R vectors from sources to sensors (N_sens, N_src, 3)
        R = sensor_pos[:, np.newaxis, :] - src_pos[np.newaxis, :, :]
        dist = np.linalg.norm(R, axis=2)
        dist[dist==0] = 1e-9 # Avoid singularity
        
        # Cross product Q x R
        # Broadcast Moment: (1, N_src, 3) -> (N_sens, N_src, 3)
        Q_broad = np.tile(src_mom[np.newaxis, :, :], (len(sensor_pos), 1, 1))
        Cross = np.cross(Q_broad, R, axis=2)
        
        # B field (N_sens, N_src, 3)
        B_vec = (self.mu0 / (4 * np.pi)) * Cross / (dist[:, :, np.newaxis]**3)
        
        # Project onto sensor orientation (N_sens, 1, 3)
        flux_per_source = np.sum(B_vec * sensor_orient[:, np.newaxis, :], axis=2)
        
        # Sum contributions from all sources
        return np.sum(flux_per_source, axis=1)
    
    def compute_lead_field_matrix_grid(self, scanner, grid_points):
        """
        Computes Lead Field Matrix for a grid of potential source locations.
        For beamforming, we need L(r) for each grid point r.
        Assumes 3 orthogonal dipoles at each grid point.
        """
        n_sensors = len(scanner.sensors)
        n_points = len(grid_points)
        L = np.zeros((n_sensors, n_points * 3))
        
        for j, pos in enumerate(grid_points):
            # 3 orthogonal dipoles
            for k, direction in enumerate(np.eye(3)):
                source = NeuralSource(pos, direction)
                # Using the simplified Biot-Savart for now
                sig = self.compute_lead_field(scanner, [source])
                L[:, j*3 + k] = sig
                
        return L

class SourceLocalizer:
    """Algorithms for reconstructing source configuration from sensor data."""
    
    @staticmethod
    def lcmv_beamformer(data, lead_field_matrix, grid_points, C_inv=None):
        """
        Linearly Constrained Minimum Variance Beamformer.
        
        data: (n_sensors, n_timepoints)
        lead_field_matrix: (n_sensors, n_gridpoints * 3)
        """
        n_sensors, n_timepoints = data.shape
        
        # Calculate covariance matrix
        if C_inv is None:
            C = np.cov(data) + 0.05 * np.eye(n_sensors) * np.trace(np.cov(data)) / n_sensors # Regularized
            C_inv = np.linalg.pinv(C)
            
        n_points = len(grid_points)
        source_power = np.zeros(n_points)
        
        for i in range(n_points):
            # Lead field for this location (3 components)
            Lf = lead_field_matrix[:, i*3:(i+1)*3]
            
            # Spatial filter weights
            # W = (Lf.T @ C_inv @ Lf)^-1 @ Lf.T @ C_inv
            denominator = Lf.T @ C_inv @ Lf
            try:
                denom_inv = np.linalg.pinv(denominator)
                weights = denom_inv @ Lf.T @ C_inv
                
                # Source activity
                y = weights @ data
                
                # Power
                source_power[i] = np.mean(np.sum(y**2, axis=0)) # Trace of source covariance
            except np.linalg.LinAlgError:
                source_power[i] = 0
                
        return source_power

    @staticmethod
    def minimum_norm_estimate(data, lead_field_matrix, alpha=0.1):
        """
        Minimum Norm Estimate (MNE).
        J = L.T @ (L @ L.T + alpha * I)^-1 @ data
        """
        n_sensors = data.shape[0]
        L = lead_field_matrix
        
        gram = L @ L.T
        reg = alpha * np.trace(gram) / n_sensors
        
        inverse_op = L.T @ np.linalg.pinv(gram + reg * np.eye(n_sensors))
        
        sources = inverse_op @ data
        
        # Combine 3 orientations
        n_points = L.shape[1] // 3
        sources_combined = np.zeros((n_points, data.shape[1]))
        
        for i in range(n_points):
            sources_combined[i] = np.linalg.norm(sources[i*3:(i+1)*3], axis=0)
            
        return sources_combined

def simulate_reconstruction():
    # 1. Setup Scanner
    scanner = MEGBoreScanner(radius=0.15, length=0.3, n_circumference=16, n_length=12)
    print(f"Scanner initialized with {len(scanner.sensors)} SQUID sensors.")

    # 2. Define Sources
    t = np.linspace(0, 1, 100)
    freq = 10 # Hz
    activation = np.sin(2 * np.pi * freq * t)
    
    source1 = NeuralSource(position=[0.05, 0.05, 0], moment=[1e-8, 0, 0], time_course=activation)
    source2 = NeuralSource(position=[-0.05, -0.02, 0.05], moment=[0, 1e-8, 0], time_course=activation) # Another source
    
    true_sources = [source1, source2]
    
    # 3. Simulate Data (Forward Pass)
    forward = ForwardPhysics()
    
    # Generate data over time
    n_sensors = len(scanner.sensors)
    data = np.zeros((n_sensors, len(t)))
    
    # Pre-compute L for specific true sources to speed up if static, 
    # but here position is static, moment direction is static, magnitude changes.
    for i, time_point in enumerate(t):
        current_sources = []
        for s in true_sources:
             # Scale moment by time course
             scaled_moment = s.moment * s.time_course[i]
             current_sources.append(NeuralSource(s.position, scaled_moment))
        
        field = forward.compute_lead_field(scanner, current_sources)
        
        # Add noise
        noise = np.random.normal(0, 1e-13, size=n_sensors) # 100 fT noise
        data[:, i] = field + noise

    print("Data simulated.")

    # 4. Inverse Problem (Reconstruction)
    # Define a search grid
    lim = 0.1
    grid_res = 10
    x = np.linspace(-lim, lim, grid_res)
    y = np.linspace(-lim, lim, grid_res)
    z = np.linspace(-lim, lim, grid_res)
    X, Y, Z = np.meshgrid(x, y, z)
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    print(f"Computing Lead Field for {len(grid_points)} grid points...")
    L_grid = forward.compute_lead_field_matrix_grid(scanner, grid_points)
    
    print("Running Beamformer...")
    localizer = SourceLocalizer()
    power_map = localizer.lcmv_beamformer(data, L_grid, grid_points)
    
    # normalize power map
    power_map /= np.max(power_map)
    
    return scanner, true_sources, grid_points, power_map, data

if __name__ == "__main__":
    scanner, sources, grid, power, data = simulate_reconstruction()
    
    # Visualization with Plotly
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # 1. Sensors
        sensor_pos = scanner.get_sensor_positions()
        sensor_trace = go.Scatter3d(
            x=sensor_pos[:, 0], y=sensor_pos[:, 1], z=sensor_pos[:, 2],
            mode='markers', marker=dict(size=4, color='blue', opacity=0.8),
            name='SQUID Sensors'
        )
        
        # 2. True Sources
        source_x = [s.position[0] for s in sources]
        source_y = [s.position[1] for s in sources]
        source_z = [s.position[2] for s in sources]
        source_trace = go.Scatter3d(
            x=source_x, y=source_y, z=source_z,
            mode='markers', marker=dict(size=10, color='red', symbol='diamond'),
            name='True Sources'
        )
        
        # 3. Reconstructed Activity
        # Filter for visualization
        threshold = 0.6 * np.max(power)
        mask = power > threshold
        
        if np.any(mask):
            active_grid = grid[mask]
            active_power = power[mask]
            
            recon_trace = go.Scatter3d(
                x=active_grid[:, 0], y=active_grid[:, 1], z=active_grid[:, 2],
                mode='markers', 
                marker=dict(
                    size=6, 
                    color=active_power, 
                    colorscale='Hot', 
                    opacity=0.8
                ),
                name='Reconstructed Source (Beamformer)'
            )
        else:
            recon_trace = go.Scatter3d(x=[], y=[], z=[], mode='markers', name='No active sources found')

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('MEG Scanner Geometry & True Sources', 'Source Localization (LCMV Beamformer)')
        )

        fig.add_trace(sensor_trace, row=1, col=1)
        fig.add_trace(source_trace, row=1, col=1)
        
        # Add simpler scanner context to second plot or just the reconstruction?
        # Let's add scanner as ghost to second plot for reference
        sensor_ghost = go.Scatter3d(
            x=sensor_pos[:, 0], y=sensor_pos[:, 1], z=sensor_pos[:, 2],
            mode='markers', marker=dict(size=2, color='gray', opacity=0.2),
            showlegend=False
        )
        fig.add_trace(sensor_ghost, row=1, col=2)
        fig.add_trace(recon_trace, row=1, col=2)

        fig.update_layout(
            title_text="MEG Signal Reconstruction Simulation",
            height=800,
            showlegend=True,
            scene=dict(aspectmode='data'),
            scene2=dict(aspectmode='data')
        )
        
        fig.write_html("meg_simulation_interactive.html")
        print("Interactive visualization saved to 'meg_simulation_interactive.html'")
        
    except ImportError:
        print("Plotly not installed, skipping interactive visualization.")
        
    # Keep Matplotlib as fallback or static image generator
    fig_mpl = plt.figure(figsize=(12, 5))
    
    # 1. Sensor Geometry
    ax1 = fig_mpl.add_subplot(121, projection='3d')
    sensor_pos = scanner.get_sensor_positions()
    ax1.scatter(sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2], c='b', marker='.', alpha=0.5, label='Sensors')
    
    # True sources
    for s in sources:
        ax1.scatter(s.position[0], s.position[1], s.position[2], c='r', s=100, marker='*', label='True Source')
    
    ax1.set_title("MEG Bore & Sources")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    
    # 2. Reconstruction (Maximum Intensity Projection or Thresholded)
    ax2 = fig_mpl.add_subplot(122, projection='3d')
    
    # Filter grid points with high power
    threshold = 0.5
    mask = power > threshold
    
    if np.any(mask):
        active_grid = grid[mask]
        active_power = power[mask]
        
        img = ax2.scatter(active_grid[:, 0], active_grid[:, 1], active_grid[:, 2], 
                         c=active_power, cmap='hot', s=50, alpha=0.8, label='Reconstruction')
        fig_mpl.colorbar(img, ax=ax2, label='Beamformer Power')
    else:
        ax2.text(0,0,0, "No source found above threshold")
        
    ax2.set_title("Reconstructed Source Power")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    
    plt.tight_layout()
    plt.savefig('meg_simulation_result.png')
    print("Simulation complete. Result saved to 'meg_simulation_result.png'")

class GeodesicPathFinder:
    """
    Computes 'Continuable Geodesics' between regions of interest (ROI) in the brain volume.
    Simulates white matter tracts or functional connectivity paths using a 
    distance-weighted metric.
    """
    @staticmethod
    def compute_path(start_pos, end_pos, obstacles=None, resolution=50):
        """
        Generates a geodesic path. In this simplified model, it's a 
        perturbed line (arc) to simulate biological variance/folding.
        """
        # Linear interpolation
        t = np.linspace(0, 1, resolution)
        path = np.outer(1-t, start_pos) + np.outer(t, end_pos)
        
        # Add 'Continuable' curvature (Geodesic effect on manifold)
        # We simulate a manifold curvature by adding a sine wave perpendicular to the path
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)
        if length < 1e-6: return path
        
        # Perpendicular vector
        perp = np.cross(direction, np.array([0,0,1]))
        if np.linalg.norm(perp) < 1e-6: perp = np.cross(direction, np.array([0,1,0]))
        perp /= np.linalg.norm(perp)
        
        # Curvature amplitude depends on length
        curvature = 0.2 * length * np.sin(np.pi * t)
        
        path_curved = path + np.outer(curvature, perp)
        return path_curved

class MRISliceGenerator:
    """Generates 2D Grayscale MRI slices with functional overlays."""
    
    @staticmethod
    def generate_slices(grid_points, power_map, grid_res):
        """
        Projects 3D power map onto 2D slices (Axial, Coronal, Sagittal).
        Generates synthetic MRI background.
        """
        # Reshape to 3D volume
        dim = int(round(len(grid_points)**(1/3))) # assuming cubic grid
        if dim**3 != len(grid_points):
            # Fallback if not perfectly cubic, try to approximate or use raw grid_res
            dim = grid_res
            
        vol_power = power_map.reshape((dim, dim, dim))
        
        # Synthetic MRI T1 background (Skull, Gray Matter, White Matter)
        # Simple spherical model
        x = np.linspace(-1, 1, dim)
        y = np.linspace(-1, 1, dim)
        z = np.linspace(-1, 1, dim)
        X, Y, Z = np.meshgrid(x, y, z)
        R = np.sqrt(X**2 + Y**2 + Z**2)
        
        mri_vol = np.zeros_like(R)
        mri_vol[R < 0.9] = 0.3 # Scalp
        mri_vol[R < 0.8] = 0.1 # Skull (Dark)
        mri_vol[R < 0.75] = 0.6 # Gray Matter
        mri_vol[R < 0.6] = 0.8 # White Matter
        
        # Generate Slices at center
        center = dim // 2
        
        slices = {
            'Axial': (mri_vol[:, :, center], vol_power[:, :, center]),
            'Coronal': (mri_vol[center, :, :].T, vol_power[center, :, :].T), # Transpose for better orientation
            'Sagittal': (mri_vol[:, center, :].T, vol_power[:, center, :].T)
        }
        
        return slices

if __name__ == "__main__":
    scanner, sources, grid, power, data = simulate_reconstruction()
    
    # --- Geodesic Computation ---
    print("Computing Continuable Geodesics between sources...")
    if len(sources) >= 2:
        path = GeodesicPathFinder.compute_path(sources[0].position, sources[1].position)
    else:
        path = np.array([])

    # --- 2D Grayscale Slice Generation ---
    print("Generating Grayscale MRI Slices with Beamformer Overlay...")
    slice_gen = MRISliceGenerator()
    # grid_res was 10 in simulate_reconstruction, let's assume it matches or pass it
    # We need to know grid resolution used. In simulate_reconstruction it is 10.
    slices = slice_gen.generate_slices(grid, power, grid_res=10)
    
    # Plot Slices
    fig_slices, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, (mri, func)) in enumerate(slices.items()):
        # Background: Grayscale MRI
        axes[i].imshow(mri, cmap='gray', origin='lower', extent=[-0.1, 0.1, -0.1, 0.1])
        
        # Overlay: Beamformer Power (Thresholded)
        # Mask low values for transparency
        func_masked = np.ma.masked_where(func < 0.4, func)
        axes[i].imshow(func_masked, cmap='jet', alpha=0.6, origin='lower', extent=[-0.1, 0.1, -0.1, 0.1])
        
        axes[i].set_title(f"{name} Slice\nwith Beamformer Overlay")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('meg_mri_slices.png')
    print("Generated 'meg_mri_slices.png'")

    # --- Update Interactive Visualization ---
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # 1. Sensors
        sensor_pos = scanner.get_sensor_positions()
        sensor_trace = go.Scatter3d(
            x=sensor_pos[:, 0], y=sensor_pos[:, 1], z=sensor_pos[:, 2],
            mode='markers', marker=dict(size=4, color='blue', opacity=0.8),
            name='SQUID Sensors'
        )
        
        # 2. True Sources
        source_x = [s.position[0] for s in sources]
        source_y = [s.position[1] for s in sources]
        source_z = [s.position[2] for s in sources]
        source_trace = go.Scatter3d(
            x=source_x, y=source_y, z=source_z,
            mode='markers', marker=dict(size=10, color='red', symbol='diamond'),
            name='True Sources'
        )
        
        # 3. Geodesic Path
        if len(path) > 0:
            geo_trace = go.Scatter3d(
                x=path[:, 0], y=path[:, 1], z=path[:, 2],
                mode='lines', line=dict(color='lime', width=5),
                name='Continuable Geodesic'
            )
        else:
            geo_trace = go.Scatter3d(x=[], y=[], z=[])
        
        # 4. Reconstructed Activity
        threshold = 0.6 * np.max(power)
        mask = power > threshold
        if np.any(mask):
            active_grid = grid[mask]
            active_power = power[mask]
            recon_trace = go.Scatter3d(
                x=active_grid[:, 0], y=active_grid[:, 1], z=active_grid[:, 2],
                mode='markers', 
                marker=dict(size=6, color=active_power, colorscale='Hot', opacity=0.8),
                name='Reconstructed Source'
            )
        else:
            recon_trace = go.Scatter3d(x=[], y=[], z=[], mode='markers', name='No active sources')

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('MEG Scanner Geometry & True Sources', 'Source, Geodesics & Beamformer')
        )

        fig.add_trace(sensor_trace, row=1, col=1)
        fig.add_trace(source_trace, row=1, col=1)
        if len(path) > 0: fig.add_trace(geo_trace, row=1, col=1)
        
        # Right Plot
        fig.add_trace(recon_trace, row=1, col=2)
        if len(path) > 0: fig.add_trace(geo_trace, row=1, col=2)
        fig.add_trace(source_trace, row=1, col=2) # Show true sources on recon too

        fig.update_layout(
            title_text="MEG Signal Reconstruction: Beamformer & Geodesics",
            height=800,
            showlegend=True,
            scene=dict(aspectmode='data'),
            scene2=dict(aspectmode='data'),
            template='plotly_dark'
        )
        
        fig.write_html("meg_simulation_interactive.html")
        print("Interactive visualization saved to 'meg_simulation_interactive.html'")
        
    except ImportError:
        print("Plotly not installed, skipping interactive visualization.")
