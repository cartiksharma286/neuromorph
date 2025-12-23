import io
import base64
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

def run_mri_fea_simulation(params=None):
    if params is None:
        params = {}
        
    print(f"Initializing MRI Coil FEA with params: {params}")
    
    # --- 1. CONFIGURATION & CONSTANTS ---
    mu0 = 4 * np.pi * 1e-7
    
    # Physics Parameters from Input
    b0_target = float(params.get('b0_strength', 3.0)) # Tesla
    coil_radius_mm = float(params.get('coil_radius', 120))
    coil_radius = coil_radius_mm / 1000.0 # convert to meters
    
    # Map B0 target to approximate current density (simplified solenoid model relation)
    # B ~= mu0 * n * I. Here we just scale J proportionally.
    # Base reference: 1e7 A/m^2 gives approx 0.12 T in this 2D config.
    # So 3T needs approx 2.5e8. Let's calibrate roughly:
    # 0.128T was achieved with 1e7.
    current_density_b0 = (b0_target / 0.128) * 1e7
    
    # RF Parameters
    # Assume RF current is smaller
    current_density_rf = current_density_b0 * 0.01 

    # Domain
    L = 0.8  # 80cm FOV
    N = 200  # Optimized grid for higher fidelity
    h = L / (N - 1)
    
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    
    # --- 2. DEFINE COIL GEOMETRIES ---
    J_main = np.zeros((N, N))
    
    # Main Magnet positions (fixed for bore)
    magnet_radius = 0.35 
    magnet_width = 0.1
    magnet_thickness = 0.05
    
    mask_main_top = (np.abs(Y - magnet_radius) < magnet_thickness) & (np.abs(X) < magnet_width)
    mask_main_bot = (np.abs(Y + magnet_radius) < magnet_thickness) & (np.abs(X) < magnet_width)
    J_main[mask_main_top] = current_density_b0
    J_main[mask_main_bot] = -current_density_b0
    
    # RF Coil (Dynamic Radius)
    J_rf = np.zeros((N, N))
    rf_elem_size = 0.015
    
    # Simple loop pair model for the head coil
    mask_rf_left = ((X + coil_radius)**2 + Y**2) < rf_elem_size**2
    mask_rf_right = ((X - coil_radius)**2 + Y**2) < rf_elem_size**2
    
    J_rf[mask_rf_left] = current_density_rf
    J_rf[mask_rf_right] = -current_density_rf
    
    b_vec_main = -mu0 * J_main.flatten() * h**2
    b_vec_rf = -mu0 * J_rf.flatten() * h**2

    # --- 3. SOLVER (Laplacian) ---
    main_diag = -4 * np.ones(N*N)
    off_diag_x = np.ones(N*N)
    off_diag_y = np.ones(N*N)
    diagonals = [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y]
    offsets = [0, -1, 1, -N, N]
    Laplacian = sp.diags(diagonals, offsets, shape=(N*N, N*N), format='csc')
    
    # Solve
    A_main = sp.linalg.spsolve(Laplacian, b_vec_main).reshape((N, N))
    A_rf = sp.linalg.spsolve(Laplacian, b_vec_rf).reshape((N, N))
    
    # --- 4. FIELDS ---
    By_m, Bx_m = np.gradient(A_main, h) # Remember gradient returns (rows, cols) -> (y, x)
    By_m = -By_m 
    Bx_m = np.gradient(A_main, axis=0) / h
    By_m = -np.gradient(A_main, axis=1) / h
    B_mag_m = np.sqrt(Bx_m**2 + By_m**2)

    Bx_rf = np.gradient(A_rf, axis=0) / h
    By_rf = -np.gradient(A_rf, axis=1) / h
    B_mag_rf = np.sqrt(Bx_rf**2 + By_rf**2) * 1000 # Scale for visibility (mT)
    
    # --- 5. VISUALIZATION (In-Memory) ---
    plt.switch_backend('Agg') # Ensure non-interactive backend
    plt.style.use('dark_background') # Use dark style for plots
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    fig.patch.set_facecolor('#0f172a') # Dark blue-gray background matching UI likely
    
    # B0 Plot
    im1 = axes[0].pcolormesh(X, Y, B_mag_m, cmap='inferno', shading='auto')
    axes[0].set_title(f"Main Field B0\nTarget: {b0_target}T", color='white')
    axes[0].set_aspect('equal')
    axes[0].axis('off')
    
    # B1 Plot
    im2 = axes[1].pcolormesh(X, Y, B_mag_rf, cmap='viridis', shading='auto')
    axes[1].set_title(f"RF B1 Field (mT)\nRadius: {coil_radius_mm}mm", color='white')
    axes[1].set_aspect('equal')
    axes[1].axis('off')
    
    # Overlay coil geometry
    coil_circ1 = Circle((-coil_radius, 0), rf_elem_size, fill=False, color='white', lw=1.5)
    coil_circ2 = Circle((coil_radius, 0), rf_elem_size, fill=False, color='white', lw=1.5)
    axes[1].add_patch(coil_circ1)
    axes[1].add_patch(coil_circ2)

    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    
    image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Stats
    center_idx = N // 2
    actual_b0 = B_mag_m[center_idx, center_idx]
    max_b1 = np.max(B_mag_rf)
    
    stats = {
        "b0_strength": float(actual_b0),
        "max_b1_strength": float(max_b1 / 1000.0), # convert back to T
        "deformation": "None (Elasticity Disabled)"
    }
    
    return stats, f"data:image/png;base64,{image_b64}"
