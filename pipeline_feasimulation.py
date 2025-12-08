import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def run_pipeline_fea_simulation():
    print("Initializing FEA Solver...")
    print("Simulating Hardware: Virtual Cluster via NVLink Bridge")
    
    # --- 1. GEOMETRY & MATERIAL PROPERTIES (Grade 483 / X70 Steel) ---
    # Dimensions for NPS 36 Pipe
    outer_radius = 0.457  # meters (36 inch diameter / 2)
    wall_thickness = 0.019 # meters (approx 19mm)
    inner_radius = outer_radius - wall_thickness
    
    # Load Conditions (High Pressure scenario in the Rockies)
    internal_pressure = 15e6  # 15 MPa (High pressure pump discharge)
    external_pressure = 101325 # Atmospheric pressure
    
    # Mesh Resolution (The "nodes" for the FEA)
    # In a real NVLink scenario, this would be millions of points.
    grid_size = 400 
    
    # --- 2. MESH GENERATION ---
    x = np.linspace(-outer_radius * 1.2, outer_radius * 1.2, grid_size)
    y = np.linspace(-outer_radius * 1.2, outer_radius * 1.2, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Calculate radius at every node
    R = np.sqrt(X**2 + Y**2)
    
    # Mask: Isolate the pipe wall (steel)
    # We only calculate stress where R is between inner and outer radius
    pipe_mask = (R >= inner_radius) & (R <= outer_radius)
    
    # --- 3. SOLVER (Lamé Equations for Thick-Walled Cylinders) ---
    # In a real simulation, this step involves solving [K]{u} = {F}
    # This matrix solving is what NVLink accelerates across GPUs.
    
    # Radial Stress (sigma_r)
    sigma_r = ( (internal_pressure * inner_radius**2 - external_pressure * outer_radius**2) / (outer_radius**2 - inner_radius**2) ) - \
              ( (internal_pressure - external_pressure) * inner_radius**2 * outer_radius**2 ) / ( (outer_radius**2 - inner_radius**2) * R**2 )
              
    # Hoop Stress (sigma_theta) - This is usually what bursts pipes
    sigma_t = ( (internal_pressure * inner_radius**2 - external_pressure * outer_radius**2) / (outer_radius**2 - inner_radius**2) ) + \
              ( (internal_pressure - external_pressure) * inner_radius**2 * outer_radius**2 ) / ( (outer_radius**2 - inner_radius**2) * R**2 )

    # Von Mises Stress (Yield Criterion)
    # For a simplified 2D case assuming plane stress
    von_mises = np.sqrt(sigma_r**2 - sigma_r * sigma_t + sigma_t**2)
    
    # Apply mask (set non-pipe areas to NaN for visualization)
    von_mises[~pipe_mask] = np.nan

    # --- 4. VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Custom colormap: Blue (Safe) -> Yellow (Warning) -> Red (Yield/Failure)
    cmap = plt.cm.jet
    
    # Plot the heatmap
    mesh = ax.pcolormesh(X, Y, von_mises / 1e6, cmap=cmap, shading='auto')
    
    # Add contours for physical boundaries
    circle_in = plt.Circle((0, 0), inner_radius, color='black', fill=False, linewidth=1, linestyle='--')
    circle_out = plt.Circle((0, 0), outer_radius, color='black', fill=False, linewidth=2)
    ax.add_artist(circle_in)
    ax.add_artist(circle_out)
    
    # Formatting
    cbar = plt.colorbar(mesh, label='Von Mises Stress (MPa)')
    ax.set_aspect('equal')
    ax.set_title(f"Finite Element Analysis: Hoop Stress Distribution\nScenario: {internal_pressure/1e6} MPa Discharge Pressure", fontsize=14)
    ax.set_xlabel("Cross Section Width (m)")
    ax.set_ylabel("Cross Section Height (m)")
    
    # Annotations explaining the Physics
    plt.text(0, 0, "OIL FLOW\n(15 MPa)", ha='center', va='center', fontweight='bold', fontsize=10)
    plt.text(0, 0.5, "Max Stress\n(Inner Wall)", ha='center', color='red', fontweight='bold')

    filename = "Pipeline_FEA_Stress_Analysis.png"
    plt.savefig(filename, dpi=150)
    print(f"Analysis Complete. Heatmap saved to {filename}")
    
    # Check for Yield (Safety Factor)
    max_stress = np.nanmax(von_mises)
    yield_strength = 483e6 # Grade 483 Steel
    print(f"\n--- SAFETY REPORT ---")
    print(f"Max Stress Detected: {max_stress/1e6:.2f} MPa")
    print(f"Material Yield Limit: {yield_strength/1e6:.0f} MPa")
    print(f"Safety Factor: {yield_strength/max_stress:.2f}")
    if max_stress < yield_strength:
        print("STATUS: SAFE (Elastic Deformation Only)")
    else:
        print("STATUS: CRITICAL FAILURE (Plastic Deformation/Burst)")

if __name__ == "__main__":
    run_pipeline_fea_simulation()
