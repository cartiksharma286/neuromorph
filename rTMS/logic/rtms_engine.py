import numpy as np
import time

def simulate_fea(resolution=20):
    """
    Simulates Finite Element Analysis (FEA) of cortical manifolds for E-field distribution.
    Returns a mock 2D grid of electromagnetic field intensities.
    """
    # Create a 2D grid representing cortical surface E-field
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Simulate a focal hotspot (e.g., motor cortex or DLPFC)
    hotspot_x, hotspot_y = np.random.uniform(-0.5, 0.5, 2)
    Z = np.exp(-((X - hotspot_x)**2 + (Y - hotspot_y)**2) / 0.1)
    
    # Add some structural noise (sulci/gyri geometry effects)
    noise = np.sin(X * 10) * np.cos(Y * 10) * 0.1
    Z = Z + noise
    
    return Z.tolist()

def simulate_bem(nodes=50):
    """
    Simulates Boundary Element Method (BEM) for head tissue boundaries to account
    for skin, skull, CSF, and gray matter conductivity differences.
    """
    # Return a 3D boundary geometry array (x,y,z) with scalar potentials
    theta = np.linspace(0, 2 * np.pi, nodes)
    phi = np.linspace(0, np.pi, nodes // 2)
    THETA, PHI = np.meshgrid(theta, phi)
    
    R = 1.0 + 0.05 * np.sin(4 * THETA) * np.cos(4 * PHI)  # Simple deformed sphere model
    
    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)
    
    # Flatten it to just a list of vertices and a mock potential value
    vertices = []
    for i in range(len(X)):
        for j in range(len(X[0])):
            potential = np.exp(-Z[i][j]**2) * 10.0 # simulated BEM potential
            vertices.append({
                "x": round(float(X[i][j]), 3),
                "y": round(float(Y[i][j]), 3),
                "z": round(float(Z[i][j]), 3),
                "c": round(float(potential), 3)
            })
            
    # Sample down to keep payload manageable
    np.random.shuffle(vertices)
    return vertices[:500]

def optimize_protocol(condition="stroke"):
    """
    Uses statistical optimization techniques to dynamically adjust rTMS frequency,
    intensity, and pulse train duration for optimal neuromodulation based on condition.
    """
    # Mocking convergence over several iterations
    iterations = []
    current_freq = 1.0 # Hz
    current_intensity = 40.0 # % MSO
    
    target_freq = 10.0 if condition == "stroke" else 20.0
    target_intensity = 80.0 if condition == "stroke" else 100.0 # higher for dementia deep brain targets
    
    for i in range(20):
        # Statistical gradient descent step
        current_freq += (target_freq - current_freq) * 0.2 + np.random.normal(0, 0.5)
        current_intensity += (target_intensity - current_intensity) * 0.15 + np.random.normal(0, 1.0)
        
        # Calculate a mock fitness/cost function score (closer to 1.0 is better)
        fitness = 1.0 - (abs(target_freq - current_freq) / target_freq + abs(target_intensity - current_intensity) / target_intensity) / 2
        
        iterations.append({
            "iteration": i + 1,
            "frequency_hz": round(max(1.0, current_freq), 2),
            "intensity_mso": round(max(10.0, current_intensity), 2),
            "fitness": round(max(0.0, float(fitness)), 4)
        })
        
    return {
        "final_parameters": iterations[-1],
        "convergence_trajectory": iterations,
        "protocol_summary": f"Optimized {condition.capitalize()} Protocol: {iterations[-1]['frequency_hz']} Hz at {iterations[-1]['intensity_mso']}% MSO"
    }

def run_full_simulation(condition="stroke"):
    """
    Simulates Google Cloud Server execution of complex BEM/FEA and Statistical Optimization.
    """
    return {
        "condition": condition,
        "timestamp": time.time(),
        "cloud_node": "gcp-us-central1-c-tensor-node",
        "optimization": optimize_protocol(condition),
        "fea_grid": simulate_fea(),
        "bem_mesh": simulate_bem()
    }
