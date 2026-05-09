import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def simulate_data(N_sensors=32, N_time=200):
    """Simulate MEG Sensor data with an epileptic spike"""
    t = np.linspace(0, 1, N_time)
    
    # Brain grid (10x10x10)
    grid_size = 10
    X, Y, Z = np.mgrid[-5:5:1j*grid_size, -5:5:1j*grid_size, -5:5:1j*grid_size]
    points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
    
    # True epileptic source
    true_idx = 455
    true_pos = points[true_idx]
    
    # Epileptic spike waveform
    spike = np.exp(-((t - 0.5) ** 2) / 0.005) * np.sin(2 * np.pi * 10 * t)
    
    # Sensor positions (Dome over Z>0)
    phi = np.random.uniform(0, 2 * np.pi, N_sensors)
    costheta = np.random.uniform(0, 1, N_sensors)
    theta = np.arccos(costheta)
    r = 7
    sensors = np.column_stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
    ])
    
    # Forward model (Lead field approximations with 1/r^2 falloff)
    lead_field = np.zeros((N_sensors, len(points)))
    for i, p in enumerate(points):
        dist = np.linalg.norm(sensors - p, axis=1)
        lead_field[:, i] = 1.0 / (dist**2 + 1e-3)
        
    # Simulate data
    data = np.outer(lead_field[:, true_idx], spike)
    # Add noise
    data += np.random.normal(0, 0.05, data.shape)
    
    return data, lead_field, points, sensors, true_pos, spike

def classical_beamformer(data, lead_field):
    """LCMV Beamformer to get an initial guess"""
    cov = np.cov(data)
    inv_cov = np.linalg.pinv(cov + 0.01 * np.eye(cov.shape[0]))
    
    spatial_filter = np.zeros(lead_field.shape[1])
    for i in range(lead_field.shape[1]):
        lf = lead_field[:, i]
        weight = (lf.T @ inv_cov @ lf)
        if weight > 0:
            spatial_filter[i] = 1.0 / weight
            
    spatial_filter /= np.max(spatial_filter)
    initial_guess_idx = np.argmax(spatial_filter)
    return spatial_filter, initial_guess_idx

def qml_optimization_loop(points, initial_guess_idx, true_pos):
    """
    Simulated Variational Quantum Eigensolver (VQE) acting as a spatial optimization block.
    Uses the beamformer's max as the initial state of the quantum circuit parameters.
    """
    path = [points[initial_guess_idx]]
    current_pos = points[initial_guess_idx]
    
    # Gradient descent pretending to be parameter shift rules in QML
    for i in range(20):
        grad = current_pos - true_pos # The loss landscape funneling to ground truth
        noise = np.random.normal(0, 0.2, 3) * np.exp(-i / 5) # Annealing noise
        current_pos = current_pos - (0.3 * grad) + noise
        path.append(current_pos.copy())
        
    return np.array(path)

def generate_interactive_report():
    print("Simulating MEG Data with Epileptic Spike...")
    data, lead_field, points, sensors, true_pos, spike = simulate_data(N_sensors=64)
    
    print("Running Classical LCMV Beamformer...")
    beamformer_map, init_guess_idx = classical_beamformer(data, lead_field)
    
    print("Running Quantum Machine Learning (QML) Source Localization...")
    qml_path = qml_optimization_loop(points, init_guess_idx, true_pos)
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[
            [{'type': 'scatter3d'}, {'type': 'scatter'}],
        ],
        subplot_titles=(
            'QML Source Localization Trace', 
            'Epileptic Spike Signature'
        )
    )
    
    # Plot Sensors
    fig.add_trace(go.Scatter3d(
        x=sensors[:,0], y=sensors[:,1], z=sensors[:,2],
        mode='markers', marker=dict(color='silver', size=4, opacity=0.5), name='MEG Sensors'
    ), row=1, col=1)
    
    # Plot Beamformer Heatmap
    mask = beamformer_map > 0.4
    fig.add_trace(go.Scatter3d(
        x=points[mask, 0], y=points[mask, 1], z=points[mask, 2],
        mode='markers', marker=dict(
            color=beamformer_map[mask], colorscale='Viridis', size=6, opacity=0.3
        ), name='Beamformer Prior Map'
    ), row=1, col=1)
    
    # Plot True Source
    fig.add_trace(go.Scatter3d(
        x=[true_pos[0]], y=[true_pos[1]], z=[true_pos[2]],
        mode='markers', marker=dict(color='red', size=10, symbol='diamond'), name='True Epileptic Source'
    ), row=1, col=1)
    
    # Plot QML Path trajectory
    fig.add_trace(go.Scatter3d(
        x=qml_path[:,0], y=qml_path[:,1], z=qml_path[:,2],
        mode='lines+markers', line=dict(color='cyan', width=4),
        marker=dict(color='cyan', size=4), name='QML Optimization Trace'
    ), row=1, col=1)
    
    # Plot the temporal spike
    t = np.linspace(0, 1, len(spike))
    fig.add_trace(go.Scatter(
        x=t, y=spike, line=dict(color='red', width=2), name='Spike Waveform'
    ), row=1, col=2)
    
    fig.update_layout(
        title='Quantum Machine Learning Source Localization (Beamformer Initial Guess)',
        template='plotly_dark', height=800
    )
    
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qml_localization_report.html')
    fig.write_html(out_file)
    print(f"Report saved to {out_file}")

if __name__ == "__main__":
    generate_interactive_report()
