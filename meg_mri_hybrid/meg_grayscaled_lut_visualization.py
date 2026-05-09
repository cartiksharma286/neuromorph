import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_interactive_report():
    print("Generating MEG Visualization with Grayscaled Neuroimages and Waveforms...")
    
    # 1. Simulate a synthetic MRI slice (grayscale LUT)
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    
    # Synthesize cortical structures
    brain = np.exp(-(X**2 + Y**2)/12) * 0.85
    ventricles = np.exp(-((X-1)**2 + Y**2)/1.5) * -0.4 + np.exp(-((X+1)**2 + Y**2)/1.5) * -0.4
    sulfi = np.sin(X*5) * np.cos(Y*5) * 0.05
    neuroimage = np.clip(brain + ventricles + sulfi + np.random.normal(0, 0.02, X.shape), 0, 1)

    # 2. Simulate MEG Temporal Waveforms
    t = np.linspace(0, 3, 1000)
    waveforms = []
    colors = ['cyan', 'magenta', '#00ff00', 'orange', '#ff3333', 'yellow']
    
    for i in range(6):
        # Oscillatory bursts representing neural rhythms (alpha, beta, gamma bands)
        freq = np.random.uniform(8, 30)
        envelope = np.exp(-((t - np.random.uniform(0.5, 2.5))**2) / 0.15)
        wave = envelope * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.05, len(t))
        waveforms.append(wave - i * 2.5) # Spatial vertical offset for multi-channel plotting

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.45, 0.55],
        horizontal_spacing=0.08,
        subplot_titles=('LUT Mapped Grayscale Neuroimage (Structural)', 'MEG Coregistered Waveforms (Functional)')
    )
    
    # Heatmap for the neuroimage (Grayscale LUT)
    fig.add_trace(go.Heatmap(
        z=neuroimage, colorscale='gray',
        showscale=False, name='Neuroimage Slice'
    ), row=1, col=1)
    
    # Add a mock MEG sensor projection overlay
    theta = np.linspace(0, 2*np.pi, 20)
    r = 75
    sensor_x = 100 + r * np.cos(theta)
    sensor_y = 100 + r * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=sensor_x, y=sensor_y, mode='markers',
        marker=dict(color='rgba(0, 255, 255, 0.5)', size=6, symbol='circle-open'),
        name='Sensor Array Array Projection'
    ), row=1, col=1)

    # Add active focal spike points
    fig.add_trace(go.Scatter(
        x=[60, 140], y=[120, 130], mode='markers',
        marker=dict(color='red', size=12, symbol='cross'),
        name='Active Epileptic Foci'
    ), row=1, col=1)

    # Plot waveforms
    for i, wave in enumerate(waveforms):
        fig.add_trace(go.Scatter(
            x=t, y=wave, mode='lines', line=dict(color=colors[i], width=1.5),
            name=f'SQUID Channel {i+1}'
        ), row=1, col=2)
        
    fig.update_layout(
        template='plotly_dark',
        height=800,
        title='Coregistered MEG Functional Waveforms with Structural MRI Context',
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        xaxis2=dict(title='Time (seconds)', showgrid=True, gridcolor='#333'),
        yaxis2=dict(title='Magnetic Flux (fT)', showgrid=False, zeroline=False, showticklabels=False)
    )
    
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meg_neuroimage_lut_report.html')
    fig.write_html(out_file)
    print(f"Report saved to {out_file}")

if __name__ == "__main__":
    generate_interactive_report()