import dash
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy.fft import fft, ifft, fftfreq

# Initialize the Dash app
app = dash.Dash(__name__)

# --- 1. Pipeline Map Data & Schematics ---
# MB -> SK -> AB -> BC Route
pipeline_route = pd.DataFrame({
    'Lat': [49.8951, 50.4452, 51.0447, 53.5444, 53.9171, 49.2827],
    'Lon': [-97.1384, -104.6189, -114.0719, -113.4909, -122.7497, -123.1207],
    'Site': ['Winnipeg Node (MB)', 'Regina Hub (SK)', 'Calgary Modeling Site (AB)', 'Edmonton Hub (AB)', 'Prince George Site (BC)', 'Vancouver Terminal (BC)'],
    'Type': ['Hub', 'Waystation', 'Major Modeling Site', 'Major Modeling Site', 'Major Modeling Site', 'Terminal Hub'],
    'Geophysics_Score': [0.5, 0.6, 0.95, 0.9, 0.85, 0.7]
})

fig_map = go.Figure()

# Add Pipeline Line Schematic
fig_map.add_trace(go.Scattermap(
    mode="lines",
    lat=pipeline_route['Lat'],
    lon=pipeline_route['Lon'],
    line=dict(width=4, color="#ff00ff"),
    name="MB-AB-BC Primary Pipeline"
))

# Add Strategic Modeling Sites
fig_map.add_trace(go.Scattermap(
    mode="markers+text",
    lat=pipeline_route['Lat'],
    lon=pipeline_route['Lon'],
    text=pipeline_route['Site'],
    textposition="bottom right",
    marker=dict(
        size=pipeline_route['Geophysics_Score'] * 20, 
        color=pipeline_route['Geophysics_Score'], 
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title=dict(text="Geophys Signature Match", font=dict(color="white")), x=0.02, y=0.5, tickcolor="white", tickfont=dict(color="white"))
    ),
    name="Geophysical Sites"
))

fig_map.update_layout(
    map_style="carto-darkmatter",
    map=dict(center=dict(lat=51.5, lon=-110.0), zoom=3.5),
    margin={"r":0,"t":50,"l":0,"b":0},
    paper_bgcolor="#111111", 
    plot_bgcolor="#111111",
    title="MB-AB-BC Pipeline Route Schematics & Strategic Modeling Sites", 
    font=dict(color="white"),
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0.5)")
)


# --- 2. Signal Reconstruction for Oil Well Signatures ---
def generate_reconstruction_data():
    """Generates synthetic noisy geophysical telemetry and reconstructed signal"""
    np.random.seed(42)
    t = np.linspace(0, 1.0, 800)
    
    # Base structural geology frequencies + Oil signature frequency
    clean_sig = np.sin(2 * np.pi * 4 * t) + 0.6 * np.sin(2 * np.pi * 18 * t) + 0.3 * np.cos(2 * np.pi * 30 * t)
    
    # Extreme sensor noise (simulating deep well scatter)
    noisy_sig = clean_sig + np.random.normal(0, 1.2, t.shape)
    
    # Fast Fourier Transform (FFT) Signal Reconstruction
    yf = fft(noisy_sig)
    xf = fftfreq(len(t), 1/800)
    
    # Apply Low-Pass / Band-Pass Filter (optimize for < 35Hz oil signatures)
    yf_filtered = yf.copy()
    yf_filtered[np.abs(xf) > 35] = 0
    reconstructed_sig = np.real(ifft(yf_filtered))
    
    return t, clean_sig, noisy_sig, reconstructed_sig

t, clean_sig, noisy_sig, reconstructed_sig = generate_reconstruction_data()

fig_sig = make_subplots(
    rows=2, cols=1, 
    subplot_titles=("Raw Telemetry from Modeling Sites (High Noise)", "Optimized FFT Signal Reconstruction (Oil Signature Identified)"),
    vertical_spacing=0.15
)

# Raw Noisy Signal
fig_sig.add_trace(go.Scatter(x=t, y=noisy_sig, mode='lines', line=dict(color='rgba(255, 100, 100, 0.4)', width=1), name='Raw Sensor Data'), row=1, col=1)

# Reconstructed vs True
fig_sig.add_trace(go.Scatter(x=t, y=reconstructed_sig, mode='lines', line=dict(color='#00ffcc', width=2), name='Reconstructed Filtered Signal'), row=2, col=1)
fig_sig.add_trace(go.Scatter(x=t, y=clean_sig, mode='lines', line=dict(color='#ffffff', width=2, dash='dot'), name='True Model Signature'), row=2, col=1)

fig_sig.update_layout(
    paper_bgcolor="#111111", 
    plot_bgcolor="#1a1a1a", 
    font=dict(color="white"),
    height=600, 
    margin={"r":20,"t":60,"l":20,"b":20},
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
)
fig_sig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333333', title_text="Time (s)")
fig_sig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333', title_text="Amplitude")


# --- 3. 3D Geomorphic Oil Well Models ---
def generate_3d_geomorphology():
    # Create geological layers representing a subsurface reservoir
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    
    # Anticline trap (Dome structure)
    Z_cap = -100 - (X**2 + Y**2) * 2 + 15 * np.sin(X) * np.cos(Y)
    Z_res = Z_cap - 30 # Oil Reservoir layer
    Z_base = Z_res - 40 # Bedrock layer

    fig = go.Figure()

    # Add Cap rock surface
    fig.add_trace(go.Surface(z=Z_cap, x=X, y=Y, colorscale='Greys', opacity=0.5, name='Cap Rock', showscale=False))
    
    # Add Oil Reservoir surface
    fig.add_trace(go.Surface(z=Z_res, x=X, y=Y, colorscale='YlOrRd', opacity=0.9, name='Oil Reservoir', showscale=True, 
                             colorbar=dict(title=dict(text="Hydrocarbon<br>Saturation", font=dict(color="white")), x=0.9, tickfont=dict(color="white"))))
    
    # Add Base rock surface
    fig.add_trace(go.Surface(z=Z_base, x=X, y=Y, colorscale='Earth', opacity=0.5, name='Base Rock', showscale=False))

    # Add Wellbores
    # Well 1 (Strategic Vertical well tapping the dome crest)
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, -180], mode='lines', line=dict(color='cyan', width=8), name='Exploratory Well (Vertical)'))
    
    # Well 2 (Directional drilling)
    fig.add_trace(go.Scatter3d(x=[3, 1], y=[3, 1], z=[0, -190], mode='lines', line=dict(color='#ff00ff', width=8), name='Production Well (Directional)'))

    fig.update_layout(
        title='3D Geomorphic Model: Subsurface Reservoir & Wellbores',
        scene=dict(
            xaxis_title='Easting (km)',
            yaxis_title='Northing (km)',
            zaxis_title='Depth (m)',
            zaxis=dict(range=[-250, 0]),
            xaxis=dict(backgroundcolor="#1a1a1a", gridcolor="#333", showbackground=True),
            yaxis=dict(backgroundcolor="#1a1a1a", gridcolor="#333", showbackground=True)
        ),
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font=dict(color="white"),
        height=750,
        margin={"r":0,"t":50,"l":0,"b":0},
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.05, bgcolor="rgba(0,0,0,0.5)")
    )
    return fig

fig_3d = generate_3d_geomorphology()

# --- 4. App Layout Construction ---
app.layout = html.Div(style={'backgroundColor': '#111111', 'minHeight': '100vh', 'padding': '20px', 'color': 'white', 'fontFamily': 'sans-serif'}, children=[
    html.H1("MB-AB-BC Pipeline: Geophysical Signal Reconstruction", style={'textAlign': 'center', 'color': '#00ffcc', 'letterSpacing': '2px'}),
    html.P("Mapping strategic pipeline modeling sites across Manitoba, Alberta, and British Columbia, and optimizing sub-surface oil well telemetry.", style={'textAlign': 'center', 'fontSize': '16px', 'color': '#aaaaaa'}),
    
    html.Div(style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '20px'}, children=[
        dcc.Tabs(colors={'border': '#333333', 'primary': '#00ffcc', 'background': '#1a1a1a'}, children=[
            dcc.Tab(label='Pipeline Mapping & Signal Reconstruction', style={'backgroundColor': '#1a1a1a', 'color': 'white', 'border': 'none'}, selected_style={'backgroundColor': '#111111', 'color': '#00ffcc', 'borderTop': '2px solid #00ffcc', 'borderBottom': 'none'}, children=[
                html.Div([
                    # Pipeline map section
                    html.Div([
                        dcc.Graph(figure=fig_map, style={'height': '45vh', 'borderRadius': '12px', 'overflow': 'hidden', 'boxShadow': '0 4px 15px rgba(255, 0, 255, 0.15)'}),
                    ], style={'width': '100%', 'marginBottom': '30px', 'marginTop': '20px'}),
                    
                    # Signal reconstruction section
                    html.Div([
                        dcc.Graph(figure=fig_sig, style={'height': '600px', 'borderRadius': '12px', 'overflow': 'hidden', 'boxShadow': '0 4px 15px rgba(0, 255, 204, 0.15)'}),
                    ], style={'width': '100%'}),
                ])
            ]),
            dcc.Tab(label='3D Geomorphic Oil Wells', style={'backgroundColor': '#1a1a1a', 'color': 'white', 'border': 'none'}, selected_style={'backgroundColor': '#111111', 'color': '#00ffcc', 'borderTop': '2px solid #00ffcc', 'borderBottom': 'none'}, children=[
                html.Div([
                    dcc.Graph(figure=fig_3d, style={'height': '80vh', 'borderRadius': '12px', 'overflow': 'hidden', 'boxShadow': '0 4px 15px rgba(0, 255, 204, 0.15)'})
                ], style={'marginTop': '20px'})
            ])
        ])
    ])
])

if __name__ == '__main__':
    # Running on port 8060 to avoid conflicts with previous Dash apps
    app.run(debug=True, host='0.0.0.0', port=8060)
