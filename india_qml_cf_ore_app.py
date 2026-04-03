import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

app = dash.Dash(__name__)
np.random.seed(108)

# ── Mathematical Engine: QML + Continued Fractions ────────────────────────────
def qml_continued_fraction_hybrid_model(n_points, bounding_box, centers, phase_shift=0.0, cf_depth=3):
    """
    Hybrid Physics/Math Engine:
    1. QML Superposition: Computes probability amplitude P = |Ψ|^2 across multiple nodes.
    2. Continued Fractions (CF): Applies fractal noise modulation representing
       structural fracturing and mineral ore permeation anomalies.
    """
    lat_min, lat_max, lon_min, lon_max = bounding_box
    
    # Base spatial sampling
    lats = np.random.uniform(lat_min, lat_max, n_points)
    lons = np.random.uniform(lon_min, lon_max, n_points)
    
    # 1. QML Superposition
    psi = np.zeros(n_points, dtype=np.complex128)
    classical_prob = np.zeros(n_points)
    
    for c in centers:
        # Distance squared from geometric quantum node
        r_sq = ((lats - c['lat'])**2 + (lons - c['lon'])**2) / (2 * c['spread']**2)
        # Quantum state amplitude
        amp = c['weight'] * np.exp(-r_sq)
        phase = c['k'] * r_sq + c['phase'] + phase_shift
        
        psi += amp * np.exp(1j * phase)
        classical_prob += (amp)**2
        
    probability_density = np.abs(psi)**2
    
    # Normalized interference (Entanglement proxy)
    interference = probability_density - classical_prob
    entanglement = np.abs(interference) / (probability_density + 1e-4)

    # 2. Continued Fraction (CF) Modulator (Fractal Ore Veining)
    cf_multiplier = np.zeros(n_points)
    
    for i in range(n_points):
        # We compute CF based on distance to the closest center
        min_dist = min([np.sqrt((lats[i]-c['lat'])**2 + (lons[i]-c['lon'])**2) for c in centers])
        if min_dist == 0: 
            cf_multiplier[i] = 1.0
            continue
            
        r = min_dist
        eps = 0.05
        cf = 0.0
        for _ in range(cf_depth):
            cf = 1.0 / (abs(np.sin(np.pi / (r + eps))) + cf + eps)
        # Normalize
        cf_multiplier[i] = min(abs(cf), 2.5) / 2.5

    # 3. Hybrid Vis-à-Vis Score
    # The final composite signature relies on high QML probability *AND* favorable CF veining patterns
    composite_signature = probability_density * (0.3 + 0.7 * cf_multiplier)
    
    return lats, lons, probability_density, cf_multiplier, composite_signature, entanglement

# ── Region Constraints & Subsurface Targets ────────────────────────────────────

INDIA_ORE_CENTERS = [
    # Chota Nagpur Plateau (Jharkhand/Odisha/Chhattisgarh) 
    {"lat": 23.00, "lon": 85.00, "weight": 2.2, "spread": 2.0, "k": 12, "phase": 0.0, "label": "Chota Nagpur Superstate (Fe/Coal/Al)"},
    # Dharwar Craton (Karnataka)
    {"lat": 15.00, "lon": 76.00, "weight": 1.7, "spread": 1.8, "k": 8, "phase": np.pi/3, "label": "Dharwar Craton (Au/Fe/Mn)"},
    # Aravalli Craton (Rajasthan)
    {"lat": 25.00, "lon": 74.00, "weight": 1.5, "spread": 1.5, "k": 10, "phase": -np.pi/4, "label": "Aravalli Phase-State (Zn/Pb/Cu/Ag)"},
    # Singhbhum Shear Zone (Jharkhand)
    {"lat": 22.50, "lon": 86.00, "weight": 1.8, "spread": 1.0, "k": 14, "phase": np.pi/2, "label": "Singhbhum Manifold (Cu/U/Fe)"},
    # Eastern Ghats / Andhra Pradesh
    {"lat": 18.00, "lon": 82.50, "weight": 1.3, "spread": 1.4, "k": 9, "phase": -np.pi/6, "label": "Eastern Ghats Node (Bauxite)"}
]

# Generate static dataframe
def generate_df(phase_shift):
    bounds = (8.0, 35.0, 68.0, 98.0) # Lat min, Lat max, Lon min, Lon max (India roughly)
    lats, lons, prob, cf_mod, composite, ent = qml_continued_fraction_hybrid_model(
        7000, bounds, INDIA_ORE_CENTERS, phase_shift=phase_shift, cf_depth=4
    )
    
    labels = []
    for i in range(len(lats)):
        dist = [((lats[i]-c["lat"])**2 + (lons[i]-c["lon"])**2) for c in INDIA_ORE_CENTERS]
        labels.append(INDIA_ORE_CENTERS[np.argmin(dist)]["label"])
        
    df = pd.DataFrame({
        "Latitude": lats,
        "Longitude": lons,
        "QML_Density": prob,
        "Continued_Fraction_Veining": cf_mod,
        "Composite_Signature": composite,
        "Entanglement": ent,
        "Craton_Target": labels
    })
    
    # Filter to simulate spatial threshold collapses for prominent mineral belts
    df = df[df["Composite_Signature"] > df["Composite_Signature"].quantile(0.60)]
    return df

# ── App Layout ────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"backgroundColor": "#1a0f00", "minHeight": "100vh", "padding": "24px", "fontFamily": "Segoe UI, sans-serif"},
    children=[
        html.H1("Indian Mineral Ore Discovery Engine", style={"color": "#ffaa00", "textAlign": "center", "marginBottom": "5px"}),
        html.P("Quantum Machine Learning Superpositions Vis-à-Vis Continued Fraction Fracturing", 
               style={"color": "#aa8866", "textAlign": "center", "marginBottom": "20px"}),
        
        html.Div([
            html.Div([
                html.Label("QML Ansatz Phase Angle Shift (radians θ)", style={"color": "#ffcc88"}),
                dcc.Slider(id="phase-slider", min=0, max=2*np.pi, step=0.1, value=0.0, marks={0: "0", 3.14: "π", 6.28: "2π"}),
                html.Br(),
                html.Label("Min Composite Ore Signature Strength Filter", style={"color": "#ffcc88"}),
                dcc.Slider(id="prob-slider", min=0, max=1.0, step=0.05, value=0.2, marks={i/10: f"{i/10}" for i in range(0, 11, 2)}),
            ], style={"padding": "20px", "backgroundColor": "#2a1a00", "borderRadius": "10px", "margin": "15px 0", "border": "1px solid #4a2a00"}),
            
            dcc.Graph(id="hybrid-map", style={"height": "65vh", "borderRadius": "10px", "overflow": "hidden"}),
            
            html.Div(id="stats-panels", style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "15px", "marginTop": "20px"})
        ])
    ]
)

@app.callback(
    Output("hybrid-map", "figure"),
    Output("stats-panels", "children"),
    Input("phase-slider", "value"),
    Input("prob-slider", "value")
)
def update_view(phase, min_prob):
    df = generate_df(phase)
    
    max_c = df["Composite_Signature"].max()
    df = df[df["Composite_Signature"] >= min_prob * max_c]
    
    fig = px.scatter_map(
        df, lat="Latitude", lon="Longitude", color="Craton_Target",
        size="Composite_Signature", size_max=16, opacity=0.85,
        hover_name="Craton_Target",
        hover_data={
            "Latitude": ":.4f", "Longitude": ":.4f",
            "QML_Density": ":.3f",
            "Continued_Fraction_Veining": ":.3f",
            "Composite_Signature": ":.4f"
        },
        zoom=4.0, center={"lat": 22.0, "lon": 82.0},
        map_style="carto-darkmatter",
        title="Composite Mineral Ore Potential (Indian Cratons & Basins)"
    )
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0}, paper_bgcolor="#1a0f00", plot_bgcolor="#1a0f00",
        font=dict(color="#ffddaa"), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(26,15,0,0.7)")
    )
    
    # Graphs
    h1 = go.Figure(go.Histogram(x=df["QML_Density"], marker_color="#ff5555", nbinsx=40))
    h1.update_layout(title="QML Probability Amplitude |Ψ|²", paper_bgcolor="#2a1a00", plot_bgcolor="#2a1a00", font=dict(color="#ccc"), margin=dict(l=30,r=10,t=30,b=20))
    
    h2 = go.Figure(go.Histogram(x=df["Continued_Fraction_Veining"], marker_color="#00ffcc", nbinsx=40))
    h2.update_layout(title="Continued Fraction (CF) Veining Index", paper_bgcolor="#2a1a00", plot_bgcolor="#2a1a00", font=dict(color="#ccc"), margin=dict(l=30,r=10,t=30,b=20))
    
    h3 = go.Figure(go.Histogram(x=df["Composite_Signature"], marker_color="#ffaa00", nbinsx=40))
    h3.update_layout(title="Hybrid Composite Ore Target Strength", paper_bgcolor="#2a1a00", plot_bgcolor="#2a1a00", font=dict(color="#ccc"), margin=dict(l=30,r=10,t=30,b=20))
    
    return fig, [
        dcc.Graph(figure=h1, style={"height": "250px", "borderRadius": "8px"}),
        dcc.Graph(figure=h2, style={"height": "250px", "borderRadius": "8px"}),
        dcc.Graph(figure=h3, style={"height": "250px", "borderRadius": "8px"})
    ]

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8058)