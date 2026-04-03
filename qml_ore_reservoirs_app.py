import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

app = dash.Dash(__name__)

np.random.seed(88)

# ── Simulated Quantum Machine Learning Engine ────────────────────────────────
def simulate_quantum_wavefunction(n_points, bounding_box, quantum_centers, phase_shift=0.0):
    """
    Simulates a spatial probability distribution of ore reservoirs
    using a Quantum Machine Learning (QML) Ansatz state formulation.
    The density is defined by the probability amplitude P = |Ψ|^2, 
    where Ψ is a superposition of Gaussian basis states with phase interference.
    Additional metrics like Simulated Entanglement Entropy are derived from phase cross-terms.
    """
    lat_min, lat_max, lon_min, lon_max = bounding_box
    
    # Generate background states
    lats = np.random.uniform(lat_min, lat_max, n_points)
    lons = np.random.uniform(lon_min, lon_max, n_points)
    
    # Superposition state
    psi = np.zeros(n_points, dtype=np.complex128)
    
    for qc in quantum_centers:
        r_sq = ((lats - qc['lat'])**2 + (lons - qc['lon'])**2) / (2 * qc['spread']**2)
        # Quantum state component: Amplitude * exp(i * phase)
        amplitude = qc['weight'] * np.exp(-r_sq)
        phase = qc['k'] * r_sq + qc['phase'] + phase_shift
        psi += amplitude * np.exp(1j * phase)
        
    # Probability distribution P(x) = |Ψ(x)|^2
    probability_density = np.abs(psi)**2
    
    # Simulated Entanglement Score: a metric representing multi-center interference
    # Max when probability density differs drastically from a classical sum of probabilities
    classical_prob = np.zeros(n_points)
    for qc in quantum_centers:
        r_sq = ((lats - qc['lat'])**2 + (lons - qc['lon'])**2) / (2 * qc['spread']**2)
        classical_prob += (qc['weight'] * np.exp(-r_sq))**2
    
    # Normalized interference term (quantum minus classical)
    interference = probability_density - classical_prob
    entanglement_score = np.abs(interference) / (probability_density + 1e-5)
    
    return lats, lons, probability_density, entanglement_score


# ── Region Constraints & Definitions ─────────────────────────────────────────

# Ground truth simulated centers for quantum superposition
MASSACHUSETTS_CENTERS = [
    {"lat": 42.25, "lon": -73.35, "weight": 1.2, "spread": 0.15, "k": 15, "phase": 0.0, "label": "Berkshire Node"},
    {"lat": 42.50, "lon": -71.80, "weight": 0.9, "spread": 0.20, "k": 10, "phase": np.pi/3, "label": "Central MA Node"},
    {"lat": 41.80, "lon": -70.90, "weight": 0.7, "spread": 0.12, "k": 20, "phase": -np.pi/4, "label": "Southeastern Node"}
]

NORTH_AMERICA_CENTERS = [
    # Sudbury
    {"lat": 46.50, "lon": -81.00, "weight": 1.5, "spread": 1.5, "k": 8, "phase": 0.0, "label": "Sudbury Superstate"},
    # Carlin
    {"lat": 40.80, "lon": -116.3, "weight": 1.4, "spread": 2.0, "k": 5, "phase": np.pi/2, "label": "Nevada Superstate"},
    # Duluth
    {"lat": 47.50, "lon": -91.50, "weight": 1.1, "spread": 1.8, "k": 12, "phase": -np.pi/3, "label": "Duluth Superstate"},
    # Massachusetts (macro scale)
    {"lat": 42.30, "lon": -72.00, "weight": 0.8, "spread": 1.2, "k": 10, "phase": np.pi/6, "label": "New England Superstate"}
]

CANADA_OIL_ORE_CENTERS = [
    # WCSB (Alberta) - Oil/Gas dominant
    {"lat": 56.50, "lon": -113.00, "weight": 1.8, "spread": 2.5, "k": 10, "phase": np.pi/4, "label": "WCSB Phase-State (Oil)"},
    # Athabasca Basin - Uranium dominant + oil sands proximity
    {"lat": 58.20, "lon": -104.00, "weight": 1.5, "spread": 1.8, "k": 8, "phase": -np.pi/2, "label": "Athabasca Phase-State (Ore)"},
    # Sudbury - Multi-metallic ore
    {"lat": 46.50, "lon": -81.00, "weight": 1.3, "spread": 1.5, "k": 12, "phase": 0.0, "label": "Sudbury Phase-State (Ore)"},
    # Williston Basin - Oil
    {"lat": 49.50, "lon": -103.50, "weight": 1.4, "spread": 1.6, "k": 6, "phase": np.pi, "label": "Williston Phase-State (Oil)"}
]

CANADIAN_NW_CENTERS = [
    # Norman Wells - Oil Manifold
    {"lat": 65.28, "lon": -126.83, "weight": 1.7, "spread": 2.2, "k": 9, "phase": 0.0, "label": "Norman Wells Manifold (Oil)"},
    # Yellowknife / Slave Craton - Gold, Silver
    {"lat": 62.45, "lon": -114.37, "weight": 1.4, "spread": 1.5, "k": 14, "phase": np.pi/3, "label": "Slave Craton (Au/Ag)"},
    # Great Bear Magmatic Zone - Nickel, Titanium, Palladium (Ni/Ti/Pd)
    {"lat": 66.00, "lon": -118.00, "weight": 1.5, "spread": 1.8, "k": 11, "phase": -np.pi/4, "label": "Great Bear Phase (Ni/Ti/Pd)"},
    # Beaufort Delta - Oil / Gas
    {"lat": 68.35, "lon": -133.72, "weight": 1.6, "spread": 2.0, "k": 7, "phase": np.pi/2, "label": "Beaufort Delta Manifold (Oil)"}
]

SAUDI_ARABIA_CENTERS = [
    # Ghawar Field
    {"lat": 25.50, "lon": 49.50, "weight": 2.0, "spread": 1.5, "k": 12, "phase": 0.0, "label": "Ghawar Superstate (Oil)"},
    # Safaniya Field (offshore)
    {"lat": 28.00, "lon": 49.00, "weight": 1.7, "spread": 1.2, "k": 8, "phase": np.pi/4, "label": "Safaniya Phase-State (Oil)"},
    # Shaybah Field
    {"lat": 22.50, "lon": 53.90, "weight": 1.5, "spread": 1.8, "k": 10, "phase": -np.pi/3, "label": "Shaybah Manifold (Oil)"},
    # Khurais Field
    {"lat": 25.00, "lon": 48.00, "weight": 1.4, "spread": 1.0, "k": 15, "phase": np.pi/2, "label": "Khurais Node (Oil)"}
]

REGIONS = {
    "Massachusetts": {
        "bounds": (41.2, 42.9, -73.6, -69.8),
        "centers": MASSACHUSETTS_CENTERS,
        "n_points": 4000,
        "center_map": {"lat": 42.1, "lon": -71.7},
        "zoom": 7.3
    },
    "North America": {
        "bounds": (25.0, 60.0, -125.0, -65.0),
        "centers": NORTH_AMERICA_CENTERS,
        "n_points": 8000,
        "center_map": {"lat": 45.0, "lon": -95.0},
        "zoom": 3.0
    },
    "Canadian Oil-Ore Context": {
        "bounds": (44.0, 64.0, -125.0, -70.0),
        "centers": CANADA_OIL_ORE_CENTERS,
        "n_points": 6000,
        "center_map": {"lat": 55.0, "lon": -100.0},
        "zoom": 3.5
    },
    "Canadian NW Quantum Manifold": {
        "bounds": (59.0, 71.0, -142.0, -108.0),
        "centers": CANADIAN_NW_CENTERS,
        "n_points": 5000,
        "center_map": {"lat": 65.5, "lon": -125.0},
        "zoom": 3.8
    },
    "Saudi Arabia Oil Signatures": {
        "bounds": (15.0, 33.0, 34.0, 56.0),
        "centers": SAUDI_ARABIA_CENTERS,
        "n_points": 6000,
        "center_map": {"lat": 24.0, "lon": 47.0},
        "zoom": 4.5
    }
}

# ── Dynamic Data Generator ───────────────────────────────────────────────────

def generate_df(region, phase_shift):
    cfg = REGIONS[region]
    lats, lons, prob_dens, ent_score = simulate_quantum_wavefunction(
        cfg["n_points"], cfg["bounds"], cfg["centers"], phase_shift
    )
    
    # Map points to closest superposition states for labeling
    labels = []
    for i in range(len(lats)):
        dist = [((lats[i]-c["lat"])**2 + (lons[i]-c["lon"])**2) for c in cfg["centers"]]
        labels.append(cfg["centers"][np.argmin(dist)]["label"])
        
    df = pd.DataFrame({
        "Latitude": lats,
        "Longitude": lons,
        "Q_Probability": prob_dens,
        "Entanglement_Score": ent_score,
        "State_Label": labels,
        "Est_Grade": np.clip(prob_dens * (1.0 + ent_score) * 40, 0, 100) # Grade proxy
    })
    
    # Filter out near-zero probabilities to simulate "collapse" to likely prospects
    df = df[df["Q_Probability"] > df["Q_Probability"].quantile(0.65)]
    return df

# ── App Layout ────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"backgroundColor": "#050510", "minHeight": "100vh", "padding": "20px", "fontFamily": "Consolas, sans-serif"},
    children=[
        html.H1("Quantum Machine Learning Ore Prospecting", style={"color": "#aa88ff", "textAlign": "center", "marginBottom": "5px"}),
        html.P("Simulated QML Wavefunction Superposition and Entanglement Scoring", 
               style={"color": "#666699", "textAlign": "center", "marginBottom": "20px"}),
        
        dcc.Tabs(
            id="tabs", value="Massachusetts",
            colors={"border": "#111122", "primary": "#aa88ff", "background": "#0a0a1a"},
            children=[
                dcc.Tab(label="Massachusetts Resonance", value="Massachusetts", style={"backgroundColor":"#0a0a1a","color":"#888"}, selected_style={"backgroundColor":"#050510","color":"#aa88ff"}),
                dcc.Tab(label="North American Macro-States", value="North America", style={"backgroundColor":"#0a0a1a","color":"#888"}, selected_style={"backgroundColor":"#050510","color":"#00ffcc"}),
                dcc.Tab(label="Canadian Oil-Ore Context", value="Canadian Oil-Ore Context", style={"backgroundColor":"#0a0a1a","color":"#888"}, selected_style={"backgroundColor":"#050510","color":"#ffaa00"}),
                dcc.Tab(label="Canadian NW Quantum Manifold", value="Canadian NW Quantum Manifold", style={"backgroundColor":"#0a0a1a","color":"#888"}, selected_style={"backgroundColor":"#050510","color":"#ff44aa"}),
                dcc.Tab(label="Saudi Arabia Oil Signatures", value="Saudi Arabia Oil Signatures", style={"backgroundColor":"#0a0a1a","color":"#888"}, selected_style={"backgroundColor":"#050510","color":"#00ff00"}),
            ]
        ),
        
        html.Div([
            html.Div([
                html.Label("Ansatz Phase Angle Shift (radians θ)", style={"color": "#8888aa"}),
                dcc.Slider(id="phase-slider", min=0, max=2*np.pi, step=0.1, value=0.0, marks={0: "0", 3.14: "π", 6.28: "2π"}),
                html.Br(),
                html.Label("Min Wavefunction Probability P(x)", style={"color": "#8888aa"}),
                dcc.Slider(id="prob-slider", min=0, max=1.0, step=0.05, value=0.1, marks={i/10: f"{i/10}" for i in range(0, 11, 2)}),
            ], style={"padding": "20px", "backgroundColor": "#0d0d1a", "borderRadius": "10px", "margin": "15px 0", "border": "1px solid #1a1a33"}),
            
            dcc.Graph(id="qml-map", style={"height": "65vh", "borderRadius": "10px", "overflow": "hidden"}),
            
            html.Div(id="qml-hists", style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "15px", "marginTop": "20px"})
        ])
    ]
)

@app.callback(
    Output("qml-map", "figure"),
    Output("qml-hists", "children"),
    Input("tabs", "value"),
    Input("phase-slider", "value"),
    Input("prob-slider", "value")
)
def update_dashboard(region, phase, min_prob):
    df = generate_df(region, phase)
    
    # Scale relative to max prob for slider filtering
    max_p = df["Q_Probability"].max()
    df = df[df["Q_Probability"] >= min_prob * max_p]
    
    cfg = REGIONS[region]
    
    fig = px.scatter_map(
        df, lat="Latitude", lon="Longitude", color="State_Label",
        size="Q_Probability", size_max=16, opacity=0.85,
        hover_name="State_Label",
        hover_data={
            "Latitude": ":.4f", "Longitude": ":.4f",
            "Q_Probability": ":.4f",
            "Entanglement_Score": ":.4f",
            "Est_Grade": ":.2f"
        },
        zoom=cfg["zoom"], center=cfg["center_map"],
        map_style="carto-darkmatter",
        title=f"|Ψ|² Subsurface Distribution — {region}"
    )
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0}, paper_bgcolor="#050510", plot_bgcolor="#050510",
        font=dict(color="#ccccff"), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.6)")
    )
    
    # Histograms
    h1 = go.Figure(go.Histogram(x=df["Q_Probability"], marker_color="#aa88ff", nbinsx=40))
    h1.update_layout(title="Wavefunction Probability Amplitude |Ψ|²", paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a", font=dict(color="#bbb"), margin=dict(l=30,r=10,t=30,b=20), xaxis=dict(gridcolor="#223"), yaxis=dict(gridcolor="#223"))
    
    h2 = go.Figure(go.Histogram(x=df["Entanglement_Score"], marker_color="#00ffcc", nbinsx=40))
    h2.update_layout(title="Simulated Entanglement Entropy Score", paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a", font=dict(color="#bbb"), margin=dict(l=30,r=10,t=30,b=20), xaxis=dict(gridcolor="#223"), yaxis=dict(gridcolor="#223"))
    
    return fig, [
        dcc.Graph(figure=h1, style={"height": "300px", "borderRadius": "8px"}),
        dcc.Graph(figure=h2, style={"height": "300px", "borderRadius": "8px"})
    ]

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8054)