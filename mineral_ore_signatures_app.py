import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scipy.stats as stats

app = dash.Dash(__name__)

np.random.seed(42)

# ── Statistical Measure Theory & Continued Fraction Engine ────────────────────
def generate_measure_theoretic_cf_distribution(n_points, center_lat, center_lon, spread, depth=3):
    """
    Generates spatial distributions mapping Mineral Ore Signatures.
    Combines:
    1. Continued Fractions: Models the fractal fracturing of geological faults.
    2. Statistical Measure Theory: Uses a Radon-Nikodym derivative approach 
       (weighting factor with exponential decay and harmonic oscillation) 
       to define the density measure dν/dμ over the standard Lebesgue measure.
    """
    lats, lons, cf_densities, rn_derivatives = [], [], [], []
    
    while len(lats) < n_points:
        # Standard Lebesgue background sampling space (uniform disk mapping)
        u = np.random.uniform(-1, 1)
        v = np.random.uniform(-1, 1)
        r = np.sqrt(u**2 + v**2)
        if r == 0 or r > 1:
            continue
            
        # 1. Continued Fraction Factor (Fractal Fault lines)
        eps = 1e-4
        cf = 0.0
        for _ in range(depth):
            # Applying nested inversion logic for chaotic spatial fracturing
            cf = 1.0 / (abs(np.sin(np.pi / (r + eps))) + cf + eps)
        
        # 2. Measure Theory Factor (Radon-Nikodym derivative dν/dμ)
        # Represents concentration probability decaying from the anomaly epicenter
        # multiplied by a periodic geological bedding variation.
        rn_deriv = np.exp(-(r**2) / 0.15) * (1.0 + 0.6 * np.cos(8 * np.pi * r))
        
        # Combine measures
        final_weight = (cf * 0.4) + (rn_deriv * 0.6)
        
        # Rejection sampling based on the new measure theoretic density
        if np.random.uniform(0, max(1.5, final_weight)) > final_weight:
            continue

        lat = center_lat + v * spread
        lon = center_lon + u * spread
        lats.append(lat)
        lons.append(lon)
        cf_densities.append(min(cf, 5.0))
        rn_derivatives.append(rn_deriv)
        
    return lats, lons, cf_densities, rn_derivatives


# ── Region definitions (Canada & US) ──────────────────────────────────────────
REGIONS = {
    "Canada": {
        "clusters": [
            dict(n=300, clat=46.500, clon=-81.000, spread=1.2, label="Sudbury Basin (Ni/Cu/PGE)"),
            dict(n=250, clat=58.200, clon=-104.00, spread=1.5, label="Athabasca Basin (Uranium)"),
            dict(n=200, clat=48.200, clon=-79.000, spread=1.0, label="Abitibi Gold Belt (Au)"),
        ],
        "center": {"lat": 52.0, "lon": -90.0},
        "zoom": 4.0,
    },
    "United States": {
        "clusters": [
            dict(n=350, clat=40.800, clon=-116.30, spread=1.1, label="Carlin Trend, NV (Gold)"),
            dict(n=200, clat=47.500, clon=-91.500, spread=1.3, label="Duluth Complex, MN (Cu/Ni)"),
            dict(n=150, clat=39.000, clon=-110.00, spread=1.6, label="Colorado Plateau (U/V)"),
        ],
        "center": {"lat": 42.0, "lon": -101.0},
        "zoom": 4.0,
    },
    "North America": {
        "clusters": [], # Built dynamically
        "center": {"lat": 48.0, "lon": -95.0},
        "zoom": 3.0,
    },
}

region_dfs = {}

for region_name, cfg in REGIONS.items():
    if region_name == "North America": continue
    frames = []
    for cl in cfg["clusters"]:
        l, ln, cf_d, rn_d = generate_measure_theoretic_cf_distribution(
            cl["n"], cl["clat"], cl["clon"], cl["spread"]
        )
        density_measure = np.array(cf_d) * np.array(rn_d)
        # Normalize thickness/grade for visuals
        grade_pct = np.clip((np.array(rn_d) * np.random.uniform(0.5, 2.0, len(l))), 0, 100)
        
        frames.append(pd.DataFrame({
            "Latitude": l,
            "Longitude": ln,
            "Deposit_Cluster": cl["label"],
            "CF_Fractal_Density": cf_d,
            "Radon_Nikodym_Derivative": rn_d,
            "Combined_Density_Measure": density_measure,
            "Est_Grade_Index": grade_pct,
            "Depth_m": np.random.lognormal(mean=6.0, sigma=0.8, size=len(l)) # depth distribution
        }))
    region_dfs[region_name] = pd.concat(frames, ignore_index=True)

# Combined Overivew
region_dfs["North America"] = pd.concat(
    [df.assign(Territory=rn) for rn, df in region_dfs.items()],
    ignore_index=True
)

COLORS = {
    "Sudbury Basin (Ni/Cu/PGE)": "#ff00ff",   # Magenta
    "Athabasca Basin (Uranium)": "#00ff00",   # Lime
    "Abitibi Gold Belt (Au)": "#ffd700",      # Gold
    "Carlin Trend, NV (Gold)": "#ffaa00",     # Orange-Gold
    "Duluth Complex, MN (Cu/Ni)": "#00aaff",  # Blue
    "Colorado Plateau (U/V)": "#ff3333",      # Red
}

def build_map(region_name, min_density_measure, max_depth):
    df = region_dfs[region_name]
    df = df[(df["Combined_Density_Measure"] >= min_density_measure) & (df["Depth_m"] <= max_depth)]
    cfg = REGIONS[region_name]
    
    color_col = "Territory" if region_name == "North America" else "Deposit_Cluster"
    c_map = {"Canada": "#ff0000", "United States": "#0000ff"} if region_name == "North America" else COLORS

    fig = px.scatter_map(
        df, lat="Latitude", lon="Longitude", color=color_col,
        color_discrete_map=c_map,
        size="Est_Grade_Index", size_max=15, opacity=0.8,
        hover_name="Deposit_Cluster",
        hover_data={
            "Latitude": ":.3f", "Longitude": ":.3f",
            "CF_Fractal_Density": ":.3f",
            "Radon_Nikodym_Derivative": ":.3f",
            "Combined_Density_Measure": ":.4f",
            "Est_Grade_Index": ":.2f", "Depth_m": ":.0f"
        },
        zoom=cfg["zoom"], center=cfg["center"],
        map_style="carto-darkmatter",
        title=f"Mineral Ore Anomalies — {region_name}"
    )
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0}, paper_bgcolor="#080808", plot_bgcolor="#080808",
        font=dict(color="#bbb"), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.6)")
    )
    return fig

def build_hists(region_name):
    df = region_dfs[region_name]
    metrics = [
        ("Combined_Density_Measure", "Density Measure dν/dμ", "#00ffcc"),
        ("Est_Grade_Index", "Estimated Grade Index", "#ffd700"),
        ("Depth_m", "Depth Profile (m)", "#ff44aa"),
        ("Radon_Nikodym_Derivative", "Radon-Nikodym Derivative", "#44aaff")
    ]
    figs = []
    for col, label, color in metrics:
        h = go.Figure(go.Histogram(x=df[col], marker_color=color, nbinsx=50, opacity=0.8))
        h.update_layout(
            title=label, paper_bgcolor="#111", plot_bgcolor="#111",
            font=dict(color="#ccc", size=10), margin=dict(l=30, r=10, t=30, b=20),
            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333")
        )
        figs.append(dcc.Graph(figure=h, style={"height": "250px", "borderRadius": "8px", "overflow": "hidden"}))
    return figs


# ── App Layout ────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"backgroundColor": "#080808", "minHeight": "100vh", "padding": "20px", "fontFamily": "Segoe UI, sans-serif"},
    children=[
        html.H1("North American Mineral Ore Exploration", style={"color": "#00ffcc", "textAlign": "center", "marginBottom": "5px"}),
        html.P("Statistical Measure Theory (Radon-Nikodym) & Continued Fraction Spatial Modeling", 
               style={"color": "#888", "textAlign": "center", "marginBottom": "20px"}),
        
        dcc.Tabs(
            id="tabs", value="Canada",
            colors={"border": "#222", "primary": "#00ffcc", "background": "#111"},
            children=[
                dcc.Tab(label="Canadian Shield & Basins", value="Canada", style={"backgroundColor":"#111","color":"#888"}, selected_style={"backgroundColor":"#080808","color":"#ff0033"}),
                dcc.Tab(label="United States Ore Trends", value="United States", style={"backgroundColor":"#111","color":"#888"}, selected_style={"backgroundColor":"#080808","color":"#0077ff"}),
                dcc.Tab(label="North America (Combined)", value="North America", style={"backgroundColor":"#111","color":"#888"}, selected_style={"backgroundColor":"#080808","color":"#00ffcc"}),
            ]
        ),
        
        html.Div([
            html.Div([
                html.Label("Min Density Measure (dν/dμ)", style={"color": "#aaa"}),
                dcc.Slider(id="density-slider", min=0, max=1.5, step=0.1, value=0.0, marks={i/10: str(i/10) for i in range(0, 16, 3)}),
                html.Br(),
                html.Label("Max Depth (m)", style={"color": "#aaa"}),
                dcc.Slider(id="depth-slider", min=0, max=5000, step=250, value=5000, marks={i: str(i) for i in range(0, 5001, 1000)}),
            ], style={"padding": "20px", "backgroundColor": "#111", "borderRadius": "10px", "margin": "15px 0"}),
            
            dcc.Graph(id="main-map", style={"height": "65vh", "borderRadius": "10px", "overflow": "hidden"}),
            
            html.H3("Statistical Distributions", style={"color": "#fff", "marginTop": "20px"}),
            html.Div(id="histograms", style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "15px"})
        ])
    ]
)

@app.callback(
    Output("main-map", "figure"),
    Output("histograms", "children"),
    Input("tabs", "value"),
    Input("density-slider", "value"),
    Input("depth-slider", "value")
)
def update_ui(tab, min_density, max_depth):
    return build_map(tab, min_density, max_depth), build_hists(tab)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8052)