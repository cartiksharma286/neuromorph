import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

app = dash.Dash(__name__)

np.random.seed(7)

# ── Continued-fraction distribution engine ────────────────────────────────────
def cf_distribution(n_points, center_lat, center_lon, spread, depth=3):
    """
    Generates spatially-clustered points whose density is shaped by a
    depth-`depth` continued-fraction kernel:
        CF(r) = a0 + 1/(a1 + 1/(a2 + ... ))
    where each a_i = floor(1/(r + epsilon)).
    This creates natural fractal-like clustering that mimics subsurface
    geological fault / anticline patterns.
    """
    lats, lons, cf_vals = [], [], []
    while len(lats) < n_points:
        u = np.random.uniform(-1, 1)
        v = np.random.uniform(-1, 1)
        r = np.sqrt(u**2 + v**2)
        if r == 0:
            continue
        # Build continued-fraction value
        eps = 0.05
        cf = 0.0
        for _ in range(depth):
            cf = 1.0 / (abs(np.sin(np.pi / (r + eps))) + cf + eps)
        cf_factor = min(abs(cf), 2.0) / 2.0          # normalise to [0,1]
        lat = center_lat + v * spread * cf_factor
        lon = center_lon + u * spread * cf_factor
        lats.append(lat)
        lons.append(lon)
        cf_vals.append(round(cf_factor, 4))
    return lats, lons, cf_vals


# ── Region definitions ────────────────────────────────────────────────────────
REGIONS = {
    "Houston Energy Corridor": {
        # West Houston / Katy / Energy Corridor  (29.76 N, 95.57 W)
        "clusters": [
            dict(n=200, clat=29.762, clon=-95.570, spread=0.30, label="Energy Corridor Core"),
            dict(n=150, clat=29.700, clon=-95.680, spread=0.25, label="Katy Field"),
            dict(n=100, clat=29.820, clon=-95.450, spread=0.20, label="NW Houston Spur"),
        ],
        "center": {"lat": 29.76, "lon": -95.57},
        "zoom": 9.5,
        "color": "#ff6600",
    },
    "North Texas": {
        # Barnett Shale / Fort Worth Basin  (32.8 N, 97.3 W)
        "clusters": [
            dict(n=250, clat=32.800, clon=-97.300, spread=0.70, label="Fort Worth Basin"),
            dict(n=180, clat=33.100, clon=-96.800, spread=0.50, label="Barnett Shale N"),
            dict(n=120, clat=32.400, clon=-97.700, spread=0.45, label="Cleburne-Glen Rose"),
            dict(n=80,  clat=33.500, clon=-97.800, spread=0.35, label="Wise County"),
        ],
        "center": {"lat": 32.9, "lon": -97.2},
        "zoom": 8.0,
        "color": "#00ccff",
    },
    "Combined — Texas Overview": {
        "clusters": [],          # built dynamically below
        "center": {"lat": 31.5, "lon": -96.8},
        "zoom": 5.5,
        "color": None,
    },
}

# ── Generate data for each region ────────────────────────────────────────────
region_dfs = {}

for region_name, cfg in REGIONS.items():
    if region_name == "Combined — Texas Overview":
        continue
    frames = []
    for cl in cfg["clusters"]:
        lats, lons, cf_vals = cf_distribution(cl["n"], cl["clat"], cl["clon"], cl["spread"])
        frames.append(pd.DataFrame({
            "Latitude":         lats,
            "Longitude":        lons,
            "CF_Density":       cf_vals,
            "Cluster":          cl["label"],
            "Yield_Prob":       np.random.uniform(0.35, 0.95, len(lats)),
            "Est_Depth_m":      np.abs(np.random.normal(2200, 450, len(lats))),
            "GOR_scf_bbl":      np.random.uniform(200, 1800, len(lats)),
        }))
    df = pd.concat(frames, ignore_index=True)
    region_dfs[region_name] = df

# Combined
region_dfs["Combined — Texas Overview"] = pd.concat(
    [df.assign(Region=rn) for rn, df in region_dfs.items()],
    ignore_index=True
)

# ── Figure builders ───────────────────────────────────────────────────────────
PALETTE_HOUSTON = [
    "#ff6600", "#ff9933", "#ffcc00", "#ff3300", "#cc3300"
]
PALETTE_NTEXAS = [
    "#00ccff", "#0099cc", "#33ffcc", "#0066ff", "#6600ff"
]
PALETTE_COMBINED = [
    "#ff6600", "#00ccff",   # per-region tint
]


def build_figure(region_name, depth_filter=None, min_yield=0.0):
    df = region_dfs[region_name].copy()
    if depth_filter:
        df = df[df["Est_Depth_m"] <= depth_filter]
    df = df[df["Yield_Prob"] >= min_yield]

    cfg = REGIONS[region_name]

    if region_name == "Combined — Texas Overview":
        color_col = "Region"
        color_map = {
            "Houston Energy Corridor": "#ff6600",
            "North Texas":            "#00ccff",
        }
    else:
        color_col = "Cluster"
        palette = PALETTE_HOUSTON if "Houston" in region_name else PALETTE_NTEXAS
        clusters = df["Cluster"].unique()
        color_map = {c: palette[i % len(palette)] for i, c in enumerate(clusters)}

    fig = px.scatter_map(
        df,
        lat="Latitude",
        lon="Longitude",
        color=color_col,
        color_discrete_map=color_map,
        size="CF_Density",
        size_max=14,
        opacity=0.82,
        hover_name=color_col,
        hover_data={
            "Latitude":    ":.4f",
            "Longitude":   ":.4f",
            "CF_Density":  ":.3f",
            "Yield_Prob":  ":.2f",
            "Est_Depth_m": ":.0f",
            "GOR_scf_bbl": ":.0f",
        },
        zoom=cfg["zoom"],
        center=cfg["center"],
        map_style="carto-darkmatter",
        title=f"Oil Well Signatures — {region_name} (Continued Fraction Model)",
    )
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        paper_bgcolor="#0d0d0d",
        font=dict(color="#e0e0e0", size=13),
        title_font_size=20,
        legend=dict(
            yanchor="top", y=0.98, xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.55)",
            bordercolor="#444", borderwidth=1,
        ),
    )
    return fig


# ── Distribution histogram builder ───────────────────────────────────────────
def build_distributions(region_name):
    df = region_dfs[region_name]
    metrics = [
        ("CF_Density",  "Continued-Fraction Density",   "#ff9900"),
        ("Yield_Prob",  "Yield Probability",             "#00ffcc"),
        ("Est_Depth_m", "Estimated Depth (m)",           "#cc44ff"),
        ("GOR_scf_bbl", "Gas-Oil Ratio (scf/bbl)",       "#ff4488"),
    ]
    figs = []
    for col, label, color in metrics:
        h = go.Figure()
        h.add_trace(go.Histogram(
            x=df[col], nbinsx=40,
            marker_color=color, opacity=0.85,
            name=label,
        ))
        h.update_layout(
            title=label, paper_bgcolor="#0d0d0d", plot_bgcolor="#1a1a1a",
            font=dict(color="#e0e0e0", size=11),
            margin=dict(l=40, r=10, t=40, b=30),
            bargap=0.05,
            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
        )
        figs.append(h)
    return figs


# ── Layout ────────────────────────────────────────────────────────────────────
CARD = {
    "backgroundColor": "#1a1a1a",
    "borderRadius": "10px",
    "padding": "16px",
    "boxShadow": "0 4px 12px rgba(0,200,255,0.15)",
    "marginBottom": "16px",
}

def filter_panel(tab_id):
    return html.Div([
        html.Div([
            html.Label("Max Depth (m)", style={"color": "#aaa", "fontSize": "13px"}),
            dcc.Slider(
                id=f"depth-slider-{tab_id}",
                min=500, max=5000, step=250, value=5000,
                marks={v: str(v) for v in range(500, 5001, 500)},
                tooltip={"placement": "bottom"},
            ),
        ], style={"marginBottom": "10px"}),
        html.Div([
            html.Label("Min Yield Probability", style={"color": "#aaa", "fontSize": "13px"}),
            dcc.Slider(
                id=f"yield-slider-{tab_id}",
                min=0.0, max=0.9, step=0.05, value=0.0,
                marks={v: f"{v:.1f}" for v in np.arange(0, 1.0, 0.1)},
                tooltip={"placement": "bottom"},
            ),
        ]),
    ], style=CARD)


app.layout = html.Div(
    style={"backgroundColor": "#0d0d0d", "minHeight": "100vh",
           "padding": "24px", "fontFamily": "Arial, sans-serif"},
    children=[
        html.H1(
            "US Oil Well Signature Explorer — Continued Fraction Model",
            style={"textAlign": "center", "color": "#00ccff",
                   "marginBottom": "6px", "fontSize": "26px"},
        ),
        html.P(
            "Fractal-clustered prospecting signatures for Houston Energy Corridor & North Texas "
            "derived from depth-3 continued-fraction density kernels.",
            style={"textAlign": "center", "color": "#888", "fontSize": "14px",
                   "marginBottom": "20px"},
        ),

        dcc.Tabs(
            id="region-tabs",
            value="tab-houston",
            colors={"border": "#333", "primary": "#00ccff", "background": "#1a1a1a"},
            style={"marginBottom": "16px"},
            children=[
                # ── Tab 1: Houston ──────────────────────────────────────────
                dcc.Tab(
                    label="Houston Energy Corridor",
                    value="tab-houston",
                    style={"color": "#888", "backgroundColor": "#1a1a1a"},
                    selected_style={"color": "#ff6600", "backgroundColor": "#0d0d0d",
                                    "borderTop": "3px solid #ff6600"},
                    children=[
                        filter_panel("houston"),
                        dcc.Graph(
                            id="map-houston",
                            figure=build_figure("Houston Energy Corridor"),
                            style={"height": "65vh", "borderRadius": "10px",
                                   "overflow": "hidden"},
                        ),
                        html.H3("N-Distributions", style={"color": "#ff9900",
                                "marginTop": "20px", "marginBottom": "10px"}),
                        html.Div(id="dist-houston", style={
                            "display": "grid",
                            "gridTemplateColumns": "1fr 1fr",
                            "gap": "16px",
                        }),
                    ],
                ),

                # ── Tab 2: North Texas ──────────────────────────────────────
                dcc.Tab(
                    label="North Texas (Barnett / Fort Worth)",
                    value="tab-ntexas",
                    style={"color": "#888", "backgroundColor": "#1a1a1a"},
                    selected_style={"color": "#00ccff", "backgroundColor": "#0d0d0d",
                                    "borderTop": "3px solid #00ccff"},
                    children=[
                        filter_panel("ntexas"),
                        dcc.Graph(
                            id="map-ntexas",
                            figure=build_figure("North Texas"),
                            style={"height": "65vh", "borderRadius": "10px",
                                   "overflow": "hidden"},
                        ),
                        html.H3("N-Distributions", style={"color": "#00ccff",
                                "marginTop": "20px", "marginBottom": "10px"}),
                        html.Div(id="dist-ntexas", style={
                            "display": "grid",
                            "gridTemplateColumns": "1fr 1fr",
                            "gap": "16px",
                        }),
                    ],
                ),

                # ── Tab 3: Combined ─────────────────────────────────────────
                dcc.Tab(
                    label="Texas Overview (Combined)",
                    value="tab-combined",
                    style={"color": "#888", "backgroundColor": "#1a1a1a"},
                    selected_style={"color": "#cc44ff", "backgroundColor": "#0d0d0d",
                                    "borderTop": "3px solid #cc44ff"},
                    children=[
                        dcc.Graph(
                            id="map-combined",
                            figure=build_figure("Combined — Texas Overview"),
                            style={"height": "75vh", "borderRadius": "10px",
                                   "overflow": "hidden"},
                        ),
                        html.H3("N-Distributions (All Regions)", style={"color": "#cc44ff",
                                "marginTop": "20px", "marginBottom": "10px"}),
                        html.Div(id="dist-combined", style={
                            "display": "grid",
                            "gridTemplateColumns": "1fr 1fr",
                            "gap": "16px",
                        }),
                    ],
                ),
            ],
        ),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("map-houston", "figure"),
    Output("dist-houston", "children"),
    Input("depth-slider-houston", "value"),
    Input("yield-slider-houston", "value"),
)
def update_houston(max_depth, min_yield):
    fig = build_figure("Houston Energy Corridor", depth_filter=max_depth, min_yield=min_yield)
    hists = build_distributions("Houston Energy Corridor")
    graphs = [dcc.Graph(figure=h, style={"borderRadius": "8px"}) for h in hists]
    return fig, graphs


@app.callback(
    Output("map-ntexas", "figure"),
    Output("dist-ntexas", "children"),
    Input("depth-slider-ntexas", "value"),
    Input("yield-slider-ntexas", "value"),
)
def update_ntexas(max_depth, min_yield):
    fig = build_figure("North Texas", depth_filter=max_depth, min_yield=min_yield)
    hists = build_distributions("North Texas")
    graphs = [dcc.Graph(figure=h, style={"borderRadius": "8px"}) for h in hists]
    return fig, graphs


@app.callback(
    Output("dist-combined", "children"),
    Input("region-tabs", "value"),
)
def update_combined_dists(tab):
    if tab != "tab-combined":
        return []
    hists = build_distributions("Combined — Texas Overview")
    return [dcc.Graph(figure=h, style={"borderRadius": "8px"}) for h in hists]


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8051)
