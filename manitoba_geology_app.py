import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scipy.stats as stats

# Initialize the Dash app
app = dash.Dash(__name__)

# --- 1. Statistical Data Generation (Northern Manitoba bounds) ---
# Northern Manitoba Rough Bounds: Lat (54.0 to 60.0), Lon (-102.0 to -90.0)
np.random.seed(42)

def generate_continued_fraction_distribution(n_points, center_lat, center_lon, spread):
    """
    Generates a distribution where the density is modulated by a simple continued fraction-like
    function to simulate realistic, clustered geological formations.
    """
    lats = []
    lons = []
    for _ in range(n_points):
        # A simple mathematical variation inspired by continued fraction chaos patterns
        u = np.random.uniform(-1, 1)
        v = np.random.uniform(-1, 1)
        r = np.sqrt(u**2 + v**2)
        if r > 0:
            # Applying a non-linear transformation
            cf_factor = abs(np.sin(1/(r + 0.1))) 
            lat = center_lat + (v * spread * cf_factor)
            lon = center_lon + (u * spread * cf_factor)
            lats.append(lat)
            lons.append(lon)
    return lats, lons

# Generate Oil Well Sites
oil_lats, oil_lons = generate_continued_fraction_distribution(150, 56.5, -98.0, 2.5)
oil_df = pd.DataFrame({
    'Latitude': oil_lats,
    'Longitude': oil_lons,
    'Type': 'Oil Well Signature',
    'Yield_Probability': np.random.uniform(0.4, 0.95, 150),
    'Depth_m': np.random.normal(1500, 300, 150)
})

# Generate Mineral Ore Sites
mineral_lats, mineral_lons = generate_continued_fraction_distribution(200, 58.0, -96.5, 2.0)
mineral_df = pd.DataFrame({
    'Latitude': mineral_lats,
    'Longitude': mineral_lons,
    'Type': 'Mineral Ore Site',
    'Purity_Index': np.random.uniform(0.6, 0.99, 200),
    'Depth_m': np.random.normal(500, 150, 200)
})

# Combine into a single DataFrame
df = pd.concat([oil_df, mineral_df], ignore_index=True)

# Generate a vibrant map using Plotly Express
fig = px.scatter_map(
    df,
    lat="Latitude",
    lon="Longitude",
    color="Type",
    size="Depth_m",
    color_discrete_map={
        "Oil Well Signature": "#ff9900",  # Vibrant Orange
        "Mineral Ore Site": "#00ffcc"     # Electric Cyan
    },
    hover_name="Type",
    hover_data={
        "Latitude": ":.4f",
        "Longitude": ":.4f",
        "Depth_m": ":.0f",
        "Yield_Probability": ":.2f",
        "Purity_Index": ":.2f"
    },
    zoom=4.5,
    center={"lat": 57.0, "lon": -97.0},
    map_style="carto-darkmatter",
    title="Northern Manitoba Geological Signatures (Continued Fraction Model)"
)

# Enhance map styling
fig.update_layout(
    margin={"r":0,"t":50,"l":0,"b":0},
    paper_bgcolor="#111111",
    plot_bgcolor="#111111",
    font=dict(color="#ffffff"),
    legend_title_text="Site Classification",
    title_font_size=24,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(0,0,0,0.5)"
    )
)

# --- 2. Build the Dashboard Layout ---
app.layout = html.Div(style={'backgroundColor': '#111111', 'minHeight': '100vh', 'padding': '20px', 'color': 'white', 'fontFamily': 'Arial'}, children=[
    html.H1("Northern Manitoba Exploratory Data Visualizer", style={'textAlign': 'center', 'color': '#00ffcc'}),
    html.P("Mapping Oil Well Signatures and Mineral Ore Sites derived from Statistical Continued Fraction distributions.", style={'textAlign': 'center', 'fontSize': '18px'}),
    
    html.Div([
        dcc.Graph(
            id='geological-map',
            figure=fig,
            style={'height': '80vh', 'width': '100%', 'borderRadius': '10px', 'overflow': 'hidden', 'boxShadow': '0 4px 8px 0 rgba(0, 255, 204, 0.2)'}
        )
    ], style={'padding': '20px'})
])

if __name__ == '__main__':
    # Launch on a specific port to avoid conflicts
    app.run(debug=True, host='0.0.0.0', port=8050)
