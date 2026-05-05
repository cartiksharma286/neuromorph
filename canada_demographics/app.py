import os
from flask import Flask, render_template, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

app = Flask(__name__)

# Base Statistics (Approximate 2026)
BASE_POPULATION = 40_000_000
TARGET_MULTIPLIER = 1.30
MINERAL_REGIONS = [
    {"name": "Ontario (Shield)", "base_pop": 15000000, "mineral_index": 0.8},
    {"name": "Quebec (Nord-du-Québec)", "base_pop": 9000000, "mineral_index": 0.75},
    {"name": "British Columbia", "base_pop": 5500000, "mineral_index": 0.85},
    {"name": "Alberta & Prairies", "base_pop": 7500000, "mineral_index": 0.6},
    {"name": "Yukon/NWT/Nunavut", "base_pop": 150000, "mineral_index": 1.0}, # High mineral potential
    {"name": "Atlantic Canada", "base_pop": 2850000, "mineral_index": 0.4}
]

def generate_quantum_surface():
    """
    Generates a 3D surface plot representing the quantum phase-space
    of population density attracted to mineral ore potentials.
    """
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    
    # Simulate a potential well structure based on mineral deposits
    # Psi^2 represents the probability density of population settlement
    R = np.sqrt(X**2 + Y**2)
    Z_mineral_potential = np.sin(R) * np.exp(-0.1 * R**2) # Interference pattern
    Z_population_density = Z_mineral_potential**2 * min(TARGET_MULTIPLIER, 1.5)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_population_density, cmap='viridis', edgecolor='none', alpha=0.9)
    
    ax.set_title("Quantum Surface Integral: Demographic Concentration over Mineral Veins")
    ax.set_xlabel("East-West Geomagnetic Coordinate")
    ax.set_ylabel("North-South Geomagnetic Coordinate")
    ax.set_zlabel("Probability Amplitude $|\Psi|^2$")
    plt.tight_layout()
    
    os.makedirs('static', exist_ok=True)
    filepath = 'static/quantum_demographics.png'
    plt.savefig(filepath, dpi=150, format='png', bbox_inches='tight')
    plt.close()
    return filepath

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate')
def simulate():
    generate_quantum_surface()
    
    # Perform mathematical scaling using the 'Quantum Surface Integral' principle
    # Conceptually: we integrate over the probability surface, which gives a total 1.0 probability
    # and scalar multiply by TARGET_MULTIPLIER
    total_pop_new = BASE_POPULATION * TARGET_MULTIPLIER
    
    # Calculate new populations based on mineral indexing (higher minerals = higher growth absorption)
    total_mineral_weight = sum([r['mineral_index'] for r in MINERAL_REGIONS])
    growth_pool = total_pop_new - BASE_POPULATION
    
    results = []
    
    for r in MINERAL_REGIONS:
        # Standard growth + mineral-weighted absorption
        base_growth = r['base_pop'] * 1.05 # Baseline 5% growth
        extra_growth = growth_pool * (r['mineral_index'] / total_mineral_weight)
        new_pop = int(base_growth + extra_growth)
        
        # Estimate new GDP conditionally on high tech + mining expansion
        # Base GDP per capita around $55,000; minerals amplify it up to $85,000
        gdp_per_capita = 55000 + (r['mineral_index'] * 35000)
        
        results.append({
            "region": r['name'],
            "base_pop": r['base_pop'],
            "new_pop": new_pop,
            "growth_pct": round(((new_pop - r['base_pop']) / r['base_pop']) * 100, 1),
            "est_gdp_pc": gdp_per_capita,
            "mineral_index": r['mineral_index']
        })
        
    return jsonify({
        "status": "success",
        "total_base_pop": BASE_POPULATION,
        "total_new_pop": int(total_pop_new),
        "target_multiplier": TARGET_MULTIPLIER,
        "regions": results,
        "surface_img": "/static/quantum_demographics.png"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5063, debug=False, threaded=True)
