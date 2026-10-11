"""
Lake Tahoe Undersea Robotics & Molecular Gene Combing Platform
Main Application Server (Flask)
Integrates:
- 6-DOF Autonomous Underwater Vehicle (AUV) Kinematics & Deep-Trench Bathymetry
- Microfluidic Molecular Gene Combing & eDNA Fluorescence Barcoding
- p-Adic Ultrametric Number-Theoretic Topological Placement
- Environmental Secchi Clarity & Particulate Microstate Diagnostics
- Nature Preprint Generation and PDF Serving
"""

import os
import sys
import json
import numpy as np
from flask import Flask, render_template, jsonify, request, send_file

# Add backend directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, 'backend'))

from tahoe_auv_robotics_engine import TahoeAUVRoboticsEngine
from gene_combing_particle_engine import GeneCombingParticleEngine
from padic_topological_engine import PadicTopologicalEngine
from nature_tahoe_preprint_generator import generate_nature_tahoe_preprint_pdf

app = Flask(
    __name__,
    template_folder=os.path.join(CURRENT_DIR, 'templates'),
    static_folder=os.path.join(CURRENT_DIR, 'static')
)

# Instantiate engines
auv_engine = TahoeAUVRoboticsEngine()
combing_engine = GeneCombingParticleEngine()
padic_engine = PadicTopologicalEngine()

PREPRINT_PDF_PATH = os.path.join(CURRENT_DIR, 'Nature_Lake_Tahoe_Undersea_Robotics_pAdic_Preprint.pdf')

# Ensure PDF exists at startup
if not os.path.exists(PREPRINT_PDF_PATH):
    try:
        generate_nature_tahoe_preprint_pdf(PREPRINT_PDF_PATH)
    except Exception as e:
        print(f"Warning: Could not pre-generate PDF: {e}")

@app.route('/')
def index():
    """Serves the main application interface."""
    return render_template('index.html')

# --- AUV Robotics & Bathymetry Routes ---

@app.route('/api/auv/stations', methods=['GET'])
def get_stations():
    """Returns list of Lake Tahoe benthic stations."""
    return jsonify({
        'status': 'success',
        'stations': TahoeAUVRoboticsEngine.BENTHIC_STATIONS
    })

@app.route('/api/auv/telemetry', methods=['POST', 'GET'])
def get_auv_telemetry():
    """Computes 6-DOF kinematics, drag, power draw, and acoustic telemetry."""
    if request.method == 'POST':
        data = request.json or {}
    else:
        data = request.args
        
    station_key = data.get('station', 'crystal_bay_trench')
    thruster_rpm = float(data.get('thruster_rpm', 1200.0))
    buoyancy_trim = float(data.get('buoyancy_trim_cc', 0.0))
    
    result = auv_engine.compute_auv_telemetry(
        station_key=station_key,
        thruster_rpm=thruster_rpm,
        buoyancy_trim_cc=buoyancy_trim
    )
    return jsonify({
        'status': 'success',
        'telemetry': result
    })

# --- Molecular Gene Combing & Particle States Routes ---

@app.route('/api/combing/species', methods=['GET'])
def get_species():
    """Returns candidate eDNA bioindicator species."""
    return jsonify({
        'status': 'success',
        'species': GeneCombingParticleEngine.SPECIES_BARCODES
    })

@app.route('/api/combing/simulate', methods=['POST', 'GET'])
def simulate_gene_combing():
    """Simulates DNA meniscus elongation and fluorescence barcode readout."""
    if request.method == 'POST':
        data = request.json or {}
    else:
        data = request.args
        
    species_key = data.get('species', 'cyclotella_gordonensis')
    shear_rate = float(data.get('shear_rate', 1200.0))
    
    result = combing_engine.simulate_gene_combing(
        species_key=species_key,
        shear_rate_s_minus_1=shear_rate
    )
    return jsonify({
        'status': 'success',
        'combing': result
    })

# --- p-Adic Topological Placement Routes ---

@app.route('/api/padic/network', methods=['POST', 'GET'])
def get_padic_network():
    """Computes p-adic ultrametric valuations, simplicial complexes, and Betti numbers."""
    if request.method == 'POST':
        data = request.json or {}
    else:
        data = request.args
        
    prime_p = int(data.get('prime', 5))
    threshold = float(data.get('threshold', 0.65))
    
    result = padic_engine.compute_topological_network(
        chosen_prime=prime_p,
        distance_threshold=threshold
    )
    return jsonify({
        'status': 'success',
        'network': result
    })

# --- Preprint Routes ---

@app.route('/api/preprint/generate', methods=['POST', 'GET'])
def regenerate_preprint():
    """Recompiles the Nature Preprint PDF."""
    try:
        path = generate_nature_tahoe_preprint_pdf(PREPRINT_PDF_PATH)
        return jsonify({
            'status': 'success',
            'message': 'Nature preprint PDF generated successfully',
            'pdf_path': path,
            'size_bytes': os.path.getsize(path)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/preprint/download', methods=['GET'])
def download_preprint():
    """Serves the Nature Preprint PDF as a downloadable attachment."""
    if not os.path.exists(PREPRINT_PDF_PATH):
        generate_nature_tahoe_preprint_pdf(PREPRINT_PDF_PATH)
    return send_file(
        PREPRINT_PDF_PATH,
        as_attachment=True,
        download_name='Nature_Lake_Tahoe_Undersea_Robotics_pAdic_Preprint.pdf',
        mimetype='application/pdf'
    )

@app.route('/api/preprint/view', methods=['GET'])
def view_preprint():
    """Serves the Nature Preprint PDF inline."""
    if not os.path.exists(PREPRINT_PDF_PATH):
        generate_nature_tahoe_preprint_pdf(PREPRINT_PDF_PATH)
    return send_file(
        PREPRINT_PDF_PATH,
        as_attachment=False,
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    port = int(os.environ.get('FLASK_RUN_PORT', 5090))
    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    print(f"\n{'='*70}")
    print(f"🌊 Lake Tahoe Undersea Robotics & p-Adic Topological Placement Platform")
    print(f"{'='*70}")
    print(f"Starting Server on http://{host}:{port}")
    print(f"Serving Nature Preprint at: http://{host}:{port}/api/preprint/view")
    print(f"{'='*70}\n")
    app.run(host=host, port=port, debug=False, threaded=True)
