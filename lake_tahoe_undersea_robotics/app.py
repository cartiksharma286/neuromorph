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
from ontario_qubit_spectroscopy_engine import OntarioQubitSpectroscopyEngine
from nature_ontario_preprint_generator import generate_nature_ontario_preprint_pdf
from backend.carbon_routes import carbon

app = Flask(
    __name__,
    template_folder=os.path.join(CURRENT_DIR, 'templates'),
    static_folder=os.path.join(CURRENT_DIR, 'static')
)
app.register_blueprint(carbon)

# Instantiate engines
auv_engine = TahoeAUVRoboticsEngine()
combing_engine = GeneCombingParticleEngine()
padic_engine = PadicTopologicalEngine()
ontario_engine = OntarioQubitSpectroscopyEngine()

PREPRINT_PDF_PATH = os.path.join(CURRENT_DIR, 'Nature_Lake_Tahoe_Undersea_Robotics_pAdic_Preprint.pdf')
ONTARIO_PREPRINT_PDF_PATH = os.path.join(CURRENT_DIR, 'Nature_Lake_Ontario_Qubit_Robotics_Spectroscopy_Preprint.pdf')

# Ensure PDFs exist at startup
if not os.path.exists(PREPRINT_PDF_PATH):
    try:
        generate_nature_tahoe_preprint_pdf(PREPRINT_PDF_PATH)
    except Exception as e:
        print(f"Warning: Could not pre-generate Tahoe PDF: {e}")

if not os.path.exists(ONTARIO_PREPRINT_PDF_PATH):
    try:
        generate_nature_ontario_preprint_pdf(ONTARIO_PREPRINT_PDF_PATH)
    except Exception as e:
        print(f"Warning: Could not pre-generate Ontario PDF: {e}")

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

# --- Lake Ontario Qubit Circuitry, Spectroscopy & 10-Year Recovery Routes ---

@app.route('/api/ontario/stations', methods=['GET'])
def get_ontario_stations():
    """Returns Lake Ontario benthic stations."""
    return jsonify({
        'status': 'success',
        'stations': ontario_engine.LAKE_ONTARIO_STATIONS
    })

@app.route('/api/ontario/qubit/specs', methods=['GET'])
def get_qubit_specs():
    """Returns superconducting transmon qubit hardware specifications."""
    return jsonify({
        'status': 'success',
        'specs': ontario_engine.QUBIT_HARDWARE_SPEC
    })

@app.route('/api/ontario/kicad/download', methods=['GET'])
def download_kicad_schematic():
    """Generates and serves the complete KiCad S-Expression schematic file (.kicad_sch)."""
    kicad_content = ontario_engine.generate_kicad_schematic_sexpr()
    import io
    buf = io.BytesIO(kicad_content.encode('utf-8'))
    return send_file(
        buf,
        as_attachment=True,
        download_name='AUV_Superconducting_Qubit_Spectrometer.kicad_sch',
        mimetype='application/x-kicad-schematic'
    )

@app.route('/api/ontario/kicad/raw', methods=['GET'])
def get_kicad_raw():
    """Returns raw KiCad S-Expression content for in-browser inspection."""
    return jsonify({
        'status': 'success',
        'kicad_sch': ontario_engine.generate_kicad_schematic_sexpr()
    })

@app.route('/api/ontario/spectroscopy/simulate', methods=['POST', 'GET'])
def simulate_ontario_spectroscopy():
    """Simulates AUV robotic movement, water intake, and quantum-enhanced spectroscopy."""
    if request.method == 'POST':
        data = request.json or {}
    else:
        data = request.args
        
    station_key = data.get('station', 'toronto_inner_harbour')
    depth = float(data.get('depth_m', 12.0))
    intake_rate = float(data.get('intake_rate', 150.0))
    
    result = ontario_engine.simulate_qubit_spectroscopy(
        station_key=station_key,
        auv_depth_m=depth,
        intake_rate_ml_min=intake_rate
    )
    return jsonify({
        'status': 'success',
        'data': result
    })

@app.route('/api/ontario/recovery/10yr', methods=['GET'])
def get_ontario_10yr_recovery():
    """Returns 10-year Lake Ontario ecological recovery trajectory and roadmap."""
    result = ontario_engine.compute_lake_ontario_10yr_recovery()
    return jsonify({
        'status': 'success',
        'recovery': result
    })

@app.route('/api/ontario/preprint/generate', methods=['POST', 'GET'])
def regenerate_ontario_preprint():
    """Recompiles the Nature Lake Ontario Qubit Preprint PDF."""
    try:
        path = generate_nature_ontario_preprint_pdf(ONTARIO_PREPRINT_PDF_PATH)
        return jsonify({
            'status': 'success',
            'message': 'Nature Lake Ontario preprint PDF generated successfully',
            'pdf_path': path,
            'size_bytes': os.path.getsize(path)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ontario/preprint/view', methods=['GET'])
def view_ontario_preprint():
    """Serves the Nature Lake Ontario Qubit Preprint PDF inline."""
    if not os.path.exists(ONTARIO_PREPRINT_PDF_PATH):
        generate_nature_ontario_preprint_pdf(ONTARIO_PREPRINT_PDF_PATH)
    return send_file(
        ONTARIO_PREPRINT_PDF_PATH,
        as_attachment=False,
        mimetype='application/pdf'
    )

@app.route('/api/ontario/preprint/download', methods=['GET'])
def download_ontario_preprint():
    """Serves the Nature Lake Ontario Qubit Preprint PDF as a download."""
    if not os.path.exists(ONTARIO_PREPRINT_PDF_PATH):
        generate_nature_ontario_preprint_pdf(ONTARIO_PREPRINT_PDF_PATH)
    return send_file(
        ONTARIO_PREPRINT_PDF_PATH,
        as_attachment=True,
        download_name='Nature_Lake_Ontario_Qubit_Robotics_Spectroscopy_Preprint.pdf',
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    port = int(os.environ.get('FLASK_RUN_PORT', 5070))
    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    print(f"\n{'='*70}")
    print(f"🌊 Lake Tahoe & Lake Ontario Undersea Robotics Platform")
    print(f"⚡ Superconducting Qubit Circuitry • KiCad Schematics • 10-Yr Recovery")
    print(f"{'='*70}")
    print(f"Starting Server on http://{host}:{port}")
    print(f"Serving Nature Preprint at: http://{host}:{port}/api/ontario/preprint/view")
    print(f"{'='*70}\n")
    app.run(host=host, port=port, debug=False, threaded=True)
