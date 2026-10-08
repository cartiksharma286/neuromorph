"""
Tourette Syndrome Neurogenomics & QML Drug Discovery Platform
Main Application Server (Flask)
Integrates:
- Quantum Machine Learning Variational Quantum Eigensolvers (VQE)
- Number Theoretic Galois field lattices & Ramanujan character sums
- Jacobi Elliptic Integrals for non-linear proteomic folding
- Molecular Conjugates & Formulas Library with 2D vector structural graphics
- 2-Year Longitudinal Clinical Trajectories & 4-Phase Treatment Plans
- Health Economic Outcomes (5-state Markov, ICER Dominance, CEAC, PSA)
- Nature Preprint Compilation and PDF serving
"""

import os
import sys
import json
import numpy as np
from flask import Flask, render_template, jsonify, request, send_file

# Add backend directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, 'backend'))

from qml_number_theoretic_engine import (
    QuantumProteomicEngine,
    jacobi_elliptic_functions,
    complete_elliptic_integral_k,
    complete_elliptic_integral_e,
    incomplete_elliptic_integral_f,
    ramanujan_sum
)
from molecular_conjugates import COMPOUND_LIBRARY, generate_svg_structure
from clinical_and_economics_engine import ClinicalAndEconomicsEngine
from nature_preprint_generator import generate_nature_preprint_pdf

app = Flask(
    __name__,
    template_folder=os.path.join(CURRENT_DIR, 'templates'),
    static_folder=os.path.join(CURRENT_DIR, 'static')
)

# Instantiate engines
qml_engine = QuantumProteomicEngine(num_qubits=6)
clinical_econ_engine = ClinicalAndEconomicsEngine()

PREPRINT_PDF_PATH = os.path.join(CURRENT_DIR, 'Nature_Tourette_QML_Drug_Discovery_Preprint.pdf')

# Ensure PDF exists at startup
if not os.path.exists(PREPRINT_PDF_PATH):
    try:
        generate_nature_preprint_pdf(PREPRINT_PDF_PATH)
    except Exception as e:
        print(f"Warning: Could not pre-generate PDF: {e}")

@app.route('/')
def index():
    """Serves the main application interface."""
    return render_template('index.html')

@app.route('/api/targets', methods=['GET'])
def get_targets():
    """Returns list of genomic targets for Tourette Syndrome."""
    return jsonify({
        'status': 'success',
        'targets': QuantumProteomicEngine.GENOMIC_TARGETS
    })

@app.route('/api/compounds', methods=['GET'])
def get_compounds():
    """Returns all candidate molecular conjugates with SVG diagrams."""
    compounds = []
    for cid, data in COMPOUND_LIBRARY.items():
        comp_copy = dict(data)
        comp_copy['svg'] = generate_svg_structure(data)
        compounds.append(comp_copy)
    return jsonify({
        'status': 'success',
        'compounds': compounds
    })

@app.route('/api/compound/<compound_id>', methods=['GET'])
def get_compound(compound_id):
    """Returns a single compound's full dossier."""
    comp = COMPOUND_LIBRARY.get(compound_id)
    if not comp:
        return jsonify({'status': 'error', 'message': f'Compound {compound_id} not found'}), 404
    comp_copy = dict(comp)
    comp_copy['svg'] = generate_svg_structure(comp)
    return jsonify({
        'status': 'success',
        'compound': comp_copy
    })

@app.route('/api/qml/optimize', methods=['POST', 'GET'])
def run_qml_optimization():
    """
    Executes the Variational Quantum Eigensolver (VQE) with number-theoretic
    and Jacobi elliptic potential integration.
    """
    if request.method == 'POST':
        data = request.json or {}
    else:
        data = request.args
        
    target_key = data.get('target', 'DRD2')
    num_params = int(data.get('num_params', 6))
    
    # Custom or pseudo-random variational parameters theta
    seed = int(data.get('seed', 42))
    np.random.seed(seed)
    theta = np.random.uniform(0, 2 * 3.14159, num_params)
    
    result = qml_engine.simulate_vqe_ansatz(theta, target_key=target_key)
    result['theta_parameters'] = [round(float(t), 4) for t in theta]
    
    return jsonify({
        'status': 'success',
        'vqe_result': result
    })

@app.route('/api/folding/simulate', methods=['POST', 'GET'])
def simulate_folding():
    """
    Simulates 3D proteomic backbone folding trajectory using Jacobi elliptic solitons.
    """
    if request.method == 'POST':
        data = request.json or {}
    else:
        data = request.args
        
    compound_id = data.get('compound_id', 'NM-TS-701')
    target_key = data.get('target', 'DRD2')
    residues = int(data.get('residues', 100))
    
    folding_data = qml_engine.compute_elliptic_proteomic_folding(
        compound_name=compound_id,
        target_key=target_key,
        num_residues=residues
    )
    
    return jsonify({
        'status': 'success',
        'folding': folding_data
    })

@app.route('/api/clinical/trajectories', methods=['GET'])
def get_clinical_trajectories():
    """Returns 2-year longitudinal YGTSS trajectories and recovery treatment plan."""
    compound_id = request.args.get('compound_id', 'NM-TS-701')
    data = clinical_econ_engine.generate_ygtss_trajectories(compound_id=compound_id)
    return jsonify({
        'status': 'success',
        'data': data
    })

@app.route('/api/economics/markov', methods=['GET'])
def get_health_economics():
    """Returns 2-year 5-state Markov decision model and health economic outcomes."""
    data = clinical_econ_engine.run_health_economics_markov_model()
    return jsonify({
        'status': 'success',
        'economics': data
    })

@app.route('/api/preprint/generate', methods=['POST', 'GET'])
def regenerate_preprint():
    """Recompiles the Nature Preprint PDF."""
    try:
        path = generate_nature_preprint_pdf(PREPRINT_PDF_PATH)
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
    """Serves the Nature Preprint PDF as a downloadable file."""
    if not os.path.exists(PREPRINT_PDF_PATH):
        generate_nature_preprint_pdf(PREPRINT_PDF_PATH)
    return send_file(
        PREPRINT_PDF_PATH,
        as_attachment=True,
        download_name='Nature_Tourette_QML_Drug_Discovery_Preprint.pdf',
        mimetype='application/pdf'
    )

@app.route('/api/preprint/view', methods=['GET'])
def view_preprint():
    """Serves the Nature Preprint PDF inline for browser viewing."""
    if not os.path.exists(PREPRINT_PDF_PATH):
        generate_nature_preprint_pdf(PREPRINT_PDF_PATH)
    return send_file(
        PREPRINT_PDF_PATH,
        as_attachment=False,
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    port = int(os.environ.get('FLASK_RUN_PORT', 5050))
    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    print(f"\n{'='*70}")
    print(f"🧬 Tourette Syndrome QML & Number-Theoretic Drug Discovery Platform")
    print(f"{'='*70}")
    print(f"Starting Server on http://{host}:{port}")
    print(f"Serving Nature Preprint at: http://{host}:{port}/api/preprint/view")
    print(f"{'='*70}\n")
    app.run(host=host, port=port, debug=False, threaded=True)
