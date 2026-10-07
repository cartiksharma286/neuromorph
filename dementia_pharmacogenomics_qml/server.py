#!/usr/bin/env python3
"""
Flask Web Application Server:
Pharmacogenomics & Combinatorial Photonic Neural Excitation Engine
with Quantum Machine Learning and Elliptic Moduli Theory for Dementia Reversal
"""

import os, sys, json
from flask import Flask, request, jsonify, render_template, send_file
import numpy as np

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from quantum_elliptic_engine import EllipticModuliEngine, QuantumPharmacogenomicOptimizer
from pharmacogenomics_model import PharmacogenomicsEngine
from photonic_circuit_model import PhotonicNeuralCircuitEngine
from clinical_horizons import ClinicalHorizonsSimulator
from generate_nature_preprint import build_preprint_pdf
from quantum_molecular_design import QuantumMolecularDesignEngine

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'), static_folder=os.path.join(BASE_DIR, 'static'))

# Global optimizer instance
qml_optimizer = QuantumPharmacogenomicOptimizer(num_qubits=6)
PDF_PATH = os.path.join(BASE_DIR, 'Nature_Preprint_Pharmacogenomics_Photonic_Quantum_Dementia.pdf')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health():
    return jsonify({
        "status": "online",
        "app_name": "Neuromorph Quantum-Elliptic Pharmacogenomics & Photonic Dementia Suite",
        "preprint_exists": os.path.exists(PDF_PATH),
        "pdf_size_bytes": os.path.getsize(PDF_PATH) if os.path.exists(PDF_PATH) else 0
    })

@app.route('/api/simulate', methods=['POST'])
def simulate():
    data = request.get_json() or {}
    
    disease_type = data.get('disease_type', 'LOAD')
    apoe = data.get('apoe', 'e3/e4')
    cyp2d6 = data.get('cyp2d6', 'IM')
    early_onset = (disease_type == 'EOAD') or data.get('early_onset_mutation', False)
    selected_drugs = data.get('selected_drugs', ['Donepezil', 'Lecanemab'])
    photonic_bbb = data.get('photonic_assisted_bbb', True)
    
    wavelengths = data.get('wavelengths', [810, 1064])
    frequency_hz = float(data.get('frequency_hz', 40.0))
    irradiance_mw = float(data.get('irradiance_mw', 85.0))
    exposure_min = float(data.get('exposure_min', 30.0))
    target_circuits = data.get('target_circuits', None)
    
    # 1. Pharmacogenomic evaluation
    pgx_eval = PharmacogenomicsEngine.evaluate_patient_profile(
        apoe=apoe,
        cyp2d6=cyp2d6,
        early_onset_mutation=early_onset,
        selected_drugs=selected_drugs,
        photonic_assisted_bbb=photonic_bbb
    )
    
    # 2. Photonic neural circuit excitation
    pho_eval = PhotonicNeuralCircuitEngine.evaluate_combinatorial_excitation(
        wavelengths=wavelengths,
        frequency_hz=frequency_hz,
        surface_irradiance_mw=irradiance_mw,
        exposure_time_min=exposure_min,
        target_circuits=target_circuits
    )
    
    # 3. Elliptic moduli state computation
    # Pathological tau begins at 0.35 + 2.4i; therapeutic attractor is 0.05 + 1.25i
    synergy_factor = (pho_eval["gamma_synchrony_score"] * 0.5 + 
                      (pho_eval["overall_atp_boost_percent"] / 85.0) * 0.3 + 
                      (1.0 / max(0.5, pgx_eval["pathology_progression_index"])) * 0.2)
    synergy_factor = min(1.0, max(0.2, synergy_factor))
    
    tau_current = complex(
        0.35 * (1.0 - synergy_factor) + 0.05 * synergy_factor,
        2.40 * (1.0 - synergy_factor) + 1.25 * synergy_factor
    )
    tau_reduced, operations = EllipticModuliEngine.map_to_fundamental_domain(tau_current)
    j_inv = EllipticModuliEngine.klein_j_invariant(tau_current)
    weier = EllipticModuliEngine.weierstrass_p_orbit(tau_current)
    finite_curve = EllipticModuliEngine.finite_field_elliptic_curve(p=257, a=1, b=3)
    poincare_dist = EllipticModuliEngine.poincare_distance(tau_current, complex(0.05, 1.25))
    
    # 4. Long-term clinical horizons (0 to 60+ months)
    horizons = ClinicalHorizonsSimulator.simulate_cohort_trajectories(
        disease_type=disease_type,
        apoe_genotype=apoe,
        cyp2d6_status=cyp2d6,
        synergy_qml_score=synergy_factor,
        photonic_active=True,
        drug_active=True
    )
    
    return jsonify({
        "status": "success",
        "pharmacogenomics": pgx_eval,
        "photonic": pho_eval,
        "elliptic_moduli": {
            "tau_real": round(float(tau_current.real), 4),
            "tau_imag": round(float(tau_current.imag), 4),
            "tau_reduced_real": round(float(tau_reduced.real), 4),
            "tau_reduced_imag": round(float(tau_reduced.imag), 4),
            "modular_operations": operations,
            "poincare_distance_to_attractor": round(float(poincare_dist), 4),
            "klein_j_invariant_real": round(float(j_inv.real), 1),
            "klein_j_invariant_imag": round(float(j_inv.imag), 1),
            "weierstrass_p": weier,
            "finite_field_curve": finite_curve
        },
        "clinical_horizons": horizons
    })

@app.route('/api/optimize_qml', methods=['POST'])
def optimize_qml():
    data = request.get_json() or {}
    apoe = data.get('apoe', 'e3/e4')
    cyp2d6 = data.get('cyp2d6', 'IM')
    drug_dose = float(data.get('drug_dose', 10.0))
    wavelength = float(data.get('wavelength', 810.0))
    frequency_hz = float(data.get('frequency_hz', 40.0))
    tau = complex(float(data.get('tau_real', 0.25)), float(data.get('tau_imag', 1.6)))
    
    apoe_risk = PharmacogenomicsEngine.APOE_PROFILES.get(apoe, {}).get('risk_multiplier', 1.0)
    cyp_activity = PharmacogenomicsEngine.CYP2D6_PROFILES.get(cyp2d6, {}).get('activity_score', 1.0)
    
    result = qml_optimizer.optimize_step(
        apoe_risk=apoe_risk,
        cyp2d6_activity=cyp_activity,
        drug_dose=drug_dose,
        wavelength=wavelength,
        frequency_hz=frequency_hz,
        tau_modulus=tau,
        lr=0.09,
        steps=18
    )
    return jsonify(result)

@app.route('/api/pk_curve', methods=['GET'])
def pk_curve():
    drug = request.args.get('drug', 'Donepezil')
    cyp = request.args.get('cyp', 'IM')
    photonic = request.args.get('photonic', 'true').lower() == 'true'
    dose = float(request.args.get('dose', 10.0))
    
    res = PharmacogenomicsEngine.simulate_pharmacokinetic_trajectory(
        drug_name=drug,
        dose_mg=dose,
        cyp_status=cyp,
        photonic_active=photonic,
        hours=72,
        dt=0.5
    )
    return jsonify(res)

@app.route('/api/download_preprint')
def download_preprint():
    if not os.path.exists(PDF_PATH):
        build_preprint_pdf(PDF_PATH)
    return send_file(
        PDF_PATH,
        as_attachment=True,
        download_name='Nature_Preprint_Pharmacogenomics_Photonic_Quantum_Dementia.pdf',
        mimetype='application/pdf'
    )

@app.route('/api/regenerate_preprint', methods=['POST'])
def regenerate_preprint():
    try:
        build_preprint_pdf(PDF_PATH)
        return jsonify({
            "status": "success",
            "message": "Nature Preprint PDF successfully generated and compiled.",
            "path": PDF_PATH,
            "size_bytes": os.path.getsize(PDF_PATH)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/molecules')
def list_molecules():
    mols = QuantumMolecularDesignEngine.list_molecules()
    return jsonify({"status": "success", "molecules": mols})

@app.route('/api/molecule/<mol_id>')
def get_molecule(mol_id):
    mol = QuantumMolecularDesignEngine.get_molecule(mol_id)
    if not mol:
        return jsonify({"status": "error", "message": f"Molecule {mol_id} not found"}), 404
    return jsonify({"status": "success", "molecule": mol})

@app.route('/api/qml_dock', methods=['POST'])
def qml_dock():
    data = request.get_json() or {}
    mol_id = data.get('molecule_id', 'NQML-8101')
    target = data.get('target_protein', 'AChE')
    wavelength = float(data.get('wavelength', 810.0))
    res = QuantumMolecularDesignEngine.simulate_qml_electronic_docking(
        mol_id=mol_id,
        target_protein=target,
        photonic_wavelength=wavelength
    )
    return jsonify({"status": "success", "docking": res})

@app.route('/api/generate_molecular_candidate', methods=['POST'])
def generate_molecular_candidate():
    data = request.get_json() or {}
    scaffold = data.get('core_scaffold', 'Indanone_Piperidine')
    switch = data.get('photonic_switch', 'o-Nitrobenzyl_Ester')
    warhead = data.get('warhead', 'Fluorobenzyl_AChE')
    candidate = QuantumMolecularDesignEngine.generate_custom_qml_candidate(
        core_scaffold=scaffold,
        photonic_switch=switch,
        warhead=warhead
    )
    return jsonify({"status": "success", "candidate": candidate})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5080))
    print(f"Starting Neuromorph QML Pharmacogenomics & Photonic Dementia Suite on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
