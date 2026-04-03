import os, io
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from logic.rtms_engine import (
    run_full_simulation,
    get_equipment_list,
    get_tremor_clinical_data,
    get_treatment_paradigm,
    get_dementia_longterm_care
)
from logic.monteris_cf_treatment import (
    full_treatment_paradigm,
    intraop_thermometry,
    preop_dti_sequence,
    preop_fmri_bold,
    postop_flair_swi,
    jc_state_transfer,
    risk_stratification,
)
from logic.dbs_statistical_manifold import generate_dbs_treatment_protocol

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    data = request.json
    condition = data.get('condition', 'stroke')
    results = run_full_simulation(condition)
    return jsonify({
        "status": "success",
        "message": "Optimal protocol calculated via GCP clustered FEA/BEM optimization.",
        "data": results
    })

@app.route('/api/equipment', methods=['GET'])
def equipment():
    return jsonify({"status": "success", "data": get_equipment_list()})

@app.route('/api/tremor-clinical', methods=['GET'])
def tremor_clinical():
    return jsonify({"status": "success", "data": get_tremor_clinical_data()})

@app.route('/api/treatment-paradigm', methods=['GET'])
def treatment_paradigm():
    condition = request.args.get('condition', 'stroke')
    return jsonify({"status": "success", "data": get_treatment_paradigm(condition)})

@app.route('/api/dementia-longterm', methods=['GET'])
def dementia_longterm():
    return jsonify({"status": "success", "data": get_dementia_longterm_care()})

@app.route('/api/dbs-imaging', methods=['GET'])
def dbs_imaging_protocol():
    return jsonify({"status": "success", "data": generate_dbs_treatment_protocol()})

# ── Monteris CF Treatment Paradigm Routes ────────────────────────────────────

@app.route('/api/monteris/full-paradigm', methods=['POST'])
def monteris_full_paradigm():
    data = request.json or {}
    condition = data.get('condition', 'glioma')
    preset    = data.get('monteris_preset', 'standard')
    result    = full_treatment_paradigm(condition, preset)
    return jsonify({"status": "success", "data": result})

@app.route('/api/monteris/preop-dti', methods=['GET'])
def monteris_preop_dti():
    return jsonify({"status": "success", "data": preop_dti_sequence()})

@app.route('/api/monteris/preop-fmri', methods=['GET'])
def monteris_preop_fmri():
    return jsonify({"status": "success", "data": preop_fmri_bold()})

@app.route('/api/monteris/intraop-thermometry', methods=['POST'])
def monteris_intraop():
    data   = request.json or {}
    preset = data.get('preset', 'standard')
    return jsonify({"status": "success", "data": intraop_thermometry(preset)})

@app.route('/api/monteris/postop-monitoring', methods=['GET'])
def monteris_postop():
    return jsonify({"status": "success", "data": postop_flair_swi()})

@app.route('/api/monteris/qnc-state-transfer', methods=['POST'])
def monteris_qnc():
    data      = request.json or {}
    n_qubits  = int(data.get('n_qubits', 8))
    abl_frac  = float(data.get('ablation_fraction', 0.3))
    omega_c   = float(data.get('omega_c_MHz', 5.0))
    g_kHz     = float(data.get('g_coupling_kHz', 50.0))
    return jsonify({"status": "success",
                    "data": jc_state_transfer(n_qubits, omega_c, g_kHz, abl_frac)})

@app.route('/api/monteris/risk-stratification', methods=['POST'])
def monteris_risk():
    data = request.json or {}
    return jsonify({"status": "success", "data": risk_stratification(
        age=float(data.get('age', 55)),
        kps=float(data.get('kps', 80)),
        tumour_vol_cm3=float(data.get('tumour_vol_cm3', 5.0)),
        eloquent_proximity_mm=float(data.get('eloquent_proximity_mm', 15.0)),
    )})

@app.route('/api/monteris/nature-report', methods=['GET'])
def monteris_nature_report():
    """Stream the Nature publication PDF."""
    pdf_path = os.path.join(os.path.dirname(__file__),
                            'seqs', 'Nature_Monteris_CF_Treatment.pdf')
    if not os.path.exists(pdf_path):
        # Generate on demand
        try:
            from generate_nature_monteris_report import generate_nature_monteris_report
            generate_nature_monteris_report(pdf_path)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return send_file(pdf_path, mimetype='application/pdf',
                     as_attachment=False,
                     download_name='Nature_Monteris_CF_Treatment.pdf')

if __name__ == '__main__':
    app.run(debug=True, port=5002)
