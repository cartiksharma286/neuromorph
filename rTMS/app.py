from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from logic.rtms_engine import (
    run_full_simulation,
    get_equipment_list,
    get_tremor_clinical_data,
    get_treatment_paradigm
)

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

if __name__ == '__main__':
    app.run(debug=True, port=5002)
