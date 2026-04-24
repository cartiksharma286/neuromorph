import os
from flask import Flask, render_template, request, jsonify
from ehr_logic import analyze_patient_record, process_pacs_image

app = Flask(__name__)

# In-memory mock database
patients_db = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/patient', methods=['POST'])
def add_patient():
    data = request.json
    patient_id = data.get('id', 'Unknown')
    name = data.get('name', 'Unknown')
    records = data.get('records', '')
    pacs_image_data = data.get('pacs_data', None)
    
    # Process optional PACS data
    pacs_analysis = None
    if pacs_image_data:
        pacs_analysis = process_pacs_image(pacs_image_data)
    
    # Run the multimodal LLM reasoning engine
    ai_outcome = analyze_patient_record(records, pacs_analysis)
    
    patients_db[patient_id] = {
        'name': name,
        'records': records,
        'ai_analysis': ai_outcome
    }
    
    return jsonify({
        'status': 'success',
        'patient_id': patient_id,
        'llm_outcome': ai_outcome
    })

@app.route('/api/patients', methods=['GET'])
def get_patients():
    return jsonify(patients_db)

if __name__ == '__main__':
    app.run(debug=True, port=5003)
