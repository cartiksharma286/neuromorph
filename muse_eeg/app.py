
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import pandas as pd
import numpy as np
from cortical_connectivity import generate_cortical_connectivity
from eeg_stimulus import run_stimulus_response_experiment
from math_modeling import continued_fraction_model
from fea_simulation import simulate_fea
from visualization import visualize_cortical_flow

app = Flask(__name__)
app.secret_key = 'museeegsecret'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
SAMPLE_EEG = os.path.join(os.path.dirname(__file__), 'sample_eeg.csv')

def get_default_eeg():
    return pd.read_csv(SAMPLE_EEG).values

@app.route('/', methods=['GET', 'POST'])
def index():
    tab = request.args.get('tab', 'connectivity')
    # Connectivity
    conn_result = None
    conn_data = get_default_eeg()
    if request.method == 'POST' and request.form.get('action') == 'connectivity':
        file = request.files.get('eeg_csv')
        if file and file.filename:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            conn_data = pd.read_csv(path).values
        if conn_data.shape[1] != 40:
            flash('EEG data must have 40 columns (sensors)')
        else:
            conn_result = generate_cortical_connectivity(conn_data)
    # Stimulus
    stim_result = None
    stim_data = get_default_eeg()
    stim_str = ''
    if request.method == 'POST' and request.form.get('action') == 'stimulus':
        file = request.files.get('eeg_csv')
        stim_str = request.form.get('stimulus', '')
        if file and file.filename:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            stim_data = pd.read_csv(path).values
        try:
            stimulus = [float(x) for x in stim_str.split(',')]
        except Exception:
            flash('Invalid stimulus format')
            stimulus = None
        if stimulus and len(stimulus) == 40 and stim_data.shape[1] == 40:
            stim_result = run_stimulus_response_experiment(stim_data, stimulus)
        else:
            if stim_str:
                flash('Stimulus and EEG data must have 40 values each')
    # Math
    math_result = None
    math_data = get_default_eeg()
    if request.method == 'POST' and request.form.get('action') == 'math':
        file = request.files.get('data_csv')
        if file and file.filename:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            math_data = pd.read_csv(path).values
        if math_data.shape[1] != 40:
            flash('Data must have 40 columns (sensors)')
        else:
            math_result = continued_fraction_model(math_data)
    # FEA
    fea_result = None
    if request.method == 'POST' and request.form.get('action') == 'fea':
        config = list(range(40))
        fea_result = simulate_fea(config)
    # Visualization
    vis_message = None
    if request.method == 'POST' and request.form.get('action') == 'visualize':
        conn_file = request.files.get('conn_csv')
        fea_file = request.files.get('fea_csv')
        if conn_file and fea_file and conn_file.filename and fea_file.filename:
            conn_path = os.path.join(UPLOAD_FOLDER, conn_file.filename)
            fea_path = os.path.join(UPLOAD_FOLDER, fea_file.filename)
            conn_file.save(conn_path)
            fea_file.save(fea_path)
            conn = pd.read_csv(conn_path, header=None).values.flatten()
            fea = pd.read_csv(fea_path, header=None).values.flatten()
            if len(conn) != 40 or len(fea) != 40:
                vis_message = 'Both files must have 40 values (sensors)'
            else:
                fig = visualize_cortical_flow(conn, fea)
                fig.write_html('uploads/visualization.html')
                return send_file('uploads/visualization.html')
    return render_template('main_tabs.html',
        tab=tab,
        conn_result=conn_result.tolist() if isinstance(conn_result, np.ndarray) else conn_result,
        stim_result=stim_result.tolist() if isinstance(stim_result, np.ndarray) else stim_result,
        math_result=math_result.tolist() if isinstance(math_result, np.ndarray) else math_result,
        fea_result=fea_result.tolist() if isinstance(fea_result, np.ndarray) else fea_result,
        vis_message=vis_message,
        stim_str=stim_str
    )

if __name__ == '__main__':
    app.run(debug=True, port=5002, host='0.0.0.0')
