from flask import Flask, render_template_string
import threading
import webbrowser
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from meg_analysis_bet_sss import run_analysis, run_realtime_nvqlink, run_prime_vortex_simulation
from meg_dewar_simulation import simulate_with_dewar

app = Flask(__name__)

PORT = 5003

@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>MEG Advanced Processing Node</title>
        <style>
            body { background: #0f172a; color: #e2e8f0; font-family: 'Inter', sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; height: 100vh; margin: 0; padding-top: 2rem; }
            .card { background: #1e293b; padding: 2rem; border-radius: 12px; border: 1px solid #334155; text-align: center; max-width: 800px; width: 100%; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5); z-index: 10; }
            h1 { margin-top: 0; color: #818cf8; }
            p { color: #94a3b8; margin-bottom: 1rem; }
            .controls { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; }
            .btn { display: inline-block; background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 10px 20px; border-radius: 8px; text-decoration: none; font-weight: 600; cursor: pointer; border: none; font-size: 0.95rem; transition: transform 0.2s; }
            .btn-secondary { background: linear-gradient(135deg, #3b82f6, #06b6d4); }
            .btn-accent { background: linear-gradient(135deg, #ec4899, #f43f5e); }
            .btn-success { background: linear-gradient(135deg, #10b981, #34d399); }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.4); }
            .spinner { border: 4px solid #334155; border-top: 4px solid #818cf8; border-radius: 50%; width: 24px; height: 24px; animation: spin 1s linear infinite; display: none; margin: 0 auto 1rem; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            #viewer { width: 95vw; height: 75vh; border: none; border-radius: 12px; margin-top: 20px; background: #1e293b; display: none; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5); }
        </style>
        <script>
            function startSim(route) {
                document.getElementById('spinner').style.display = 'block';
                document.getElementById('status').innerText = 'Running Simulation...';
                const viewer = document.getElementById('viewer');
                viewer.style.display = 'block';
                viewer.src = route;
                viewer.onload = function() {
                    document.getElementById('spinner').style.display = 'none';
                    document.getElementById('status').innerText = 'Simulation Complete.';
                };
            }
        </script>
    </head>
    <body>
        <div class="card">
            <h1>MEG Processing Node</h1>
            <p>Dedicated compute node for MEG/MRI Hybrid Simulations.</p>
            <div id="spinner" class="spinner"></div>
            <p id="status" style="font-weight: bold;">Ready to compute.</p>
            <div class="controls">
                <button class="btn" onclick="startSim('/run_simulation')">Run BET & SSS Analysis</button>
                <button class="btn btn-secondary" onclick="startSim('/run_beamformer')">Run Beamformer & Geodesics</button>
                <button class="btn btn-accent" onclick="startSim('/run_qml_source_localization')">Run QML Source Localization</button>
                <button class="btn btn-secondary" onclick="startSim('/run_dewar')">Run Dewar Quantum CFD</button>
                <button class="btn btn-accent" onclick="startSim('/run_nvqlink')">Run Real-Time NVQLink SSS</button>
                <button class="btn btn-success" onclick="startSim('/run_prime_vortex')">Run Prime Vortex Generator</button>
            </div>
        </div>
        <iframe id="viewer"></iframe>
    </body>
    </html>
    """

@app.route('/run_prime_vortex')
def run_prime():
    try:
        html_report = run_prime_vortex_simulation()
        return html_report
    except Exception as e:
        import traceback
        return f"<h1>Error Running Prime Vortex Simulation</h1><pre>{traceback.format_exc()}</pre>"

@app.route('/run_simulation')
def run_sim():
    try:
        html_report = run_analysis()
        return html_report
    except Exception as e:
        return f"<h1>Error Running BET Simulation</h1><pre>{str(e)}</pre>"
        
@app.route('/run_beamformer')
def run_beamformer():
    # Run the simulation script
    import subprocess
    # Run synchronously for now (simulation is faster with vectorization)
    cwd_path = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run([sys.executable, 'meg_simulation.py'], cwd=cwd_path)
    
    if result.returncode == 0:
        # Read the generated HTML
        with open(os.path.join(cwd_path, 'meg_simulation_interactive.html'), 'r') as f:
            return f.read()
    else:
        return "<h1>Error running Beamformer Simulation</h1>"

@app.route('/run_qml_source_localization')
def run_qml_source_localization():
    import subprocess
    cwd_path = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run([sys.executable, 'meg_qml_source_localization.py'], cwd=cwd_path)
    
    if result.returncode == 0:
        with open(os.path.join(cwd_path, 'qml_localization_report.html'), 'r') as f:
            return f.read()
    else:
        return "<h1>Error running QML Source Localization Simulation</h1>"

@app.route('/run_nvqlink')
def run_nvq():
    try:
        html_report = run_realtime_nvqlink()
        return html_report
    except Exception as e:
        return f"<h1>Error Running NVQLink Simulation</h1><pre>{str(e)}</pre>"

@app.route('/run_dewar')
def run_dewar():
    try:
        html_report = simulate_with_dewar()
        return html_report
    except Exception as e:
        return f"<h1>Error Running Dewar Simulation</h1><pre>{str(e)}</pre>"

def run_flask():
    print(f"Starting MEG Processing App on http://localhost:{PORT}")
    webbrowser.open(f"http://localhost:{PORT}")
    app.run(port=PORT, debug=False, threaded=True)

if __name__ == "__main__":
    run_flask()
