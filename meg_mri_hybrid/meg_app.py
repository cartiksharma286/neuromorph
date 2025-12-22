from flask import Flask, render_template_string
import threading
import webbrowser
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from meg_analysis_bet_sss import run_analysis, run_realtime_nvqlink
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
            body { background: #0f172a; color: #e2e8f0; font-family: 'Inter', sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
            .card { background: #1e293b; padding: 2rem; border-radius: 12px; border: 1px solid #334155; text-align: center; max-width: 500px; width: 100%; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5); }
            h1 { margin-top: 0; color: #818cf8; }
            p { color: #94a3b8; margin-bottom: 2rem; }
            .btn { display: inline-block; background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; cursor: pointer; border: none; font-size: 1rem; transition: transform 0.2s; margin: 5px; }
            .btn-secondary { background: linear-gradient(135deg, #3b82f6, #06b6d4); }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.4); }
            .btn-accent { background: linear-gradient(135deg, #ec4899, #f43f5e); }
            .spinner { border: 4px solid #334155; border-top: 4px solid #818cf8; border-radius: 50%; width: 24px; height: 24px; animation: spin 1s linear infinite; display: none; margin: 0 auto 1rem; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
        <script>
            function startSim(route) {
                document.querySelectorAll('.btn').forEach(b => b.style.display = 'none');
                document.getElementById('spinner').style.display = 'block';
                document.getElementById('status').innerText = 'Running Simulation (this may take a few seconds)...';
                window.location.href = route;
            }
        </script>
    </head>
    <body>
        <div class="card">
            <h1>MEG Processing Node</h1>
            <p>Dedicated compute node for MEG/MRI Hybrid Simulations.</p>
            <div id="spinner" class="spinner"></div>
            <p id="status">Ready to compute.</p>
            <button class="btn" onclick="startSim('/run_simulation')">Run BET & SSS Analysis</button>
            <button class="btn btn-secondary" onclick="startSim('/run_dewar')">Run Dewar Quantum CFD</button>
            <button class="btn btn-accent" onclick="startSim('/run_nvqlink')">Run Real-Time NVQLink SSS</button>
        </div>
    </body>
    </html>
    """

@app.route('/run_simulation')
def run_sim():
    try:
        html_report = run_analysis()
        return html_report
    except Exception as e:
        return f"<h1>Error Running BET Simulation</h1><pre>{str(e)}</pre>"

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
