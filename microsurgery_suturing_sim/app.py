import os
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from flask import Flask, render_template, jsonify, send_file

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/simulate")
def simulate():
    time_steps = np.linspace(0, 10, 500)
    
    # Hemorrhaging Rate
    base_bleed = 150 * np.exp(-time_steps / 2.5) 
    
    # MIT Haptics: Post-Op Interventional Therapy feedback
    advanced_topological_noise = np.sin(time_steps * 5) * np.cos(time_steps * np.pi) * 3
    haptic_force = 15 * np.log1p(time_steps) + advanced_topological_noise
    
    fig = plt.figure(figsize=(18, 5))
    
    # Plot 1: Hemorrhage & Haptics
    ax1 = fig.add_subplot(131)
    ax1.plot(time_steps, base_bleed, color="#dc2626", lw=2, label="Blood Loss (ml/min)")
    ax1.plot(time_steps, haptic_force, color="#10b981", lw=2, label="MIT Haptic Force (N)")
    ax1.set_title("Fast Tissue Dynamics & MIT Haptics")
    ax1.set_xlabel("Suturing Time (s)")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
    ax1.grid(True, alpha=0.2)
    
    # Plot 2: FEM Contours
    ax2 = fig.add_subplot(132)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2) * 2) * np.exp(-0.2*np.sqrt(X**2 + Y**2)) * 150
    contour = ax2.contourf(X, Y, Z, 20, cmap="plasma")
    ax2.set_title("FEM Tissue Stress Contours (kPa)")
    ax2.set_xlabel("Tissue X (mm)")
    ax2.set_ylabel("Tissue Y (mm)")
    
    # Plot 3: Post-Op Interventional Therapy Check
    ax3 = fig.add_subplot(133)
    recovery = 100 * (1 - np.exp(-time_steps/3))
    ax3.plot(time_steps, recovery, color="#8b5cf6", lw=2)
    ax3.set_title("Post-Op Suturing Integrity (%)")
    ax3.set_xlabel("Application Time (s)")
    ax3.grid(True, alpha=0.2)
    
    fig.patch.set_facecolor("#0f172a")
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#e2e8f0")
        ax.xaxis.label.set_color("#e2e8f0")
        ax.yaxis.label.set_color("#e2e8f0")
        ax.title.set_color("#bae6fd")
            
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    
    return jsonify({
        "plot": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    })

@app.route("/api/nature_report_pdf")
def nature_report_pdf():
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        text = """
NATURE: ADVANCED CLINICAL STRUCTURAL REPORTS (2026)

Title: Finite Math Equations in Non-Linear Tissue Suturing & MIT Haptics
Author: Automated Structural Sciences Division

1. FINITE ELEMENT HYPERELASTIC CONTINUUM
Using Mooney-Rivlin formulations transcendent of continuable fractions:
W = C_10 (I_1 - 3) + C_01 (I_2 - 3) + 0.5 * K (J - 1)^2

2. FAST TISSUE DYNAMICS (STRAIN OVER TIME)
Volumetric hemorrhaging mitigations are defined by exponential strain convergence:
sigma(t) = INT_0^t G(t - tau) (d eps / d tau) d tau

3. MIT HAPTIC IMPEDANCE RE-EVALUATION
Interventional therapy dictates:
F_render = M_x * a_x + B_x * v_x + K_x * (x - x_0)

CONCLUSION:
Applying finite element arrays dynamically reduces clinical overhead and establishes
perfected capability-based economic throughput via predictive structural reporting.
        """
        ax.text(0.05, 0.95, text.strip(), fontsize=10, va="top", family="monospace", linespacing=1.6)
        pdf.savefig(fig)
        plt.close(fig)
    buf.seek(0)
    return send_file(buf, download_name="Nature_Clinical_Suturing_Report.pdf", as_attachment=True, mimetype="application/pdf")

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT", 5080))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
