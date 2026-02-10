from flask import Flask, render_template, request, jsonify, send_file
from simulator_core import MRIReconstructionSimulator
from llm_modules import GeminiRFDesigner, LLMPulseDesigner
from statistical_adaptive_pulse import create_adaptive_sequence, ADAPTIVE_SEQUENCES
from quantum_vascular_coils import get_coil_summary, QUANTUM_VASCULAR_COIL_LIBRARY
import os
import generate_pdf
import generate_report_images
import numpy as np

app = Flask(__name__)
LATEST_CONTEXT = {}
NVQLINK_STATUS = {
    'enabled': False,
    'bandwidth_gbps': 400,
    'latency_ns': 12,
    'quantum_state': 'Entangled',
    'uptime_hours': 0
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/human_readable_status')
def status():
    return "MRI Simulator Online"

GLOBAL_SIM = None

@app.route('/api/simulate', methods=['POST'])
def simulate():
    global GLOBAL_SIM
    try:
        data = request.json or {}
        res = int(data.get('resolution', 128))
        seq_type = data.get('sequence', 'SE')
        tr = float(data.get('tr', 2000))
        te = float(data.get('te', 100))
        ti = float(data.get('ti', 500))
        flip_angle = float(data.get('flip_angle', 30))
        coil_mode = data.get('coils', 'standard')
        num_coils = int(data.get('num_coils', 8))
        noise = float(data.get('noise_level', 0.0)) # Allow noise from UI
        recon_method = data.get('recon_method', 'SoS')
        use_shimming = data.get('shimming', False)
        
        slice_orientation = data.get('slice_orientation', 'axial')
        slice_pos = float(data.get('slice_pos', 0.5))
        
        phantom_type = 'brain'
        if coil_mode == 'cardiothoracic_array':
            phantom_type = 'cardiac'
        elif coil_mode == 'knee_vascular_array':
            phantom_type = 'knee'

        # Initialize or Re-initialize Simulator if resolution changes
        if GLOBAL_SIM is None or GLOBAL_SIM.resolution != res:
            GLOBAL_SIM = MRIReconstructionSimulator(resolution=res)
            # Reset state trackers
            GLOBAL_SIM.last_phantom_type = None
            GLOBAL_SIM.last_coil_config = None
            GLOBAL_SIM.last_shim_report = None
        
        sim = GLOBAL_SIM
        
        # 1. Update Phantom if needed
        if getattr(sim, 'last_phantom_type', None) != phantom_type:
            sim.setup_phantom(use_real_data=True, phantom_type=phantom_type)
            sim.last_phantom_type = phantom_type
            # If phantom changes, we might need to redo coils if they depend on phantom geometry (shimming does)
            # But generate_coil_sensitivities mainly uses grid. 
            # However, optimize_shimming uses the phantom. So invalidate coils.
            sim.last_coil_config = None 

        # 2. Update Coils if needed
        # Config tuple: (coil_mode, num_coils, use_shimming)
        current_coil_config = (coil_mode, num_coils, use_shimming)
        
        if getattr(sim, 'last_coil_config', None) != current_coil_config:
            sim.generate_coil_sensitivities(num_coils=num_coils, coil_type=coil_mode, optimal_shimming=use_shimming)
            
            shim_report = None
            if use_shimming:
                w_opt, shim_info = sim.optimize_shimming_b_field(sim.coils)
                shim_report = sim.generate_shim_report_data(w_opt, shim_info)
                if coil_mode != 'n25_array':
                    new_coils = []
                    for i, c in enumerate(sim.coils):
                        if i < len(w_opt):
                            new_coils.append(c * w_opt[i])
                    sim.coils = new_coils
            
            sim.last_shim_report = shim_report
            sim.last_coil_config = current_coil_config
        else:
            shim_report = getattr(sim, 'last_shim_report', None)

        # 3. Set View (Slicing) - Fast operation
        sim.set_view(slice_orientation, slice_pos)
        
        # 4. Acquire Signal & Reconstruct (Sequence Dependent)
        kspace, M_ref = sim.acquire_signal(sequence_type=seq_type, TR=tr, TE=te, TI=ti, flip_angle=flip_angle, noise_level=noise)
        
        if recon_method == 'DeepLearning':
            recon_img = sim.deep_learning_reconstruct(kspace)
        else:
            recon_img, coil_imgs = sim.reconstruct_image(kspace, method=recon_method)
        
        metrics = sim.compute_metrics(recon_img, M_ref)
        # Statistical Classifier might be fast enough, or we can cache?
        # It analyzes the image content. Since image changes with sequence, we must run it.
        stat_metrics = sim.classifier.analyze_image(recon_img)
        metrics.update(stat_metrics)
        
        plots = sim.generate_plots(kspace, recon_img, M_ref)
        signal_study = sim.generate_signal_study(seq_type)
        
        # Save plots to disk (Optional for persistence, but slow? Let's keep it for report generation)
        img_dir = os.path.join(os.getcwd(), 'static', 'report_images')
        os.makedirs(img_dir, exist_ok=True)
        import base64
        for key, b64_str in plots.items():
            # Optimize: Only save if requested? 
            # The original code saved them. We keep it for now.
            with open(os.path.join(img_dir, f"{key}.png"), "wb") as f:
                f.write(base64.b64decode(b64_str))
                
        LATEST_CONTEXT.update({
            "coil_mode": coil_mode,
            "seq_type": seq_type,
            "metrics": metrics,
            "signal_study": signal_study,
            "timestamp": "January 10, 2026",
            "shim_report": shim_report
        })
        
        return jsonify({
            "success": True,
            "metrics": metrics,
            "plots": plots,
            "shim_report": shim_report,
            "signal_study": signal_study
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/design_rf', methods=['POST'])
def design_rf():
    try:
        data = request.json or {}
        prompt = data.get('prompt', '')
        field = data.get('field', '3T')
        designer = GeminiRFDesigner()
        result = designer.generate_design(prompt, field)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/render_cortical', methods=['POST'])
def render_cortical():
    try:
        sim = MRIReconstructionSimulator()
        pd_surface = sim.renderCorticalSurface2D()
        metadata = sim.renderCorticalSurface3D()
        import matplotlib.pyplot as plt
        import io
        import base64
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(6,6))
        fig.patch.set_facecolor('#0f172a')
        ax.imshow(pd_surface, cmap='gray', origin='lower')
        ax.set_title("Cortical Surface Reconstruction")
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return jsonify({"success": True, "plot": plot_b64, "metadata": metadata})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/report')
def get_report():
    try:
        if not LATEST_CONTEXT:
            return jsonify({"success": False, "error": "No simulation run yet."}), 400
        c = LATEST_CONTEXT
        coil_name = c["coil_mode"].replace('_', ' ').title()
        
        physics_desc = "Standard MRI Acquisition Loop."
        math_block = r"$$ S(k) = \int M(r) e^{-i k r} dr $$"
        
        cmd = c['coil_mode']
        if 'birdcage' in cmd or cmd == 'standard':
            physics_desc = "The Birdcage Coil utilizes a ladder network to create a homogeneous B1 field."
            math_block = r"$$ \omega_m = \frac{1}{\sqrt{L_{leg} C_{ring}}} [2 \sin(\frac{m\pi}{N})]^{-1} $$"
        elif 'gemini' in cmd:
            physics_desc = "The Gemini Optimized 3T array uses an AI-driven current shimming."
            math_block = r"$$ \mathbf{w}_{opt} = \operatorname*{argmin}_{\mathbf{w}} (\| |\mathbf{A}\mathbf{w}| - \mathbf{T} \|_2^2 + \lambda \|\mathbf{w}\|_2^2) $$"
        elif 'quantum' in cmd:
            physics_desc = "This coil topology exploits the Berry Phase of adiabatic spin transport."
            math_block = r"$$ \gamma_n(C) = i \oint_C \langle \psi_n | \nabla | \psi_n \rangle \cdot d\mathbf{R} $$"
        
        seq_math = ""
        st = c['seq_type']
        if st == 'Gemini3.0':
            seq_math = r"$$ \mathcal{J}(\mathbf{m}) =  -\sum p_i \ln(p_i) + \lambda \int |\nabla \mathbf{m}|^2 d\Omega $$"
        elif st == 'QuantumNVQLink':
            seq_math = r"$$ M_{eff}(v) = M_0 \sin(\alpha) \frac{1-e^{-TR/T1}}{1-\cos(\alpha)e^{-TR/T1}} + \beta v \delta t $$"
        elif st == 'QuantumBerryPhase':
            seq_math = r"$$ S \propto M_0 e^{-TE/T2^*} e^{i(\Phi_{dyn} + \Phi_B)} $$"
        elif st == 'QuantumLowEnergyBeam':
            seq_math = r"$$ I(r) = I_0 (1 + \eta \mathcal{H}(\chi)) $$"
        else:
            seq_math = r"$$ S \propto \rho (1-e^{-TR/T1})e^{-TE/T2} $$"

        content = """
# NeuroPulse Clinical Physics Report
**Date:** {timestamp}
**Simulation ID:** {seq_type}-{coil_mode}

---

## 1. Executive Summary
This report details the simulation results for the **{coil_name}** operating with **{seq_type}**.

## 2. Physics & Circuit Topology
{physics_desc}

### Coil Derivation
{math_block}

### Pulse Sequence Physics
{seq_math}

---

## 3. Finite Math & Discrete Derivations
$$ M_z^{sub} = M_z(t) \cdot e^{-\Delta t/T1} + M_0(1 - e^{-\Delta t/T1}) $$
$$ Z_{ij} = \sum \\frac{\mu_0}{4\pi} \\frac{\mathbf{J}_i \cdot \mathbf{J}_j}{|\mathbf{r}_{ij}|} \Delta A_k $$

---

## 4. Visual Reconstruction Data
![Reconstruction](static/report_images/recon.png)

## 5. Metrics
* **Contrast:** {contrast:.4f}
* **Sharpness:** {sharpness:.2f}
"""
        # Using simple replace to avoid any .format() KeyError with LaTeX braces
        report_md = content.replace("{timestamp}", str(c['timestamp'])) \
                          .replace("{seq_type}", str(c['seq_type'])) \
                          .replace("{coil_mode}", str(c['coil_mode'])) \
                          .replace("{coil_name}", str(coil_name)) \
                          .replace("{physics_desc}", str(physics_desc)) \
                          .replace("{math_block}", str(math_block)) \
                          .replace("{seq_math}", str(seq_math)) \
                          .replace("{contrast:.4f}", f"{c['metrics'].get('contrast', 0):.4f}") \
                          .replace("{sharpness:.2f}", f"{c['metrics'].get('sharpness', 0):.2f}")
        
        md_path = os.path.join(os.getcwd(), 'detailed_coil_report.md')
        with open(md_path, 'w') as f:
            f.write(report_md)
            
        pdf_path = os.path.join(os.getcwd(), 'NeuroPulse_Coil_Report.pdf')
        generate_pdf.md_to_pdf(md_path, pdf_path)
        
        return send_file(pdf_path, as_attachment=True, download_name='NeuroPulse_Detailed_Report.pdf')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5050, debug=True)
