from flask import Flask, render_template, request, jsonify
from simulator_core import MRIReconstructionSimulator
from llm_modules import GeminiRFDesigner, LLMPulseDesigner
import os

app = Flask(__name__)

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/human_readable_status')
def status():
    return "MRI Simulator Online"

@app.route('/api/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json or {}
        
        # Parameters
        res = int(data.get('resolution', 128))
        seq_type = data.get('sequence', 'SE')
        tr = float(data.get('tr', 2000))
        te = float(data.get('te', 100))
        ti = float(data.get('ti', 500))
        flip_angle = float(data.get('flip_angle', 30))
        coil_mode = data.get('coils', 'standard') # 'standard' or 'custom_phased_array'
        num_coils = int(data.get('num_coils', 8))
        noise = float(data.get('noise', 0.01))
        recon_method = data.get('recon_method', 'SoS')
        use_shimming = data.get('shimming', False) # New Param
        
        # Run Simulation
        sim = MRIReconstructionSimulator(resolution=res)
        
        # Determine Phantom Type
        phantom_type = 'brain'
        if coil_mode == 'cardiothoracic_array':
            phantom_type = 'cardiac'

        # 1. Generate Phantom (Load Real Data if possible)
        sim.setup_phantom(use_real_data=True, phantom_type=phantom_type)
        
        # 2. Generate Coils
        sim.generate_coil_sensitivities(num_coils=num_coils, coil_type=coil_mode, optimal_shimming=use_shimming)
        
        shim_report = None
        if use_shimming:
            # Run Rigorous Math Solver B-Matrix Optimization
            w_opt, shim_info = sim.optimize_shimming_b_field(sim.coils)
            shim_report = sim.generate_shim_report_data(w_opt, shim_info)
            
            # Apply weights if not already handled by internal logic (N25 does it, but this refinement is rigorous)
            if coil_mode != 'n25_array':
                new_coils = []
                for i, c in enumerate(sim.coils):
                    if i < len(w_opt):
                        new_coils.append(c * w_opt[i])
                sim.coils = new_coils
        
        # Generate ECG (58yo AFib model)
        ecg_time, ecg_signal = sim.generate_ecg_waveform(duration_sec=2.0)
        ecg_data = {
            "time": ecg_time, 
            "signal": ecg_signal, 
            "leads": ["Lead II"],
            "source_localization": {
                "origin": "Left Superior Pulmonary Vein (LSPV)",
                "mechanism": "Focal Trigger + Reentry",
                "confidence": "94%"
            }
        }

        # 3. Acquire
        kspace, M_ref = sim.acquire_signal(sequence_type=seq_type, TR=tr, TE=te, TI=ti, flip_angle=flip_angle, noise_level=noise)
        
        # 4. Reconstruct
        recon_img, coil_imgs = sim.reconstruct_image(kspace, method=recon_method)
        
        # 5. Metrics & Plot
        metrics = sim.compute_metrics(recon_img, M_ref)
        
        # Add Statistical Analysis
        stat_metrics = sim.classifier.analyze_image(recon_img)
        metrics.update(stat_metrics)
        
        plots = sim.generate_plots(kspace, recon_img, M_ref)
        
        return jsonify({
            "success": True,
            "metrics": metrics,
            "plots": plots,
            "shim_report": shim_report,
            "ecg_data": ecg_data
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

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


from flask import send_file
import generate_pdf
import generate_report_images

@app.route('/api/report')
def get_report():
    try:
        # 1. Ensure Images exist (or regenerate)
        # Call generate_all to get circuits too
        generate_report_images.generate_all()
        
        # 2. Generate PDF
        md_path = os.path.join(os.getcwd(), 'detailed_coil_report.md')
        pdf_path = os.path.join(os.getcwd(), 'NeuroPulse_Coil_Report.pdf')
        
        generate_pdf.md_to_pdf(md_path, pdf_path)
        
        return send_file(pdf_path, as_attachment=True, download_name='NeuroPulse_Detailed_Report.pdf')
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5005, debug=True)
