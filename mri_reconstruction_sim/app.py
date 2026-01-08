from flask import Flask, render_template, request, jsonify
from simulator_core import MRIReconstructionSimulator
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
        
        # Run Simulation
        sim = MRIReconstructionSimulator(resolution=res)
        
        # 1. Generate Phantom
        sim.generate_brain_phantom()
        
        # 2. Generate Coils
        sim.generate_coil_sensitivities(num_coils=num_coils, coil_type=coil_mode)
        
        # 3. Acquire
        kspace, M_ref = sim.acquire_signal(sequence_type=seq_type, TR=tr, TE=te, TI=ti, flip_angle=flip_angle, noise_level=noise)
        
        # 4. Reconstruct
        recon_img, coil_imgs = sim.reconstruct_image(kspace, method='SoS')
        
        # 5. Metrics & Plot
        metrics = sim.compute_metrics(recon_img, M_ref)
        plots = sim.generate_plots(kspace, recon_img, M_ref)
        
        return jsonify({
            "success": True,
            "metrics": metrics,
            "plots": plots
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5005, debug=True)
