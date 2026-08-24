"""
Flask Backend: Pelvic Floor Implant BEM / HPC / NVQLink Design Studio
Wires together chamfer geometry generation, continued-fraction manifold
repair, boundary-element simulation, HPC job scheduling, and the NVQLink
hybrid quantum-classical acceleration layer, plus Nature preprint export.
"""

import os
import subprocess
import sys
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

from chamfer_geometry import ChamferGeometryEngine
from continued_fraction_manifold import ContinuedFractionManifold
from bem_engine import BoundaryElementEngine
from hpc_scheduler import HPCScheduler
from nvqlink_interface import NVQLinkInterface

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

chamfer_engine = ChamferGeometryEngine()
cf_manifold = ContinuedFractionManifold()
bem_engine = BoundaryElementEngine()
hpc = HPCScheduler()
nvqlink = NVQLinkInterface()

state = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'Pelvic Floor BEM/HPC/NVQLink Studio', 'version': '1.0.0'})


@app.route('/api/generate-geometry', methods=['POST'])
def generate_geometry():
    """Generate implant boundary + chamfer profile"""
    try:
        data = request.json or {}
        length = float(data.get('length_mm', 40.0))
        width = float(data.get('width_mm', 30.0))
        thickness = float(data.get('thickness_mm', 1.0))
        corner_radius = float(data.get('corner_radius_mm', 4.0))
        chamfer_width = float(data.get('chamfer_width_mm', 1.5))
        chamfer_angle = float(data.get('chamfer_angle_deg', 45.0))
        material = data.get('material', 'composite')

        boundary = chamfer_engine.generate_implant_boundary(length, width, corner_radius)
        chamfer = chamfer_engine.apply_chamfer(boundary, chamfer_width, chamfer_angle, thickness)

        state['boundary'] = boundary
        state['chamfer'] = chamfer
        state['design'] = {
            'material': material,
            'dimensions': {'length_mm': length, 'width_mm': width, 'thickness_mm': thickness}
        }

        return jsonify({
            'status': 'success',
            'boundary_points': boundary.tolist(),
            'chamfer': chamfer,
            'design': state['design'],
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/repair-manifold', methods=['POST'])
def repair_manifold():
    """Run geometric mesh repair + continued-fraction manifold blending"""
    try:
        data = request.json or {}
        if 'chamfer' not in state:
            return jsonify({'status': 'error', 'message': 'Generate geometry first'}), 400

        chamfer = state['chamfer']
        rotation_number = float(data.get('rotation_number', 1.6180339887))
        depth = int(data.get('cf_depth', 8))

        repair = chamfer_engine.repair_manifold(chamfer['vertices'], chamfer['faces'])
        blend = cf_manifold.manifold_blend_field(state['boundary'], chamfer['chamfer_depth_mm'],
                                                  rotation_number=rotation_number, depth=depth)
        consistency = cf_manifold.check_manifold_consistency(repair['euler_characteristic'])

        state['repair'] = repair
        state['blend'] = blend

        return jsonify({'status': 'success', 'repair': repair, 'manifold_blend': blend,
                         'gauss_bonnet': consistency})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/run-bem', methods=['POST'])
def run_bem():
    """Run the boundary element method simulation"""
    try:
        data = request.json or {}
        if 'boundary' not in state:
            return jsonify({'status': 'error', 'message': 'Generate geometry first'}), 400

        pressure_kpa = float(data.get('pressure_kpa', 15.0))
        n_panels_target = int(data.get('n_panels_target', 96))

        result = bem_engine.run_bem_analysis(state['design'], state['boundary'],
                                              pressure_kpa=pressure_kpa, n_panels_target=n_panels_target)
        state['bem'] = result
        return jsonify({'status': 'success', 'bem': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/submit-hpc-job', methods=['POST'])
def submit_hpc_job():
    """Submit the BEM dense-solve workload to the simulated HPC cluster"""
    try:
        data = request.json or {}
        n_panels = int(state.get('bem', {}).get('n_panels', 96))
        nodes = int(data.get('nodes', 4))
        tasks_per_node = int(data.get('tasks_per_node', 16))
        parallel_fraction = float(data.get('parallel_fraction', 0.94))

        job = hpc.submit_job(n_panels, nodes=nodes, tasks_per_node=tasks_per_node,
                              parallel_fraction=parallel_fraction)
        curve = hpc.scaling_curve(n_panels, parallel_fraction=parallel_fraction)
        state['hpc_job'] = job
        state['hpc_curve'] = curve
        return jsonify({'status': 'success', 'job': job, 'scaling_curve': curve})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/nvqlink-accelerate', methods=['POST'])
def nvqlink_accelerate():
    """Run the NVQLink hybrid GPU-QPU acceleration simulation for the BEM solve"""
    try:
        bem = state.get('bem')
        if not bem:
            return jsonify({'status': 'error', 'message': 'Run BEM analysis first'}), 400

        dof = int(bem['dense_system_size'])
        condition_number = float(bem.get('condition_estimate', 1e3))

        link = nvqlink.link_status()
        accel = nvqlink.accelerate_bem_solve(dof, condition_number)
        state['nvqlink'] = accel
        return jsonify({'status': 'success', 'link_status': link, 'acceleration': accel})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/generate-preprint', methods=['POST'])
def generate_preprint():
    """Run the Nature-style preprint PDF generator and return the file name"""
    try:
        script_path = os.path.join(BASE_DIR, 'generate_nature_preprint_pelvic_bem.py')
        result = subprocess.run([sys.executable, script_path], cwd=BASE_DIR,
                                 capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return jsonify({'status': 'error', 'message': result.stderr}), 500
        pdf_name = 'Nature_Preprint_Pelvic_Floor_BEM_HPC_NVQLink.pdf'
        return jsonify({'status': 'success', 'pdf': pdf_name, 'log': result.stdout})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/preprint/<path:filename>')
def download_preprint(filename):
    return send_from_directory(BASE_DIR, filename, as_attachment=False)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5057))
    app.run(host='0.0.0.0', port=port, debug=True)
