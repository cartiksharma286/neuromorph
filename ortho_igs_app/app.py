from flask import Flask, jsonify, send_from_directory, request
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from gemini_optimizer import GeminiKneeOptimizer
from health_economics import HealthEconomicsEngine
from robot_kinematics import OrthoRobotController

app = Flask(__name__, static_folder='frontend', static_url_path='')

optimizer = GeminiKneeOptimizer()
economics = HealthEconomicsEngine()
robot = OrthoRobotController()

@app.route('/')
def home():
    return send_from_directory('frontend', 'index.html')

@app.route('/api/optimize', methods=['GET'])
def run_optimization():
    # Mock constraints for demo
    constraints = {'varus_valgus_deformity': 5.0}
    try:
        result = optimizer.optimize_surgical_plan(constraints)
        return jsonify({
            'status': 'success',
            'result': {
                'parameters': result.optimal_parameters,
                'energy': result.energy,
                'method': result.method
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/economics', methods=['GET'])
def get_economics():
    surgery_type = request.args.get('type', 'knee')
    try:
        data = economics.generate_cost_effectiveness_data(surgery_type=surgery_type)
        return jsonify({
            'status': 'success',
            'data': data
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/economics/cost', methods=['GET'])
def get_implant_cost():
    try:
        size = float(request.args.get('size', 1.0))
        complexity = request.args.get('complexity', 'standard')
        cost_data = economics.estimate_production_cost(size_param=size, complexity=complexity)
        return jsonify({
            'status': 'success',
            'data': cost_data
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/geometry/genai', methods=['GET'])
def get_genai_geometry():
    try:
        # Generative AI / LLM Rendering
        geometry = optimizer.generate_genai_geometry(size_param=1.0)
        return jsonify({
            'status': 'success',
            'type': 'Generative AI & LLM',
            'data': geometry
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/geometry/nvqlink', methods=['GET'])
def get_nvqlink_geometry():
    try:
        # NVQLink Rendering
        geometry = optimizer.generate_nvqlink_geometry(size_param=1.0)
        return jsonify({
            'status': 'success',
            'type': 'NVQLink Optimized',
            'data': geometry
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/geometry/hip/nvqlink', methods=['GET'])
def get_hip_nvqlink():
    try:
        # NVQLink Clean Parametric
        # Simulate optimization returning a specific size
        geometry = optimizer.generate_hip_nvqlink_geometry(size_param=1.2)
        return jsonify({
            'status': 'success',
            'type': 'NVQLink Hip',
            'data': geometry
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/geometry/hip/genai', methods=['GET'])
def get_hip_genai():
    try:
        # GenAI Organic/Porous
        geometry = optimizer.generate_hip_genai_geometry(size_param=1.2)
        return jsonify({
            'status': 'success',
            'type': 'GenAI Hip',
            'data': geometry
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- New Workflow Endpoints ---

@app.route('/api/workflow/acquire', methods=['GET'])
def run_acquisition():
    try:
        data = optimizer.simulate_image_acquisition()
        return jsonify({'status': 'success', 'data': data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/workflow/register', methods=['POST'])
def run_registration():
    try:
        data = optimizer.perform_registration([])
        return jsonify({'status': 'success', 'data': data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/workflow/track', methods=['GET'])
def run_tracking():
    try:
        data = optimizer.track_tools()
        return jsonify({'status': 'success', 'data': data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/planning/generate', methods=['GET'])
def run_gemini_planning():
    try:
        data = optimizer.generate_resction_plan()
        return jsonify({'status': 'success', 'data': data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/planning/optimize_fit', methods=['POST'])
def run_implant_fit():
    try:
        constraints = request.json or {}
        data = optimizer.optimize_implant_fit(constraints)
        return jsonify({'status': 'success', 'data': data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/robot/trajectory', methods=['POST'])
def get_robot_trajectory():
    try:
        cut_type = request.json.get('cut_type', 'distal')
        data = optimizer.genai_robot_trajectory(cut_type)
        return jsonify({'status': 'success', 'data': data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/workflow/resection', methods=['POST'])
def run_resection():
    try:
        # Pass plan from request if needed
        req_data = request.json or {}
        profile_type = req_data.get('profile', 'CR')
        result = optimizer.simulate_resection_process(plan_type=profile_type)
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/workflow/balancing', methods=['POST'])
def run_balancing():
    try:
        # Full kinematic curve
        result = optimizer.generate_kinematics_simulation()
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/workflow/implant', methods=['POST'])
def run_implant():
    try:
        result = optimizer.simulate_implant_insertion()
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
         return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/workflow/postop', methods=['GET'])
def get_postop_data():
    try:
        # Generate 5000 procedure stats
        result = optimizer.simulate_postop_analytics()
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/robot/resect', methods=['POST'])
def run_robot_resection():
    try:
        plan = request.json or {}
        result = robot.execute_knee_resection(plan)
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/robot/status', methods=['GET'])
def get_robot_status():
    return jsonify({
        'status': robot.status,
        'serial_joints': robot.serial_arm.joint_angles,
        'parallel_base': robot.parallel_arm.base_radius
    })



if __name__ == '__main__':
    print("Starting NVQLink Orthopedic Server...")
    print("Access the dashboard at http://localhost:5000")
    app.run(port=5000, debug=True)
