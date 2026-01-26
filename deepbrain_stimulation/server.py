"""
Flask Backend Server for DBS-PTSD & Dementia Treatment System
Provides REST API for circuit generation, AI optimization, quantum optimization, and safety validation
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('Agg')  # Optimize for server-side non-GUI rendering
import matplotlib.pyplot as plt

# Import our modules
from dbs_circuit_generator import DBSCircuitGenerator
from generative_ai_engine import GenerativeAIEngine
from ptsd_neural_model import PTSDNeuralModel
from safety_validator import SafetyValidator
from fea_simulator import DBSFEASimulator
from treatment_optimizer import TreatmentProtocolOptimizer

# Import dementia care modules
from dementia_neural_model import DementiaNeuralModel
from gemini_optimizer import GeminiQuantumOptimizer
from dementia_biomarkers import DementiaBiomarkerTracker
from ocd_neural_model import OCDNeuralModel
from ocd_quantum_optimizer import OCDQuantumOptimizer
from asd_neural_model import ASDNeuralRepairModel



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize lightweight components immediately
circuit_generator = DBSCircuitGenerator()
ai_engine = GenerativeAIEngine()
neural_model = PTSDNeuralModel()
safety_validator = SafetyValidator()
fea_simulator = DBSFEASimulator()
protocol_optimizer = TreatmentProtocolOptimizer(neural_model)
executor = ThreadPoolExecutor(max_workers=4)

# Lazy-load heavy components (quantum optimizers, neural models)
# These will be initialized on first use to speed up server startup
_dementia_model = None
_quantum_optimizer = None
_biomarker_tracker = None
_ocd_model = None
_ocd_quantum = None
_asd_model = None
# Global state
ai_models_trained = False

# Lazy initialization helpers
def get_dementia_model():
    global _dementia_model
    if _dementia_model is None:
        _dementia_model = DementiaNeuralModel(disease_duration_years=2.0)
    return _dementia_model

def get_quantum_optimizer():
    global _quantum_optimizer
    if _quantum_optimizer is None:
        _quantum_optimizer = GeminiQuantumOptimizer()
    return _quantum_optimizer

def get_biomarker_tracker():
    global _biomarker_tracker
    if _biomarker_tracker is None:
        _biomarker_tracker = DementiaBiomarkerTracker()
    return _biomarker_tracker

def get_ocd_model():
    global _ocd_model
    if _ocd_model is None:
        _ocd_model = OCDNeuralModel()
    return _ocd_model

def get_ocd_quantum():
    global _ocd_quantum
    if _ocd_quantum is None:
        _ocd_quantum = OCDQuantumOptimizer()
    return _ocd_quantum

def get_asd_model():
    global _asd_model
    if _asd_model is None:
        _asd_model = ASDNeuralRepairModel(severity='severe')
    if _asd_model is None:
        _asd_model = ASDNeuralRepairModel(severity='severe')
    return _asd_model









@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_trained': ai_models_trained,
        'components': {
            'circuit_generator': True,
            'ai_engine': True,
            'neural_model': True,
            'safety_validator': True,
            'dementia_model': True,
            'quantum_optimizer': True,
            'biomarker_tracker': True,
            'ocd_model': True
        },
        'quantum_available': False  # Don't initialize on health check
    })



# ==================== Circuit Generation Endpoints ====================

@app.route('/api/circuit/electrode-array', methods=['GET'])
def get_electrode_array():
    """Get electrode array schematic"""
    schematic = circuit_generator.generate_electrode_array_schematic()
    return jsonify(schematic)


@app.route('/api/circuit/pulse-generator', methods=['GET'])
def get_pulse_generator():
    """Get pulse generator schematic"""
    schematic = circuit_generator.generate_pulse_generator_schematic()
    return jsonify(schematic)


@app.route('/api/circuit/power-management', methods=['GET'])
def get_power_management():
    """Get power management schematic"""
    schematic = circuit_generator.generate_power_management_schematic()
    return jsonify(schematic)


@app.route('/api/circuit/safety-system', methods=['GET'])
def get_safety_system():
    """Get safety system schematic"""
    schematic = circuit_generator.generate_safety_system_schematic()
    return jsonify(schematic)


@app.route('/api/circuit/signal-processing', methods=['GET'])
def get_signal_processing():
    """Get signal processing schematic"""
    schematic = circuit_generator.generate_signal_processing_schematic()
    return jsonify(schematic)


@app.route('/api/circuit/complete', methods=['GET'])
def get_complete_circuit():
    """Get complete system schematic"""
    schematic = circuit_generator.generate_complete_system_schematic()
    return jsonify(schematic)


@app.route('/api/circuit/svg', methods=['GET'])
def get_circuit_svg():
    """Get SVG circuit diagram"""
    svg = circuit_generator.generate_svg_schematic()
    return svg, 200, {'Content-Type': 'image/svg+xml'}


@app.route('/api/circuit/export', methods=['POST'])
def export_circuit():
    """Export circuit schematics to files"""
    try:
        result = circuit_generator.export_schematics()
        return jsonify({
            'success': True,
            'files': {
                'json': result['json_path'],
                'svg': result['svg_path']
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== AI Engine Endpoints ====================

@app.route('/api/ai/train', methods=['POST'])
def train_ai_models():
    """Train all AI models"""
    global ai_models_trained
    
    data = request.json
    epochs = data.get('epochs', 50)
    
    try:
        # Train VAE
        vae_losses = ai_engine.train_vae(epochs=epochs)
        
        # Train GAN
        gan_losses = ai_engine.train_gan(epochs=epochs)
        
        ai_models_trained = True
        
        return jsonify({
            'success': True,
            'vae_final_loss': vae_losses[-1],
            'gan_final_d_loss': gan_losses['d_loss'][-1],
            'gan_final_g_loss': gan_losses['g_loss'][-1],
            'epochs': epochs
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai/generate/vae', methods=['POST'])
def generate_vae_parameters():
    """Generate parameters using VAE"""
    data = request.json
    num_samples = data.get('num_samples', 10)
    
    try:
        params = ai_engine.generate_vae_parameters(num_samples)
        return jsonify({
            'success': True,
            'parameters': params,
            'method': 'VAE'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai/generate/gan', methods=['POST'])
def generate_gan_parameters():
    """Generate parameters using GAN"""
    data = request.json
    num_samples = data.get('num_samples', 10)
    
    try:
        params = ai_engine.generate_gan_parameters(num_samples)
        return jsonify({
            'success': True,
            'parameters': params,
            'method': 'GAN'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai/optimize/rl', methods=['POST'])
def optimize_with_rl():
    """Optimize parameters using RL"""
    data = request.json
    initial_state = np.array(data.get('initial_state', [0.7, 0.6, 0.5, 0.6, 0.8, 0.3, 0.5, 0.7]))
    num_steps = data.get('num_steps', 100)
    
    try:
        trajectory = ai_engine.optimize_with_rl(initial_state, num_steps)
        return jsonify({
            'success': True,
            'trajectory': [state.tolist() for state in trajectory],
            'method': 'Reinforcement Learning'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== Neural Model Endpoints ====================

@app.route('/api/neural/state', methods=['GET'])
def get_neural_state():
    """Get current neural model state"""
    state = neural_model.export_state()
    return jsonify(state)


@app.route('/api/neural/simulate', methods=['POST'])
def simulate_stimulation():
    """Simulate DBS stimulation"""
    data = request.json
    
    try:
        result = neural_model.apply_dbs_stimulation(
            target_region=data['target_region'],
            amplitude_ma=data['amplitude_ma'],
            frequency_hz=data['frequency_hz'],
            pulse_width_us=data['pulse_width_us'],
            duration_s=data.get('duration_s', 1.0)
        )
        
        return jsonify({
            'success': True,
            'activity': result['activity'],
            'symptoms': {
                'hyperarousal': result['symptoms'].hyperarousal,
                're_experiencing': result['symptoms'].re_experiencing,
                'avoidance': result['symptoms'].avoidance,
                'negative_cognition': result['symptoms'].negative_cognition,
                'total_severity': result['symptoms'].total_severity()
            },
            'efficacy': result['efficacy'],
            'biomarkers': neural_model.get_biomarkers()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/neural/predict', methods=['POST'])
def predict_treatment():
    """Predict long-term treatment response"""
    data = request.json
    
    try:
        prediction = neural_model.predict_treatment_response(
            target_region=data['target_region'],
            amplitude_ma=data['amplitude_ma'],
            frequency_hz=data['frequency_hz'],
            pulse_width_us=data['pulse_width_us'],
            treatment_weeks=data.get('treatment_weeks', 12)
        )
        
        return jsonify({
            'success': True,
            'weekly_progression': prediction['weekly_progression'],
            'response_rate': prediction['response_rate'],
            'responder': prediction['responder'],
            'final_symptoms': {
                'hyperarousal': prediction['final_symptoms'].hyperarousal,
                're_experiencing': prediction['final_symptoms'].re_experiencing,
                'avoidance': prediction['final_symptoms'].avoidance,
                'negative_cognition': prediction['final_symptoms'].negative_cognition,
                'total_severity': prediction['final_symptoms'].total_severity()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/neural/optimize', methods=['POST'])
def optimize_parameters():
    """Optimize DBS parameters"""
    data = request.json
    target_region = data.get('target_region', 'amygdala')
    
    try:
        optimization = neural_model.optimize_parameters(target_region=target_region)
        return jsonify({
            'success': True,
            'best_parameters': optimization['best_parameters'],
            'best_efficacy': optimization['best_efficacy'],
            'top_results': optimization['all_results']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/neural/reset', methods=['POST'])
def reset_neural_model():
    """Reset neural model to baseline"""
    global neural_model
    neural_model = PTSDNeuralModel()
    return jsonify({'success': True, 'message': 'Neural model reset to baseline'})


# ==================== Safety Validation Endpoints ====================

@app.route('/api/safety/validate', methods=['POST'])
def validate_parameters():
    """Validate stimulation parameters"""
    data = request.json
    
    try:
        validation = safety_validator.validate_parameters(
            amplitude_ma=data['amplitude_ma'],
            frequency_hz=data['frequency_hz'],
            pulse_width_us=data['pulse_width_us'],
            electrode_area_cm2=data.get('electrode_area_cm2', 0.06),
            impedance_ohms=data.get('impedance_ohms', 1000)
        )
        
        return jsonify({
            'success': True,
            'validation': validation
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/safety/charge-balance', methods=['POST'])
def validate_charge_balance():
    """Validate charge balance"""
    data = request.json
    
    try:
        balance = safety_validator.validate_charge_balance(
            cathodic_charge_uc=data['cathodic_charge_uc'],
            anodic_charge_uc=data['anodic_charge_uc'],
            tolerance_percent=data.get('tolerance_percent', 1.0)
        )
        
        return jsonify({
            'success': True,
            'balance': balance
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/safety/thermal', methods=['POST'])
def validate_thermal():
    """Validate thermal safety"""
    data = request.json
    
    try:
        thermal = safety_validator.validate_thermal_safety(
            power_mw=data['power_mw'],
            duration_s=data['duration_s'],
            tissue_thermal_conductivity=data.get('tissue_thermal_conductivity', 0.5)
        )
        
        return jsonify({
            'success': True,
            'thermal': thermal
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/safety/biocompatibility', methods=['POST'])
def check_biocompatibility():
    """Check material biocompatibility"""
    data = request.json
    material = data.get('material', 'Platinum-Iridium')
    
    try:
        result = safety_validator.check_biocompatibility(material)
        return jsonify({
            'success': True,
            'biocompatibility': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/safety/compliance', methods=['POST'])
def check_compliance():
    """Check regulatory compliance"""
    data = request.json
    device_specs = data.get('device_specs', {})
    
    try:
        compliance = safety_validator.validate_regulatory_compliance(device_specs)
        return jsonify({
            'success': True,
            'compliance': compliance
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/safety/report', methods=['POST'])
def generate_safety_report():
    """Generate comprehensive safety report"""
    data = request.json
    
    try:
        report = safety_validator.generate_safety_report(
            parameters=data['parameters'],
            device_specs=data['device_specs']
        )
        
        return jsonify({
            'success': True,
            'report': report
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500




# ==================== Dementia Model Endpoints ====================

@app.route('/api/dementia/state', methods=['GET'])
def get_dementia_state():
    """Get current dementia model state"""
    state = get_dementia_model().export_state()
    return jsonify(state)


@app.route('/api/dementia/simulate', methods=['POST'])
def simulate_dementia_stimulation():
    """Simulate DBS effects on dementia"""
    data = request.json
    
    try:
        # Offload Dementia Step
        future = executor.submit(get_dementia_model().apply_dbs_stimulation, 
            target_region=data['target_region'],
            amplitude_ma=data['amplitude_ma'],
            frequency_hz=data['frequency_hz'],
            pulse_width_us=data['pulse_width_us'],
            duration_s=data.get('duration_s', 1.0)
        )
        result = future.result()
        
        return jsonify({
            'success': True,
            'activity': result['activity'],
            'cognitive_scores': {
                'mmse': result['cognitive_scores'].mmse,
                'moca': result['cognitive_scores'].moca,
                'memory_encoding': result['cognitive_scores'].memory_encoding,
                'memory_retrieval': result['cognitive_scores'].memory_retrieval,
                'disease_stage': result['cognitive_scores'].get_disease_stage()
            },
            'efficacy': result['efficacy']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dementia/predict', methods=['POST'])
def predict_dementia_treatment():
    """Predict long-term dementia treatment response"""
    data = request.json
    
    try:
        prediction = get_dementia_model().predict_treatment_response(
            target_region=data['target_region'],
            amplitude_ma=data['amplitude_ma'],
            frequency_hz=data['frequency_hz'],
            pulse_width_us=data['pulse_width_us'],
            treatment_months=data.get('treatment_months', 6)
        )
        
        return jsonify({
            'success': True,
            'monthly_progression': prediction['monthly_progression'],
            'response_rate': prediction['response_rate'],
            'responder': prediction['responder']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dementia/biomarkers', methods=['GET'])
def get_dementia_biomarkers():
    """Get dementia biomarkers"""
    biomarkers = get_dementia_model().get_biomarkers()
    tracker_summary = get_biomarker_tracker().get_biomarker_summary()
    
    return jsonify({
        'model_biomarkers': biomarkers,
        'tracker_summary': tracker_summary
    })


# ==================== OCD Model Endpoints ====================

@app.route('/api/ocd/simulate', methods=['POST'])
def simulate_ocd_dbs():
    """Simulate OCD DBS treatment on a single subject"""
    data = request.json
    
    # Create a fresh model instance for simulation
    sim_model = OCDNeuralModel()
    
    pre_gain = sim_model.calculate_cycle_gain()
    pre_ybocs = sim_model.calculate_ybocs()
    
    # Offload OCD simulation
    def _run_ocd_sim():
        sim_model.apply_dbs(
            target=data.get('target', 'caudate'),
            frequency=data.get('frequency', 130),
            amplitude=data.get('amplitude', 3.0)
        )
        return sim_model

    future = executor.submit(_run_ocd_sim)
    sim_model = future.result()
    
    post_gain = sim_model.calculate_cycle_gain()
    post_ybocs = sim_model.calculate_ybocs()
    
    return jsonify({
        'success': True,
        'pre': {'gain': pre_gain, 'ybocs': pre_ybocs},
        'post': {'gain': post_gain, 'ybocs': post_ybocs},
        'reduction_eff': (pre_ybocs - post_ybocs) / pre_ybocs * 100
    })


@app.route('/api/ocd/trial', methods=['POST'])
def run_ocd_trial():
    """Run a statistical clinical trial simulation for OCD DBS"""
    data = request.json
    
    n_subjects = data.get('n_subjects', 20)
    target = data.get('target', 'caudate')
    freq = data.get('frequency', 130)
    amp = data.get('amplitude', 3.0)
    
    # Use the model class method (or instance method we made)
    # We can use the global instance to run the trial method since it creates new subjects inside
    results = get_ocd_model().run_clinical_trial(n_subjects, target, freq, amp)
    
    return jsonify({
        'success': True,
        'results': results
    })


@app.route('/api/ocd/schematic', methods=['GET'])
def get_ocd_schematic():
    """Get OCD-specific circuit schematic and specs"""
    # Using the global circuit_generator instance
    # We rely on the method we just added
    
    try:
        schematic = circuit_generator.generate_ocd_specific_schematic()
        return jsonify({
            'success': True,
            'schematic': schematic
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ocd/optimize_quantum_fea', methods=['POST'])
def optimize_ocd_quantum():
    """Optimize OCD protocol using Quantum Surface Integrals and FEA"""
    data = request.json
    
    # In a full system, we would get the raw FEA array here.
    # For now, we accept summary stats and generating a synthetic field in the optimizer.
    fea_input = {
        'max_field': data.get('max_field', 1.5),
        'vta': data.get('vta', 0.8)
    }
    
    try:
        result = get_ocd_quantum().optimize_protocol_with_fea(fea_input)
        return jsonify({
            'success': True,
            'optimization': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== ASD Model Endpoints ====================

@app.route('/api/asd/optimize', methods=['POST'])
def optimize_asd_treatment():
    """Optimize ASD treatment using Gemini 3.0 with Quantum Surface Integrals & Continued Fractions"""
    try:
        data = request.json
        severity = data.get('severity', 'moderate')
        target = data.get('target', 'ACC')
        
        # Handle cases where frontend might send null/None for numeric values
        frequency = data.get('frequency')
        if frequency is None: frequency = 130.0
        else: frequency = float(frequency)
        
        amplitude = data.get('amplitude')
        if amplitude is None: amplitude = 2.5
        else: amplitude = float(amplitude)
        
        # Get or reinitialize model if severity changed
        global _asd_model
        if _asd_model is None or _asd_model.severity != severity:
            _asd_model = ASDNeuralRepairModel(severity=severity)
        
        # Run Repair Session with parameters
        def _run_asd_repair():
             return _asd_model.simulate_repair_session(
                target_region=target,
                frequency=frequency,
                amplitude=amplitude
            )
        
        future = executor.submit(_run_asd_repair)
        result = future.result()
        
        # Get plotting data
        plot_data = _asd_model.get_plotting_data()
        
        return jsonify({
            'success': True,
            'optimization': result,
            'plot_data': plot_data
        })
    except Exception as e:
        print(f"Error in ASD optimization: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/asd/plotting_data', methods=['GET'])
def get_asd_plot_data():
    """Get connectivity matrices for ASD plotting"""
    try:
        return jsonify(get_asd_model().get_plotting_data())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500








from sad_neural_model import SADNeuralModel

# ... (imports continue) ...

# Lazy-load global
_sad_model = None

def get_sad_model():
    global _sad_model
    if _sad_model is None:
        _sad_model = SADNeuralModel()
    return _sad_model


# ==================== SAD Model Endpoints ====================

@app.route('/api/sad/treat', methods=['POST'])
def treat_sad():
    """Simulate SAD Treatment with DBS and Statistical ML"""
    try:
        data = request.json
        target = data.get('target', 'Lateral Habenula')
        freq = float(data.get('frequency', 135.0))
        amp = float(data.get('amplitude', 2.5))
        duration = float(data.get('duration', 1.0))
        
        result = get_sad_model().simulate_treatment_session(target, freq, amp, duration)
        
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== Quantum Optimization Endpoints ====================

@app.route('/api/quantum/optimize/vqe', methods=['POST'])
def optimize_with_vqe():
    """Optimize parameters using quantum VQE"""
    data = request.json
    
    try:
        initial_params = data['initial_params']
        bounds = data['bounds']
        
        # Create objective function (simplified)
        def objective(params):
            # Simulate with dementia model
            result = get_dementia_model().apply_dbs_stimulation(
                target_region=params.get('target_region', 'nucleus_basalis'),
                amplitude_ma=params['amplitude_ma'],
                frequency_hz=params['frequency_hz'],
                pulse_width_us=params['pulse_width_us']
            )
            # Negative efficacy (minimize)
            return -result['efficacy']
        
        future = executor.submit(
            get_quantum_optimizer().optimize_vqe,
            objective_function=objective,
            initial_params=initial_params,
            bounds=bounds,
            max_iterations=data.get('max_iterations', 50)
        )
        result = future.result()
        
        return jsonify({
            'success': True,
            'optimal_parameters': result.optimal_parameters,
            'energy': result.energy,
            'iterations': result.iterations,
            'method': result.method,
            'gemini_insights': result.gemini_insights,
            'confidence_score': result.confidence_score
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/quantum/info', methods=['GET'])
def get_quantum_optimizer_info():
    """Get Gemini optimizer information"""
    info = get_quantum_optimizer().get_optimizer_info()
    return jsonify(info)


@app.route('/api/quantum/compare', methods=['POST'])
def compare_quantum_classical():
    """Compare quantum vs classical optimization"""
    data = request.json
    
    try:
        initial_params = data['initial_params']
        bounds = data['bounds']
        
        def objective(params):
            result = get_dementia_model().apply_dbs_stimulation(
                target_region='nucleus_basalis',
                amplitude_ma=params['amplitude_ma'],
                frequency_hz=params['frequency_hz'],
                pulse_width_us=params['pulse_width_us']
            )
            return -result['efficacy']
        
        comparison = get_quantum_optimizer().compare_quantum_classical(
            objective_function=objective,
            initial_params=initial_params,
            bounds=bounds
        )
        
        return jsonify({
            'success': True,
            'comparison': {
                'gemini_energy': comparison.get('gemini_energy'),
                'classical_energy': comparison.get('classical_energy'),
                'gemini_iterations': comparison.get('gemini_iterations'),
                'classical_iterations': comparison.get('classical_iterations'),
                'speedup': comparison.get('speedup'),
                'quality_improvement': comparison.get('quality_improvement'),
                'gemini_advantage': comparison.get('gemini_advantage'),
                'gemini_insights': comparison.get('gemini_insights')
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== Cognitive Assessment Endpoints ====================

@app.route('/api/cognitive/mmse', methods=['POST'])
def calculate_mmse():
    """Calculate MMSE score"""
    data = request.json
    responses = data.get('responses', {})
    
    score = biomarker_tracker.calculate_mmse(responses)
    return jsonify({'mmse_score': score})


@app.route('/api/cognitive/moca', methods=['POST'])
def calculate_moca():
    """Calculate MoCA score"""
    data = request.json
    responses = data.get('responses', {})
    
    score = biomarker_tracker.calculate_moca(responses)
    return jsonify({'moca_score': score})


@app.route('/api/cognitive/timeline', methods=['GET'])
def get_cognitive_timeline():
    """Get cognitive assessment timeline"""
    trajectory = biomarker_tracker.get_cognitive_trajectory()
    return jsonify(trajectory)


# ==================== FEA Simulation Endpoints ====================

@app.route('/api/fea/simulate', methods=['POST'])
def run_fea_simulation():
    """Run Finite Element Analysis for Field Distribution"""
    data = request.json
    config = data.get('electrode_config', {'voltages': {'c1': -3.0}})
    
    try:
        fea_simulator.generate_tissue_model()
        phi, e_field, vta = fea_simulator.solve_electric_field(config)
        
        # Generate Visuals
        phi_img = fea_simulator.generate_heatmap_plot(phi, "Electrical Potential ($\phi$)")
        efield_img = fea_simulator.generate_heatmap_plot(e_field, "E-Field Magnitude (|E|)")
        vta_img = fea_simulator.generate_heatmap_plot(vta, "Volume of Tissue Activated (VTA)")
        
        return jsonify({
            'success': True,
            'plots': {
                'potential': phi_img,
                'e_field': efield_img,
                'vta': vta_img
            },
            'max_field': float(np.max(e_field))
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/fea/optimize', methods=['POST'])
def optimize_fea_platform():
    """Optimize Stimulation Settings based on Field Models"""
    data = request.json
    target = data.get('target_coords', [32, 32])
    
    try:
        best_config, detailed_results = fea_simulator.optimize_stimulation(tuple(target))
        return jsonify({
            'success': True,
            'recommended_config': best_config,
            'optimization_log': detailed_results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== Treatment Protocol Endpoints ====================

@app.route('/api/protocol/optimize', methods=['POST'])
def optimize_protocol():
    """Generate optimized treatment protocol sequence"""
    data = request.json
    target_zones = data.get('target_zones', ['amygdala'])
    duration = data.get('duration_weeks', 12)
    
    try:
        result = protocol_optimizer.generate_protocol_sequence(target_zones, duration)
        return jsonify({
            'success': True,
            'protocol': result['protocol'],
            'projected_improvement': result['final_projected_symptom_reduction'],
            'bem_data': result.get('bem_data'),
            'connectome_data': result.get('connectome_data')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/protocol/bem_simulation', methods=['POST'])
def run_dedicated_bem():
    """Run dedicated BEM simulation for cortical surface"""
    try:
        data = request.json or {}
        targets = data.get('targets', ['amygdala'])
        result = protocol_optimizer.generate_bem_surface_data(targets)
        return jsonify({
            'success': True,
            'bem_data': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/protocol/simulate_outcome', methods=['POST'])
def simulate_protocol_outcome():
    """Simulate weekly progress of a protocol"""
    data = request.json
    protocol = data.get('protocol', [])
    
    # Infer condition or get from request
    condition = 'ptsd'
    if data.get('condition') == 'dementia':
        condition = 'dementia'
    else:
        # Auto-detect if any dementia targets
        for stage in protocol:
            if stage.get('target_focus') in ['nucleus_basalis', 'fornix']:
                condition = 'dementia'
                break
    
    try:
        weeks = protocol_optimizer.simulate_long_term_outcome(protocol, condition=condition)
        return jsonify({
            'success': True,
            'timeline': weeks,
            'condition': condition
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== Utility Endpoints ====================


# ==================== Static File Serving ====================

@app.route('/')
def serve_index():
    """Serve the main HTML file"""
    file_path = os.path.join(os.path.dirname(__file__), 'index.html')
    return send_file(file_path)

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, etc.)"""
    # Prevent API routes from falling through to static serving (which would cause 405 if POST)
    if filename.startswith('api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
        
    try:
        file_path = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/brain-regions', methods=['GET'])
def get_brain_regions():
    """Get available brain regions for targeting"""
    return jsonify({
        'regions': [
            {
                'id': 'amygdala',
                'name': 'Basolateral Amygdala (BLA)',
                'description': 'Fear processing and emotional memory',
                'coordinates': {'x': -20, 'y': -5, 'z': -15}
            },
            {
                'id': 'hippocampus',
                'name': 'Hippocampus (CA1)',
                'description': 'Memory consolidation and context',
                'coordinates': {'x': -25, 'y': -20, 'z': -10}
            },
            {
                'id': 'vmPFC',
                'name': 'Ventromedial Prefrontal Cortex',
                'description': 'Fear extinction and emotional regulation',
                'coordinates': {'x': 0, 'y': 45, 'z': -10}
            },
            {
                'id': 'hypothalamus',
                'name': 'Hypothalamus (HPA axis)',
                'description': 'Stress response and arousal',
                'coordinates': {'x': 0, 'y': -5, 'z': -10}
            }
        ]
    })


@app.route('/api/parameter-ranges', methods=['GET'])
def get_parameter_ranges():
    """Get safe parameter ranges"""
    return jsonify({
        'amplitude_ma': {'min': 0.5, 'max': 8.0, 'recommended': 3.0},
        'frequency_hz': {'min': 20, 'max': 185, 'recommended': 130, 'common': [60, 130, 185]},
        'pulse_width_us': {'min': 60, 'max': 210, 'recommended': 90},
        'duty_cycle': {'min': 0.1, 'max': 0.9, 'recommended': 0.5}
    })


if __name__ == '__main__':
    print("="*60)
    print("DBS-PTSD Treatment System Backend Server")
    print("="*60)
    print("\nStarting server on http://localhost:5002")
    print("\nAvailable endpoints:")
    print("  Circuit Generation: /api/circuit/*")
    print("  AI Engine: /api/ai/*")
    print("  Neural Model: /api/neural/*")
    print("  Safety Validation: /api/safety/*")
    print("\n[!] FOR RESEARCH AND EDUCATIONAL USE ONLY")
    print("="*60)
    print("\nOpen http://localhost:5002 in your browser")
    
    # Run with threading enabled and debug off for performance
    app.run(debug=False, host='0.0.0.0', port=5002, threaded=True)
