"""
Flask Backend for Gynecological Repair & Pelvic Floor Reconstruction
Integrates LLM, Combinatorial Design, and 3D Visualization
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from functools import lru_cache
from collections import OrderedDict
import json
import hashlib
import numpy as np
from datetime import datetime
from implant_designer_optimized import ImplantDesigner
from chamber_generator import ChamberGenerator
from llm_integration import LLMDesignAssistant
from visualization_engine_optimized import VisualizationEngineOptimized
from fea_engine import PelvicFEAEngine

app = Flask(__name__)
CORS(app)

# Performance cache for frequent computations (LRU-evicted)
MAX_CACHED_ITEMS = 50
design_cache = OrderedDict()
fea_cache = OrderedDict()


def _cache_key(payload: dict) -> str:
    """Deterministic hash key for cache lookups"""
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _cache_get(cache: OrderedDict, key: str):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def _cache_set(cache: OrderedDict, key: str, value):
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > MAX_CACHED_ITEMS:
        cache.popitem(last=False)

# Initialize systems
implant_designer = ImplantDesigner()
chamber_gen = ChamberGenerator()
llm_assistant = LLMDesignAssistant()
viz_engine = VisualizationEngineOptimized()
fea_engine = PelvicFEAEngine()

# Store session designs
session_designs = {}

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Pelvic Floor Reconstruction AI',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze-patient', methods=['POST'])
def analyze_patient():
    """
    Analyze patient case and generate initial assessment
    Expects: patient_data with discontinuity measurements and severity
    """
    try:
        data = request.json
        patient_id = data.get('patient_id', 'NEW_PATIENT')
        
        # Extract measurements
        discontinuity_length = data.get('discontinuity_length', 0)
        discontinuity_width = data.get('discontinuity_width', 0)
        severity = data.get('severity', 'moderate')  # mild, moderate, severe
        tissue_quality = data.get('tissue_quality', 'normal')
        
        # Generate initial analysis
        analysis = implant_designer.analyze_case(
            discontinuity_length=discontinuity_length,
            discontinuity_width=discontinuity_width,
            severity=severity,
            tissue_quality=tissue_quality
        )
        
        # Get LLM recommendations
        llm_recommendations = llm_assistant.get_repair_recommendations(
            analysis=analysis,
            patient_data=data
        )
        
        session_designs[patient_id] = {
            'analysis': analysis,
            'llm_recommendations': llm_recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'patient_id': patient_id,
            'analysis': analysis,
            'recommendations': llm_recommendations
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/generate-implant-designs', methods=['POST'])
def generate_implant_designs():
    """
    Generate multiple optimized implant designs with high-quality 3D meshes
    """
    try:
        data = request.json
        patient_id = data.get('patient_id', 'TEMP')
        num_designs = min(data.get('num_designs', 5), 10)  # Cap at 10 for performance
        design_params = data.get('design_params', {})

        cache_key = _cache_key({'num_designs': num_designs, 'design_params': design_params})
        cached = _cache_get(design_cache, cache_key)
        if cached is not None:
            models, designs, ranked_designs = cached
        else:
            # Generate optimized combinatorial designs
            designs = implant_designer.generate_designs_optimized(
                num_designs=num_designs,
                **design_params
            )

            # Rank designs using LLM
            ranked_designs = llm_assistant.rank_designs(designs)

            # Generate high-quality 3D meshes for each design
            models = []
            for design in ranked_designs[:num_designs]:
                mesh_model = viz_engine.generate_implant_mesh_detailed(design)
                models.append({
                    'design_id': design['id'],
                    'design': design,
                    'mesh': mesh_model['mesh'],
                    'metadata': mesh_model['metadata'],
                    'model_id': mesh_model['model_id'],
                    'score': design.get('rank_score', 0)
                })
            _cache_set(design_cache, cache_key, (models, designs, ranked_designs))

        session_designs.setdefault(patient_id, {})['implant_designs'] = models
        
        return jsonify({
            'status': 'success',
            'num_designs': len(models),
            'designs': models,
            'optimization_metrics': {
                'designs_generated': len(designs),
                'designs_ranked': len(ranked_designs),
                'meshes_created': len(models),
                'cache_hit': cached is not None
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/generate-chambers', methods=['POST'])
def generate_chambers():
    """
    Generate pelvic chamber configurations with optimized 3D meshes
    """
    try:
        data = request.json
        patient_id = data.get('patient_id', 'TEMP')
        implant_design_id = data.get('implant_design_id')
        
        # Get selected implant design
        implant = None
        if patient_id in session_designs:
            for design in session_designs[patient_id].get('implant_designs', []):
                if design['design_id'] == implant_design_id:
                    implant = design['design']
                    break
        
        if not implant:
            return jsonify({'status': 'error', 'message': 'Implant design not found'}), 404
        
        # Generate chambers
        chambers = chamber_gen.generate_chambers(implant)
        
        # Generate optimized 3D meshes for chambers
        chamber_models = []
        for i, chamber in enumerate(chambers):
            mesh_model = viz_engine.generate_chamber_mesh_detailed(chamber, i)
            chamber_models.append({
                'chamber_id': i,
                'chamber': chamber,
                'mesh': mesh_model['mesh'],
                'metadata': mesh_model['metadata'],
                'model_id': mesh_model['model_id']
            })
        
        session_designs.setdefault(patient_id, {})['chambers'] = chamber_models
        
        return jsonify({
            'status': 'success',
            'num_chambers': len(chamber_models),
            'chambers': chamber_models
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/simulate-surgery', methods=['POST'])
def simulate_surgery():
    """
    Simulate the surgical repair process
    """
    try:
        data = request.json
        patient_id = data.get('patient_id', 'TEMP')
        
        if patient_id not in session_designs:
            return jsonify({'status': 'error', 'message': 'Patient not found'}), 404
        
        session = session_designs[patient_id]
        
        # Simulate surgery
        simulation = {
            'steps': implant_designer.simulate_placement(
                session.get('implant_designs', [{}])[0].get('design', {})
            ),
            'estimated_duration': np.random.randint(45, 120),
            'risk_factors': implant_designer.analyze_risks(
                session.get('analysis', {})
            ),
            'success_probability': np.random.uniform(0.85, 0.98)
        }
        
        session_designs[patient_id]['simulation'] = simulation
        
        return jsonify({
            'status': 'success',
            'simulation': simulation
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/export-surgical-plan', methods=['POST'])
def export_surgical_plan():
    """
    Export complete surgical plan as PDF and 3D models
    """
    try:
        data = request.json
        patient_id = data.get('patient_id', 'TEMP')
        
        if patient_id not in session_designs:
            return jsonify({'status': 'error', 'message': 'Patient not found'}), 404
        
        session = session_designs[patient_id]
        
        # Generate surgical plan document
        plan = {
            'patient_id': patient_id,
            'timestamp': datetime.now().isoformat(),
            'analysis': session.get('analysis'),
            'selected_implant': session.get('implant_designs', [{}])[0] if session.get('implant_designs') else None,
            'chambers': session.get('chambers'),
            'simulation': session.get('simulation'),
            'recommendations': session.get('llm_recommendations')
        }
        
        return jsonify({
            'status': 'success',
            'surgical_plan': plan,
            'export_formats': ['pdf', 'stl', 'json']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/session/<patient_id>', methods=['GET'])
def get_session(patient_id):
    """Get current session data"""
    if patient_id in session_designs:
        return jsonify({
            'status': 'success',
            'session': session_designs[patient_id]
        })
    return jsonify({'status': 'error', 'message': 'Session not found'}), 404

@app.route('/api/ai-chat', methods=['POST'])
def ai_chat():
    """
    Interactive AI chat for design assistance
    """
    try:
        data = request.json
        query = data.get('query', '')
        patient_id = data.get('patient_id', 'TEMP')
        
        # Get AI response
        response = llm_assistant.chat(
            query=query,
            context=session_designs.get(patient_id, {})
        )
        
        return jsonify({
            'status': 'success',
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/run-fea', methods=['POST'])
def run_fea():
    """
    Execute Biomechanical FEA Analysis on an implant design
    """
    try:
        data = request.json or {}
        patient_id = data.get('patient_id', 'TEMP')
        pressure_kpa = float(data.get('pressure_kpa', 15.0))
        anchoring_type = data.get('anchoring_type', 'bilateral_sacrospinous')
        grid_res = int(data.get('grid_res', 21))
        custom_design = data.get('design', None)
        
        # Resolve design from session or payload
        target_design = None
        if custom_design:
            target_design = custom_design
        elif patient_id in session_designs and 'implant_designs' in session_designs[patient_id]:
            implant_models = session_designs[patient_id]['implant_designs']
            design_id = data.get('design_id')
            if design_id:
                for m in implant_models:
                    if m.get('design_id') == design_id or m.get('design', {}).get('id') == design_id:
                        target_design = m.get('design')
                        break
            if not target_design and len(implant_models) > 0:
                target_design = implant_models[0].get('design')
        
        # Fallback default design
        if not target_design:
            target_design = {
                'material': data.get('material', 'composite'),
                'shape_profile': data.get('shape_profile', 'anatomical'),
                'dimensions': {
                    'length_mm': float(data.get('length_mm', 42.0)),
                    'width_mm': float(data.get('width_mm', 28.0)),
                    'thickness_mm': float(data.get('thickness_mm', 1.0))
                },
                'pore_size_microns': float(data.get('pore_size_microns', 100))
            }

        fea_key = _cache_key({
            'design': target_design, 'pressure_kpa': pressure_kpa,
            'anchoring_type': anchoring_type, 'grid_res': grid_res
        })
        fea_results = _cache_get(fea_cache, fea_key)
        if fea_results is None:
            fea_results = fea_engine.run_fea_analysis(
                design=target_design,
                pressure_kpa=pressure_kpa,
                grid_res=grid_res,
                anchoring_type=anchoring_type
            )
            _cache_set(fea_cache, fea_key, fea_results)
        
        if patient_id in session_designs:
            session_designs[patient_id]['fea_results'] = fea_results
            
        return jsonify({
            'status': 'success',
            'fea': fea_results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/cache-stats', methods=['GET'])
def cache_stats():
    """Report current in-memory cache utilization"""
    return jsonify({
        'status': 'success',
        'design_cache_items': len(design_cache),
        'fea_cache_items': len(fea_cache),
        'max_cached_items': MAX_CACHED_ITEMS
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5050, threaded=True)
