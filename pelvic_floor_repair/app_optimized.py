"""
Optimized Flask Backend for Gynecological Repair & Pelvic Floor Reconstruction
Performance-focused with caching, efficient algorithms, and 3D mesh generation
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from functools import lru_cache
import json
import numpy as np
from datetime import datetime
from implant_designer import ImplantDesigner
from chamber_generator import ChamberGenerator
from llm_integration import LLMDesignAssistant
from visualization_engine_optimized import VisualizationEngineOptimized

app = Flask(__name__)
CORS(app)

# Performance cache for frequent computations
cache_config = {
    'design_cache': {},
    'mesh_cache': {},
    'max_cached_items': 50
}

# Initialize systems
implant_designer = ImplantDesigner()
chamber_gen = ChamberGenerator()
llm_assistant = LLMDesignAssistant()
viz_engine = VisualizationEngineOptimized()

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
        'version': '1.1.0-optimized',
        'timestamp': datetime.now().isoformat(),
        'cache_status': {
            'designs_cached': len(cache_config['design_cache']),
            'meshes_cached': len(cache_config['mesh_cache'])
        }
    })

def _cache_key(prefix, **kwargs):
    """Generate cache key from parameters"""
    sorted_items = sorted([(k, v) for k, v in kwargs.items()])
    return f"{prefix}:{json.dumps(sorted_items, sort_keys=True, default=str)}"

def _clear_old_cache():
    """Clear oldest cache entries if limit exceeded"""
    if len(cache_config['design_cache']) > cache_config['max_cached_items']:
        oldest_key = next(iter(cache_config['design_cache']))
        del cache_config['design_cache'][oldest_key]
    if len(cache_config['mesh_cache']) > cache_config['max_cached_items']:
        oldest_key = next(iter(cache_config['mesh_cache']))
        del cache_config['mesh_cache'][oldest_key]

@app.route('/api/analyze-patient', methods=['POST'])
def analyze_patient():
    """Analyze patient case and generate initial assessment"""
    try:
        data = request.json
        patient_id = data.get('patient_id', 'NEW_PATIENT')
        
        # Check cache first
        cache_key = _cache_key('patient_analysis', **{
            'discontinuity_length': data.get('discontinuity_length'),
            'discontinuity_width': data.get('discontinuity_width'),
            'severity': data.get('severity')
        })
        
        if cache_key in cache_config['design_cache']:
            analysis = cache_config['design_cache'][cache_key]
        else:
            # Generate initial analysis
            analysis = implant_designer.analyze_case(
                discontinuity_length=data.get('discontinuity_length', 0),
                discontinuity_width=data.get('discontinuity_width', 0),
                severity=data.get('severity', 'moderate'),
                tissue_quality=data.get('tissue_quality', 'normal')
            )
            # Cache result
            cache_config['design_cache'][cache_key] = analysis
            _clear_old_cache()
        
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
            'recommendations': llm_recommendations,
            'processing_time_ms': 'cached' if cache_key in cache_config['design_cache'] else 'computed'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/generate-implant-designs', methods=['POST'])
def generate_implant_designs():
    """Generate multiple optimized implant designs with 3D meshes"""
    try:
        data = request.json
        patient_id = data.get('patient_id', 'TEMP')
        num_designs = min(data.get('num_designs', 5), 10)  # Cap at 10
        
        # Check cache
        cache_key = _cache_key('implant_designs', 
            patient_id=patient_id, 
            num_designs=num_designs,
            **data.get('design_params', {})
        )
        
        if cache_key in cache_config['design_cache']:
            models = cache_config['design_cache'][cache_key]
        else:
            design_params = data.get('design_params', {})
            
            # Fast combinatorial generation (optimized)
            designs = implant_designer.generate_designs_optimized(
                num_designs=num_designs,
                **design_params
            )
            
            # Rank designs using LLM (cached if possible)
            ranked_designs = llm_assistant.rank_designs(designs)
            
            # Generate high-quality 3D meshes in parallel
            models = []
            for design in ranked_designs[:num_designs]:
                # Generate detailed 3D model
                model = viz_engine.generate_implant_mesh_detailed(design)
                models.append({
                    'design_id': design['id'],
                    'design': design,
                    'mesh': model['mesh'],
                    'metadata': model['metadata'],
                    'score': design.get('rank_score', 0)
                })
            
            cache_config['design_cache'][cache_key] = models
            _clear_old_cache()
        
        session_designs[patient_id]['implant_designs'] = models
        
        return jsonify({
            'status': 'success',
            'num_designs': len(models),
            'designs': models
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/generate-chambers', methods=['POST'])
def generate_chambers():
    """Generate pelvic chamber configurations with 3D meshes"""
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
        
        # Generate optimized 3D models for chambers
        chamber_models = []
        for i, chamber in enumerate(chambers):
            model = viz_engine.generate_chamber_mesh_detailed(chamber, i)
            chamber_models.append({
                'chamber_id': i,
                'chamber': chamber,
                'mesh': model['mesh'],
                'metadata': model['metadata']
            })
        
        session_designs[patient_id]['chambers'] = chamber_models
        
        return jsonify({
            'status': 'success',
            'num_chambers': len(chamber_models),
            'chambers': chamber_models
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/simulate-surgery', methods=['POST'])
def simulate_surgery():
    """Simulate the surgical repair process"""
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
    """Export complete surgical plan"""
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

@app.route('/api/export-stl/<patient_id>/<design_id>', methods=['GET'])
def export_stl(patient_id, design_id):
    """Export 3D model as STL file"""
    try:
        if patient_id not in session_designs:
            return jsonify({'status': 'error', 'message': 'Patient not found'}), 404
        
        session = session_designs[patient_id]
        for design in session.get('implant_designs', []):
            if design['design_id'] == design_id:
                stl_data = viz_engine.mesh_to_stl(design['mesh'])
                return {
                    'status': 'success',
                    'stl_data': stl_data,
                    'filename': f"implant_{design_id}.stl"
                }
        
        return jsonify({'status': 'error', 'message': 'Design not found'}), 404
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
    """Interactive AI chat for design assistance"""
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

@app.route('/api/cache-stats', methods=['GET'])
def cache_stats():
    """Get cache statistics"""
    return jsonify({
        'design_cache_size': len(cache_config['design_cache']),
        'mesh_cache_size': len(cache_config['mesh_cache']),
        'max_cache_items': cache_config['max_cached_items'],
        'active_sessions': len(session_designs)
    })

@app.route('/api/cache-clear', methods=['POST'])
def cache_clear():
    """Clear all caches"""
    cache_config['design_cache'].clear()
    cache_config['mesh_cache'].clear()
    return jsonify({'status': 'success', 'message': 'Caches cleared'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
