"""
Flask Server for Cell Tracking and Differentiation System
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import json
import torch

from tracking_engine import CellDetector, CellTracker, FeatureExtractor, TrackingVisualizer
from heuristic_algorithms import (GeneticTrackingOptimizer, ParticleSwarmOptimizer,
                                  AntColonyLineage, HybridOptimizer)
from parametric_learning import (CellStateModel, DifferentiationModel,
                                 ParameterEstimator, TrajectoryAnalyzer)
from generative_models import MorphologyVAE, TrajectoryGAN, DifferentiationPredictor, TrajectoryGenerator
from utils import image_to_base64, generate_sample_data

app = Flask(__name__, static_folder='.')
CORS(app)

# Global state
current_images = []
current_tracks = []
detector = CellDetector()
tracker = CellTracker()
feature_extractor = FeatureExtractor()
visualizer = TrackingVisualizer()


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


@app.route('/api/upload', methods=['POST'])
def upload_data():
    """Upload time-lapse microscopy data"""
    global current_images
    
    try:
        data = request.json
        
        if 'use_sample' in data and data['use_sample']:
            # Generate sample data
            current_images = generate_sample_data(n_frames=30, n_cells=15)
            return jsonify({
                'success': True,
                'n_frames': len(current_images),
                'message': 'Sample data generated successfully'
            })
        
        # Handle uploaded images
        if 'images' in data:
            current_images = []
            for img_data in data['images']:
                # Decode base64 image
                img_bytes = base64.b64decode(img_data.split(',')[1])
                img = Image.open(BytesIO(img_bytes))
                img_array = np.array(img)
                current_images.append(img_array)
            
            return jsonify({
                'success': True,
                'n_frames': len(current_images),
                'message': 'Images uploaded successfully'
            })
        
        return jsonify({'success': False, 'message': 'No data provided'}), 400
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/track', methods=['POST'])
def track_cells():
    """Run cell tracking algorithm"""
    global current_tracks, tracker
    
    try:
        if len(current_images) == 0:
            return jsonify({'success': False, 'message': 'No images loaded'}), 400
        
        # Reset tracker
        tracker = CellTracker()
        all_detections = []
        
        # Detect cells in each frame
        for frame_idx, image in enumerate(current_images):
            cells = detector.detect(image, frame_idx)
            all_detections.append(cells)
            tracker.update(cells)
        
        current_tracks = tracker.tracks
        
        # Generate visualization for first frame
        first_frame_viz = visualizer.draw_tracks(current_images[0], current_tracks, 0)
        viz_base64 = image_to_base64(first_frame_viz)
        
        # Get statistics
        active_tracks = [t for t in current_tracks if t.active]
        
        return jsonify({
            'success': True,
            'n_tracks': len(current_tracks),
            'n_active_tracks': len(active_tracks),
            'n_frames': len(current_images),
            'visualization': viz_base64,
            'tracks': [{
                'id': t.track_id,
                'length': len(t.cells),
                'active': t.active
            } for t in current_tracks[:20]]  # Limit to first 20 for display
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/get_frame/<int:frame_idx>', methods=['GET'])
def get_frame(frame_idx):
    """Get specific frame with tracking overlay"""
    try:
        if frame_idx >= len(current_images):
            return jsonify({'success': False, 'message': 'Frame index out of range'}), 400
        
        frame_viz = visualizer.draw_tracks(current_images[frame_idx], current_tracks, frame_idx)
        viz_base64 = image_to_base64(frame_viz)
        
        return jsonify({
            'success': True,
            'frame_idx': frame_idx,
            'visualization': viz_base64
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/optimize/tracking', methods=['POST'])
def optimize_tracking():
    """Optimize tracking using genetic algorithms"""
    try:
        data = request.json
        n_generations = data.get('n_generations', 50)
        
        if len(current_tracks) == 0:
            return jsonify({'success': False, 'message': 'No tracks available'}), 400
        
        # Extract features from last frame
        last_frame_cells = [t.cells[-1] for t in current_tracks if t.active and t.cells]
        
        if len(last_frame_cells) < 2:
            return jsonify({'success': False, 'message': 'Not enough cells for optimization'}), 400
        
        cell_features = feature_extractor.extract_batch_features(last_frame_cells)
        track_features = cell_features.copy()  # Simplified
        
        # Create motion coherence matrix
        motion_coherence = np.random.rand(len(cell_features), len(track_features))
        
        # Run genetic algorithm
        ga_optimizer = GeneticTrackingOptimizer(len(cell_features), len(track_features))
        best_assignment, fitness_history = ga_optimizer.optimize(
            cell_features, track_features, motion_coherence, n_generations=n_generations
        )
        
        return jsonify({
            'success': True,
            'best_fitness': float(fitness_history[-1]),
            'fitness_history': [float(f) for f in fitness_history],
            'n_generations': len(fitness_history),
            'assignment': best_assignment.tolist()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/optimize/parameters', methods=['POST'])
def optimize_parameters():
    """Tune parameters using PSO"""
    try:
        data = request.json
        max_iter = data.get('max_iter', 50)
        
        # Define objective function (minimize tracking error)
        def objective(params):
            # Simplified objective - in practice would re-run tracking
            return np.sum((params - 0.5) ** 2)
        
        # Parameter bounds
        bounds = [(0.1, 0.9), (0.1, 0.9), (0.5, 2.0), (0.1, 0.5), (0.1, 0.9)]
        
        # Run PSO
        pso_optimizer = ParticleSwarmOptimizer(n_particles=30, n_dimensions=5)
        best_params, cost_history = pso_optimizer.optimize(objective, bounds, max_iter=max_iter)
        
        return jsonify({
            'success': True,
            'best_parameters': best_params.tolist(),
            'cost_history': [float(c) for c in cost_history],
            'final_cost': float(cost_history[-1])
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/optimize/lineage', methods=['POST'])
def optimize_lineage():
    """Reconstruct lineage using ACO"""
    try:
        if len(current_tracks) == 0:
            return jsonify({'success': False, 'message': 'No tracks available'}), 400
        
        # Extract features
        all_cells = []
        for track in current_tracks:
            all_cells.extend(track.cells)
        
        if len(all_cells) < 5:
            return jsonify({'success': False, 'message': 'Not enough cells'}), 400
        
        cell_features = feature_extractor.extract_batch_features(all_cells[:20])  # Limit for demo
        
        # Simulate division events (in practice would detect from data)
        division_events = [0, 1, 2]
        
        data = request.json
        n_iterations = data.get('iterations', 30)
        
        # Run ACO
        aco_optimizer = AntColonyLineage(n_cells=len(cell_features), n_ants=20)
        best_lineage, quality_history = aco_optimizer.optimize(
            cell_features, division_events, n_iterations=n_iterations
        )
        
        return jsonify({
            'success': True,
            'lineage_edges': [(int(p), int(d)) for p, d in best_lineage],
            'quality_history': [float(q) for q in quality_history],
            'final_quality': float(quality_history[-1])
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/differentiation/analyze', methods=['POST'])
def analyze_differentiation():
    """Analyze differentiation trajectories"""
    try:
        if len(current_tracks) == 0:
            return jsonify({'success': False, 'message': 'No tracks available'}), 400
        
        # Extract trajectories
        trajectories = []
        for track in current_tracks:
            if len(track.cells) > 5:
                traj = np.array([cell.centroid for cell in track.cells])
                trajectories.append(traj)
        
        if len(trajectories) == 0:
            return jsonify({'success': False, 'message': 'No suitable trajectories'}), 400
        
        # Analyze trajectories
        analyzer = TrajectoryAnalyzer()
        stats = analyzer.compute_trajectory_statistics(trajectories)
        
        # Cluster trajectories
        clusters = []
        if len(trajectories) >= 3:
            labels = analyzer.cluster_trajectories(trajectories, n_clusters=3)
            stats['clusters'] = labels.tolist()
            clusters = labels.tolist()
        else:
            clusters = [0] * len(trajectories)
            
        # Convert trajectories to list format for JSON
        traj_list = [t.tolist() for t in trajectories]
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'n_trajectories': len(trajectories),
            'trajectories': traj_list,
            'clusters': clusters
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# Global model instances
vae = MorphologyVAE()
gan = TrajectoryGAN()
diff_predictor = DifferentiationPredictor()
param_estimator = ParameterEstimator()

# Initialize models (simplified for demo - in production would load weights)
def init_models():
    pass

init_models()


@app.route('/api/differentiation/predict', methods=['POST'])
def predict_differentiation():
    """Predict future cell states using LSTM and ParameterEstimator"""
    try:
        data = request.json
        n_steps = data.get('n_steps', 10)
        
        # Use actual model for prediction
        # Simulate input trajectory (batch_size=1, seq_len=10, input_dim=3)
        dummy_input = torch.randn(1, 10, 3)
        
        with torch.no_grad():
            outcome_probs = diff_predictor(dummy_input)
            probs = outcome_probs[0].numpy().tolist()
        
        # Use ParameterEstimator for fate trajectory
        # Initialize with dummy states if not fitted
        if param_estimator.transition_matrix is None:
            # Create dummy states for initialization
            dummy_states = np.random.rand(100, 3)
            param_estimator.estimate_transition_probabilities(dummy_states)
            
        current_state = np.random.rand(3)
        cell_fate = param_estimator.predict_cell_fate(current_state, n_steps)
        
        # Generate predicted states using simple projection for demo
        # In real scenario, would use DifferentiationModel (GP)
        predicted_states = []
        state = current_state
        for _ in range(n_steps):
            state = state + np.random.randn(3) * 0.1
            predicted_states.append(state.tolist())
        
        return jsonify({
            'success': True,
            'prediction': {
                'predicted_states': predicted_states,
                'confidence': [max(probs)] * n_steps,  # Simplified confidence
                'cell_fate': cell_fate,
                'outcome_probs': probs
            },
            'n_steps': n_steps
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/generate/trajectories', methods=['POST'])
def generate_trajectories():
    """Generate synthetic trajectories using GAN"""
    try:
        data = request.json
        n_samples = data.get('n_samples', 5)
        trajectory_length = data.get('length', 50)
        
        # Update GAN configuration if needed
        if gan.trajectory_length != trajectory_length:
            gan.trajectory_length = trajectory_length
            gan.generator = TrajectoryGenerator(trajectory_length=trajectory_length)
        
        # Generate conditions (random for demo)
        conditions = torch.randn(n_samples, gan.condition_dim)
        
        # Generate trajectories
        generated_data = gan.generate_trajectories(conditions, n_samples)
        
        # Convert to list format
        trajectories = generated_data.tolist()
        
        return jsonify({
            'success': True,
            'trajectories': trajectories,
            'n_samples': n_samples
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/generate/morphology', methods=['POST'])
def generate_morphology():
    """Generate cell morphology variations using VAE (simulated for demo)"""
    try:
        data = request.json
        n_samples = data.get('n_samples', 5)
        
        # For demo purposes, we generate procedural cell images
        # In a real scenario with trained weights, we would use:
        # generated_tensors = vae.generate(n_samples)
        
        morphologies = []
        for _ in range(n_samples):
            # Create blank image
            img = np.zeros((64, 64), dtype=np.uint8)
            
            # Random parameters
            center_x = np.random.randint(28, 36)
            center_y = np.random.randint(28, 36)
            radius = np.random.randint(15, 22)
            
            # Draw main cell body (irregular blob)
            # Use multiple overlapping circles to create irregularity
            cv2.circle(img, (center_x, center_y), radius, 200, -1)
            for _ in range(3):
                ox = center_x + np.random.randint(-5, 5)
                oy = center_y + np.random.randint(-5, 5)
                r = np.random.randint(10, 18)
                cv2.circle(img, (ox, oy), r, 200, -1)
            
            # Add texture/noise
            noise = np.random.randint(-20, 20, (64, 64))
            img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
            
            # Smooth
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
            # Apply color map
            img_color = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
            
            morphologies.append(image_to_base64(img_color))
        
        return jsonify({
            'success': True,
            'morphologies': morphologies,
            'n_samples': n_samples
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


if __name__ == '__main__':
    print("Cell Tracking and Differentiation System")
    print("=" * 50)
    print("Server starting on http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000, host='0.0.0.0')
