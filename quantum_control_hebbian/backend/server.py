"""
Flask REST API Server for Quantum Neural Circuitry Platform
Provides endpoints for simulation control and real-time updates
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import json
from threading import Thread, Lock
import time

# Import our modules
try:
    from continued_fractions import (
        ContinuedFraction, PadeApproximant, NeuralActivationCF,
        ConvergenceAcceleration
    )
    from combinatorial_game_theory import (
        NimGame, NashEquilibrium, TBIRecoveryGame
    )
    MODULES_LOADED = True
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    MODULES_LOADED = False

app = Flask(__name__)
CORS(app)

# Global simulation state
simulation_state = {
    'running': False,
    'time': 0,
    'step': 0,
    'network': None,
    'metrics': {
        'firingRate': 0,
        'avgWeight': 0.5,
        'synchrony': 0,
        'quantumFidelity': 1.0
    }
}

state_lock = Lock()


class SimulationEngine:
    """Main simulation engine"""
    
    def __init__(self):
        self.neurons = 100
        self.topology = 'small-world'
        self.learning_rate = 0.05
        self.plasticity_rule = 'hebbian'
        self.quantum_optimization = True
        
        # Initialize network
        self.positions = []
        self.connections = []
        self.weights = []
        self.activities = []
        
        # Continued fractions
        self.cf_data = None
        
        # Game theory
        self.tbi_game = TBIRecoveryGame() if MODULES_LOADED else None
        self.nim_game = NimGame([3, 5, 7]) if MODULES_LOADED else None
        
    def initialize_network(self):
        """Initialize neural network"""
        # Generate neuron positions in 3D
        self.positions = []
        for i in range(self.neurons):
            theta = np.random.random() * np.pi * 2
            phi = np.arccos(2 * np.random.random() - 1)
            r = 50 + np.random.random() * 20
            
            self.positions.append({
                'x': r * np.sin(phi) * np.cos(theta),
                'y': r * np.sin(phi) * np.sin(theta),
                'z': r * np.cos(phi)
            })
        
        # Generate connections (small-world network)
        self.connections = []
        self.weights = []
        
        for i in range(self.neurons):
            num_connections = 3 + int(np.random.random() * 5)
            for _ in range(num_connections):
                target = int(np.random.random() * self.neurons)
                if target != i:
                    self.connections.append({'source': i, 'target': target})
                    self.weights.append(0.3 + np.random.random() * 0.4)
        
        # Initialize activities
        self.activities = [1 if np.random.random() > 0.8 else 0 for _ in range(self.neurons)]
    
    def update_step(self):
        """Update simulation by one step"""
        # Update neural activities (simplified)
        new_activities = []
        for i in range(self.neurons):
            # Random firing with some persistence
            if self.activities[i] == 1:
                new_activities.append(1 if np.random.random() > 0.3 else 0)
            else:
                new_activities.append(1 if np.random.random() > 0.9 else 0)
        
        self.activities = new_activities
        
        # Hebbian weight update
        for idx, conn in enumerate(self.connections):
            source_active = self.activities[conn['source']]
            target_active = self.activities[conn['target']]
            
            if source_active and target_active:
                # Strengthen connection
                self.weights[idx] = min(1.0, self.weights[idx] + self.learning_rate * 0.01)
            else:
                # Weak decay
                self.weights[idx] = max(0.0, self.weights[idx] - 0.001)
        
        # Update metrics
        firing_neurons = sum(self.activities)
        simulation_state['metrics']['firingRate'] = (firing_neurons / self.neurons) * 100
        simulation_state['metrics']['avgWeight'] = np.mean(self.weights) if self.weights else 0.5
        simulation_state['metrics']['synchrony'] = np.std(self.activities)
    
    def get_network_data(self):
        """Get current network state"""
        return {
            'positions': self.positions,
            'connections': self.connections,
            'weights': self.weights,
            'activities': self.activities
        }
    
    def get_continued_fraction_data(self):
        """Generate continued fraction visualization data"""
        if not MODULES_LOADED:
            return {}
        
        # Generate convergent sequence for π
        pi_coeffs = [3, 7, 15, 1, 292, 1, 1, 1, 2, 1]
        cf = ContinuedFraction(pi_coeffs)
        convergents_data = cf.convergence_sequence(20)
        
        # Generate Padé approximant data for sigmoid
        x_range = np.linspace(-5, 5, 100)
        comparison = NeuralActivationCF.compare_approximations(x_range)
        
        pade_data = []
        for i, x in enumerate(comparison['x']):
            pade_data.append({
                'x': float(x),
                'original': float(comparison['sigmoid']['original'][i]),
                'pade': float(comparison['sigmoid']['pade'][i])
            })
        
        return {
            'convergents': [
                {'n': n, 'value': float(val), 'error': float(err)}
                for n, val, err in convergents_data
            ],
            'padeApproximants': pade_data
        }
    
    def get_game_theory_data(self):
        """Get game theory visualization data"""
        if not MODULES_LOADED:
            return {}
        
        # Nim game
        nim_data = {
            'heaps': self.nim_game.heaps,
            'grundyValue': self.nim_game.grundy_value(),
            'isWinning': self.nim_game.is_winning_position()
        }
        
        # TBI recovery game
        equilibrium = self.tbi_game.find_equilibrium()
        
        # Nash equilibrium for simple 2x2 game
        payoff_2x2 = np.array([
            [[3, 3], [0, 5]],
            [[5, 0], [1, 1]]
        ])
        nash = NashEquilibrium(payoff_2x2)
        p1_strat, p2_strat = nash.find_mixed_nash_2x2()
        
        return {
            'nimHeaps': nim_data['heaps'],
            'grundyValue': nim_data['grundyValue'],
            'nashEquilibrium': {
                'player1': float(p1_strat[0]),
                'player2': float(p2_strat[0])
            },
            'payoffMatrix': payoff_2x2.tolist(),
            'tbiGame': equilibrium
        }


# Global simulation engine
sim_engine = SimulationEngine()


@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize simulation with parameters"""
    data = request.json
    
    sim_engine.neurons = data.get('neurons', 100)
    sim_engine.topology = data.get('topology', 'small-world')
    sim_engine.neuronModel = data.get('neuronModel', 'lif')
    
    sim_engine.initialize_network()
    
    return jsonify({'success': True, 'message': 'Network initialized'})


@app.route('/api/start', methods=['POST'])
def start_simulation():
    """Start simulation"""
    with state_lock:
        simulation_state['running'] = True
    
    return jsonify({'success': True, 'message': 'Simulation started'})


@app.route('/api/pause', methods=['POST'])
def pause_simulation():
    """Pause simulation"""
    with state_lock:
        simulation_state['running'] = False
    
    return jsonify({'success': True, 'message': 'Simulation paused'})


@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation"""
    with state_lock:
        simulation_state['running'] = False
        simulation_state['time'] = 0
        simulation_state['step'] = 0
    
    sim_engine.initialize_network()
    
    return jsonify({'success': True, 'message': 'Simulation reset'})


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current simulation state"""
    with state_lock:
        state = {
            'time': simulation_state['time'],
            'step': simulation_state['step'],
            'metrics': simulation_state['metrics'],
            'network': sim_engine.get_network_data(),
            'quantum': {
                'circuitDepth': 10,
                'gateCount': 45,
                'fidelity': 0.95 + np.random.random() * 0.05,
                'vqeEnergy': -2.5 + np.random.random() * 0.5,
                'stateVector': {
                    'theta': np.random.random() * np.pi,
                    'phi': np.random.random() * 2 * np.pi
                }
            },
            'statistics': {
                'weightDistribution': [float(w) for w in sim_engine.weights[:50]],
                'spikeTrains': [[int(np.random.random() > 0.9) for _ in range(100)] for _ in range(20)],
                'isiHistogram': [float(np.random.random() * 100) for _ in range(30)],
                'learningCurve': [float(1 - np.exp(-i / 20) + np.random.random() * 0.1) for i in range(100)]
            },
            'gameTheory': sim_engine.get_game_theory_data(),
            'continuedFractions': sim_engine.get_continued_fraction_data()
        }
    
    return jsonify(state)


@app.route('/api/apply-damage', methods=['POST'])
def apply_damage():
    """Apply TBI damage to network"""
    data = request.json
    severity = data.get('severity', 0)
    damage_type = data.get('type', 'diffuse-axonal')
    
    # Simulate damage by reducing weights
    damage_factor = severity / 100.0
    for i in range(len(sim_engine.weights)):
        if np.random.random() < damage_factor:
            sim_engine.weights[i] *= (1 - damage_factor)
    
    return jsonify({'success': True, 'message': f'Applied {severity}% {damage_type} damage'})


@app.route('/api/start-recovery', methods=['POST'])
def start_recovery():
    """Start recovery simulation"""
    data = request.json
    quantum_optimization = data.get('quantumOptimization', True)
    
    # Increase learning rate for recovery
    sim_engine.learning_rate = 0.1 if quantum_optimization else 0.05
    
    return jsonify({'success': True, 'message': 'Recovery simulation started'})


def simulation_loop():
    """Background simulation loop"""
    while True:
        with state_lock:
            if simulation_state['running']:
                sim_engine.update_step()
                simulation_state['step'] += 1
                simulation_state['time'] += 0.1
        
        time.sleep(0.1)  # 10 Hz update rate


# Start simulation thread
simulation_thread = Thread(target=simulation_loop, daemon=True)
simulation_thread.start()


if __name__ == '__main__':
    print("=" * 60)
    print("Quantum Neural Circuitry Platform - Backend Server")
    print("=" * 60)
    print(f"Modules loaded: {MODULES_LOADED}")
    print(f"Server starting on http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
