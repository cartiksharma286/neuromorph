import os
import sys
import time
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# University of Toronto - Advanced QML & Agentic VR-LMS Proof of Concept
# Simulates neural plasticity & quantum-driven attention mechanics

class HebbianPlasticityEngine:
    """
    Models cognitive synaptic weight updates based on Hebbian Learning:
    Δw = η * x_i * y_j - λ * w
    where x_i represents VR stim intensity, y_j represents cognitive focus, and λ is decay.
    """
    def __init__(self, learning_rate=0.05, decay=0.01):
        self.learning_rate = learning_rate
        self.decay = decay
        # Synaptic weights for 5 critical cognitive sectors:
        # Spatial Memory, Executive Attention, Sensory Integration, Memory Consolidation, Language
        self.weights = np.array([0.2, 0.15, 0.25, 0.1, 0.3])
        self.labels = ["Spatial Navigation", "Executive Attention", "Sensory Integration", "Memory Consolidation", "Cognitive Flexibility"]

    def step(self, stimuli, attention):
        stimuli = np.clip(stimuli, 0.0, 1.0)
        attention = np.clip(attention, 0.0, 1.0)
        
        # Hebbian rule: neurons that fire together, wire together
        dw = self.learning_rate * stimuli * attention - self.decay * self.weights
        self.weights = np.clip(self.weights + dw, 0.01, 5.0)
        return self.weights.tolist()


class VariationalQuantumClassifierSim:
    """
    Emulates a Parameterized Quantum Circuit (PQC) classifier on a topological lattice.
    Converts student feature dimension (Focus, Completion Rate, Latency) into Hilbert space angles,
    applies entanglement gates (CNOT), and performs measurement expectations on Z-basis.
    """
    def __init__(self):
        pass

    def predict_competence(self, focus, completion_rate, latency):
        # Angle embedding of student state
        theta1 = focus * np.pi
        theta2 = completion_rate * np.pi
        theta3 = (1.0 - latency) * np.pi  # faster is better

        # Parameterized rotations (Variational Ansätz)
        # We simulate expectation <Z0 Z1> of a 2-qubit system
        coupling = np.sin(theta1) * np.cos(theta2) + np.sin(theta3)
        expect_z1_z2 = np.tanh(1.5 * coupling)

        # Map expect (-1 to +1) to target competence prediction model
        competence_score = float(50.0 + 50.0 * expect_z1_z2)
        
        # Simulate circuit parameter optimization trace (VQE history)
        epochs = 15
        loss_history = []
        val = 1.8
        for i in range(epochs):
            val = val * 0.7 + 0.3 * (100.0 - competence_score) / 100.0
            loss_history.append(float(val + 0.05 * np.random.normal()))

        return {
            'competence_score': competence_score,
            'z_expectation': expect_z1_z2,
            'circuit_depth': 12,
            'topological_qubits': 2,
            'vqe_loss': loss_history,
            'bloch_coordinates': {
                'x': float(np.cos(theta1) * np.sin(theta2)),
                'y': float(np.sin(theta1) * np.sin(theta2)),
                'z': float(np.cos(theta2))
            }
        }


class AgenticAICopilot:
    """
    Emulates a cognitive state-engine running personalized learning paths,
    adjusting environment difficulty, generating conversational advice and scaffolding.
    """
    def __init__(self):
        self.states = ["Diagnostics", "Scaffolded Progression", "Frictionless Acceleration", "Synaptic Potentiation", "Topological Hyperlearning"]

    def determine_intervention(self, weight_avg, competence):
        if competence < 40.0:
            state = self.states[1] # Scaffold
            advice = "Agent Advisor (UT-Copilot): The current attention weight falls below the critical consolidation threshold. Activating sensory scaffolding in the VR theater. Adjusting visual cues to stimulate occipital pathways."
        elif competence < 75.0:
            if weight_avg > 0.4:
                state = self.states[3] # Hebbian Synaptic Potentiation
                advice = "Agent Advisor (UT-Copilot): High Hebbian plasticity detected! Accelerating visual complexity of your King's College spatial model. Introducing multi-agent Socratic debate modules."
            else:
                state = self.states[2] # Frictionless
                advice = "Agent Advisor (UT-Copilot): Attention and focus profile is balanced. Extending standard neural plasticity reinforcement sequences. VR navigation speed calibrated."
        else:
            state = self.states[4] # Hyperlearning
            advice = "Agent Advisor (UT-Copilot): Phenomenal hyperlearning state observed! Activating Quantum telemetric reinforcement protocols. Loading advanced cognitive challenge matrices from the Vector Institute."

        return {
            'agentic_state': state,
            'prescribed_scaffolding_level': float(100.0 - competence),
            'actionable_advice': advice,
            'active_policies': ["Socratic Scaffolding", "Dynamic Luminescence", "Hebbian Synaptic Potentiation"]
        }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/simulate-learning', methods=['POST'])
def simulate_learning():
    try:
        data = request.json or {}
        # Extracts sliders
        learning_rate = float(data.get('learning_rate', 0.05))
        decay = float(data.get('decay', 0.01))
        
        # 5 physical inputs for VR simulation
        stimuli = np.array(data.get('stimuli', [0.5, 0.6, 0.4, 0.7, 0.5]))
        attention = np.array(data.get('attention', [0.6, 0.5, 0.7, 0.8, 0.4]))

        engine = HebbianPlasticityEngine(learning_rate, decay)
        updated_weights = engine.step(stimuli, attention)
        avg_w = float(np.mean(updated_weights))

        # Perform Quantum ML Telemetry classification
        qml = VariationalQuantumClassifierSim()
        competence_data = qml.predict_competence(
            focus=float(np.mean(attention)),
            completion_rate=float(np.clip(avg_w * 0.3 + 0.4, 0.1, 0.95)),
            latency=float(np.clip(0.9 - avg_w * 0.1, 0.05, 0.95))
        )

        # Agentic Intervention Selection
        copilot = AgenticAICopilot()
        intervention = copilot.determine_intervention(avg_w, competence_data['competence_score'])

        # Create history trace for visualization
        epochs = 20
        history_weights = []
        for ep in range(epochs):
            # Model progression
            factor = (1.0 - np.exp(-ep / 8.0))
            history_weights.append([
                float(w * (0.4 + 0.6 * factor) + 0.02 * np.random.normal()) for w in updated_weights
            ])

        return jsonify({
            'success': True,
            'labels': engine.labels,
            'weights': updated_weights,
            'average_plasticity': avg_w,
            'history_weights_epochs': history_weights,
            'quantum_telemetry': competence_data,
            'agentic_intervention': intervention
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7505))
    app.run(debug=True, host='0.0.0.0', port=port)
