#!/usr/bin/env python3
"""
DejaVu Neural Circuitry Web App
Flask application for deja vu effect simulation
"""

from flask import Flask, jsonify, request, render_template_string
import numpy as np
from deja_vu_simulator import DejaVuSimulator
import json
from datetime import datetime

app = Flask(__name__)

# Global simulator instance
simulator = DejaVuSimulator(num_neurons=256)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DejaVu Neural Circuitry Simulator</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            color: #f8fafc;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: rgba(30, 41, 59, 0.9);
            border-radius: 12px;
            border: 2px solid #38bdf8;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            color: #38bdf8;
        }
        
        .subtitle {
            color: #94a3b8;
            font-size: 1.1em;
            margin-bottom: 15px;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(30, 41, 59, 0.9);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 8px;
            padding: 20px;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            border-color: #38bdf8;
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.1);
        }
        
        .card h2 {
            color: #38bdf8;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .control-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            color: #cbd5e1;
            font-weight: 500;
        }
        
        select, input, button {
            width: 100%;
            padding: 10px;
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 6px;
            background: rgba(15, 23, 42, 0.8);
            color: #f8fafc;
            font-size: 0.95em;
            transition: all 0.2s;
        }
        
        select:focus, input:focus {
            outline: none;
            border-color: #38bdf8;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
        }
        
        button {
            background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 10px;
        }
        
        button:hover {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(56, 189, 248, 0.3);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .result-display {
            background: rgba(15, 23, 42, 0.9);
            border-left: 4px solid #38bdf8;
            padding: 15px;
            border-radius: 6px;
            margin-top: 15px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(56, 189, 248, 0.1);
        }
        
        .metric-label {
            color: #cbd5e1;
        }
        
        .metric-value {
            color: #38bdf8;
            font-weight: 600;
            font-family: 'Courier New', monospace;
        }
        
        .deja-vu-indicator {
            text-align: center;
            font-size: 3em;
            margin: 20px 0;
            transition: all 0.3s;
        }
        
        .deja-vu-indicator.active {
            color: #f59e0b;
            text-shadow: 0 0 20px rgba(245, 158, 11, 0.5);
            animation: pulse 0.6s infinite;
        }
        
        .deja-vu-indicator.inactive {
            color: #64748b;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.8; }
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
        }
        
        .stats-display {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        
        .stat-box {
            background: rgba(15, 23, 42, 0.9);
            padding: 15px;
            border-radius: 6px;
            border-left: 3px solid #38bdf8;
        }
        
        .stat-label {
            color: #94a3b8;
            font-size: 0.85em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        
        .stat-value {
            color: #38bdf8;
            font-size: 1.5em;
            font-weight: 600;
            font-family: 'Courier New', monospace;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
        }
        
        .button-group button {
            flex: 1;
        }
        
        .loading {
            display: none;
            text-align: center;
            color: #38bdf8;
            margin: 20px 0;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            display: inline-block;
            width: 30px;
            height: 30px;
            border: 3px solid rgba(56, 189, 248, 0.3);
            border-top-color: #38bdf8;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 DejaVu Neural Circuitry Simulator</h1>
            <p class="subtitle">Simulating memory recognition with continued fractions and optimal pattern indexing</p>
        </header>
        
        <div class="grid">
            <!-- Stimulus Generation -->
            <div class="card">
                <h2>🌊 Stimulus Generation</h2>
                
                <div class="control-group">
                    <label for="pattern-type">Pattern Type:</label>
                    <select id="pattern-type">
                        <option value="sine">Sinusoidal (Musical Waves)</option>
                        <option value="gaussian">Gaussian (Bump Pattern)</option>
                        <option value="random">Random Noise</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label for="num-stimuli">Number of Stimuli:</label>
                    <input type="number" id="num-stimuli" min="1" max="50" value="10">
                </div>
                
                <div class="control-group">
                    <label for="exposure-count">Familiarization Exposures:</label>
                    <input type="number" id="exposure-count" min="1" max="15" value="5">
                </div>
                
                <button onclick="generateStimuli()">Generate & Process</button>
                
                <div class="loading" id="loading-indicator">
                    <div class="spinner"></div>
                    <p>Processing through neural circuit...</p>
                </div>
            </div>
            
            <!-- Neural Circuit Status -->
            <div class="card">
                <h2>⚡ Neural Circuit Status</h2>
                
                <div class="result-display" id="circuit-status" style="display: none;">
                    <div class="metric">
                        <span class="metric-label">Circuit State:</span>
                        <span class="metric-value" id="circuit-state">Idle</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Activation Level:</span>
                        <span class="metric-value" id="activation-level">0%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Memory Buffers Filled:</span>
                        <span class="metric-value" id="memory-buffers">0</span>
                    </div>
                </div>
                
                <div id="quick-stats" style="margin-top: 15px;"></div>
            </div>
        </div>
        
        <div class="grid">
            <!-- DejaVu Detector -->
            <div class="card">
                <h2>✨ DejaVu Detector</h2>
                
                <div class="deja-vu-indicator inactive" id="deja-vu-icon">✓</div>
                
                <div class="result-display" id="deja-vu-results" style="display: none;">
                    <div class="metric">
                        <span class="metric-label">DejaVu Score:</span>
                        <span class="metric-value" id="deja-vu-score">0.00</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Status:</span>
                        <span class="metric-value" id="deja-vu-status">Not Detected</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Pattern Match Confidence:</span>
                        <span class="metric-value" id="pattern-match">0.00</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Similar Patterns Found:</span>
                        <span class="metric-value" id="similar-count">0</span>
                    </div>
                </div>
            </div>
            
            <!-- Statistics -->
            <div class="card">
                <h2>📊 Overall Statistics</h2>
                
                <div class="stats-display" id="stats-display" style="display: none;">
                    <div class="stat-box">
                        <div class="stat-label">Mean DejaVu</div>
                        <div class="stat-value" id="stat-mean">0.00</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Std Dev</div>
                        <div class="stat-value" id="stat-std">0.00</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Max Score</div>
                        <div class="stat-value" id="stat-max">0.00</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">DejaVu Frequency</div>
                        <div class="stat-value" id="stat-freq">0%</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Visualization -->
        <div class="card" style="grid-column: 1 / -1;">
            <h2>📈 DejaVu Score Timeline</h2>
            <div class="chart-container">
                <canvas id="dejaVuChart"></canvas>
            </div>
        </div>
        
        <!-- Information -->
        <div class="card" style="grid-column: 1 / -1;">
            <h2>ℹ️ How It Works</h2>
            <p style="line-height: 1.8; color: #cbd5e1;">
                <strong>Neural Circuitry:</strong> Three-layer neural network (sensory → memory → recognition) simulates how the brain processes stimuli.
                Deja vu occurs when the recognition layer strongly activates previously learned memory patterns.<br><br>
                
                <strong>Continued Fractions:</strong> Each memory pattern is encoded as a continued fraction, enabling efficient similarity comparisons
                through convergent analysis. Patterns with similar continued fraction representations are more likely to trigger deja vu.<br><br>
                
                <strong>Optimal Pattern Indexing:</strong> Locality-sensitive hashing enables fast retrieval of similar patterns from memory.
                When a new stimulus arrives, the system queries the index to find familiar patterns, computing confidence scores based on
                both neural activation and mathematical pattern similarity.<br><br>
                
                <strong>Familiarization:</strong> Repeated exposure to the same stimulus strengthens memory through Hebbian learning,
                increasing the probability of deja vu detection on subsequent encounters.
            </p>
        </div>
    </div>
    
    <script>
        let dejaVuChart = null;
        let dejaVuScores = [];
        
        function showLoading(show) {
            document.getElementById('loading-indicator').classList.toggle('active', show);
        }
        
        function generateStimuli() {
            showLoading(true);
            
            const patternType = document.getElementById('pattern-type').value;
            const numStimuli = parseInt(document.getElementById('num-stimuli').value);
            const exposureCount = parseInt(document.getElementById('exposure-count').value);
            
            fetch('/api/deja-vu/generate-sequence', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    pattern_type: patternType,
                    num_stimuli: numStimuli,
                    familiarization_exposures: exposureCount
                })
            })
            .then(r => r.json())
            .then(data => {
                dejaVuScores = data.deja_vu_scores;
                
                updateDejaVuIndicator(data.last_result);
                updateStatistics(data.statistics);
                updateChart();
                
                document.getElementById('circuit-status').style.display = 'block';
                document.getElementById('deja-vu-results').style.display = 'block';
                document.getElementById('stats-display').style.display = 'grid';
                
                showLoading(false);
            })
            .catch(err => {
                console.error(err);
                showLoading(false);
            });
        }
        
        function updateDejaVuIndicator(result) {
            const icon = document.getElementById('deja-vu-icon');
            icon.classList.toggle('active', result.is_deja_vu);
            icon.classList.toggle('inactive', !result.is_deja_vu);
            icon.textContent = result.is_deja_vu ? '★' : '✓';
            
            document.getElementById('deja-vu-score').textContent = result.deja_vu_score.toFixed(4);
            document.getElementById('deja-vu-status').textContent = result.is_deja_vu ? 'DEJA VU DETECTED' : 'Familiar';
            document.getElementById('pattern-match').textContent = result.top_match.toFixed(4);
            document.getElementById('similar-count').textContent = result.similar_patterns;
        }
        
        function updateStatistics(stats) {
            document.getElementById('stat-mean').textContent = stats.mean_deja_vu.toFixed(4);
            document.getElementById('stat-std').textContent = stats.std_deja_vu.toFixed(4);
            document.getElementById('stat-max').textContent = stats.max_deja_vu.toFixed(4);
            document.getElementById('stat-freq').textContent = (stats.deja_vu_frequency * 100).toFixed(1) + '%';
        }
        
        function updateChart() {
            const ctx = document.getElementById('dejaVuChart').getContext('2d');
            
            if (dejaVuChart) {
                dejaVuChart.destroy();
            }
            
            dejaVuChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({length: dejaVuScores.length}, (_, i) => `T${i+1}`),
                    datasets: [{
                        label: 'DejaVu Score',
                        data: dejaVuScores,
                        borderColor: '#38bdf8',
                        backgroundColor: 'rgba(56, 189, 248, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 4,
                        pointBackgroundColor: dejaVuScores.map(s => s > 0.3 ? '#f59e0b' : '#38bdf8'),
                        pointBorderColor: '#0f172a',
                        pointBorderWidth: 2
                    }, {
                        label: 'DejaVu Threshold (0.3)',
                        data: Array(dejaVuScores.length).fill(0.3),
                        borderColor: '#f59e0b',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {color: '#cbd5e1', font: {size: 12}}
                        }
                    },
                    scales: {
                        y: {
                            ticks: {color: '#94a3b8'},
                            grid: {color: 'rgba(56, 189, 248, 0.1)'},
                            beginAtZero: true,
                            max: 1
                        },
                        x: {
                            ticks: {color: '#94a3b8'},
                            grid: {color: 'rgba(56, 189, 248, 0.1)'}
                        }
                    }
                }
            });
        }
        
        // Initial chart
        updateChart();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/deja-vu/generate-sequence', methods=['POST'])
def generate_sequence():
    """Generate stimulus sequence and process through simulator"""
    data = request.json
    
    pattern_type = data.get('pattern_type', 'sine')
    num_stimuli = data.get('num_stimuli', 10)
    familiarization_exposures = data.get('familiarization_exposures', 5)
    
    try:
        # Generate stimulus sequence
        stimuli = simulator.generate_stimulus_sequence(
            num_stimuli=num_stimuli,
            pattern_type=pattern_type
        )
        
        deja_vu_scores = []
        last_result = None
        
        # Process each stimulus
        for i, stimulus in enumerate(stimuli):
            is_new = i < 3  # First 3 are learning phase
            result = simulator.experience_stimulus(stimulus, is_new_learning=is_new)
            deja_vu_scores.append(result['deja_vu_score'])
            last_result = result
            
            # Familiarization experiment on first stimulus
            if i == 0:
                fam_scores = simulator.simulate_familiarization(stimulus, num_exposures=familiarization_exposures)
                deja_vu_scores.extend(fam_scores)
        
        # Get statistics
        stats = simulator.get_statistics()
        
        return jsonify({
            'success': True,
            'deja_vu_scores': deja_vu_scores,
            'last_result': last_result,
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/deja-vu/single-stimulus', methods=['POST'])
def single_stimulus():
    """Process a single stimulus"""
    data = request.json
    stimulus_type = data.get('stimulus_type', 'random')
    
    try:
        # Generate single stimulus
        if stimulus_type == 'sine':
            t = np.linspace(0, 2 * np.pi, 64)
            stimulus = np.sin(t) + 0.1 * np.random.randn(64)
        elif stimulus_type == 'gaussian':
            x = np.arange(64)
            stimulus = np.exp(-((x - 32) ** 2) / 100.0) + 0.05 * np.random.randn(64)
        else:
            stimulus = np.random.randn(64)
        
        # Process
        result = simulator.experience_stimulus(stimulus, is_new_learning=True)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/deja-vu/stats', methods=['GET'])
def get_stats():
    """Get current simulator statistics"""
    stats = simulator.get_statistics()
    return jsonify({
        'success': True,
        'statistics': stats,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/deja-vu/reset', methods=['POST'])
def reset_simulator():
    """Reset simulator state"""
    global simulator
    simulator = DejaVuSimulator(num_neurons=256)
    return jsonify({
        'success': True,
        'message': 'Simulator reset',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5051, host='0.0.0.0')
