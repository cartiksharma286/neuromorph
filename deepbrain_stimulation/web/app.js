// ===== Application State =====
const AppState = {
    simulation: {
        running: false,
        speed: 1.0,
        time: 0,
        step: 0
    },
    network: {
        neurons: 100,
        topology: 'small-world',
        neuronModel: 'lif',
        connections: []
    },
    learning: {
        rate: 0.05,
        rule: 'hebbian',
        quantumOptimization: true
    },
    tbi: {
        severity: 0,
        type: 'diffuse-axonal',
        applied: false
    },
    metrics: {
        firingRate: 0,
        avgWeight: 0.5,
        synchrony: 0,
        quantumFidelity: 1.0,
        networkHealth: 100
    }
};

// ===== API Client =====
class APIClient {
    constructor(baseURL = 'http://127.0.0.1:5000') {
        this.baseURL = baseURL;
        this.ws = null;
    }

    async post(endpoint, data) {
        try {
            const response = await fetch(`${this.baseURL}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            return await response.json();
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            return null;
        }
    }

    async get(endpoint) {
        try {
            const response = await fetch(`${this.baseURL}${endpoint}`);
            return await response.json();
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            return null;
        }
    }

    connectWebSocket(onMessage) {
        this.ws = new WebSocket('ws://127.0.0.1:5000/ws');
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onMessage(data);
        };
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    sendWebSocket(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }
}

const api = new APIClient();

// ===== UI Controllers =====
class UIController {
    constructor() {
        this.initializeControls();
        this.initializeTabs();
        this.initializeEventListeners();
    }

    initializeControls() {
        // Neuron count slider
        const neuronSlider = document.getElementById('neuron-slider');
        const neuronValue = neuronSlider.nextElementSibling;
        neuronSlider.addEventListener('input', (e) => {
            const value = e.target.value;
            neuronValue.textContent = value;
            AppState.network.neurons = parseInt(value);
            document.getElementById('neuron-count').textContent = value;
        });

        // Learning rate slider
        const learningSlider = document.getElementById('learning-rate-slider');
        const learningValue = learningSlider.nextElementSibling;
        learningSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            learningValue.textContent = value.toFixed(3);
            AppState.learning.rate = value;
            document.getElementById('learning-rate').textContent = value.toFixed(2);
        });

        // Damage severity slider
        const damageSlider = document.getElementById('damage-severity-slider');
        const damageValue = damageSlider.nextElementSibling;
        damageSlider.addEventListener('input', (e) => {
            const value = e.target.value;
            damageValue.textContent = `${value}%`;
            AppState.tbi.severity = parseInt(value);
        });

        // Speed slider
        const speedSlider = document.getElementById('speed-slider');
        const speedValue = speedSlider.nextElementSibling;
        speedSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            speedValue.textContent = `${value.toFixed(1)}x`;
            AppState.simulation.speed = value;
        });

        // Topology select
        document.getElementById('topology-select').addEventListener('change', (e) => {
            AppState.network.topology = e.target.value;
        });

        // Neuron model select
        document.getElementById('neuron-model-select').addEventListener('change', (e) => {
            AppState.network.neuronModel = e.target.value;
        });

        // Plasticity rule select
        document.getElementById('plasticity-rule-select').addEventListener('change', (e) => {
            AppState.learning.rule = e.target.value;
        });

        // Damage type select
        document.getElementById('damage-type-select').addEventListener('change', (e) => {
            AppState.tbi.type = e.target.value;
        });

        // Quantum optimization checkbox
        document.getElementById('quantum-optimization').addEventListener('change', (e) => {
            AppState.learning.quantumOptimization = e.target.checked;
        });
    }

    initializeTabs() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                // Remove active class from all tabs
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));

                // Add active class to clicked tab
                button.classList.add('active');
                const tabId = button.getAttribute('data-tab');
                document.getElementById(`${tabId}-tab`).classList.add('active');
            });
        });
    }

    initializeEventListeners() {
        // Start button
        document.getElementById('start-btn').addEventListener('click', () => {
            this.startSimulation();
        });

        // Pause button
        document.getElementById('pause-btn').addEventListener('click', () => {
            this.pauseSimulation();
        });

        // Reset button
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetSimulation();
        });

        // Apply damage button
        document.getElementById('apply-damage-btn').addEventListener('click', () => {
            this.applyTBIDamage();
        });

        // Start recovery button
        document.getElementById('start-recovery-btn').addEventListener('click', () => {
            this.startRecovery();
        });

        // Rotate toggle
        document.getElementById('rotate-toggle').addEventListener('click', () => {
            if (window.neuralViz) {
                window.neuralViz.toggleRotation();
            }
        });

        // Fullscreen toggle
        document.getElementById('fullscreen-toggle').addEventListener('click', () => {
            const container = document.getElementById('neural-network-viz');
            if (!document.fullscreenElement) {
                container.requestFullscreen();
            } else {
                document.exitFullscreen();
            }
        });
    }

    async startSimulation() {
        if (AppState.simulation.running) return;

        AppState.simulation.running = true;
        document.getElementById('start-btn').disabled = true;
        document.getElementById('pause-btn').disabled = false;

        // Initialize network
        const response = await api.post('/api/initialize', {
            neurons: AppState.network.neurons,
            topology: AppState.network.topology,
            neuronModel: AppState.network.neuronModel
        });

        if (response && response.success) {
            // Start simulation
            await api.post('/api/start', {
                learningRate: AppState.learning.rate,
                plasticityRule: AppState.learning.rule,
                quantumOptimization: AppState.learning.quantumOptimization
            });

            // Start polling for updates
            this.startPolling();
            this.showNotification('Simulation started', 'success');
        } else {
            this.showNotification('Failed to start simulation - using demo mode', 'warning');
            AppState.simulation.running = false;
            // Fall back to mock data
            this.startMockMode();
        }
    }

    startPolling() {
        // Poll server every 100ms for updates
        this.pollingInterval = setInterval(async () => {
            if (!AppState.simulation.running) {
                clearInterval(this.pollingInterval);
                return;
            }

            const data = await api.get('/api/state');
            if (data) {
                this.handleSimulationUpdate(data);
            }
        }, 100);
    }

    startMockMode() {
        // Fallback to mock data if backend unavailable
        AppState.simulation.running = true;
        setInterval(() => {
            if (AppState.simulation.running && mockData) {
                const data = mockData.update();
                this.handleSimulationUpdate(data);
            }
        }, 100);
    }

    pauseSimulation() {
        if (!AppState.simulation.running) return;

        AppState.simulation.running = false;
        document.getElementById('start-btn').disabled = false;
        document.getElementById('pause-btn').disabled = true;

        api.sendWebSocket({ command: 'pause' });
        this.showNotification('Simulation paused', 'info');
    }

    async resetSimulation() {
        AppState.simulation.running = false;
        AppState.simulation.time = 0;
        AppState.simulation.step = 0;
        AppState.tbi.applied = false;

        document.getElementById('start-btn').disabled = false;
        document.getElementById('pause-btn').disabled = true;

        await api.post('/api/reset');

        // Reset visualizations
        if (window.neuralViz) window.neuralViz.reset();
        if (window.quantumViz) window.quantumViz.reset();
        if (window.statsViz) window.statsViz.reset();

        this.showNotification('Simulation reset', 'info');
    }

    async applyTBIDamage() {
        if (!AppState.simulation.running) {
            this.showNotification('Start simulation first', 'warning');
            return;
        }

        const response = await api.post('/api/apply-damage', {
            severity: AppState.tbi.severity,
            type: AppState.tbi.type
        });

        if (response && response.success) {
            AppState.tbi.applied = true;
            AppState.metrics.networkHealth = 100 - AppState.tbi.severity;
            document.getElementById('network-health').textContent = `${AppState.metrics.networkHealth}%`;
            this.showNotification(`TBI damage applied: ${AppState.tbi.severity}%`, 'warning');
        }
    }

    async startRecovery() {
        if (!AppState.tbi.applied) {
            this.showNotification('Apply TBI damage first', 'warning');
            return;
        }

        const response = await api.post('/api/start-recovery', {
            quantumOptimization: AppState.learning.quantumOptimization
        });

        if (response && response.success) {
            this.showNotification('Recovery simulation started', 'success');
        }
    }

    handleSimulationUpdate(data) {
        // Update metrics
        if (data.metrics) {
            AppState.metrics = { ...AppState.metrics, ...data.metrics };
            this.updateMetricsDisplay();
        }

        // Update visualizations
        if (data.network && window.neuralViz) {
            window.neuralViz.update(data.network);
        }

        if (data.quantum && window.quantumViz) {
            window.quantumViz.update(data.quantum);
        }

        if (data.statistics && window.statsViz) {
            window.statsViz.update(data.statistics);
        }

        if (data.gameTheory && window.gameTheoryViz) {
            window.gameTheoryViz.update(data.gameTheory);
        }

        if (data.continuedFractions && window.cfViz) {
            window.cfViz.update(data.continuedFractions);
        }

        // Update simulation time
        AppState.simulation.time = data.time || AppState.simulation.time;
        AppState.simulation.step = data.step || AppState.simulation.step;
    }

    updateMetricsDisplay() {
        const { firingRate, avgWeight, synchrony, quantumFidelity } = AppState.metrics;

        document.getElementById('firing-rate').textContent = `${firingRate.toFixed(1)} Hz`;
        document.getElementById('avg-weight').textContent = avgWeight.toFixed(2);
        document.getElementById('synchrony').textContent = synchrony.toFixed(2);
        document.getElementById('quantum-fidelity').textContent = quantumFidelity.toFixed(2);

        // Update sparklines (simplified - would use actual charting library)
        this.updateSparkline('firing-rate-sparkline', firingRate);
        this.updateSparkline('weight-sparkline', avgWeight);
        this.updateSparkline('synchrony-sparkline', synchrony);
        this.updateSparkline('fidelity-sparkline', quantumFidelity);
    }

    updateSparkline(elementId, value) {
        const element = document.getElementById(elementId);
        if (!element.sparklineData) {
            element.sparklineData = [];
        }

        element.sparklineData.push(value);
        if (element.sparklineData.length > 50) {
            element.sparklineData.shift();
        }

        // Simple visualization (would be replaced with actual charting)
        const max = Math.max(...element.sparklineData);
        const min = Math.min(...element.sparklineData);
        const range = max - min || 1;

        const gradient = element.sparklineData.map((v, i) => {
            const normalized = (v - min) / range;
            const opacity = 0.3 + (normalized * 0.7);
            return `rgba(99, 102, 241, ${opacity})`;
        }).join(', ');

        element.style.background = `linear-gradient(90deg, ${gradient})`;
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            padding: 1rem 1.5rem;
            background: var(--bg-glass);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            color: var(--text-primary);
            box-shadow: var(--shadow-lg);
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;

        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// ===== Mock Data Generator (for demo without backend) =====
class MockDataGenerator {
    constructor() {
        this.time = 0;
    }

    generateNetworkData(neurons = 100) {
        const positions = [];
        const connections = [];
        const activities = [];
        const weights = [];

        // Generate neuron positions in 3D space
        for (let i = 0; i < neurons; i++) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = 50 + Math.random() * 20;

            positions.push({
                x: r * Math.sin(phi) * Math.cos(theta),
                y: r * Math.sin(phi) * Math.sin(theta),
                z: r * Math.cos(phi)
            });

            activities.push(Math.random() > 0.8 ? 1 : 0);
        }

        // Generate connections (small-world network)
        for (let i = 0; i < neurons; i++) {
            const numConnections = 3 + Math.floor(Math.random() * 5);
            for (let j = 0; j < numConnections; j++) {
                const target = Math.floor(Math.random() * neurons);
                if (target !== i) {
                    connections.push({ source: i, target });
                    weights.push(0.3 + Math.random() * 0.4);
                }
            }
        }

        return { positions, connections, activities, weights };
    }

    generateQuantumData() {
        return {
            circuitDepth: 10,
            gateCount: 45,
            fidelity: 0.95 + Math.random() * 0.05,
            vqeEnergy: -2.5 + Math.random() * 0.5,
            convergence: Math.min(1, this.time / 100)
        };
    }

    generateStatistics() {
        return {
            weightDistribution: Array.from({ length: 50 }, () => Math.random()),
            spikeTrains: Array.from({ length: 20 }, () =>
                Array.from({ length: 100 }, () => Math.random() > 0.9 ? 1 : 0)
            ),
            isiHistogram: Array.from({ length: 30 }, () => Math.random() * 100),
            learningCurve: Array.from({ length: 100 }, (_, i) =>
                1 - Math.exp(-i / 20) + Math.random() * 0.1
            )
        };
    }

    generateGameTheoryData() {
        return {
            payoffMatrix: [
                [3, 0],
                [5, 1]
            ],
            nashEquilibrium: { player1: 0.6, player2: 0.4 },
            nimHeaps: [3, 5, 7],
            grundyValue: 3 ^ 5 ^ 7
        };
    }

    generateContinuedFractionData() {
        const convergents = [];
        for (let i = 1; i <= 20; i++) {
            convergents.push({
                n: i,
                value: Math.PI - 1 / Math.pow(2, i),
                error: 1 / Math.pow(2, i)
            });
        }

        return {
            convergents,
            padeApproximants: Array.from({ length: 100 }, (_, i) => ({
                x: (i - 50) / 10,
                original: 1 / (1 + Math.exp(-(i - 50) / 10)),
                pade: 1 / (1 + Math.exp(-(i - 50) / 10)) + Math.random() * 0.01
            }))
        };
    }

    update() {
        this.time++;
        return {
            time: this.time,
            step: this.time,
            metrics: {
                firingRate: 10 + Math.random() * 20,
                avgWeight: 0.4 + Math.random() * 0.2,
                synchrony: Math.random(),
                quantumFidelity: 0.95 + Math.random() * 0.05
            },
            network: this.generateNetworkData(AppState.network.neurons),
            quantum: this.generateQuantumData(),
            statistics: this.generateStatistics(),
            gameTheory: this.generateGameTheoryData(),
            continuedFractions: this.generateContinuedFractionData()
        };
    }
}

// ===== Initialize Application =====
let uiController;
let mockData;

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Quantum Neural Circuitry Platform Initialized');

    // Initialize UI controller
    uiController = new UIController();

    // Initialize mock data generator for demo
    mockData = new MockDataGenerator();

    // Start demo mode with mock data
    setInterval(() => {
        if (AppState.simulation.running) {
            const data = mockData.update();
            uiController.handleSimulationUpdate(data);
        }
    }, 100);

    console.log('✓ UI Controllers initialized');
    console.log('✓ Mock data generator ready');
    console.log('✓ Visualization modules loading...');
});

// ===== Export for use in other modules =====
window.AppState = AppState;
window.api = api;
window.uiController = uiController;
