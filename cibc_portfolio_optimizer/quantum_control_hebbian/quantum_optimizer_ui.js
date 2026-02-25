/**
 * Quantum Optimizer UI Module
 * NVQLink quantum optimization visualization
 */

class QuantumOptimizerUI {
    constructor() {
        this.quantumAvailable = false;
        this.canvas = null;
        this.ctx = null;
    }

    async init() {
        this.setupCanvas();
        this.setupButtons();
        await this.checkQuantumAvailability();
    }

    setupCanvas() {
        this.canvas = document.getElementById('quantumCanvas');
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = this.canvas.offsetHeight;

        this.drawQuantumCircuit();
    }

    async checkQuantumAvailability() {
        try {
            const response = await window.app.get('/quantum/circuit');
            this.quantumAvailable = response.available;

            const statusEl = document.getElementById('quantumStatus');
            if (statusEl) {
                if (this.quantumAvailable) {
                    statusEl.innerHTML = `
                        <div class="safety-status safe">
                            ✓ CUDA-Q Available
                            <br>
                            <small>${response.num_qubits} qubits, ${response.num_parameters} parameters</small>
                        </div>
                    `;
                } else {
                    statusEl.innerHTML = `
                        <div class="safety-status warning">
                            ⚠ CUDA-Q Not Available
                            <br>
                            <small>Using classical fallback</small>
                        </div>
                    `;
                }
            }

            // Display circuit info
            const infoEl = document.getElementById('quantumCircuitInfo');
            if (infoEl && this.quantumAvailable) {
                infoEl.innerHTML = `
                    <h4>Quantum Circuit</h4>
                    <ul>
                        <li>Framework: ${response.framework}</li>
                        <li>Qubits: ${response.num_qubits}</li>
                        <li>Parameters: ${response.num_parameters}</li>
                        <li>Circuit Depth: ${response.circuit_depth}</li>
                        <li>Gates: ${response.gates.join(', ')}</li>
                        <li>Backend: ${response.backend}</li>
                    </ul>
                `;
            }
        } catch (error) {
            console.error('Failed to check quantum availability:', error);
        }
    }

    setupButtons() {
        // VQE Optimize button
        const vqeBtn = document.getElementById('optimizeVQEBtn');
        if (vqeBtn) {
            vqeBtn.addEventListener('click', () => this.runVQEOptimization());
        }

        // Compare button
        const compareBtn = document.getElementById('compareQuantumBtn');
        if (compareBtn) {
            compareBtn.addEventListener('click', () => this.compareQuantumClassical());
        }
    }

    async runVQEOptimization() {
        const btn = document.getElementById('optimizeVQEBtn');
        const resultsDiv = document.getElementById('vqeOptimizationResults');

        btn.disabled = true;
        btn.innerHTML = '<div class="loading"></div> Optimizing...';
        resultsDiv.innerHTML = '<p>Running VQE optimization...</p>';

        try {
            const response = await window.app.post('/quantum/optimize/vqe', {
                initial_params: {
                    amplitude_ma: 3.0,
                    frequency_hz: 20,
                    pulse_width_us: 90,
                    duty_cycle: 0.5
                },
                bounds: {
                    amplitude_ma: [0.5, 8.0],
                    frequency_hz: [4, 100],
                    pulse_width_us: [60, 210],
                    duty_cycle: [0.1, 0.9]
                },
                max_iterations: 50
            });

            if (response.success) {
                resultsDiv.innerHTML = `
                    <div class="result-item">
                        <strong>VQE Optimization Complete!</strong><br>
                        Method: ${response.method}<br>
                        Iterations: ${response.iterations}<br>
                        Final Energy: ${response.energy.toFixed(4)}<br><br>
                        <strong>Optimal Parameters:</strong><br>
                        Amplitude: ${response.optimal_parameters.amplitude_ma.toFixed(2)} mA<br>
                        Frequency: ${response.optimal_parameters.frequency_hz.toFixed(0)} Hz<br>
                        Pulse Width: ${response.optimal_parameters.pulse_width_us.toFixed(0)} μs<br>
                        Duty Cycle: ${response.optimal_parameters.duty_cycle.toFixed(2)}
                    </div>
                `;

                this.visualizeOptimization(response);
                btn.innerHTML = '✓ VQE Complete';
            }
        } catch (error) {
            resultsDiv.innerHTML = `<div class="error-message">Optimization failed: ${error.message}</div>`;
            btn.innerHTML = 'Optimize with VQE';
        } finally {
            btn.disabled = false;
        }
    }

    async compareQuantumClassical() {
        const btn = document.getElementById('compareQuantumBtn');
        const resultsDiv = document.getElementById('comparisonResults');

        btn.disabled = true;
        btn.innerHTML = '<div class="loading"></div> Comparing...';
        resultsDiv.innerHTML = '<p>Comparing quantum vs classical optimization...</p>';

        try {
            const response = await window.app.post('/quantum/compare', {
                initial_params: {
                    amplitude_ma: 3.0,
                    frequency_hz: 20,
                    pulse_width_us: 90,
                    duty_cycle: 0.5
                },
                bounds: {
                    amplitude_ma: [0.5, 8.0],
                    frequency_hz: [4, 100],
                    pulse_width_us: [60, 210],
                    duty_cycle: [0.1, 0.9]
                }
            });

            if (response.success) {
                const comparison = response.comparison;

                let html = '<div class="result-item">';
                html += '<strong>Optimization Comparison</strong><br><br>';
                html += `Classical Energy: ${comparison.classical_energy.toFixed(4)}<br>`;

                if (comparison.quantum_energy !== null) {
                    html += `Quantum Energy: ${comparison.quantum_energy.toFixed(4)}<br>`;
                    html += `Speedup: ${comparison.speedup ? comparison.speedup.toFixed(2) + 'x' : 'N/A'}<br>`;
                    html += `Quantum Advantage: ${comparison.quantum_advantage ? '✓ Yes' : '✗ No'}<br>`;
                } else {
                    html += '<br><em>Quantum optimization not available (CUDA-Q required)</em>';
                }

                html += '</div>';
                resultsDiv.innerHTML = html;

                btn.innerHTML = 'Compare Methods';
            }
        } catch (error) {
            resultsDiv.innerHTML = `<div class="error-message">Comparison failed: ${error.message}</div>`;
            btn.innerHTML = 'Compare Methods';
        } finally {
            btn.disabled = false;
        }
    }

    drawQuantumCircuit() {
        if (!this.ctx) return;

        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear canvas
        this.ctx.fillStyle = '#0a0a0a';
        this.ctx.fillRect(0, 0, width, height);

        // Draw simplified quantum circuit diagram
        const numQubits = 4;
        const qubitSpacing = height / (numQubits + 1);
        const gateWidth = 40;
        const gateHeight = 30;

        // Draw qubit lines
        this.ctx.strokeStyle = '#00d4ff';
        this.ctx.lineWidth = 2;

        for (let i = 0; i < numQubits; i++) {
            const y = qubitSpacing * (i + 1);
            this.ctx.beginPath();
            this.ctx.moveTo(50, y);
            this.ctx.lineTo(width - 50, y);
            this.ctx.stroke();

            // Qubit label
            this.ctx.fillStyle = '#b0b0b0';
            this.ctx.font = '14px Inter';
            this.ctx.fillText(`q${i}`, 10, y + 5);
        }

        // Draw gates
        const gatePositions = [100, 200, 300, 400];
        const gateTypes = ['RY(θ)', 'CNOT', 'RZ(φ)', 'RY(θ)'];

        gatePositions.forEach((x, idx) => {
            for (let i = 0; i < numQubits; i++) {
                const y = qubitSpacing * (i + 1);

                if (gateTypes[idx] === 'CNOT' && i < numQubits - 1) {
                    // Draw CNOT gate
                    this.ctx.fillStyle = '#00d4ff';
                    this.ctx.beginPath();
                    this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
                    this.ctx.fill();

                    const targetY = qubitSpacing * (i + 2);
                    this.ctx.strokeStyle = '#00d4ff';
                    this.ctx.beginPath();
                    this.ctx.moveTo(x, y);
                    this.ctx.lineTo(x, targetY);
                    this.ctx.stroke();

                    this.ctx.beginPath();
                    this.ctx.arc(x, targetY, 10, 0, 2 * Math.PI);
                    this.ctx.stroke();
                    this.ctx.beginPath();
                    this.ctx.moveTo(x - 10, targetY);
                    this.ctx.lineTo(x + 10, targetY);
                    this.ctx.moveTo(x, targetY - 10);
                    this.ctx.lineTo(x, targetY + 10);
                    this.ctx.stroke();
                } else if (gateTypes[idx] !== 'CNOT') {
                    // Draw rotation gate
                    this.ctx.fillStyle = 'rgba(0, 212, 255, 0.3)';
                    this.ctx.fillRect(x - gateWidth / 2, y - gateHeight / 2, gateWidth, gateHeight);
                    this.ctx.strokeStyle = '#00d4ff';
                    this.ctx.strokeRect(x - gateWidth / 2, y - gateHeight / 2, gateWidth, gateHeight);

                    this.ctx.fillStyle = '#ffffff';
                    this.ctx.font = '12px Inter';
                    this.ctx.textAlign = 'center';
                    this.ctx.fillText(gateTypes[idx], x, y + 4);
                }
            }
        });

        // Title
        this.ctx.fillStyle = '#00d4ff';
        this.ctx.font = 'bold 16px Inter';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('VQE Ansatz for DBS Parameter Optimization', width / 2, 30);
    }

    visualizeOptimization(result) {
        // Draw optimization progress on canvas
        if (!this.ctx) return;

        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear and redraw circuit
        this.drawQuantumCircuit();

        // Add optimization result overlay
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(width - 250, height - 150, 240, 140);

        this.ctx.fillStyle = '#00ff88';
        this.ctx.font = 'bold 14px Inter';
        this.ctx.textAlign = 'left';
        this.ctx.fillText('Optimization Result:', width - 240, height - 130);

        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '12px Inter';
        this.ctx.fillText(`Energy: ${result.energy.toFixed(4)}`, width - 240, height - 110);
        this.ctx.fillText(`Iterations: ${result.iterations}`, width - 240, height - 90);
        this.ctx.fillText(`Method: ${result.method}`, width - 240, height - 70);

        this.ctx.fillStyle = '#00d4ff';
        this.ctx.fillText(`Amp: ${result.optimal_parameters.amplitude_ma.toFixed(2)} mA`, width - 240, height - 45);
        this.ctx.fillText(`Freq: ${result.optimal_parameters.frequency_hz.toFixed(0)} Hz`, width - 240, height - 25);
    }
}

window.QuantumOptimizerUI = QuantumOptimizerUI;
