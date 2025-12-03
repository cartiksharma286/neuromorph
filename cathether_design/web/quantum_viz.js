/**
 * Quantum Circuit Visualization and Convergence Tracking
 * 
 * Visualizes quantum circuit execution and optimization progress
 */

let convergenceChart = null;
let circuitCanvas = null;
let circuitCtx = null;

// Initialize visualizations when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    initConvergenceChart();
    initQuantumCircuitViz();
});

/**
 * Initialize convergence chart using Chart.js
 */
function initConvergenceChart() {
    const canvas = document.getElementById('convergence-canvas');
    const ctx = canvas.getContext('2d');

    convergenceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Cost Function',
                data: [],
                borderColor: '#00D9FF',
                backgroundColor: 'rgba(0, 217, 255, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 4
            }, {
                label: 'Gradient Norm',
                data: [],
                borderColor: '#9D4EDD',
                backgroundColor: 'rgba(157, 78, 221, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 300
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#B8C1EC',
                        font: {
                            family: 'Inter',
                            size: 11
                        }
                    }
                },
                tooltip: {
                    enabled: true,
                    backgroundColor: 'rgba(20, 25, 40, 0.95)',
                    titleColor: '#FFFFFF',
                    bodyColor: '#B8C1EC',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Iteration',
                        color: '#B8C1EC',
                        font: {
                            family: 'Inter',
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#6B7AA1',
                        font: {
                            family: 'JetBrains Mono',
                            size: 10
                        }
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Value',
                        color: '#B8C1EC',
                        font: {
                            family: 'Inter',
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#6B7AA1',
                        font: {
                            family: 'JetBrains Mono',
                            size: 10
                        }
                    }
                }
            }
        }
    });
}

/**
 * Update convergence chart with new data point
 */
function updateConvergenceChart(iteration, cost, gradientNorm = null) {
    if (!convergenceChart) return;

    // Add data point
    convergenceChart.data.labels.push(iteration);
    convergenceChart.data.datasets[0].data.push(cost);

    if (gradientNorm !== null) {
        convergenceChart.data.datasets[1].data.push(gradientNorm);
    }

    // Keep only last 100 points for performance
    if (convergenceChart.data.labels.length > 100) {
        convergenceChart.data.labels.shift();
        convergenceChart.data.datasets[0].data.shift();
        if (convergenceChart.data.datasets[1].data.length > 0) {
            convergenceChart.data.datasets[1].data.shift();
        }
    }

    convergenceChart.update('none'); // Update without animation for smooth real-time
}

/**
 * Reset convergence chart
 */
function resetConvergenceChart() {
    if (!convergenceChart) return;

    convergenceChart.data.labels = [];
    convergenceChart.data.datasets[0].data = [];
    convergenceChart.data.datasets[1].data = [];
    convergenceChart.update();
}

/**
 * Initialize quantum circuit visualization
 */
function initQuantumCircuitViz() {
    const container = document.getElementById('quantum-circuit');
    const canvas = document.getElementById('circuit-canvas');

    // Set canvas size
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    circuitCanvas = canvas;
    circuitCtx = canvas.getContext('2d');

    // Draw initial circuit
    drawQuantumCircuit();
}

/**
 * Draw quantum circuit diagram
 */
function drawQuantumCircuit(nQubits = 8, nLayers = 3) {
    if (!circuitCtx) return;

    const ctx = circuitCtx;
    const width = circuitCanvas.width;
    const height = circuitCanvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Circuit parameters
    const padding = 30;
    const qubitSpacing = (height - 2 * padding) / (nQubits + 1);
    const layerSpacing = (width - 2 * padding) / (nLayers + 2);

    // Draw qubit wires
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1;

    for (let i = 0; i < nQubits; i++) {
        const y = padding + (i + 1) * qubitSpacing;

        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();

        // Qubit label
        ctx.fillStyle = '#B8C1EC';
        ctx.font = '12px JetBrains Mono';
        ctx.textAlign = 'right';
        ctx.fillText(`q${i}`, padding - 10, y + 4);
    }

    // Draw gates
    for (let layer = 0; layer < nLayers; layer++) {
        const x = padding + (layer + 1) * layerSpacing;

        // Single-qubit gates (Ry, Rz)
        for (let i = 0; i < nQubits; i++) {
            const y = padding + (i + 1) * qubitSpacing;

            // Ry gate
            drawGate(ctx, x, y, 'Ry', '#00D9FF');

            // Rz gate (offset)
            if (layer < nLayers - 1) {
                drawGate(ctx, x + layerSpacing * 0.3, y, 'Rz', '#9D4EDD');
            }
        }

        // Entangling gates (CNOT)
        if (layer < nLayers) {
            const entangleX = x + layerSpacing * 0.6;

            for (let i = 0; i < nQubits - 1; i++) {
                const y1 = padding + (i + 1) * qubitSpacing;
                const y2 = padding + (i + 2) * qubitSpacing;

                drawCNOT(ctx, entangleX, y1, y2);
            }

            // Wrap-around CNOT
            if (nQubits > 2) {
                const y1 = padding + nQubits * qubitSpacing;
                const y2 = padding + qubitSpacing;
                drawCNOT(ctx, entangleX, y1, y2, true);
            }
        }
    }

    // Measurement symbols
    const measureX = padding + (nLayers + 1) * layerSpacing;
    for (let i = 0; i < nQubits; i++) {
        const y = padding + (i + 1) * qubitSpacing;
        drawMeasurement(ctx, measureX, y);
    }
}

/**
 * Draw a single-qubit gate
 */
function drawGate(ctx, x, y, label, color) {
    const size = 20;

    // Gate box
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(x - size / 2, y - size / 2, size, size);

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x - size / 2, y - size / 2, size, size);

    // Label
    ctx.fillStyle = color;
    ctx.font = 'bold 10px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(label, x, y + 4);
}

/**
 * Draw a CNOT gate
 */
function drawCNOT(ctx, x, y1, y2, dashed = false) {
    // Connection line
    ctx.strokeStyle = '#FF006E';
    ctx.lineWidth = 2;

    if (dashed) {
        ctx.setLineDash([4, 4]);
    }

    ctx.beginPath();
    ctx.moveTo(x, y1);
    ctx.lineTo(x, y2);
    ctx.stroke();

    ctx.setLineDash([]);

    // Control dot
    ctx.fillStyle = '#FF006E';
    ctx.beginPath();
    ctx.arc(x, y1, 4, 0, Math.PI * 2);
    ctx.fill();

    // Target circle
    ctx.strokeStyle = '#FF006E';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y2, 8, 0, Math.PI * 2);
    ctx.stroke();

    // Target cross
    ctx.beginPath();
    ctx.moveTo(x - 8, y2);
    ctx.lineTo(x + 8, y2);
    ctx.moveTo(x, y2 - 8);
    ctx.lineTo(x, y2 + 8);
    ctx.stroke();
}

/**
 * Draw measurement symbol
 */
function drawMeasurement(ctx, x, y) {
    const size = 16;

    // Box
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(x - size / 2, y - size / 2, size, size);

    ctx.strokeStyle = '#06FFA5';
    ctx.lineWidth = 2;
    ctx.strokeRect(x - size / 2, y - size / 2, size, size);

    // Meter symbol
    ctx.strokeStyle = '#06FFA5';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, y + 2, 6, Math.PI, 0, false);
    ctx.stroke();

    // Needle
    ctx.beginPath();
    ctx.moveTo(x, y + 2);
    ctx.lineTo(x + 4, y - 2);
    ctx.stroke();
}

/**
 * Update quantum circuit visualization
 */
function updateQuantumCircuit() {
    const nQubits = parseInt(document.getElementById('qubits').value) || 8;
    const nLayers = parseInt(document.getElementById('layers').value) || 3;

    drawQuantumCircuit(nQubits, nLayers);
}

// Listen for quantum setting changes
document.addEventListener('DOMContentLoaded', () => {
    const qubitsSlider = document.getElementById('qubits');
    const layersSlider = document.getElementById('layers');

    if (qubitsSlider) {
        qubitsSlider.addEventListener('input', updateQuantumCircuit);
    }

    if (layersSlider) {
        layersSlider.addEventListener('input', updateQuantumCircuit);
    }
});

console.log('Quantum visualization initialized');
