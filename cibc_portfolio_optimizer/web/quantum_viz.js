/**
 * Quantum Visualization Module
 * Real-time visualization of quantum optimization progress
 */

class QuantumVisualizer {
    constructor() {
        this.convergenceData = [];
        this.energyChart = null;
    }

    initializeConvergenceChart(canvasId) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        this.energyChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Energy',
                    data: [],
                    borderColor: '#8B5CF6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#B8C1EC' }
                    },
                    title: {
                        display: true,
                        text: 'VQE Convergence',
                        color: '#FFFFFF'
                    }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'Energy', color: '#B8C1EC' },
                        ticks: { color: '#B8C1EC' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        title: { display: true, text: 'Iteration', color: '#B8C1EC' },
                        ticks: { color: '#B8C1EC' },
                        grid: { display: false }
                    }
                }
            }
        });
    }

    updateConvergence(optimizationHistory) {
        if (!optimizationHistory || optimizationHistory.length === 0) return;

        this.convergenceData = optimizationHistory;

        if (this.energyChart) {
            const labels = optimizationHistory.map(h => h.iteration);
            const energies = optimizationHistory.map(h => h.energy);

            this.energyChart.data.labels = labels;
            this.energyChart.data.datasets[0].data = energies;
            this.energyChart.update();
        }
    }

    visualizeQuantumCircuit(numQubits, circuitDepth) {
        // Create ASCII-style quantum circuit visualization
        const gates = ['H', 'RY', 'RZ', 'CX'];
        let circuit = '';

        for (let qubit = 0; qubit < numQubits; qubit++) {
            circuit += `q${qubit}: `;
            for (let depth = 0; depth < circuitDepth; depth++) {
                const gate = gates[Math.floor(Math.random() * gates.length)];
                circuit += `--[${gate}]--`;
            }
            circuit += '\n';
        }

        return circuit;
    }

    getQuantumMetricsSummary(metrics) {
        return {
            efficiency: this.calculateCircuitEfficiency(metrics),
            convergenceRate: this.calculateConvergenceRate(),
            quantumAdvantage: this.estimateQuantumAdvantage(metrics)
        };
    }

    calculateCircuitEfficiency(metrics) {
        // Efficiency based on circuit depth vs qubits
        if (!metrics.circuit_depth || !metrics.num_qubits) return 0;

        const idealDepth = metrics.num_qubits * 2;
        const efficiency = Math.max(0, 100 - Math.abs(metrics.circuit_depth - idealDepth) / idealDepth * 100);
        return efficiency.toFixed(1);
    }

    calculateConvergenceRate() {
        if (this.convergenceData.length < 2) return 0;

        const firstEnergy = this.convergenceData[0].energy;
        const lastEnergy = this.convergenceData[this.convergenceData.length - 1].energy;
        const improvement = Math.abs(lastEnergy - firstEnergy);

        return (improvement / this.convergenceData.length).toFixed(4);
    }

    estimateQuantumAdvantage(metrics) {
        // Estimate quantum advantage based on problem size
        const numQubits = metrics.num_qubits || 0;

        if (numQubits < 10) return 'Limited';
        if (numQubits < 20) return 'Moderate';
        return 'Significant';
    }

    animateQuantumState(elementId) {
        const element = document.getElementById(elementId);
        if (!element) return;

        // Create animated quantum state visualization
        const states = ['|0⟩', '|1⟩', '|+⟩', '|-⟩', '|ψ⟩'];
        let index = 0;

        setInterval(() => {
            element.textContent = states[index];
            index = (index + 1) % states.length;
        }, 500);
    }

    visualizeEntanglement(numQubits) {
        // Create visual representation of qubit entanglement
        const connections = [];

        for (let i = 0; i < numQubits - 1; i++) {
            connections.push({
                from: i,
                to: i + 1,
                strength: Math.random()
            });
        }

        // Ring connection
        if (numQubits > 2) {
            connections.push({
                from: numQubits - 1,
                to: 0,
                strength: Math.random()
            });
        }

        return connections;
    }

    getGateCount(optimizationHistory) {
        if (!optimizationHistory || optimizationHistory.length === 0) return 0;

        // Estimate total gate count
        const params = optimizationHistory[0].params || [];
        return params.length;
    }

    formatQuantumMetrics(metrics) {
        return {
            'Circuit Depth': metrics.circuit_depth || 'N/A',
            'Qubits': metrics.num_qubits || 'N/A',
            'Parameters': metrics.num_parameters || 'N/A',
            'Iterations': metrics.convergence_iterations || 'N/A',
            'Final Energy': (metrics.final_energy || 0).toFixed(4),
            'Gate Count': this.getGateCount(metrics.optimization_history || [])
        };
    }
}

// Export for use in main app
window.QuantumVisualizer = QuantumVisualizer;
