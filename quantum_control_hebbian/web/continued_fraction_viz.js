// ===== Continued Fractions Visualization =====

class ContinuedFractionVisualization {
    constructor() {
        this.padeContainer = document.getElementById('pade-convergence-viz');
        this.convergentContainer = document.getElementById('convergent-sequence-viz');
        this.errorContainer = document.getElementById('cf-error-viz');

        this.initCharts();
    }

    initCharts() {
        // Padé Approximant Convergence Chart
        const padeCtx = this.padeContainer;
        if (padeCtx) {
            this.padeChart = new Chart(padeCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Original Function',
                            data: [],
                            borderColor: '#6366f1',
                            backgroundColor: 'rgba(99, 102, 241, 0.1)',
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0.4
                        },
                        {
                            label: 'Padé [4/4]',
                            data: [],
                            borderColor: '#ec4899',
                            backgroundColor: 'rgba(236, 72, 153, 0.1)',
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0.4,
                            borderDash: [5, 5]
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            labels: { color: '#adb5bd' }
                        },
                        title: {
                            display: true,
                            text: 'Sigmoid Function vs Padé Approximant',
                            color: '#f8f9fa',
                            font: { size: 14 }
                        }
                    },
                    scales: {
                        y: {
                            title: { display: true, text: 'f(x)', color: '#adb5bd' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#adb5bd' }
                        },
                        x: {
                            title: { display: true, text: 'x', color: '#adb5bd' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#adb5bd' }
                        }
                    }
                }
            });

            // Initialize with sigmoid function
            this.updatePadeChart();
        }

        // Convergent Sequence Chart
        const convergentCtx = this.convergentContainer;
        if (convergentCtx) {
            this.convergentChart = new Chart(convergentCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Convergent Value',
                        data: [],
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        borderWidth: 2,
                        pointRadius: 4,
                        pointBackgroundColor: '#8b5cf6',
                        tension: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        title: {
                            display: true,
                            text: 'Convergent Sequence',
                            color: '#f8f9fa',
                            font: { size: 14 }
                        }
                    },
                    scales: {
                        y: {
                            title: { display: true, text: 'Cₙ', color: '#adb5bd' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#adb5bd' }
                        },
                        x: {
                            title: { display: true, text: 'n', color: '#adb5bd' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#adb5bd' }
                        }
                    }
                }
            });
        }

        // Error Analysis Chart
        const errorCtx = this.errorContainer;
        if (errorCtx) {
            this.errorChart = new Chart(errorCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Convergence Error',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        pointRadius: 3,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        title: {
                            display: true,
                            text: 'Convergence Error (log scale)',
                            color: '#f8f9fa',
                            font: { size: 14 }
                        }
                    },
                    scales: {
                        y: {
                            type: 'logarithmic',
                            title: { display: true, text: 'Error', color: '#adb5bd' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#adb5bd' }
                        },
                        x: {
                            title: { display: true, text: 'Iteration', color: '#adb5bd' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#adb5bd' }
                        }
                    }
                }
            });
        }
    }

    updatePadeChart() {
        if (!this.padeChart) return;

        const xValues = [];
        const originalValues = [];
        const padeValues = [];

        // Generate sigmoid and Padé approximation
        for (let x = -5; x <= 5; x += 0.1) {
            xValues.push(x.toFixed(1));

            // Original sigmoid
            const sigmoid = 1 / (1 + Math.exp(-x));
            originalValues.push(sigmoid);

            // Padé [4/4] approximation for sigmoid
            const pade = this.padeApproximant(x);
            padeValues.push(pade);
        }

        this.padeChart.data.labels = xValues;
        this.padeChart.data.datasets[0].data = originalValues;
        this.padeChart.data.datasets[1].data = padeValues;
        this.padeChart.update();
    }

    padeApproximant(x) {
        // Padé [4/4] approximation for sigmoid
        // This is a simplified version
        const num = 1 + 0.5 * x + 0.125 * x * x + 0.0208 * x * x * x + 0.0026 * x * x * x * x;
        const den = 1 + 0.5 * Math.abs(x) + 0.125 * x * x + 0.0208 * Math.abs(x * x * x) + 0.0026 * x * x * x * x;
        return num / den;
    }

    update(cfData) {
        if (!cfData) return;

        // Update convergent sequence
        if (this.convergentChart && cfData.convergents) {
            const labels = cfData.convergents.map(c => c.n);
            const values = cfData.convergents.map(c => c.value);

            this.convergentChart.data.labels = labels;
            this.convergentChart.data.datasets[0].data = values;
            this.convergentChart.update('none');
        }

        // Update error chart
        if (this.errorChart && cfData.convergents) {
            const labels = cfData.convergents.map(c => c.n);
            const errors = cfData.convergents.map(c => c.error);

            this.errorChart.data.labels = labels;
            this.errorChart.data.datasets[0].data = errors;
            this.errorChart.update('none');
        }

        // Update Padé chart if new data provided
        if (cfData.padeApproximants) {
            const xValues = cfData.padeApproximants.map(p => p.x.toFixed(1));
            const originalValues = cfData.padeApproximants.map(p => p.original);
            const padeValues = cfData.padeApproximants.map(p => p.pade);

            this.padeChart.data.labels = xValues;
            this.padeChart.data.datasets[0].data = originalValues;
            this.padeChart.data.datasets[1].data = padeValues;
            this.padeChart.update('none');
        }
    }

    reset() {
        if (this.padeChart) {
            this.updatePadeChart();
        }

        if (this.convergentChart) {
            this.convergentChart.data.labels = [];
            this.convergentChart.data.datasets[0].data = [];
            this.convergentChart.update();
        }

        if (this.errorChart) {
            this.errorChart.data.labels = [];
            this.errorChart.data.datasets[0].data = [];
            this.errorChart.update();
        }
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.cfViz = new ContinuedFractionVisualization();
    console.log('✓ Continued Fractions Visualization initialized');
});
