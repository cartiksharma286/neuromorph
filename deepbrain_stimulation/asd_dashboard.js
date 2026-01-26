/**
 * ASD Dashboard Controller
 * Quantum Neural Repair with Continued Fractions & Stochastic Correlations
 */

class ASDDashboard {
    constructor() {
        this.charts = {};
        this.correlationMatrix = null;
    }

    init() {
        console.log("ASD Dashboard Initialized");
        this.setupEventListeners();
        this.initializeContinuedFractionViz();
    }

    setupEventListeners() {
        const optimizeBtn = document.getElementById('runAsdOptimizationBtn');
        if (optimizeBtn) {
            optimizeBtn.addEventListener('click', () => this.runQuantumOptimization());
        }

        const schematicBtn = document.getElementById('loadAsdSchematicBtn');
        if (schematicBtn) {
            schematicBtn.addEventListener('click', () => this.loadSchematic());
        }
    }

    async runQuantumOptimization() {
        const severity = document.getElementById('asdSeverity').value;
        const target = document.getElementById('asdTarget').value;
        const freq = parseFloat(document.getElementById('asdFreq').value);
        const amp = parseFloat(document.getElementById('asdAmp').value);

        const btn = document.getElementById('runAsdOptimizationBtn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<span>⚛️</span> Computing Quantum Integrals...';
        btn.disabled = true;

        try {
            const response = await window.app.post('/asd/optimize', {
                severity: severity,
                target: target,
                frequency: freq,
                amplitude: amp
            });

            if (response.success) {
                const result = response.optimization;

                // Update Quantum Surface Integrals
                document.getElementById('surfaceIntegral').textContent = result.quantum_surface_integral.toFixed(6);
                document.getElementById('transProb').textContent = (result.transition_probability * 100).toFixed(2) + '%';
                document.getElementById('corrEnergy').textContent = result.correlation_energy.toFixed(4);

                // Update Continued Fraction Metrics
                document.getElementById('convRate').textContent = result.convergence_rate.toFixed(4);
                document.getElementById('repairIdx').textContent = result.repair_index.toFixed(3);

                // Update Gedanken Experiment
                const gedanken = result.gedanken_experiment;
                if (gedanken) {
                    document.getElementById('gedankenSuperposition').textContent = gedanken.superposition_state;
                    document.getElementById('gedankenCollapse').textContent = gedanken.collapse_probability;
                    document.getElementById('gedankenEntropy').textContent = gedanken.von_neumann_entropy.toFixed(4);
                    document.getElementById('gedankenCoherence').textContent = gedanken.eigenstate_coherence;
                    document.getElementById('gedankenZeno').textContent = gedanken.pulse_observer_effect;
                }

                // Update Statistical Characteristics
                this.updateStatistics(result.pre_treatment, result.post_treatment);

                // Render visualizations
                this.renderContinuedFractionPlot(result.continued_fraction_sequence);
                this.renderCorrelationMatrix(result.correlation_matrix);
                // Bar plot removed: this.renderDistributionPlot(result.pre_treatment, result.post_treatment);

                // New Statistical Visualizations
                this.renderCongruencePlot(result.repair_timeline, result.statistical_congruence);
                this.renderInflectionPlot(result.statistical_congruence);

            }
        } catch (error) {
            console.error("Optimization failed", error);
            alert("Quantum optimization failed: " + error.message);
        } finally {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    }

    updateStatistics(pre, post) {
        // Updated to remove bar width manipulation, just text updates
        // Social Communication
        const socialImprovement = ((post.social_communication - pre.social_communication) / pre.social_communication) * 100;
        document.getElementById('socialScore').textContent = '+' + socialImprovement.toFixed(1) + '%';

        // Repetitive Behaviors (reduction is good)
        const repetitiveReduction = ((pre.repetitive_behaviors - post.repetitive_behaviors) / pre.repetitive_behaviors) * 100;
        document.getElementById('repetitiveScore').textContent = '-' + repetitiveReduction.toFixed(1) + '%';

        // Sensory Processing
        const sensoryImprovement = ((post.sensory_processing - pre.sensory_processing) / pre.sensory_processing) * 100;
        document.getElementById('sensoryScore').textContent = '+' + sensoryImprovement.toFixed(1) + '%';

        // Executive Function
        const executiveImprovement = ((post.executive_function - pre.executive_function) / pre.executive_function) * 100;
        document.getElementById('executiveScore').textContent = '+' + executiveImprovement.toFixed(1) + '%';
    }

    initializeContinuedFractionViz() {
        const canvas = document.getElementById('continuedFractionCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#1c1c1e';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.strokeStyle = '#555';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, canvas.height / 2);
        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();

        ctx.fillStyle = '#888';
        ctx.font = '12px monospace';
        ctx.fillText('Awaiting optimization...', 20, canvas.height / 2 - 10);
    }

    renderContinuedFractionPlot(sequence) {
        const canvas = document.getElementById('continuedFractionCanvas');
        if (!canvas || !sequence) return;

        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#1c1c1e';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw grid
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        for (let i = 0; i < 5; i++) {
            const y = (canvas.height / 4) * i;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
        }

        // Draw continued fraction convergence
        ctx.strokeStyle = '#00d4ff';
        ctx.lineWidth = 2;
        ctx.beginPath();

        const step = canvas.width / (sequence.length - 1);
        const maxVal = Math.max(...sequence);
        const minVal = Math.min(...sequence);
        const range = maxVal - minVal;

        sequence.forEach((val, i) => {
            const x = i * step;
            const y = canvas.height - ((val - minVal) / range) * (canvas.height - 40) - 20;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();

        // Draw convergence line
        const convergedValue = sequence[sequence.length - 1];
        const convergedY = canvas.height - ((convergedValue - minVal) / range) * (canvas.height - 40) - 20;

        ctx.strokeStyle = '#4cd964';
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(0, convergedY);
        ctx.lineTo(canvas.width, convergedY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Labels
        ctx.fillStyle = '#888';
        ctx.font = '10px monospace';
        ctx.fillText('Iterations →', canvas.width - 80, canvas.height - 5);
        ctx.fillText('Convergence', 5, 15);
    }

    renderCorrelationMatrix(matrix) {
        const canvas = document.getElementById('correlationMatrixCanvas');
        if (!canvas || !matrix) return;

        const ctx = canvas.getContext('2d');
        const size = matrix.length;
        const cellSize = canvas.width / size;

        // Clear canvas
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw correlation matrix
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const value = matrix[i][j];

                // Color mapping: -1 (blue) to 0 (black) to 1 (cyan)
                let r, g, b;
                if (value >= 0) {
                    r = 0;
                    g = Math.floor(212 * value);
                    b = Math.floor(255 * value);
                } else {
                    r = Math.floor(255 * Math.abs(value));
                    g = 0;
                    b = Math.floor(128 * Math.abs(value));
                }

                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
            }
        }

        // Draw grid
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        for (let i = 0; i <= size; i++) {
            ctx.beginPath();
            ctx.moveTo(i * cellSize, 0);
            ctx.lineTo(i * cellSize, canvas.height);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, i * cellSize);
            ctx.lineTo(canvas.width, i * cellSize);
            ctx.stroke();
        }
    }

    renderCongruencePlot(timeline, congruenceData) {
        const ctx = document.getElementById('congruenceCanvas');
        if (!ctx || !timeline) return;

        if (this.charts.congruence) this.charts.congruence.destroy();

        // Generate ideal curve for comparison
        const phases = timeline.map((_, i) => i / (timeline.length - 1));
        const ideal = phases.map(p => 1.0 / (1.0 + Math.exp(-10 * (p - 0.5))));

        // Update correlation score
        document.getElementById('congScore').textContent = `Corr: ${congruenceData.congruence_score.toFixed(3)}`;

        this.charts.congruence = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timeline.map((_, i) => `Wk ${i}`),
                datasets: [
                    {
                        label: 'Actual Repair',
                        data: timeline,
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        borderWidth: 2,
                        tension: 0.3
                    },
                    {
                        label: 'Ideal Sigmoid Model',
                        data: ideal,
                        borderColor: '#666',
                        borderDash: [5, 5],
                        borderWidth: 1,
                        pointRadius: 0,
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        grid: { color: '#333' },
                        ticks: { color: '#ccc' },
                        title: { display: true, text: 'Recovery Index' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#ccc' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#fff' } }
                }
            }
        });
    }

    renderInflectionPlot(data) {
        const ctx = document.getElementById('inflectionCanvas');
        if (!ctx || !data) return;

        if (this.charts.inflection) this.charts.inflection.destroy();

        const velocity = data.derivatives.velocity;
        const acceleration = data.derivatives.acceleration;
        const labels = velocity.map((_, i) => `Wk ${i}`);

        // Update Reasoning Text
        const reasoningDiv = document.getElementById('inflectionReasoning');
        const pts = data.inflection_points;
        reasoningDiv.innerHTML = `
            <div style="margin-bottom:4px"><span style="color:#4cd964">● Initiation (Wk ${pts.initiation_week.toFixed(0)}):</span> ${data.reasoning.phase_1}</div>
            <div style="margin-bottom:4px"><span style="color:#00d4ff">● Max Velocity (Wk ${pts.max_velocity_week.toFixed(0)}):</span> ${data.reasoning.phase_2}</div>
            <div><span style="color:#ff3b30">● Saturation (Wk ${pts.saturation_week.toFixed(0)}):</span> ${data.reasoning.phase_3}</div>
        `;

        this.charts.inflection = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Velocity (Rate of Change)',
                        data: velocity,
                        borderColor: '#9d34da', // Purple
                        borderWidth: 2,
                        yAxisID: 'y',
                        tension: 0.4
                    },
                    {
                        label: 'Acceleration (2nd Derivative)',
                        data: acceleration,
                        borderColor: '#ff9500', // Orange
                        borderWidth: 1.5,
                        yAxisID: 'y1',
                        borderDash: [2, 2],
                        tension: 0.4,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        grid: { color: '#333' },
                        title: { display: true, text: 'Velocity' }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: { display: false },
                        title: { display: true, text: 'Acceleration' }
                    },
                    x: {
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#fff' } }
                }
            }
        });
    }

    async loadSchematic() {
        const btn = document.getElementById('loadAsdSchematicBtn');
        const container = document.getElementById('asdSchematicContent');

        if (container.style.display === 'block') {
            container.style.display = 'none';
            btn.textContent = 'Show Circuit Design';
            return;
        }

        btn.textContent = 'Loading...';
        btn.disabled = true;

        try {
            // Generate schematic specs
            const specs = {
                application: 'Autism Spectrum Disorder DBS',
                target_regions: ['ACC', 'Amygdala', 'Striatum', 'Thalamus'],
                frequency: '130 Hz (optimal for ASD)',
                amplitude: '2.5 mA (therapeutic range)',
                pulse_width: '90 μs',
                mode: 'Bilateral stimulation',
                safety: 'Charge-balanced biphasic'
            };

            let specsHtml = `<strong>Application:</strong> ${specs.application}<br/><br/>`;
            specsHtml += `<strong>Target Regions:</strong><br/>${specs.target_regions.map(t => `- ${t}`).join('<br/')}<br/><br/>`;
            specsHtml += `<strong>Electrical Parameters:</strong><br/>`;
            specsHtml += `- Frequency: ${specs.frequency}<br/>`;
            specsHtml += `- Amplitude: ${specs.amplitude}<br/>`;
            specsHtml += `- Pulse Width: ${specs.pulse_width}<br/>`;
            specsHtml += `- Mode: ${specs.mode}<br/>`;
            specsHtml += `- Safety: ${specs.safety}`;

            document.getElementById('asdSpecsList').innerHTML = specsHtml;

            // Simple SVG schematic
            const svg = `
                <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
                    <rect width="400" height="300" fill="#000"/>
                    <text x="200" y="30" fill="#00d4ff" font-size="16" text-anchor="middle" font-weight="bold">ASD DBS Circuit</text>
                    
                    <!-- Pulse Generator -->
                    <rect x="50" y="60" width="100" height="60" fill="none" stroke="#00d4ff" stroke-width="2"/>
                    <text x="100" y="95" fill="#fff" font-size="12" text-anchor="middle">Pulse Gen</text>
                    
                    <!-- Electrodes -->
                    <rect x="250" y="60" width="100" height="60" fill="none" stroke="#4cd964" stroke-width="2"/>
                    <text x="300" y="95" fill="#fff" font-size="12" text-anchor="middle">Electrodes</text>
                    
                    <!-- Connection -->
                    <line x1="150" y1="90" x2="250" y2="90" stroke="#00d4ff" stroke-width="2"/>
                    
                    <!-- Brain Regions -->
                    <circle cx="100" cy="200" r="30" fill="none" stroke="#ff3366" stroke-width="2"/>
                    <text x="100" y="205" fill="#fff" font-size="10" text-anchor="middle">ACC</text>
                    
                    <circle cx="200" cy="200" r="30" fill="none" stroke="#ff3366" stroke-width="2"/>
                    <text x="200" y="205" fill="#fff" font-size="10" text-anchor="middle">Amygdala</text>
                    
                    <circle cx="300" cy="200" r="30" fill="none" stroke="#ff3366" stroke-width="2"/>
                    <text x="300" y="205" fill="#fff" font-size="10" text-anchor="middle">Striatum</text>
                    
                    <!-- Connections to brain -->
                    <line x1="300" y1="120" x2="100" y2="170" stroke="#4cd964" stroke-width="1" stroke-dasharray="5,5"/>
                    <line x1="300" y1="120" x2="200" y2="170" stroke="#4cd964" stroke-width="1" stroke-dasharray="5,5"/>
                    <line x1="300" y1="120" x2="300" y2="170" stroke="#4cd964" stroke-width="1" stroke-dasharray="5,5"/>
                </svg>
            `;

            document.getElementById('asdSvgContainer').innerHTML = svg;
            container.style.display = 'block';
            btn.textContent = 'Hide Circuit Design';

        } catch (error) {
            console.error("Failed to load schematic", error);
            alert("Could not load schematic.");
        } finally {
            btn.disabled = false;
        }
    }
}

// Auto-initialize if DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        if (typeof window.asdDashboard === 'undefined') {
            window.asdDashboard = new ASDDashboard();
        }
    });
} else {
    if (typeof window.asdDashboard === 'undefined') {
        window.asdDashboard = new ASDDashboard();
    }
}
