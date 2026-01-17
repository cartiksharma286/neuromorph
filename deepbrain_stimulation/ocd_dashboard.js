
/**
 * OCD Dashboard Controller
 * Handles OCD simulation and statistical trial UI
 */

class OCDDashboard {
    constructor() {
        this.chart = null;
    }

    init() {
        console.log("OCD Dashboard Initialized");
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Single Simulation Button
        const singleBtn = document.getElementById('runSingleOcdBtn');
        if (singleBtn) {
            singleBtn.addEventListener('click', () => this.runSingleSimulation());
        }

        // Trial Button
        const trialBtn = document.getElementById('runTrialBtn');
        if (trialBtn) {
            trialBtn.addEventListener('click', () => this.runStatisticalTrial());
        }

        // Schematic Button
        const schematicBtn = document.getElementById('loadOcdSchematicBtn');
        if (schematicBtn) {
            schematicBtn.addEventListener('click', () => this.loadSchematic());
        }
    }

    async runSingleSimulation() {
        const target = document.getElementById('ocdTarget').value;
        const freq = parseFloat(document.getElementById('ocdFreq').value);
        const amp = parseFloat(document.getElementById('ocdAmp').value);

        const btn = document.getElementById('runSingleOcdBtn');
        const originalText = btn.innerHTML;
        btn.innerHTML = 'Running...';
        btn.disabled = true;

        try {
            const result = await window.app.post('/ocd/simulate', {
                target: target,
                frequency: freq,
                amplitude: amp
            });

            if (result.success) {
                // Update Values
                document.getElementById('valGain').textContent = result.post.gain.toFixed(2);
                document.getElementById('valYbocs').textContent = result.post.ybocs.toFixed(1);

                // Color coding
                document.getElementById('valGain').style.color = result.post.gain > 1.0 ? '#ff3366' : '#00ff88';
                document.getElementById('valYbocs').style.color = result.post.ybocs > 24 ? '#ff3366' : '#00ff88';
            }
        } catch (error) {
            console.error("Simulation failed", error);
            alert("Simulation failed: " + error.message);
        } finally {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    }

    async runStatisticalTrial() {
        const target = document.getElementById('ocdTarget').value;
        const freq = parseFloat(document.getElementById('ocdFreq').value);
        const amp = parseFloat(document.getElementById('ocdAmp').value);
        const n = parseInt(document.getElementById('trialN').value);
        const alpha = parseFloat(document.getElementById('trialAlpha').value);

        const btn = document.getElementById('runTrialBtn');
        btn.innerHTML = 'Running Clinical Trial...';
        btn.disabled = true;

        try {
            const response = await window.app.post('/ocd/trial', {
                n_subjects: n,
                target: target,
                frequency: freq,
                amplitude: amp
            });

            if (response.success) {
                const res = response.results;

                // Show Results Div
                document.getElementById('trialResults').style.display = 'block';

                // Calcs
                const reduction = ((res.mean_pre - res.mean_post) / res.mean_pre * 100).toFixed(1);

                document.getElementById('resRed').textContent = `${reduction}%`;
                document.getElementById('resP').textContent = res.p_value.toExponential(2);

                const sigDiv = document.getElementById('resSig');
                if (res.p_value < alpha) {
                    sigDiv.textContent = "STATISTICALLY SIGNIFICANT IMPROVEMENT";
                    sigDiv.style.color = "#00ff88";
                } else {
                    sigDiv.textContent = "NO SIGNIFICANT IMPROVEMENT";
                    sigDiv.style.color = "#ff3366";
                }

                // Plot
                this.plotTrialResults(res.pre_scores, res.post_scores);
            }

        } catch (error) {
            console.error("Trial failed", error);
            alert("Trial failed: " + error.message);
        } finally {
            btn.innerHTML = '<span>📊</span> Run Statistical Trial';
            btn.disabled = false;
        }
    }

    plotTrialResults(preScores, postScores) {
        const ctx = document.getElementById('ocdTrialCanvas').getContext('2d');

        if (this.chart) {
            this.chart.destroy();
        }

        // Create histograms (simplified as bins)
        const bins = Array(9).fill(0).map((_, i) => i * 5); // 0, 5, 10... 40
        const preHist = bins.map(b => preScores.filter(s => s >= b && s < b + 5).length);
        const postHist = bins.map(b => postScores.filter(s => s >= b && s < b + 5).length);

        this.chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: bins.map(b => `${b}-${b + 5}`),
                datasets: [
                    {
                        label: 'Pre-Treatment (Baseline)',
                        data: preHist,
                        backgroundColor: 'rgba(255, 99, 132, 0.5)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Post-DBS Treatment',
                        data: postHist,
                        backgroundColor: 'rgba(75, 192, 192, 0.5)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Patients',
                            color: '#ccc'
                        },
                        grid: { color: '#333' }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'YBOCS Score Range',
                            color: '#ccc'
                        },
                        grid: { color: '#333' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#fff' }
                    },
                    title: {
                        display: true,
                        text: 'Patient Distribution shifts Left (Improvement)',
                        color: '#fff'
                    }
                }
            }
        });
    }

    async loadSchematic() {
        const btn = document.getElementById('loadOcdSchematicBtn');
        const container = document.getElementById('ocdSchematicContent');

        if (container.style.display === 'block') {
            container.style.display = 'none';
            btn.textContent = 'Show Schematic';
            return;
        }

        btn.textContent = 'Loading...';
        btn.disabled = true;

        try {
            const response = await window.app.get('/ocd/schematic');
            if (response.success) {
                const s = response.schematic;

                // Render Specs
                let specsHtml = `<strong>Application:</strong> ${s.application}<br/><br/>`;
                specsHtml += `<strong>Target Structures:</strong><br/>${s.target_structures.map(t => `- ${t}`).join('<br/>')}<br/><br/>`;

                specsHtml += `<strong>Electrical Specs:</strong><br/>`;
                specsHtml += `- Mode: ${s.electrical_specifications.stimulation_mode}<br/>`;
                specsHtml += `- Freq: ${s.electrical_specifications.frequency_range.optimal}<br/>`;
                specsHtml += `- Amp: ${s.electrical_specifications.amplitude_range.typical_therapeutic}<br/>`;
                specsHtml += `- Pulse Width: ${s.electrical_specifications.pulse_width.typical}<br/><br/>`;

                specsHtml += `<strong>Safety:</strong><br/>`;
                specsHtml += `- Max Charge: ${s.safety_limits.max_charge_density}<br/>`;
                specsHtml += `- Compliance: ${s.safety_limits.max_voltage_compliance}<br/>`;

                document.getElementById('ocdSpecsList').innerHTML = specsHtml;

                // Render SVG
                document.getElementById('ocdSvgContainer').innerHTML = s.svg_schematic;

                // Show
                container.style.display = 'block';
                btn.textContent = 'Hide Schematic';
            }
        } catch (error) {
            console.error("Failed to load schematic", error);
            alert("Could not load schematic.");
        } finally {
            btn.disabled = false;
        }
    }
}
