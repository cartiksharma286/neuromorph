/**
 * Dementia Dashboard Module
 * Cognitive assessment tracking and disease progression monitoring
 */

class DementiaDashboard {
    constructor() {
        this.cognitiveScores = {
            mmse: 18.0,
            moca: 16.0,
            memory_encoding: 0.5,
            memory_retrieval: 0.5
        };
        this.predictionCanvas = null;
        this.activityCanvas = null;
    }

    async init() {
        this.setupCanvases();
        this.setupSimulateButton();
        this.setupPredictButton();
        await this.loadDementiaState();
    }

    setupCanvases() {
        this.predictionCanvas = document.getElementById('dementiaPredictionCanvas');
        this.activityCanvas = document.getElementById('dementiaActivityCanvas');
    }

    async loadDementiaState() {
        try {
            const response = await window.app.get('/dementia/state');
            this.updateCognitiveScores(response.cognitive_scores);
            this.updateBiomarkers(response.biomarkers);
            this.visualizeNeuralActivity(response.activity);
        } catch (error) {
            console.error('Failed to load dementia state:', error);
        }
    }

    updateCognitiveScores(scores) {
        this.cognitiveScores = scores;

        // Update MMSE display
        const mmseValue = document.getElementById('metricMMSE');
        const mmseBar = document.getElementById('barMMSE');
        if (mmseValue && mmseBar) {
            mmseValue.textContent = scores.mmse.toFixed(1);
            mmseBar.style.width = `${(scores.mmse / 30) * 100}%`;

            // Color based on severity
            if (scores.mmse < 10) {
                mmseBar.style.background = 'linear-gradient(90deg, #ff3366, #ffaa00)';
            } else if (scores.mmse < 20) {
                mmseBar.style.background = 'linear-gradient(90deg, #ffaa00, #00d4ff)';
            } else {
                mmseBar.style.background = 'linear-gradient(90deg, #00d4ff, #00ff88)';
            }
        }

        // Update MoCA display
        const mocaValue = document.getElementById('metricMoCA');
        const mocaBar = document.getElementById('barMoCA');
        if (mocaValue && mocaBar) {
            mocaValue.textContent = scores.moca.toFixed(1);
            mocaBar.style.width = `${(scores.moca / 30) * 100}%`;
        }

        // Update memory scores
        const memoryEncValue = document.getElementById('metricMemoryEncoding');
        const memoryEncBar = document.getElementById('barMemoryEncoding');
        if (memoryEncValue && memoryEncBar) {
            memoryEncValue.textContent = (scores.memory_encoding * 100).toFixed(0);
            memoryEncBar.style.width = `${scores.memory_encoding * 100}%`;
        }

        const memoryRetValue = document.getElementById('metricMemoryRetrieval');
        const memoryRetBar = document.getElementById('barMemoryRetrieval');
        if (memoryRetValue && memoryRetBar) {
            memoryRetValue.textContent = (scores.memory_retrieval * 100).toFixed(0);
            memoryRetBar.style.width = `${scores.memory_retrieval * 100}%`;
        }

        // Update disease stage
        const stageEl = document.getElementById('diseaseStage');
        if (stageEl) {
            stageEl.textContent = scores.disease_stage || 'Unknown';
        }
    }

    updateBiomarkers(biomarkers) {
        const container = document.getElementById('dementiaBiomarkers');
        if (!container) return;

        container.innerHTML = `
            <div class="metric-item">
                <div class="metric-item-label">Acetylcholine</div>
                <div class="metric-item-value">${biomarkers.acetylcholine_level_percent}%</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Amyloid-β</div>
                <div class="metric-item-value">${biomarkers.amyloid_beta_burden.toFixed(2)}</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Tau Tangles</div>
                <div class="metric-item-value">${biomarkers.tau_tangles_burden.toFixed(2)}</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Hippocampal Activity</div>
                <div class="metric-item-value">${(biomarkers.hippocampal_activity * 100).toFixed(0)}%</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Entorhinal Activity</div>
                <div class="metric-item-value">${(biomarkers.entorhinal_activity * 100).toFixed(0)}%</div>
            </div>
        `;
    }

    setupSimulateButton() {
        const btn = document.getElementById('simulateDementiaBtn');
        if (!btn) return;

        btn.addEventListener('click', async () => {
            btn.disabled = true;
            btn.innerHTML = '<div class="loading"></div> Simulating...';

            try {
                const params = {
                    target_region: document.getElementById('dementiaTargetRegion')?.value || 'nucleus_basalis',
                    amplitude_ma: parseFloat(document.getElementById('dementiaAmplitude')?.value || 3.0),
                    frequency_hz: parseFloat(document.getElementById('dementiaFrequency')?.value || 20),
                    pulse_width_us: parseFloat(document.getElementById('dementiaPulseWidth')?.value || 90)
                };

                const response = await window.app.post('/dementia/simulate', params);

                if (response.success) {
                    this.updateCognitiveScores(response.cognitive_scores);
                    this.visualizeNeuralActivity(response.activity);

                    btn.innerHTML = '✓ Simulation Complete';
                    setTimeout(() => {
                        btn.innerHTML = '<span>▶</span> Simulate Stimulation';
                        btn.disabled = false;
                    }, 2000);
                }
            } catch (error) {
                console.error('Simulation failed:', error);
                btn.innerHTML = '✗ Simulation Failed';
                btn.disabled = false;
            }
        });
    }

    setupPredictButton() {
        const btn = document.getElementById('predictDementiaBtn');
        if (!btn) return;

        btn.addEventListener('click', async () => {
            const months = parseInt(document.getElementById('treatmentMonths')?.value || 6);
            await this.predictTreatment(months);
        });
    }

    async predictTreatment(months) {
        const btn = document.getElementById('predictDementiaBtn');
        btn.disabled = true;
        btn.innerHTML = '<div class="loading"></div> Predicting...';

        try {
            const params = {
                target_region: 'nucleus_basalis',
                amplitude_ma: 3.0,
                frequency_hz: 20,
                pulse_width_us: 90,
                treatment_months: months
            };

            const response = await window.app.post('/dementia/predict', params);

            if (response.success) {
                this.visualizePrediction(response.monthly_progression);

                const responder = response.responder ? '✓ Responder' : '✗ Non-responder';
                const responseRate = (response.response_rate * 100).toFixed(1);

                alert(`Treatment Prediction:\n${responder}\nResponse Rate: ${responseRate}%`);
            }
        } catch (error) {
            console.error('Prediction failed:', error);
        } finally {
            btn.disabled = false;
            btn.innerHTML = 'Predict Response';
        }
    }

    visualizePrediction(progression) {
        if (!this.predictionCanvas) return;

        const ctx = this.predictionCanvas.getContext('2d');
        const width = this.predictionCanvas.width = this.predictionCanvas.offsetWidth;
        const height = this.predictionCanvas.height = this.predictionCanvas.offsetHeight;

        // Clear canvas
        ctx.fillStyle = '#0a0a0a';
        ctx.fillRect(0, 0, width, height);

        // Draw grid
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        for (let y = 0; y < height; y += 40) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Draw MMSE and MoCA progression
        const metrics = ['mmse', 'moca'];
        const colors = ['#00d4ff', '#00ff88'];
        const labels = ['MMSE', 'MoCA'];

        metrics.forEach((metric, idx) => {
            ctx.strokeStyle = colors[idx];
            ctx.lineWidth = 2;
            ctx.beginPath();

            progression.forEach((month, i) => {
                const x = (i / (progression.length - 1)) * width;
                const y = height - (month[metric] / 30 * height * 0.8) - height * 0.1;

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });

            ctx.stroke();

            // Label
            ctx.fillStyle = colors[idx];
            ctx.font = '12px Inter';
            ctx.fillText(labels[idx], 10, 20 + idx * 15);
        });

        // Axis labels
        ctx.fillStyle = '#b0b0b0';
        ctx.font = '12px Inter';
        ctx.fillText('Months →', width - 70, height - 10);
        ctx.fillText('Score', 10, height - 10);
    }

    visualizeNeuralActivity(activity) {
        if (!this.activityCanvas) return;

        const ctx = this.activityCanvas.getContext('2d');
        const width = this.activityCanvas.width = this.activityCanvas.offsetWidth;
        const height = this.activityCanvas.height = this.activityCanvas.offsetHeight;

        // Clear canvas
        ctx.fillStyle = '#0a0a0a';
        ctx.fillRect(0, 0, width, height);

        // Draw brain regions as bars
        const regions = Object.entries(activity);
        const barWidth = width / (regions.length * 2);
        const spacing = barWidth * 0.5;

        regions.forEach(([region, value], i) => {
            const x = i * (barWidth + spacing) + spacing;
            const barHeight = value * height * 0.8;
            const y = height - barHeight;

            // Gradient based on activity level
            const gradient = ctx.createLinearGradient(x, y, x, height);
            if (value > 0.6) {
                gradient.addColorStop(0, '#00ff88');
                gradient.addColorStop(1, '#00d4ff');
            } else {
                gradient.addColorStop(0, '#ffaa00');
                gradient.addColorStop(1, '#ff3366');
            }

            ctx.fillStyle = gradient;
            ctx.fillRect(x, y, barWidth, barHeight);

            // Border
            ctx.strokeStyle = '#00d4ff';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, barWidth, barHeight);

            // Label
            ctx.fillStyle = '#b0b0b0';
            ctx.font = '10px Inter';
            ctx.save();
            ctx.translate(x + barWidth / 2, height - 5);
            ctx.rotate(-Math.PI / 4);
            ctx.fillText(region.replace('_', ' '), 0, 0);
            ctx.restore();

            // Value
            ctx.fillStyle = '#ffffff';
            ctx.font = '12px Inter';
            ctx.textAlign = 'center';
            ctx.fillText((value * 100).toFixed(0), x + barWidth / 2, y - 5);
        });
    }
}

window.DementiaDashboard = DementiaDashboard;
