/**
 * Clinical Dashboard Module
 * PTSD symptom tracking and treatment monitoring
 */

class ClinicalDashboard {
    constructor() {
        this.symptoms = {
            hyperarousal: 0.7,
            re_experiencing: 0.6,
            avoidance: 0.5,
            negative_cognition: 0.6
        };
        this.predictionCanvas = null;
        this.neuralCanvas = null;
    }

    init() {
        this.setupPredictionButton();
        this.setupCanvases();
        this.loadNeuralState();
    }

    setupPredictionButton() {
        const btn = document.getElementById('predictBtn');
        if (!btn) return;

        btn.addEventListener('click', async () => {
            const weeks = parseInt(document.getElementById('treatmentWeeks').value);
            await this.predictTreatment(weeks);
        });
    }

    setupCanvases() {
        this.predictionCanvas = document.getElementById('predictionCanvas');
        this.neuralCanvas = document.getElementById('neuralCanvas');
    }

    updateSymptoms(symptoms) {
        this.symptoms = symptoms;

        // Update metric cards
        const metrics = [
            { id: 'Hyperarousal', value: symptoms.hyperarousal },
            { id: 'Reexperiencing', value: symptoms.re_experiencing },
            { id: 'Avoidance', value: symptoms.avoidance },
            { id: 'NegativeCognition', value: symptoms.negative_cognition }
        ];

        metrics.forEach(metric => {
            const valueEl = document.getElementById(`metric${metric.id}`);
            const barEl = document.getElementById(`bar${metric.id}`);

            if (valueEl && barEl) {
                valueEl.textContent = (metric.value * 100).toFixed(0);
                barEl.style.width = `${metric.value * 100}%`;

                // Color based on severity
                if (metric.value > 0.7) {
                    barEl.style.background = 'linear-gradient(90deg, #ff3366, #ffaa00)';
                } else if (metric.value > 0.4) {
                    barEl.style.background = 'linear-gradient(90deg, #ffaa00, #00d4ff)';
                } else {
                    barEl.style.background = 'linear-gradient(90deg, #00d4ff, #00ff88)';
                }
            }
        });
    }

    updateBiomarkers(biomarkers) {
        const container = document.getElementById('biomarkers');
        if (!container) return;

        container.innerHTML = `
            <div class="metric-item">
                <div class="metric-item-label">Heart Rate Variability</div>
                <div class="metric-item-value">${biomarkers.heart_rate_variability_ms} ms</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Cortisol</div>
                <div class="metric-item-value">${biomarkers.cortisol_ug_dl} μg/dL</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Skin Conductance</div>
                <div class="metric-item-value">${biomarkers.skin_conductance_us} μS</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Sleep Quality</div>
                <div class="metric-item-value">${(biomarkers.sleep_quality_0_1 * 100).toFixed(0)}%</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Amygdala Activity</div>
                <div class="metric-item-value">${(biomarkers.amygdala_activity * 100).toFixed(0)}%</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">vmPFC Activity</div>
                <div class="metric-item-value">${(biomarkers.vmPFC_activity * 100).toFixed(0)}%</div>
            </div>
        `;
    }

    async loadNeuralState() {
        try {
            const response = await window.app.get('/neural/state');
            this.updateSymptoms(response.symptoms);
            this.updateBiomarkers(response.biomarkers);
            this.visualizeNeuralActivity(response.activity);
        } catch (error) {
            console.error('Failed to load neural state:', error);
        }
    }

    async predictTreatment(weeks) {
        const btn = document.getElementById('predictBtn');
        btn.disabled = true;
        btn.innerHTML = '<div class="loading"></div> Predicting...';

        try {
            const response = await window.app.post('/neural/predict', {
                target_region: 'amygdala',
                amplitude_ma: 3.0,
                frequency_hz: 130,
                pulse_width_us: 90,
                treatment_weeks: weeks
            });

            if (response.success) {
                this.visualizePrediction(response.weekly_progression);

                // Show summary
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

        // Draw symptom progression
        const symptoms = ['hyperarousal', 're_experiencing', 'avoidance', 'negative_cognition'];
        const colors = ['#00d4ff', '#00ff88', '#ff3366', '#ffaa00'];
        const labels = ['Hyperarousal', 'Re-experiencing', 'Avoidance', 'Neg. Cognition'];

        symptoms.forEach((symptom, idx) => {
            ctx.strokeStyle = colors[idx];
            ctx.lineWidth = 2;
            ctx.beginPath();

            progression.forEach((week, i) => {
                const x = (i / (progression.length - 1)) * width;
                const y = height - (week[symptom] * height * 0.8) - height * 0.1;

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
        ctx.fillText('Weeks →', width - 60, height - 10);
        ctx.fillText('Severity', 10, height - 10);
    }

    visualizeNeuralActivity(activity) {
        if (!this.neuralCanvas) return;

        const ctx = this.neuralCanvas.getContext('2d');
        const width = this.neuralCanvas.width = this.neuralCanvas.offsetWidth;
        const height = this.neuralCanvas.height = this.neuralCanvas.offsetHeight;

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
                gradient.addColorStop(0, '#ff3366');
                gradient.addColorStop(1, '#ffaa00');
            } else {
                gradient.addColorStop(0, '#00d4ff');
                gradient.addColorStop(1, '#00ff88');
            }

            ctx.fillStyle = gradient;
            ctx.fillRect(x, y, barWidth, barHeight);

            // Border
            ctx.strokeStyle = '#00d4ff';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, barWidth, barHeight);

            // Label
            ctx.fillStyle = '#b0b0b0';
            ctx.font = '12px Inter';
            ctx.save();
            ctx.translate(x + barWidth / 2, height - 5);
            ctx.rotate(-Math.PI / 4);
            ctx.fillText(region, 0, 0);
            ctx.restore();

            // Value
            ctx.fillStyle = '#ffffff';
            ctx.font = '14px Inter';
            ctx.textAlign = 'center';
            ctx.fillText((value * 100).toFixed(0), x + barWidth / 2, y - 5);
        });
    }
}

window.ClinicalDashboard = ClinicalDashboard;
