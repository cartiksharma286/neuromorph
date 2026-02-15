/**
 * Waveform Generator Module
 * Real-time stimulation waveform design and safety validation
 */

class WaveformGenerator {
    constructor() {
        this.parameters = {
            amplitude_ma: 3.0,
            frequency_hz: 130,
            pulse_width_us: 90,
            target_region: 'amygdala'
        };
        this.canvas = null;
        this.ctx = null;
    }

    init() {
        this.setupControls();
        this.setupCanvas();
        this.setupSimulateButton();
        this.drawWaveform();
    }

    setupControls() {
        const controls = document.getElementById('parameterControls');
        if (!controls) return;

        controls.innerHTML = `
            <div>
                <label>Target Region</label>
                <select id="targetRegion">
                    <option value="amygdala">Amygdala (BLA)</option>
                    <option value="hippocampus">Hippocampus</option>
                    <option value="vmPFC">vmPFC</option>
                    <option value="hypothalamus">Hypothalamus</option>
                </select>
            </div>
            <div>
                <label>Amplitude: <span id="amplitudeValue">3.0</span> mA</label>
                <input type="range" id="amplitude" min="0.5" max="8.0" step="0.1" value="3.0">
            </div>
            <div>
                <label>Frequency: <span id="frequencyValue">130</span> Hz</label>
                <input type="range" id="frequency" min="20" max="185" step="1" value="130">
            </div>
            <div>
                <label>Pulse Width: <span id="pulseWidthValue">90</span> μs</label>
                <input type="range" id="pulseWidth" min="60" max="210" step="10" value="90">
            </div>
        `;

        // Add event listeners
        ['amplitude', 'frequency', 'pulseWidth', 'targetRegion'].forEach(id => {
            const element = document.getElementById(id);
            element.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                const param = id.replace(/([A-Z])/g, '_$1').toLowerCase();

                if (id === 'targetRegion') {
                    this.parameters.target_region = e.target.value;
                } else {
                    this.parameters[param] = value;
                    document.getElementById(`${id}Value`).textContent = value;
                }

                this.drawWaveform();
                this.validateSafety();
            });
        });
    }

    setupCanvas() {
        this.canvas = document.getElementById('waveformCanvas');
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = this.canvas.offsetHeight;
    }

    drawWaveform() {
        if (!this.ctx) return;

        const { amplitude_ma, frequency_hz, pulse_width_us } = this.parameters;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear canvas with slight transparency for trail effect? No, clean clear.
        this.ctx.clearRect(0, 0, width, height);

        // Background
        // this.ctx.fillStyle = '#0a0a0a';
        // this.ctx.fillRect(0, 0, width, height);

        // Draw grid
        this.drawGrid();

        // Calculate waveform parameters
        const period_ms = 1000 / frequency_hz;
        const pulse_width_ms = pulse_width_us / 1000;
        const pixels_per_ms = width / (period_ms * 3.5); // Show 3.5 periods for continuity look

        const centerY = height / 2;
        const scaleY = (height * 0.7) / (8.0); // static scale based on max amplitude 8.0 for stability

        // Setup Neon Style
        this.ctx.shadowBlur = 10;
        this.ctx.shadowColor = '#00d4ff';
        this.ctx.strokeStyle = '#00d4ff';
        this.ctx.lineWidth = 3;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';

        this.ctx.beginPath();
        this.ctx.moveTo(0, centerY);

        for (let period = 0; period < 5; period++) {
            const offsetX = period * period_ms * pixels_per_ms;
            if (offsetX > width) break;

            // Cathodic phase (down)
            this.ctx.lineTo(offsetX, centerY); // Start
            this.ctx.lineTo(offsetX, centerY + amplitude_ma * scaleY); // Drop (note: Y increases downwards in canvas, but usually cathodic is drawn negative... let's stick to user preference or std. Let's draw Down as negative/cathodic)
            // Actually, convention varies. Let's make "positive" Up.
            // If amp is 3.0, let's draw it going UP for Anodic and DOWN for Cathodic.
            // Wait, previous code drew: lineTo(..., centerY - amp). centerY - amp is UP.

            // Let's implement symmetric biphasic charge balanced.
            // Phase 1 (Cathodic/Negative? Usually Stim is Cathodic First)
            // Let's draw DOWN first for Cathodic.

            this.ctx.lineTo(offsetX, centerY + amplitude_ma * scaleY);
            this.ctx.lineTo(offsetX + pulse_width_ms * pixels_per_ms, centerY + amplitude_ma * scaleY);
            this.ctx.lineTo(offsetX + pulse_width_ms * pixels_per_ms, centerY);

            // Inter-phase gap (tiny)
            const gap = pulse_width_ms * 0.2 * pixels_per_ms;
            this.ctx.lineTo(offsetX + pulse_width_ms * pixels_per_ms + gap, centerY);

            // Phase 2 (Anodic/Positive)
            this.ctx.lineTo(offsetX + pulse_width_ms * pixels_per_ms + gap, centerY - amplitude_ma * scaleY);
            this.ctx.lineTo(offsetX + (pulse_width_ms * 2) * pixels_per_ms + gap, centerY - amplitude_ma * scaleY);
            this.ctx.lineTo(offsetX + (pulse_width_ms * 2) * pixels_per_ms + gap, centerY);

            // Return to baseline until next period
            this.ctx.lineTo(offsetX + period_ms * pixels_per_ms, centerY);
        }

        this.ctx.stroke();

        // Reset Shadow for text/grid
        this.ctx.shadowBlur = 0;

        // Draw Area Fill (Gradient)
        // Re-trace path for fill
        this.ctx.lineTo(width, height);
        this.ctx.lineTo(0, height);
        this.ctx.closePath();

        const gradient = this.ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, 'rgba(0, 212, 255, 0.0)');
        gradient.addColorStop(0.5, 'rgba(0, 212, 255, 0.1)');
        gradient.addColorStop(1, 'rgba(0, 212, 255, 0.0)');
        this.ctx.fillStyle = gradient;
        // this.ctx.fill(); // Fill creates a mess with the baseline return. Keep it simple line for now or complex close path.
        // Let's skip fill to keep it clean neon.

        // Draw labels overlay
        this.ctx.fillStyle = '#fff';
        this.ctx.font = 'bold 14px Inter';
        this.ctx.fillText(`${amplitude_ma.toFixed(1)} mA`, 10, 25);
        this.ctx.font = '12px Inter';
        this.ctx.fillStyle = '#888';
        this.ctx.fillText(`${frequency_hz} Hz  |  ${pulse_width_us} μs`, 10, 45);
    }

    drawGrid() {
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        this.ctx.lineWidth = 1;

        // Horizontal lines (Voltage)
        for (let y = 0; y < this.canvas.height; y += 20) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }

        // Vertical lines (Time)
        for (let x = 0; x < this.canvas.width; x += 40) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }

        // Center line (Baseline)
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([5, 5])
        this.ctx.beginPath();
        this.ctx.moveTo(0, this.canvas.height / 2);
        this.ctx.lineTo(this.canvas.width, this.canvas.height / 2);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
    }

    async validateSafety() {
        try {
            const response = await window.app.post('/safety/validate', this.parameters);
            const validation = response.validation;

            // Update safety status
            const statusDiv = document.getElementById('safetyStatus');
            statusDiv.className = `safety-status ${validation.safety_level}`;

            let statusText = validation.safety_level.toUpperCase();
            if (validation.violations.length > 0) {
                statusText += ': ' + validation.violations[0];
            } else if (validation.warnings.length > 0) {
                statusText += ': ' + validation.warnings[0];
            } else {
                statusText += ': All parameters within safe limits';
            }

            statusDiv.textContent = statusText;

            // Update metrics
            this.displayMetrics(validation.metrics);
        } catch (error) {
            console.error('Safety validation failed:', error);
        }
    }

    displayMetrics(metrics) {
        const container = document.getElementById('safetyMetrics');
        if (!container) return;

        container.innerHTML = `
            <div class="metric-item">
                <div class="metric-item-label">Charge Density</div>
                <div class="metric-item-value">${metrics.charge_density_uc_cm2} μC/cm²</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Current Density</div>
                <div class="metric-item-value">${metrics.current_density_ma_cm2} mA/cm²</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Power</div>
                <div class="metric-item-value">${metrics.power_dissipation_mw} mW</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-label">Voltage</div>
                <div class="metric-item-value">${metrics.voltage_v} V</div>
            </div>
        `;
    }

    setupSimulateButton() {
        const btn = document.getElementById('simulateBtn');
        if (!btn) return;

        btn.addEventListener('click', async () => {
            btn.disabled = true;
            btn.innerHTML = '<div class="loading"></div> Simulating...';

            try {
                const response = await window.app.post('/neural/simulate', this.parameters);

                if (response.success) {
                    // Update clinical dashboard
                    if (window.clinicalDashboard) {
                        window.clinicalDashboard.updateSymptoms(response.symptoms);
                        window.clinicalDashboard.updateBiomarkers(response.biomarkers);
                    }

                    // Show success
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
}

window.WaveformGenerator = WaveformGenerator;
