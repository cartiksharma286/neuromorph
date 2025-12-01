// Visualization Engine for Pulse Sequences and K-space

const Visualizer = {
    canvases: {},
    contexts: {},

    /**
     * Initialize all canvases
     */
    init() {
        const canvasIds = ['kspace-canvas', 'cine-canvas', 'perfusion-canvas', 'lge-canvas', 'mapping-canvas', 'flow-canvas'];

        canvasIds.forEach(id => {
            const canvas = document.getElementById(id);
            if (canvas) {
                this.canvases[id] = canvas;
                this.contexts[id] = canvas.getContext('2d');
                // Set high DPI
                const dpr = window.devicePixelRatio || 1;
                const rect = canvas.getBoundingClientRect();
                canvas.width = rect.width * dpr;
                canvas.height = rect.height * dpr;
                canvas.style.width = rect.width + 'px';
                canvas.style.height = rect.height + 'px';
                this.contexts[id].scale(dpr, dpr);
            }
        });

        console.log('Visualizer initialized');
    },

    /**
     * Draw K-space undersampling pattern
     */
    drawKSpace(config) {
        const canvas = this.canvases['kspace-canvas'];
        const ctx = this.contexts['kspace-canvas'];
        if (!canvas || !ctx) return;

        const width = canvas.width / (window.devicePixelRatio || 1);
        const height = canvas.height / (window.devicePixelRatio || 1);

        // Clear canvas
        ctx.fillStyle = '#0f1419';
        ctx.fillRect(0, 0, width, height);

        // Draw k-space grid
        const kSpaceSize = 256;
        const cellWidth = (width * 0.7) / kSpaceSize;
        const cellHeight = (height * 0.8) / kSpaceSize;
        const offsetX = width * 0.15;
        const offsetY = height * 0.1;

        // Generate sampling pattern
        const R = config.accelerationFactor;
        const technique = config.technique;

        // Draw sampled lines
        for (let ky = 0; ky < kSpaceSize; ky++) {
            let isSampled = false;

            if (technique === 'sense' || technique === 'grappa') {
                // Regular undersampling
                isSampled = (ky % Math.round(R) === 0);

                // ACS region for GRAPPA
                if (technique === 'grappa') {
                    const centerStart = Math.floor(kSpaceSize / 2 - config.acsLines / 2);
                    const centerEnd = centerStart + config.acsLines;
                    if (ky >= centerStart && ky < centerEnd) {
                        isSampled = true;
                    }
                }
            } else if (technique === 'compressed-sensing') {
                // Variable density random
                const distFromCenter = Math.abs(ky - kSpaceSize / 2) / (kSpaceSize / 2);
                const prob = 1 / R * (1 + Math.exp(-distFromCenter * 3));
                isSampled = Math.random() < prob;
            } else if (technique === 'hybrid') {
                // SENSE + CS
                isSampled = (ky % Math.round(R * 0.7) === 0) || (Math.random() < 0.15 / R);
            }

            if (isSampled) {
                // Draw line across k-space
                const y = offsetY + ky * cellHeight;

                // Gradient from center to edge
                const distFromCenter = Math.abs(ky - kSpaceSize / 2) / (kSpaceSize / 2);
                const brightness = Math.floor(120 + (1 - distFromCenter) * 135);

                ctx.fillStyle = `rgb(${brightness * 0.6}, ${brightness * 0.8}, ${brightness})`;
                ctx.fillRect(offsetX, y, width * 0.7, Math.max(1, cellHeight));
            }
        }

        // Draw k-space center marker
        const centerY = offsetY + (kSpaceSize / 2) * cellHeight;
        ctx.strokeStyle = '#e53e3e';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 3]);
        ctx.beginPath();
        ctx.moveTo(offsetX, centerY);
        ctx.lineTo(offsetX + width * 0.7, centerY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw labels
        ctx.fillStyle = '#a0aec0';
        ctx.font = '14px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('ky (phase encode)', offsetX, offsetY - 10);

        ctx.textAlign = 'center';
        ctx.fillText('kx (readout) →', offsetX + width * 0.35, height - 10);

        // Draw info text
        ctx.fillStyle = '#4299e1';
        ctx.font = 'bold 16px Inter, sans-serif';
        ctx.textAlign = 'right';

        const sampledLines = Math.ceil(kSpaceSize / R);
        ctx.fillText(`R = ${R.toFixed(1)}x`, width - 20, offsetY + 30);
        ctx.fillStyle = '#a0aec0';
        ctx.font = '13px Inter, sans-serif';
        ctx.fillText(`${sampledLines}/${kSpaceSize} lines`, width - 20, offsetY + 50);
        ctx.fillText(technique.toUpperCase(), width - 20, offsetY + 70);
    },

    /**
     * Draw CINE cardiac phase timing diagram
     */
    drawCINE(heartRate, phases, temporalRes) {
        const canvas = this.canvases['cine-canvas'];
        const ctx = this.contexts['cine-canvas'];
        if (!canvas || !ctx) return;

        const width = canvas.width / (window.devicePixelRatio || 1);
        const height = canvas.height / (window.devicePixelRatio || 1);

        // Clear
        ctx.fillStyle = '#0f1419';
        ctx.fillRect(0, 0, width, height);

        const RR = Utils.bpmToRR(heartRate);
        const padding = 60;
        const graphWidth = width - padding * 2;
        const graphHeight = height - padding * 2;

        // Draw ECG-like trace
        ctx.strokeStyle = '#48bb78';
        ctx.lineWidth = 2;
        ctx.beginPath();

        for (let x = 0; x < graphWidth; x++) {
            const t = (x / graphWidth) * RR;
            let y;

            // Simplified ECG waveform
            const normalizedT = t / RR;
            if (normalizedT < 0.1) {
                // QRS complex
                y = Math.sin(normalizedT * 10 * Math.PI * 2) * 50;
            } else if (normalizedT < 0.3) {
                // T wave
                y = Math.sin((normalizedT - 0.1) * 5 * Math.PI) * 20;
            } else {
                // Baseline
                y = 0;
            }

            const screenY = padding + graphHeight / 2 - y;
            if (x === 0) {
                ctx.moveTo(padding + x, screenY);
            } else {
                ctx.lineTo(padding + x, screenY);
            }
        }

        ctx.stroke();

        // Draw cardiac phases
        const phaseWidth = graphWidth / phases;

        for (let i = 0; i < phases; i++) {
            const x = padding + i * phaseWidth;
            const isSystemle = (i / phases) < 0.35; // First 35% is systole

            ctx.fillStyle = isSystemle ? 'rgba(237, 100, 166, 0.15)' : 'rgba(66, 153, 225, 0.15)';
            ctx.fillRect(x, padding, phaseWidth, graphHeight);

            // Phase dividers
            ctx.strokeStyle = '#2d3748';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x, padding);
            ctx.lineTo(x, padding + graphHeight);
            ctx.stroke();
        }

        // Labels
        ctx.fillStyle = '#a0aec0';
        ctx.font = '14px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`RR Interval: ${Math.round(RR)}ms`, width / 2, height - 15);

        ctx.fillStyle = '#ed64a6';
        ctx.fillText('Systole', padding + graphWidth * 0.17, padding - 10);
        ctx.fillStyle = '#4299e1';
        ctx.fillText('Diastole', padding + graphWidth * 0.65, padding - 10);

        // Temporal resolution indicator
        ctx.fillStyle = '#48bb78';
        ctx.font = 'bold 13px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(`Temporal Res: ${temporalRes}ms`, width - padding, padding + 25);
        ctx.fillText(`${phases} phases/cycle`, width - padding, padding + 45);
    },

    /**
     * Draw pulse sequence diagram
     */
    drawPulseSequence(sequenceType, params) {
        // Generic pulse sequence diagram
        // Would have RF, Gx, Gy, Gz, ADC rows
        // This is a simplified version
        console.log(`Drawing ${sequenceType} pulse sequence`, params);
    },

    /**
     * Draw g-factor map
     */
    drawGFactorMap(gFactorData) {
        // 2D g-factor spatial distribution
        // Would show color-coded map with hot spots
        console.log('Drawing g-factor map', gFactorData);
    },

    /**
     * Clear specific canvas
     */
    clear(canvasId) {
        const ctx = this.contexts[canvasId];
        const canvas = this.canvases[canvasId];
        if (ctx && canvas) {
            const width = canvas.width / (window.devicePixelRatio || 1);
            const height = canvas.height / (window.devicePixelRatio || 1);
            ctx.fillStyle = '#0f1419';
            ctx.fillRect(0, 0, width, height);
        }
    }
};

// Export
window.Visualizer = Visualizer;
