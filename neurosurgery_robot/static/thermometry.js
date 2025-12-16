class ThermalViz {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        this.mode = 'TEMP'; // TEMP or DAMAGE
        this.onHover = null;
        this.lastData = null;

        // Offscreen buffer for the 64x64 data
        this.bufferCanvas = document.createElement('canvas');
        this.bufferCanvas.width = 64;
        this.bufferCanvas.height = 64;
        this.bufferCtx = this.bufferCanvas.getContext('2d');
        this.imageData = this.bufferCtx.createImageData(64, 64);

        // Load MR Background for "Color on Grayscale"
        this.bgImage = new Image();
        this.bgImage.src = '/static/mr_cortex_tumor.png';
        this.bgLoaded = false;
        this.bgImage.onload = () => { this.bgLoaded = true; };
        // Initialize Colormaps
        this.tempLUT = this.generateTempColormap();
        this.damageLUT = this.generateDamageColormap();

        // Interaction
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseleave', () => {
            if (this.onHover) this.onHover(null);
        });
    }

    generateTempColormap() {
        // High-Precision Gradient Generator
        const steps = 1000;
        const lut = new Uint8ClampedArray(steps * 4);

        // Control Points: Temp -> [R, G, B, A]
        // Alpha bumped for better visibility
        const stops = [
            { t: 37.0, c: [0, 0, 0, 0] },
            { t: 37.5, c: [48, 18, 59, 80] },    // 37.5: Dark Violet (More visible)
            { t: 40.0, c: [70, 131, 240, 120] }, // 40.0: Blue
            { t: 43.0, c: [26, 211, 211, 160] }, // 43.0: Cyan
            { t: 46.0, c: [108, 247, 114, 190] },// 46.0: Green
            { t: 52.0, c: [254, 186, 44, 210] }, // 52.0: Orange
            { t: 60.0, c: [246, 55, 18, 230] },  // 60.0: Red
            { t: 80.0, c: [98, 6, 6, 245] },     // 80.0: Dark Red
            { t: 100.0, c: [255, 255, 255, 255] }// 100.0: White
        ];

        // Helper to find range
        const getRange = (temp) => {
            for (let i = 0; i < stops.length - 1; i++) {
                if (temp >= stops[i].t && temp < stops[i + 1].t) {
                    return [stops[i], stops[i + 1]];
                }
            }
            return [stops[stops.length - 2], stops[stops.length - 1]];
        };

        for (let i = 0; i < steps; i++) {
            // Map step 0..999 to 37.0..100.0
            const temp = 37.0 + (i / steps) * 63.0;

            // Fixed clamping to 100.0
            if (temp >= 100.0) {
                const c = stops[stops.length - 1].c;
                lut[i * 4] = c[0]; lut[i * 4 + 1] = c[1]; lut[i * 4 + 2] = c[2]; lut[i * 4 + 3] = c[3];
                continue;
            }

            const [s1, s2] = getRange(temp);

            // Interpolate
            const ratio = (temp - s1.t) / (s2.t - s1.t);
            const r = Math.floor(s1.c[0] + (s2.c[0] - s1.c[0]) * ratio);
            const g = Math.floor(s1.c[1] + (s2.c[1] - s1.c[1]) * ratio);
            const b = Math.floor(s1.c[2] + (s2.c[2] - s1.c[2]) * ratio);
            const a = Math.floor(s1.c[3] + (s2.c[3] - s1.c[3]) * ratio);

            lut[i * 4] = r;
            lut[i * 4 + 1] = g;
            lut[i * 4 + 2] = b;
            lut[i * 4 + 3] = a;
        }

        return lut;
    }

    generateDamageColormap() {
        const steps = 256;
        const lut = new Uint8ClampedArray(steps * 4);
        for (let i = 0; i < steps; i++) {
            let val = i; // 0..255
            let r = 0, g = 0, b = 0, a = 0;

            if (val < 1) {
                a = 0;
            } else {
                // Gradient: Orange (Damage) -> Dark Red -> Black (Necrosis)
                const t = val / 240.0; // Normalized 0..1 for critical range
                const tClamped = Math.min(1.0, t);

                // Orange: 255, 165, 0
                // Black: 0, 0, 0

                r = Math.floor(255 * (1 - tClamped));
                g = Math.floor(165 * (1 - tClamped));
                b = 0;
                a = 120 + Math.floor(135 * tClamped); // 120 -> 255 opacity
            }
            lut[i * 4] = r;
            lut[i * 4 + 1] = g;
            lut[i * 4 + 2] = b;
            lut[i * 4 + 3] = a;
        }
        return lut;
    }

    setHoverCallback(fn) {
        this.onHover = fn;
    }

    setMode(mode) {
        this.mode = mode;
    }

    handleMouseMove(e) {
        if (!this.lastData || !this.onHover) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const gx = Math.floor((x / rect.width) * 64);
        const gy = Math.floor((y / rect.height) * 64);

        if (gx >= 0 && gx < 64 && gy >= 0 && gy < 64) {
            if (this.lastData[gy] && this.lastData[gy][gx] !== undefined) {
                this.onHover(this.lastData[gy][gx]);
            }
        }
    }

    getTempColorUnsafe(temp) {
        if (temp < 37.0) temp = 37.0;
        if (temp > 99.9) temp = 99.9;

        let idx = Math.floor((temp - 37.0) / 63.0 * 1000);
        if (idx < 0) idx = 0;
        if (idx >= 1000) idx = 999;

        const i = idx * 4;
        return [this.tempLUT[i], this.tempLUT[i + 1], this.tempLUT[i + 2], this.tempLUT[i + 3]];
    }

    getDamageColorUnsafe(val) {
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        let idx = Math.floor(val);
        const i = idx * 4;
        return [this.damageLUT[i], this.damageLUT[i + 1], this.damageLUT[i + 2], this.damageLUT[i + 3]];
    }

    getColor(temp) { return this.getTempColorUnsafe(temp); }
    getDamageColor(val) { return this.getDamageColorUnsafe(val); }

    update(tempData, damageData, anatomyData) {
        if (!tempData) return;

        let activeData = (this.mode === 'DAMAGE' && damageData) ? damageData : tempData;
        this.lastData = activeData;

        // 1. Draw Background
        if (this.bgLoaded) {
            this.ctx.drawImage(this.bgImage, 0, 0, this.width, this.height);
        } else {
            this.ctx.fillStyle = '#111';
            this.ctx.fillRect(0, 0, this.width, this.height);
        }

        // 2. Prepare Overlay
        const pixels = this.imageData.data;
        let p = 0;
        let maxVal = 0;

        let hasHeat = false;

        for (let y = 0; y < 64; y++) {
            for (let x = 0; x < 64; x++) {
                const val = activeData[y][x];
                if (val > maxVal) maxVal = val;

                // Color Logic
                let r = 0, g = 0, b = 0, a = 0;

                // 1. Heatmap / Damage Map
                if (this.mode === 'DAMAGE') {
                    if (val >= 1.0) hasHeat = true;
                    const i = Math.min(255, Math.max(0, Math.floor(val))) * 4;
                    r = this.damageLUT[i]; g = this.damageLUT[i + 1]; b = this.damageLUT[i + 2]; a = this.damageLUT[i + 3];
                } else {
                    if (val >= 37.05) hasHeat = true;
                    let idx = Math.floor((val - 37.0) / 63.0 * 1000);
                    if (idx < 0) idx = 0;
                    if (idx >= 1000) idx = 999;
                    const i = idx * 4;
                    r = this.tempLUT[i]; g = this.tempLUT[i + 1]; b = this.tempLUT[i + 2]; a = this.tempLUT[i + 3];
                }

                pixels[p++] = r;
                pixels[p++] = g;
                pixels[p++] = b;
                pixels[p++] = a;
            }
        }

        // 3. Draw Overlay if necessary
        if (hasHeat) {
            this.bufferCtx.putImageData(this.imageData, 0, 0);
            this.ctx.save();
            this.ctx.globalAlpha = 0.85;
            this.ctx.imageSmoothingEnabled = false;
            this.ctx.drawImage(this.bufferCanvas, 0, 0, this.width, this.height);
            this.ctx.restore();
        }

        return maxVal;
    }

    initChart(history, targetProfile = []) {
        if (this.chart) return;

        const ctx = document.getElementById('temp-profile-chart').getContext('2d');
        const maxLen = Math.max(history.length, targetProfile.length, 100);

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array(maxLen).fill(''),
                datasets: [{
                    label: 'Actual Temp (°C)',
                    data: history,
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    label: 'GenAI Target (°C)',
                    data: targetProfile,
                    borderColor: '#06b6d4', // Cyan
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: true, labels: { color: '#cbd5e1' } } },
                scales: {
                    x: { display: false },
                    y: {
                        beginAtZero: false,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#94a3b8', font: { size: 10 } },
                        suggestedMin: 37,
                        suggestedMax: 80
                    }
                }
            }
        });
    }

    updateChart(history, targetProfile = []) {
        if (!this.chart) {
            this.initChart(history, targetProfile);
        } else {
            const maxLen = Math.max(history.length, targetProfile.length, 100);
            if (this.chart.data.labels.length !== maxLen) {
                this.chart.data.labels = Array(maxLen).fill('');
            }

            this.chart.data.datasets[0].data = history;
            this.chart.data.datasets[1].data = targetProfile;
            this.chart.update();
        }
    }
}
