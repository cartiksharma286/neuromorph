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

        // Animation state
        this.frame = 0;
    }

    generateTempColormap() {
        // High-Vibrancy "Turbo" inspired LUT for Grayscale MRI backgrounds
        const steps = 1000;
        const lut = new Uint8ClampedArray(steps * 4);

        // Professional Surgical Color Palette
        // 37-38: Transparent
        // 38-42: Violets/Blues (Near Target)
        // 42-50: Aquas/Greens (Coagulation range)
        // 50-65: Yellows/Oranges (Ablation start)
        // 65-80: Deep Reds (High intensity)
        // 80-100: White-Hot Core
        const stops = [
            { t: 37.0, c: [0, 0, 0, 0] },
            { t: 38.0, c: [60, 0, 150, 40] },
            { t: 40.0, c: [0, 80, 255, 100] },
            { t: 43.0, c: [0, 255, 200, 140] },
            { t: 48.0, c: [0, 255, 0, 180] },
            { t: 55.0, c: [255, 255, 0, 210] },
            { t: 65.0, c: [255, 120, 0, 230] },
            { t: 80.0, c: [255, 0, 0, 250] },
            { t: 95.0, c: [255, 255, 255, 255] },
            { t: 100.0, c: [255, 255, 255, 255] }
        ];

        const getRange = (temp) => {
            for (let i = 0; i < stops.length - 1; i++) {
                if (temp >= stops[i].t && temp < stops[i + 1].t) {
                    return [stops[i], stops[i + 1]];
                }
            }
            return [stops[stops.length - 2], stops[stops.length - 1]];
        };

        for (let i = 0; i < steps; i++) {
            const temp = 37.0 + (i / steps) * 63.0;
            const [s1, s2] = getRange(temp);
            const ratio = (temp - s1.t) / (s2.t - s1.t);
            lut[i * 4] = s1.c[0] + (s2.c[0] - s1.c[0]) * ratio;
            lut[i * 4 + 1] = s1.c[1] + (s2.c[1] - s1.c[1]) * ratio;
            lut[i * 4 + 2] = s1.c[2] + (s2.c[2] - s1.c[2]) * ratio;
            lut[i * 4 + 3] = s1.c[3] + (s2.c[3] - s1.c[3]) * ratio;
        }

        return lut;
    }

    generateDamageColormap() {
        // High-Contrast Necrosis Map
        const steps = 256;
        const lut = new Uint8ClampedArray(steps * 4);
        for (let i = 0; i < steps; i++) {
            const t = i / 240.0;
            if (i < 1) {
                lut[i * 4 + 3] = 0;
            } else if (i >= 240) {
                // Dead tissue: Black with high alpha
                lut[i * 4] = 0; lut[i * 4 + 1] = 0; lut[i * 4 + 2] = 0; lut[i * 4 + 3] = 230;
            } else {
                // Healing/Damage gradient: Orange/Magenta
                lut[i * 4] = 255;
                lut[i * 4 + 1] = 100 * (1 - t);
                lut[i * 4 + 2] = 255 * (t);
                lut[i * 4 + 3] = 100 + 100 * t;
            }
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

    update(tempData, damageData, anatomyData, laserActive = false, laserPos = null) {
        if (!tempData) return;
        this.frame++;

        let activeData = (this.mode === 'DAMAGE' && damageData) ? damageData : tempData;
        this.lastData = activeData;

        // 1. Draw Background (MRI Image)
        if (this.bgLoaded) {
            this.ctx.drawImage(this.bgImage, 0, 0, this.width, this.height);
        } else {
            this.ctx.fillStyle = '#0a0a0a';
            this.ctx.fillRect(0, 0, this.width, this.height);
        }

        // 2. Prepare Thermal Overlay
        const pixels = this.imageData.data;
        let p = 0;
        let maxVal = 0;
        let hasHeat = false;

        for (let y = 0; y < 64; y++) {
            for (let x = 0; x < 64; x++) {
                const val = activeData[y][x];
                if (val > maxVal) maxVal = val;

                if (this.mode === 'DAMAGE') {
                    if (val >= 1.0) hasHeat = true;
                    const i = Math.min(255, Math.floor(val)) * 4;
                    pixels[p++] = this.damageLUT[i];
                    pixels[p++] = this.damageLUT[i + 1];
                    pixels[p++] = this.damageLUT[i + 2];
                    pixels[p++] = this.damageLUT[i + 3];
                } else {
                    if (val >= 37.1) hasHeat = true;
                    let idx = Math.floor((val - 37.0) / 63.0 * 1000);
                    idx = Math.max(0, Math.min(999, idx));
                    const i = idx * 4;
                    pixels[p++] = this.tempLUT[i];
                    pixels[p++] = this.tempLUT[i + 1];
                    pixels[p++] = this.tempLUT[i + 2];
                    pixels[p++] = this.tempLUT[i + 3];
                }
            }
        }

        // 3. Draw Overlay with Smoothing
        if (hasHeat) {
            this.bufferCtx.putImageData(this.imageData, 0, 0);
            this.ctx.save();
            this.ctx.imageSmoothingEnabled = true;
            this.ctx.globalAlpha = laserActive ? 0.9 : 0.75;
            this.ctx.drawImage(this.bufferCanvas, 0, 0, this.width, this.height);
            this.ctx.restore();
        }

        // 4. Enhanced Laser Visuals
        if (laserActive && laserPos) {
            // Map grid pos to canvas pos
            const lx = (laserPos.x / 64.0) * this.width;
            const ly = (laserPos.y / 64.0) * this.height;

            // Pulsing effect frequency
            const pulse = 1.0 + 0.15 * Math.sin(this.frame * 0.4);

            // Thermal Core Glow
            this.ctx.save();
            this.ctx.beginPath();
            const grad = this.ctx.createRadialGradient(lx, ly, 2, lx, ly, 18 * pulse);
            grad.addColorStop(0, 'rgba(255, 255, 255, 1.0)'); // White hot center
            grad.addColorStop(0.2, 'rgba(255, 255, 0, 0.9)'); // Yellow core
            grad.addColorStop(0.5, 'rgba(255, 100, 0, 0.6)'); // Orange glow
            grad.addColorStop(1.0, 'rgba(255, 0, 0, 0.0)');   // Fade

            this.ctx.fillStyle = grad;
            this.ctx.arc(lx, ly, 18 * pulse, 0, Math.PI * 2);
            this.ctx.fill();

            // Precision Crosshair
            this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
            this.ctx.setLineDash([2, 4]);
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.moveTo(lx - 25, ly); this.ctx.lineTo(lx + 25, ly);
            this.ctx.moveTo(lx, ly - 25); this.ctx.lineTo(lx, ly + 25);
            this.ctx.stroke();
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
