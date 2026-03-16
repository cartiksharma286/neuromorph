class ThermalViz {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        this.mode = 'TEMP'; // TEMP or DAMAGE
        this.onHover = null;
        this.lastData = null;
        this._endEffectorNX = null;
        this._endEffectorNZ = null;
        this._activeLUTName = 'prf';

        // Level-set segmentation overlay state
        this._ls = {
            boundary_ds:   [],
            tumor_mask_ds: [],
            phi_ds:        [],
            safe_zone_ds:  [],
            tumor_center:  [0.5, 0.5],
            active:        true,   // toggleable
            showSafeZone:  true,
            showPhiLines:  true,
        };

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
        this.luts = {
            prf:       this._buildPRFLUT(),
            hotbody:   this._buildHotBodyLUT(),
            rainbow:   this._buildRainbowLUT(),
            coolwarm:  this._buildCoolWarmLUT(),
        };
        this.tempLUT   = this.luts['prf'];
        this.damageLUT = this._buildDamageLUT();

        // Interaction
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseleave', () => {
            if (this.onHover) this.onHover(null);
        });

        // Animation state
        this.frame = 0;
    }

    /** Public: switch the active LUT by name ('prf'|'hotbody'|'rainbow'|'coolwarm') */
    setLUT(name) {
        if (this.luts[name]) {
            this.tempLUT = this.luts[name];
            this._activeLUTName = name;
        }
    }

    /** Store normalised end-effector position for overlay */
    setEndEffectorOverlay(nx, nz) {
        this._endEffectorNX = nx;
        this._endEffectorNZ = nz;
    }

    /** Update level-set segmentation overlay from API response.
     *  @param {object} d - response from /api/segmentation/level_set
     */
    setLevelSetContour(d) {
        this._ls.boundary_ds   = d.boundary_ds   || [];
        this._ls.tumor_mask_ds = d.tumor_mask_ds  || [];
        this._ls.phi_ds        = d.phi_ds         || [];
        this._ls.safe_zone_ds  = d.safe_zone_ds   || [];
        this._ls.tumor_center  = d.tumor_center   || [0.5, 0.5];
    }

    /** Show/hide the level-set overlay */
    toggleLevelSet(show) {
        this._ls.active = (show !== undefined) ? show : !this._ls.active;
    }

    /** Draw level-set segmentation layers onto the main canvas */
    _drawLevelSetOverlay() {
        const bd  = this._ls.boundary_ds;
        const tm  = this._ls.tumor_mask_ds;
        const phi = this._ls.phi_ds;
        const sz  = this._ls.safe_zone_ds;
        const tc  = this._ls.tumor_center;
        const nR  = bd.length;
        const nC  = bd[0].length;
        const cW  = this.width  / nC;
        const cH  = this.height / nR;

        this.ctx.save();

        // (a) Tumour interior — semi-transparent red fill
        this.ctx.fillStyle = 'rgba(239,68,68,0.18)';
        for (let r = 0; r < nR; r++) {
            for (let c = 0; c < nC; c++) {
                if (tm[r] && tm[r][c]) {
                    this.ctx.fillRect(c * cW, r * cH, cW, cH);
                }
            }
        }

        // (b) Tumour boundary — solid red outline
        this.ctx.strokeStyle = '#ef4444';
        this.ctx.lineWidth   = 1.5;
        this.ctx.setLineDash([]);
        for (let r = 0; r < nR; r++) {
            for (let c = 0; c < nC; c++) {
                if (bd[r] && bd[r][c]) {
                    this.ctx.strokeRect(c * cW + 0.5, r * cH + 0.5, cW - 1, cH - 1);
                }
            }
        }

        // (c) Safe-zone margin — dashed yellow at safe_zone boundary
        if (this._ls.showSafeZone && sz.length > 0) {
            this.ctx.strokeStyle = 'rgba(250,204,21,0.75)';
            this.ctx.lineWidth   = 1;
            this.ctx.setLineDash([3, 3]);
            for (let r = 1; r < nR - 1; r++) {
                for (let c = 1; c < nC - 1; c++) {
                    const inSafe  = sz[r] && sz[r][c];
                    const inTumor = tm[r] && tm[r][c];
                    if (inSafe && !inTumor) {
                        const edge =
                            !(sz[r-1] && sz[r-1][c]) ||
                            !(sz[r+1] && sz[r+1][c]) ||
                            !(sz[r]   && sz[r][c-1]) ||
                            !(sz[r]   && sz[r][c+1]);
                        if (edge) this.ctx.strokeRect(c * cW, r * cH, cW, cH);
                    }
                }
            }
            this.ctx.setLineDash([]);
        }

        // (d) Phi zero-crossing (active contour front) — cyan glow
        if (this._ls.showPhiLines && phi.length > 0) {
            this.ctx.strokeStyle = 'rgba(34,211,238,0.70)';
            this.ctx.lineWidth   = 1;
            this.ctx.setLineDash([]);
            for (let r = 1; r < nR - 1; r++) {
                for (let c = 1; c < nC - 1; c++) {
                    const v = phi[r][c];
                    if (!isFinite(v)) continue;
                    const crossH = (phi[r][c+1] !== undefined) && (Math.sign(v) !== Math.sign(phi[r][c+1]));
                    const crossV = (phi[r+1]   !== undefined) && (Math.sign(v) !== Math.sign(phi[r+1][c]));
                    if (crossH || crossV) {
                        this.ctx.strokeRect(c * cW, r * cH, cW, cH);
                    }
                }
            }
        }

        // (e) Tumour centroid crosshair — magenta
        const tcx = tc[0] * this.width;
        const tcy = tc[1] * this.height;
        this.ctx.strokeStyle = '#f0abfc';
        this.ctx.lineWidth   = 1.5;
        this.ctx.setLineDash([2, 3]);
        this.ctx.beginPath();
        this.ctx.moveTo(tcx - 10, tcy); this.ctx.lineTo(tcx + 10, tcy);
        this.ctx.moveTo(tcx, tcy - 10); this.ctx.lineTo(tcx, tcy + 10);
        this.ctx.stroke();
        this.ctx.setLineDash([]);

        // (f) LS label near centroid
        this.ctx.fillStyle  = '#f0abfc';
        this.ctx.font       = 'bold 9px monospace';
        this.ctx.fillText('LS', tcx + 11, tcy - 4);

        this.ctx.restore();
    }

    // ── Clinical PRF-shift thermometry LUT ────────────────────────────
    _buildPRFLUT() {
        const steps = 1000;
        const lut = new Uint8ClampedArray(steps * 4);
        const stops = [
            { t: 37.0,  c: [  0,   0,   0,   0] },
            { t: 38.5,  c: [  0,   0, 210,  18] },
            { t: 41.0,  c: [  0,  60, 255,  60] },
            { t: 45.0,  c: [  0, 190, 255, 120] },
            { t: 50.0,  c: [  0, 255, 190, 160] },
            { t: 55.0,  c: [140, 255,   0, 190] },
            { t: 60.0,  c: [255, 230,   0, 215] },
            { t: 68.0,  c: [255, 110,   0, 235] },
            { t: 78.0,  c: [255,  10,   0, 248] },
            { t: 90.0,  c: [255, 255, 255, 255] },
            { t: 100.0, c: [255, 255, 255, 255] }
        ];
        return this._interpolateLUT(lut, stops, steps);
    }

    // ── Hot-body (black→red→yellow→white) LUT ─────────────────────────
    _buildHotBodyLUT() {
        const steps = 1000;
        const lut = new Uint8ClampedArray(steps * 4);
        const stops = [
            { t: 37.0,  c: [  0,  0,  0,   0] },
            { t: 39.0,  c: [ 20,  0,  0,  30] },
            { t: 45.0,  c: [180,  0,  0, 160] },
            { t: 55.0,  c: [255, 80,  0, 210] },
            { t: 65.0,  c: [255,220,  0, 235] },
            { t: 78.0,  c: [255,255,180, 248] },
            { t: 100.0, c: [255,255,255, 255] }
        ];
        return this._interpolateLUT(lut, stops, steps);
    }

    // ── Rainbow (cool→warm) LUT ────────────────────────────────────────
    _buildRainbowLUT() {
        const steps = 1000;
        const lut = new Uint8ClampedArray(steps * 4);
        const stops = [
            { t: 37.0,  c: [  0,   0,   0,   0] },
            { t: 39.0,  c: [  0,   0, 255,  30] },
            { t: 45.0,  c: [  0, 200, 255, 120] },
            { t: 52.0,  c: [  0, 255,   0, 180] },
            { t: 60.0,  c: [255, 255,   0, 215] },
            { t: 70.0,  c: [255, 100,   0, 240] },
            { t: 80.0,  c: [255,   0,   0, 250] },
            { t: 100.0, c: [255,   0, 255, 255] }
        ];
        return this._interpolateLUT(lut, stops, steps);
    }

    // ── Cool-warm diverging LUT ────────────────────────────────────────
    _buildCoolWarmLUT() {
        const steps = 1000;
        const lut = new Uint8ClampedArray(steps * 4);
        const stops = [
            { t: 37.0,  c: [ 59, 76,192,  0] },
            { t: 42.0,  c: [142,185,232, 80] },
            { t: 50.0,  c: [220,220,220,160] },
            { t: 58.0,  c: [244,109, 67, 220] },
            { t: 70.0,  c: [215, 48, 39, 245] },
            { t: 100.0, c: [165,  0, 38, 255] }
        ];
        return this._interpolateLUT(lut, stops, steps);
    }

    _interpolateLUT(lut, stops, steps) {
        const getRange = (temp) => {
            for (let i = 0; i < stops.length - 1; i++) {
                if (temp >= stops[i].t && temp < stops[i + 1].t) return [stops[i], stops[i + 1]];
            }
            return [stops[stops.length - 2], stops[stops.length - 1]];
        };
        for (let i = 0; i < steps; i++) {
            const temp = 37.0 + (i / steps) * 63.0;
            const [s1, s2] = getRange(temp);
            const r = (temp - s1.t) / (s2.t - s1.t);
            lut[i * 4]     = Math.round(s1.c[0] + (s2.c[0] - s1.c[0]) * r);
            lut[i * 4 + 1] = Math.round(s1.c[1] + (s2.c[1] - s1.c[1]) * r);
            lut[i * 4 + 2] = Math.round(s1.c[2] + (s2.c[2] - s1.c[2]) * r);
            lut[i * 4 + 3] = Math.round(s1.c[3] + (s2.c[3] - s1.c[3]) * r);
        }
        return lut;
    }

    _buildDamageLUT() {
        const steps = 256;
        const lut = new Uint8ClampedArray(steps * 4);
        for (let i = 0; i < steps; i++) {
            const t = i / 240.0;
            if (i < 1) {
                lut[i * 4 + 3] = 0;
            } else if (i >= 240) {
                lut[i * 4] = 0; lut[i * 4 + 1] = 0; lut[i * 4 + 2] = 0; lut[i * 4 + 3] = 230;
            } else {
                lut[i * 4]     = 255;
                lut[i * 4 + 1] = Math.round(100 * (1 - t));
                lut[i * 4 + 2] = Math.round(255 * t);
                lut[i * 4 + 3] = Math.round(100 + 100 * t);
            }
        }
        return lut;
    }

    // Legacy aliases so existing call-sites still work
    generateTempColormap()   { return this._buildPRFLUT(); }
    generateDamageColormap() { return this._buildDamageLUT(); }

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
            // Account for stride-2 downsampling if data is 128x128
            const stride = this.lastData.length > 64 ? 2 : 1;
            const srcY = Math.min(gy * stride, this.lastData.length - 1);
            const srcX = Math.min(gx * stride, (this.lastData[0] || []).length - 1);
            if (this.lastData[srcY] && this.lastData[srcY][srcX] !== undefined) {
                this.onHover(this.lastData[srcY][srcX]);
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
        // Data from backend may be 128x128 — stride-2 downsample to fill 64x64 buffer
        const stride = activeData.length > 64 ? 2 : 1;
        const pixels = this.imageData.data;
        let p = 0;
        let maxVal = 0;
        let hasHeat = false;

        for (let y = 0; y < 64; y++) {
            for (let x = 0; x < 64; x++) {
                const val = activeData[y * stride][x * stride];
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

        // 3b. Level-Set Tumour Contour Overlay
        if (this._ls.active && this._ls.boundary_ds.length > 0) {
            this._drawLevelSetOverlay();
        }

        // 4. Enhanced Laser Visuals
        if (laserActive && laserPos) {
            // laserPos is [x, z] normalised 0-1 from backend
            const lpx = Array.isArray(laserPos) ? laserPos[0] : (laserPos.x || 0);
            const lpy = Array.isArray(laserPos) ? laserPos[1] : (laserPos.y || 0);
            const lx = lpx * this.width;
            const ly = lpy * this.height;

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

        // 5. End-Effector Overlay — cyan diamond (always shown when position available)
        if (this._endEffectorNX !== null && this._endEffectorNZ !== null) {
            const ex = this._endEffectorNX * this.width;
            const ez = this._endEffectorNZ * this.height;
            this.ctx.save();
            // Outer glow ring
            const eglow = this.ctx.createRadialGradient(ex, ez, 3, ex, ez, 12);
            eglow.addColorStop(0, 'rgba(34,211,238,0.7)');
            eglow.addColorStop(1, 'rgba(34,211,238,0.0)');
            this.ctx.beginPath();
            this.ctx.arc(ex, ez, 12, 0, Math.PI * 2);
            this.ctx.fillStyle = eglow;
            this.ctx.fill();
            // Diamond shape
            this.ctx.fillStyle = 'rgba(34,211,238,0.95)';
            this.ctx.beginPath();
            this.ctx.moveTo(ex,      ez - 6);
            this.ctx.lineTo(ex + 5,  ez);
            this.ctx.lineTo(ex,      ez + 6);
            this.ctx.lineTo(ex - 5,  ez);
            this.ctx.closePath();
            this.ctx.fill();
            // Label
            this.ctx.fillStyle = '#22d3ee';
            this.ctx.font = 'bold 9px monospace';
            this.ctx.fillText('EE', ex + 8, ez + 4);
            this.ctx.restore();
        }

        // 6. Temperature Colorbar (right edge, 12px wide)
        if (this.mode === 'TEMP') {
            const barX = this.width - 14;
            const barH = this.height - 20;
            const barY = 10;
            for (let py = 0; py < barH; py++) {
                const frac = 1 - py / barH;
                let idx = Math.floor(frac * 999);
                idx = Math.max(0, Math.min(999, idx));
                const i = idx * 4;
                this.ctx.fillStyle =
                    `rgba(${this.tempLUT[i]},${this.tempLUT[i+1]},${this.tempLUT[i+2]},1.0)`;
                this.ctx.fillRect(barX, barY + py, 12, 1);
            }
            // Bar labels
            this.ctx.font = '8px monospace';
            this.ctx.fillStyle = '#e2e8f0';
            this.ctx.fillText('100°', barX - 18, barY + 6);
            this.ctx.fillText('68°',  barX - 14, barY + barH * 0.45);
            this.ctx.fillText('37°',  barX - 14, barY + barH + 4);
            // Bar border
            this.ctx.strokeStyle = 'rgba(255,255,255,0.3)';
            this.ctx.lineWidth = 0.5;
            this.ctx.strokeRect(barX, barY, 12, barH);
        }

        return maxVal;
    }

    initChart(history, targetProfile = []) {
        if (this.chart) return;

        const ctx = document.getElementById('temp-profile-chart').getContext('2d');
        const safeHistory = Array.isArray(history) ? history : [];
        const safeTargetProfile = Array.isArray(targetProfile) ? targetProfile : [];
        const maxLen = Math.max(safeHistory.length, safeTargetProfile.length, 100);

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array(maxLen).fill(''),
                datasets: [{
                    label: 'Actual Temp (°C)',
                    data: safeHistory,
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    label: 'GenAI Target (°C)',
                    data: safeTargetProfile,
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
        const safeHistory = Array.isArray(history) ? history : [];
        const safeTargetProfile = Array.isArray(targetProfile) ? targetProfile : [];
        if (!this.chart) {
            this.initChart(safeHistory, safeTargetProfile);
        } else {
            const maxLen = Math.max(safeHistory.length, safeTargetProfile.length, 100);
            if (this.chart.data.labels.length !== maxLen) {
                this.chart.data.labels = Array(maxLen).fill('');
            }

            this.chart.data.datasets[0].data = safeHistory;
            this.chart.data.datasets[1].data = safeTargetProfile;
            this.chart.update();
        }
    }
}
