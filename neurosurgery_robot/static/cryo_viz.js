class CryoViz {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        // Ensure canvas exists to avoid crash
        if (!this.canvas) {
            console.error("CryoViz: Canvas not found " + canvasId);
            return;
        }
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;

        this.bufferCanvas = document.createElement('canvas');
        this.bufferCanvas.width = 64;
        this.bufferCanvas.height = 64;
        this.bufferCtx = this.bufferCanvas.getContext('2d');
        this.imageData = this.bufferCtx.createImageData(64, 64);

        // Load MR Image
        this.bgImage = new Image();
        this.bgImage.src = '/static/mr_cortex_tumor.png';
        this.bgImageLoaded = false;
        this.bgImage.onload = () => {
            this.bgImageLoaded = true;
        }

        this.lut = this.generateIceFractionColormap();
    }

    generateIceFractionColormap() {
        // Maps ice volume fraction (0 – 1) to clinical cryo-ablation colours.
        // 256 linear steps.
        // 0.00 = no ice → transparent
        // 0.15 = early cooling → pale blue mist
        // 0.35 = partial freeze → medium blue
        // 0.55 = 50 % frozen   → bright cyan
        // 0.75 = deep freeze   → cyan-white
        // 1.00 = fully frozen  → pure white (lethal cryo)
        const steps = 256;
        const lut = new Uint8ClampedArray(steps * 4);

        const stops = [
            { f: 0.00, c: [  0,   0,   0,   0] },
            { f: 0.04, c: [180, 220, 255,  20] },
            { f: 0.15, c: [ 80, 160, 255,  70] },
            { f: 0.30, c: [  0, 110, 255, 130] },
            { f: 0.50, c: [  0, 200, 245, 180] },
            { f: 0.70, c: [100, 235, 255, 215] },
            { f: 0.85, c: [200, 248, 255, 240] },
            { f: 1.00, c: [255, 255, 255, 255] },
        ];

        for (let i = 0; i < steps; i++) {
            const frac = i / (steps - 1);

            // Find interpolation segment
            let s1 = stops[0], s2 = stops[1];
            for (let j = 0; j < stops.length - 1; j++) {
                if (frac >= stops[j].f && frac <= stops[j + 1].f) {
                    s1 = stops[j]; s2 = stops[j + 1];
                    break;
                }
            }

            let ratio = 0;
            const df = s2.f - s1.f;
            if (Math.abs(df) > 0.0001) ratio = (frac - s1.f) / df;

            lut[i * 4]     = Math.round(s1.c[0] + (s2.c[0] - s1.c[0]) * ratio);
            lut[i * 4 + 1] = Math.round(s1.c[1] + (s2.c[1] - s1.c[1]) * ratio);
            lut[i * 4 + 2] = Math.round(s1.c[2] + (s2.c[2] - s1.c[2]) * ratio);
            lut[i * 4 + 3] = Math.round(s1.c[3] + (s2.c[3] - s1.c[3]) * ratio);
        }
        return lut;
    }

    getColor(frac) {
        // Input: ice volume fraction 0-1
        if (frac < 0) frac = 0;
        if (frac > 1) frac = 1;
        const idx = Math.min(255, Math.floor(frac * 255));
        const i = idx * 4;
        return [this.lut[i], this.lut[i + 1], this.lut[i + 2], this.lut[i + 3]];
    }

    update(cryoData, anatomyData) {
        if (!cryoData || !this.ctx) return;

        // 1. Draw Background
        if (this.bgImageLoaded) {
            // High-Res Image
            this.ctx.drawImage(this.bgImage, 0, 0, this.width, this.height);
        } else if (anatomyData) {
            // Fallback: Low-Res Generated Anatomy
            const fw = this.width / 64;
            const fh = this.height / 64;
            for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) {
                    const val = Math.floor(anatomyData[y][x] * 255); // usage [y][x]
                    this.ctx.fillStyle = `rgb(${val},${val},${val})`;
                    this.ctx.fillRect(x * fw, y * fh, fw, fh);
                }
            }
        } else {
            this.ctx.fillStyle = "#000";
            this.ctx.fillRect(0, 0, this.width, this.height);
        }

        // 2. Prepare Overlay
        // Backend data is 128x128; buffer is 64x64 — stride-2 downsample
        const stride = cryoData.length > 64 ? 2 : 1;
        const pixels = this.imageData.data;
        let p = 0;

        for (let y = 0; y < 64; y++) {
            for (let x = 0; x < 64; x++) {
                // Ice volume fraction 0-1
                const frac = cryoData[y * stride][x * stride];
                const [cr, cg, cb, ca] = this.getColor(frac);

                pixels[p++] = cr;
                pixels[p++] = cg;
                pixels[p++] = cb;
                pixels[p++] = ca;
            }
        }

        this.bufferCtx.putImageData(this.imageData, 0, 0);

        // 3. Draw Overlay with bilinear smoothing for clean ice-ball edges
        this.ctx.save();
        this.ctx.globalCompositeOperation = 'source-over';
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'high';
        this.ctx.drawImage(this.bufferCanvas, 0, 0, this.width, this.height);
        this.ctx.restore();

        // 4. Cryo probe crosshair at ice-ball centre
        if (this.bgImageLoaded) {
            const cx = this.width * 0.5;
            const cy = this.height * 0.5;
            this.ctx.save();
            this.ctx.strokeStyle = 'rgba(0, 220, 255, 0.75)';
            this.ctx.lineWidth = 1;
            this.ctx.setLineDash([3, 5]);
            this.ctx.beginPath();
            this.ctx.moveTo(cx - 20, cy); this.ctx.lineTo(cx + 20, cy);
            this.ctx.moveTo(cx, cy - 20); this.ctx.lineTo(cx, cy + 20);
            this.ctx.stroke();
            this.ctx.restore();
        }
    }
}
