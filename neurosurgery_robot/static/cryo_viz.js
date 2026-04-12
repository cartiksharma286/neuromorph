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
        this.nvqStatus = { connected: false, latency: 0, coherence: 0 };
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

    update(packet, anatomyData) {
        if (!packet || !this.ctx) return;
        
        // Handle NVQLink Packet structure
        const cryoRGB = packet.data; // Already RGB arrays
        this.nvqStatus = {
            connected: true,
            latency: packet.latency,
            coherence: packet.coherence,
            id: packet.nvq_id
        };

        // 1. Draw Background (Anatomy)
        if (this.bgImageLoaded) {
            this.ctx.globalAlpha = 1.0;
            this.ctx.drawImage(this.bgImage, 0, 0, this.width, this.height);
        } else if (anatomyData) {
            const fw = this.width / 64;
            const fh = this.height / 64;
            for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) {
                    const val = Math.floor(anatomyData[y][x] * 180); // Slightly darker background for contrast
                    this.ctx.fillStyle = `rgb(${val},${val},${val})`;
                    this.ctx.fillRect(x * fw, y * fh, fw + 0.5, fh + 0.5);
                }
            }
        } else {
            this.ctx.fillStyle = "#000";
            this.ctx.fillRect(0, 0, this.width, this.height);
        }

        // 2. Draw Colorized Cryo (from RGB Map)
        // Data is 64x64 for performance
        const rows = cryoRGB.length;
        const cols = cryoRGB[0].length;
        const cW = this.width / cols;
        const cH = this.height / rows;

        this.ctx.globalAlpha = 0.85; // Give it an icy transparency over the anatomy
        for (let y = 0; y < rows; y++) {
            for (let x = 0; x < cols; x++) {
                const [r, g, b] = cryoRGB[y][x];
                // Only draw if not the background color (approx)
                if (r > 0.05 || g > 0.05 || b > 0.1) {
                    this.ctx.fillStyle = `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`;
                    this.ctx.fillRect(x * cW, y * cH, cW + 0.5, cH + 0.5); // overlapping slightly to avoid seams
                }
            }
        }
        this.ctx.globalAlpha = 1.0;

        // 3. Draw NVQLink Overlay
        this.drawNVQOverlay();

        // 4. Cryo probe crosshair
        this.drawProbeCrosshair();
    }

    drawNVQOverlay() {
        this.ctx.save();
        this.ctx.font = "bold 9px 'Inter', sans-serif";
        this.ctx.fillStyle = "rgba(6, 182, 212, 0.9)";
        this.ctx.fillText(`NVQLINK: ${this.nvqStatus.id || 'ACTIVE'}`, 10, 20);
        this.ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
        this.ctx.fillText(`LATENCY: ${this.nvqStatus.latency.toFixed(2)}ms`, 10, 32);
        this.ctx.fillText(`COHERENCE: ${(this.nvqStatus.coherence * 100).toFixed(2)}%`, 10, 44);
        
        // Status indicator LED
        this.ctx.fillStyle = "#10b981";
        this.ctx.beginPath();
        this.ctx.arc(140, 17, 3, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.restore();
    }

    drawProbeCrosshair() {
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
