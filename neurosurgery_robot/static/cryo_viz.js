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

        this.lut = this.generateIceColormap();
    }

    generateIceColormap() {
        // Range -200 to +50 C (250 steps approx? Let's do 10 steps per degree -> 2500)
        // Offset +200. Index 0 = -200C. Index 2000 = 0C. Index 2500 = 50C.
        const steps = 2500;
        const lut = new Uint8ClampedArray(steps * 4);

        for (let i = 0; i < steps; i++) {
            const temp = (i / 10.0) - 200.0; // Map index to temp

            let r = 0, g = 0, b = 0, a = 0;

            if (temp > 30.0) {
                // Body temp / Warm -> Transparent
                a = 0;
            } else if (temp > 0.0) {
                // 0..30 -> Cooling Mist
                const t = (30.0 - temp) / 30.0; // 0..1
                r = 150; g = 200; b = 255;
                a = Math.floor(100 * t);
            } else {
                // Freezing < 0
                const absT = Math.abs(temp);
                if (absT < 50) {
                    // 0..-50: Blue -> Cyan
                    const t = absT / 50.0;
                    r = 0;
                    g = Math.floor(150 + (105 * t));
                    b = 255;
                    a = Math.floor(100 + (80 * t));
                } else if (absT < 100) {
                    // -50..-100: Cyan -> White (Ice Ball Center)
                    const t = (absT - 50) / 50.0;
                    r = Math.floor(255 * t);
                    g = 255;
                    b = 255;
                    a = 180;
                } else {
                    // Deep Freeze
                    r = 230; g = 240; b = 255; a = 200;
                }
            }
            lut[i * 4] = r;
            lut[i * 4 + 1] = g;
            lut[i * 4 + 2] = b;
            lut[i * 4 + 3] = a;
        }
        return lut;
    }

    getColor(temp) {
        // LUT Lookup
        // Range -200 to +50
        if (temp < -200.0) temp = -200.0;
        if (temp > 49.9) temp = 49.9;

        let idx = Math.floor((temp + 200.0) * 10.0);
        if (idx < 0) idx = 0;

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
        const pixels = this.imageData.data;
        let p = 0;

        for (let y = 0; y < 64; y++) {
            for (let x = 0; x < 64; x++) {
                // Fix: Row-Major [y][x]
                const temp = cryoData[y][x];
                const [cr, cg, cb, ca] = this.getColor(temp);

                pixels[p++] = cr;
                pixels[p++] = cg;
                pixels[p++] = cb;
                pixels[p++] = ca;
            }
        }

        this.bufferCtx.putImageData(this.imageData, 0, 0);

        // 3. Draw Overlay
        this.ctx.save();
        this.ctx.globalCompositeOperation = 'source-over';
        this.ctx.imageSmoothingEnabled = false;
        this.ctx.drawImage(this.bufferCanvas, 0, 0, this.width, this.height);
        this.ctx.restore();
    }
}
