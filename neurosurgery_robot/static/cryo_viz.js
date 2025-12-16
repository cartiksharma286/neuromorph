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
    }

    getColor(temp) {
        // Temp range: -180 to 37

        if (temp > 30) {
            return [0, 0, 0, 0];
        }

        // Ice visualization
        // Cap alpha at 200 (approx 80%) to allow seeing tumor underneath

        if (temp > 0) {
            // 0..30C -> Faint cooling mist
            const t = (30 - temp) / 30.0;
            return [150, 200, 255, 100 * t];
        }

        const absT = Math.abs(temp);

        if (absT < 50) {
            // Blue -> Cyan
            const t = absT / 50.0;
            return [0, 150 + 105 * t, 255, 100 + 80 * t];
        } else if (absT < 100) {
            // Cyan -> White
            const t = (absT - 50) / 50.0;
            return [255 * t, 255, 255, 180];
        } else {
            // Deep Freeze (White)
            return [230, 240, 255, 200];
        }
    }

    update(cryoData, anatomyData) {
        if (!cryoData || !this.ctx) return;

        // 1. Draw Background
        if (this.bgImageLoaded) {
            // High-Res Image
            this.ctx.drawImage(this.bgImage, 0, 0, this.width, this.height);
        } else if (anatomyData) {
            // Fallback: Low-Res Generated Anatomy
            // Render anatomy to buffer first? No, just draw rectangles or similar.
            // Let's use the pixel loop for fallback compositing if needed.
            // Actually, simplest is to just fill black and rely on alpha blending if we want to be lazy,
            // But let's try to verify if we can render the low-res anatomy.

            // We can just use the ctx to draw it directly scaled
            const fw = this.width / 64;
            const fh = this.height / 64;
            for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) {
                    const val = Math.floor(anatomyData[x][y] * 255);
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
                const temp = cryoData[x][y];
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
