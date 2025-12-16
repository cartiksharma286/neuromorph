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
        // High-Precision Cryo Gradient
        // Range -200 to +50 C (2500 steps, 0.1C res)
        // Offset +200. Index 0 = -200C.
        const steps = 2500;
        const lut = new Uint8ClampedArray(steps * 4);

        // Control Points: Temp -> [R, G, B, A]
        const stops = [
            { t: 37.0, c: [0, 0, 0, 0] },
            { t: 30.0, c: [150, 200, 255, 30] },   // Cool Mist
            { t: 10.0, c: [100, 180, 255, 80] },   // Cooling
            { t: 0.0, c: [0, 150, 255, 120] },     // Freezing Point (Blue)
            { t: -20.0, c: [0, 255, 255, 160] },   // Ice Formation (Cyan)
            { t: -40.0, c: [150, 255, 255, 200] }, // Deep Freeze
            { t: -100.0, c: [240, 250, 255, 240] },// Cryo Core (White-ish)
            { t: -180.0, c: [255, 255, 255, 255] } // Max Cryo
        ];

        // Sort descending because we iterate typical range or just handle logic
        // Stops are descending in standard "cooling" view, but let's just use standard interpolation
        // Helper to find range
        const getRange = (temp) => {
            // Find s1, s2 such that s2.t <= temp <= s1.t 
            // OR s1.t <= temp <= s2.t

            // Stops are mixed order? No, let's look at them: 37, 30, 10... descending.
            // Let's sort them ascending to make interpolation logic standard
            const sortedStops = stops.sort((a, b) => a.t - b.t);

            for (let i = 0; i < sortedStops.length - 1; i++) {
                if (temp >= sortedStops[i].t && temp < sortedStops[i + 1].t) {
                    return [sortedStops[i], sortedStops[i + 1]];
                }
            }
            if (temp < sortedStops[0].t) return [sortedStops[0], sortedStops[0]];
            return [sortedStops[sortedStops.length - 1], sortedStops[sortedStops.length - 1]];
        };

        for (let i = 0; i < steps; i++) {
            const temp = (i / 10.0) - 200.0;

            // Clamping
            if (temp > 37.0) {
                lut[i * 4] = 0; lut[i * 4 + 1] = 0; lut[i * 4 + 2] = 0; lut[i * 4 + 3] = 0;
                continue;
            }
            if (temp < -180.0) {
                lut[i * 4] = 255; lut[i * 4 + 1] = 255; lut[i * 4 + 2] = 255; lut[i * 4 + 3] = 255;
                continue;
            }

            const [s1, s2] = getRange(temp);

            // Interpolate
            let ratio = 0;
            if (Math.abs(s2.t - s1.t) > 0.001) {
                ratio = (temp - s1.t) / (s2.t - s1.t);
            }

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
