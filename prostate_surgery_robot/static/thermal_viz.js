
class ThermalViz {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;

        // Colormaps
        this.tempLUT = this.createThermoLUT();
    }

    createThermoLUT() {
        // Range: -100.0 to +100.0 C
        // Step: 0.1 C
        // Size: 2001 entries
        // Index = (Temp + 100) * 10
        const lut = new Uint8ClampedArray(2001 * 4);

        for (let i = 0; i <= 2000; i++) {
            let t = (i - 1000) / 10.0;
            let r = 0, g = 0, b = 0, a = 0;

            if (t < -40) {
                // Lethal Freeze (Necrosis) - Bright White/Cyan
                r = 200; g = 255; b = 255; a = 200;
            } else if (t < 0) {
                // Ice Ball - Deep Blue/Cyan
                // Gradient: -40 -> 0
                r = 0;
                g = Math.floor(100 + ((t + 40) / 40) * 155);
                b = 255;
                a = 180;
            } else if (t < 35) {
                // Cool Tissue - Faint Blue
                r = 0; g = 0; b = 255; a = 60;
            } else if (t < 40) {
                // Body Temp (37C) - Transparent
                a = 0;
            } else if (t < 60) {
                // Hyperthermia (Red)
                // 40 -> 60
                r = 255;
                g = Math.floor(255 - ((t - 40) / 20) * 255);
                b = 0;
                a = 150;
            } else {
                // Ablation/Char (Black/Red)
                r = 100; g = 0; b = 0; a = 200;
            }

            let idx = i * 4;
            lut[idx] = r; lut[idx + 1] = g; lut[idx + 2] = b; lut[idx + 3] = a;
        }
        return lut;
    }

    update(tempMap, anatomy, gridPos) {
        if (!tempMap || !anatomy) return;

        // Draw Anatomy Background (Grayscale)
        const frame = this.ctx.createImageData(128, 128); // 128x128 native
        const d = frame.data;

        // ZOOM IMPLEMENTATION: 2x Digital Zoom on Prostate (Center)
        // Canvas is 128x128. Source is 128x128.
        // We want source [32..96] mapped to [0..128].

        for (let cy = 0; cy < 128; cy++) {
            for (let cx = 0; cx < 128; cx++) {
                // Map Canvas (cy, cx) -> Source (sy, sx)
                // Linear interp
                const sy = Math.floor(32 + (cy / 128.0) * 64);
                const sx = Math.floor(32 + (cx / 128.0) * 64);

                // --- FETCH PIXEL ---
                let val = 0;
                let tVal = 0;

                if (Array.isArray(anatomy[0])) {
                    val = anatomy[sy][sx];
                    tVal = tempMap[sy][sx];
                } else {
                    let idx = (sy * 128 + sx);
                    val = anatomy[idx];
                    tVal = tempMap[idx];
                }

                let gray = Math.floor(val * 255);
                let tr = 0, tg = 0, tb = 0, ta = 0;

                // New Index Calculation
                let lutIdx = Math.floor((tVal + 100) * 10);
                lutIdx = Math.max(0, Math.min(2000, lutIdx));

                // Only Look up if deviation from body temp (simplifies rendering)
                if (tVal < 36 || tVal > 38) {
                    const lutBase = lutIdx * 4;
                    tr = this.tempLUT[lutBase];
                    tg = this.tempLUT[lutBase + 1];
                    tb = this.tempLUT[lutBase + 2];
                    ta = this.tempLUT[lutBase + 3];
                }

                let aNorm = ta / 255.0;
                let rOut = (tr * aNorm) + (gray * (1.0 - aNorm));
                let gOut = (tg * aNorm) + (gray * (1.0 - aNorm));
                let bOut = (tb * aNorm) + (gray * (1.0 - aNorm));

                // Draw to Canvas (cy, cx)
                let pIdx = (cy * 128 + cx) * 4;
                d[pIdx] = rOut;
                d[pIdx + 1] = gOut;
                d[pIdx + 2] = bOut;
                d[pIdx + 3] = 255;
            }
        }

        this.ctx.putImageData(frame, 0, 0);

        // Draw Reticle
        if (gridPos) {
            const x = gridPos[0];
            const y = gridPos[1];

            this.ctx.strokeStyle = "#fbbf24"; // Amber
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.moveTo(x - 5, y); this.ctx.lineTo(x + 5, y);
            this.ctx.moveTo(x, y - 5); this.ctx.lineTo(x, y + 5);
            this.ctx.stroke();
        }

    }
}
