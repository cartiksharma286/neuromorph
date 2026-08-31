/**
 * Ultrasound Transducer & Geodesic Fusion Visualizer
 * Renders:
 * 1. B-Mode RF Reconstruction with Worsley Euler Characteristic topological overlay
 * 2. Geodesic Minimum-Energy Trajectory avoiding Neurovascular Bundles and Urethra
 * 3. Transducer Radiation Beampattern (Polar & Cartesian dB)
 */

class UltrasoundViz {
    constructor(usCanvasId, beampatternCanvasId, alineCanvasId) {
        this.usCanvas = document.getElementById(usCanvasId);
        this.beamCanvas = document.getElementById(beampatternCanvasId);
        this.alineCanvas = document.getElementById(alineCanvasId);
        
        this.usCtx = this.usCanvas ? this.usCanvas.getContext('2d') : null;
        this.beamCtx = this.beamCanvas ? this.beamCanvas.getContext('2d') : null;
        this.alineCtx = this.alineCanvas ? this.alineCanvas.getContext('2d') : null;
        
        this.currentData = null;
        this.displayMode = 'fusion';
        
        this.init();
    }
    
    init() {
        if (this.usCanvas) {
            this.usCanvas.width = 256;
            this.usCanvas.height = 256;
        }
        if (this.beamCanvas) {
            this.beamCanvas.width = 280;
            this.beamCanvas.height = 130;
        }
        if (this.alineCanvas) {
            this.alineCanvas.width = 280;
            this.alineCanvas.height = 70;
        }
        this.renderPlaceholders();
    }
    
    renderPlaceholders() {
        if (this.usCtx) {
            this.usCtx.fillStyle = "#020617";
            this.usCtx.fillRect(0, 0, this.usCanvas.width, this.usCanvas.height);
            this.usCtx.fillStyle = "#38bdf8";
            this.usCtx.font = "11px Courier New";
            this.usCtx.textAlign = "center";
            this.usCtx.fillText("Ultrasound Beamformer Ready", this.usCanvas.width / 2, this.usCanvas.height / 2 - 10);
            this.usCtx.fillStyle = "#64748b";
            this.usCtx.fillText("Click 'RUN BEAMFORMER SIMULATION'", this.usCanvas.width / 2, this.usCanvas.height / 2 + 10);
        }
        if (this.beamCtx) {
            this.beamCtx.fillStyle = "#020617";
            this.beamCtx.fillRect(0, 0, this.beamCanvas.width, this.beamCanvas.height);
            this.beamCtx.fillStyle = "#64748b";
            this.beamCtx.font = "10px Courier New";
            this.beamCtx.textAlign = "center";
            this.beamCtx.fillText("Directivity Pattern [dB]", this.beamCanvas.width / 2, this.beamCanvas.height / 2);
        }
        if (this.alineCtx) {
            this.alineCtx.fillStyle = "#020617";
            this.alineCtx.fillRect(0, 0, this.alineCanvas.width, this.alineCanvas.height);
            this.alineCtx.fillStyle = "#64748b";
            this.alineCtx.font = "10px Courier New";
            this.alineCtx.textAlign = "center";
            this.alineCtx.fillText("RF A-Line Oscilloscope Trace", this.alineCanvas.width / 2, this.alineCanvas.height / 2);
        }
    }
    
    setDisplayMode(mode) {
        this.displayMode = mode;
        if (this.currentData) {
            this.renderUltrasoundView(this.currentData);
        }
    }
    
    update(data) {
        this.currentData = data;
        this.renderUltrasoundView(data);
        if (data.beampattern) {
            this.renderBeampattern(data.beampattern);
        }
        if (data.rf_aline) {
            this.renderAline(data.rf_aline);
        }
    }
    
    renderUltrasoundView(data) {
        if (!this.usCtx || !data) return;
        
        const W = this.usCanvas.width;
        const H = this.usCanvas.height;
        const recons = data.reconstructions || {};
        const worsley = data.worsley_saliency;
        const distance = data.distance_map;
        const path = data.geodesic_path || [];
        
        // Select active 2D grid matrix based on displayMode
        let activeGrid = data.bmode_image;
        if (this.displayMode === 'das' && recons.das) activeGrid = recons.das;
        else if (this.displayMode === 'mvdr' && recons.mvdr) activeGrid = recons.mvdr;
        else if (this.displayMode === 'plane_wave' && recons.plane_wave) activeGrid = recons.plane_wave;
        else if (this.displayMode === 'harmonic' && recons.harmonic) activeGrid = recons.harmonic;
        else if (this.displayMode === 'raw_rf' && recons.raw_rf) activeGrid = recons.raw_rf;
        else if (this.displayMode === 'worsley' && worsley) activeGrid = worsley;
        else if (this.displayMode === 'geodesic' && distance) activeGrid = distance;
        
        if (!activeGrid || activeGrid.length === 0) return;
        
        const gridH = activeGrid.length;
        const gridW = activeGrid[0].length;
        const imgData = this.usCtx.createImageData(gridW, gridH);
        
        for (let r = 0; r < gridH; r++) {
            for (let c = 0; c < gridW; c++) {
                const idx = (r * gridW + c) * 4;
                const val = activeGrid[r][c];
                const bmVal = data.bmode_image ? data.bmode_image[r][c] : val;
                const worVal = worsley ? worsley[r][c] : 0.0;
                const distVal = distance ? distance[r][c] : 0.0;
                
                let red = 0, green = 0, blue = 0;
                
                if (this.displayMode === 'das' || this.displayMode === 'mvdr' || this.displayMode === 'plane_wave') {
                    const lum = Math.floor(val * 255);
                    red = lum; green = lum; blue = lum;
                } else if (this.displayMode === 'harmonic') {
                    const lum = Math.floor(val * 255);
                    red = Math.min(255, Math.floor(lum * 1.1));
                    green = Math.min(255, Math.floor(lum * 0.95));
                    blue = Math.floor(lum * 0.7);
                } else if (this.displayMode === 'raw_rf') {
                    const centered = (val - 0.5) * 2.0;
                    if (centered > 0) {
                        red = Math.floor(255 * centered);
                        green = Math.floor(255 * (1 - centered * 0.5));
                        blue = Math.floor(255 * (1 - centered));
                    } else {
                        const neg = -centered;
                        red = Math.floor(255 * (1 - neg));
                        green = Math.floor(255 * (1 - neg * 0.5));
                        blue = Math.floor(255 * neg);
                    }
                } else if (this.displayMode === 'worsley') {
                    red = Math.floor(worVal * 255);
                    green = Math.floor(worVal * 190);
                    blue = Math.floor((1 - worVal) * 160);
                } else if (this.displayMode === 'geodesic') {
                    const wave = Math.sin(distVal * 36.0) > 0.65 ? 255 : 40;
                    red = Math.floor(distVal * 140);
                    green = Math.floor(wave * 0.8 + distVal * 60);
                    blue = Math.floor(255 - distVal * 180);
                } else {
                    const usLum = Math.floor(bmVal * 165);
                    red = usLum; green = usLum; blue = usLum;
                    
                    if (worVal > 0.35) {
                        red = Math.min(255, red + Math.floor(worVal * 170));
                        green = Math.min(255, green + Math.floor(worVal * 120));
                        blue = Math.max(0, blue - Math.floor(worVal * 60));
                    }
                    
                    if (distance && Math.sin(distVal * 32.0) > 0.85) {
                        green = Math.min(255, green + 80);
                        blue = Math.min(255, blue + 140);
                    }
                }
                
                imgData.data[idx] = red;
                imgData.data[idx + 1] = green;
                imgData.data[idx + 2] = blue;
                imgData.data[idx + 3] = 255;
            }
        }
        
        createImageBitmap(imgData).then(bmp => {
            this.usCtx.clearRect(0, 0, W, H);
            this.usCtx.drawImage(bmp, 0, 0, W, H);
            
            this.usCtx.fillStyle = "#06b6d4";
            this.usCtx.fillRect(W * 0.15, 2, W * 0.7, 5);
            this.usCtx.fillStyle = "#ffffff";
            this.usCtx.font = "8px Inter, sans-serif";
            this.usCtx.textAlign = "center";
            this.usCtx.fillText("TRUS ARRAY APERTURE", W / 2, 16);
            
            if (path.length > 1 && (this.displayMode === 'fusion' || this.displayMode === 'geodesic')) {
                this.usCtx.beginPath();
                this.usCtx.strokeStyle = "#22c55e";
                this.usCtx.lineWidth = 2.5;
                this.usCtx.setLineDash([4, 2]);
                
                for (let i = 0; i < path.length; i++) {
                    const py = (path[i][0] / gridH) * H;
                    const px = (path[i][1] / gridW) * W;
                    if (i === 0) this.usCtx.moveTo(px, py);
                    else this.usCtx.lineTo(px, py);
                }
                this.usCtx.stroke();
                this.usCtx.setLineDash([]);
                
                const endY = (path[0][0] / gridH) * H;
                const endX = (path[0][1] / gridW) * W;
                this.usCtx.beginPath();
                this.usCtx.arc(endX, endY, 6, 0, 2 * Math.PI);
                this.usCtx.fillStyle = "#ef4444";
                this.usCtx.fill();
                this.usCtx.strokeStyle = "#ffffff";
                this.usCtx.lineWidth = 1.5;
                this.usCtx.stroke();
                
                const startY = (path[path.length - 1][0] / gridH) * H;
                const startX = (path[path.length - 1][1] / gridW) * W;
                this.usCtx.beginPath();
                this.usCtx.arc(startX, startY, 4, 0, 2 * Math.PI);
                this.usCtx.fillStyle = "#06b6d4";
                this.usCtx.fill();
            }
            
            if (this.displayMode === 'fusion' || this.displayMode === 'geodesic') {
                this.usCtx.fillStyle = "rgba(239, 68, 68, 0.4)";
                this.usCtx.beginPath();
                this.usCtx.arc(W * 0.32, H * 0.60, 10, 0, 2 * Math.PI);
                this.usCtx.arc(W * 0.68, H * 0.60, 10, 0, 2 * Math.PI);
                this.usCtx.fill();
                
                this.usCtx.fillStyle = "rgba(59, 130, 246, 0.4)";
                this.usCtx.beginPath();
                this.usCtx.arc(W * 0.50, H * 0.45, 9, 0, 2 * Math.PI);
                this.usCtx.fill();
                
                this.usCtx.fillStyle = "#94a3b8";
                this.usCtx.font = "8px sans-serif";
                this.usCtx.textAlign = "left";
                this.usCtx.fillText("NVB-L", W * 0.22, H * 0.62);
                this.usCtx.fillText("NVB-R", W * 0.72, H * 0.62);
                this.usCtx.fillText("URETHRA", W * 0.44, H * 0.43);
            }
            
            this.usCtx.fillStyle = "rgba(15, 23, 42, 0.75)";
            this.usCtx.fillRect(4, H - 20, 140, 16);
            this.usCtx.fillStyle = "#38bdf8";
            this.usCtx.font = "9px Inter, sans-serif";
            this.usCtx.textAlign = "left";
            this.usCtx.fillText("MODE: " + this.displayMode.toUpperCase(), 8, H - 8);
        });
    }
    
    renderBeampattern(beampattern) {
        if (!this.beamCtx || !beampattern) return;
        const W = this.beamCanvas.width;
        const H = this.beamCanvas.height;
        const angles = beampattern.angles_deg;
        const dbs = beampattern.beampattern_db;
        
        this.beamCtx.fillStyle = "#020617";
        this.beamCtx.fillRect(0, 0, W, H);
        
        this.beamCtx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        this.beamCtx.lineWidth = 1;
        
        const dbLevels = [0, -3, -20, -40, -60];
        dbLevels.forEach(db => {
            const y = ((-db) / 60.0) * (H - 24) + 12;
            this.beamCtx.beginPath();
            this.beamCtx.moveTo(28, y);
            this.beamCtx.lineTo(W - 10, y);
            this.beamCtx.stroke();
            
            this.beamCtx.fillStyle = "#64748b";
            this.beamCtx.font = "8px Courier New";
            this.beamCtx.textAlign = "right";
            this.beamCtx.fillText(`${db}dB`, 25, y + 3);
        });
        
        const midX = 28 + (W - 38) / 2;
        this.beamCtx.beginPath();
        this.beamCtx.moveTo(midX, 10);
        this.beamCtx.lineTo(midX, H - 10);
        this.beamCtx.strokeStyle = "rgba(6, 182, 212, 0.3)";
        this.beamCtx.stroke();
        
        this.beamCtx.beginPath();
        this.beamCtx.strokeStyle = "#38bdf8";
        this.beamCtx.lineWidth = 1.8;
        
        for (let i = 0; i < angles.length; i++) {
            const ang = angles[i];
            const x = 28 + ((ang + 90) / 180.0) * (W - 38);
            const db = Math.max(-60, dbs[i]);
            const y = ((-db) / 60.0) * (H - 24) + 12;
            
            if (i === 0) this.beamCtx.moveTo(x, y);
            else this.beamCtx.lineTo(x, y);
        }
        this.beamCtx.stroke();
        
        this.beamCtx.fillStyle = "#f8fafc";
        this.beamCtx.font = "9px Inter, sans-serif";
        this.beamCtx.textAlign = "left";
        this.beamCtx.fillText(`FWHM: ${beampattern.fwhm_deg}° | PSLL: ${beampattern.psll_db} dB`, 30, 18);
    }
    
    renderAline(aline) {
        if (!this.alineCtx || !aline) return;
        const W = this.alineCanvas.width;
        const H = this.alineCanvas.height;
        
        this.alineCtx.fillStyle = "#020617";
        this.alineCtx.fillRect(0, 0, W, H);
        
        this.alineCtx.strokeStyle = "rgba(255, 255, 255, 0.1)";
        this.alineCtx.lineWidth = 1;
        this.alineCtx.beginPath();
        this.alineCtx.moveTo(10, H - 10);
        this.alineCtx.lineTo(W - 10, H - 10);
        this.alineCtx.stroke();
        
        this.alineCtx.beginPath();
        this.alineCtx.strokeStyle = "#4ade80";
        this.alineCtx.lineWidth = 1.5;
        
        const N = aline.length;
        for (let i = 0; i < N; i++) {
            const x = 10 + (i / (N - 1)) * (W - 20);
            const y = (H - 12) - aline[i] * (H - 20);
            if (i === 0) this.alineCtx.moveTo(x, y);
            else this.alineCtx.lineTo(x, y);
        }
        this.alineCtx.stroke();
        
        this.alineCtx.fillStyle = "#94a3b8";
        this.alineCtx.font = "8px Courier New";
        this.alineCtx.textAlign = "left";
        this.alineCtx.fillText("A-LINE ENVELOPE (FOCAL DEPTH PROFILE)", 12, 14);
    }
}

window.UltrasoundViz = UltrasoundViz;
