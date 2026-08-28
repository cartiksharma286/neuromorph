/**
 * Ultrasound Transducer & Geodesic Fusion Visualizer
 * Renders:
 * 1. B-Mode RF Reconstruction with Worsley Euler Characteristic topological overlay
 * 2. Geodesic Minimum-Energy Trajectory avoiding Neurovascular Bundles and Urethra
 * 3. Transducer Radiation Beampattern (Polar & Cartesian dB)
 */

class UltrasoundViz {
    constructor(usCanvasId, beampatternCanvasId) {
        this.usCanvas = document.getElementById(usCanvasId);
        this.beamCanvas = document.getElementById(beampatternCanvasId);
        
        this.usCtx = this.usCanvas ? this.usCanvas.getContext('2d') : null;
        this.beamCtx = this.beamCanvas ? this.beamCanvas.getContext('2d') : null;
        
        this.currentData = null;
        this.beampatternData = null;
        this.displayMode = 'fusion'; // 'bmode', 'worsley', 'geodesic', 'fusion'
        
        this.init();
    }
    
    init() {
        if (this.usCanvas) {
            this.usCanvas.width = 256;
            this.usCanvas.height = 256;
        }
        if (this.beamCanvas) {
            this.beamCanvas.width = 280;
            this.beamCanvas.height = 140;
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
            this.usCtx.fillText("Click 'Run Beamformer & Fusion'", this.usCanvas.width / 2, this.usCanvas.height / 2 + 10);
        }
        if (this.beamCtx) {
            this.beamCtx.fillStyle = "#020617";
            this.beamCtx.fillRect(0, 0, this.beamCanvas.width, this.beamCanvas.height);
            this.beamCtx.fillStyle = "#64748b";
            this.beamCtx.font = "10px Courier New";
            this.beamCtx.textAlign = "center";
            this.beamCtx.fillText("Directivity Pattern [dB]", this.beamCanvas.width / 2, this.beamCanvas.height / 2);
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
    }
    
    renderUltrasoundView(data) {
        if (!this.usCtx || !data) return;
        
        const W = this.usCanvas.width;
        const H = this.usCanvas.height;
        const bmode = data.bmode_image; // 128x128
        const worsley = data.worsley_saliency; // 128x128
        const distance = data.distance_map; // 128x128
        const path = data.geodesic_path || [];
        
        if (!bmode || bmode.length === 0) return;
        
        const gridH = bmode.length;
        const gridW = bmode[0].length;
        const imgData = this.usCtx.createImageData(gridW, gridH);
        
        for (let r = 0; r < gridH; r++) {
            for (let c = 0; c < gridW; c++) {
                const idx = (r * gridW + c) * 4;
                const bmVal = bmode[r][c]; // 0.0 to 1.0
                const worVal = worsley ? worsley[r][c] : 0.0;
                const distVal = distance ? distance[r][c] : 0.0;
                
                let red = 0, green = 0, blue = 0;
                
                if (this.displayMode === 'bmode') {
                    // Standard Ultrasound Greyscale (Acoustic dynamic range)
                    const lum = Math.floor(bmVal * 255);
                    red = lum; green = lum; blue = lum;
                } else if (this.displayMode === 'worsley') {
                    // Worsley Topological Excursion Heatmap (Purple to Cyan)
                    red = Math.floor(worVal * 240);
                    green = Math.floor(worVal * 180 + bmVal * 40);
                    blue = Math.floor((1 - worVal) * 120 + worVal * 255);
                } else if (this.displayMode === 'geodesic') {
                    // Geodesic Distance Equipotentials (Iso-distance waves)
                    const contour = Math.sin(distVal * 40.0) > 0.6 ? 255 : 40;
                    red = Math.floor(distVal * 180);
                    green = Math.floor(contour * 0.8 + distVal * 80);
                    blue = Math.floor(255 - distVal * 150);
                } else {
                    // 'fusion' Multi-modal Composite
                    // Base: Ultrasound speckle
                    const usLum = Math.floor(bmVal * 160);
                    red = usLum;
                    green = usLum;
                    blue = usLum;
                    
                    // Add Worsley Excursion Signature in Vivid Gold/Amber
                    if (worVal > 0.35) {
                        red = Math.min(255, red + Math.floor(worVal * 160));
                        green = Math.min(255, green + Math.floor(worVal * 110));
                        blue = Math.max(0, blue - Math.floor(worVal * 70));
                    }
                    
                    // Add Subtle Geodesic wavefront contours in Cyan
                    if (distance && Math.sin(distVal * 32.0) > 0.85) {
                        green = Math.min(255, green + 90);
                        blue = Math.min(255, blue + 140);
                    }
                }
                
                imgData.data[idx] = red;
                imgData.data[idx + 1] = green;
                imgData.data[idx + 2] = blue;
                imgData.data[idx + 3] = 255;
            }
        }
        
        // Draw grid scaled to canvas
        createImageBitmap(imgData).then(bmp => {
            this.usCtx.clearRect(0, 0, W, H);
            this.usCtx.drawImage(bmp, 0, 0, W, H);
            
            // Draw TRUS Transducer Array at top
            this.usCtx.fillStyle = "#06b6d4";
            this.usCtx.fillRect(W * 0.15, 2, W * 0.7, 5);
            this.usCtx.fillStyle = "#ffffff";
            this.usCtx.font = "8px Inter, sans-serif";
            this.usCtx.textAlign = "center";
            this.usCtx.fillText("TRUS ARRAY APERTURE", W / 2, 16);
            
            // Draw Geodesic Path Overlay
            if (path.length > 1) {
                this.usCtx.beginPath();
                this.usCtx.strokeStyle = "#22c55e"; // Bright Green Geodesic
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
                
                // Target Point (Ablation Focal Node)
                const endY = (path[0][0] / gridH) * H;
                const endX = (path[0][1] / gridW) * W;
                this.usCtx.beginPath();
                this.usCtx.arc(endX, endY, 6, 0, 2 * Math.PI);
                this.usCtx.fillStyle = "#ef4444";
                this.usCtx.fill();
                this.usCtx.strokeStyle = "#ffffff";
                this.usCtx.lineWidth = 1.5;
                this.usCtx.stroke();
                
                // Start Point at Transducer
                const startY = (path[path.length - 1][0] / gridH) * H;
                const startX = (path[path.length - 1][1] / gridW) * W;
                this.usCtx.beginPath();
                this.usCtx.arc(startX, startY, 4, 0, 2 * Math.PI);
                this.usCtx.fillStyle = "#06b6d4";
                this.usCtx.fill();
            }
            
            // Annotations (NVB Hazards & Urethra)
            this.usCtx.fillStyle = "rgba(239, 68, 68, 0.4)";
            // Left & Right NVB
            this.usCtx.beginPath();
            this.usCtx.arc(W * 0.32, H * 0.60, 10, 0, 2 * Math.PI);
            this.usCtx.arc(W * 0.68, H * 0.60, 10, 0, 2 * Math.PI);
            this.usCtx.fill();
            
            this.usCtx.fillStyle = "rgba(59, 130, 246, 0.4)";
            // Central Urethra
            this.usCtx.beginPath();
            this.usCtx.arc(W * 0.50, H * 0.45, 9, 0, 2 * Math.PI);
            this.usCtx.fill();
            
            this.usCtx.fillStyle = "#94a3b8";
            this.usCtx.font = "8px sans-serif";
            this.usCtx.textAlign = "left";
            this.usCtx.fillText("NVB-L", W * 0.22, H * 0.62);
            this.usCtx.fillText("NVB-R", W * 0.72, H * 0.62);
            this.usCtx.fillText("URETHRA", W * 0.44, H * 0.43);
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
        
        // Grid lines
        this.beamCtx.strokeStyle = "rgba(255, 255, 255, 0.1)";
        this.beamCtx.lineWidth = 1;
        
        // -3dB, -20dB, -40dB lines
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
        
        // Center angle line
        const midX = 28 + (W - 38) / 2;
        this.beamCtx.beginPath();
        this.beamCtx.moveTo(midX, 10);
        this.beamCtx.lineTo(midX, H - 10);
        this.beamCtx.strokeStyle = "rgba(6, 182, 212, 0.3)";
        this.beamCtx.stroke();
        
        // Plot Beampattern curve
        this.beamCtx.beginPath();
        this.beamCtx.strokeStyle = "#38bdf8";
        this.beamCtx.lineWidth = 1.8;
        
        for (let i = 0; i < angles.length; i++) {
            const ang = angles[i]; // -90 to +90
            const x = 28 + ((ang + 90) / 180.0) * (W - 38);
            const db = Math.max(-60, dbs[i]);
            const y = ((-db) / 60.0) * (H - 24) + 12;
            
            if (i === 0) this.beamCtx.moveTo(x, y);
            else this.beamCtx.lineTo(x, y);
        }
        this.beamCtx.stroke();
        
        // Info label
        this.beamCtx.fillStyle = "#f8fafc";
        this.beamCtx.font = "9px Inter, sans-serif";
        this.beamCtx.textAlign = "left";
        this.beamCtx.fillText(`FWHM: ${beampattern.fwhm_deg}° | PSLL: ${beampattern.psll_db} dB`, 30, 18);
    }
}

window.UltrasoundViz = UltrasoundViz;
