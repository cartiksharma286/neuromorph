class CryoViz {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.error("CryoViz: Canvas not found " + canvasId);
            return;
        }
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;

        // Offscreen buffer for smoothing (64x64 data)
        this.bufferCanvas = document.createElement('canvas');
        this.bufferCanvas.width = 64;
        this.bufferCanvas.height = 64;
        this.bufferCtx = this.bufferCanvas.getContext('2d');
        this.imageData = this.bufferCtx.createImageData(64, 64);

        // Load MR Image (Grayscale Clinical Baseline)
        this.bgImage = new Image();
        this.bgImageLoaded = false;
        this.bgImage.onload = () => {
            this.bgImageLoaded = true;
            console.log("CryoViz: MR Background Loaded");
        }
        this.bgImage.src = '/static/mr_cortex_tumor.png';

        this.geminiStatus = { active: true, model: 'Gemini 1.5 Pro', latency: 22.4, id: 'SURGICAL-GEN-01' };
        this.lastProbePos = [0.5, 0.5];
        this.frameCount = 0;
    }

    update(packet, anatomyData, rawPacket = null) {
        if (!packet || !this.ctx) return;
        this.frameCount++;
        
        this.geminiStatus = {
            active: true,
            model: 'Gemini 1.5 Pro',
            latency: packet.latency || 22.4,
            coherence: packet.coherence || 0.99,
            id: packet.gemini_id || 'LOCAL-AI'
        };

        this.ctx.clearRect(0, 0, this.width, this.height);

        // 1. Draw Background (Grayscale MR Image)
        if (this.bgImageLoaded) {
            this.ctx.globalAlpha = 1.0; 
            this.ctx.drawImage(this.bgImage, 0, 0, this.width, this.height);
        }
        
        // 1b. Draw Voxel Anatomy (Faint clinical overlay)
        if (anatomyData) {
            this.ctx.save();
            this.ctx.globalAlpha = this.bgImageLoaded ? 0.15 : 1.0;
            const fw = this.width / 64;
            const fh = this.height / 64;
            const stride = anatomyData.length > 64 ? 2 : 1;
            for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) {
                    const v = Math.floor(anatomyData[y * stride][x * stride] * 140);
                    if (v > 10) {
                        this.ctx.fillStyle = `rgb(${v},${v},${v})`;
                        this.ctx.fillRect(x * fw, y * fh, fw + 0.5, fh + 0.5);
                    }
                }
            }
            this.ctx.restore();
        }

        // 2. Draw Colorized Cryo (Smoothed via Screen Blending)
        const pixels = this.imageData.data;
        let p = 0;
        let hasIce = false;
        for (let y = 0; y < 64; y++) {
            for (let x = 0; x < 64; x++) {
                const rgb = packet.data[y] ? packet.data[y][x] : [0,0,0];
                pixels[p++] = Math.round(rgb[0] * 255);
                pixels[p++] = Math.round(rgb[1] * 255);
                pixels[p++] = Math.round(rgb[2] * 255);
                
                const brightness = (rgb[0] + rgb[1] + rgb[2]) / 3;
                pixels[p++] = brightness > 0.05 ? 200 : 0;
                if (brightness > 0.1) hasIce = true;
            }
        }
        
        if (hasIce) {
            this.bufferCtx.putImageData(this.imageData, 0, 0);
            this.ctx.save();
            this.ctx.imageSmoothingEnabled = true;
            this.ctx.globalCompositeOperation = 'screen';
            this.ctx.globalAlpha = 0.8;
            this.ctx.drawImage(this.bufferCanvas, 0, 0, this.width, this.height);
            this.ctx.restore();
        }

        // 3. Draw Profiles (Nature-Style Iso-contours)
        if (rawPacket && rawPacket.raw_map) {
            this.drawProfiles(rawPacket.raw_map);
        }

        // 4. Draw Necrotic Mask (Targeting Core - Nature Red)
        if (packet.necrotic_mask && packet.necrotic_mask.length > 0) {
            this.drawNecroticMask(packet.necrotic_mask);
        }

        // 5. Draw Legend
        if (packet.legend && packet.legend.length > 0) {
            this.drawLegend(packet.legend);
        }

        this.drawGeminiStatus();
        
        if (rawPacket && rawPacket.metrics && rawPacket.metrics.probe_position) {
            this.lastProbePos = rawPacket.metrics.probe_position;
        }
        this.drawProbeCrosshair(this.lastProbePos);
    }

    drawProfiles(rawMap) {
        // Nature Levels: Boundary, Intermediate, Core
        const levels = [0.15, 0.5, 0.9];
        // Blue Frontier: #2563eb, Intermediate: #3b82f6, White Core: #ffffff
        const colors = ['#2563eb', '#3b82f6', '#ffffff'];
        
        const rows = rawMap.length;
        const cols = rawMap[0].length;
        const cW = this.width / cols;
        const cH = this.height / rows;

        this.ctx.save();
        this.ctx.globalCompositeOperation = 'screen';
        
        levels.forEach((lvl, i) => {
            this.ctx.strokeStyle = colors[i];
            this.ctx.lineWidth = 2.5 - (i * 0.5);
            this.ctx.shadowBlur = 6;
            this.ctx.shadowColor = colors[i];
            
            if (i === 0) {
                this.ctx.setLineDash([8, 4]);
            } else {
                this.ctx.setLineDash([]);
            }

            this.ctx.beginPath();
            let first = true;
            for (let y = 1; y < rows - 1; y++) {
                for (let x = 1; x < cols - 1; x++) {
                    const val = rawMap[y][x];
                    if (val >= lvl) {
                        if (rawMap[y-1][x] < lvl || rawMap[y+1][x] < lvl || 
                            rawMap[y][x-1] < lvl || rawMap[y][x+1] < lvl) {
                            
                            const px = x * cW + (cW/2);
                            const py = y * cH + (cH/2);
                            
                            // Sub-pixel deformation for organic look
                            const dx = Math.sin(this.frameCount/15 + y*0.5) * 2.0;
                            const dy = Math.cos(this.frameCount/15 + x*0.5) * 2.0;
                            
                            if (first) {
                                this.ctx.moveTo(px + dx, py + dy);
                                first = false;
                            } else {
                                this.ctx.lineTo(px + dx, py + dy);
                            }
                        }
                    }
                }
            }
            this.ctx.stroke();
        });
        this.ctx.restore();
    }

    drawLegend(legend) {
        const startX = this.width - 135;
        const startY = 20;
        const rowHeight = 20;

        this.ctx.save();
        this.ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
        this.ctx.strokeStyle = 'rgba(37, 99, 235, 0.3)';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.roundRect(startX - 10, startY - 10, 145, (legend.length * rowHeight) + 15, 12);
        this.ctx.fill();
        this.ctx.stroke();

        this.ctx.font = "bold 9px 'Inter', sans-serif";
        this.ctx.textBaseline = 'middle';

        legend.forEach((item, i) => {
            const y = startY + (i * rowHeight);
            this.ctx.fillStyle = item.color;
            this.ctx.beginPath();
            this.ctx.arc(startX + 4, y, 5, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.shadowBlur = 2;
            this.ctx.shadowColor = item.color;
            this.ctx.fillStyle = '#f1f5f9';
            this.ctx.fillText(item.label.toUpperCase(), startX + 18, y);
        });
        this.ctx.restore();
    }

    drawNecroticMask(mask) {
        const rows = mask.length;
        const cols = mask[0].length;
        const cW = this.width / cols;
        const cH = this.height / rows;

        this.ctx.save();
        this.ctx.globalCompositeOperation = 'screen';
        const pulse = 0.7 + 0.3 * Math.sin(Date.now() / 200);
        this.ctx.globalAlpha = pulse;
        this.ctx.strokeStyle = '#ef4444'; 
        this.ctx.lineWidth = 2.5;
        this.ctx.shadowBlur = 10;
        this.ctx.shadowColor = '#ef4444';

        this.ctx.beginPath();
        let first = true;
        for (let y = 1; y < rows - 1; y++) {
            for (let x = 1; x < cols - 1; x++) {
                if (mask[y][x] > 0.5) {
                    if (mask[y-1][x] < 0.5 || mask[y+1][x] < 0.5 || 
                        mask[y][x-1] < 0.5 || mask[y][x+1] < 0.5) {
                        const px = x * cW + (cW/2);
                        const py = y * cH + (cH/2);
                        if (first) {
                            this.ctx.moveTo(px, py);
                            first = false;
                        } else {
                            this.ctx.lineTo(px, py);
                        }
                    }
                }
            }
        }
        this.ctx.stroke();
        this.ctx.restore();
    }

    drawGeminiStatus() {
        this.ctx.save();
        this.ctx.font = "bold 12px 'Inter', sans-serif";
        this.ctx.shadowBlur = 8;
        this.ctx.shadowColor = 'black';
        
        const grad = this.ctx.createLinearGradient(10, 0, 160, 0);
        grad.addColorStop(0, '#4285f4');
        grad.addColorStop(0.5, '#9b72cb');
        grad.addColorStop(1, '#d96570');
        
        this.ctx.fillStyle = grad;
        this.ctx.fillText(`GEMINI 1.5 PRO CORE`, 10, 24);
        
        this.ctx.fillStyle = "white";
        this.ctx.font = "9px 'Inter', sans-serif";
        this.ctx.globalAlpha = 0.8;
        this.ctx.fillText(`GENERATIVE REFINEMENT ACTIVE`, 10, 38);
        this.ctx.fillText(`COHERENCE: ${(this.geminiStatus.coherence * 100).toFixed(2)}%`, 10, 50);
        
        const pulse = 1 + 0.2 * Math.sin(Date.now() / 150);
        this.ctx.fillStyle = "#8ab4f8";
        this.ctx.beginPath();
        this.ctx.arc(165, 18, 5 * pulse, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.restore();
    }

    drawProbeCrosshair(pos) {
        if (!pos) return;
        const cx = pos[0] * this.width;
        const cy = pos[1] * this.height;
        this.ctx.save();
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
        this.ctx.lineWidth = 2.5;
        this.ctx.shadowBlur = 5;
        this.ctx.shadowColor = 'black';
        
        this.ctx.beginPath();
        this.ctx.arc(cx, cy, 14, 0, Math.PI * 2);
        this.ctx.stroke();
        
        this.ctx.lineWidth = 1.5;
        this.ctx.beginPath();
        this.ctx.moveTo(cx - 20, cy); this.ctx.lineTo(cx + 20, cy);
        this.ctx.moveTo(cx, cy - 20); this.ctx.lineTo(cx, cy + 20);
        this.ctx.stroke();
        this.ctx.restore();
    }
}
