/**
 * 3D Visualization Engine for Coil Generator
 * Renders coil geometries, field maps, and anatomy
 */

class CoilVisualization {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;

        this.camera = {
            x: 0,
            y: 0,
            z: 3,
            rotX: 0.3,
            rotY: 0.4,
            zoom: 1.0
        };

        this.viewMode = '3d';
    }

    /**
     * Render 3D coil geometry
     */
    renderCoilGeometry(coilData) {
        const positions = coilData.positions;
        const params = coilData.parameters;

        this.clear();

        // Draw coordinate system
        this.drawCoordinateSystem();

        // Draw knee anatomy outline
        this.drawKneeOutline();

        // Draw coil elements
        positions.forEach((pos, idx) => {
            this.drawCoilElement(pos, idx, params);
        });

        // Draw connections between adjacent elements
        this.drawConnections(positions);

        // Draw labels
        this.drawLabels(positions);
    }

    /**
     * Clear canvas
     */
    clear() {
        this.ctx.fillStyle = '#0f1729';
        this.ctx.fillRect(0, 0, this.width, this.height);
    }

    /**
     * Project 3D point to 2D canvas
     */
    project3D(x, y, z) {
        // Apply camera rotation
        let rotX = x;
        let rotY = y * Math.cos(this.camera.rotX) - z * Math.sin(this.camera.rotX);
        let rotZ = y * Math.sin(this.camera.rotX) + z * Math.cos(this.camera.rotX);

        let finalX = rotX * Math.cos(this.camera.rotY) + rotZ * Math.sin(this.camera.rotY);
        let finalY = rotY;
        let finalZ = -rotX * Math.sin(this.camera.rotY) + rotZ * Math.cos(this.camera.rotY);

        // Perspective projection
        const scale = (300 * this.camera.zoom) / (this.camera.z + finalZ);
        const screenX = this.width / 2 + finalX * scale;
        const screenY = this.height / 2 - finalY * scale;

        return { x: screenX, y: screenY, z: finalZ, scale: scale };
    }

    /**
     * Draw coordinate system
     */
    drawCoordinateSystem() {
        const origin = this.project3D(0, 0, 0);
        const xAxis = this.project3D(0.2, 0, 0);
        const yAxis = this.project3D(0, 0.2, 0);
        const zAxis = this.project3D(0, 0, 0.2);

        // X axis (red)
        this.ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(origin.x, origin.y);
        this.ctx.lineTo(xAxis.x, xAxis.y);
        this.ctx.stroke();

        // Y axis (green)
        this.ctx.strokeStyle = 'rgba(16, 185, 129, 0.5)';
        this.ctx.beginPath();
        this.ctx.moveTo(origin.x, origin.y);
        this.ctx.lineTo(yAxis.x, yAxis.y);
        this.ctx.stroke();

        // Z axis (blue)
        this.ctx.strokeStyle = 'rgba(59, 130, 246, 0.5)';
        this.ctx.beginPath();
        this.ctx.moveTo(origin.x, origin.y);
        this.ctx.lineTo(zAxis.x, zAxis.y);
        this.ctx.stroke();
    }

    /**
     * Draw knee anatomy outline
     */
    drawKneeOutline() {
        // Draw cylindrical representation of knee
        const segments = 32;
        const radius = 0.08;
        const length = 0.25;

        // Top circle
        this.ctx.strokeStyle = 'rgba(100, 116, 139, 0.3)';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();

        for (let i = 0; i <= segments; i++) {
            const angle = (2 * Math.PI * i) / segments;
            const x = radius * Math.cos(angle);
            const y = radius * Math.sin(angle);
            const z = length / 2;

            const proj = this.project3D(x, y, z);
            if (i === 0) {
                this.ctx.moveTo(proj.x, proj.y);
            } else {
                this.ctx.lineTo(proj.x, proj.y);
            }
        }
        this.ctx.stroke();

        // Bottom circle
        this.ctx.beginPath();
        for (let i = 0; i <= segments; i++) {
            const angle = (2 * Math.PI * i) / segments;
            const x = radius * Math.cos(angle);
            const y = radius * Math.sin(angle);
            const z = -length / 2;

            const proj = this.project3D(x, y, z);
            if (i === 0) {
                this.ctx.moveTo(proj.x, proj.y);
            } else {
                this.ctx.lineTo(proj.x, proj.y);
            }
        }
        this.ctx.stroke();

        // Vertical lines
        for (let i = 0; i < 4; i++) {
            const angle = (2 * Math.PI * i) / 4;
            const x = radius * Math.cos(angle);
            const y = radius * Math.sin(angle);

            const top = this.project3D(x, y, length / 2);
            const bottom = this.project3D(x, y, -length / 2);

            this.ctx.beginPath();
            this.ctx.moveTo(top.x, top.y);
            this.ctx.lineTo(bottom.x, bottom.y);
            this.ctx.stroke();
        }
    }

    /**
     * Draw individual coil element
     */
    drawCoilElement(position, index, params) {
        const proj = this.project3D(position.x, position.y, position.z);

        // Calculate coil size based on area parameter
        const coilRadius = Math.sqrt(params.area / Math.PI) * 100; // Convert to pixels
        const visualRadius = coilRadius * proj.scale / 300;

        // Draw coil circle
        const gradient = this.ctx.createRadialGradient(
            proj.x, proj.y, 0,
            proj.x, proj.y, visualRadius
        );
        gradient.addColorStop(0, 'rgba(6, 182, 212, 0.6)');
        gradient.addColorStop(0.7, 'rgba(6, 182, 212, 0.3)');
        gradient.addColorStop(1, 'rgba(6, 182, 212, 0.1)');

        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(proj.x, proj.y, visualRadius, 0, 2 * Math.PI);
        this.ctx.fill();

        // Draw border
        this.ctx.strokeStyle = '#06b6d4';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();

        // Draw center point
        this.ctx.fillStyle = '#8b5cf6';
        this.ctx.beginPath();
        this.ctx.arc(proj.x, proj.y, 4, 0, 2 * Math.PI);
        this.ctx.fill();

        // Draw channel number
        this.ctx.fillStyle = '#f0f9ff';
        this.ctx.font = 'bold 12px Inter';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(`${index + 1}`, proj.x, proj.y);
    }

    /**
     * Draw connections between coil elements
     */
    drawConnections(positions) {
        this.ctx.strokeStyle = 'rgba(139, 92, 246, 0.2)';
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([5, 5]);

        for (let i = 0; i < positions.length; i++) {
            const next = (i + 1) % positions.length;
            const p1 = this.project3D(positions[i].x, positions[i].y, positions[i].z);
            const p2 = this.project3D(positions[next].x, positions[next].y, positions[next].z);

            this.ctx.beginPath();
            this.ctx.moveTo(p1.x, p1.y);
            this.ctx.lineTo(p2.x, p2.y);
            this.ctx.stroke();
        }

        this.ctx.setLineDash([]);
    }

    /**
     * Draw labels
     */
    drawLabels(positions) {
        this.ctx.fillStyle = '#cbd5e1';
        this.ctx.font = '11px Inter';
        this.ctx.textAlign = 'left';

        // Title
        this.ctx.fillStyle = '#f0f9ff';
        this.ctx.font = 'bold 14px Inter';
        this.ctx.fillText(`${positions.length}-Channel Knee Coil Array`, 20, 30);

        // Camera info
        this.ctx.fillStyle = '#64748b';
        this.ctx.font = '10px Inter';
        this.ctx.fillText(`View: ${this.viewMode.toUpperCase()}`, 20, this.height - 20);
    }

    /**
     * Render B1 field map
     */
    renderB1FieldMap(canvasId, fieldMap) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        ctx.clearRect(0, 0, width, height);

        // Find min/max B1 values
        const b1Values = fieldMap.map(p => p.B1);
        const maxB1 = Math.max(...b1Values);
        const minB1 = Math.min(...b1Values);

        // Determine resolution
        const resolution = Math.round(Math.cbrt(fieldMap.length));
        const pixelSize = Math.min(width, height) / resolution;

        // Draw heatmap (slice through volume)
        const midSlice = Math.floor(resolution / 2);

        for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
                const idx = i * resolution * resolution + j * resolution + midSlice;
                if (idx < fieldMap.length) {
                    const b1 = fieldMap[idx].B1;
                    const normalized = (b1 - minB1) / (maxB1 - minB1);

                    // Color mapping (blue to cyan to purple)
                    const color = this.heatmapColor(normalized);

                    ctx.fillStyle = color;
                    ctx.fillRect(
                        i * pixelSize,
                        j * pixelSize,
                        pixelSize,
                        pixelSize
                    );
                }
            }
        }

        // Draw border
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 2;
        ctx.strokeRect(0, 0, width, height);

        // Draw title and legend
        ctx.fillStyle = '#f0f9ff';
        ctx.font = 'bold 12px Inter';
        ctx.fillText('B1 Field Strength', 10, 20);

        this.drawColorbar(ctx, width - 40, 40, 20, height - 80, minB1, maxB1);
    }

    /**
     * Render SNR distribution
     */
    renderSNRMap(canvasId, snrMap) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        ctx.clearRect(0, 0, width, height);

        // Find min/max SNR values
        const snrValues = snrMap.map(p => p.SNR);
        const maxSNR = Math.max(...snrValues);
        const minSNR = Math.min(...snrValues);

        // Determine resolution
        const resolution = Math.round(Math.cbrt(snrMap.length));
        const pixelSize = Math.min(width, height) / resolution;

        // Draw heatmap
        const midSlice = Math.floor(resolution / 2);

        for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
                const idx = i * resolution * resolution + j * resolution + midSlice;
                if (idx < snrMap.length) {
                    const snr = snrMap[idx].SNR;
                    const normalized = (snr - minSNR) / (maxSNR - minSNR);

                    const color = this.heatmapColor(normalized);

                    ctx.fillStyle = color;
                    ctx.fillRect(
                        i * pixelSize,
                        j * pixelSize,
                        pixelSize,
                        pixelSize
                    );
                }
            }
        }

        // Draw border
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
        ctx.strokeRect(0, 0, width, height);

        // Draw title
        ctx.fillStyle = '#f0f9ff';
        ctx.font = 'bold 12px Inter';
        ctx.fillText('SNR Distribution', 10, 20);

        this.drawColorbar(ctx, width - 40, 40, 20, height - 80, minSNR, maxSNR);
    }

    /**
     * Generate heatmap color
     */
    heatmapColor(value) {
        // Value from 0 to 1
        const r = Math.round(59 + value * (139 - 59));
        const g = Math.round(130 + value * (92 - 130));
        const b = Math.round(246 + value * (246 - 246));
        return `rgb(${r}, ${g}, ${b})`;
    }

    /**
     * Draw colorbar legend
     */
    drawColorbar(ctx, x, y, width, height, minVal, maxVal) {
        const steps = 50;
        const stepHeight = height / steps;

        for (let i = 0; i < steps; i++) {
            const value = i / steps;
            ctx.fillStyle = this.heatmapColor(value);
            ctx.fillRect(x, y + (steps - i - 1) * stepHeight, width, stepHeight);
        }

        // Border
        ctx.strokeStyle = '#64748b';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, width, height);

        // Labels
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '9px Inter';
        ctx.textAlign = 'left';
        ctx.fillText(maxVal.toExponential(2), x + width + 5, y + 10);
        ctx.fillText(minVal.toExponential(2), x + width + 5, y + height);
    }

    /**
     * Set camera rotation
     */
    rotate(deltaX, deltaY) {
        this.camera.rotY += deltaX * 0.01;
        this.camera.rotX += deltaY * 0.01;

        // Clamp X rotation
        this.camera.rotX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.camera.rotX));
    }

    /**
     * Set view mode
     */
    setViewMode(mode) {
        this.viewMode = mode;

        switch (mode) {
            case 'top':
                this.camera.rotX = Math.PI / 2;
                this.camera.rotY = 0;
                break;
            case 'side':
                this.camera.rotX = 0;
                this.camera.rotY = Math.PI / 2;
                break;
            case '3d':
            default:
                this.camera.rotX = 0.3;
                this.camera.rotY = 0.4;
                break;
        }
    }

    /**
     * Animate rotation
     */
    animateRotation() {
        this.camera.rotY += 0.005;
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CoilVisualization;
}
