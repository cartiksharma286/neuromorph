// k-Space Visualizer
// Renders k-space trajectories for various readout types

const KSpaceVisualizer = {
    /**
     * Draw k-space trajectory on canvas
     */
    drawTrajectory(canvas, trajectory, options = {}) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.fillStyle = '#1a2235';
        ctx.fillRect(0, 0, width, height);

        // Configuration
        const margin = 40;
        const plotSize = Math.min(width, height) - 2 * margin;
        const centerX = width / 2;
        const centerY = height / 2;

        // Draw axes
        this.drawAxes(ctx, centerX, centerY, plotSize);

        // Scale trajectory to fit
        const kmax = trajectory.kmax || this.findKMax(trajectory);
        const scale = plotSize / (2 * kmax);

        // Draw trajectory
        if (options.showAllInterleaves && trajectory.interleaves) {
            this.drawMultipleInterleaves(ctx, trajectory, centerX, centerY, scale, options);
        } else {
            this.drawSingleTrajectory(ctx, trajectory, centerX, centerY, scale, options);
        }

        // Draw sampling density if requested
        if (options.showDensity) {
            this.drawSamplingDensity(ctx, trajectory, centerX, centerY, scale);
        }
    },

    /**
     * Draw coordinate axes
     */
    drawAxes(ctx, centerX, centerY, size) {
        ctx.strokeStyle = 'rgba(139, 148, 158, 0.3)';
        ctx.lineWidth = 1;

        // kx axis
        ctx.beginPath();
        ctx.moveTo(centerX - size / 2, centerY);
        ctx.lineTo(centerX + size / 2, centerY);
        ctx.stroke();

        // ky axis
        ctx.beginPath();
        ctx.moveTo(centerX, centerY - size / 2);
        ctx.lineTo(centerX, centerY + size / 2);
        ctx.stroke();

        // Circle at k=0
        ctx.strokeStyle = 'rgba(88, 166, 255, 0.5)';
        ctx.beginPath();
        ctx.arc(centerX, centerY, 5, 0, 2 * Math.PI);
        ctx.stroke();

        // Labels
        ctx.fillStyle = '#8b949e';
        ctx.font = '12px Inter';
        ctx.fillText('kₓ', centerX + size / 2 + 10, centerY);
        ctx.fillText('kᵧ', centerX, centerY - size / 2 - 10);
    },

    /**
     * Find maximum k-space extent
     */
    findKMax(trajectory) {
        let kmax = 0;

        for (let i = 0; i < trajectory.kx.length; i++) {
            const k = Math.sqrt(trajectory.kx[i] ** 2 + trajectory.ky[i] ** 2);
            kmax = Math.max(kmax, k);
        }

        return kmax;
    },

    /**
     * Draw single trajectory
     */
    drawSingleTrajectory(ctx, trajectory, centerX, centerY, scale, options) {
        const gradient = options.gradient !== false;

        ctx.lineWidth = 2;

        for (let i = 1; i < trajectory.kx.length; i++) {
            const x1 = centerX + trajectory.kx[i - 1] * scale;
            const y1 = centerY - trajectory.ky[i - 1] * scale; // Flip y for canvas
            const x2 = centerX + trajectory.kx[i] * scale;
            const y2 = centerY - trajectory.ky[i] * scale;

            // Color based on position or ADC status
            if (gradient) {
                const hue = (i / trajectory.kx.length) * 360;
                ctx.strokeStyle = `hsl(${hue}, 70%, 60%)`;
            } else {
                const isADCOn = !trajectory.adc || trajectory.adc[i];
                ctx.strokeStyle = isADCOn ? '#3fb950' : 'rgba(139, 148, 158, 0.3)';
            }

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }

        // Starting point
        ctx.fillStyle = '#3fb950';
        ctx.beginPath();
        ctx.arc(centerX + trajectory.kx[0] * scale, centerY - trajectory.ky[0] * scale, 4, 0, 2 * Math.PI);
        ctx.fill();

        // Ending point
        ctx.fillStyle = '#f85149';
        const lastIdx = trajectory.kx.length - 1;
        ctx.beginPath();
        ctx.arc(centerX + trajectory.kx[lastIdx] * scale, centerY - trajectory.ky[lastIdx] * scale, 4, 0, 2 * Math.PI);
        ctx.fill();
    },

    /**
     * Draw multiple interleaves
     */
    drawMultipleInterleaves(ctx, baseTrajectory, centerX, centerY, scale, options) {
        const numInterleaves = baseTrajectory.interleaves || 8;
        const angleStep = (2 * Math.PI) / numInterleaves;

        for (let i = 0; i < numInterleaves; i++) {
            const angle = i * angleStep;
            const cos_a = Math.cos(angle);
            const sin_a = Math.sin(angle);

            const hue = (i / numInterleaves) * 360;
            ctx.strokeStyle = `hsla(${hue}, 70%, 60%, 0.7)`;
            ctx.lineWidth = 1.5;

            ctx.beginPath();

            for (let j = 0; j < baseTrajectory.kx.length; j++) {
                const kx_rot = baseTrajectory.kx[j] * cos_a - baseTrajectory.ky[j] * sin_a;
                const ky_rot = baseTrajectory.kx[j] * sin_a + baseTrajectory.ky[j] * cos_a;

                const x = centerX + kx_rot * scale;
                const y = centerY - ky_rot * scale;

                if (j === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }

            ctx.stroke();
        }
    },

    /**
     * Draw sampling density heatmap
     */
    drawSamplingDensity(ctx, trajectory, centerX, centerY, scale) {
        // Create density map
        const gridSize = 64;
        const density = Array(gridSize).fill(0).map(() => Array(gridSize).fill(0));

        // Accumulate samples
        for (let i = 0; i < trajectory.kx.length; i++) {
            const kx_norm = (trajectory.kx[i] / trajectory.kmax + 1) / 2; // 0 to 1
            const ky_norm = (trajectory.ky[i] / trajectory.kmax + 1) / 2;

            const ix = Math.floor(kx_norm * (gridSize - 1));
            const iy = Math.floor(ky_norm * (gridSize - 1));

            if (ix >= 0 && ix < gridSize && iy >= 0 && iy < gridSize) {
                density[iy][ix]++;
            }
        }

        // Find max density
        const maxDensity = Math.max(...density.map(row => Math.max(...row)));

        // Draw heatmap
        const cellSize = (scale * 2 * trajectory.kmax) / gridSize;

        for (let iy = 0; iy < gridSize; iy++) {
            for (let ix = 0; ix < gridSize; ix++) {
                if (density[iy][ix] > 0) {
                    const alpha = density[iy][ix] / maxDensity;
                    ctx.fillStyle = `rgba(88, 166, 255, ${alpha * 0.5})`;

                    const x = centerX - scale * trajectory.kmax + ix * cellSize;
                    const y = centerY - scale * trajectory.kmax + iy * cellSize;

                    ctx.fillRect(x, y, cellSize, cellSize);
                }
            }
        }
    },

    /**
     * Animate trajectory acquisition
     */
    animateTrajectory(canvas, trajectory, duration = 2000) {
        const startTime = Date.now();
        const totalPoints = trajectory.kx.length;

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const currentPoint = Math.floor(progress * totalPoints);

            // Draw trajectory up to current point
            const partialTraj = {
                kx: trajectory.kx.slice(0, currentPoint),
                ky: trajectory.ky.slice(0, currentPoint),
                kmax: trajectory.kmax
            };

            this.drawTrajectory(canvas, partialTraj, { gradient: true });

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        animate();
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = KSpaceVisualizer;
}
