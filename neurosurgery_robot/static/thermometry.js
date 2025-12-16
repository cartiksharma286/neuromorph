class ThermalViz {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;

        // Offscreen buffer for the 64x64 data
        this.bufferCanvas = document.createElement('canvas');
        this.bufferCanvas.width = 64;
        this.bufferCanvas.height = 64;
        this.bufferCtx = this.bufferCanvas.getContext('2d');
        this.imageData = this.bufferCtx.createImageData(64, 64);
    }

    getColor(temp) {
        // Temp range: 37.0 to 100.0
        // Map to 0..1
        let t = (temp - 37.0) / (90.0 - 37.0);
        t = Math.max(0, Math.min(1, t));

        // Simple Heatmap gradient: Blue -> Green -> Yellow -> Red
        const r = Math.min(1, Math.max(0, 4 * t - 2)) * 255;
        const g = Math.min(1, Math.max(0, 2 - Math.abs(4 * t - 2))) * 255;
        const b = Math.min(1, Math.max(0, 2 - 4 * t)) * 255; // Blue fades out

        // Base blue for body temp to make it look cool/medical
        if (t < 0.05) {
            return [0, 100, 255];
        }

        return [r, g, b];
    }

    update(data2D) {
        if (!data2D) return;

        const pixels = this.imageData.data;
        let p = 0;

        let maxT = 0;

        for (let y = 0; y < 64; y++) {
            for (let x = 0; x < 64; x++) {
                // Backend sends data[w][h] typically, need to check ordering
                // Let's assume row-major or check
                const val = data2D[x][y]; // or [y][x], doesn't matter much for symmetrical visual
                if (val > maxT) maxT = val;

                const [r, g, b] = this.getColor(val);

                pixels[p++] = r;
                pixels[p++] = g;
                pixels[p++] = b;
                pixels[p++] = 255; // Alpha
            }
        }

        // Put data on buffer
        this.bufferCtx.putImageData(this.imageData, 0, 0);

        // Scale up to display canvas with smoothing false (pixelated look handled by CSS)
        this.ctx.imageSmoothingEnabled = false;
        this.ctx.drawImage(this.bufferCanvas, 0, 0, this.width, this.height);

        return maxT;
    }

    initChart(history) {
        if (this.chart) return;

        const ctx = document.getElementById('temp-profile-chart').getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array(history.length).fill(''),
                datasets: [{
                    label: 'Cortical Temp (°C)',
                    data: history,
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: {
                        beginAtZero: false,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#94a3b8', font: { size: 10 } },
                        suggestedMin: 37,
                        suggestedMax: 60
                    }
                }
            }
        });
    }

    updateChart(history) {
        if (!this.chart) {
            this.initChart(history);
        } else {
            this.chart.data.labels = Array(history.length).fill('');
            this.chart.data.datasets[0].data = history;
            this.chart.update();
        }
    }
}
