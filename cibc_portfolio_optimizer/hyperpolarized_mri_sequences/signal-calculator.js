// Signal Calculator
// Calculates and visualizes signal evolution for hyperpolarized imaging

const SignalCalculator = {
    /**
     * Draw signal evolution chart
     */
    drawSignalEvolution(canvas, signals, times, options = {}) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.fillStyle = '#1a2235';
        ctx.fillRect(0, 0, width, height);

        // Configuration
        const margin = { top: 30, right: 30, bottom: 50, left: 60 };
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;

        // Draw axes
        this.drawAxesSignal(ctx, margin, plotWidth, plotHeight, times);

        // Draw signal curve
        this.drawSignalCurve(ctx, signals, times, margin, plotWidth, plotHeight, options);

        // Draw legend if multiple signals
        if (options.labels && options.labels.length > 1) {
            this.drawLegend(ctx, options.labels, width, margin);
        }
    },

    /**
     * Draw axes for signal plot
     */
    drawAxesSignal(ctx, margin, plotWidth, plotHeight, times) {
        const maxTime = Math.max(...times);

        // Y-axis
        ctx.strokeStyle = '#8b949e';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(margin.left, margin.top);
        ctx.lineTo(margin.left, margin.top + plotHeight);
        ctx.stroke();

        // X-axis
        ctx.beginPath();
        ctx.moveTo(margin.left, margin.top + plotHeight);
        ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
        ctx.stroke();

        // Y-axis labels (normalized signal)
        ctx.fillStyle = '#8b949e';
        ctx.font = '12px Inter';
        ctx.textAlign = 'right';

        for (let i = 0; i <= 5; i++) {
            const value = i / 5;
            const y = margin.top + plotHeight - (i / 5) * plotHeight;
            ctx.fillText(value.toFixed(1), margin.left - 10, y + 4);

            // Grid line
            ctx.strokeStyle = 'rgba(139, 148, 158, 0.1)';
            ctx.beginPath();
            ctx.moveTo(margin.left, y);
            ctx.lineTo(margin.left + plotWidth, y);
            ctx.stroke();
        }

        // X-axis labels (time)
        ctx.textAlign = 'center';
        const numTicks = 5;

        for (let i = 0; i <= numTicks; i++) {
            const time = (i / numTicks) * maxTime;
            const x = margin.left + (i / numTicks) * plotWidth;
            ctx.fillText(time.toFixed(1) + ' s', x, margin.top + plotHeight + 25);
        }

        // Axis titles
        ctx.save();
        ctx.translate(20, margin.top + plotHeight / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = '#e6edf3';
        ctx.font = '14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Normalized Signal', 0, 0);
        ctx.restore();

        ctx.fillText('Time (s)', margin.left + plotWidth / 2, margin.top + plotHeight + 45);
    },

    /**
     * Draw signal curve
     */
    drawSignalCurve(ctx, signals, times, margin, plotWidth, plotHeight, options) {
        if (!Array.isArray(signals[0])) {
            // Single signal
            signals = [signals];
        }

        const colors = options. = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#bc8cff'];

        signals.forEach((signalArray, idx) => {
            const color = colors[idx % colors.length];
            const maxSignal = Math.max(...signalArray);
            const maxTime = Math.max(...times);

            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.shadowBlur = 8;
            ctx.shadowColor = color;

            ctx.beginPath();

            for (let i = 0; i < signalArray.length; i++) {
                const x = margin.left + (times[i] / maxTime) * plotWidth;
                const y = margin.top + plotHeight - (signalArray[i] / maxSignal) * plotHeight;

                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }

            ctx.stroke();
            ctx.shadowBlur = 0;

            // Draw points
            ctx.fillStyle = color;
            for (let i = 0; i < signalArray.length; i++) {
                const x = margin.left + (times[i] / maxTime) * plotWidth;
                const y = margin.top + plotHeight - (signalArray[i] / maxSignal) * plotHeight;

                ctx.beginPath();
                ctx.arc(x, y, 3, 0, 2 * Math.PI);
                ctx.fill();
            }
        });
    },

    /**
     * Draw legend
     */
    drawLegend(ctx, labels, width, margin) {
        const colors = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#bc8cff'];
        const legendX = width - margin.right - 150;
        const legendY = margin.top;

        ctx.fillStyle = 'rgba(26, 34, 53, 0.8)';
        ctx.fillRect(legendX - 10, legendY - 10, 160, labels.length * 25 + 20);

        labels.forEach((label, i) => {
            const color = colors[i % colors.length];
            const y = legendY + i * 25;

            // Color box
            ctx.fillStyle = color;
            ctx.fillRect(legendX, y, 20, 3);

            // Label
            ctx.fillStyle = '#e6edf3';
            ctx.font = '12px Inter';
            ctx.textAlign = 'left';
            ctx.fillText(label, legendX + 30, y + 4);
        });
    },

    /**
     * Calculate SNR over time
     */
    calculateSNR(signals, noiseLevel = 0.01) {
        return signals.map(signal => {
            const snr = signal / noiseLevel;
            return 20 * Math.log10(snr); // dB
        });
    },

    /**
     * Calculate total integrated signal
     */
    calculateIntegratedSignal(signals, times) {
        let integral = 0;

        for (let i = 1; i < signals.length; i++) {
            const dt = times[i] - times[i - 1];
            const avgSignal = (signals[i] + signals[i - 1]) / 2;
            integral += avgSignal * dt;
        }

        return integral;
    },

    /**
     * Compare signal strategies
     */
    compareStrategies(strategies) {
        const comparison = [];

        for (const [name, data] of Object.entries(strategies)) {
            comparison.push({
                name: name,
                totalSignal: this.calculateIntegratedSignal(data.signals, data.times),
                peakSignal: Math.max(...data.signals),
                efficiency: data.efficiency || 0
            });
        }

        return comparison;
    },

    /**
     * Export signal data as CSV
     */
    exportCSV(signals, times, labels = []) {
        let csv = 'Time (s)';

        if (labels.length > 0) {
            labels.forEach(label => {
                csv += ',' + label;
            });
        } else {
            csv += ',Signal';
        }

        csv += '\n';

        const signalArrays = Array.isArray(signals[0]) ? signals : [signals];

        for (let i = 0; i < times.length; i++) {
            csv += times[i].toFixed(4);

            signalArrays.forEach(signalArray => {
                csv += ',' + signalArray[i].toFixed(6);
            });

            csv += '\n';
        }

        return csv;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SignalCalculator;
}
