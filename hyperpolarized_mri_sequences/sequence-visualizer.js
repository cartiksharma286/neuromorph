// Sequence Visualizer
// Creates timing diagrams for pulse sequences

const SequenceVisualizer = {
    /**
     * Draw sequence timing diagram on canvas
     */
    drawTimingDiagram(canvas, sequence, options = {}) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.fillStyle = '#1a2235';
        ctx.fillRect(0, 0, width, height);

        // Configuration
        const margin = { top: 40, right: 40, bottom: 40, left: 80 };
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;

        // Channels to display
        const channels = ['RF', 'Gx', 'Gy', 'Gz', 'ADC'];
        const channelHeight = plotHeight / channels.length;

        // Draw channel labels and separators
        ctx.font = '14px Inter';
        ctx.fillStyle = '#8b949e';
        ctx.strokeStyle = 'rgba(88, 166, 255, 0.15)';

        channels.forEach((channel, i) => {
            const y = margin.top + i * channelHeight;
            ctx.fillText(channel, 20, y + channelHeight / 2);
            ctx.beginPath();
            ctx.moveTo(margin.left, y);
            ctx.lineTo(width - margin.right, y);
            ctx.stroke();
        });

        // Draw time axis
        this.drawTimeAxis(ctx, margin, plotWidth, options.duration || 100);

        // Draw sequence events
        if (sequence.events) {
            this.drawEvents(ctx, sequence.events, margin, plotWidth, channelHeight, options);
        }
    },

    /**
     * Draw time axis
     */
    drawTimeAxis(ctx, margin, width, duration) {
        const numTicks = 10;
        const tickSpacing = width / numTicks;

        ctx.strokeStyle = '#8b949e';
        ctx.fillStyle = '#8b949e';
        ctx.font = '12px Inter';

        for (let i = 0; i <= numTicks; i++) {
            const x = margin.left + i * tickSpacing;
            const time = (i / numTicks) * duration;

            ctx.beginPath();
            ctx.moveTo(x, margin.top);
            ctx.lineTo(x, margin.top - 5);
            ctx.stroke();

            ctx.fillText(time.toFixed(1) + ' ms', x - 15, margin.top - 10);
        }
    },

    /**
     * Draw sequence events
     */
    drawEvents(ctx, events, margin, plotWidth, channelHeight, options) {
        const duration = options.duration || 100;
        const timeScale = plotWidth / duration;

        events.forEach(event => {
            const startX = margin.left + event.startTime * timeScale;
            const endX = margin.left + event.endTime * timeScale;
            const channelIndex = this.getChannelIndex(event.channel);

            if (channelIndex === -1) return;

            const y = margin.top + channelIndex * channelHeight + channelHeight / 2;

            if (event.type === 'rf') {
                this.drawRFPulse(ctx, startX, endX, y, event);
            } else if (event.type === 'gradient') {
                this.drawGradient(ctx, startX, endX, y, event);
            } else if (event.type === 'adc') {
                this.drawADC(ctx, startX, endX, y, event);
            }
        });
    },

    /**
     * Get channel index
     */
    getChannelIndex(channel) {
        const channels = ['RF', 'Gx', 'Gy', 'Gz', 'ADC'];
        return channels.indexOf(channel);
    },

    /**
     * Draw RF pulse
     */
    drawRFPulse(ctx, startX, endX, y, event) {
        ctx.fillStyle = 'rgba(102, 126, 234, 0.3)';
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 2;

        // Draw sinc envelope
        const numPoints = 50;
        ctx.beginPath();

        for (let i = 0; i <= numPoints; i++) {
            const x = startX + (endX - startX) * (i / numPoints);
            const t = (i / numPoints - 0.5) * 2;
            const amplitude = t === 0 ? 1 : Math.sin(Math.PI * t * 3) / (Math.PI * t * 3);
            const yPos = y - amplitude * 30;

            if (i === 0) ctx.moveTo(x, yPos);
            else ctx.lineTo(x, yPos);
        }

        ctx.stroke();

        // Fill area
        ctx.lineTo(endX, y);
        ctx.lineTo(startX, y);
        ctx.closePath();
        ctx.fill();

        // Label
        if (event.flipAngle) {
            ctx.fillStyle = '#667eea';
            ctx.font = '11px Inter';
            ctx.fillText(event.flipAngle.toFixed(0) + '°', (startX + endX) / 2 - 10, y - 40);
        }
    },

    /**
     * Draw gradient
     */
    drawGradient(ctx, startX, endX, y, event) {
        ctx.strokeStyle = '#58a6ff';
        ctx.lineWidth = 2;
        ctx.fillStyle = 'rgba(88, 166, 255, 0.2)';

        const amplitude = event.amplitude || 1;
        const height = amplitude * 25;

        // Trapezoid
        ctx.beginPath();
        ctx.moveTo(startX, y);
        ctx.lineTo(startX + 5, y - height);
        ctx.lineTo(endX - 5, y - height);
        ctx.lineTo(endX, y);
        ctx.closePath();

        ctx.fill();
        ctx.stroke();
    },

    /**
     * Draw ADC window
     */
    drawADC(ctx, startX, endX, y, event) {
        ctx.fillStyle = 'rgba(62, 185, 80, 0.3)';
        ctx.strokeStyle = '#3fb950';
        ctx.lineWidth = 2;

        const height = 20;

        // Rectangle with dots
        ctx.fillRect(startX, y - height / 2, endX - startX, height);
        ctx.strokeRect(startX, y - height / 2, endX - startX, height);

        // Draw sampling points
        const numDots = Math.min(20, Math.floor((endX - startX) / 5));
        ctx.fillStyle = '#3fb950';

        for (let i = 0; i < numDots; i++) {
            const x = startX + (endX - startX) * (i / (numDots - 1));
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, 2 * Math.PI);
            ctx.fill();
        }
    },

    /**
     * Create example sequence for testing
     */
    createExampleSequence() {
        return {
            duration: 100, // ms
            events: [
                { type: 'rf', channel: 'RF', startTime: 0, endTime: 2, flipAngle: 90 },
                { type: 'gradient', channel: 'Gz', startTime: 0, endTime: 2, amplitude: 1 },
                { type: 'gradient', channel: 'Gx', startTime: 10, endTime: 50, amplitude: 0.8 },
                { type: 'adc', channel: 'ADC', startTime: 12, endTime: 48 },
                { type: 'gradient', channel: 'Gy', startTime: 52, endTime: 54, amplitude: 0.5 }
            ]
        };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SequenceVisualizer;
}
