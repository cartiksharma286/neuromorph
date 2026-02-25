// Variable Flip Angle (VFA) Calculator
// Implements optimal flip angle strategies for hyperpolarized imaging

const VFACalculator = {
    /**
     * Calculate constant signal approach (CSA) flip angles
     * Maintains constant signal across all acquisitions
     * Reference: Nagashima K. MRM 2008
     */
    calculateConstantSignal(numFrames, t1, tr) {
        const flipAngles = [];
        const trSeconds = tr / 1000; // Convert to seconds

        for (let n = 1; n <= numFrames; n++) {
            // Constant signal formula: θ_n = arctan(1/sqrt(N-n))
            // where N is total frames and n is current frame
            const angle = Math.atan(1 / Math.sqrt(numFrames - n + 1));
            flipAngles.push(angle * 180 / Math.PI); // Convert to degrees
        }

        return flipAngles;
    },

    /**
     * Calculate maximum SNR flip angles
     * Maximizes total SNR across all frames
     * Reference: Larson PEZ et al. MRM 2013
     */
    calculateMaxSNR(numFrames, t1, tr) {
        const flipAngles = [];
        const trSeconds = tr / 1000;
        const e1 = Math.exp(-trSeconds / t1);

        for (let n = 1; n <= numFrames; n++) {
            // Max SNR formula: cos(θ_n) = sqrt((N-n)/(N-n+1)) * e^(-TR/T1)
            const ratio = Math.sqrt((numFrames - n) / (numFrames - n + 1));
            const cosTheta = ratio * e1;

            // Ensure cosTheta is in valid range
            const validCos = Math.min(Math.max(cosTheta, 0), 1);
            const angle = Math.acos(validCos) * 180 / Math.PI;

            flipAngles.push(angle);
        }

        return flipAngles;
    },

    /**
     * Calculate custom linear ramp flip angles
     */
    calculateLinearRamp(numFrames, startAngle, endAngle) {
        const flipAngles = [];
        const step = (endAngle - startAngle) / (numFrames - 1);

        for (let n = 0; n < numFrames; n++) {
            flipAngles.push(startAngle + n * step);
        }

        return flipAngles;
    },

    /**
     * Calculate custom exponential ramp flip angles
     */
    calculateExponentialRamp(numFrames, startAngle, endAngle, exponent = 2) {
        const flipAngles = [];

        for (let n = 0; n < numFrames; n++) {
            const t = n / (numFrames - 1);
            const expT = Math.pow(t, exponent);
            const angle = startAngle + (endAngle - startAngle) * expT;
            flipAngles.push(angle);
        }

        return flipAngles;
    },

    /**
     * Simulate signal evolution with VFA
     */
    simulateSignalEvolution(flipAngles, t1, tr, initialMagnetization = 1.0) {
        const signals = [];
        let magnetization = initialMagnetization;
        const trSeconds = tr / 1000;

        for (let i = 0; i < flipAngles.length; i++) {
            const flipRad = flipAngles[i] * Math.PI / 180;

            // Signal acquired
            const signal = magnetization * Math.sin(flipRad);
            signals.push(signal);

            // Remaining magnetization after excitation
            magnetization *= Math.cos(flipRad);

            // T1 decay during TR (for hyperpolarized, no recovery)
            magnetization *= Math.exp(-trSeconds / t1);
        }

        return signals;
    },

    /**
     * Calculate total SNR for a flip angle schedule
     */
    calculateTotalSNR(flipAngles, t1, tr) {
        const signals = this.simulateSignalEvolution(flipAngles, t1, tr);

        // Total SNR is proportional to sum of squared signals
        const totalSNR = Math.sqrt(signals.reduce((sum, s) => sum + s * s, 0));

        return totalSNR;
    },

    /**
     * Calculate SNR efficiency (normalized to constant flip angle baseline)
     */
    calculateSNREfficiency(flipAngles, t1, tr) {
        const vfaSNR = this.calculateTotalSNR(flipAngles, t1, tr);

        // Baseline: constant 90° flip angles
        const baseline90 = new Array(flipAngles.length).fill(90);
        const baseline90SNR = this.calculateTotalSNR(baseline90, t1, tr);

        return (vfaSNR / baseline90SNR) * 100; // Percentage
    },

    /**
     * Optimize VFA for specific constraints
     */
    optimizeVFA(numFrames, t1, tr, strategy = 'constant-signal', options = {}) {
        let flipAngles;

        switch (strategy) {
            case 'constant-signal':
                flipAngles = this.calculateConstantSignal(numFrames, t1, tr);
                break;
            case 'max-snr':
                flipAngles = this.calculateMaxSNR(numFrames, t1, tr);
                break;
            case 'linear':
                flipAngles = this.calculateLinearRamp(
                    numFrames,
                    options.startAngle || 10,
                    options.endAngle || 90
                );
                break;
            case 'exponential':
                flipAngles = this.calculateExponentialRamp(
                    numFrames,
                    options.startAngle || 10,
                    options.endAngle || 90,
                    options.exponent || 2
                );
                break;
            default:
                flipAngles = this.calculateConstantSignal(numFrames, t1, tr);
        }

        // Apply constraints
        if (options.maxFlipAngle) {
            flipAngles = flipAngles.map(angle => Math.min(angle, options.maxFlipAngle));
        }

        if (options.minFlipAngle) {
            flipAngles = flipAngles.map(angle => Math.max(angle, options.minFlipAngle));
        }

        return flipAngles;
    },

    /**
     * Generate VFA schedule with metadata
     */
    generateSchedule(numFrames, t1, tr, strategy = 'constant-signal', options = {}) {
        const flipAngles = this.optimizeVFA(numFrames, t1, tr, strategy, options);
        const signals = this.simulateSignalEvolution(flipAngles, t1, tr);
        const totalSNR = this.calculateTotalSNR(flipAngles, t1, tr);
        const efficiency = this.calculateSNREfficiency(flipAngles, t1, tr);

        // Generate time points
        const times = [];
        for (let i = 0; i < numFrames; i++) {
            times.push((i * tr) / 1000); // seconds
        }

        return {
            flipAngles,
            signals,
            times,
            totalSNR,
            efficiency,
            strategy,
            parameters: {
                numFrames,
                t1,
                tr
            }
        };
    },

    /**
     * Export VFA schedule as table data
     */
    exportTable(schedule) {
        const rows = [];

        for (let i = 0; i < schedule.flipAngles.length; i++) {
            rows.push({
                frame: i + 1,
                time: schedule.times[i].toFixed(2),
                flipAngle: schedule.flipAngles[i].toFixed(1),
                signal: schedule.signals[i].toFixed(3),
                signalPercent: (schedule.signals[i] * 100).toFixed(1)
            });
        }

        return rows;
    },

    /**
     * Compare different VFA strategies
     */
    compareStrategies(numFrames, t1, tr) {
        const strategies = ['constant-signal', 'max-snr'];
        const results = {};

        for (const strategy of strategies) {
            results[strategy] = this.generateSchedule(numFrames, t1, tr, strategy);
        }

        return results;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VFACalculator;
}
