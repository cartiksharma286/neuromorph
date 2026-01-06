// Cardiac Gating and Triggering Logic
const CardiacGating = {
    state: {
        method: 'retrospective', // retrospective or prospective
        triggerDelay: 0, // ms
        windowWidth: 10, // % of RR
        arrhythmiaRejection: true,
        detectedHeartRate: 60
    },

    init() {
        console.log('Cardiac Gating module initialized');
    },

    /**
     * Calculate available acquisition window
     */
    calculateAcquisitionWindow(heartRate, temporalRes) {
        const rrInterval = 60000 / heartRate;

        // Diastolic window is typically 15-20% of RR in shorter cycles, more in longer
        const diastolicWindow = rrInterval * (heartRate > 90 ? 0.3 : 0.6);

        const availablePhases = Math.floor(diastolicWindow / temporalRes);

        return {
            rrInterval: rrInterval,
            diastolicWindow: diastolicWindow,
            availablePhases: availablePhases,
            triggerTime: rrInterval * 0.1 // Simulated R-wave peak
        };
    },

    /**
     * Simulate Arrhythmia Rejection
     */
    checkArrhythmia(currentRR, tolerance = 0.1) {
        if (!this.state.arrhythmiaRejection) return true;

        const avgRR = 60000 / this.state.detectedHeartRate;
        const diff = Math.abs(currentRR - avgRR);
        const percentDiff = diff / avgRR;

        return percentDiff <= tolerance;
    },

    /**
     * Get gating suggestions based on HR
     */
    getRecommendations(heartRate) {
        if (heartRate > 90) {
            return {
                method: 'retrospective',
                recommendation: 'High HR detected. Use Retrospective Gating to cover full cardiac cycle.',
                triggerDelay: 'Minimal'
            };
        } else if (heartRate < 50) {
            return {
                method: 'prospective',
                recommendation: 'Low HR. Prospective Triggering may be more efficient but excludes late diastole.',
                triggerDelay: 'Adaptive'
            };
        } else {
            return {
                method: 'retrospective',
                recommendation: 'Standard Retrospective Gating recommended for CINE.',
                triggerDelay: 'Adaptive'
            };
        }
    }
};

window.CardiacGating = CardiacGating;
