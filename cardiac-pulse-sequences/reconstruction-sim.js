// Reconstruction Simulation Module
const ReconstructionSim = {
    init() {
        console.log('Reconstruction Sim module initialized');
    },

    /**
     * Simulate reconstruction artifacts
     */
    simulateArtifacts(config) {
        const artifacts = [];
        const R = config.parallelImaging.accelerationFactor;

        // Aliasing artifacts
        if (config.parallelImaging.technique === 'sense') {
            if (R > 3) {
                artifacts.push({
                    type: 'aliasing',
                    severity: 'high',
                    description: 'Residual aliasing expected in center due to high R'
                });
            }
        }

        // Noise amplification
        const gFactor = parseFloat(config.gFactor || 1);
        if (gFactor > 1.5) {
            artifacts.push({
                type: 'noise',
                severity: gFactor > 2.5 ? 'high' : 'medium',
                description: `Significant noise amplification (g-factor ${gFactor})`
            });
        }

        return artifacts;
    },

    /**
     * Calculate effective resolution
     */
    calculateEffectiveResolution(fov, matrix, blurFactor = 1.0) {
        const nominalRes = fov / matrix;
        return {
            nominal: nominalRes,
            effective: nominalRes * blurFactor,
            blur: (blurFactor - 1) * 100
        };
    },

    /**
     * Estimate reconstruction time
     */
    estimateReconTime(matrix, slices, technique) {
        const baseTime = (matrix * matrix * slices) / 1e6; // Scale factor

        switch (technique) {
            case 'grappa': return baseTime * 1.5;
            case 'sense': return baseTime * 1.2;
            case 'compressed-sensing': return baseTime * 10; // Iterative
            case 'quantum': return baseTime * 5 + 0.5; // API overhead + complex recon
            default: return baseTime;
        }
    }
};

window.ReconstructionSim = ReconstructionSim;
