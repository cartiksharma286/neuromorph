// MRI Physics Base Calculations

const Physics = {
    /**
     * Constants
     */
    GAMMA_HYDROGEN: 42.58, // MHz/T

    /**
     * Calculate SNR
     */
    calculateSNR(voxelVolume, bandwidth, averages, fieldStrength) {
        // Simplified SNR calculation
        // SNR ∝ voxel_volume × √(averages) × B0 / √(bandwidth)
        const baseSnr = voxelVolume * Math.sqrt(averages) * fieldStrength / Math.sqrt(bandwidth);
        return baseSnr * 100; // Scaled for display
    },

    /**
     * Calculate readout time
     */
    calculateReadoutTime(matrixSize, bandwidth) {
        // Readout time = matrix_size / bandwidth
        return matrixSize / (bandwidth * 1000); // Convert kHz to Hz
    },

    /**
     * Calculate echo spacing
     */
    calculateEchoSpacing(matrixX, bandwidth) {
        return 1 / (bandwidth * 1000) * matrixX;
    }
};

window.Physics = Physics;
