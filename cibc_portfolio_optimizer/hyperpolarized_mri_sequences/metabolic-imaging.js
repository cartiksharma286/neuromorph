// Metabolic Imaging Sequence Designer
// Spectral-spatial excitation for multi-metabolite imaging

const MetabolicImaging = {
    /**
     * Design spectral-spatial pulse
     */
    designSpectralSpatial(metabolites, b0Field, sliceThickness, pulseType) {
        const pulse = {
            type: pulseType,
            metabolites: metabolites,
            b0Field: b0Field,
            sliceThickness: sliceThickness,
            subpulses: []
        };

        // Get chemical shifts
        const shifts = this.getMetaboliteShifts(metabolites, b0Field);
        pulse.chemicalShifts = shifts;

        if (pulseType === 'spectral-spatial') {
            pulse.duration = 8; // ms typical
            pulse.numSubpulses = 16;
            pulse.subpulseDuration = pulse.duration / pulse.numSubpulses;

            // Generate subpulse pattern
            for (let i = 0; i < pulse.numSubpulses; i++) {
                const phase = this.calculateSubpulsePhase(i, shifts);
                const amplitude = this.calculateSubpulseAmplitude(i, pulse.numSubpulses);

                pulse.subpulses.push({
                    index: i,
                    phase: phase,
                    amplitude: amplitude,
                    duration: pulse.subpulseDuration
                });
            }
        } else if (pulseType === 'multiband') {
            pulse.duration = 4; // ms
            pulse.bands = shifts.length;
        }

        // Calculate gradient for slice selection
        const gamma = 42.58; // MHz/T for 1H (13C is 10.7)
        const gammaC13 = 10.7084;
        const sliceThicknessCm = sliceThickness / 10; // Convert mm to cm

        pulse.sliceGradient = (2 * Math.PI) / (gamma * 2 * Math.PI * sliceThicknessCm * pulse.duration * 0.001);

        return pulse;
    },

    /**
     * Get chemical shifts for metabolites
     */
    getMetaboliteShifts(metabolites, b0Field) {
        const shiftsPPM = {
            'pyruvate': 171.0,
            'lactate': 183.3,
            'alanine': 176.5,
            'bicarbonate': 161.0
        };

        const gamma13C = 10.7084; // MHz/T
        const larmorFreq = gamma13C * b0Field; // MHz

        const shifts = [];

        for (const metabolite of metabolites) {
            if (shiftsPPM[metabolite]) {
                shifts.push({
                    name: metabolite,
                    ppm: shiftsPPM[metabolite],
                    frequencyOffset: shiftsPPM[metabolite] * larmorFreq // Hz
                });
            }
        }

        return shifts;
    },

    /**
     * Calculate subpulse phase for spectral selectivity
     */
    calculateSubpulsePhase(index, shifts) {
        // Simplified phase modulation for spectral selectivity
        // In practice, this uses iterative optimization (SLR algorithm)

        if (shifts.length === 0) return 0;

        // Basic linear phase for first target frequency
        const targetFreq = shifts[0].frequencyOffset;
        const phase = 2 * Math.PI * targetFreq * index * 0.001; // Simplified

        return phase % (2 * Math.PI);
    },

    /**
     * Calculate subpulse amplitude (envelope)
     */
    calculateSubpulseAmplitude(index, numPulses) {
        // Hamming window for spatial profile
        const n = index / (numPulses - 1);
        const amplitude = 0.54 - 0.46 * Math.cos(2 * Math.PI * n);

        return amplitude;
    },

    /**
     * Design flyback EPSI sequence
     */
    designFlybackEPSI(matrixSize, fov, spectralPoints, spectralWidth) {
        const epsi = {
            spatialMatrix: matrixSize,
            spectralPoints: spectralPoints,
            fov: fov,
            spectralWidth: spectralWidth, // Hz
            dwellTime: 1000 / spectralWidth, // ms
            totalReadoutTime: 0
        };

        // Each spatial point has a spectral dimension
        epsi.readoutDuration = spectralPoints * epsi.dwellTime;
        epsi.totalReadoutTime = epsi.readoutDuration * matrixSize;

        // Flyback gradient
        epsi.flybackDuration = epsi.readoutDuration * 0.3;
        epsi.lineTime = epsi.readoutDuration + epsi.flybackDuration;

        epsi.totalAcquisitionTime = epsi.lineTime * matrixSize;

        return epsi;
    },

    /**
     * Calculate spectral profile
     */
    calculateSpectralProfile(pulse, frequencyRange) {
        const profile = [];
        const frequencies = [];

        // Generate frequency points
        for (let f = frequencyRange.min; f <= frequencyRange.max; f += frequencyRange.step) {
            frequencies.push(f);

            // Calculate excitation profile at this frequency
            // using Bloch simulation (simplified)
            let mx = 0, my = 0, mz = 1;

            for (const subpulse of pulse.subpulses) {
                const dt = subpulse.duration * 0.001; // Convert to seconds
                const b1 = subpulse.amplitude;
                const phase = subpulse.phase;
                const detuning = 2 * Math.PI * f; // rad/s

                // Simplified rotation (should use full Bloch equations)
                const angle = b1 * 360 * dt; // degrees, simplified
                const angleRad = angle * Math.PI / 180;

                // Rotation around effective field
                const rotation = angleRad * Math.cos(phase);
                my += Math.sin(rotation) * mz;
                mz *= Math.cos(rotation);
            }

            const magnitude = Math.sqrt(mx * mx + my * my);
            profile.push(magnitude);
        }

        return {
            frequencies: frequencies,
            profile: profile
        };
    },

    /**
     * Calculate spatial profile
     */
    calculateSpatialProfile(pulse, positionRange) {
        const profile = [];
        const positions = [];

        // Spatial profile is similar to standard slice-selective pulse
        const sliceThicknessCm = pulse.sliceThickness / 10;

        for (let z = positionRange.min; z <= positionRange.max; z += positionRange.step) {
            positions.push(z);

            // Sinc-like profile (simplified)
            const x = (z / sliceThicknessCm) * Math.PI;
            const sinc = x === 0 ? 1 : Math.sin(x) / x;
            const magnitude = Math.abs(sinc);

            profile.push(magnitude);
        }

        return {
            positions: positions,
            profile: profile
        };
    },

    /**
     * Export spectral-spatial pulse
     */
    exportForPyPulseq(pulse) {
        const code = `# Spectral-Spatial Pulse for Metabolic Imaging
# Generated by Hyperpolarized Sequence Generator

import numpy as np
from pypulseq import Sequence, make_sinc_pulse, make_arbitrary_rf

# Pulse parameters
pulse_duration = ${pulse.duration}  # ms
num_subpulses = ${pulse.numSubpulses || 16}
b0_field = ${pulse.b0Field}  # Tesla
slice_thickness = ${pulse.sliceThickness}  # mm

# Target metabolites and chemical shifts
metabolites = ${JSON.stringify(pulse.metabolites)}
chemical_shifts_hz = [${pulse.chemicalShifts ? pulse.chemicalShifts.map(s => s.frequencyOffset.toFixed(1)).join(', ') : ''}]

# Subpulse amplitudes and phases
amplitudes = np.array([${pulse.subpulses ? pulse.subpulses.slice(0, 10).map(p => p.amplitude.toFixed(4)).join(', ') : ''}...])
phases = np.array([${pulse.subpulses ? pulse.subpulses.slice(0, 10).map(p => p.phase.toFixed(4)).join(', ') : ''}...])

# Slice selection gradient
slice_grad = ${pulse.sliceGradient ? pulse.sliceGradient.toFixed(3) : 0}  # mT/m

# Create RF pulse
# rf_pulse = make_arbitrary_rf(signal=amplitudes * np.exp(1j * phases), ...)

# Create sequence with spectral-spatial excitation
seq = Sequence()

# Add metabolite-selective excitation
# For each metabolite, adjust frequency offset and apply pulse

# See full implementation in pypulseq_generator.py
`;

        return code;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MetabolicImaging;
}
