// EPI Configurator
// Echo Planar Imaging sequence design for hyperpolarized imaging

const EPIConfigurator = {
    /**
     * Design EPI trajectory
     */
    designEPI(matrixSize, fov, shots, partialFourier, trajectoryType, maxGrad = 40, maxSlew = 150) {
        const matrix = matrixSize;
        const fovCm = fov;
        const resolution = (fovCm * 10) / matrix; // mm

        // Calculate k-space parameters
        const kmax = 1 / (2 * resolution * 0.1); // cm^-1
        const dk = (2 * kmax) / matrix;

        // Partial Fourier
        const pfFactor = parseFloat(partialFourier);
        const numLines = Math.round(matrix * pfFactor);
        const linesPerShot = Math.ceil(numLines / shots);

        // Timing calculations
        const gamma = 42.58; // MHz/T for 1H
        const gradRampTime = (maxGrad * 0.001) / maxSlew * 1000; // ms
        const dwellTime = 1 / (gamma * fovCm); // ms, sets bandwidth
        const echoSpacing = dwellTime * matrix + 2 * gradRampTime; // ms

        const trajectory = {
            kx: [],
            ky: [],
            gx: [],
            gy: [],
            adc: [],
            echoSpacing: echoSpacing,
            effectiveTE: echoSpacing * (matrix / 2),
            acquisitionTime: echoSpacing * linesPerShot,
            matrixSize: matrix,
            shots: shots,
            linesPerShot: linesPerShot,
            readoutGradient: maxGrad * 0.8, // mT/m
            phaseGradient: 0,
            bandwidthPerPixel: 1 / (dwellTime * 1e-3) / matrix // Hz
        };

        // Generate trajectory for one shot
        let currentKy = shots > 1 ? -kmax : -kmax * pfFactor;

        for (let line = 0; line < linesPerShot; line++) {
            const direction = line % 2 === 0 ? 1 : -1; // Alternate readout direction

            if (trajectoryType === 'blipped') {
                // Blipped EPI - standard
                this.addBlippedLine(trajectory, kmax, currentKy, direction, matrix, dk);
            } else {
                // Flyback EPI
                this.addFlybackLine(trajectory, kmax, currentKy, matrix, dk);
            }

            currentKy += dk * shots; // Account for shot interleaving
        }

        return trajectory;
    },

    /**
     * Add blipped EPI line to trajectory
     */
    addBlippedLine(trajectory, kmax, ky, direction, numPoints, dk) {
        const kxStart = direction > 0 ? -kmax : kmax;
        const kxEnd = direction > 0 ? kmax : -kmax;
        const dkx = (kxEnd - kxStart) / (numPoints - 1);

        for (let i = 0; i < numPoints; i++) {
            const kx = kxStart + i * dkx;
            trajectory.kx.push(kx);
            trajectory.ky.push(ky);
            trajectory.adc.push(true);
        }

        // Gradient calculation (simplified)
        const readGrad = trajectory.readoutGradient * direction;
        const phaseBlip = dk; // Simplified

        for (let i = 0; i < numPoints; i++) {
            trajectory.gx.push(readGrad);
            trajectory.gy.push(i === numPoints - 1 ? phaseBlip : 0);
        }
    },

    /**
     * Add flyback EPI line to trajectory
     */
    addFlybackLine(trajectory, kmax, ky, numPoints, dk) {
        const kxStart = -kmax;
        const kxEnd = kmax;
        const dkx = (kxEnd - kxStart) / (numPoints - 1);

        // Forward readout only
        for (let i = 0; i < numPoints; i++) {
            const kx = kxStart + i * dkx;
            trajectory.kx.push(kx);
            trajectory.ky.push(ky);
            trajectory.adc.push(true);
            trajectory.gx.push(trajectory.readoutGradient);
            trajectory.gy.push(0);
        }

        // Flyback (no acquisition)
        const flybackPoints = Math.floor(numPoints / 4);
        for (let i = 0; i < flybackPoints; i++) {
            const kx = kxEnd - (kxEnd - kxStart) * (i / flybackPoints);
            trajectory.kx.push(kx);
            trajectory.ky.push(ky + dk);
            trajectory.adc.push(false);
            trajectory.gx.push(-trajectory.readoutGradient * 2); // Faster flyback
            trajectory.gy.push(dk / flybackPoints);
        }
    },

    /**
     * Generate multi-shot trajectories
     */
    generateMultiShot(baseTrajectory, shots) {
        const shotTrajectories = [];

        for (let shot = 0; shot < shots; shot++) {
            const shotTraj = {
                kx: [],
                ky: [],
                gx: [],
                gy: [],
                adc: [],
                shot: shot
            };

            // Interleave lines
            for (let i = shot; i < baseTrajectory.kx.length; i += shots) {
                shotTraj.kx.push(baseTrajectory.kx[i]);
                shotTraj.ky.push(baseTrajectory.ky[i]);
                shotTraj.gx.push(baseTrajectory.gx[i]);
                shotTraj.gy.push(baseTrajectory.gy[i]);
                shotTraj.adc.push(baseTrajectory.adc[i]);
            }

            shotTrajectories.push(shotTraj);
        }

        return shotTrajectories;
    },

    /**
     * Calculate EPI metrics
     */
    calculateMetrics(trajectory) {
        return {
            echoSpacing: trajectory.echoSpacing.toFixed(3) + ' ms',
            effectiveTE: trajectory.effectiveTE.toFixed(2) + ' ms',
            acquisitionTime: trajectory.acquisitionTime.toFixed(2) + ' ms',
            bandwidthPerPixel: (trajectory.bandwidthPerPixel / 1000).toFixed(2) + ' kHz',
            totalBandwidth: ((trajectory.bandwidthPerPixel * trajectory.matrixSize) / 1000).toFixed(2) + ' kHz',
            linesPerShot: trajectory.linesPerShot,
            totalAcquisitionTime: (trajectory.acquisitionTime * trajectory.shots).toFixed(2) + ' ms'
        };
    },

    /**
     * Calculate distortion metrics
     */
    calculateDistortion(trajectory, b0Field = 3.0) {
        // Estimate geometric distortion
        const echoSpacingMs = trajectory.echoSpacing;
        const bwPerPixel = trajectory.bandwidthPerPixel;

        // Typical off-resonance
        const offResonanceHz = 100; // Hz at 3T
        const distortionPixels = offResonanceHz / bwPerPixel;

        return {
            estimatedDistortion: distortionPixels.toFixed(2) + ' pixels',
            recommendedShimming: distortionPixels > 2 ? 'High-order shimming recommended' : 'Standard shimming sufficient'
        };
    },

    /**
     * Export for PyPulseq
     */
    exportForPyPulseq(trajectory) {
        const code = `# EPI Sequence - Generated by Hyperpolarized Sequence Generator
import numpy as np
from pypulseq import Sequence, make_adc, make_trapezoid, make_delay

# EPI parameters
matrix_size = ${trajectory.matrixSize}
echo_spacing = ${trajectory.echoSpacing} # ms
lines_per_shot = ${trajectory.linesPerShot}
num_shots = ${trajectory.shots}

# Gradient parameters
readout_grad = ${trajectory.readoutGradient} # mT/m
bandwidth_per_pixel = ${trajectory.bandwidthPerPixel} # Hz

# k-space trajectory (first shot)
kx = np.array([${trajectory.kx.slice(0, 20).map(k => k.toFixed(6)).join(', ')}...])  # ${trajectory.kx.length} points
ky = np.array([${trajectory.ky.slice(0, 20).map(k => k.toFixed(6)).join(', ')}...])

# ADC windows
adc_on = np.array([${trajectory.adc.slice(0, 20).map(a => a ? '1' : '0').join(', ')}...])

# Build sequence
seq = Sequence()

# Add EPI readout module
# See pypulseq_generator.py for complete implementation
`;

        return code;
    },

    /**
     * Validate EPI parameters
     */
    validateParameters(matrixSize, fov, shots, maxGrad, maxSlew) {
        const warnings = [];
        const errors = [];

        // Check matrix size
        if (matrixSize % 2 !== 0) {
            warnings.push('Matrix size should be even for optimal EPI performance');
        }

        if (matrixSize > 128) {
            warnings.push('Large matrix sizes may exceed gradient duty cycle limits');
        }

        // Check FOV
        if (fov < 10 || fov > 50) {
            warnings.push('FOV outside typical range (10-50 cm)');
        }

        // Check gradient limits
        if (maxGrad > 50) {
            warnings.push('Gradient amplitude exceeds typical clinical limits');
        }

        if (maxSlew > 200) {
            warnings.push('Slew rate exceeds typical clinical limits');
        }

        return { warnings, errors };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EPIConfigurator;
}
