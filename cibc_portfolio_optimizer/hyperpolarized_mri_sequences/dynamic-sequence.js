// Dynamic Time-Resolved Sequence Designer
// For kinetic imaging of hyperpolarized contrast agents

const DynamicSequence = {
    /**
     * Design dynamic time-resolved sequence
     */
    designDynamic(numFrames, temporalResolution, readoutType, vfaEnabled, nucleiProps) {
        const sequence = {
            numFrames: numFrames,
            temporalResolution: temporalResolution, // seconds
            readoutType: readoutType,
            vfaEnabled: vfaEnabled,
            frames: [],
            totalDuration: numFrames * temporalResolution
        };

        // Calculate TR based on temporal resolution and readout
        const readoutTime = this.estimateReadoutTime(readoutType);
        const tr = (temporalResolution * 1000) - readoutTime; // ms

        if (tr < 10) {
            console.warn('TR too short for selected temporal resolution and readout');
        }

        sequence.tr = tr;
        sequence.readoutDuration = readoutTime;

        // Generate VFA schedule if enabled
        if (vfaEnabled && nucleiProps) {
            const vfaSchedule = VFACalculator.generateSchedule(
                numFrames,
                nucleiProps.t1,
                tr,
                'constant-signal'
            );
            sequence.flipAngles = vfaSchedule.flipAngles;
            sequence.predictedSignals = vfaSchedule.signals;
        } else {
            // Constant flip angle
            sequence.flipAngles = new Array(numFrames).fill(20); // Conservative 20°
            sequence.predictedSignals = this.predictConstantFA(numFrames, 20, nucleiProps, tr);
        }

        // Generate frame definitions
        for (let i = 0; i < numFrames; i++) {
            sequence.frames.push({
                number: i + 1,
                startTime: i * temporalResolution,
                flipAngle: sequence.flipAngles[i],
                predictedSignal: sequence.predictedSignals ? sequence.predictedSignals[i] : 1.0
            });
        }

        return sequence;
    },

    /**
     * Estimate readout duration for different trajectory types
     */
    estimateReadoutTime(readoutType) {
        const estimates = {
            'spiral': 20, // ms
            'epi': 50,
            'radial': 30,
            'cartesian': 100
        };

        return estimates[readoutType] || 50;
    },

    /**
     * Predict signal with constant flip angle
     */
    predictConstantFA(numFrames, flipAngle, nucleiProps, tr) {
        if (!nucleiProps) return null;

        const signals = [];
        let magnetization = 1.0;
        const trSeconds = tr / 1000;
        const flipRad = flipAngle * Math.PI / 180;

        for (let i = 0; i < numFrames; i++) {
            const signal = magnetization * Math.sin(flipRad);
            signals.push(signal);

            magnetization *= Math.cos(flipRad);
            magnetization *= Math.exp(-trSeconds / nucleiProps.t1);
        }

        return signals;
    },

    /**
     * Design golden angle radial trajectory
     */
    designGoldenAngleRadial(numFrames, spokesPerFrame) {
        const goldenAngle = 111.246; // degrees
        const trajectory = {
            frames: [],
            goldenAngle: goldenAngle,
            spokesPerFrame: spokesPerFrame
        };

        for (let frame = 0; frame < numFrames; frame++) {
            const frameAngles = [];

            for (let spoke = 0; spoke < spokesPerFrame; spoke++) {
                const spokeIndex = frame * spokesPerFrame + spoke;
                const angle = (spokeIndex * goldenAngle) % 360;
                frameAngles.push(angle);
            }

            trajectory.frames.push({
                frameNumber: frame + 1,
                angles: frameAngles
            });
        }

        return trajectory;
    },

    /**
     * Calculate temporal sampling pattern
     */
    calculateTemporalSampling(numFrames, temporalResolution, strategy = 'uniform') {
        const sampling = {
            strategy: strategy,
            frames: []
        };

        if (strategy === 'uniform') {
            // Uniform temporal sampling
            for (let i = 0; i < numFrames; i++) {
                sampling.frames.push({
                    frame: i + 1,
                    time: i * temporalResolution,
                    weight: 1.0
                });
            }
        } else if (strategy === 'exponential') {
            // Exponential sampling (more frequent early)
            let time = 0;
            for (let i = 0; i < numFrames; i++) {
                const dt = temporalResolution * Math.exp(i / numFrames);
                sampling.frames.push({
                    frame: i + 1,
                    time: time,
                    weight: 1.0
                });
                time += dt;
            }
        }

        return sampling;
    },

    /**
     * Simulate kinetic model
     */
    simulateKinetics(model, parameters, timePoints) {
        if (model === 'two-site') {
            return this.twoSiteExchange(parameters, timePoints);
        } else if (model === 'three-site') {
            return this.threeSiteExchange(parameters, timePoints);
        } else if (model === 'perfusion') {
            return this.perfusionModel(parameters, timePoints);
        }

        return null;
    },

    /**
     * Two-site exchange model (e.g., pyruvate <-> lactate)
     */
    twoSiteExchange(params, timePoints) {
        // Parameters: kPL (forward rate), kLP (reverse rate), T1_pyr, T1_lac
        const kPL = params.kPL || 0.02; // s^-1
        const kLP = params.kLP || 0.0;  // s^-1 (often negligible)
        const t1_pyr = params.t1_pyr || 43; // s
        const t1_lac = params.t1_lac || 33; // s

        const pyruvate = [];
        const lactate = [];

        // Initial conditions
        let pyr = 1.0;
        let lac = 0.0;

        for (let i = 0; i < timePoints.length; i++) {
            const t = timePoints[i];
            const dt = i > 0 ? timePoints[i] - timePoints[i - 1] : timePoints[0];

            // Differential equations
            const dpyr = -(1 / t1_pyr + kPL) * pyr + kLP * lac;
            const dlac = kPL * pyr - (1 / t1_lac + kLP) * lac;

            pyr += dpyr * dt;
            lac += dlac * dt;

            pyruvate.push(Math.max(0, pyr));
            lactate.push(Math.max(0, lac));
        }

        return {
            pyruvate: pyruvate,
            lactate: lactate,
            parameters: params
        };
    },

    /**
     * Three-site exchange model
     */
    threeSiteExchange(params, timePoints) {
        // Simplified three-site model
        const result = this.twoSiteExchange(params, timePoints);

        // Add bicarbonate pathway
        const bicarbonate = timePoints.map(t =>
            0.1 * Math.exp(-t / 15) // Simplified
        );

        result.bicarbonate = bicarbonate;
        return result;
    },

    /**
     * Perfusion model
     */
    perfusionModel(params, timePoints) {
        const ktrans = params.ktrans || 0.1; // min^-1
        const ve = params.ve || 0.2; // extravascular fraction

        const tissue = timePoints.map(t => {
            // Parker AIF convolved with exponential
            const aif = Math.exp(-t / 10);
            const concentration = ktrans * aif * (1 - Math.exp(-t * ktrans / ve));
            return concentration;
        });

        return {
            tissue: tissue,
            parameters: params
        };
    },

    /**
     * Export dynamic sequence
     */
    exportForPyPulseq(sequence) {
        const code = `# Dynamic Time-Resolved Sequence
# Generated by Hyperpolarized Sequence Generator

import numpy as np
from pypulseq import Sequence

# Sequence parameters
num_frames = ${sequence.numFrames}
temporal_resolution = ${sequence.temporalResolution}  # seconds
tr = ${sequence.tr}  # ms
readout_type = "${sequence.readoutType}"

# Flip angle schedule
flip_angles = np.array([${sequence.flipAngles.slice(0, 10).map(a => a.toFixed(2)).join(', ')}...])  # degrees

# Frame timing
frame_times = np.array([${sequence.frames.slice(0, 10).map(f => f.startTime.toFixed(2)).join(', ')}...])  # seconds

# Build dynamic sequence
seq = Sequence()

for frame in range(num_frames):
    flip_angle_rad = np.deg2rad(flip_angles[frame])
    
    # Add RF pulse with VFA
    # Add readout gradient/trajectory
    # Add delays
    
    # See specific readout module for details

# Export sequence
seq.write('dynamic_hyperpolarized.seq')
`;

        return code;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DynamicSequence;
}
