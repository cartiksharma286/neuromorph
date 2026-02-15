// Spiral Trajectory Designer
// Generates Archimedean spiral k-space trajectories for fast imaging

const SpiralDesigner = {
    /**
     * Design uniform density Archimedean spiral
     */
    designUniformSpiral(fov, resolution, interleaves, maxGrad, maxSlew, gamma = 42.58) {
        // Convert units
        const fovCm = fov; // cm
        const resMm = resolution; // mm
        const gmax = maxGrad * 0.001; // Convert mT/m to T/m
        const smax = maxSlew; // T/m/s
        const gammaHz = gamma * 1e6; // Hz/T

        // Calculate k-space parameters
        const kmax = 1 / (2 * resMm * 0.1); // cm^-1
        const npoints = Math.ceil(Math.PI * kmax * kmax * fovCm * fovCm / interleaves);

        // Time step based on gradient constraints
        const dt = 4e-6; // 4 microseconds

        // Generate trajectory
        const trajectory = {
            kx: [],
            ky: [],
            gx: [],
            gy: [],
            time: [],
            interleaves: interleaves,
            kmax: kmax
        };

        let k = 0;
        let theta = 0;
        let t = 0;

        // Initial gradient direction
        let gx = 0;
        let gy = 0;

        while (k < kmax) {
            // Current k-space position
            const kx_val = k * Math.cos(theta);
            const ky_val = k * Math.sin(theta);

            trajectory.kx.push(kx_val);
            trajectory.ky.push(ky_val);
            trajectory.time.push(t);

            // Calculate required gradients
            // For Archimedean spiral: k(t) = (1/FOV) * (t/T_readout)
            // dk/dt determines gradient amplitude

            const dk = Math.sqrt(1 / (Math.PI * fovCm * fovCm * interleaves));
            const dtheta = dk / Math.max(k, 1e-10);

            // Gradient calculation
            const gx_new = dk * Math.cos(theta) / (gammaHz * 2 * Math.PI * dt);
            const gy_new = dk * Math.sin(theta) / (gammaHz * 2 * Math.PI * dt);

            // Apply slew rate constraint
            const dgx = (gx_new - gx) / dt;
            const dgy = (gy_new - gy) / dt;
            const slew = Math.sqrt(dgx * dgx + dgy * dgy);

            if (slew > smax) {
                // Limit by slew rate
                const scale = smax / slew;
                gx += dgx * scale * dt;
                gy += dgy * scale * dt;
            } else {
                gx = gx_new;
                gy = gy_new;
            }

            // Apply gradient amplitude constraint
            const gmag = Math.sqrt(gx * gx + gy * gy);
            if (gmag > gmax) {
                const gscale = gmax / gmag;
                gx *= gscale;
                gy *= gscale;
            }

            trajectory.gx.push(gx * 1000); // Convert back to mT/m
            trajectory.gy.push(gy * 1000);

            // Update k-space position based on actual gradients
            k += dk;
            theta += dtheta;
            t += dt;
        }

        trajectory.readoutDuration = t * 1000; // Convert to ms
        trajectory.numPoints = trajectory.kx.length;

        return trajectory;
    },

    /**
     * Design variable density spiral
     */
    designVariableDensitySpiral(fov, resolution, interleaves, maxGrad, maxSlew, gamma = 42.58, undersamplingFactor = 2) {
        // Start with uniform spiral
        const uniform = this.designUniformSpiral(fov, resolution, interleaves, maxGrad, maxSlew, gamma);

        // Modify for variable density (higher density at center)
        const vd = {
            kx: [],
            ky: [],
            gx: [],
            gy: [],
            time: [],
            interleaves: interleaves,
            kmax: uniform.kmax
        };

        for (let i = 0; i < uniform.kx.length; i++) {
            const k = Math.sqrt(uniform.kx[i] ** 2 + uniform.ky[i] ** 2);
            const knorm = k / uniform.kmax;

            // Variable density function: more points at center
            const density = 1 + (undersamplingFactor - 1) * knorm;

            // Subsample based on density
            if (i % Math.round(density) === 0) {
                vd.kx.push(uniform.kx[i]);
                vd.ky.push(uniform.ky[i]);
                vd.gx.push(uniform.gx[i]);
                vd.gy.push(uniform.gy[i]);
                vd.time.push(uniform.time[i]);
            }
        }

        vd.readoutDuration = vd.time[vd.time.length - 1] * 1000;
        vd.numPoints = vd.kx.length;

        return vd;
    },

    /**
     * Generate all interleaves
     */
    generateAllInterleaves(baseTrajectory, interleaves) {
        const allTrajectories = [];
        const angleStep = (2 * Math.PI) / interleaves;

        for (let i = 0; i < interleaves; i++) {
            const angle = i * angleStep;
            const cos_a = Math.cos(angle);
            const sin_a = Math.sin(angle);

            const trajectory = {
                kx: [],
                ky: [],
                gx: [],
                gy: [],
                interleave: i
            };

            for (let j = 0; j < baseTrajectory.kx.length; j++) {
                // Rotate trajectory
                const kx_rot = baseTrajectory.kx[j] * cos_a - baseTrajectory.ky[j] * sin_a;
                const ky_rot = baseTrajectory.kx[j] * sin_a + baseTrajectory.ky[j] * cos_a;

                const gx_rot = baseTrajectory.gx[j] * cos_a - baseTrajectory.gy[j] * sin_a;
                const gy_rot = baseTrajectory.gx[j] * sin_a + baseTrajectory.gy[j] * cos_a;

                trajectory.kx.push(kx_rot);
                trajectory.ky.push(ky_rot);
                trajectory.gx.push(gx_rot);
                trajectory.gy.push(gy_rot);
            }

            allTrajectories.push(trajectory);
        }

        return allTrajectories;
    },

    /**
     * Calculate sequence metrics
     */
    calculateMetrics(trajectory, tr, interleaves) {
        return {
            readoutDuration: trajectory.readoutDuration.toFixed(2) + ' ms',
            totalScanTime: ((tr * interleaves) / 1000).toFixed(2) + ' s',
            kmaxValue: trajectory.kmax.toFixed(3) + ' cm⁻¹',
            numPoints: trajectory.numPoints,
            bandwidthPerPoint: (1000 / (trajectory.time[1] - trajectory.time[0]) / 1000).toFixed(1) + ' kHz'
        };
    },

    /**
     * Export trajectory for PyPulseq
     */
    exportForPyPulseq(trajectory, interleaves) {
        const code = `# Spiral Trajectory - Generated by Hyperpolarized Sequence Generator
import numpy as np

# Trajectory parameters
num_interleaves = ${interleaves}
readout_duration = ${trajectory.readoutDuration} # ms
kmax = ${trajectory.kmax} # cm^-1

# Base interleave k-space trajectory
kx = np.array([${trajectory.kx.slice(0, 20).map(k => k.toFixed(6)).join(', ')}...])  # ${trajectory.kx.length} points total
ky = np.array([${trajectory.ky.slice(0, 20).map(k => k.toFixed(6)).join(', ')}...])  # ${trajectory.ky.length} points total

# Gradient waveforms (mT/m)
gx = np.array([${trajectory.gx.slice(0, 20).map(g => g.toFixed(3)).join(', ')}...])
gy = np.array([${trajectory.gy.slice(0, 20).map(g => g.toFixed(3)).join(', ')}...])

# Time vector (seconds)
time = np.array([${trajectory.time.slice(0, 20).map(t => t.toFixed(6)).join(', ')}...])

# Generate all interleaves by rotating base trajectory
def generate_interleaves(kx, ky, gx, gy, num_interleaves):
    all_kx, all_ky = [], []
    all_gx, all_gy = [], []
    
    for i in range(num_interleaves):
        angle = 2 * np.pi * i / num_interleaves
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        kx_rot = kx * cos_a - ky * sin_a
        ky_rot = kx * sin_a + ky * cos_a
        
        gx_rot = gx * cos_a - gy * sin_a
        gy_rot = gx * sin_a + gy * cos_a
        
        all_kx.append(kx_rot)
        all_ky.append(ky_rot)
        all_gx.append(gx_rot)
        all_gy.append(gy_rot)
    
    return all_kx, all_ky, all_gx, all_gy

# Use with PyPulseq
# See pypulseq_generator.py for complete sequence implementation
`;

        return code;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SpiralDesigner;
}
