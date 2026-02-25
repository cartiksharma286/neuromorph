// Parallel Imaging Physics and Reconstruction Calculations

const ParallelPhysics = {
    /**
     * Calculate g-factor for SENSE reconstruction
     * Simplified geometric model based on coil geometry and acceleration
     */
    calculateGFactor(accelerationFactor, coilElements, coilGeometry = 'circular') {
        // G-factor depends on coil arrangement and acceleration
        // This is a simplified analytical model

        // Minimum coil elements needed for acceleration R
        const minCoils = accelerationFactor * 2;

        if (coilElements < minCoils) {
            return {
                gFactor: Infinity,
                error: `Need at least ${minCoils} coil elements for R=${accelerationFactor}`
            };
        }

        // Simplified g-factor calculation
        // Real g-factor depends on coil sensitivity profiles and noise correlation
        const coilRatio = coilElements / accelerationFactor;

        let baseGFactor;
        switch (coilGeometry) {
            case 'circular':
                // Better geometry factor for circular arrays
                baseGFactor = 1 + (accelerationFactor - 1) * 0.15;
                break;
            case 'linear':
                // Higher g-factor for linear arrays
                baseGFactor = 1 + (accelerationFactor - 1) * 0.25;
                break;
            default:
                baseGFactor = 1 + (accelerationFactor - 1) * 0.2;
        }

        // Account for coil redundancy
        const redundancyFactor = Math.max(1, 1.5 - (coilRatio - 2) * 0.1);
        const gFactor = baseGFactor * redundancyFactor;

        // Spatial variation (simplified)
        const gFactorCenter = gFactor;
        const gFactorPeripheral = gFactor * 1.3;

        return {
            gFactorMean: parseFloat(gFactor.toFixed(3)),
            gFactorCenter: parseFloat(gFactorCenter.toFixed(3)),
            gFactorPeripheral: parseFloat(gFactorPeripheral.toFixed(3)),
            gFactorMax: parseFloat((gFactorPeripheral * 1.2).toFixed(3)),
            isValid: coilElements >= minCoils
        };
    },

    /**
     * Calculate SNR penalty from parallel imaging
     */
    calculateSNRPenalty(gFactor, accelerationFactor) {
        // SNR_parallel = SNR_full / (sqrt(R) * g)
        const snrRatio = 1 / (Math.sqrt(accelerationFactor) * gFactor);
        const snrPenalty = (1 - snrRatio) * 100; // as percentage

        return {
            snrRatio: parseFloat(snrRatio.toFixed(3)),
            snrPenaltyPercent: parseFloat(snrPenalty.toFixed(1)),
            effectiveSNR: (snrRatio * 100).toFixed(1)
        };
    },

    /**
     * Calculate effective acceleration for GRAPPA with ACS lines
     */
    calculateEffectiveAcceleration(nominalR, kSpaceLines, acsLines) {
        // Effective R accounts for overhead of ACS lines
        const totalAcquiredLines = (kSpaceLines / nominalR) + acsLines;
        const effectiveR = kSpaceLines / totalAcquiredLines;

        return {
            nominalR: nominalR,
            effectiveR: parseFloat(effectiveR.toFixed(2)),
            totalLines: Math.ceil(totalAcquiredLines),
            speedup: parseFloat((kSpaceLines / totalAcquiredLines).toFixed(2))
        };
    },

    /**
     * Optimize ACS lines for GRAPPA
     */
    optimizeACSLines(accelerationFactor, kSpaceLines) {
        // Rule of thumb: ACS lines ≈ 24-32 for standard imaging
        // Need at least 3-4 times the kernel size

        const kernelSize = accelerationFactor * 2;
        const minACS = kernelSize * 4;
        const recommendedACS = Math.min(32, Math.max(minACS, 24));

        return {
            recommendedACS: recommendedACS,
            minACS: minACS,
            kernelSize: kernelSize,
            percentageOfKSpace: parseFloat((recommendedACS / kSpaceLines * 100).toFixed(1))
        };
    },

    /**
     * Calculate compressed sensing undersampling pattern
     */
    generateCSPattern(matrixSize, accelerationFactor, densityType = 'variable') {
        // Generate incoherent undersampling pattern for compressed sensing
        const totalSamples = matrixSize;
        const sampledLines = Math.ceil(totalSamples / accelerationFactor);

        const pattern = new Array(matrixSize).fill(0);

        if (densityType === 'variable') {
            // Variable density: more sampling in k-space center
            const centerWidth = Math.floor(matrixSize * 0.08); // 8% fully sampled center

            // Fully sample center
            for (let i = Math.floor(matrixSize / 2) - centerWidth;
                i < Math.floor(matrixSize / 2) + centerWidth; i++) {
                pattern[i] = 1;
            }

            // Randomly sample periphery with decreasing probability
            let samplesNeeded = sampledLines - (centerWidth * 2);
            let attempts = 0;
            const maxAttempts = matrixSize * 10;

            while (samplesNeeded > 0 && attempts < maxAttempts) {
                const idx = Math.floor(Math.random() * matrixSize);
                const distFromCenter = Math.abs(idx - matrixSize / 2) / (matrixSize / 2);
                const probability = 1 - Math.pow(distFromCenter, 0.5); // Higher prob at center

                if (pattern[idx] === 0 && Math.random() < probability) {
                    pattern[idx] = 1;
                    samplesNeeded--;
                }
                attempts++;
            }
        } else {
            // Uniform random sampling
            let samplesNeeded = sampledLines;
            while (samplesNeeded > 0) {
                const idx = Math.floor(Math.random() * matrixSize);
                if (pattern[idx] === 0) {
                    pattern[idx] = 1;
                    samplesNeeded--;
                }
            }
        }

        const actualSamples = pattern.reduce((a, b) => a + b, 0);
        const actualR = matrixSize / actualSamples;

        return {
            pattern: pattern,
            actualR: parseFloat(actualR.toFixed(2)),
            sampledLines: actualSamples,
            incoherence: this.calculateIncoherence(pattern)
        };
    },

    /**
     * Calculate incoherence metric for CS pattern
     */
    calculateIncoherence(pattern) {
        // Simplified incoherence measure
        // Real incoherence depends on transform domain (wavelets, etc.)
        let maxGap = 0;
        let currentGap = 0;

        for (let i = 0; i < pattern.length; i++) {
            if (pattern[i] === 0) {
                currentGap++;
            } else {
                maxGap = Math.max(maxGap, currentGap);
                currentGap = 0;
            }
        }

        const avgGap = (pattern.length - pattern.reduce((a, b) => a + b, 0)) /
            (pattern.reduce((a, b) => a + b, 0) || 1);

        return {
            maxGap: maxGap,
            averageGap: parseFloat(avgGap.toFixed(2)),
            coherence: parseFloat((1 - maxGap / pattern.length).toFixed(3))
        };
    },

    /**
     * Calculate SMS (Simultaneous Multi-Slice) parameters
     */
    calculateSMSParameters(numSlices, multiband, sliceGap = 0) {
        const sliceGroups = Math.ceil(numSlices / multiband);
        const totalExcitations = sliceGroups;
        const accelerationFactor = multiband;

        // CAIPIRINHA FOV shift for each band
        const fovShift = 1 / multiband;

        // Estimate leakage based on slice separation
        const sliceThickness = 8; // mm, typical
        const sliceSeparation = (sliceThickness + sliceGap) * multiband;
        const leakageFactor = sliceThickness / sliceSeparation;
        const estimatedLeakage = Math.min(25, leakageFactor * 30); // percentage

        return {
            multibandFactor: multiband,
            sliceGroups: sliceGroups,
            accelerationFactor: accelerationFactor,
            fovShift: fovShift,
            sliceSeparation: parseFloat(sliceSeparation.toFixed(1)),
            estimatedLeakage: parseFloat(estimatedLeakage.toFixed(1)),
            recommendations: {
                minSliceGap: sliceThickness * 0.2,
                optimalSliceGap: sliceThickness * 0.5
            }
        };
    },

    /**
     * Calculate combined SENSE + Compressed Sensing parameters
     */
    calculateHybridAcceleration(senseR, csR) {
        const totalR = senseR * csR;

        // SNR penalty is multiplicative but not quite sqrt(R1*R2) due to synergy
        const senseGFactor = 1 + (senseR - 1) * 0.15;
        const csPenalty = Math.sqrt(csR * 0.8); // CS has inherent denoising
        const effectiveSNRPenalty = senseGFactor * csPenalty * Math.sqrt(senseR);

        return {
            totalAcceleration: parseFloat(totalR.toFixed(2)),
            senseComponent: senseR,
            csComponent: csR,
            effectiveSNRPenalty: parseFloat(effectiveSNRPenalty.toFixed(2)),
            recommendedIterations: Math.ceil(csR * 10),
            estimatedReconTime: csR * 2 // seconds, simplified
        };
    },

    /**
     * Estimate scan time reduction
     */
    calculateScanTimeReduction(baselineScanTime, accelerationFactor, overhead = 0) {
        const acceleratedTime = (baselineScanTime / accelerationFactor) + overhead;
        const timeSaved = baselineScanTime - acceleratedTime;
        const percentReduction = (timeSaved / baselineScanTime) * 100;

        return {
            baselineTime: Utils.formatTime(baselineScanTime),
            acceleratedTime: Utils.formatTime(acceleratedTime),
            timeSaved: Utils.formatTime(timeSaved),
            percentReduction: parseFloat(percentReduction.toFixed(1)),
            accelerationFactor: accelerationFactor
        };
    },

    /**
     * Generate aliasing pattern for undersampled k-space
     */
    simulateAliasing(accelerationFactor, fov) {
        // In undersampled images, aliasing occurs with FOV/R spacing
        const aliasFOV = fov / accelerationFactor;
        const numGhosts = Math.floor(accelerationFactor) - 1;

        return {
            aliasFOV: aliasFOV,
            numGhosts: numGhosts,
            ghostSpacing: aliasFOV,
            description: `${numGhosts} ghost(s) at ${aliasFOV.toFixed(0)}mm intervals`
        };
    },

    /**
     * Calculate noise correlation matrix influence
     */
    estimateNoiseCorrelation(coilElements, coilSpacing) {
        // Noise correlation increases with proximity
        // Simplified model: correlation ~ exp(-distance)

        const avgCorrelation = Math.exp(-coilSpacing / 30); // Decay constant ~30mm
        const maxCorrelation = Math.min(0.5, avgCorrelation * 1.3);

        return {
            averageCorrelation: parseFloat(avgCorrelation.toFixed(3)),
            maxCorrelation: parseFloat(maxCorrelation.toFixed(3)),
            impact: maxCorrelation > 0.3 ? 'significant' : 'moderate',
            recommendation: maxCorrelation > 0.3 ?
                'Consider noise decorrelation in reconstruction' :
                'Noise correlation within acceptable limits'
        };
    }
};

// Export for use in other modules
window.ParallelPhysics = ParallelPhysics;
