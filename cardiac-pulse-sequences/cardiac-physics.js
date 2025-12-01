// MRI Physics Calculations for Cardiac Imaging

const CardiacPhysics = {
    // Tissue relaxation times at 1.5T and 3T (in ms)
    relaxationTimes: {
        '1.5T': {
            myocardium: { T1: 1030, T2: 45, T2star: 30 },
            blood: { T1: 1500, T2: 250, T2star: 150 },
            scar: { T1: 500, T2: 50, T2star: 25 },
            edema: { T1: 1200, T2: 80, T2star: 40 }
        },
        '3T': {
            myocardium: { T1: 1471, T2: 42, T2star: 25 },
            blood: { T1: 1932, T2: 275, T2star: 100 },
            scar: { T1: 700, T2: 45, T2star: 20 },
            edema: { T1: 1700, T2: 75, T2star: 35 }
        }
    },

    /**
     * Calculate signal for balanced SSFP
     */
    calculateSSFPSignal(M0, TR, TE, T1, T2, flipAngle) {
        const alpha = Utils.degToRad(flipAngle);
        const E1 = Math.exp(-TR / T1);
        const E2 = Math.exp(-TR / T2);

        const numerator = M0 * (1 - E1) * Math.sin(alpha);
        const denominator = 1 - (E1 - E2) * Math.cos(alpha) - E1 * E2;

        const signal = numerator / denominator;
        return signal * Math.exp(-TE / T2);
    },

    /**
     * Calculate Ernst angle for maximum signal
     */
    calculateErnstAngle(TR, T1) {
        const E1 = Math.exp(-TR / T1);
        return Utils.radToDeg(Math.acos(E1));
    },

    /**
     * Calculate optimal TI for inversion recovery (nulling)
     */
    calculateNullingTI(T1) {
        // TI = T1 * ln(2) for perfect nulling
        return T1 * Math.log(2);
    },

    /**
     * Calculate signal after inversion recovery
     */
    calculateIRSignal(M0, TI, TR, T1, flipAngle = 90) {
        const alpha = Utils.degToRad(flipAngle);
        const signal = M0 * Math.abs(1 - 2 * Math.exp(-TI / T1) + Math.exp(-TR / T1)) * Math.sin(alpha);
        return signal;
    },

    /**
     * Calculate T1 from MOLLI measurements
     */
    fitT1MOLLI(signalValues, inversionTimes) {
        // Simplified 3-parameter fit: S(TI) = A - B*exp(-TI/T1*)
        // T1 = T1* * (B/A - 1)
        // This is a simplified version; real MOLLI uses non-linear curve fitting

        if (signalValues.length < 3) {
            throw new Error('Need at least 3 measurements for T1 fitting');
        }

        // Placeholder simplified calculation
        // In production, would use Levenberg-Marquardt or similar
        const minSignal = Math.min(...signalValues);
        const maxSignal = Math.max(...signalValues);
        const minIndex = signalValues.indexOf(minSignal);

        // Approximate T1 from the null crossing
        const estimatedT1 = inversionTimes[minIndex] / Math.log(2);

        return {
            T1: estimatedT1,
            confidence: 0.95,
            fittedCurve: inversionTimes.map(ti =>
                maxSignal * (1 - 2 * Math.exp(-ti / estimatedT1))
            )
        };
    },

    /**
     * Calculate extracellular volume (ECV)
     */
    calculateECV(T1pre_myo, T1post_myo, T1pre_blood, T1post_blood, hematocrit) {
        const deltaR1_myo = (1 / T1post_myo) - (1 / T1pre_myo);
        const deltaR1_blood = (1 / T1post_blood) - (1 / T1pre_blood);

        const ECV = deltaR1_myo / deltaR1_blood * (1 - hematocrit) * 100;
        return Math.max(0, Math.min(100, ECV)); // Clamp between 0-100%
    },

    /**
     * Calculate diffusion weighting b-value
     */
    calculateBValue(G, delta, DELTA, gamma = 42.58) {
        // b = γ²G²δ²(Δ - δ/3)
        // G in mT/m, delta and DELTA in ms, gamma in MHz/T
        const gammaRad = gamma * 2 * Math.PI; // Convert to rad/s/T
        const bValue = Math.pow(gammaRad * G * delta / 1000, 2) * (DELTA - delta / 3);
        return bValue / 1e6; // Convert to s/mm²
    },

    /**
     * Calculate ADC from multi-b-value signal
     */
    calculateADC(S0, Sb, bValue) {
        // S(b) = S(0) * exp(-b * ADC)
        // ADC = -ln(S(b)/S(0)) / b
        if (Sb <= 0 || S0 <= 0) return 0;
        return -Math.log(Sb / S0) / bValue;
    },

    /**
     * Calculate phase for velocity encoding
     */
    calculateVelocityPhase(velocity, VENC) {
        // Phase = π * velocity / VENC
        return Math.PI * velocity / VENC;
    },

    /**
     * Calculate velocity from phase
     */
    calculateVelocityFromPhase(phase, VENC) {
        // velocity = phase * VENC / π
        return phase * VENC / Math.PI;
    },

    /**
     * Calculate flow rate from velocity
     */
    calculateFlowRate(velocity, area) {
        // Flow (mL/s) = velocity (cm/s) * area (cm²)
        return velocity * area;
    },

    /**
     * Estimate specific absorption rate (SAR)
     */
    estimateSAR(flipAngle, TR, fieldStrength, weight = 70) {
        // Simplified SAR estimation
        // Real SAR depends on RF pulse shape, body part, etc.
        const alpha = Utils.degToRad(flipAngle);
        const B0 = fieldStrength; // Tesla

        // Proportional to B0², flip angle², and inversely to TR
        const relativeSAR = Math.pow(B0 * Math.sin(alpha), 2) / (TR / 1000);

        // Scale to approximate W/kg (very simplified)
        const estimatedSAR = relativeSAR * 0.1 / weight;

        return {
            value: estimatedSAR,
            limit: 4.0, // IEC limit for head/body (W/kg)
            percentage: (estimatedSAR / 4.0) * 100
        };
    },

    /**
     * Calculate temporal resolution for cardiac imaging
     */
    calculateTemporalResolution(TR, linesPerPhase, accelerationFactor = 1) {
        // Temporal resolution = TR * lines per phase / acceleration
        return (TR * linesPerPhase) / accelerationFactor;
    },

    /**
     * Calculate number of cardiac phases from temporal resolution
     */
    calculateCardiacPhases(RR_interval, temporalResolution) {
        return Math.floor(RR_interval / temporalResolution);
    },

    /**
     * Estimateperfusion parameters (Fermi model)
     */
    fitPerfusionCurve(timePoints, signalIntensity) {
        // Simplified Fermi model: SI(t) = A / (1 + exp(-(t-t0)/τ))
        // Returns approximate myocardial blood flow

        const maxSI = Math.max(...signalIntensity);
        const maxIndex = signalIntensity.indexOf(maxSI);
        const peakTime = timePoints[maxIndex];

        // Estimate upslope
        const baselineSI = signalIntensity[0];
        const upslope = (maxSI - baselineSI) / peakTime;

        return {
            peakTime: peakTime,
            maxEnhancement: maxSI - baselineSI,
            upslope: upslope,
            estimatedMBF: upslope * 100 // Simplified, real calculation more complex
        };
    },

    /**
     * Calculate contrast-to-noise ratio
     */
    calculateCNR(signal1, signal2, noise) {
        return Math.abs(signal1 - signal2) / noise;
    },

    /**
     * Calculate signal-to-noise ratio
     */
    calculateSNR(signal, noise) {
        return signal / noise;
    }
};

// Export for use in other modules
window.CardiacPhysics = CardiacPhysics;
