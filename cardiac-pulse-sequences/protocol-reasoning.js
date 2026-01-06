// Protocol Reasoning Engine
const ProtocolReasoning = {
    init() {
        console.log('Protocol Reasoning module initialized');
    },

    /**
     * Analyze protocol for issues
     */
    analyzeProtocol(config) {
        const warnings = [];
        const suggestions = [];

        // Check CINE temporal resolution
        if (config.cine && config.cine.temporalRes > 45) {
            warnings.push("Temporal resolution > 45ms may blur diastolic dysfunction.");
            suggestions.push("Increase parallel imaging factor to reduce temporal resolution.");
        }

        // Check SAR limits for SSFP at 3T
        if (config.fieldStrength === 3 && config.cine.type === 'bssfp') {
            warnings.push("Potential SAR issues with bSSFP at 3T.");
            suggestions.push("Ensure TR is minimized and flip angle is optimized.");
        }

        // Check Parallel Imaging
        if (config.parallelImaging.accelerationFactor > 4) {
            warnings.push("High acceleration factor (R>4) risks severe SNR penalty.");
            suggestions.push("Consider using Quantum Optimization or reducing R.");
        }

        return {
            status: warnings.length > 0 ? 'warning' : 'optimal',
            warnings: warnings,
            suggestions: suggestions
        };
    },

    /**
     * Suggest protocol based on indication
     */
    suggestProtocol(indication) {
        const lowerIndication = indication.toLowerCase();

        if (lowerIndication.includes('myocarditis')) {
            return {
                sequences: ['T2-weighted STIR', 'Early Gadolinium Enhancement', 'LGE', 'T1 Mapping', 'T2 Mapping'],
                focus: 'Tissue Characterization',
                notes: 'Lake Louise Criteria require T1 and T2 mapping.'
            };
        }

        if (lowerIndication.includes('ischemia') || lowerIndication.includes('infarct')) {
            return {
                sequences: ['CINE Stress/Rest', 'Perfusion', 'LGE'],
                focus: 'Wall Motion & Perfusion',
                notes: 'Focus on stress perfusion and scar viability.'
            };
        }

        return {
            sequences: ['Standard CINE', 'LGE'],
            focus: 'General Structure & Function',
            notes: 'Standard diagnostic protocol.'
        };
    }
};

window.ProtocolReasoning = ProtocolReasoning;
