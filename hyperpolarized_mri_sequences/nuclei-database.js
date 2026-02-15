// Nuclei Database for Hyperpolarized Imaging
// Contains gyromagnetic ratios, relaxation times, and metabolite information

const NucleiDatabase = {
    // Carbon-13 Metabolites
    'c13-pyruvate': {
        name: 'Hyperpolarized ¹³C Pyruvate',
        nucleus: '13C',
        gamma: 10.7084, // MHz/T
        t1: 43, // seconds at 3T, in vivo
        t2: 1.0, // seconds
        chemicalShift: 171.0, // ppm
        b0Field: 3.0, // Tesla
        metabolites: {
            pyruvate: { shift: 171.0, name: 'Pyruvate [1-13C]' },
            lactate: { shift: 183.3, name: 'Lactate [1-13C]' },
            alanine: { shift: 176.5, name: 'Alanine [1-13C]' },
            bicarbonate: { shift: 161.0, name: 'Bicarbonate' }
        }
    },
    'c13-lactate': {
        name: 'Hyperpolarized ¹³C Lactate',
        nucleus: '13C',
        gamma: 10.7084,
        t1: 33, // seconds at 3T
        t2: 0.8,
        chemicalShift: 183.3,
        b0Field: 3.0
    },
    'c13-bicarbonate': {
        name: 'Hyperpolarized ¹³C Bicarbonate',
        nucleus: '13C',
        gamma: 10.7084,
        t1: 15, // seconds
        t2: 0.5,
        chemicalShift: 161.0,
        b0Field: 3.0
    },
    'c13-alanine': {
        name: 'Hyperpolarized ¹³C Alanine',
        nucleus: '13C',
        gamma: 10.7084,
        t1: 30,
        t2: 0.7,
        chemicalShift: 176.5,
        b0Field: 3.0
    },

    // Xenon-129
    'xe129-gas': {
        name: 'Hyperpolarized ¹²⁹Xe Gas Phase',
        nucleus: '129Xe',
        gamma: -11.8604, // MHz/T
        t1: 20, // seconds (in lungs during breath hold)
        t2: 0.02, // seconds
        chemicalShift: 0, // ppm reference
        b0Field: 3.0
    },
    'xe129-dissolved': {
        name: 'Hyperpolarized ¹²⁹Xe Dissolved Phase',
        nucleus: '129Xe',
        gamma: -11.8604,
        t1: 5, // seconds (in tissue/blood)
        t2: 0.002, // seconds
        chemicalShift: 197, // ppm from gas
        b0Field: 3.0
    },

    // Helium-3
    'he3': {
        name: 'Hyperpolarized ³He',
        nucleus: '3He',
        gamma: -32.434, // MHz/T
        t1: 15, // seconds
        t2: 0.015,
        chemicalShift: 0,
        b0Field: 3.0
    },

    // Custom template
    'custom': {
        name: 'Custom Nucleus',
        nucleus: 'Custom',
        gamma: 1.0,
        t1: 10,
        t2: 1.0,
        chemicalShift: 0,
        b0Field: 3.0
    }
};

// Helper functions
const NucleiHelper = {
    /**
     * Get nuclei properties
     */
    getProperties(nucleiId) {
        return NucleiDatabase[nucleiId] || NucleiDatabase['custom'];
    },

    /**
     * Calculate Larmor frequency
     */
    getLarmorFrequency(nucleiId, b0Field) {
        const props = this.getProperties(nucleiId);
        return Math.abs(props.gamma * b0Field); // MHz
    },

    /**
     * Calculate chemical shift in Hz
     */
    getChemicalShiftHz(nucleiId, shiftPPM, b0Field) {
        const props = this.getProperties(nucleiId);
        const larmorFreq = this.getLarmorFrequency(nucleiId, b0Field);
        return shiftPPM * larmorFreq; // Hz
    },

    /**
     * Get all metabolites for a nucleus
     */
    getMetabolites(nucleiId) {
        const props = this.getProperties(nucleiId);
        return props.metabolites || {};
    },

    /**
     * Calculate signal decay over time
     */
    calculateDecay(initialSignal, t1, time) {
        return initialSignal * Math.exp(-time / t1);
    },

    /**
     * Calculate T2* decay
     */
    calculateT2Decay(initialSignal, t2, time) {
        return initialSignal * Math.exp(-time / t2);
    },

    /**
     * Format nucleus name for display
     */
    formatNucleusName(nucleus) {
        const superscripts = {
            '13': '¹³',
            '129': '¹²⁹',
            '3': '³',
            '15': '¹⁵'
        };

        for (const [normal, super_] of Object.entries(superscripts)) {
            nucleus = nucleus.replace(normal, super_);
        }
        return nucleus;
    },

    /**
     * Get recommended imaging parameters
     */
    getRecommendedParams(nucleiId) {
        const props = this.getProperties(nucleiId);
        const recommendations = {
            'c13-pyruvate': {
                vfaFrames: 20,
                temporalResolution: 2, // seconds
                acquisitionWindow: 60, // seconds
                minTR: 100 // ms
            },
            'c13-lactate': {
                vfaFrames: 15,
                temporalResolution: 3,
                acquisitionWindow: 45,
                minTR: 100
            },
            'xe129-gas': {
                vfaFrames: 10,
                temporalResolution: 0.5,
                acquisitionWindow: 15,
                minTR: 20
            },
            'xe129-dissolved': {
                vfaFrames: 8,
                temporalResolution: 0.3,
                acquisitionWindow: 10,
                minTR: 15
            },
            'he3': {
                vfaFrames: 10,
                temporalResolution: 0.5,
                acquisitionWindow: 15,
                minTR: 20
            }
        };

        return recommendations[nucleiId] || {
            vfaFrames: 15,
            temporalResolution: 2,
            acquisitionWindow: 30,
            minTR: 100
        };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NucleiDatabase, NucleiHelper };
}
