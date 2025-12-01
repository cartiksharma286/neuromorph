// Example Sequences
// Pre-configured sequences for common hyperpolarized imaging applications

const ExampleSequences = {
    /**
     * C-13 Pyruvate Dynamic Imaging
     */
    c13PyruvateDynamic: {
        name: 'C-13 Pyruvate Dynamic Imaging',
        type: 'dynamic',
        description: 'Time-resolved imaging of pyruvate-to-lactate conversion',
        nucleus: 'c13-pyruvate',
        parameters: {
            numFrames: 20,
            temporalResolution: 2, // seconds
            readoutType: 'spiral',
            fov: 24, // cm
            sliceThickness: 10, // mm
            tr: 100, // ms
            vfaEnabled: true
        }
    },

    /**
     * Xe-129 Lung Ventilation
     */
    xe129Ventilation: {
        name: 'Xe-129 Lung Ventilation Imaging',
        type: 'dynamic',
        description: 'Single breath-hold ventilation imaging with Xe-129',
        nucleus: 'xe129-gas',
        parameters: {
            numFrames: 10,
            temporalResolution: 0.5,
            readoutType: 'spiral',
            fov: 40,
            sliceThickness: 15,
            tr: 20,
            vfaEnabled: true
        }
    },

    /**
     * C-13 Metabolic Imaging
     */
    c13Metabolic: {
        name: 'C-13 Multi-Metabolite Imaging',
        type: 'metabolic',
        description: 'Spectral-spatial excitation for pyruvate, lactate, and bicarbonate',
        nucleus: 'c13-pyruvate',
        parameters: {
            metabolites: ['pyruvate', 'lactate', 'bicarbonate'],
            b0Field: 3.0,
            sliceThickness: 10,
            pulseType: 'spectral-spatial',
            fov: 24
        }
    },

    /**
     * High-Resolution Spiral
     */
    highResSpiral: {
        name: 'High-Resolution Spiral Imaging',
        type: 'spiral',
        description: 'High spatial resolution with variable density spiral',
        nucleus: 'c13-pyruvate',
        parameters: {
            fov: 24,
            resolution: 2, // mm
            interleaves: 16,
            maxGrad: 40,
            maxSlew: 150,
            spiralType: 'variable'
        }
    },

    /**
     * Fast EPI
     */
    fastEPI: {
        name: 'Fast Single-Shot EPI',
        type: 'epi',
        description: 'Rapid snapshot imaging with EPI',
        nucleus: 'c13-pyruvate',
        parameters: {
            matrixSize: 64,
            fov: 24,
            shots: 1,
            partialFourier: 0.75,
            trajectory: 'blipped'
        }
    },

    /**
     * Get example by name
     */
    getExample(name) {
        return this[name] || null;
    },

    /**
     * Get all examples
     */
    getAllExamples() {
        return {
            'c13PyruvateDynamic': this.c13PyruvateDynamic,
            'xe129Ventilation': this.xe129Ventilation,
            'c13Metabolic': this.c13Metabolic,
            'highResSpiral': this.highResSpiral,
            'fastEPI': this.fastEPI
        };
    },

    /**
     * Load example into application
     */
    loadExample(exampleName, app) {
        const example = this.getExample(exampleName);

        if (!example) {
            console.error('Example not found:', exampleName);
            return false;
        }

        // Load parameters based on type
        console.log('Loading example:', example.name);

        // Set nucleus
        if (example.nucleus) {
            const nucleiSelect = document.getElementById('nuclei-select');
            if (nucleiSelect) {
                nucleiSelect.value = example.nucleus;
                nucleiSelect.dispatchEvent(new Event('change'));
            }
        }

        // Load type-specific parameters
        if (app && app.loadSequenceFromExample) {
            app.loadSequenceFromExample(example);
        }

        return true;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ExampleSequences;
}
