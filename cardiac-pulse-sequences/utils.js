// Utility Functions for Cardiac Parallel Imaging Application

const Utils = {
    /**
     * Format a number with specified decimal places
     */
    formatNumber(value, decimals = 2) {
        return Number(value).toFixed(decimals);
    },

    /**
     * Format time in milliseconds
     */
    formatTime(ms) {
        if (ms < 1000) {
            return `${Math.round(ms)}ms`;
        } else {
            return `${(ms / 1000).toFixed(1)}s`;
        }
    },

    /**
     * Convert BPM to RR interval in ms
     */
    bpmToRR(bpm) {
        return (60000 / bpm);
    },

    /**
     * Convert RR interval to BPM
     */
    rrToBPM(rrMs) {
        return (60000 / rrMs);
    },

    /**
     * Clamp a value between min and max
     */
    clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    },

    /**
     * Linear interpolation
     */
    lerp(a, b, t) {
        return a + (b - a) * t;
    },

    /**
     * Convert degrees to radians
     */
    degToRad(degrees) {
        return degrees * Math.PI / 180;
    },

    /**
     * Convert radians to degrees
     */
    radToDeg(radians) {
        return radians * 180 / Math.PI;
    },

    /**

     * Validate parameter is within range
     */
    validateRange(value, min, max, name) {
        if (value < min || value > max) {
            console.warn(`${name} value ${value} is outside valid range [${min}, ${max}]`);
            return false;
        }
        return true;
    },

    /**
     * Deep clone an object
     */
    deepClone(obj) {
        return JSON.parse(JSON.stringify(obj));
    },

    /**
     * Generate unique ID
     */
    generateId() {
        return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    },

    /**
     * Download text as file
     */
    downloadFile(content, filename, contentType = 'text/plain') {
        const blob = new Blob([content], { type: contentType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    },

    /**
     * Parse clinical indication for keywords
     */
    parseClinicalIndication(text) {
        const keywords = {
            conditions: [],
            modifiers: []
        };

        const conditionMap = {
            'myocarditis': ['myocarditis', 'inflammation'],
            'cardiomyopathy': ['cardiomyopathy', 'dcm', 'hcm'],
            'ischemia': ['ischemia', 'ischemic', 'cad', 'coronary'],
            'infarction': ['infarct', 'mi', 'stemi', 'nstemi'],
            'viability': ['viability', 'viable'],
            'scar': ['scar', 'fibrosis'],
            'arrhythmia': ['arrhythmia', 'afib', 'vt'],
            'valve': ['valve', 'stenosis', 'regurgitation'],
            'mass': ['mass', 'tumor', 'thrombus']
        };

        const lowerText = text.toLowerCase();

        for (const [condition, keywords_list] of Object.entries(conditionMap)) {
            if (keywords_list.some(kw => lowerText.includes(kw))) {
                keywords.conditions.push(condition);
            }
        }

        // Extract modifiers
        if (lowerText.includes('acute')) keywords.modifiers.push('acute');
        if (lowerText.includes('chronic')) keywords.modifiers.push('chronic');
        if (lowerText.includes('suspected')) keywords.modifiers.push('suspected');
        if (lowerText.includes('known')) keywords.modifiers.push('known');

        return keywords;
    },

    /**
     * Format matrix size
     */
    formatMatrix(rows, cols) {
        return `${rows}×${cols}`;
    },

    /**
     * Calculate FOV from matrix and resolution
     */
    calculateFOV(matrix, resolution) {
        return matrix * resolution;
    },

    /**
     * Calculate resolution from FOV and matrix
     */
    calculateResolution(fov, matrix) {
        return fov / matrix;
    }
};

// Export for use in other modules
window.Utils = Utils;
