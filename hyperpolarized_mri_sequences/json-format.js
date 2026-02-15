// JSON Format Handler
// Standardized JSON format for sequence definitions

const JSONFormat = {
    /**
     * Create sequence definition in JSON format
     */
    createSequenceJSON(sequenceData) {
        return {
            version: '1.0.0',
            generator: 'Hyperpolarized Pulse Sequence Generator',
            timestamp: new Date().toISOString(),
            sequence: {
                type: sequenceData.type,
                name: sequenceData.name || `${sequenceData.type}_sequence`,
                description: sequenceData.description || '',

                // Nucleus configuration
                nucleus: {
                    name: sequenceData.nucleus,
                    gamma: sequenceData.gamma,
                    t1: sequenceData.t1,
                    t2: sequenceData.t2,
                    b0Field: sequenceData.b0Field
                },

                // Imaging parameters
                parameters: sequenceData.parameters || {},

                // Flip angle schedule (if VFA)
                flipAngleSchedule: sequenceData.flipAngles || null,

                // Trajectory data (if applicable)
                trajectory: sequenceData.trajectory || null,

                // Timing
                timing: {
                    tr: sequenceData.tr,
                    te: sequenceData.te,
                    totalDuration: sequenceData.totalDuration
                },

                // Predicted signals
                simulation: sequenceData.simulation || null
            }
        };
    },

    /**
     * Parse JSON sequence definition
     */
    parseSequenceJSON(jsonString) {
        try {
            const data = JSON.parse(jsonString);

            // Validate structure
            if (!data.sequence || !data.sequence.type) {
                throw new Error('Invalid sequence JSON format');
            }

            return data;
        } catch (error) {
            console.error('Error parsing sequence JSON:', error);
            return null;
        }
    },

    /**
     * Export as formatted JSON string
     */
    exportJSON(sequenceData) {
        const jsonData = this.createSequenceJSON(sequenceData);
        return JSON.stringify(jsonData, null, 2);
    },

    /**
     * Create library entry
     */
    createLibraryEntry(sequenceData, metadata = {}) {
        return {
            id: metadata.id || this.generateID(),
            name: metadata.name || sequenceData.name,
            category: metadata.category || sequenceData.type,
            tags: metadata.tags || [],
            created: new Date().toISOString(),
            modified: new Date().toISOString(),
            author: metadata.author || 'Sunnybrook Research Institute',
            sequence: this.createSequenceJSON(sequenceData)
        };
    },

    /**
     * Generate unique ID
     */
    generateID() {
        return 'seq_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = JSONFormat;
}
