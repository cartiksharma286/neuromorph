// Cardiac-specific Sequence Definitions and Management

const CardiacSequences = {
    state: {
        cine: {
            type: 'bssfp',
            temporalRes: 40,
            cardiacPhases: 25,
            heartRate: 70
        },
        perfusion: {
            srTime: 100,
            slices: 4,
            temporalFootprint: 150
        },
        lge: {
            ti: 300,
            recon: 'psir',
            fatSat: 'spir'
        },
        mapping: {
            type: 't1-molli',
            scheme: '5-3-3'
        },
        flow: {
            mode: '2d',
            venc: 150,
            direction: 'through-plane'
        }
    },

    /**
     * Initialize cardiac sequences
     */
    init() {
        this.bindCINEEvents();
        this.bindPerfusionEvents();
        this.bindLGEEvents();
        this.bindMappingEvents();
        this.bindFlowEvents();

        // Initial calculations
        this.updateCINECalculations();
        console.log('Cardiac Sequences module initialized');
    },

    /**
     * Bind CINE events
     */
    bindCINEEvents() {
        const elements = {
            temporalRes: document.getElementById('temporal-res'),
            cardiacPhases: document.getElementById('cardiac-phases'),
            heartRate: document.getElementById('heart-rate')
        };

        if (elements.temporalRes) {
            elements.temporalRes.addEventListener('input', (e) => {
                this.state.cine.temporalRes = parseInt(e.target.value);
                document.getElementById('temporal-value').textContent = this.state.cine.temporalRes + ' ms';
                this.updateCINECalculations();
            });
        }

        if (elements.cardiacPhases) {
            elements.cardiacPhases.addEventListener('input', (e) => {
                this.state.cine.cardiacPhases = parseInt(e.target.value);
                document.getElementById('phases-value').textContent = this.state.cine.cardiacPhases + ' phases';
                this.updateCINECalculations();
            });
        }

        if (elements.heartRate) {
            elements.heartRate.addEventListener('input', (e) => {
                this.state.cine.heartRate = parseInt(e.target.value);
                document.getElementById('hr-value').textContent = this.state.cine.heartRate + ' bpm';
                this.updateCINECalculations();
            });
        }
    },

    /**
     * Bind other sequence events
     */
    bindPerfusionEvents() {
        const srTime = document.getElementById('sr-time');
        if (srTime) {
            srTime.addEventListener('input', (e) => {
                this.state.perfusion.srTime = parseInt(e.target.value);
                document.getElementById('sr-value').textContent = this.state.perfusion.srTime + ' ms';
            });
        }
    },

    bindLGEEvents() {
        const tiValue = document.getElementById('ti-value');
        if (tiValue) {
            tiValue.addEventListener('input', (e) => {
                this.state.lge.ti = parseInt(e.target.value);
                document.getElementById('ti-display').textContent = this.state.lge.ti + ' ms';
            });
        }
    },

    bindMappingEvents() {
        // Mapping events
        const mappingType = document.getElementById('mapping-type');
        if (mappingType) {
            mappingType.addEventListener('change', (e) => {
                this.state.mapping.type = e.target.value;
            });
        }
    },

    bindFlowEvents() {
        const venc = document.getElementById('venc');
        if (venc) {
            venc.addEventListener('input', (e) => {
                this.state.flow.venc = parseInt(e.target.value);
                document.getElementById('venc-value').textContent = this.state.flow.venc + ' cm/s';
            });
        }
    },

    /**
     * Update CINE calculations and visualization
     */
    updateCINECalculations() {
        const rr = Utils.bpmToRR(this.state.cine.heartRate);

        // Update displays
        const rrDisplay = document.getElementById('rr-interval-display');
        const phasesDisplay = document.getElementById('temporal-phases-display');

        if (rrDisplay) {
            rrDisplay.textContent = Math.round(rr);
        }

        if (phasesDisplay) {
            phasesDisplay.textContent = this.state.cine.cardiacPhases;
        }

        // Update visualization
        if (window.Visualizer) {
            Visualizer.drawCINE(
                this.state.cine.heartRate,
                this.state.cine.cardiacPhases,
                this.state.cine.temporalRes
            );
        }
    },

    /**
     * Get sequence configuration
     */
    getConfig(sequenceType) {
        return { ...this.state[sequenceType] };
    }
};

// Export
window.CardiacSequences = CardiacSequences;
