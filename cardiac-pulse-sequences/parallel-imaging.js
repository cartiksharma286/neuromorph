// Parallel Imaging Configuration and State Management

const ParallelImaging = {
    state: {
        technique: 'sense',
        accelerationFactor: 2.0,
        coilElements: 18,
        acsLines: 24,
        kSpaceLines: 256,
        fov: 400, // mm
        scanTimeBaseline: 180000 // ms
    },

    /**
     * Initialize parallel imaging module
     */
    init() {
        console.log('Parallel Imaging module initialized');
        this.bindEvents();
        // Delay initial calculation slightly to ensure everything loaded
        setTimeout(() => this.updateCalculations(), 100);
    },

    /**
     * Bind UI events
     */
    bindEvents() {
        const elements = {
            technique: document.getElementById('pi-technique'),
            acceleration: document.getElementById('acceleration-factor'),
            coils: document.getElementById('coil-elements'),
            acs: document.getElementById('acs-lines')
        };

        if (elements.technique) {
            elements.technique.addEventListener('change', (e) => {
                this.state.technique = e.target.value;
                this.updateUI();
                this.updateCalculations();
            });
        }

        if (elements.acceleration) {
            elements.acceleration.addEventListener('input', (e) => {
                this.state.accelerationFactor = parseFloat(e.target.value);
                document.getElementById('acceleration-value').textContent =
                    this.state.accelerationFactor.toFixed(1) + 'x';
                this.updateCalculations();
            });
        }

        if (elements.coils) {
            elements.coils.addEventListener('input', (e) => {
                this.state.coilElements = parseInt(e.target.value);
                document.getElementById('coil-value').textContent =
                    this.state.coilElements + ' channels';
                this.updateCalculations();
            });
        }

        if (elements.acs) {
            elements.acs.addEventListener('input', (e) => {
                this.state.acsLines = parseInt(e.target.value);
                document.getElementById('acs-value').textContent =
                    this.state.acsLines + ' lines';
                this.updateCalculations();
            });
        }
    },

    /**
     * Update UI based on technique
     */
    updateUI() {
        const acsGroup = document.getElementById('acs-group');
        if (acsGroup) {
            // Show ACS controls only for GRAPPA
            acsGroup.style.display = this.state.technique === 'grappa' ? 'block' : 'none';
        }
    },

    /**
     * Update all calculations
     */
    updateCalculations() {
        if (this.state.technique === 'quantum') {
            this.performQuantumOptimization();
            return;
        }

        // Calculate g-factor
        const gResult = ParallelPhysics.calculateGFactor(
            this.state.accelerationFactor,
            this.state.coilElements,
            'circular'
        );

        // Calculate SNR penalty
        const snrResult = ParallelPhysics.calculateSNRPenalty(
            gResult.gFactorMean,
            this.state.accelerationFactor
        );

        // Calculate scan time
        const timeResult = ParallelPhysics.calculateScanTimeReduction(
            this.state.scanTimeBaseline,
            this.state.accelerationFactor,
            this.state.technique === 'grappa' ? 2000 : 0
        );

        // Updatemetrics display
        this.updateMetrics(gResult, snrResult, timeResult);

        // Update visualization
        if (window.Visualizer) {
            Visualizer.drawKSpace(this.state);
        }
    },

    /**
     * Perform Quantum Optimization via Backend API
     */
    async performQuantumOptimization() {
        const statusDisplay = document.getElementById('g-factor-display');
        if (statusDisplay) statusDisplay.textContent = 'Optimizing...';

        try {
            const response = await fetch('/api/quantum-optimize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    accelerationFactor: this.state.accelerationFactor,
                    coilElements: this.state.coilElements
                })
            });

            if (!response.ok) throw new Error('Optimization failed');

            const result = await response.json();

            // Update metrics with quantum results
            this.updateQuantumMetrics(result);

            // Update visualization with quantum pattern
            if (window.Visualizer) {
                // We inject the custom pattern into the state for the visualizer to use
                const quantumState = { ...this.state, customPattern: result.pattern };
                Visualizer.drawKSpace(quantumState);
            }

        } catch (error) {
            console.error('Quantum optimization error:', error);
            // Fallback to standard calculation
            const gResult = ParallelPhysics.calculateGFactor(
                this.state.accelerationFactor,
                this.state.coilElements
            );
            this.updateMetrics(gResult, { snrPenaltyPercent: 0 }, { timeSaved: 0 });
        }
    },

    updateQuantumMetrics(result) {
        const gFactorDisplay = document.getElementById('g-factor-display');
        const snrDisplay = document.getElementById('snr-penalty-display');
        const timeDisplay = document.getElementById('scan-time-display');

        if (gFactorDisplay) {
            gFactorDisplay.innerHTML = `${result.g_factor} <span style="color:#4ade80; font-size:0.8em">▼ Q-Optimized</span>`;
        }

        if (snrDisplay) {
            // Calculate effective SNR penalty with improvement
            snrDisplay.innerHTML = `Low <span style="color:#4ade80; font-size:0.8em">(+${result.snr_improvement})</span>`;
        }

        if (timeDisplay) {
            const timeSeconds = this.state.scanTimeBaseline / this.state.accelerationFactor / 1000;
            timeDisplay.textContent = Math.round(timeSeconds) + 's';
        }
    },

    /**
     * Update metrics display
     */
    updateMetrics(gResult, snrResult, timeResult) {
        const gFactorDisplay = document.getElementById('g-factor-display');
        const snrDisplay = document.getElementById('snr-penalty-display');
        const timeDisplay = document.getElementById('scan-time-display');

        if (gFactorDisplay) {
            gFactorDisplay.textContent = gResult.gFactorMean.toFixed(2);
        }

        if (snrDisplay) {
            snrDisplay.textContent = snrResult.snrPenaltyPercent.toFixed(0) + '%';
        }

        if (timeDisplay) {
            const timeSeconds = this.state.scanTimeBaseline / this.state.accelerationFactor / 1000;
            timeDisplay.textContent = Math.round(timeSeconds) + 's';
        }
    },

    /**
     * Get current configuration
     */
    getConfig() {
        return { ...this.state };
    },

    /**
     * Apply LLM-suggested configuration
     */
    applyConfig(config) {
        if (config.accelerationFactor) {
            this.state.accelerationFactor = config.accelerationFactor;
            const slider = document.getElementById('acceleration-factor');
            if (slider) {
                slider.value = config.accelerationFactor;
                document.getElementById('acceleration-value').textContent =
                    config.accelerationFactor.toFixed(1) + 'x';
            }
        }

        if (config.technique) {
            this.state.technique = config.technique;
            const select = document.getElementById('pi-technique');
            if (select) select.value = config.technique;
        }

        this.updateUI();
        this.updateCalculations();
    }
};

// Export
window.ParallelImaging = ParallelImaging;
