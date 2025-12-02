/**
 * Orthopedic Coil Generator - Main Application
 * Integrates variational measure theory with 3D visualization
 */

class CoilGeneratorApp {
    constructor() {
        this.vmt = new VariationalMeasureTheory();
        this.viz = new CoilVisualization('coil-canvas');
        this.currentCoil = null;
        this.animationFrame = null;

        this.init();
    }

    /**
     * Initialize application
     */
    init() {
        this.setupControls();
        this.setupViewControls();
        this.updateParameterDisplays();
        this.startIdleAnimation();
    }

    /**
     * Setup parameter controls
     */
    setupControls() {
        // Generate button
        document.getElementById('generate-btn').addEventListener('click', () => {
            this.generateCoil();
        });

        // Optimize button
        document.getElementById('optimize-btn').addEventListener('click', () => {
            this.optimizeCoil();
        });

        // Parameter sliders
        const sliders = ['num-channels', 'coil-radius', 'coil-length', 'iterations'];
        sliders.forEach(id => {
            document.getElementById(id).addEventListener('input', (e) => {
                this.updateParameterDisplays();
            });
        });
    }

    /**
     * Setup view controls
     */
    setupViewControls() {
        const viewButtons = document.querySelectorAll('.viz-btn');
        viewButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                viewButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                const view = btn.dataset.view;
                this.viz.setViewMode(view);
                if (this.currentCoil) {
                    this.viz.renderCoilGeometry(this.currentCoil);
                }
            });
        });
    }

    /**
     * Update parameter value displays
     */
    updateParameterDisplays() {
        document.getElementById('channels-value').textContent =
            document.getElementById('num-channels').value;
        document.getElementById('radius-value').textContent =
            (document.getElementById('coil-radius').value / 10).toFixed(1);
        document.getElementById('length-value').textContent =
            document.getElementById('coil-length').value;
        document.getElementById('iter-value').textContent =
            document.getElementById('iterations').value;
    }

    /**
     * Generate coil using current parameters
     */
    async generateCoil() {
        // Hide status overlay and start generation
        const overlay = document.getElementById('generation-status');
        overlay.innerHTML = `
            <div class="status-content loading">
                <div class="status-icon">⚡</div>
                <h3>Generating Coil...</h3>
                <p>Applying variational measure theory optimization</p>
            </div>
        `;

        // Get parameters from UI
        const numChannels = parseInt(document.getElementById('num-channels').value);
        const radius = parseFloat(document.getElementById('coil-radius').value) / 100; // Convert to meters
        const length = parseFloat(document.getElementById('coil-length').value) / 100;
        const iterations = parseInt(document.getElementById('iterations').value);

        // Update template
        const template = this.vmt.coilTemplates.knee;
        template.num_channels = numChannels;
        template.geometry.radius = radius;
        template.geometry.length = length;

        // Stop idle animation
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }

        // Simulate async generation (run optimization)
        await this.delay(100);

        // Generate optimal geometry
        overlay.innerHTML = `
            <div class="status-content loading">
                <div class="status-icon">🧮</div>
                <h3>Optimizing...</h3>
                <p>Running ${iterations} iterations of gradient descent</p>
            </div>
        `;

        await this.delay(100);

        this.currentCoil = this.vmt.generateOptimalCoilGeometry(template, iterations);

        // Hide overlay
        overlay.classList.add('hidden');

        // Render coil
        this.viz.renderCoilGeometry(this.currentCoil);

        // Calculate and display metrics
        this.displayMetrics();

        // Generate field maps
        this.generateFieldMaps();

        // Update status
        document.getElementById('coil-status').textContent = 'Generated';
        document.getElementById('coil-status').style.color = '#10b981';
    }

    /**
     * Optimize existing coil
     */
    async optimizeCoil() {
        if (!this.currentCoil) {
            alert('Please generate a coil first!');
            return;
        }

        const overlay = document.getElementById('generation-status');
        overlay.classList.remove('hidden');
        overlay.innerHTML = `
            <div class="status-content loading">
                <div class="status-icon">🎯</div>
                <h3>Re-optimizing...</h3>
                <p>Further minimizing energy functional</p>
            </div>
        `;

        await this.delay(100);

        const iterations = parseInt(document.getElementById('iterations').value);
        const template = this.vmt.coilTemplates.knee;

        this.currentCoil = this.vmt.generateOptimalCoilGeometry(template, iterations);

        overlay.classList.add('hidden');
        this.viz.renderCoilGeometry(this.currentCoil);
        this.displayMetrics();
        this.generateFieldMaps();
    }

    /**
     * Display performance metrics
     */
    displayMetrics() {
        const metrics = this.vmt.generatePerformanceMetrics(this.currentCoil);

        document.getElementById('coupling-metric').textContent = metrics.coupling.toFixed(4);
        document.getElementById('coverage-metric').textContent = metrics.coverage.toFixed(4);
        document.getElementById('snr-metric').textContent = metrics.avgSNR.toExponential(2);
        document.getElementById('efficiency-metric').textContent = metrics.efficiency.toFixed(3);

        // Color code based on performance
        const couplingEl = document.getElementById('coupling-metric');
        couplingEl.style.color = metrics.coupling < 0.1 ? '#10b981' : metrics.coupling < 0.2 ? '#f59e0b' : '#ef4444';

        const coverageEl = document.getElementById('coverage-metric');
        coverageEl.style.color = metrics.coverage > 0.7 ? '#10b981' : metrics.coverage > 0.5 ? '#f59e0b' : '#ef4444';
    }

    /**
     * Generate field maps
     */
    generateFieldMaps() {
        // Calculate B1 field map
        const b1Map = this.vmt.calculateB1FieldMap(
            this.currentCoil.positions,
            this.currentCoil.parameters,
            15 // resolution
        );

        // Calculate SNR map
        const snrMap = this.vmt.calculateSNRMap(
            this.currentCoil.positions,
            this.currentCoil.parameters,
            15
        );

        // Render maps
        this.viz.renderB1FieldMap('b1-field-map', b1Map);
        this.viz.renderSNRMap('snr-distribution', snrMap);
    }

    /**
     * Start idle animation
     */
    startIdleAnimation() {
        const animate = () => {
            if (!document.getElementById('generation-status').classList.contains('hidden')) {
                // Only animate when no coil is displayed
                this.animationFrame = requestAnimationFrame(animate);
            }
        };
        animate();
    }

    /**
     * Utility delay function
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.coilApp = new CoilGeneratorApp();
});
