// Main Application Controller
class MRICoilApp {
    constructor() {
        this.currentTab = 'geodesic';
        this.init();
    }

    init() {
        this.setupTabNavigation();
        this.setupGeodesicControls();
        this.setupSchematicControls();
        this.setupRFCalculatorControls();
        this.setupAIAssistantControls();
        this.setupGlobalControls();

        // Initialize modules
        initGeodesicEngine();
        initSchematicGenerator();
        initRFCalculator();
        initLLMAssistant();

        // Initial render
        this.updateGeodesicView();
        this.updateSchematicView();
    }

    // Tab navigation
    setupTabNavigation() {
        const tabs = document.querySelectorAll('.tab-btn');
        const contents = document.querySelectorAll('.tab-content');

        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const tabId = tab.dataset.tab;

                // Update active states
                tabs.forEach(t => t.classList.remove('active'));
                contents.forEach(c => c.classList.remove('active'));

                tab.classList.add('active');
                document.getElementById(`${tabId}-tab`).classList.add('active');

                this.currentTab = tabId;
            });
        });
    }

    // Geodesic controls
    setupGeodesicControls() {
        // Field strength
        const fieldStrength = document.getElementById('fieldStrength');
        if (fieldStrength) {
            fieldStrength.addEventListener('change', () => this.updateGeodesicView());
        }

        // Geodesic frequency slider
        const geodesicFreq = document.getElementById('geodesicFreq');
        const geodesicFreqValue = document.getElementById('geodesicFreqValue');
        if (geodesicFreq) {
            geodesicFreq.addEventListener('input', (e) => {
                const freq = parseInt(e.target.value);
                geodesicFreqValue.textContent = freq;

                if (geodesicEngine) {
                    const vertexCount = geodesicEngine.generateGeodesic(freq);
                    document.getElementById('vertexCount').textContent = `${vertexCount} vertices`;
                    document.getElementById('totalVertices').textContent = vertexCount;
                    this.updateGeodesicView();
                }
            });
        }

        // Coil radius
        const coilRadius = document.getElementById('coilRadius');
        if (coilRadius) {
            coilRadius.addEventListener('input', () => this.updateGeodesicView());
        }

        // Number of elements slider
        const numElements = document.getElementById('numElements');
        const numElementsValue = document.getElementById('numElementsValue');
        if (numElements) {
            numElements.addEventListener('input', (e) => {
                const num = parseInt(e.target.value);
                numElementsValue.textContent = num;
                this.updateGeodesicView();
            });
        }

        // Visualization options
        const showWireframe = document.getElementById('showWireframe');
        const showElements = document.getElementById('showElements');
        const autoRotate = document.getElementById('autoRotate');

        if (showWireframe) {
            showWireframe.addEventListener('change', () => this.updateGeodesicView());
        }
        if (showElements) {
            showElements.addEventListener('change', () => this.updateGeodesicView());
        }
        if (autoRotate) {
            autoRotate.addEventListener('change', (e) => {
                if (geodesicEngine) {
                    geodesicEngine.autoRotate = e.target.checked;
                }
            });
        }

        // Canvas controls
        const resetView = document.getElementById('resetView');
        const downloadCanvas = document.getElementById('downloadCanvas');

        if (resetView) {
            resetView.addEventListener('click', () => {
                if (geodesicEngine) geodesicEngine.resetView();
            });
        }
        if (downloadCanvas) {
            downloadCanvas.addEventListener('click', () => {
                if (geodesicEngine) geodesicEngine.downloadImage();
            });
        }
    }

    // Schematic controls
    setupSchematicControls() {
        const controls = [
            'coilTopology',
            'loopDiameter',
            'wireGauge',
            'includeMatching',
            'includeDecoupling'
        ];

        controls.forEach(id => {
            const elem = document.getElementById(id);
            if (elem) {
                elem.addEventListener('change', () => this.updateSchematicView());
            }
        });

        // Loop diameter also updates component values
        const loopDiameter = document.getElementById('loopDiameter');
        if (loopDiameter) {
            loopDiameter.addEventListener('input', () => {
                this.updateComponentValues();
                this.updateSchematicView();
            });
        }
    }

    // RF calculator controls
    setupRFCalculatorControls() {
        const controls = [
            'calcFieldStrength',
            'nucleus',
            'loopShape',
            'calcLoopDiameter',
            'calcWireDiameter',
            'resInductance',
            'targetFreq',
            'snrCoilDiameter',
            'snrDepth'
        ];

        controls.forEach(id => {
            const elem = document.getElementById(id);
            if (elem) {
                elem.addEventListener('input', () => updateRFCalculations());
            }
        });
    }

    // AI assistant controls
    setupAIAssistantControls() {
        const generateBtn = document.getElementById('aiGenerateBtn');
        if (generateBtn) {
            generateBtn.addEventListener('click', () => this.generateAIRecommendations());
        }
    }

    // Global controls
    setupGlobalControls() {
        const generateBtn = document.getElementById('generateBtn');
        const exportBtn = document.getElementById('exportBtn');

        if (generateBtn) {
            generateBtn.addEventListener('click', () => this.quickGenerate());
        }
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportDesign());
        }
    }

    // Update geodesic visualization
    updateGeodesicView() {
        if (!geodesicEngine) return;

        const showWireframe = document.getElementById('showWireframe')?.checked ?? true;
        const showElements = document.getElementById('showElements')?.checked ?? true;
        const numElements = parseInt(document.getElementById('numElements')?.value || 32);

        geodesicEngine.render({
            showWireframe,
            showElements,
            numActiveElements: numElements
        });

        // Update stats
        this.updateGeodesicStats();
    }

    // Update geodesic stats
    updateGeodesicStats() {
        const fieldStrength = parseFloat(document.getElementById('fieldStrength')?.value || 3);
        const numElements = parseInt(document.getElementById('numElements')?.value || 32);

        // Larmor frequency
        if (rfCalculator) {
            const larmorFreq = rfCalculator.calculateLarmorFrequency(fieldStrength);
            const larmorElem = document.getElementById('larmorFreq');
            if (larmorElem) larmorElem.textContent = `${larmorFreq} MHz`;
        }

        // Active elements
        const activeElem = document.getElementById('activeElements');
        if (activeElem) activeElem.textContent = numElements;

        // Coverage (simple estimate)
        const totalVertices = geodesicEngine ? geodesicEngine.vertices.length : 42;
        const coverage = Math.round((numElements / totalVertices) * 100);
        const coverageElem = document.getElementById('coverage');
        if (coverageElem) coverageElem.textContent = `${coverage}%`;
    }

    // Update schematic view
    updateSchematicView() {
        if (!schematicGenerator) return;

        const loopDiameter = parseInt(document.getElementById('loopDiameter')?.value || 80);
        const includeMatching = document.getElementById('includeMatching')?.checked ?? true;
        const includeDecoupling = document.getElementById('includeDecoupling')?.checked ?? true;

        // Get component values
        const loopInductance = parseFloat(document.getElementById('loopInductance')?.textContent || 180);
        const tuningCap = parseFloat(document.getElementById('tuningCap')?.textContent || 8.2);
        const matchingCap = parseFloat(document.getElementById('matchingCap')?.textContent || 22);

        schematicGenerator.generateCoilCircuit({
            loopDiameter,
            loopInductance,
            tuningCap,
            matchingCap,
            includeMatching,
            includeDecoupling
        });
    }

    // Update component values
    updateComponentValues() {
        if (!rfCalculator) return;

        const loopDiameter = parseInt(document.getElementById('loopDiameter')?.value || 80);
        const fieldStrength = parseFloat(document.getElementById('fieldStrength')?.value || 3);
        const wireDiameter = 1.3; // 16 AWG

        // Calculate inductance
        const inductance = rfCalculator.calculateLoopInductance(loopDiameter, wireDiameter);
        const inductanceElem = document.getElementById('loopInductance');
        if (inductanceElem) inductanceElem.textContent = `${inductance} nH`;

        // Calculate tuning capacitance
        const larmorFreq = rfCalculator.calculateLarmorFrequency(fieldStrength);
        const tuningCap = rfCalculator.calculateResonanceCapacitance(parseFloat(inductance), parseFloat(larmorFreq));
        const tuningCapElem = document.getElementById('tuningCap');
        if (tuningCapElem) tuningCapElem.textContent = `${tuningCap} pF`;

        // Matching capacitance (approximate)
        const matchingCap = (parseFloat(tuningCap) * 2.5).toFixed(1);
        const matchingCapElem = document.getElementById('matchingCap');
        if (matchingCapElem) matchingCapElem.textContent = `${matchingCap} pF`;

        // Q-factor
        const qFactor = rfCalculator.estimateQFactor(loopDiameter, parseFloat(larmorFreq), wireDiameter);
        const qFactorElem = document.getElementById('qFactor');
        if (qFactorElem) qFactorElem.textContent = qFactor;
    }

    // Generate AI recommendations
    generateAIRecommendations() {
        if (!llmAssistant) return;

        const fieldStrength = parseFloat(document.getElementById('aiFieldStrength')?.value || 3);
        const targetAnatomy = document.getElementById('targetAnatomy')?.value || 'brain';
        const designGoal = document.getElementById('designGoal')?.value || 'balanced';

        const recommendations = llmAssistant.generateRecommendations({
            fieldStrength,
            targetAnatomy,
            designGoal
        });

        // Display recommendations
        const aiOutput = document.getElementById('aiOutput');
        if (aiOutput) {
            aiOutput.innerHTML = llmAssistant.formatRecommendationsHTML(recommendations);
        }

        // Apply recommendations to geodesic tab
        this.applyRecommendations(recommendations);

        // Generate documentation
        this.generateDocumentation(recommendations);
    }

    // Apply recommendations to controls
    applyRecommendations(recommendations) {
        const { parameters, components } = recommendations;

        // Update geodesic controls
        const geodesicFreq = document.getElementById('geodesicFreq');
        if (geodesicFreq) {
            geodesicFreq.value = parameters.geodesicFrequency;
            document.getElementById('geodesicFreqValue').textContent = parameters.geodesicFrequency;

            if (geodesicEngine) {
                const vertexCount = geodesicEngine.generateGeodesic(parameters.geodesicFrequency);
                document.getElementById('vertexCount').textContent = `${vertexCount} vertices`;
            }
        }

        const numElements = document.getElementById('numElements');
        if (numElements) {
            numElements.value = parameters.channelCount;
            document.getElementById('numElementsValue').textContent = parameters.channelCount;
        }

        const coilRadius = document.getElementById('coilRadius');
        if (coilRadius) {
            coilRadius.value = parameters.coilRadius;
        }

        // Update schematic controls
        const loopDiameter = document.getElementById('loopDiameter');
        if (loopDiameter) {
            loopDiameter.value = parameters.loopDiameter;
        }

        // Refresh views
        this.updateGeodesicView();
        this.updateComponentValues();
        this.updateSchematicView();
    }

    // Generate documentation
    generateDocumentation(recommendations) {
        const docOutput = document.getElementById('docOutput');
        if (!docOutput) return;

        const { parameters, components, performance } = recommendations;

        const doc = `
## Technical Specification Document

### System Parameters
- **Operating Frequency:** ${parameters.larmorFrequency} MHz
- **Array Architecture:** ${parameters.channelCount}-channel geodesic phased array
- **Coverage Radius:** ${parameters.coilRadius} mm
- **Element Size:** ${parameters.loopDiameter} mm diameter

### Electrical Design
- **Loop Inductance:** ${components.loopInductance} nH
- **Tuning Capacitor:** ${components.tuningCapacitor} pF (non-magnetic ceramic)
- **Matching Capacitor:** ${components.matchingCapacitor} pF (variable)
- **Wire:** ${components.wireGauge} AWG copper

### Performance Metrics
- **Q-Factor:** ${performance.estimatedQFactor} (estimated)
- **Relative SNR:** ${performance.expectedSNR}×
- **Parallel Imaging:** ${performance.parallelImagingCapability}

### Construction Notes
1. Fabricate individual coil elements on rigid former
2. Tune each element to resonance (check with dipper or VNA)
3. Match to 50Ω using parallel capacitor adjustment
4. Assemble elements on geodesic former
5. Verify inter-element decoupling (target: <-15 dB S21)
6. Perform bench SNR measurements with phantom
        `.trim();

        docOutput.innerHTML = `<pre style="white-space: pre-wrap; line-height: 1.8; color: #cbd5e1;">${doc}</pre>`;
    }

    // Quick generate from header button
    quickGenerate() {
        // Switch to AI assistant tab and generate
        const aiTab = document.querySelector('[data-tab="ai-assist"]');
        if (aiTab) aiTab.click();

        setTimeout(() => {
            this.generateAIRecommendations();
        }, 300);
    }

    // Export design
    exportDesign() {
        const timestamp = new Date().toISOString().slice(0, 10);

        // Export geodesic canvas
        if (geodesicEngine) {
            geodesicEngine.downloadImage(`mri-coil-design-${timestamp}.png`);
        }

        // In a real application, would also export:
        // - PDF report with all specifications
        // - CAD files for mechanical design
        // - Netlist for circuit simulation
        // - BOM spreadsheet

        alert('Design exported! In a full implementation, this would generate:\n\n' +
            '• 3D visualization (PNG)\n' +
            '• Circuit schematics (PDF)\n' +
            '• Technical specifications (PDF)\n' +
            '• Bill of Materials (CSV)\n' +
            '• CAD files (STEP/STL)');
    }
}

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new MRICoilApp();
    window.mriCoilApp = app;
});
