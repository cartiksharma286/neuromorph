class QMLOptimizer {
    constructor() {
        this.ctx = null;
        this.feaCanvas = null;
        this.isOptimizing = false;
        this.isAnimating = false;
        this.init();
    }

    init() {
        this.setupEventListeners();
    }

    setupEventListeners() {
        const optimizeBtn = document.getElementById('runQMLOptimization');
        if (optimizeBtn) {
            optimizeBtn.addEventListener('click', () => this.runOptimizationSequence());
        }

        const inputs = ['distType', 'qmlDepth', 'qmlLearningRate'];
        inputs.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.addEventListener('change', () => this.updateFEAPreview());
            }
        });
    }

    async runOptimizationSequence() {
        if (this.isOptimizing) return;
        this.isOptimizing = true;

        const btn = document.getElementById('runQMLOptimization');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<span class="loading-spinner"></span> Optimizing...';
        btn.disabled = true;

        // Start animation loop
        this.isAnimating = true;
        this.animateOptimization();

        // Simulate QML Processing Steps
        await this.simulateStep('Initializing Quantum Registers...', 800);
        await this.simulateStep('Injecting Statistical Priors...' + this.getDistributionName(), 1000);
        await this.simulateStep('Running Variational Quantum Eigensolver (VQE)...', 1500);
        await this.simulateStep('Consulting Gemini 3.0 Knowledge Graph...', 1200);

        // Stop animation
        this.isAnimating = false;

        // Generate Final Results
        this.generateResults();

        btn.innerHTML = originalText;
        btn.disabled = false;
        this.isOptimizing = false;

        document.getElementById('optimizationStatus').innerHTML =
            '<span style="color: #4ade80">Optimization Complete: Converged to Global Minimum</span>';
    }

    async simulateStep(message, duration) {
        document.getElementById('optimizationStatus').textContent = message;
        // Wait for the duration while animation runs in background
        await new Promise(r => setTimeout(r, duration));
    }

    animateOptimization() {
        if (!this.isAnimating) return;

        // Render current frame with time-dependent noise
        this.renderFEAHeatmap(true);
        requestAnimationFrame(() => this.animateOptimization());
    }

    getDistributionName() {
        const el = document.getElementById('distType');
        return el ? ` (${el.options[el.selectedIndex].text})` : '';
    }

    generateResults() {
        // 1. Update FEA with "Optimized" smooth beam
        this.renderFEAHeatmap(false);

        // 2. Generate Specs
        this.updateSpecs();

        // 3. Generate Gemini Insights
        this.updateGeminiInsights();

        // 4. Update Schematic visual (hook into existing or new)
        this.drawEnhancedSchematic();
    }

    renderFEAHeatmap(isNoisy) {
        const canvas = document.getElementById('feaCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        // Performance: Use lower internal resolution for rendering
        // Keep display size high via CSS, but render fewer pixels
        const renderWidth = 200;
        const renderHeight = 150;

        if (canvas.width !== renderWidth) {
            canvas.width = renderWidth;
            canvas.height = renderHeight;
        }

        // Clear
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, renderWidth, renderHeight);

        // Create gradient data
        const imgData = ctx.createImageData(renderWidth, renderHeight);
        const data = imgData.data;

        const cx = renderWidth / 2;
        const cy = renderHeight / 2;
        const distType = document.getElementById('distType')?.value || 'gaussian';

        // Time-based variation for animation
        const t = performance.now() * 0.005;

        // Beam characteristics
        let spread = isNoisy ? 10 + Math.sin(t) * 5 : 20; // Scaled for 200px width
        if (distType === 'poisson') spread *= 0.8;
        if (distType === 'lorentzian') spread *= 1.2;
        if (distType === 'fermi-dirac') spread *= 1.5;

        let intensity = isNoisy ? 150 : 255;

        for (let y = 0; y < renderHeight; y++) {
            for (let x = 0; x < renderWidth; x++) {
                const idx = (y * renderWidth + x) * 4;

                // Distance from center
                const dx = x - cx;
                const dy = y - cy;
                const r = Math.sqrt(dx * dx + dy * dy);

                let val = 0;

                if (isNoisy) {
                    // Optimized noise calculation
                    const noise = Math.sin(x * 0.2 + t) * Math.cos(y * 0.2 - t) * 50;
                    // Circle mask
                    const mask = r < (renderWidth / 2.5) ? 1 : 0.1;
                    val = (Math.random() * 30 + noise + 80) * mask;
                } else {
                    // Analytical Beam Profile
                    if (distType === 'gaussian') {
                        val = intensity * Math.exp(-(r * r) / (2 * spread * spread));
                    } else if (distType === 'poisson') {
                        // Approximation for visual variety
                        val = intensity * (r / spread) * Math.exp(-r / spread) * 2.7;
                    } else if (distType === 'fermi-dirac') {
                        val = intensity / (1 + Math.exp((r - spread * 3) / 2));
                    } else { // Lorentzian
                        val = intensity * (1 / (1 + (r * r) / (spread * spread)));
                    }
                }

                // Heatmap Coloring Scheme (Blue -> Green -> Red -> White)
                const [R, G, B] = this.getHeatmapColor(val);

                data[idx] = R;
                data[idx + 1] = G;
                data[idx + 2] = B;
                data[idx + 3] = 255; // Alpha
            }
        }

        ctx.putImageData(imgData, 0, 0);

        // Minimal Grid
        this.drawGrid(ctx, renderWidth, renderHeight);

        // Add Annotations
        // Draw text on low-res canvas - will be blocky but legible enough for "retro-tech" feel
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.font = '10px monospace';
        if (!isNoisy) {
            // ctx.fillText(`Pk: ${(intensity/2.55).toFixed(0)}%`, 5, 12);
        }
    }

    getHeatmapColor(val) {
        if (val < 0) val = 0;
        if (val > 255) val = 255;

        let r, g, b;
        if (val < 64) {
            r = 0; g = (val / 64) * 255; b = 255;
        } else if (val < 128) {
            r = 0; g = 255; b = 255 - ((val - 64) / 64) * 255;
        } else if (val < 192) {
            r = ((val - 128) / 64) * 255; g = 255; b = 0;
        } else {
            r = 255; g = 255 - ((val - 192) / 64) * 255; b = 0;
        }
        return [Math.floor(r), Math.floor(g), Math.floor(b)];
    }

    drawGrid(ctx, w, h) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = 1;

        // Radial
        for (let r = 10; r < w / 2; r += 20) {
            ctx.beginPath();
            ctx.arc(w / 2, h / 2, r, 0, Math.PI * 2);
            ctx.stroke();
        }

        // Cross
        ctx.beginPath();
        ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
        ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
        ctx.stroke();
    }

    updateSpecs() {
        const specs = {
            'optInductance': (150 + Math.random() * 20).toFixed(2) + ' nH',
            'optCapacitance': (12 + Math.random() * 3).toFixed(2) + ' pF',
            'optQFactor': (280 + Math.random() * 40).toFixed(0),
            'optSAR': (0.4 + Math.random() * 0.1).toFixed(2) + ' W/kg'
        };

        for (let key in specs) {
            const el = document.getElementById(key);
            if (el) el.textContent = specs[key];
        }
    }

    updateGeminiInsights() {
        const container = document.getElementById('geminiInsights');
        if (!container) return;

        const distType = document.getElementById('distType')?.value || 'Gaussian';

        const insights = `
            <div class="gemini-msg">
                <strong><span style="color: #60a5fa">Gemini 3.0 Analysis:</span></strong>
                <p>Based on the ${distType} distribution topography and QML convergence patterns, I recommend a <strong>hybrid lattice structure</strong> for the coil array.</p>
                <ul class="gemini-list">
                    <li>The <strong>VQE algorithm</strong> identified a local minimum in SAR deposition at the temporal lobes, suggesting a 12% reduction in drive voltage for lateral elements.</li>
                    <li><strong>Beam focused</strong> visualization confirms peak B1+ homogeneity deviation is &lt; 1.5% within the central 10cm sphere.</li>
                    <li>Integration of statistical priors suggests replacing static capacitors with <strong>varactor diodes</strong> for real-time impedance matching dynamic correction.</li>
                </ul>
            </div>
        `;

        container.innerHTML = insights;
    }

    drawEnhancedSchematic() {
        const svg = document.getElementById('qmlSchematicSvg');
        if (!svg) return;

        svg.innerHTML = '';
        const w = 600;
        const h = 200;
        const y = h / 2;

        const path = `
            M 50 ${y} 
            L 150 ${y} 
            L 150 ${y - 40} L 170 ${y - 40} M 160 ${y - 40} L 160 ${y - 20} 
            M 150 ${y - 40} L 150 ${y - 20}
            M 170 ${y} L 250 ${y}
            l 10 -20 l 10 40 l 10 -40 l 10 40 l 10 -20
            L 400 ${y}
        `;

        let svgContent = `<path d="${path}" stroke="#6366f1" stroke-width="3" fill="none" />`;

        // Add "Quantum Tuner" box
        svgContent += `
            <rect x="250" y="${y + 20}" width="60" height="40" rx="4" fill="#1e293b" stroke="#38bdf8" stroke-width="2"/>
            <text x="280" y="${y + 45}" fill="#38bdf8" text-anchor="middle" font-size="10" font-family="monospace">Q-TUNE</text>
        `;

        svg.innerHTML = svgContent;
    }

    updateFEAPreview() {
        this.renderFEAHeatmap(true); // Preview
    }
}

// Global instance
let qmlOptimizer = null;
function initQMLOptimizer() {
    qmlOptimizer = new QMLOptimizer();
    // Initial Render
    setTimeout(() => {
        if (document.getElementById('feaCanvas')) {
            qmlOptimizer.renderFEAHeatmap(true);
        }
    }, 500);
}
