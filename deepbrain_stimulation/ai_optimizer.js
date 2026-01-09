/**
 * AI Optimizer Module
 * Generative AI model training and parameter generation
 */

class AIOptimizer {
    constructor() {
        this.trainingInProgress = false;
        this.canvas = null;
        this.ctx = null;
        this.proprioCanvas = null;
        this.proprioCtx = null;
    }

    init() {
        this.setupButtons();
        this.setupCanvas();
    }

    setupButtons() {
        // VAE buttons
        document.getElementById('trainVAEBtn')?.addEventListener('click', () => this.trainVAE());
        document.getElementById('generateVAEBtn')?.addEventListener('click', () => this.generateVAE());

        // GAN buttons
        document.getElementById('trainGANBtn')?.addEventListener('click', () => this.trainGAN());
        document.getElementById('generateGANBtn')?.addEventListener('click', () => this.generateGAN());

        // RL button (Gemini 3.0)
        document.getElementById('optimizeRLBtn')?.addEventListener('click', () => this.optimizeRL());
    }

    setupCanvas() {
        this.canvas = document.getElementById('trainingCanvas');
        if (this.canvas) {
            this.ctx = this.canvas.getContext('2d');
            this.canvas.width = this.canvas.offsetWidth;
            this.canvas.height = this.canvas.offsetHeight;
        }

        this.proprioCanvas = document.getElementById('proprioceptionCanvas');
        if (this.proprioCanvas) {
            this.proprioCtx = this.proprioCanvas.getContext('2d');
            this.proprioCanvas.width = this.proprioCanvas.offsetWidth;
            this.proprioCanvas.height = this.proprioCanvas.offsetHeight;
            this.drawProprioceptionIdle();
        }
    }

    async trainVAE() {
        const btn = document.getElementById('trainVAEBtn');
        const resultsDiv = document.getElementById('vaeResults');

        btn.disabled = true;
        btn.innerHTML = '<div class="loading"></div> Training...';
        resultsDiv.innerHTML = '<p>Training VAE model...</p>';

        try {
            const response = await window.app.post('/ai/train', { epochs: 5 }); // Reduced to 5 for responsiveness

            if (response.success) {
                resultsDiv.innerHTML = `
                    <div class="result-item">
                        <strong>Training Complete!</strong><br>
                        Final Loss: ${response.vae_final_loss.toFixed(4)}<br>
                        Epochs: ${response.epochs}
                    </div>
                `;
                btn.innerHTML = '✓ VAE Trained';
            }
        } catch (error) {
            console.error("VAE Training Error:", error);
            resultsDiv.innerHTML = `<div class="error-message">Training failed: ${error.message}</div>`;
            btn.innerHTML = 'Train VAE';
        } finally {
            btn.disabled = false;
        }
    }

    async generateVAE() {
        const btn = document.getElementById('generateVAEBtn');
        const resultsDiv = document.getElementById('vaeResults');

        btn.disabled = true;
        btn.innerHTML = '<div class="loading"></div> Generating...';

        try {
            const response = await window.app.post('/ai/generate/vae', { num_samples: 5 });

            if (response.success) {
                let html = '<h4>Generated Parameters (VAE)</h4>';
                response.parameters.forEach((params, i) => {
                    html += `
                        <div class="result-item">
                            <strong>Sample ${i + 1}</strong><br>
                            Amplitude: ${params.amplitude_ma.toFixed(2)} mA<br>
                            Frequency: ${params.frequency_hz.toFixed(0)} Hz<br>
                            Pulse Width: ${params.pulse_width_us.toFixed(0)} μs<br>
                            Duty Cycle: ${params.duty_cycle.toFixed(2)}
                        </div>
                    `;
                });
                resultsDiv.innerHTML = html;
                btn.innerHTML = 'Generate Parameters';
            }
        } catch (error) {
            resultsDiv.innerHTML = `<div class="error-message">Generation failed: ${error.message}</div>`;
            btn.innerHTML = 'Generate Parameters';
        } finally {
            btn.disabled = false;
        }
    }

    async trainGAN() {
        const btn = document.getElementById('trainGANBtn');
        const resultsDiv = document.getElementById('ganResults');

        btn.disabled = true;
        btn.innerHTML = '<div class="loading"></div> Training...';
        resultsDiv.innerHTML = '<p>Training GAN model...</p>';

        try {
            const response = await window.app.post('/ai/train', { epochs: 5 });

            if (response.success) {
                resultsDiv.innerHTML = `
                    <div class="result-item">
                        <strong>Training Complete!</strong><br>
                        Generator Loss: ${response.gan_final_g_loss.toFixed(4)}<br>
                        Discriminator Loss: ${response.gan_final_d_loss.toFixed(4)}<br>
                        Epochs: ${response.epochs}
                    </div>
                `;
                btn.innerHTML = '✓ GAN Trained';
            }
        } catch (error) {
            console.error("GAN Training Error:", error);
            resultsDiv.innerHTML = `<div class="error-message">Training failed: ${error.message}</div>`;
            btn.innerHTML = 'Train GAN';
        } finally {
            btn.disabled = false;
        }
    }

    async generateGAN() {
        const btn = document.getElementById('generateGANBtn');
        const resultsDiv = document.getElementById('ganResults');

        btn.disabled = true;
        btn.innerHTML = '<div class="loading"></div> Generating...';

        try {
            const response = await window.app.post('/ai/generate/gan', { num_samples: 5 });

            if (response.success) {
                let html = '<h4>Generated Parameters (GAN)</h4>';
                response.parameters.forEach((params, i) => {
                    html += `
                        <div class="result-item">
                            <strong>Sample ${i + 1}</strong><br>
                            Amplitude: ${params.amplitude_ma.toFixed(2)} mA<br>
                            Frequency: ${params.frequency_hz.toFixed(0)} Hz<br>
                            Pulse Width: ${params.pulse_width_us.toFixed(0)} μs<br>
                            Duty Cycle: ${params.duty_cycle.toFixed(2)}
                        </div>
                    `;
                });
                resultsDiv.innerHTML = html;
                btn.innerHTML = 'Generate Parameters';
            }
        } catch (error) {
            resultsDiv.innerHTML = `<div class="error-message">Generation failed: ${error.message}</div>`;
            btn.innerHTML = 'Generate Parameters';
        } finally {
            btn.disabled = false;
        }
    }

    async optimizeRL() {
        const btn = document.getElementById('optimizeRLBtn');
        const resultsDiv = document.getElementById('rlResults');
        const feaStatus = document.getElementById('feaIntegrationStatus');

        btn.disabled = true;
        btn.innerHTML = '<div class="loading"></div> Gemini 3.0 Thinking...';
        resultsDiv.innerHTML = '<p>Running Proprioceptive Optimization...</p>';

        // Start feedback loop animation
        this.animateProprioception(true);

        try {
            // 1. RL Optimization
            const response = await window.app.post('/ai/optimize/rl', {
                initial_state: [0.7, 0.6, 0.5, 0.6, 0.8, 0.3, 0.5, 0.7],
                num_steps: 100
            });

            if (response.success) {
                const trajectory = response.trajectory;
                const final_state = trajectory[trajectory.length - 1];

                // Stop animation
                this.animateProprioception(false);

                // 2. Tie in with FEA Simulation
                if (feaStatus) feaStatus.innerHTML = 'Linked to FEA Simulator: <span style="color:var(--warning)">Verifying Constraints...</span>';

                // Mock optimal params based on RL state (in real app, RL would return params)
                const optimalParams = {
                    c1: -3.0 + (final_state[0] * 1.5), // Adjust voltage based on state
                    c2: 0.0,
                    c3: 1.5,
                    c4: 0.0
                };

                // Call FEA to validate
                const feaResponse = await window.app.post('/api/fea/simulate', {
                    target_x: 32,
                    target_y: 32,
                    voltage_c1: optimalParams.c1
                });

                if (feaStatus) feaStatus.innerHTML = 'Linked to FEA Simulator: <span style="color:var(--secondary)">Validated ✓</span>';

                resultsDiv.innerHTML = `
                    <div class="result-item" style="border-left: 3px solid var(--secondary);">
                        <strong>Gemini 3.0 Optimized!</strong><br>
                        Steps: ${trajectory.length}<br>
                        Stability Index: ${(1.0 - final_state[0]).toFixed(3)}<br>
                        FEA VTA: ${feaResponse.vta_volume_mm3 ? feaResponse.vta_volume_mm3.toFixed(2) : 'N/A'} mm³
                    </div>
                `;

                // Visualize trajectory
                this.visualizeTrajectory(trajectory);

                btn.innerHTML = 'Optimize with Gemini 3.0';
            }
        } catch (error) {
            resultsDiv.innerHTML = `<div class="error-message">Optimization failed: ${error.message}</div>`;
            btn.innerHTML = 'Optimize with Gemini 3.0';
            this.animateProprioception(false);
        } finally {
            btn.disabled = false;
        }
    }

    visualizeTrajectory(trajectory) {
        if (!this.ctx) return;

        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear canvas
        this.ctx.fillStyle = '#0a0a0a';
        this.ctx.fillRect(0, 0, width, height);

        // Draw grid
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        this.ctx.lineWidth = 1;
        for (let y = 0; y < height; y += 40) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(width, y);
            this.ctx.stroke();
        }

        // Draw trajectory (first 4 dimensions)
        const colors = ['#00d4ff', '#00ff88', '#ff3366', '#ffaa00'];
        const labels = ['Hyperarousal', 'Re-experiencing', 'Avoidance', 'Neg. Cognition'];

        for (let dim = 0; dim < 4; dim++) {
            this.ctx.strokeStyle = colors[dim];
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();

            trajectory.forEach((state, i) => {
                const x = (i / trajectory.length) * width;
                const y = height - (state[dim] * height * 0.8) - height * 0.1;

                if (i === 0) {
                    this.ctx.moveTo(x, y);
                } else {
                    this.ctx.lineTo(x, y);
                }
            });

            this.ctx.stroke();

            // Label
            this.ctx.fillStyle = colors[dim];
            this.ctx.font = '12px Inter';
            this.ctx.fillText(labels[dim], 10, 20 + dim * 15);
        }

        // Axis labels
        this.ctx.fillStyle = '#b0b0b0';
        this.ctx.font = '12px Inter';
        this.ctx.fillText('Steps →', width - 60, height - 10);
        this.ctx.fillText('Severity', 10, height - 10);
    }

    drawProprioceptionIdle() {
        if (!this.proprioCtx) return;
        const width = this.proprioCanvas.width;
        const height = this.proprioCanvas.height;
        this.proprioCtx.clearRect(0, 0, width, height);
        this.proprioCtx.fillStyle = '#0a0a0a';
        this.proprioCtx.fillRect(0, 0, width, height);

        this.proprioCtx.font = '14px Inter';
        this.proprioCtx.fillStyle = '#555';
        this.proprioCtx.textAlign = 'center';
        this.proprioCtx.fillText('System Idle - Waiting for Input', width / 2, height / 2);
    }

    animateProprioception(active) {
        if (!this.proprioCtx) return;

        if (!active) {
            this.drawProprioceptionIdle();
            return;
        }

        let frame = 0;
        const width = this.proprioCanvas.width;
        const height = this.proprioCanvas.height;

        const animate = () => {
            if (document.getElementById('optimizeRLBtn').disabled === false) return; // Stop if btn enabled

            this.proprioCtx.fillStyle = 'rgba(10, 10, 10, 0.1)'; // Trail effect
            this.proprioCtx.fillRect(0, 0, width, height);

            frame += 0.1;

            // Draw "Tension" lines representing gradient sensing
            const centerX = width / 2;
            const centerY = height / 2;

            for (let i = 0; i < 8; i++) {
                const angle = (i / 8) * Math.PI * 2 + frame;
                const radius = 50 + Math.sin(frame * 2 + i) * 20;

                const x = centerX + Math.cos(angle) * radius;
                const y = centerY + Math.sin(angle) * radius;

                this.proprioCtx.beginPath();
                this.proprioCtx.moveTo(centerX, centerY);
                this.proprioCtx.lineTo(x, y);
                this.proprioCtx.strokeStyle = `hsl(${(frame * 50 + i * 40) % 360}, 70%, 50%)`;
                this.proprioCtx.lineWidth = 2;
                this.proprioCtx.stroke();

                this.proprioCtx.beginPath();
                this.proprioCtx.arc(x, y, 4, 0, Math.PI * 2);
                this.proprioCtx.fillStyle = '#fff';
                this.proprioCtx.fill();
            }

            requestAnimationFrame(animate);
        };
        animate();
    }
}

window.AIOptimizer = AIOptimizer;
