/**
 * AI Optimizer Module
 * Generative AI model training and parameter generation
 */

class AIOptimizer {
    constructor() {
        this.trainingInProgress = false;
        this.canvas = null;
        this.ctx = null;
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

        // RL button
        document.getElementById('optimizeRLBtn')?.addEventListener('click', () => this.optimizeRL());
    }

    setupCanvas() {
        this.canvas = document.getElementById('trainingCanvas');
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = this.canvas.offsetHeight;
    }

    async trainVAE() {
        const btn = document.getElementById('trainVAEBtn');
        const resultsDiv = document.getElementById('vaeResults');

        btn.disabled = true;
        btn.innerHTML = '<div class="loading"></div> Training...';
        resultsDiv.innerHTML = '<p>Training VAE model...</p>';

        try {
            const response = await window.app.post('/ai/train', { epochs: 50 });

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
            const response = await window.app.post('/ai/train', { epochs: 50 });

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

        btn.disabled = true;
        btn.innerHTML = '<div class="loading"></div> Optimizing...';
        resultsDiv.innerHTML = '<p>Running RL optimization...</p>';

        try {
            const response = await window.app.post('/ai/optimize/rl', {
                initial_state: [0.7, 0.6, 0.5, 0.6, 0.8, 0.3, 0.5, 0.7],
                num_steps: 100
            });

            if (response.success) {
                const trajectory = response.trajectory;
                const final_state = trajectory[trajectory.length - 1];

                resultsDiv.innerHTML = `
                    <div class="result-item">
                        <strong>Optimization Complete!</strong><br>
                        Steps: ${trajectory.length}<br>
                        Final State: [${final_state.map(v => v.toFixed(2)).join(', ')}]
                    </div>
                `;

                // Visualize trajectory
                this.visualizeTrajectory(trajectory);

                btn.innerHTML = 'Optimize with RL';
            }
        } catch (error) {
            resultsDiv.innerHTML = `<div class="error-message">Optimization failed: ${error.message}</div>`;
            btn.innerHTML = 'Optimize with RL';
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
}

window.AIOptimizer = AIOptimizer;
