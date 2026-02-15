// Abdominal/Prostate RF Coil Design Engine for MRI
class AbdominalEngine {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.rotation = { x: 0.2, y: 0.5 };
        this.autoRotate = false;
        this.scale = 160;

        // Abdominal/Prostate-specific parameters
        this.coilType = 'prostate-combined'; // prostate-combined, endorectal, torso-array, flexible-wrap
        this.channels = 32;
        this.prostateRegion = true;

        // Coil elements and geometry
        this.elements = [];
        this.connections = [];

        // Interaction
        this.isDragging = false;
        this.lastMouse = { x: 0, y: 0 };

        this.setupEventListeners();
        this.generateAbdominalArray(this.coilType);
    }

    // Generate abdominal coil array based on type
    generateAbdominalArray(type) {
        this.coilType = type;
        this.elements = [];
        this.connections = [];

        switch (type) {
            case 'prostate-combined':
                this.generateProstateCombined();
                break;
            case 'endorectal':
                this.generateEndorectal();
                break;
            case 'torso-array':
                this.generateTorsoArray();
                break;
            case 'flexible-wrap':
                this.generateFlexibleWrap();
                break;
            case 'liver':
                this.generateLiverArray();
                break;
            case 'kidney':
                this.generateKidneyArray();
                break;
        }

        this.generateConnections();
    }

    // Generate combined prostate coil (endorectal + external)
    generateProstateCombined() {
        // Endorectal coil (internal, high-resolution)
        const endorectalElements = 4;
        for (let i = 0; i < endorectalElements; i++) {
            const angle = (i / endorectalElements) * Math.PI * 2;
            this.elements.push({
                x: Math.cos(angle) * 0.08,
                y: -0.6,
                z: Math.sin(angle) * 0.08,
                size: 0.06,
                type: 'endorectal',
                active: true,
                channel: i
            });
        }

        // Anterior pelvic array (external, front - most important for prostate)
        const anteriorElements = 12;
        for (let i = 0; i < anteriorElements; i++) {
            const row = Math.floor(i / 4);
            const col = i % 4;
            this.elements.push({
                x: (col - 1.5) * 0.2,
                y: -0.5 + row * 0.15,
                z: 0.65, // Anterior position
                size: 0.09,
                type: 'anterior',
                active: true,
                channel: i + endorectalElements
            });
        }

        // Posterior array (back support)
        const posteriorElements = Math.min(16, this.channels - endorectalElements - anteriorElements);
        for (let i = 0; i < posteriorElements; i++) {
            const row = Math.floor(i / 4);
            const col = i % 4;
            this.elements.push({
                x: (col - 1.5) * 0.22,
                y: -0.5 + row * 0.15,
                z: -0.5, // Posterior position
                size: 0.1,
                type: 'posterior',
                active: i + endorectalElements + anteriorElements < this.channels,
                channel: i + endorectalElements + anteriorElements
            });
        }
    }

    // Generate endorectal coil only
    generateEndorectal() {
        const elements = Math.min(8, this.channels);
        for (let i = 0; i < elements; i++) {
            const angle = (i / elements) * Math.PI * 2;
            const radius = 0.1;
            this.elements.push({
                x: Math.cos(angle) * radius,
                y: -0.6,
                z: Math.sin(angle) * radius,
                size: 0.07,
                type: 'endorectal',
                active: true,
                channel: i
            });
        }
    }

    // Generate torso phased array for larger abdominal coverage
    generateTorsoArray() {
        const rows = 5;
        const cols = 6;

        // Anterior elements
        const anteriorCount = Math.ceil(this.channels / 2);
        for (let i = 0; i < anteriorCount; i++) {
            const row = Math.floor(i / cols);
            const col = i % cols;
            const angle = (col / (cols - 1)) * Math.PI * 0.8 - Math.PI * 0.4;

            this.elements.push({
                x: Math.sin(angle) * 0.7,
                y: (row - 2) * 0.25,
                z: Math.cos(angle) * 0.6 + 0.2,
                size: 0.1,
                type: 'anterior',
                active: true,
                channel: i
            });
        }

        // Posterior elements
        const posteriorCount = this.channels - anteriorCount;
        for (let i = 0; i < posteriorCount; i++) {
            const row = Math.floor(i / cols);
            const col = i % cols;

            this.elements.push({
                x: (col - (cols - 1) / 2) * 0.22,
                y: (row - 2) * 0.25,
                z: -0.55,
                size: 0.11,
                type: 'posterior',
                active: true,
                channel: i + anteriorCount
            });
        }
    }

    // Generate flexible wrap array
    generateFlexibleWrap() {
        const layers = 3;
        const elementsPerLayer = Math.floor(this.channels / layers);

        for (let layer = 0; layer < layers; layer++) {
            for (let i = 0; i < elementsPerLayer; i++) {
                const angle = (i / elementsPerLayer) * Math.PI * 1.5 - Math.PI * 0.75;
                const radius = 0.65;

                if (layer * elementsPerLayer + i < this.channels) {
                    this.elements.push({
                        x: Math.cos(angle) * radius,
                        y: (layer - 1) * 0.35,
                        z: Math.sin(angle) * radius * 0.7,
                        size: 0.09,
                        type: 'flexible',
                        active: true,
                        channel: layer * elementsPerLayer + i
                    });
                }
            }
        }
    }

    // Generate liver-focused array
    generateLiverArray() {
        // Right upper quadrant focused
        const rows = 4;
        const cols = 5;

        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                const idx = row * cols + col;
                if (idx < this.channels) {
                    // Bias towards right side (liver location)
                    const xBias = 0.3;
                    this.elements.push({
                        x: xBias + (col - cols / 2) * 0.18,
                        y: 0.3 + row * 0.2,
                        z: 0.6,
                        size: 0.11,
                        type: 'liver',
                        active: true,
                        channel: idx
                    });
                }
            }
        }
    }

    // Generate kidney-focused array
    generateKidneyArray() {
        // Bilateral kidney coverage
        const elementsPerKidney = Math.floor(this.channels / 2);

        // Left kidney
        for (let i = 0; i < elementsPerKidney; i++) {
            const row = Math.floor(i / 3);
            const col = i % 3;
            this.elements.push({
                x: -0.35 + col * 0.12,
                y: 0.2 + row * 0.2,
                z: -0.3, // Posterior-lateral
                size: 0.1,
                type: 'kidney-left',
                active: true,
                channel: i
            });
        }

        // Right kidney
        for (let i = 0; i < elementsPerKidney; i++) {
            const row = Math.floor(i / 3);
            const col = i % 3;
            this.elements.push({
                x: 0.35 - col * 0.12,
                y: 0.2 + row * 0.2,
                z: -0.3, // Posterior-lateral
                size: 0.1,
                type: 'kidney-right',
                active: i + elementsPerKidney < this.channels,
                channel: i + elementsPerKidney
            });
        }
    }

    // Generate connections between adjacent elements
    generateConnections() {
        this.connections = [];

        for (let i = 0; i < this.elements.length; i++) {
            for (let j = i + 1; j < this.elements.length; j++) {
                const e1 = this.elements[i];
                const e2 = this.elements[j];
                const dist = Math.sqrt(
                    Math.pow(e1.x - e2.x, 2) +
                    Math.pow(e1.y - e2.y, 2) +
                    Math.pow(e1.z - e2.z, 2)
                );

                // Connect nearby elements
                if (dist < 0.25 && e1.active && e2.active) {
                    this.connections.push({
                        from: i,
                        to: j,
                        type: 'neighbor',
                        strength: 1 - (dist / 0.25)
                    });
                }
            }
        }
    }

    // 3D rotation matrices
    rotateX(point, angle) {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        return {
            x: point.x,
            y: point.y * cos - point.z * sin,
            z: point.y * sin + point.z * cos
        };
    }

    rotateY(point, angle) {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        return {
            x: point.x * cos + point.z * sin,
            y: point.y,
            z: -point.x * sin + point.z * cos
        };
    }

    // Project 3D point to 2D canvas
    project(point) {
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;

        return {
            x: centerX + point.x * this.scale,
            y: centerY + point.y * this.scale,
            z: point.z
        };
    }

    // Setup mouse interaction
    setupEventListeners() {
        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            const rect = this.canvas.getBoundingClientRect();
            this.lastMouse = {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top
            };
        });

        this.canvas.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;

            const rect = this.canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const dx = mouseX - this.lastMouse.x;
            const dy = mouseY - this.lastMouse.y;

            this.rotation.y += dx * 0.01;
            this.rotation.x += dy * 0.01;

            this.lastMouse = { x: mouseX, y: mouseY };
            this.render();
        });

        this.canvas.addEventListener('mouseup', () => {
            this.isDragging = false;
        });

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.scale *= e.deltaY > 0 ? 0.95 : 1.05;
            this.scale = Math.max(50, Math.min(400, this.scale));
            this.render();
        });
    }

    // Render the abdominal coil array
    render(options = {}) {
        const {
            showConnections = true,
            showElements = true,
            showAnatomyOutline = true,
            highlightProstate = false
        } = options;

        // Clear canvas with dark background
        this.ctx.fillStyle = '#0a0a14';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Transform all elements
        const transformed = this.elements.map(e => {
            let point = this.rotateX(e, this.rotation.x);
            point = this.rotateY(point, this.rotation.y);
            const projected = this.project(point);
            return { ...e, ...projected };
        });

        // Sort by z-depth
        const sorted = [...transformed].sort((a, b) => a.z - b.z);

        // Draw anatomy outline
        if (showAnatomyOutline) {
            this.drawAnatomyOutline(highlightProstate);
        }

        // Draw connections
        if (showConnections) {
            this.ctx.strokeStyle = 'rgba(99, 102, 241, 0.12)';
            this.ctx.lineWidth = 1;

            for (const conn of this.connections) {
                const e1 = transformed[conn.from];
                const e2 = transformed[conn.to];

                this.ctx.beginPath();
                this.ctx.moveTo(e1.x, e1.y);
                this.ctx.lineTo(e2.x, e2.y);
                this.ctx.globalAlpha = conn.strength * 0.25;
                this.ctx.stroke();
                this.ctx.globalAlpha = 1;
            }
        }

        // Draw coil elements
        if (showElements) {
            for (const elem of sorted) {
                if (!elem.active) continue;

                const radius = elem.size * this.scale;

                // Determine color based on type
                let color;
                switch (elem.type) {
                    case 'endorectal':
                        color = '#ff6b6b';
                        break;
                    case 'anterior':
                        color = '#8b5cf6';
                        break;
                    case 'posterior':
                        color = '#6366f1';
                        break;
                    case 'liver':
                        color = '#f59e0b';
                        break;
                    case 'kidney-left':
                    case 'kidney-right':
                        color = '#10b981';
                        break;
                    case 'flexible':
                        color = '#3b82f6';
                        break;
                    default:
                        color = '#8b5cf6';
                }

                // Element circle
                this.ctx.beginPath();
                this.ctx.arc(elem.x, elem.y, radius, 0, Math.PI * 2);

                // Gradient fill
                const gradient = this.ctx.createRadialGradient(
                    elem.x, elem.y, 0,
                    elem.x, elem.y, radius
                );
                gradient.addColorStop(0, color);
                gradient.addColorStop(1, color + '70');
                this.ctx.fillStyle = gradient;
                this.ctx.fill();

                // Glow for elements in front or endorectal
                if (elem.z > 0 || elem.type === 'endorectal') {
                    this.ctx.shadowBlur = elem.type === 'endorectal' ? 20 : 12;
                    this.ctx.shadowColor = color;
                    this.ctx.fill();
                    this.ctx.shadowBlur = 0;
                }

                // Border
                this.ctx.strokeStyle = color;
                this.ctx.lineWidth = 2;
                this.ctx.stroke();

                // Channel number
                this.ctx.fillStyle = '#ffffff';
                this.ctx.font = `${Math.max(7, radius * 0.45)}px Inter, sans-serif`;
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                this.ctx.fillText(elem.channel + 1, elem.x, elem.y);
            }
        }
    }

    // Draw anatomical outline
    drawAnatomyOutline(highlightProstate) {
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;

        // Abdominal/pelvic outline
        this.ctx.strokeStyle = 'rgba(148, 163, 184, 0.15)';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);

        this.ctx.beginPath();
        this.ctx.ellipse(
            centerX,
            centerY,
            this.scale * 0.7,
            this.scale * 1.1,
            0,
            0,
            Math.PI * 2
        );
        this.ctx.stroke();

        // Prostate region (if applicable)
        if (highlightProstate && (this.coilType === 'prostate-combined' || this.coilType === 'endorectal')) {
            this.ctx.strokeStyle = 'rgba(255, 107, 107, 0.4)';
            this.ctx.fillStyle = 'rgba(255, 107, 107, 0.08)';
            this.ctx.beginPath();
            this.ctx.arc(
                centerX,
                centerY + this.scale * 0.6,
                this.scale * 0.15,
                0,
                Math.PI * 2
            );
            this.ctx.fill();
            this.ctx.stroke();

            // Label
            this.ctx.fillStyle = 'rgba(255, 107, 107, 0.6)';
            this.ctx.font = '11px Inter, sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('Prostate', centerX, centerY + this.scale * 0.8);
        }

        this.ctx.setLineDash([]);
    }

    // Animation loop
    animate() {
        if (this.autoRotate) {
            this.rotation.y += 0.005;
            this.render();
        }
        requestAnimationFrame(() => this.animate());
    }

    // Reset view
    resetView() {
        this.rotation = { x: 0.2, y: 0.5 };
        this.scale = 160;
        this.render();
    }

    // Download canvas
    downloadImage(filename = 'abdominal-coil-design.png') {
        const link = document.createElement('a');
        link.download = filename;
        link.href = this.canvas.toDataURL();
        link.click();
    }

    // Get design specifications
    getSpecifications() {
        const activeElements = this.elements.filter(e => e.active).length;

        return {
            coilType: this.coilType,
            totalElements: this.elements.length,
            activeElements: activeElements,
            channels: this.channels,
            coverage: Math.round((activeElements / this.channels) * 100),
            elementsPerType: this.elements.reduce((acc, e) => {
                acc[e.type] = (acc[e.type] || 0) + 1;
                return acc;
            }, {})
        };
    }
}

// Initialize engine
let abdominalEngine = null;

function initAbdominalEngine() {
    const canvas = document.getElementById('abdominalCanvas');
    if (!canvas) return;

    abdominalEngine = new AbdominalEngine(canvas);
    abdominalEngine.render({
        showConnections: true,
        showElements: true,
        showAnatomyOutline: true,
        highlightProstate: true
    });
    abdominalEngine.animate();
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AbdominalEngine };
}
