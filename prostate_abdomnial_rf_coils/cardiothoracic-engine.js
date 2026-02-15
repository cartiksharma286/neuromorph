// Cardiothoracic Coil Design Engine for MRI
class CardiothoracicEngine {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.rotation = { x: 0.3, y: 0.4 };
        this.autoRotate = false;
        this.scale = 180;

        // Cardiothoracic-specific parameters
        this.coilType = 'cardiac'; // cardiac, thoracic, combined
        this.arrayConfig = 'anterior-posterior'; // anterior-posterior, circumferential, flexible
        this.channels = 32;
        this.anatomyRegion = 'heart'; // heart, chest, spine

        // Coil elements and geometry
        this.elements = [];
        this.connections = [];

        // Interaction
        this.isDragging = false;
        this.lastMouse = { x: 0, y: 0 };

        this.setupEventListeners();
        this.generateCardiothoracicArray(this.coilType, this.arrayConfig);
    }

    // Generate cardiothoracic coil array based on type
    generateCardiothoracicArray(type, config) {
        this.coilType = type;
        this.arrayConfig = config;
        this.elements = [];
        this.connections = [];

        switch (type) {
            case 'cardiac':
                this.generateCardiacArray(config);
                break;
            case 'thoracic':
                this.generateThoracicArray(config);
                break;
            case 'combined':
                this.generateCombinedArray(config);
                break;
            case 'spine':
                this.generateSpineArray(config);
                break;
        }

        this.generateConnections();
    }

    // Generate cardiac-specific array (focused on heart region)
    generateCardiacArray(config) {
        if (config === 'anterior-posterior') {
            // Anterior elements (front of chest)
            const anteriorElements = 16;
            for (let i = 0; i < anteriorElements; i++) {
                const row = Math.floor(i / 4);
                const col = i % 4;
                this.elements.push({
                    x: (col - 1.5) * 0.25,
                    y: (row - 1.5) * 0.25,
                    z: 0.8, // Front of torso
                    size: 0.12,
                    type: 'anterior',
                    active: true,
                    channel: i
                });
            }

            // Posterior elements (back)
            const posteriorElements = 16;
            for (let i = 0; i < posteriorElements; i++) {
                const row = Math.floor(i / 4);
                const col = i % 4;
                this.elements.push({
                    x: (col - 1.5) * 0.25,
                    y: (row - 1.5) * 0.25,
                    z: -0.6, // Back of torso
                    size: 0.12,
                    type: 'posterior',
                    active: true,
                    channel: i + anteriorElements
                });
            }
        } else if (config === 'circumferential') {
            // Wrap elements around torso
            const numElements = this.channels;
            const layers = 3; // vertical layers
            for (let layer = 0; layer < layers; layer++) {
                const elementsPerLayer = Math.floor(numElements / layers);
                for (let i = 0; i < elementsPerLayer; i++) {
                    const angle = (i / elementsPerLayer) * Math.PI * 2;
                    const radius = 0.7;
                    this.elements.push({
                        x: Math.cos(angle) * radius,
                        y: (layer - 1) * 0.3,
                        z: Math.sin(angle) * radius,
                        size: 0.1,
                        type: 'circumferential',
                        active: true,
                        channel: layer * elementsPerLayer + i
                    });
                }
            }
        } else if (config === 'flexible') {
            // Flexible array that conforms to body
            const rows = 4;
            const cols = 8;
            for (let row = 0; row < rows; row++) {
                for (let col = 0; col < cols; col++) {
                    const angle = (col / cols) * Math.PI - Math.PI / 2;
                    const radius = 0.6 + Math.abs(Math.sin(angle)) * 0.2;
                    this.elements.push({
                        x: Math.cos(angle) * radius,
                        y: (row - 1.5) * 0.25,
                        z: Math.sin(angle) * radius,
                        size: 0.08,
                        type: 'flexible',
                        active: row * cols + col < this.channels,
                        channel: row * cols + col
                    });
                }
            }
        }
    }

    // Generate thoracic array (broader chest coverage)
    generateThoracicArray(config) {
        const rows = 5;
        const cols = 6;

        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                const angle = (col / (cols - 1)) * Math.PI - Math.PI / 2;
                const radius = 0.7;
                const idx = row * cols + col;

                if (idx < this.channels) {
                    this.elements.push({
                        x: Math.cos(angle) * radius,
                        y: (row - 2) * 0.3,
                        z: Math.sin(angle) * radius * 0.8,
                        size: 0.1,
                        type: 'thoracic',
                        active: true,
                        channel: idx
                    });
                }
            }
        }
    }

    // Generate combined cardiac + thoracic array
    generateCombinedArray(config) {
        // Lower region: cardiac elements (smaller, denser)
        for (let i = 0; i < 16; i++) {
            const row = Math.floor(i / 4);
            const col = i % 4;
            this.elements.push({
                x: (col - 1.5) * 0.2,
                y: -0.4 + row * 0.15,
                z: 0.7,
                size: 0.08,
                type: 'cardiac',
                active: true,
                channel: i
            });
        }

        // Upper region: thoracic elements (larger, broader coverage)
        for (let i = 0; i < 16; i++) {
            const row = Math.floor(i / 4);
            const col = i % 4;
            this.elements.push({
                x: (col - 1.5) * 0.25,
                y: 0.3 + row * 0.2,
                z: 0.6,
                size: 0.11,
                type: 'thoracic',
                active: i + 16 < this.channels,
                channel: i + 16
            });
        }
    }

    // Generate spine array (posterior focus)
    generateSpineArray(config) {
        const spineElements = Math.min(this.channels, 24);
        const cols = 3;
        const rows = Math.ceil(spineElements / cols);

        for (let i = 0; i < spineElements; i++) {
            const row = Math.floor(i / cols);
            const col = i % cols;
            this.elements.push({
                x: (col - 1) * 0.15,
                y: (row - rows / 2 + 0.5) * 0.25,
                z: -0.7, // Posterior position
                size: 0.09,
                type: 'spine',
                active: true,
                channel: i
            });
        }
    }

    // Generate connections between adjacent elements
    generateConnections() {
        this.connections = [];

        // Create connections based on proximity
        for (let i = 0; i < this.elements.length; i++) {
            for (let j = i + 1; j < this.elements.length; j++) {
                const e1 = this.elements[i];
                const e2 = this.elements[j];
                const dist = Math.sqrt(
                    Math.pow(e1.x - e2.x, 2) +
                    Math.pow(e1.y - e2.y, 2) +
                    Math.pow(e1.z - e2.z, 2)
                );

                // Connect nearby elements (for overlap decoupling)
                if (dist < 0.3 && e1.active && e2.active) {
                    this.connections.push({
                        from: i,
                        to: j,
                        type: 'neighbor',
                        strength: 1 - (dist / 0.3)
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

    // Render the cardiothoracic coil array
    render(options = {}) {
        const {
            showConnections = true,
            showElements = true,
            showAnatomyOutline = true,
            highlightCardiac = false
        } = options;

        // Clear canvas
        this.ctx.fillStyle = '#0f0f1a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Transform all elements
        const transformed = this.elements.map(e => {
            let point = this.rotateX(e, this.rotation.x);
            point = this.rotateY(point, this.rotation.y);
            const projected = this.project(point);
            return { ...e, ...projected };
        });

        // Sort by z-depth for proper rendering
        const sorted = [...transformed].sort((a, b) => a.z - b.z);

        // Draw anatomy outline
        if (showAnatomyOutline) {
            this.drawAnatomyOutline();
        }

        // Draw connections (decoupling network visualization)
        if (showConnections) {
            this.ctx.strokeStyle = 'rgba(99, 102, 241, 0.15)';
            this.ctx.lineWidth = 1;

            for (const conn of this.connections) {
                const e1 = transformed[conn.from];
                const e2 = transformed[conn.to];

                this.ctx.beginPath();
                this.ctx.moveTo(e1.x, e1.y);
                this.ctx.lineTo(e2.x, e2.y);
                this.ctx.globalAlpha = conn.strength * 0.3;
                this.ctx.stroke();
                this.ctx.globalAlpha = 1;
            }
        }

        // Draw coil elements
        if (showElements) {
            for (const elem of sorted) {
                if (!elem.active) continue;

                const radius = elem.size * this.scale;

                // Element circle
                this.ctx.beginPath();
                this.ctx.arc(elem.x, elem.y, radius, 0, Math.PI * 2);

                // Color based on type and highlighting
                let color;
                if (highlightCardiac && elem.type === 'cardiac') {
                    color = '#ff6b9d';
                } else if (elem.type === 'anterior') {
                    color = '#8b5cf6';
                } else if (elem.type === 'posterior') {
                    color = '#6366f1';
                } else if (elem.type === 'cardiac') {
                    color = '#ec4899';
                } else if (elem.type === 'thoracic') {
                    color = '#3b82f6';
                } else if (elem.type === 'spine') {
                    color = '#10b981';
                } else {
                    color = '#8b5cf6';
                }

                // Gradient fill
                const gradient = this.ctx.createRadialGradient(
                    elem.x, elem.y, 0,
                    elem.x, elem.y, radius
                );
                gradient.addColorStop(0, color);
                gradient.addColorStop(1, color + '80');
                this.ctx.fillStyle = gradient;
                this.ctx.fill();

                // Glow effect for elements in front
                if (elem.z > 0) {
                    this.ctx.shadowBlur = 15;
                    this.ctx.shadowColor = color;
                    this.ctx.fill();
                    this.ctx.shadowBlur = 0;
                }

                // Element border
                this.ctx.strokeStyle = color;
                this.ctx.lineWidth = 2;
                this.ctx.stroke();

                // Channel number
                this.ctx.fillStyle = '#ffffff';
                this.ctx.font = `${Math.max(8, radius * 0.5)}px Inter, sans-serif`;
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                this.ctx.fillText(elem.channel + 1, elem.x, elem.y);
            }
        }
    }

    // Draw anatomical outline (torso/chest)
    drawAnatomyOutline() {
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;

        // Torso outline (ellipse)
        this.ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);

        this.ctx.beginPath();
        this.ctx.ellipse(
            centerX,
            centerY,
            this.scale * 0.7,
            this.scale * 1.2,
            0,
            0,
            Math.PI * 2
        );
        this.ctx.stroke();

        // Heart region indicator (if cardiac coil)
        if (this.coilType === 'cardiac' || this.coilType === 'combined') {
            this.ctx.strokeStyle = 'rgba(236, 72, 153, 0.3)';
            this.ctx.beginPath();
            this.ctx.ellipse(
                centerX - this.scale * 0.15,
                centerY - this.scale * 0.2,
                this.scale * 0.25,
                this.scale * 0.3,
                -0.3,
                0,
                Math.PI * 2
            );
            this.ctx.stroke();
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
        this.rotation = { x: 0.3, y: 0.4 };
        this.scale = 180;
        this.render();
    }

    // Download canvas as image
    downloadImage(filename = 'cardiothoracic-coil-design.png') {
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
            arrayConfig: this.arrayConfig,
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

// Initialize engine when DOM is loaded
let cardiothoracicEngine = null;

function initCardiothoracicEngine() {
    const canvas = document.getElementById('cardiothoracicCanvas');
    if (!canvas) return;

    cardiothoracicEngine = new CardiothoracicEngine(canvas);
    cardiothoracicEngine.render({
        showConnections: true,
        showElements: true,
        showAnatomyOutline: true,
        highlightCardiac: false
    });
    cardiothoracicEngine.animate();
}

// Export for use in main app
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CardiothoracicEngine };
}
