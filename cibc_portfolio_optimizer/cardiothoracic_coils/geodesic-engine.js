// Geodesic Sphere Engine for MRI Head Coil Design
class GeodesicEngine {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.rotation = { x: 0.3, y: 0.4 };
        this.autoRotate = false;
        this.scale = 200;
        this.frequency = 2;
        
        // Geodesic sphere data
        this.vertices = [];
        this.edges = [];
        this.faces = [];
        
        // Interaction
        this.isDragging = false;
        this.lastMouse = { x: 0, y: 0 };
        
        this.setupEventListeners();
        this.generateGeodesic(this.frequency);
    }
    
    // Create initial icosahedron
    createIcosahedron() {
        const t = (1 + Math.sqrt(5)) / 2; // Golden ratio
        
        // 12 vertices of icosahedron
        const vertices = [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ];
        
        // Normalize vertices to unit sphere
        return vertices.map(v => this.normalize(v));
    }
    
    // Normalize vector to unit length
    normalize(v) {
        const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        return [v[0] / len, v[1] / len, v[2] / len];
    }
    
    // Midpoint between two vertices on sphere
    getMidpoint(v1, v2) {
        return this.normalize([
            (v1[0] + v2[0]) / 2,
            (v1[1] + v2[1]) / 2,
            (v1[2] + v2[2]) / 2
        ]);
    }
    
    // Generate geodesic sphere by subdividing icosahedron
    generateGeodesic(frequency) {
        this.frequency = frequency;
        this.vertices = this.createIcosahedron();
        
        // Icosahedron faces (20 triangular faces)
        this.faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ];
        
        // Subdivide faces
        for (let i = 0; i < frequency; i++) {
            this.subdivideFaces();
        }
        
        // Generate edges from faces
        this.generateEdges();
        
        return this.vertices.length;
    }
    
    // Subdivide each face into 4 smaller faces
    subdivideFaces() {
        const newFaces = [];
        const midpointCache = new Map();
        
        const getMidpointIndex = (i1, i2) => {
            const key = i1 < i2 ? `${i1},${i2}` : `${i2},${i1}`;
            
            if (midpointCache.has(key)) {
                return midpointCache.get(key);
            }
            
            const midpoint = this.getMidpoint(this.vertices[i1], this.vertices[i2]);
            const index = this.vertices.length;
            this.vertices.push(midpoint);
            midpointCache.set(key, index);
            return index;
        };
        
        for (const face of this.faces) {
            const [v1, v2, v3] = face;
            const a = getMidpointIndex(v1, v2);
            const b = getMidpointIndex(v2, v3);
            const c = getMidpointIndex(v3, v1);
            
            newFaces.push([v1, a, c]);
            newFaces.push([v2, b, a]);
            newFaces.push([v3, c, b]);
            newFaces.push([a, b, c]);
        }
        
        this.faces = newFaces;
    }
    
    // Generate edges from faces
    generateEdges() {
        const edgeSet = new Set();
        
        for (const face of this.faces) {
            const [v1, v2, v3] = face;
            const edges = [
                [v1, v2].sort((a, b) => a - b),
                [v2, v3].sort((a, b) => a - b),
                [v3, v1].sort((a, b) => a - b)
            ];
            
            for (const edge of edges) {
                edgeSet.add(`${edge[0]},${edge[1]}`);
            }
        }
        
        this.edges = Array.from(edgeSet).map(e => e.split(',').map(Number));
    }
    
    // 3D rotation matrices
    rotateX(point, angle) {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        return [
            point[0],
            point[1] * cos - point[2] * sin,
            point[1] * sin + point[2] * cos
        ];
    }
    
    rotateY(point, angle) {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        return [
            point[0] * cos + point[2] * sin,
            point[1],
            -point[0] * sin + point[2] * cos
        ];
    }
    
    // Project 3D point to 2D canvas
    project(point) {
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        return {
            x: centerX + point[0] * this.scale,
            y: centerY + point[1] * this.scale,
            z: point[2]
        };
    }
    
    // Setup mouse interaction
    setupEventListeners() {
        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastMouse = { x: e.clientX, y: e.clientY };
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            
            const dx = e.clientX - this.lastMouse.x;
            const dy = e.clientY - this.lastMouse.y;
            
            this.rotation.y += dx * 0.01;
            this.rotation.x += dy * 0.01;
            
            this.lastMouse = { x: e.clientX, y: e.clientY };
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
    
    // Render the geodesic sphere
    render(options = {}) {
        const {
            showWireframe = true,
            showElements = true,
            numActiveElements = 32
        } = options;
        
        this.ctx.fillStyle = '#1a1a24';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Transform vertices
        const transformed = this.vertices.map(v => {
            let point = this.rotateX(v, this.rotation.x);
            point = this.rotateY(point, this.rotation.y);
            return this.project(point);
        });
        
        // Draw wireframe
        if (showWireframe) {
            this.ctx.strokeStyle = 'rgba(99, 102, 241, 0.3)';
            this.ctx.lineWidth = 1;
            
            for (const edge of this.edges) {
                const p1 = transformed[edge[0]];
                const p2 = transformed[edge[1]];
                
                this.ctx.beginPath();
                this.ctx.moveTo(p1.x, p1.y);
                this.ctx.lineTo(p2.x, p2.y);
                this.ctx.stroke();
            }
        }
        
        // Draw coil elements (vertices)
        if (showElements) {
            // Sort vertices by z-depth for proper rendering
            const sortedIndices = transformed
                .map((p, i) => ({ index: i, z: p.z }))
                .sort((a, b) => a.z - b.z);
            
            for (let i = 0; i < sortedIndices.length; i++) {
                const { index, z } = sortedIndices[i];
                const p = transformed[index];
                const isActive = i < numActiveElements;
                
                // Draw element
                this.ctx.beginPath();
                this.ctx.arc(p.x, p.y, isActive ? 6 : 3, 0, Math.PI * 2);
                
                if (isActive) {
                    const gradient = this.ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 6);
                    gradient.addColorStop(0, '#8b5cf6');
                    gradient.addColorStop(1, '#6366f1');
                    this.ctx.fillStyle = gradient;
                } else {
                    this.ctx.fillStyle = 'rgba(148, 163, 184, 0.4)';
                }
                
                this.ctx.fill();
                
                // Add glow for active elements
                if (isActive) {
                    this.ctx.shadowBlur = 10;
                    this.ctx.shadowColor = '#6366f1';
                    this.ctx.fill();
                    this.ctx.shadowBlur = 0;
                }
            }
        }
        
        // Draw center indicator
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        this.ctx.strokeStyle = 'rgba(99, 102, 241, 0.2)';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, this.scale, 0, Math.PI * 2);
        this.ctx.stroke();
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
        this.scale = 200;
        this.render();
    }
    
    // Download canvas as image
    downloadImage(filename = 'geodesic-coil-design.png') {
        const link = document.createElement('a');
        link.download = filename;
        link.href = this.canvas.toDataURL();
        link.click();
    }
}

// Initialize geodesic engine when DOM is loaded
let geodesicEngine = null;

function initGeodesicEngine() {
    const canvas = document.getElementById('geodesicCanvas');
    if (!canvas) return;
    
    geodesicEngine = new GeodesicEngine(canvas);
    geodesicEngine.render({
        showWireframe: true,
        showElements: true,
        numActiveElements: 32
    });
    geodesicEngine.animate();
}

// Export for use in main app
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { GeodesicEngine };
}
