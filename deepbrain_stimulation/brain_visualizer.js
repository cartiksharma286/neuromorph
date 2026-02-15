/**
 * 3D Brain Visualizer Module
 * Interactive 3D brain model with target regions
 */

class BrainVisualizer {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.brain = null;
        this.regions = [];
        this.selectedRegion = null;
    }

    async init() {
        if (this.renderer) return; // Already initialized

        const container = document.getElementById('brainCanvas');
        if (!container) return;

        // Setup Three.js scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            container.clientWidth / container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.z = 150;

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(this.renderer.domElement);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        const pointLight = new THREE.PointLight(0x00d4ff, 1);
        pointLight.position.set(50, 50, 50);
        this.scene.add(pointLight);

        // Create brain model
        this.createBrainModel();

        // Load regions
        await this.loadRegions();

        // Animation loop
        this.animate();

        // Mouse controls
        this.setupControls();
    }

    createBrainModel() {
        // Create simplified brain geometry
        const geometry = new THREE.SphereGeometry(50, 32, 32);
        const material = new THREE.MeshPhongMaterial({
            color: 0x2a2a2a,
            transparent: true,
            opacity: 0.6,
            wireframe: false
        });

        this.brain = new THREE.Mesh(geometry, material);
        this.scene.add(this.brain);

        // Add wireframe overlay
        const wireframe = new THREE.WireframeGeometry(geometry);
        const line = new THREE.LineSegments(wireframe);
        line.material.color.setHex(0x00d4ff);
        line.material.opacity = 0.3;
        line.material.transparent = true;
        this.brain.add(line);
    }

    async loadRegions() {
        try {
            const data = await window.app.get('/brain-regions');
            this.displayRegionList(data.regions);
            this.createRegionMarkers(data.regions);
        } catch (error) {
            console.error('Failed to load brain regions:', error);
        }
    }

    displayRegionList(regions) {
        const list = document.getElementById('regionList');
        if (!list) return;

        list.innerHTML = '';
        regions.forEach(region => {
            const item = document.createElement('div');
            item.className = 'region-item';
            item.innerHTML = `
                <div class="region-name">${region.name}</div>
                <div class="region-desc">${region.description}</div>
            `;
            item.addEventListener('click', () => this.selectRegion(region));
            list.appendChild(item);
        });
    }

    createRegionMarkers(regions) {
        regions.forEach(region => {
            const geometry = new THREE.SphereGeometry(5, 16, 16);
            const material = new THREE.MeshPhongMaterial({
                color: 0x00ff88,
                emissive: 0x00ff88,
                emissiveIntensity: 0.5
            });

            const marker = new THREE.Mesh(geometry, material);
            marker.position.set(
                region.coordinates.x,
                region.coordinates.y,
                region.coordinates.z
            );
            marker.userData = region;

            this.scene.add(marker);
            this.regions.push(marker);

            // Add label
            this.createLabel(region.name, marker.position);
        });
    }

    createLabel(text, position) {
        // Simplified label creation
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 256;
        canvas.height = 64;

        context.fillStyle = '#00d4ff';
        context.font = '20px Inter';
        context.fillText(text, 10, 30);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(material);

        sprite.position.copy(position);
        sprite.position.y += 10;
        sprite.scale.set(30, 7.5, 1);

        this.scene.add(sprite);
    }

    selectRegion(region) {
        this.selectedRegion = region;

        // Update UI
        document.querySelectorAll('.region-item').forEach(item => {
            item.classList.remove('active');
        });
        event.currentTarget.classList.add('active');

        // Highlight in 3D
        this.regions.forEach(marker => {
            if (marker.userData.id === region.id) {
                marker.material.emissiveIntensity = 1.0;
                marker.scale.set(1.5, 1.5, 1.5);
            } else {
                marker.material.emissiveIntensity = 0.5;
                marker.scale.set(1, 1, 1);
            }
        });
    }

    setupControls() {
        let isDragging = false;
        let previousMousePosition = { x: 0, y: 0 };

        this.renderer.domElement.addEventListener('mousedown', (e) => {
            isDragging = true;
        });

        this.renderer.domElement.addEventListener('mousemove', (e) => {
            if (isDragging) {
                const deltaMove = {
                    x: e.offsetX - previousMousePosition.x,
                    y: e.offsetY - previousMousePosition.y
                };

                this.brain.rotation.y += deltaMove.x * 0.01;
                this.brain.rotation.x += deltaMove.y * 0.01;
            }

            previousMousePosition = {
                x: e.offsetX,
                y: e.offsetY
            };
        });

        this.renderer.domElement.addEventListener('mouseup', () => {
            isDragging = false;
        });

        // Zoom with mouse wheel
        this.renderer.domElement.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.camera.position.z += e.deltaY * 0.1;
            this.camera.position.z = Math.max(80, Math.min(250, this.camera.position.z));
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Gentle rotation
        if (this.brain) {
            this.brain.rotation.y += 0.001;
        }

        // Pulse region markers
        this.regions.forEach(marker => {
            const scale = 1 + Math.sin(Date.now() * 0.002) * 0.1;
            if (marker.userData.id !== this.selectedRegion?.id) {
                marker.scale.set(scale, scale, scale);
            }
        });

        this.renderer.render(this.scene, this.camera);
    }
}

window.BrainVisualizer = BrainVisualizer;
