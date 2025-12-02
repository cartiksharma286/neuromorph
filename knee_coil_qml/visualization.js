/**
 * 3D Visualization Engine using Three.js
 * Renders coil geometries, B1 field maps, and quantum circuit diagrams
 */

class Visualizer {
    constructor() {
        this.coilScene = null;
        this.fieldScene = null;
        this.coilRenderer = null;
        this.fieldRenderer = null;
        this.coilCamera = null;
        this.fieldCamera = null;
        this.coilControls = null;
        this.fieldControls = null;
        this.animationId = null;
    }

    /**
     * Initialize 3D coil visualization
     */
    initCoilVisualization(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        // Scene setup
        this.coilScene = new THREE.Scene();
        this.coilScene.background = new THREE.Color(0x151933);

        // Camera setup
        this.coilCamera = new THREE.PerspectiveCamera(
            60,
            canvas.clientWidth / canvas.clientHeight,
            1,
            1000
        );
        this.coilCamera.position.set(150, 150, 150);
        this.coilCamera.lookAt(0, 0, 0);

        // Renderer setup
        this.coilRenderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        this.coilRenderer.setSize(canvas.clientWidth, canvas.clientHeight);
        this.coilRenderer.setPixelRatio(window.devicePixelRatio);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.coilScene.add(ambientLight);

        const directionalLight1 = new THREE.DirectionalLight(0x667eea, 0.8);
        directionalLight1.position.set(100, 100, 100);
        this.coilScene.add(directionalLight1);

        const directionalLight2 = new THREE.DirectionalLight(0x00d4ff, 0.6);
        directionalLight2.position.set(-100, -100, 50);
        this.coilScene.add(directionalLight2);

        // Add coordinate axes
        const axesHelper = new THREE.AxesHelper(100);
        this.coilScene.add(axesHelper);

        // Add grid
        const gridHelper = new THREE.GridHelper(200, 20, 0x667eea, 0x2a3f5f);
        this.coilScene.add(gridHelper);

        // Orbit controls (requires OrbitControls.js)
        if (typeof THREE.OrbitControls !== 'undefined') {
            this.coilControls = new THREE.OrbitControls(this.coilCamera, this.coilRenderer.domElement);
            this.coilControls.enableDamping = true;
            this.coilControls.dampingFactor = 0.05;
        }

        this.animate();
    }

    /**
     * Initialize field map visualization
     */
    initFieldVisualization(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        this.fieldScene = new THREE.Scene();
        this.fieldScene.background = new THREE.Color(0x151933);

        this.fieldCamera = new THREE.PerspectiveCamera(
            60,
            canvas.clientWidth / canvas.clientHeight,
            1,
            1000
        );
        this.fieldCamera.position.set(120, 120, 120);
        this.fieldCamera.lookAt(0, 0, 0);

        this.fieldRenderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        this.fieldRenderer.setSize(canvas.clientWidth, canvas.clientHeight);
        this.fieldRenderer.setPixelRatio(window.devicePixelRatio);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.fieldScene.add(ambientLight);

        if (typeof THREE.OrbitControls !== 'undefined') {
            this.fieldControls = new THREE.OrbitControls(this.fieldCamera, this.fieldRenderer.domElement);
            this.fieldControls.enableDamping = true;
        }
    }

    /**
     * Render coil elements
     */
    renderCoil(coilElements) {
        // Clear previous coil meshes
        while (this.coilScene.children.length > 0) {
            const obj = this.coilScene.children[0];
            this.coilScene.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
        }

        // Re-add lights and helpers
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.coilScene.add(ambientLight);

        const directionalLight1 = new THREE.DirectionalLight(0x667eea, 0.8);
        directionalLight1.position.set(100, 100, 100);
        this.coilScene.add(directionalLight1);

        const directionalLight2 = new THREE.DirectionalLight(0x00d4ff, 0.6);
        directionalLight2.position.set(-100, -100, 50);
        this.coilScene.add(directionalLight2);

        const axesHelper = new THREE.AxesHelper(100);
        this.coilScene.add(axesHelper);

        const gridHelper = new THREE.GridHelper(200, 20, 0x667eea, 0x2a3f5f);
        this.coilScene.add(gridHelper);

        // Render each coil element
        coilElements.forEach((element, index) => {
            const hue = index / coilElements.length;
            const color = new THREE.Color().setHSL(hue, 0.8, 0.6);

            // Render wire loops
            for (const loop of element.wireLoops) {
                const points = loop.map(p => new THREE.Vector3(p.x, p.z, p.y));
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const material = new THREE.LineBasicMaterial({
                    color: color,
                    linewidth: 2
                });
                const line = new THREE.Line(geometry, material);
                this.coilScene.add(line);

                // Add glow effect with tube geometry
                const tubeGeometry = new THREE.TubeGeometry(
                    new THREE.CatmullRomCurve3(points),
                    64,
                    0.8,
                    8,
                    true
                );
                const tubeMaterial = new THREE.MeshStandardMaterial({
                    color: color,
                    emissive: color,
                    emissiveIntensity: 0.5,
                    metalness: 0.8,
                    roughness: 0.2
                });
                const tube = new THREE.Mesh(tubeGeometry, tubeMaterial);
                this.coilScene.add(tube);
            }

            // Add element label
            const labelDiv = document.createElement('div');
            labelDiv.className = 'coil-label';
            labelDiv.textContent = `Ch${index + 1}`;
            labelDiv.style.position = 'absolute';
            labelDiv.style.color = `hsl(${hue * 360}, 80%, 70%)`;
        });
    }

    /**
     * Render B1 field map as heatmap
     */
    renderFieldMap(fieldMap) {
        // Clear previous field visualization
        while (this.fieldScene.children.length > 0) {
            const obj = this.fieldScene.children[0];
            this.fieldScene.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
        }

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.fieldScene.add(ambientLight);

        // Find min/max field magnitude for color scaling
        const magnitudes = fieldMap.map(f => f.magnitude);
        const maxMag = Math.max(...magnitudes);
        const minMag = Math.min(...magnitudes);

        // Create point cloud for field visualization
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];

        fieldMap.forEach(field => {
            positions.push(field.position.x, field.position.z, field.position.y);

            // Color based on field strength (blue = low, cyan = medium, magenta = high)
            const normalized = (field.magnitude - minMag) / (maxMag - minMag + 1e-10);
            const color = new THREE.Color();

            if (normalized < 0.5) {
                // Blue to cyan
                color.setHSL(0.6 - normalized * 0.2, 1.0, 0.5);
            } else {
                // Cyan to magenta
                color.setHSL(0.8 - (normalized - 0.5) * 0.6, 1.0, 0.6);
            }

            colors.push(color.r, color.g, color.b);
        });

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 3,
            vertexColors: true,
            opacity: 0.8,
            transparent: true
        });

        const points = new THREE.Points(geometry, material);
        this.fieldScene.add(points);

        // Add isosurface at average field strength
        const avgMag = magnitudes.reduce((a, b) => a + b, 0) / magnitudes.length;
        this.addIsosurface(fieldMap, avgMag);
    }

    /**
     * Add isosurface for field visualization
     */
    addIsosurface(fieldMap, threshold) {
        // Simplified isosurface using spheres at threshold points
        const geometry = new THREE.SphereGeometry(2, 8, 8);
        const material = new THREE.MeshStandardMaterial({
            color: 0x00d4ff,
            emissive: 0x00d4ff,
            emissiveIntensity: 0.3,
            transparent: true,
            opacity: 0.4
        });

        const instancedMesh = new THREE.InstancedMesh(
            geometry,
            material,
            fieldMap.filter(f => Math.abs(f.magnitude - threshold) < threshold * 0.1).length
        );

        let idx = 0;
        const matrix = new THREE.Matrix4();

        fieldMap.forEach(field => {
            if (Math.abs(field.magnitude - threshold) < threshold * 0.1) {
                matrix.setPosition(field.position.x, field.position.z, field.position.y);
                instancedMesh.setMatrixAt(idx++, matrix);
            }
        });

        this.fieldScene.add(instancedMesh);
    }

    /**
     * Render quantum circuit diagram
     */
    renderQuantumCircuit(circuit, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = '';

        const width = container.clientWidth;
        const height = Math.max(300, circuit.qubits * 50 + 60);
        const qubitSpacing = height / (circuit.qubits + 1);
        const layerSpacing = width / (circuit.depth + 2);

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', width);
        svg.setAttribute('height', height);
        svg.classList.add('circuit-svg');

        // Draw qubit lines
        for (let q = 0; q < circuit.qubits; q++) {
            const y = qubitSpacing * (q + 1);

            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', 40);
            line.setAttribute('y1', y);
            line.setAttribute('x2', width - 20);
            line.setAttribute('y2', y);
            line.setAttribute('stroke', '#667eea');
            line.setAttribute('stroke-width', '2');
            svg.appendChild(line);

            // Qubit label
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', 20);
            text.setAttribute('y', y + 5);
            text.setAttribute('fill', '#a0aec0');
            text.setAttribute('font-size', '14');
            text.textContent = `q${q}`;
            svg.appendChild(text);
        }

        // Draw gates
        circuit.gates.forEach(gate => {
            const x = 60 + gate.layer * layerSpacing;
            const y = qubitSpacing * (gate.qubit + 1);

            if (gate.type === 'CNOT') {
                // Draw CNOT gate
                const cy = qubitSpacing * (gate.control + 1);
                const ty = qubitSpacing * (gate.target + 1);

                // Control dot
                const control = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                control.setAttribute('cx', x);
                control.setAttribute('cy', cy);
                control.setAttribute('r', '5');
                control.setAttribute('fill', '#00d4ff');
                svg.appendChild(control);

                // Target circle
                const target = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                target.setAttribute('cx', x);
                target.setAttribute('cy', ty);
                target.setAttribute('r', '12');
                target.setAttribute('fill', 'none');
                target.setAttribute('stroke', '#00d4ff');
                target.setAttribute('stroke-width', '2');
                svg.appendChild(target);

                // Target plus
                const vline = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                vline.setAttribute('x1', x);
                vline.setAttribute('y1', ty - 8);
                vline.setAttribute('x2', x);
                vline.setAttribute('y2', ty + 8);
                vline.setAttribute('stroke', '#00d4ff');
                vline.setAttribute('stroke-width', '2');
                svg.appendChild(vline);

                // Connecting line
                const conn = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                conn.setAttribute('x1', x);
                conn.setAttribute('y1', cy);
                conn.setAttribute('x2', x);
                conn.setAttribute('y2', ty);
                conn.setAttribute('stroke', '#00d4ff');
                conn.setAttribute('stroke-width', '2');
                svg.appendChild(conn);
            } else {
                // Draw rotation gate (RX, RY, RZ)
                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.setAttribute('x', x - 15);
                rect.setAttribute('y', y - 12);
                rect.setAttribute('width', '30');
                rect.setAttribute('height', '24');
                rect.setAttribute('fill', '#667eea');
                rect.setAttribute('rx', '4');
                svg.appendChild(rect);

                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', x);
                text.setAttribute('y', y + 5);
                text.setAttribute('fill', 'white');
                text.setAttribute('font-size', '12');
                text.setAttribute('text-anchor', 'middle');
                text.setAttribute('font-weight', 'bold');
                text.textContent = gate.type;
                svg.appendChild(text);
            }
        });

        container.appendChild(svg);
    }

    /**
     * Animation loop
     */
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());

        if (this.coilControls) {
            this.coilControls.update();
        }

        if (this.fieldControls) {
            this.fieldControls.update();
        }

        if (this.coilRenderer && this.coilScene && this.coilCamera) {
            this.coilRenderer.render(this.coilScene, this.coilCamera);
        }

        if (this.fieldRenderer && this.fieldScene && this.fieldCamera) {
            this.fieldRenderer.render(this.fieldScene, this.fieldCamera);
        }
    }

    /**
     * Handle window resize
     */
    onWindowResize() {
        if (this.coilCamera && this.coilRenderer) {
            const canvas = this.coilRenderer.domElement;
            this.coilCamera.aspect = canvas.clientWidth / canvas.clientHeight;
            this.coilCamera.updateProjectionMatrix();
            this.coilRenderer.setSize(canvas.clientWidth, canvas.clientHeight);
        }

        if (this.fieldCamera && this.fieldRenderer) {
            const canvas = this.fieldRenderer.domElement;
            this.fieldCamera.aspect = canvas.clientWidth / canvas.clientHeight;
            this.fieldCamera.updateProjectionMatrix();
            this.fieldRenderer.setSize(canvas.clientWidth, canvas.clientHeight);
        }
    }

    /**
     * Cleanup
     */
    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }

        if (this.coilRenderer) {
            this.coilRenderer.dispose();
        }

        if (this.fieldRenderer) {
            this.fieldRenderer.dispose();
        }
    }
}
