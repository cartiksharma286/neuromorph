// ===== 3D Neural Network Visualization using Three.js =====

class NeuralVisualization {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.neurons = [];
        this.synapses = [];
        this.autoRotate = true;
        this.mouse = { x: 0, y: 0 };

        this.init();
        this.animate();
    }

    init() {
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2(0x0a0a0f, 0.002);

        // Camera setup
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.z = 150;

        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setClearColor(0x0a0a0f, 1);
        this.container.appendChild(this.renderer.domElement);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 2);
        this.scene.add(ambientLight);

        const pointLight1 = new THREE.PointLight(0x6366f1, 2, 200);
        pointLight1.position.set(50, 50, 50);
        this.scene.add(pointLight1);

        const pointLight2 = new THREE.PointLight(0xec4899, 2, 200);
        pointLight2.position.set(-50, -50, -50);
        this.scene.add(pointLight2);

        // Add particle background
        this.addParticleBackground();

        // Mouse interaction
        this.container.addEventListener('mousemove', (e) => {
            const rect = this.container.getBoundingClientRect();
            this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        });

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }

    addParticleBackground() {
        const particleCount = 1000;
        const particles = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount * 3; i += 3) {
            positions[i] = (Math.random() - 0.5) * 400;
            positions[i + 1] = (Math.random() - 0.5) * 400;
            positions[i + 2] = (Math.random() - 0.5) * 400;

            // Color gradient from purple to pink
            const t = Math.random();
            colors[i] = 0.4 + t * 0.5;     // R
            colors[i + 1] = 0.2 + t * 0.3; // G
            colors[i + 2] = 0.8 + t * 0.2; // B
        }

        particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        particles.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const particleMaterial = new THREE.PointsMaterial({
            size: 1,
            vertexColors: true,
            transparent: true,
            opacity: 0.6,
            blending: THREE.AdditiveBlending
        });

        const particleSystem = new THREE.Points(particles, particleMaterial);
        this.scene.add(particleSystem);
        this.particleSystem = particleSystem;
    }

    createNeuron(position, activity = 0) {
        // Neuron sphere
        const geometry = new THREE.SphereGeometry(2, 16, 16);
        const material = new THREE.MeshPhongMaterial({
            color: activity > 0.5 ? 0xec4899 : 0x6366f1,
            emissive: activity > 0.5 ? 0xec4899 : 0x6366f1,
            emissiveIntensity: activity,
            transparent: true,
            opacity: 0.8
        });

        const neuron = new THREE.Mesh(geometry, material);
        neuron.position.set(position.x, position.y, position.z);

        // Add glow effect for active neurons
        if (activity > 0.5) {
            const glowGeometry = new THREE.SphereGeometry(3, 16, 16);
            const glowMaterial = new THREE.MeshBasicMaterial({
                color: 0xec4899,
                transparent: true,
                opacity: 0.3,
                side: THREE.BackSide
            });
            const glow = new THREE.Mesh(glowGeometry, glowMaterial);
            neuron.add(glow);
        }

        this.scene.add(neuron);
        return neuron;
    }

    createSynapse(source, target, weight = 0.5) {
        const points = [
            new THREE.Vector3(source.x, source.y, source.z),
            new THREE.Vector3(target.x, target.y, target.z)
        ];

        const geometry = new THREE.BufferGeometry().setFromPoints(points);

        // Color based on weight strength
        const color = new THREE.Color();
        color.setHSL(0.7 - weight * 0.3, 1, 0.5);

        const material = new THREE.LineBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.2 + weight * 0.3,
            linewidth: 1
        });

        const synapse = new THREE.Line(geometry, material);
        this.scene.add(synapse);
        return synapse;
    }

    update(networkData) {
        // Clear existing neurons and synapses
        this.neurons.forEach(n => this.scene.remove(n));
        this.synapses.forEach(s => this.scene.remove(s));
        this.neurons = [];
        this.synapses = [];

        if (!networkData || !networkData.positions) return;

        // Create neurons
        networkData.positions.forEach((pos, i) => {
            const activity = networkData.activities[i] || 0;
            const neuron = this.createNeuron(pos, activity);
            this.neurons.push(neuron);
        });

        // Create synapses
        if (networkData.connections && networkData.weights) {
            networkData.connections.forEach((conn, i) => {
                const sourcePos = networkData.positions[conn.source];
                const targetPos = networkData.positions[conn.target];
                const weight = networkData.weights[i] || 0.5;

                const synapse = this.createSynapse(sourcePos, targetPos, weight);
                this.synapses.push(synapse);
            });
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Auto-rotate
        if (this.autoRotate) {
            this.scene.rotation.y += 0.002;
        }

        // Mouse interaction
        this.camera.position.x += (this.mouse.x * 20 - this.camera.position.x) * 0.05;
        this.camera.position.y += (-this.mouse.y * 20 - this.camera.position.y) * 0.05;
        this.camera.lookAt(this.scene.position);

        // Animate particles
        if (this.particleSystem) {
            this.particleSystem.rotation.y += 0.0005;
            this.particleSystem.rotation.x += 0.0002;
        }

        // Pulse active neurons
        this.neurons.forEach((neuron, i) => {
            if (neuron.material.emissiveIntensity > 0.5) {
                const scale = 1 + Math.sin(Date.now() * 0.005 + i) * 0.2;
                neuron.scale.set(scale, scale, scale);
            }
        });

        this.renderer.render(this.scene, this.camera);
    }

    toggleRotation() {
        this.autoRotate = !this.autoRotate;
    }

    reset() {
        this.neurons.forEach(n => this.scene.remove(n));
        this.synapses.forEach(s => this.scene.remove(s));
        this.neurons = [];
        this.synapses = [];
        this.scene.rotation.set(0, 0, 0);
    }

    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.neuralViz = new NeuralVisualization('neural-network-viz');
    console.log('✓ Neural Network 3D Visualization initialized');
});
