// ===== Quantum Circuit and Bloch Sphere Visualization =====

class QuantumVisualization {
    constructor() {
        this.circuitContainer = document.getElementById('quantum-circuit-viz');
        this.blochContainer = document.getElementById('bloch-sphere-viz');
        this.vqeChart = null;

        this.initBlochSphere();
        this.initVQEChart();
    }

    initBlochSphere() {
        // Create 3D Bloch sphere using Three.js
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(
            75,
            this.blochContainer.clientWidth / this.blochContainer.clientHeight,
            0.1,
            1000
        );
        camera.position.z = 3;

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(this.blochContainer.clientWidth, this.blochContainer.clientHeight);
        renderer.setClearColor(0x0a0a0f, 1);
        this.blochContainer.appendChild(renderer.domElement);

        // Bloch sphere
        const sphereGeometry = new THREE.SphereGeometry(1, 32, 32);
        const sphereMaterial = new THREE.MeshPhongMaterial({
            color: 0x6366f1,
            transparent: true,
            opacity: 0.3,
            wireframe: true
        });
        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        scene.add(sphere);

        // Axes
        const axesHelper = new THREE.AxesHelper(1.5);
        scene.add(axesHelper);

        // State vector
        const arrowGeometry = new THREE.ConeGeometry(0.1, 0.3, 16);
        const arrowMaterial = new THREE.MeshPhongMaterial({ color: 0xec4899 });
        const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
        scene.add(arrow);

        // Lighting
        const light = new THREE.PointLight(0xffffff, 1, 100);
        light.position.set(5, 5, 5);
        scene.add(light);

        this.blochScene = { scene, camera, renderer, sphere, arrow };

        // Animation loop
        const animate = () => {
            requestAnimationFrame(animate);
            sphere.rotation.y += 0.005;
            renderer.render(scene, camera);
        };
        animate();
    }

    initVQEChart() {
        const ctx = document.getElementById('vqe-progress-chart');
        if (!ctx) return;

        this.vqeChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'VQE Energy',
                    data: [],
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'Energy', color: '#adb5bd' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#adb5bd' }
                    },
                    x: {
                        title: { display: true, text: 'Iteration', color: '#adb5bd' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#adb5bd' }
                    }
                }
            }
        });
    }

    update(quantumData) {
        if (!quantumData) return;

        // Update Bloch sphere state vector
        if (this.blochScene && quantumData.stateVector) {
            const { theta, phi } = quantumData.stateVector;
            const x = Math.sin(theta) * Math.cos(phi);
            const y = Math.sin(theta) * Math.sin(phi);
            const z = Math.cos(theta);

            this.blochScene.arrow.position.set(x, y, z);
            this.blochScene.arrow.lookAt(0, 0, 0);
        }

        // Update VQE chart
        if (this.vqeChart && quantumData.vqeEnergy !== undefined) {
            const iteration = this.vqeChart.data.labels.length;
            this.vqeChart.data.labels.push(iteration);
            this.vqeChart.data.datasets[0].data.push(quantumData.vqeEnergy);

            if (this.vqeChart.data.labels.length > 50) {
                this.vqeChart.data.labels.shift();
                this.vqeChart.data.datasets[0].data.shift();
            }

            this.vqeChart.update('none');
        }

        // Draw quantum circuit
        this.drawQuantumCircuit(quantumData);
    }

    drawQuantumCircuit(data) {
        const canvas = this.circuitContainer;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Simple circuit visualization
        const qubits = 4;
        const gateWidth = 40;
        const gateHeight = 30;
        const qubitSpacing = 60;
        const startX = 50;
        const startY = 50;

        // Draw qubit lines
        ctx.strokeStyle = '#6c757d';
        ctx.lineWidth = 2;
        for (let i = 0; i < qubits; i++) {
            const y = startY + i * qubitSpacing;
            ctx.beginPath();
            ctx.moveTo(startX, y);
            ctx.lineTo(canvas.width - 50, y);
            ctx.stroke();
        }

        // Draw gates (simplified)
        const gates = ['H', 'RY', 'CNOT', 'RZ'];
        gates.forEach((gate, i) => {
            const x = startX + 100 + i * 100;
            const y = startY + (i % qubits) * qubitSpacing;

            ctx.fillStyle = '#6366f1';
            ctx.fillRect(x - gateWidth / 2, y - gateHeight / 2, gateWidth, gateHeight);

            ctx.fillStyle = '#ffffff';
            ctx.font = '14px Inter';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(gate, x, y);
        });
    }

    reset() {
        if (this.vqeChart) {
            this.vqeChart.data.labels = [];
            this.vqeChart.data.datasets[0].data = [];
            this.vqeChart.update();
        }
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.quantumViz = new QuantumVisualization();
    console.log('✓ Quantum Visualization initialized');
});
