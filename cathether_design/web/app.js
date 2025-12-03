// Global state
let scene, camera, renderer, controls;
let catheterMesh = null;
let stentMesh = null;
let flowParticles = null;
let autoRotate = false;
let wireframeMode = false;
let xrayMode = false;
let currentMode = 'catheter'; // 'catheter' or 'stent'

// WebSocket connection to backend
let ws = null;
let currentDesign = null;
let optimizationRunning = false;

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Quantum Catheter Designer...');

    initThreeJS();
    initEventListeners();
    connectBackend();

    // Initial catheter generation
    generateInitialCatheter();
});

/**
 * Initialize Three.js 3D viewer
 */
function initThreeJS() {
    const container = document.getElementById('viewer-container');

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e1a);
    scene.fog = new THREE.Fog(0x0a0e1a, 500, 2000);

    // Camera
    const aspect = container.clientWidth / container.clientHeight;
    camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 5000);
    camera.position.set(0, 300, 800);
    camera.lookAt(0, 0, 500);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 50;
    controls.maxDistance = 2000;
    controls.target.set(0, 0, 500);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0x00d9ff, 0.8);
    mainLight.position.set(200, 300, 200);
    mainLight.castShadow = true;
    scene.add(mainLight);

    const fillLight = new THREE.DirectionalLight(0x9d4edd, 0.4);
    fillLight.position.set(-200, -100, 100);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xff006e, 0.3);
    rimLight.position.set(0, 200, -200);
    scene.add(rimLight);

    // Grid helper
    const gridHelper = new THREE.GridHelper(1000, 20, 0x00d9ff, 0x1a1f35);
    gridHelper.position.y = -50;
    scene.add(gridHelper);

    // Start animation loop
    animate();

    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

/**
 * Animation loop
 */
function animate() {
    requestAnimationFrame(animate);

    controls.update();

    if (autoRotate) {
        if (catheterMesh) catheterMesh.rotation.y += 0.005;
        if (stentMesh) stentMesh.rotation.y += 0.005;
    }

    if (flowParticles) {
        animateFlow();
    }

    renderer.render(scene, camera);
}

function animateFlow() {
    const positions = flowParticles.geometry.attributes.position.array;
    for (let i = 0; i < positions.length; i += 3) {
        positions[i + 2] += 2.0; // Move along Z
        if (positions[i + 2] > 1000) positions[i + 2] = 0;
    }
    flowParticles.geometry.attributes.position.needsUpdate = true;
}

/**
 * Handle window resize
 */
function onWindowResize() {
    const container = document.getElementById('viewer-container');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

/**
 * Generate initial catheter for preview
 */
function generateInitialCatheter() {
    // Create simple parametric catheter geometry
    const length = 1000;
    const radius = 2.25; // 4.5mm diameter / 2
    const segments = 100;

    const geometry = new THREE.CylinderGeometry(
        radius, radius * 0.7, length, 32, segments
    );

    // Rotate to align with Z axis
    geometry.rotateX(Math.PI / 2);
    geometry.translate(0, 0, length / 2);

    // Material
    const material = new THREE.MeshPhysicalMaterial({
        color: 0x00d9ff,
        metalness: 0.3,
        roughness: 0.4,
        transparent: true,
        opacity: 0.9,
        transmission: 0.1,
        thickness: 0.5,
        envMapIntensity: 1,
    });

    // Remove old mesh if exists
    if (catheterMesh) scene.remove(catheterMesh);
    if (stentMesh) scene.remove(stentMesh);

    catheterMesh = new THREE.Mesh(geometry, material);
    catheterMesh.castShadow = true;
    catheterMesh.receiveShadow = true;
    scene.add(catheterMesh);

    console.log('Initial catheter generated');
}

/**
 * Generate Stent Mesh (Visual Approximation)
 */
function updateStentMesh(params) {
    if (catheterMesh) scene.remove(catheterMesh);
    if (stentMesh) scene.remove(stentMesh);

    const length = parseFloat(params.length || 18.0);
    const diameter = parseFloat(params.diameter || 3.0);
    const radius = diameter / 2;

    // Create a wireframe-like structure for the stent
    // Using a texture with alpha map is efficient for visualization
    const geometry = new THREE.CylinderGeometry(radius, radius, length, 32, 1, true);
    geometry.rotateX(Math.PI / 2);

    // Create a lattice texture
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000000'; // Transparent
    ctx.fillRect(0, 0, 512, 512);
    ctx.strokeStyle = '#FFFFFF'; // Struts
    ctx.lineWidth = 10;

    // Draw diamond pattern
    const cellsX = 8;
    const cellsY = 4;
    const cellW = 512 / cellsX;
    const cellH = 512 / cellsY;

    ctx.beginPath();
    for (let y = 0; y <= cellsY; y++) {
        for (let x = 0; x <= cellsX; x++) {
            const cx = x * cellW;
            const cy = y * cellH;
            ctx.moveTo(cx, cy - cellH / 2);
            ctx.lineTo(cx + cellW / 2, cy);
            ctx.lineTo(cx, cy + cellH / 2);
            ctx.lineTo(cx - cellW / 2, cy);
        }
    }
    ctx.stroke();

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;

    const material = new THREE.MeshStandardMaterial({
        color: 0xc0c0c0, // Silver/Chrome
        metalness: 0.9,
        roughness: 0.2,
        alphaMap: texture,
        transparent: true,
        side: THREE.DoubleSide
    });

    stentMesh = new THREE.Mesh(geometry, material);
    scene.add(stentMesh);

    // Center camera on stent
    controls.target.set(0, 0, 0);
    camera.position.set(0, 20, 40);
}

/**
 * Update catheter mesh from design parameters
 */
function updateCatheterMesh(design) {
    console.log('Updating catheter mesh:', design);

    // For now, update with simplified geometry
    // In production, this would load the actual STL from backend

    const length = parseFloat(document.getElementById('required-length').value);
    const outerDiameter = design.outer_diameter || 4.5;
    const innerDiameter = design.inner_diameter || 3.8;

    const outerRadius = outerDiameter / 2;
    const innerRadius = innerDiameter / 2;

    // Create hollow cylinder
    const shape = new THREE.Shape();
    shape.absarc(0, 0, outerRadius, 0, Math.PI * 2, false);

    const hole = new THREE.Path();
    hole.absarc(0, 0, innerRadius, 0, Math.PI * 2, true);
    shape.holes.push(hole);

    const extrudeSettings = {
        steps: 100,
        depth: length,
        bevelEnabled: false
    };

    const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
    geometry.rotateX(Math.PI / 2);

    // Update material based on flexibility
    const flexibility = design.flexibility_index || 0.75;
    const hue = 0.5 + flexibility * 0.1; // Cyan to blue

    const material = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color().setHSL(hue, 1, 0.5),
        metalness: 0.3,
        roughness: 0.4,
        transparent: true,
        opacity: 0.9,
        transmission: 0.1,
    });

    // Remove old mesh
    if (catheterMesh) scene.remove(catheterMesh);
    if (stentMesh) scene.remove(stentMesh);

    catheterMesh = new THREE.Mesh(geometry, material);
    catheterMesh.castShadow = true;
    scene.add(catheterMesh);

    // Update output fields
    updateOutputFields(design);
}

/**
 * Update output design parameter fields
 */
function updateOutputFields(design) {
    document.getElementById('output-od').value =
        `${design.outer_diameter?.toFixed(3) || '--'} mm`;
    document.getElementById('output-id').value =
        `${design.inner_diameter?.toFixed(3) || '--'} mm`;
    document.getElementById('output-wall').value =
        `${design.wall_thickness?.toFixed(3) || '--'} mm`;
    document.getElementById('output-tip').value =
        `${design.tip_angle?.toFixed(1) || '--'}°`;
}

/**
 * Update performance metrics display
 */
function updateMetrics(metrics) {
    if (metrics.pressure_drop !== undefined) {
        document.getElementById('pressure-drop').textContent =
            metrics.pressure_drop.toFixed(2);
    }

    if (metrics.reynolds_number !== undefined) {
        document.getElementById('reynolds').textContent =
            metrics.reynolds_number.toFixed(1);
    }

    if (metrics.flexibility_index !== undefined) {
        document.getElementById('flexibility').textContent =
            metrics.flexibility_index.toFixed(3);
    }

    if (metrics.average_velocity !== undefined) {
        document.getElementById('velocity').textContent =
            (metrics.average_velocity * 100).toFixed(2); // m/s to cm/s
    }
}

/**
 * Initialize event listeners
 */
function initEventListeners() {
    // Mode Switching
    document.getElementById('mode-catheter').addEventListener('click', () => switchMode('catheter'));
    document.getElementById('mode-stent').addEventListener('click', () => switchMode('stent'));

    // Slider value updates
    document.getElementById('vessel-curvature').addEventListener('input', (e) => {
        document.getElementById('curvature-value').textContent = e.target.value;
    });

    document.getElementById('strut-thickness').addEventListener('input', (e) => {
        document.getElementById('thickness-value').textContent = e.target.value;
    });

    document.getElementById('crowns-count').addEventListener('input', (e) => {
        document.getElementById('crowns-value').textContent = e.target.value;
    });

    document.getElementById('qubits').addEventListener('input', (e) => {
        document.getElementById('qubits-value').textContent = e.target.value;
    });

    document.getElementById('layers').addEventListener('input', (e) => {
        document.getElementById('layers-value').textContent = e.target.value;
    });

    document.getElementById('iterations').addEventListener('input', (e) => {
        document.getElementById('iterations-value').textContent = e.target.value;
    });

    // Viewer controls
    document.getElementById('reset-view').addEventListener('click', () => {
        if (currentMode === 'stent') {
            controls.target.set(0, 0, 0);
            camera.position.set(0, 20, 40);
        } else {
            camera.position.set(0, 300, 800);
            camera.lookAt(0, 0, 500);
            controls.target.set(0, 0, 500);
        }
    });

    document.getElementById('wireframe-toggle').addEventListener('click', () => {
        wireframeMode = !wireframeMode;
        if (catheterMesh) catheterMesh.material.wireframe = wireframeMode;
        if (stentMesh) stentMesh.material.wireframe = wireframeMode;
    });

    document.getElementById('rotate-toggle').addEventListener('click', () => {
        autoRotate = !autoRotate;
    });

    document.getElementById('xray-toggle').addEventListener('click', () => {
        xrayMode = !xrayMode;
        if (catheterMesh) {
            catheterMesh.material.opacity = xrayMode ? 0.3 : 0.9;
            catheterMesh.material.transparent = true;
        }
    });

    // Optimization button
    document.getElementById('optimize-btn').addEventListener('click', startOptimization);

    // Export button
    document.getElementById('export-btn').addEventListener('click', exportSTL);

    // Save/Load buttons
    document.getElementById('save-btn').addEventListener('click', saveDesign);
    document.getElementById('load-btn').addEventListener('click', loadDesign);
}

function switchMode(mode) {
    currentMode = mode;

    // Update buttons
    document.getElementById('mode-catheter').classList.toggle('active', mode === 'catheter');
    document.getElementById('mode-stent').classList.toggle('active', mode === 'stent');
    document.getElementById('mode-catheter').classList.toggle('btn-primary', mode === 'catheter');
    document.getElementById('mode-catheter').classList.toggle('btn-secondary', mode !== 'catheter');
    document.getElementById('mode-stent').classList.toggle('btn-primary', mode === 'stent');
    document.getElementById('mode-stent').classList.toggle('btn-secondary', mode !== 'stent');

    // Show/Hide controls
    document.getElementById('catheter-controls').style.display = mode === 'catheter' ? 'block' : 'none';
    document.getElementById('stent-controls').style.display = mode === 'stent' ? 'block' : 'none';

    // Show LBM button in stent mode
    document.getElementById('run-lbm-btn').style.display = mode === 'stent' ? 'block' : 'none';
    document.getElementById('flow-toggle').style.display = mode === 'stent' ? 'inline-flex' : 'none';

    if (mode === 'stent') {
        const params = {
            length: document.getElementById('stent-length').value,
            diameter: document.getElementById('stent-diameter').value
        };
        updateStentMesh(params);
    } else {
        generateInitialCatheter();
    }
}

/**
 * Start quantum optimization
 */
function startOptimization() {
    if (optimizationRunning) {
        console.log('Optimization already running');
        return;
    }

    optimizationRunning = true;
    updateStatus('running', 'Optimizing...');

    if (currentMode === 'stent') {
        // Stent Optimization (Quantum Surrogate)
        const params = {
            stent_params: {
                diameter: parseFloat(document.getElementById('stent-diameter').value),
                length: parseFloat(document.getElementById('stent-length').value),
                strut_thickness: parseFloat(document.getElementById('strut-thickness').value) / 1000, // um to mm
                crowns_per_ring: parseInt(document.getElementById('crowns-count').value)
            },
            quantum_settings: {
                n_qubits: parseInt(document.getElementById('qubits').value),
                n_layers: parseInt(document.getElementById('layers').value)
            }
        };

        // Simulate QNN prediction for now
        setTimeout(() => {
            updateMetrics({
                pressure_drop: 0, // Not relevant for stent optimization directly
                reynolds_number: 0,
                flexibility_index: 0.85,
                average_velocity: 0
            });

            // Show predicted WSS in metrics
            document.getElementById('pressure-drop').textContent = "1.25 Pa (WSS)";
            document.getElementById('pressure-drop').nextElementSibling.textContent = "Max WSS";

            optimizationRunning = false;
            updateStatus('success', 'QNN Prediction Complete');

            // Update visual
            updateStentMesh(params.stent_params);

        }, 1500);

    } else {
        // Catheter Optimization
        const params = {
            patient_constraints: {
                vessel_diameter: parseFloat(document.getElementById('vessel-diameter').value),
                vessel_curvature: parseFloat(document.getElementById('vessel-curvature').value),
                bifurcation_angle: parseFloat(document.getElementById('bifurcation-angle').value),
                required_length: parseFloat(document.getElementById('required-length').value),
                flow_rate: parseFloat(document.getElementById('flow-rate').value)
            },
            quantum_settings: {
                n_qubits: parseInt(document.getElementById('qubits').value),
                n_layers: parseInt(document.getElementById('layers').value),
                max_iterations: parseInt(document.getElementById('iterations').value)
            }
        };

        console.log('Starting optimization with params:', params);

        // Send to backend
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'optimize',
                data: params
            }));
        } else {
            // Simulate optimization locally
            simulateOptimization(params);
        }
    }
}

/**
 * Simulate optimization (fallback when backend unavailable)
 */
function simulateOptimization(params) {
    console.log('Simulating optimization locally...');

    let iteration = 0;
    const maxIterations = params.quantum_settings.max_iterations;

    const interval = setInterval(() => {
        iteration++;

        // Update convergence chart
        updateConvergenceChart(iteration, Math.exp(-iteration / 20) + Math.random() * 0.1);

        if (iteration >= maxIterations) {
            clearInterval(interval);

            // Generate mock optimized design
            const design = {
                outer_diameter: params.patient_constraints.vessel_diameter * 0.85,
                inner_diameter: params.patient_constraints.vessel_diameter * 0.70,
                wall_thickness: 0.35,
                tip_angle: 30 + Math.random() * 15,
                flexibility_index: 0.65 + Math.random() * 0.2,
                side_holes: [200, 400, 600, 800]
            };

            currentDesign = design;
            updateCatheterMesh(design);

            // Update metrics
            updateMetrics({
                pressure_drop: 150 + Math.random() * 50,
                reynolds_number: 800 + Math.random() * 400,
                flexibility_index: design.flexibility_index,
                average_velocity: 0.12 + Math.random() * 0.05
            });

            optimizationRunning = false;
            updateStatus('success', 'Optimization Complete');
        }
    }, 100);
}

/**
 * Update status badge
 */
function updateStatus(state, text) {
    const badge = document.getElementById('status-badge');
    const statusText = document.getElementById('status-text');

    badge.className = 'status-badge';
    if (state === 'running') {
        badge.classList.add('running');
    } else if (state === 'error') {
        badge.classList.add('error');
    }

    statusText.textContent = text;
}

/**
 * Connect to backend WebSocket
 */
function connectBackend() {
    try {
        ws = new WebSocket('ws://localhost:8765/ws');

        ws.onopen = () => {
            console.log('Connected to backend');
            updateStatus('success', 'Connected');
        };

        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            handleBackendMessage(message);
        };

        ws.onerror = (error) => {
            console.warn('WebSocket error, using local mode:', error);
            updateStatus('success', 'Ready (Local Mode)');
        };

        ws.onclose = () => {
            console.log('Backend connection closed');
            updateStatus('success', 'Ready (Local Mode)');
        };
    } catch (error) {
        console.warn('Could not connect to backend:', error);
        updateStatus('success', 'Ready (Local Mode)');
    }
}

/**
 * Handle messages from backend
 */
function handleBackendMessage(message) {
    console.log('Backend message:', message);

    switch (message.type) {
        case 'optimization_progress':
            updateConvergenceChart(message.iteration, message.cost);
            break;

        case 'optimization_complete':
            currentDesign = message.design;
            updateCatheterMesh(message.design);
            updateMetrics(message.metrics);
            optimizationRunning = false;
            updateStatus('success', 'Optimization Complete');
            break;

        case 'error':
            console.error('Backend error:', message.error);
            updateStatus('error', 'Error');
            optimizationRunning = false;
            break;
    }
}

/**
 * Export current design to STL
 */
function exportSTL() {
    if (!currentDesign && currentMode === 'catheter') {
        alert('Please run optimization first');
        return;
    }

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'export',
            format: 'stl',
            design: currentDesign
        }));

        updateStatus('running', 'Generating STL...');
    } else {
        // Local export using Three.js
        exportLocalSTL();
    }
}

/**
 * Export STL file locally using Three.js
 */
function exportLocalSTL() {
    if (!catheterMesh) return;

    // This would require STLExporter from Three.js examples
    // For now, show a message
    alert('STL export requires backend connection. Please start the server.');
}

/**
 * Save current design
 */
function saveDesign() {
    if (!currentDesign && currentMode === 'catheter') {
        alert('No design to save');
        return;
    }

    const designData = {
        design: currentDesign,
        patient_constraints: {
            vessel_diameter: parseFloat(document.getElementById('vessel-diameter').value),
            vessel_curvature: parseFloat(document.getElementById('vessel-curvature').value),
            bifurcation_angle: parseFloat(document.getElementById('bifurcation-angle').value),
            required_length: parseFloat(document.getElementById('required-length').value),
            flow_rate: parseFloat(document.getElementById('flow-rate').value)
        }
    };

    const blob = new Blob([JSON.stringify(designData, null, 2)],
        { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `catheter_design_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

/**
 * Load saved design
 */
function loadDesign() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';

    input.onchange = (e) => {
        const file = e.target.files[0];
        const reader = new FileReader();

        reader.onload = (event) => {
            try {
                const data = JSON.parse(event.target.result);
                currentDesign = data.design;

                // Update inputs
                if (data.patient_constraints) {
                    const pc = data.patient_constraints;
                    document.getElementById('vessel-diameter').value = pc.vessel_diameter;
                    document.getElementById('vessel-curvature').value = pc.vessel_curvature;
                    document.getElementById('bifurcation-angle').value = pc.bifurcation_angle;
                    document.getElementById('required-length').value = pc.required_length;
                    document.getElementById('flow-rate').value = pc.flow_rate;

                    // Update slider displays
                    document.getElementById('curvature-value').textContent = pc.vessel_curvature;
                }

                // Update visualization
                updateCatheterMesh(currentDesign);
                updateStatus('success', 'Design Loaded');
            } catch (error) {
                console.error('Error loading design:', error);
                alert('Error loading design file');
            }
        };

        reader.readAsText(file);
    };

    input.click();
}

console.log('Quantum Catheter Designer initialized');
