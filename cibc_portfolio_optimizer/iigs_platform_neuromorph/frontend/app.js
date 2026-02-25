
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Global State
const state = {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    implantPoints: null,
    abutmentPoints: null,
    probe: null
};

// Initialization
function init() {
    // Canvas Setup
    const container = document.getElementById('canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    state.scene = new THREE.Scene();
    state.scene.background = new THREE.Color(0x0a0b10);
    state.scene.fog = new THREE.FogExp2(0x0a0b10, 0.02);

    // Camera
    state.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    state.camera.position.set(10, 10, 20);

    // Renderer
    state.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    state.renderer.setSize(width, height);
    state.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(state.renderer.domElement);

    // Controls
    state.controls = new OrbitControls(state.camera, state.renderer.domElement);
    state.controls.enableDamping = true;
    state.controls.dampingFactor = 0.05;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    state.scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0x00f2ff, 1);
    pointLight.position.set(10, 10, 10);
    state.scene.add(pointLight);

    // Grid
    const gridHelper = new THREE.GridHelper(50, 50, 0x333333, 0x111111);
    state.scene.add(gridHelper);

    // Instrument Probe (Simulated)
    const probeGeo = new THREE.ConeGeometry(0.5, 5, 8);
    const probeMat = new THREE.MeshStandardMaterial({
        color: 0xff3366,
        emissive: 0xff3366,
        emissiveIntensity: 0.5,
        transparent: true,
        opacity: 0.8
    });
    state.probe = new THREE.Mesh(probeGeo, probeMat);
    state.probe.rotation.x = Math.PI; // Flashlight style
    state.probe.position.set(5, 5, 5);
    state.scene.add(state.probe);

    // Listeners
    window.addEventListener('resize', onWindowResize);
    document.getElementById('load-session-btn').addEventListener('click', fetchLatestSession);

    // Mouse movement for probe simulation
    container.addEventListener('mousemove', onMouseMove);

    // Start Loop
    animate();

    // Auto-load
    console.log("App initialized. Fetching session...");
    fetchLatestSession();
}

function onWindowResize() {
    const container = document.getElementById('canvas-container');
    state.camera.aspect = container.clientWidth / container.clientHeight;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(container.clientWidth, container.clientHeight);
}

function onMouseMove(event) {
    // Simple screen-space to world mapping for "fake" probe movement
    // Real tracking would come from websocket/server
    const rect = state.renderer.domElement.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Move probe slightly based on mouse to feel "alive"
    if (state.probe) {
        state.probe.position.x += x * 0.1;
        state.probe.position.y += y * 0.1;

        // Update UI coords
        document.getElementById('nav-x').innerText = state.probe.position.x.toFixed(2);
        document.getElementById('nav-y').innerText = state.probe.position.y.toFixed(2);
        document.getElementById('nav-z').innerText = state.probe.position.z.toFixed(2);
    }
}

async function fetchLatestSession() {
    try {
        const response = await fetch('/api/latest-session');
        if (!response.ok) throw new Error("Failed to fetch session");

        const session = await response.json();
        console.log("Session loaded:", session);
        updateUI(session);
        loadGeometry(session.files);
    } catch (err) {
        console.error(err);
        document.getElementById('connection-status').innerText = "Connection Failed";
        document.getElementById('connection-status').style.color = "#ff3366";
    }
}

function updateUI(session) {
    // Patient
    document.getElementById('p-id').innerText = session.patient.patient_id;
    document.getElementById('p-density').innerText = session.patient.bone_density;
    document.getElementById('p-gap').innerText = session.patient.gap_size + " mm";

    // Design
    document.getElementById('d-type').innerText = session.optimization_params.implant_type;
    document.getElementById('d-density').innerText = session.optimization_params.density.toFixed(2) + " g/cc";
    document.getElementById('d-conf').innerText = (session.optimization_params.optimization_confidence * 100).toFixed(1) + "%";

    // FEA
    const res = session.results;

    // Implant FEA
    const impSf = res.implant_fea.safety_factor;
    document.getElementById('fea-imp-sf').innerText = impSf.toFixed(1);
    document.getElementById('fea-imp-stress').innerText = res.implant_fea.max_stress.toFixed(1);
    updateProgressBar('fea-imp-bar', impSf);

    // Abutment FEA
    const abuSf = res.abutment_fea.safety_factor;
    document.getElementById('fea-abu-sf').innerText = abuSf.toFixed(1);
    document.getElementById('fea-abu-stress').innerText = res.abutment_fea.max_stress.toFixed(1);
    updateProgressBar('fea-abu-bar', abuSf);

    // IGS
    document.getElementById('igs-fre').innerText = res.igs_registration_fre.toFixed(4) + " mm";
}

function updateProgressBar(id, safetyFactor) {
    // Scale: SF 1.0 = 20%, SF 5.0 = 100%
    let pct = Math.min((safetyFactor / 5.0) * 100, 100);
    const el = document.getElementById(id);
    el.style.width = pct + "%";

    if (safetyFactor < 1.5) el.style.backgroundColor = "#ff3366"; // Danger
    else if (safetyFactor < 3.0) el.style.backgroundColor = "#ffae00"; // Warning
    else el.style.backgroundColor = "#00ff9d"; // Good
}

async function loadGeometry(files) {
    // Clear old
    if (state.implantPoints) state.scene.remove(state.implantPoints);
    if (state.abutmentPoints) state.scene.remove(state.abutmentPoints);

    // Fetch Implant Geometry
    try {
        const impRes = await fetch(`/data/implant_design/${files.implant_design}`);
        const impData = await impRes.json();
        state.implantPoints = createMesh(impData, 0x00f2ff, [0, 0, 0], 0.8); // Titanium-like
        state.scene.add(state.implantPoints);

        const abuRes = await fetch(`/data/implant_design/${files.abutment_design}`);
        const abuData = await abuRes.json();
        // Offset abutment slightly up based on Z-height logic (approx 10mm for implant length)
        state.abutmentPoints = createMesh(abuData, 0xbd00ff, [0, 0, 10], 0.2); // Ceramic-like (less metallic)
        state.scene.add(state.abutmentPoints);

    } catch (e) {
        console.error("Failed to load geometry", e);
    }
}

function createMesh(jsonData, fallbackHex, offset, metalness = 0.5) {
    const vertices = jsonData.vertices;
    const faces = jsonData.faces;
    const stress = jsonData.stress_values;

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(vertices.length * 3);
    const colors = new Float32Array(vertices.length * 3);

    // Color Helpers
    const color = new THREE.Color();
    const fallbackColor = new THREE.Color(fallbackHex);

    let minStress = 0, maxStress = 100;
    if (stress && stress.length > 0) {
        minStress = Math.min(...stress);
        maxStress = Math.max(...stress);
    }

    for (let i = 0; i < vertices.length; i++) {
        // Position
        positions[i * 3] = vertices[i][0] + offset[0];
        positions[i * 3 + 1] = vertices[i][1] + offset[1];
        positions[i * 3 + 2] = vertices[i][2] + offset[2];

        // Color (Stress Heatmap)
        if (stress && stress.length > 0) {
            const t = (maxStress - minStress === 0) ? 0 : (stress[i] - minStress) / (maxStress - minStress);
            const hue = (1.0 - t) * 0.66; // Blue -> Red
            color.setHSL(hue, 1.0, 0.5);

            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        } else {
            colors[i * 3] = fallbackColor.r;
            colors[i * 3 + 1] = fallbackColor.g;
            colors[i * 3 + 2] = fallbackColor.b;
        }
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Handle Faces (Indices)
    if (faces && faces.length > 0) {
        const indices = [];
        for (let i = 0; i < faces.length; i++) {
            indices.push(faces[i][0], faces[i][1], faces[i][2]);
        }
        geometry.setIndex(indices);
        geometry.computeVertexNormals();

        // Premium Material
        const material = new THREE.MeshPhysicalMaterial({
            vertexColors: true,
            metalness: metalness,
            roughness: 0.4,
            clearcoat: 0.8,
            clearcoatRoughness: 0.2,
            side: THREE.DoubleSide
        });

        return new THREE.Mesh(geometry, material);
    } else {
        // Fallback to Points if no faces generated
        const material = new THREE.PointsMaterial({
            size: 0.2,
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        });
        return new THREE.Points(geometry, material);
    }
}

function animate() {
    requestAnimationFrame(animate);

    if (state.controls) state.controls.update();

    // Rotate models slowly
    if (state.implantPoints) state.implantPoints.rotation.z += 0.005;
    if (state.abutmentPoints) state.abutmentPoints.rotation.z += 0.005;

    if (state.renderer && state.scene && state.camera) {
        state.renderer.render(state.scene, state.camera);
    }
}

// Boot
init();
