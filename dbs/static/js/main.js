// Neuromorph DBS Main Logic

let scene, camera, renderer, headGeometry, FEA_particles;
let voltage = 30;
let pulseWidth = 0.2;
let coilRadius = 0.05;

// Initialize 3D Scene
function init3D() {
    const container = document.getElementById('canvas-container');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Add a "Head" proxy
    const headGeo = new THREE.IcosahedronGeometry(2, 4);
    const headMat = new THREE.MeshPhongMaterial({
        color: 0x1a1a2e,
        wireframe: true,
        transparent: true,
        opacity: 0.3
    });
    const head = new THREE.Mesh(headGeo, headMat);
    scene.add(head);

    // Add a Coil
    const coilGeo = new THREE.TorusGeometry(1, 0.05, 16, 100);
    const coilMat = new THREE.MeshBasicMaterial({ color: 0x00f2ff });
    const coil = new THREE.Mesh(coilGeo, coilMat);
    coil.rotation.x = Math.PI / 2;
    coil.position.y = 2.2;
    scene.add(coil);

    // Lighting
    const pointLight = new THREE.PointLight(0xff00c8, 1);
    pointLight.position.set(5, 5, 5);
    scene.add(pointLight);
    scene.add(new THREE.AmbientLight(0x404040));

    camera.position.z = 5;
    camera.position.y = 2;
    camera.lookAt(0, 0, 0);

    animate();
}

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

// FEA Cortical Simulation Visualization
function drawCorticalFEA() {
    const canvas = document.getElementById('fea-cortical-canvas');
    if (!canvas) {
        requestAnimationFrame(drawCorticalFEA);
        return;
    }
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.parentElement.clientWidth;
    const h = canvas.height = canvas.parentElement.clientHeight;

    ctx.clearRect(0, 0, w, h);
    
    const time = Date.now() * 0.001;

    // Draw cortical patches
    for (let x = 0; x < w; x += 10) {
        for (let y = 0; y < h; y += 10) {
            // Perlin noise-like or wave-like pattern for cortical current density 
            const dx = x - w/2;
            const dy = y - h/2;
            const dist = Math.sqrt(dx*dx + dy*dy);
            
            // Activation spread originating from center
            const intensity = Math.max(0, Math.sin(dist * 0.05 - time * 3) * Math.cos(x * 0.02) * Math.sin(y * 0.02));
            
            if (intensity > 0.1) {
                // Determine color based on intensity (deep blue to bright cyan/pink)
                ctx.fillStyle = `rgba(${Math.floor(intensity * 255)}, ${Math.floor(intensity * 100)}, 255, ${intensity})`;
                ctx.fillRect(x, y, 8, 8);
            }
        }
    }
    
    // Draw the main electrode in the center
    ctx.beginPath();
    ctx.arc(w/2, h/2, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.shadowBlur = 15;
    ctx.shadowColor = '#00f2ff';
    
    // Draw propagating field lines
    ctx.strokeStyle = 'rgba(0, 242, 255, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(w/2, h/2, 10 + (time * 10) % 50, 0, Math.PI * 2);
    ctx.stroke();

    requestAnimationFrame(drawCorticalFEA);
}

// Fetch System Specs
async function fetchSystemSpecs() {
    const response = await fetch('/api/system-specs');
    const data = await response.json();
    const list = document.getElementById('system-specs-list');
    list.innerHTML = Object.entries(data).map(([key, val]) => `
        <div style="margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 4px;">
            <strong style="color:var(--accent-cyan); text-transform:uppercase; font-size:9px;">${key.replace('_', ' ')}</strong><br>
            <span style="color:var(--text-primary); font-family: monospace;">${val}</span>
        </div>
    `).join('');
}

// Logic for Simulation & Bio-Signals
async function runSimulation() {
    const nodes = [
        { id: 'primary', x: 0, y: 1.5, z: 0 },
        { id: 'secondary', x: 0.5, y: 1.2, z: 0.5 }
    ];

    const response = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes, voltage, pulseWidth })
    });

    const data = await response.json();
    if (data.status === 'success') {
        const res = data.results[0];
        document.getElementById('yield-value').textContent = (res.optimized_yield).toFixed(3) + "%";
        document.getElementById('field-strength-val').textContent = res.field.toExponential(2) + ' T';

        // Update Quantum Freq
        if (res.quantum_optimal_freq) {
            document.getElementById('quantum-freq-val').textContent = res.quantum_optimal_freq.toFixed(2);
        }

        // Fetch companion bio-signal analysis
        const bioResponse = await fetch('/api/analyze-biosignals', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frequency: res.quantum_optimal_freq })
        });
        const bioData = await bioResponse.json();
        // Update any specific bio-signal UI if needed, for now we let signal-viz run
    }
}

// Event Listeners
document.getElementById('btn-simulate').addEventListener('click', runSimulation);

document.getElementById('voltage-range').addEventListener('input', (e) => {
    voltage = parseFloat(e.target.value);
});
document.getElementById('pulse-range').addEventListener('input', (e) => {
    pulseWidth = parseFloat(e.target.value);
});

// Tab Switching Logic
function switchTab(tabId, event) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    if (event) {
        event.currentTarget.classList.add('active');
    } else {
        // Find the button if event not provided (initial load)
        const btn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.textContent.toLowerCase().includes(tabId));
        if (btn) btn.classList.add('active');
    }

    document.getElementById(`${tabId}-sidebar`).classList.add('active');
    document.getElementById(`${tabId}-main`).classList.add('active');

    if (tabId === 'conductivity') {
        fetchConductivity();
    }
}

// Fetch Fornix Protocol
async function fetchFornixProtocol() {
    const response = await fetch('/api/fornix-protocol');
    const data = await response.json();
    const container = document.getElementById('fornix-protocol-stages');
    container.innerHTML = data.stages.map(s => `
        <div class="stat-card" style="border-left: 3px solid var(--accent-cyan); margin-bottom: 10px;">
            <div style="font-size: 11px; font-weight: 800; color: var(--accent-cyan);">${s.name}</div>
            <div style="font-size: 10px; color: var(--text-dim); margin: 4px 0;">${s.description}</div>
            <div style="font-size: 9px; color: var(--accent-pink);">V: ${s.parameters.voltage} | F: ${s.parameters.freq}</div>
        </div>
    `).join('');
}

// Fetch and Render Conductivity Map
async function fetchConductivity() {
    const response = await fetch('/api/fornix-conductivity');
    const data = await response.json();
    const container = document.getElementById('conductivity-grid-container');
    container.innerHTML = '';

    let total = 0;
    data.grid.forEach(row => {
        row.forEach(val => {
            total += val;
            const cell = document.createElement('div');
            cell.style.aspectRatio = '1';
            // Scale color from deep blue to cyan based on conductivity
            const intensity = Math.min(100, (val - 0.15) * 500);
            cell.style.background = `rgba(0, 242, 255, ${0.1 + intensity / 100})`;
            cell.style.borderRadius = '2px';
            cell.title = `Cond: ${val.toFixed(3)} S/m`;
            container.appendChild(cell);
        });
    });

    document.getElementById('avg-cond-val').textContent = (total / 100).toFixed(3);
}

// Start everything
window.onload = () => {
    init3D();
    drawCorticalFEA();
    fetchSystemSpecs();
    fetchFornixProtocol();
    fetchConductivity(); // Pre-load conductivity
    runSimulation(); // Initial run

    // Auto-refresh telemetry every 5 seconds
    setInterval(fetchSystemSpecs, 5000);
};
