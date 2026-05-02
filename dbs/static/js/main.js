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
    if (!canvas) return;

    if (!window.feaInitialized) {
        window.feaScene = new THREE.Scene();
        window.feaCamera = new THREE.PerspectiveCamera(60, canvas.clientWidth/canvas.clientHeight, 0.1, 100);
        window.feaCamera.position.z = 20;
        
        window.feaRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
        window.feaRenderer.setSize(canvas.clientWidth, canvas.clientHeight);

        const feaNodes = document.getElementById('fea-nodes') ? parseInt(document.getElementById('fea-nodes').value) : 10000;
        // High density detail for volumetric surface
        const detail = Math.min(15, Math.max(5, Math.floor(feaNodes / 200)));
        
        const geo = new THREE.IcosahedronGeometry(7.0, detail); 
        const positions = geo.attributes.position;
        const scalars = new Float32Array(positions.count);
        geo.setAttribute('scalar', new THREE.BufferAttribute(scalars, 1));
        
        for(let i = 0; i < positions.count; i++) {
            let x = positions.getX(i);
            let y = positions.getY(i);
            let z = positions.getZ(i);
            
            // Volumetric Human Cortex Procedural Approximation
            let fissure = 1.0;
            if (Math.abs(x) < 1.5) {
                fissure = 0.4 + 0.6 * (Math.abs(x) / 1.5);
            }
            
            let noise = Math.sin(x*2.0)*Math.cos(y*3.0)*Math.sin(z*2.5) + 
                        0.5*Math.sin(x*5.0+y)*Math.cos(z*4.0) +
                        0.25*Math.cos(x*10.0)*Math.sin(y*10.0+z);
            
            let gyri = Math.pow(Math.abs(noise), 0.6); 
            let baseRadius = 1.0 - 0.12 * gyri; 
            
            let nx = x * baseRadius * fissure * 0.85; 
            let ny = y * baseRadius * 0.8; 
            let nz = z * baseRadius * 1.15; 
            
            if (nz > 0) nx *= (1.0 - 0.1*(nz/7.0));
            if (ny < 0 && Math.abs(x) > 2.0 && nz < 2.0 && nz > -2.0) {
                nx *= 1.1;
                ny *= 1.05;
            }

            positions.setXYZ(i, nx, ny, nz);
        }
        geo.computeVertexNormals();

        // Custom BEM Boundary Element Contours Shader
        const vertexShader = `
            varying vec3 vNormal;
            varying vec3 vPosition;
            attribute float scalar;
            varying float vScalar;
            void main() {
                vNormal = normalize(normalMatrix * normal);
                vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
                vScalar = scalar;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;

        const fragmentShader = `
            varying vec3 vNormal;
            varying vec3 vPosition;
            varying float vScalar;
            
            vec3 colormap(float t) {
                float r = clamp(1.5 - abs(2.0 * t - 1.0), 0.0, 1.0);
                float g = clamp(1.5 - abs(2.0 * t - 1.5), 0.0, 1.0);
                float b = clamp(1.5 - abs(2.0 * t - 2.0), 0.0, 1.0);
                vec3 brainColor = vec3(0.6, 0.5, 0.5);
                return mix(brainColor, vec3(r, g, b), clamp(t * 3.0, 0.0, 1.0));
            }

            void main() {
                float t = clamp(vScalar / 1.5, 0.0, 1.0);
                vec3 color = colormap(t);
                
                // Isolines / Contours 
                float numContours = 15.0; 
                float contour = fract(t * numContours);
                float lineThick = 0.08; 
                
                if (t > 0.02 && (contour < lineThick || contour > 1.0 - lineThick)) {
                    color = vec3(1.0); // White Contour Band
                }
                
                vec3 lightDir = normalize(vec3(0.5, 1.0, 1.0));
                float diff = max(dot(vNormal, lightDir), 0.2);
                
                vec3 viewDir = normalize(-vPosition);
                float rim = 1.0 - max(dot(viewDir, vNormal), 0.0);
                rim = smoothstep(0.6, 1.0, rim);
                
                vec3 finalColor = color * diff + vec3(0.3) * rim;
                gl_FragColor = vec4(finalColor, 0.95);
            }
        `;

        const mat = new THREE.ShaderMaterial({
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            transparent: true,
            side: THREE.DoubleSide
        });
        
        const brainMesh = new THREE.Mesh(geo, mat);
        window.feaScene.add(brainMesh);
        window.brainMesh = brainMesh; 

        const rfEmitterGeo = new THREE.SphereGeometry(0.4, 16, 16);
        const rfEmitterMat = new THREE.MeshBasicMaterial({ color: 0xff00ff });
        const rfEmitter = new THREE.Mesh(rfEmitterGeo, rfEmitterMat);
        rfEmitter.position.set(2.5, 3.5, 2.5); 
        window.feaScene.add(rfEmitter);
        window.rfEmitter = rfEmitter;
        
        window.bemTime = 0;
        window.feaInitialized = true;
    }

    if (window.brainMesh && window.rfEmitter) {
        window.bemTime += 0.05;
        const sourcePos = window.rfEmitter.position;
        const posArray = window.brainMesh.geometry.attributes.position.array;
        const scalarArray = window.brainMesh.geometry.attributes.scalar.array;
        
        const rfRange = document.getElementById('rf-freq-range');
        const pwrRange = document.getElementById('power-eff-range');
        const rf = rfRange ? parseFloat(rfRange.value) : 2.4;
        const pwr = pwrRange ? parseFloat(pwrRange.value) : 90;
        const efficiencyFactor = pwr / 100.0;
        
        for(let i = 0, j = 0; i < posArray.length; i+=3, j++) {
            let dx = posArray[i] - sourcePos.x;
            let dy = posArray[i+1] - sourcePos.y;
            let dz = posArray[i+2] - sourcePos.z;
            let r = Math.sqrt(dx*dx + dy*dy + dz*dz);
            
            // Adjust mathematical manifold with proprioceptive feedback parameters
            let rf_pulse = Math.max(0, Math.sin(window.bemTime * rf - r * (1.5 / efficiencyFactor))); 
            let targetE = (1.0 / (r * r + 0.1)) * rf_pulse * 12.0 * efficiencyFactor;
            
            scalarArray[j] = scalarArray[j] * 0.85 + targetE * 0.15;
        }
        window.brainMesh.geometry.attributes.scalar.needsUpdate = true;
        
        window.brainMesh.rotation.y += 0.003;
        window.brainMesh.rotation.z = Math.sin(window.bemTime * 0.1) * 0.05;
    }
    
    window.feaRenderer.render(window.feaScene, window.feaCamera);
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
    let base_cond = 0.20;
    let anisotropy = 0.1;
    let curvature = 2.0;

    const baseCondMap = document.getElementById('base-cond-map');
    const anisoMap = document.getElementById('aniso-map');
    const curveMap = document.getElementById('curve-map');
    
    if (baseCondMap) {
        base_cond = parseFloat(baseCondMap.value);
        anisotropy = parseFloat(anisoMap.value);
        curvature = parseFloat(curveMap.value);
        
        document.getElementById('base-cond-disp').textContent = base_cond.toFixed(2);
        document.getElementById('aniso-disp').textContent = anisotropy.toFixed(2);
        document.getElementById('curve-disp').textContent = curvature.toFixed(1);
    }
    
    const response = await fetch('/api/fornix-conductivity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            base_cond: base_cond,
            anisotropy: anisotropy,
            curvature: curvature
        })
    });
    
    const data = await response.json();
    const container = document.getElementById('conductivity-grid-container');
    container.innerHTML = '';

    let total = 0;
    let elements = 0;
    data.grid.forEach(row => {
        row.forEach(val => {
            total += val;
            elements += 1;
            const cell = document.createElement('div');
            cell.style.aspectRatio = '1';
            // Adjusted scaling for higher conductivity
            const intensity = Math.min(100, Math.max(0, (val - (base_cond - 0.05)) * (500 / Math.max(0.1, anisotropy * 5))));
            cell.style.background = `rgba(0, 242, 255, ${0.1 + intensity / 100})`;
            cell.style.borderRadius = '2px';
            cell.title = `Cond: ${val.toFixed(3)} S/m`;
            container.appendChild(cell);
        });
    });

    document.getElementById('avg-cond-val').textContent = (total / elements).toFixed(3);
}

document.addEventListener('DOMContentLoaded', () => {
    const updateBtn = document.getElementById('btn-update-conductivity');
    if (updateBtn) {
        updateBtn.addEventListener('click', fetchConductivity);
    }
    const sliders = ['base-cond-map', 'aniso-map', 'curve-map'];
    sliders.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', fetchConductivity);
    });
});

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

// --- DEMENTIA STAGING GENERATIVE PROTOCOL ---

let dementiaChart = null;

async function runDementiaStaging() {
    const prompt = document.getElementById('gen-ai-prompt').value;
    const declineRate = parseFloat(document.getElementById('decline-range').value);
    const dbsAmp = parseFloat(document.getElementById('dementia-dbs-amp').value);

    // Call backend endpoint
    const response = await fetch('/api/dementia-staging', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt, decline_rate: declineRate, dbs_amplitude: dbsAmp })
    });
    
    if (!response.ok) return;
    const res = await response.json();
    
    // Update text
    document.getElementById('dementia-insight').innerText = res.generative_insight;
    
    // Render Chart
    const ctx = document.getElementById('dementia-chart');
    if (!ctx) return;
    if (dementiaChart) dementiaChart.destroy();
    
    const times = res.time_months;
    const traj = res.cognitive_trajectory;
    // Map variance
    const upper_bound = res.clinical_distributions.map(d => d.mean + d.std/1.5);
    const lower_bound = res.clinical_distributions.map(d => Math.max(0, d.mean - d.std/1.5));

    dementiaChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: times.map(t => Math.round(t)),
            datasets: [
                {
                    label: 'Clinical Variance (Upper)',
                    data: upper_bound,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(0, 242, 255, 0.2)',
                    fill: '+1',
                    pointRadius: 0,
                    tension: 0.4
                },
                {
                    label: 'Clinical Variance (Lower)',
                    data: lower_bound,
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    fill: false,
                    pointRadius: 0,
                    tension: 0.4
                },
                {
                    label: 'Temporal Cognitive Trajectory',
                    data: traj,
                    borderColor: '#ff00c8',
                    borderWidth: 3,
                    fill: false,
                    pointRadius: 1,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, labels: { color: '#a0aab5' } }
            },
            scales: {
                x: { title: { display: true, text: 'Treatment Timeline (Months)', color: '#00f2ff'}, ticks: { color: '#a0aab5' } },
                y: { title: { display: true, text: 'Structural Neurological Retention (Score)', color: '#00f2ff'}, ticks: { color: '#a0aab5' }, min: 0, max: 35 }
            }
        }
    });
}


// Auto trigger bindings
document.addEventListener("DOMContentLoaded", () => {
    const dr = document.getElementById('decline-range');
    const da = document.getElementById('dementia-dbs-amp');
    const ga = document.getElementById('gen-ai-prompt');
    
    if (dr) dr.addEventListener('input', runDementiaStaging);
    if (da) da.addEventListener('input', runDementiaStaging);
    if (ga) ga.addEventListener('change', runDementiaStaging);
    
    // Attempt init run
    setTimeout(runDementiaStaging, 500); 
});


// --- Added Dementia Optimization & FEA Functionality ---

let dementiaChartInstance = null;

function updateDementiaChart() {
    const dbsAmp = document.getElementById('voltage-range') ? document.getElementById('voltage-range').value : 30; // mapping voltage to amplitude 
    const declineRate = document.getElementById('dementia-decline-range') ? document.getElementById('dementia-decline-range').value : 0.05;
    const prompt = document.getElementById('dementia-prompt') ? document.getElementById('dementia-prompt').value : 'baseline';
    
    fetch('/api/dementia-staging', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            dbs_amplitude: dbsAmp, 
            decline_rate: declineRate,
            prompt: prompt
        })
    })
    .then(res => res.json())
    .then(data => {
        const ctx = document.getElementById('dementia-chart')?.getContext('2d');
        if (!ctx) return;
        
        if (dementiaChartInstance) {
            dementiaChartInstance.destroy();
        }
        
        const upperBounds = data.cognitive_trajectory.map((val, i) => val + data.clinical_distributions[i].std);
        const lowerBounds = data.cognitive_trajectory.map((val, i) => Math.max(0, val - data.clinical_distributions[i].std));
        
        dementiaChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.time_months,
                datasets: [
                    {
                        label: 'Mean Trajectory',
                        data: data.cognitive_trajectory,
                        borderColor: '#00ffcc',
                        backgroundColor: 'rgba(0, 255, 204, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: '+1 Std Dev',
                        data: upperBounds,
                        borderColor: 'rgba(255, 0, 127, 0.5)',
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0,
                    },
                    {
                        label: '-1 Std Dev',
                        data: lowerBounds,
                        borderColor: 'rgba(255, 0, 127, 0.5)',
                        borderDash: [5, 5],
                        fill: '-1',
                        backgroundColor: 'rgba(255, 0, 127, 0.1)',
                        pointRadius: 0,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Months' } },
                    y: { title: { display: true, text: 'Cognitive Score (MMSE)' }, min: 0, max: 30 }
                }
            }
        });

        const insightElem = document.getElementById('dementia-insight');
        const varianceElem = document.getElementById('dementia-variance');
        if (insightElem) insightElem.innerText = data.generative_insight || 'Temporal projection stabilized via non-linear constraints.';
        
        let initialStd = data.clinical_distributions[0].std;
        let finalStd = data.clinical_distributions[data.clinical_distributions.length - 1].std;
        if (varianceElem) varianceElem.innerText = `Std Dev variance ranges from $\pm${initialStd} to $\pm${finalStd} over 60 months governed by continuous Markovian decay mappings.`;

    }).catch(e => console.error("Error charting dementia:", e));
}

let largeFeaScene, largeFeaCamera, largeFeaRenderer, largeFeaMesh, rfCoilMesh, emFieldParticles;
let emParticlesGeo;

function initLargerFEA() {
    const container = document.getElementById('fea-large-container');
    if (!container) return;
    
    // Clear past children
    container.innerHTML = '';
    
    largeFeaScene = new THREE.Scene();
    largeFeaCamera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
    largeFeaCamera.position.z = 20;
    
    largeFeaRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    largeFeaRenderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(largeFeaRenderer.domElement);
    
    // --- CORTICAL SURFACE MANIFOLD
    const feaNodes = document.getElementById('fea-nodes') ? parseInt(document.getElementById('fea-nodes').value) : 1000;
    const detail = Math.min(6, Math.max(1, Math.floor(feaNodes / 200)));
    
    const geo = new THREE.IcosahedronGeometry(6, detail);
    const positions = geo.attributes.position;
    for(let i = 0; i < positions.count; i++) {
        let x = positions.getX(i);
        let y = positions.getY(i);
        let z = positions.getZ(i);
        let bump = 1 + 0.15 * Math.sin(x*2) * Math.cos(y*2) + 0.05 * Math.sin(z*4);
        positions.setXYZ(i, x*bump, y*bump, z*bump);
    }
    geo.computeVertexNormals();
    
    // Heatmap style coloring based on Z position for basic FEA visual
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(positions.count * 3), 3));
    const colors = geo.attributes.color;
    for(let i = 0; i < positions.count; i++) {
        const val = (positions.getY(i) + 6) / 12; // roughly 0 to 1
        colors.setXYZ(i, 1.0, val, 0.2); // yellow-red FEA heat gradient
    }

    const mat = new THREE.MeshPhongMaterial({
        vertexColors: true,
        emissive: 0x221100,
        wireframe: true,
        transparent: true,
        opacity: 0.8,
        side: THREE.DoubleSide
    });
    
    largeFeaMesh = new THREE.Mesh(geo, mat);
    largeFeaScene.add(largeFeaMesh);

    // --- RF COIL CIRCUITRY
    const coilGeo = new THREE.TorusKnotGeometry( 8.5, 0.3, 150, 16, 2, 5 );
    const coilMat = new THREE.MeshStandardMaterial({ 
        color: 0xaaaaaa, 
        metalness: 0.9, 
        roughness: 0.1,
        emissive: 0x001155
    });
    rfCoilMesh = new THREE.Mesh(coilGeo, coilMat);
    largeFeaScene.add(rfCoilMesh);

    // --- ELECTROMAGNETIC FIELD PARTICLES
    emParticlesGeo = new THREE.BufferGeometry();
    const pCount = 1000;
    const pArray = new Float32Array(pCount * 3);
    for(let i=0; i<pCount*3; i++) {
        pArray[i] = (Math.random() - 0.5) * 35; // Random spread
    }
    emParticlesGeo.setAttribute('position', new THREE.BufferAttribute(pArray, 3));
    const particleMat = new THREE.PointsMaterial({
        color: 0x00ffff,
        size: 0.25,
        transparent: true,
        opacity: 0.6,
        blending: THREE.AdditiveBlending
    });
    emFieldParticles = new THREE.Points(emParticlesGeo, particleMat);
    largeFeaScene.add(emFieldParticles);
    
    // LIGHTING
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(10, 20, 10);
    largeFeaScene.add(light);
    const ambientLight = new THREE.AmbientLight(0x404040); 
    largeFeaScene.add(ambientLight);
    
    animateLargeFEA();
}

function animateLargeFEA() {
    if(!largeFeaRenderer) return;
    requestAnimationFrame(animateLargeFEA);
    
    const time = Date.now() * 0.001;

    if(largeFeaMesh) {
        largeFeaMesh.rotation.y += 0.002;
    }
    if(rfCoilMesh) {
        rfCoilMesh.rotation.y -= 0.005;
        rfCoilMesh.rotation.x = Math.sin(time*0.5) * 0.2;
        // Pulse the emissive color of the coil to simulate RF activation
        rfCoilMesh.material.emissiveIntensity = 0.5 + 0.5 * Math.sin(time * 8);
    }
    
    // Animate EM Field particles to simulate magnetic flux toroidal vortex
    if(emFieldParticles) {
        const positions = emParticlesGeo.attributes.position.array;
        for(let i=0; i<positions.length; i+=3) {
            let x = positions[i];
            let y = positions[i+1];
            let z = positions[i+2];

            const r = Math.sqrt(x*x + z*z) + 0.0001;
            const theta = Math.atan2(z, x) + 0.015; 
            positions[i] = r * Math.cos(theta);
            positions[i+2] = r * Math.sin(theta);
            positions[i+1] += (15 / r) * 0.1 * Math.sin(time * 3 + r); 

            if(positions[i+1] > 17) positions[i+1] = -17; 
            if(positions[i+1] < -17) positions[i+1] = 17;
        }
        emParticlesGeo.attributes.position.needsUpdate = true;
    }

    largeFeaRenderer.render(largeFeaScene, largeFeaCamera);
}

// Automatically load models if clicking their tab
window.addEventListener('click', (e) => {
    if(e.target.classList.contains('tab-btn')) {
        setTimeout(() => {
            if(document.getElementById('dementia-sidebar') && document.getElementById('dementia-sidebar').classList.contains('active')) {
                updateDementiaChart();
            }
            if(document.getElementById('fea-sidebar') && document.getElementById('fea-sidebar').classList.contains('active') && !largeFeaRenderer) {
                initLargerFEA();
            }
        }, 100);
    }
});



// Stage-Gated Dementia Protocol (Queueing Theory)
async function fetchStageProtocol() {
    try {
        const response = await fetch('/api/stage-gated-protocol');
        const data = await response.json();
        
        const container = document.getElementById('stage-protocol-container');
        container.innerHTML = data.protocol.map(stage => `
            <div class="stat-card" style="border-left: 3px solid var(--accent-cyan); display: flex; flex-direction: column; gap: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin:0; color: #00f2ff;">${stage.name}</h3>
                    <span style="font-size:10px; padding: 3px 8px; background: rgba(255,0,200,0.2); border-radius: 12px; color: var(--accent-pink);">
                        Stage ${stage.stage}
                    </span>
                </div>
                <p style="font-size: 11px; margin: 0; color: var(--text-dim);">${stage.desc}</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 5px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
                    <div>
                        <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase;">Electrical Protocol</div>
                        <ul style="margin: 5px 0 0 15px; font-size: 11px; color: #fff;">
                            <li>Voltage: <span style="color:var(--accent-cyan);">${stage.electrical.voltage_v} V</span></li>
                            <li>Frequency: <span style="color:var(--accent-cyan);">${stage.electrical.frequency_hz} Hz</span></li>
                            <li>Pulse Width: <span style="color:var(--accent-cyan);">${stage.electrical.pulse_width_us} µs</span></li>
                            <li>Target: <span style="color:var(--accent-cyan);">${stage.electrical.target}</span></li>
                        </ul>
                    </div>
                    <div>
                        <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase;">Molecular Queueing (M/M/1)</div>
                        <ul style="margin: 5px 0 0 15px; font-size: 11px; color: #fff;">
                            <li>Tau Aggregation Rate (λ): <span style="color:var(--accent-pink);">${stage.queueing.lambda_arrival} /yr</span></li>
                            <li>Glymphatic Clearance (μ): <span style="color:var(--accent-pink);">${stage.queueing.mu_clearance} /yr</span></li>
                            <li>System Utilization (ρ): <span style="color:var(--accent-pink);">${stage.queueing.rho_utilization}</span></li>
                            <li>Queue Length (Lq): <span style="color:var(--accent-pink);">${stage.queueing.l_q}</span></li>
                        </ul>
                    </div>
                </div>
            </div>
        `).join('');
    } catch(err) {
        console.error(err);
    }
}

function loadClinicalProtocols() {
    const listContainer = document.getElementById('protocols-list-container');
    listContainer.innerHTML = '<p style="color: #00f2ff;">Analyzing deep brain targets...<br>Compiling dementia stimulation parameters...</p>';
    
    fetch('/api/clinical-protocols', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(r => r.json())
    .then(data => {
        let html = '';
        data.protocols.forEach((p, idx) => {
            html += `
                <div style="background: rgba(0, 242, 255, 0.05); border: 1px solid rgba(0, 242, 255, 0.2); padding: 15px; margin-bottom: 20px; border-radius: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: top; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; margin-bottom: 10px;">
                        <h3 style="color: #fff; margin: 0; font-size: 16px;">Target ${idx + 1}: ${p.lobe}</h3>
                        <span style="background: var(--accent-pink); padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; color: white;">Analysis Complete</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 10px; font-family: monospace;">
                        <div style="background: rgba(0,0,0,0.5); padding: 8px; border: 1px dashed rgba(255,255,255,0.15); border-radius: 4px;">
                            <div style="color: var(--text-dim); font-size: 10px;">FREQUENCY</div>
                            <div style="color: #00ff00; font-size: 14px; font-weight: bold;">${p.frequency}</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.5); padding: 8px; border: 1px dashed rgba(255,255,255,0.15); border-radius: 4px;">
                            <div style="color: var(--text-dim); font-size: 10px;">PULSE WIDTH</div>
                            <div style="color: #00f2ff; font-size: 14px; font-weight: bold;">${p.pulse_width}</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.5); padding: 8px; border: 1px dashed rgba(255,255,255,0.15); border-radius: 4px;">
                            <div style="color: var(--text-dim); font-size: 10px;">VOLTAGE OPTIMA</div>
                            <div style="color: #ff00ff; font-size: 14px; font-weight: bold;">${p.voltage}</div>
                        </div>
                    </div>
                    <div>
                        <div style="color: var(--text-dim); font-size: 11px; text-transform: uppercase; margin-bottom: 5px;">Mechanistic Rationale</div>
                        <div style="color: #ddd; font-size: 13px; line-height: 1.4;">${p.description}</div>
                    </div>
                </div>
            `;
        });
        listContainer.innerHTML = html;
    })
    .catch(e => {
        console.error(e);
        listContainer.innerHTML = '<p style="color: red;">Error fetching protocol generation analysis.</p>';
    });
}

let paretoChartInstance = null;

async function runParetoOptimization() {
    const lambda = document.getElementById("pareto-lambda-map").value;
    
    // API request to math engine
    try {
        const res = await fetch('/api/pareto_frontier', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lambda: lambda })
        });
        const data = await res.json();
        
        // Update Chart
        const ctx = document.getElementById('pareto-chart').getContext('2d');
        if (paretoChartInstance) {
            paretoChartInstance.destroy();
        }
        
        // x axis represents generic trade-off continuous variable
        const labels = Array.from({length: 100}, (_, i) => (i / 100).toFixed(2));
        
        paretoChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Striatal Activation Target (%)',
                        data: data.striatal,
                        borderColor: '#00ffcc',
                        tension: 0.4
                    },
                    {
                        label: 'Serotonin Release Yield (%)',
                        data: data.serotonin,
                        borderColor: '#ff00ff',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#a0a0b0' } },
                    annotation: {
                        annotations: {
                            line1: {
                                type: 'line',
                                xMin: data.optimal_x.toFixed(2),
                                xMax: data.optimal_x.toFixed(2),
                                borderColor: 'white',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {
                                    content: 'Nash Equilibrium',
                                    enabled: true,
                                    position: 'top'
                                }
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 120,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#a0a0b0' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#a0a0b0', maxTicksLimit: 10 }
                    }
                }
            }
        });
        
        document.getElementById('pareto-yield').innerText = data.optimal_striatal.toFixed(1) + "%";
        document.getElementById('pareto-serotonin').innerText = data.optimal_serotonin.toFixed(1) + "%";
        
    } catch(err) {
        console.error("Error drawing pareto chart: ", err);
    }
}
