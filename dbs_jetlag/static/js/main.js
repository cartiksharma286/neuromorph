
let scene, camera, renderer, neurons = [];
let isSimulating = false;
let phaseChart, mitigationChart;
let history = [];

// Initialize 3D
function init3D() {
    const container = document.getElementById('brain-viz-container');
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Create a sphere of neurons representing the SCN
    const geo = new THREE.SphereGeometry(0.1, 16, 16);
    for (let i = 0; i < 150; i++) {
        const mat = new THREE.MeshBasicMaterial({ color: 0x00f2ff });
        const mesh = new THREE.Mesh(geo, mat);
        // Random shell distribution
        const phi = Math.acos(-1 + (2 * i) / 150);
        const theta = Math.sqrt(150 * Math.PI) * phi;
        mesh.position.set(
            4 * Math.cos(theta) * Math.sin(phi),
            4 * Math.sin(theta) * Math.sin(phi),
            4 * Math.cos(phi)
        );
        scene.add(mesh);
        neurons.push(mesh);
    }

    // Add central "target phase" guide
    const ringGeo = new THREE.TorusGeometry(5, 0.02, 16, 100);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0xff00c8, transparent: true, opacity: 0.3 });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    scene.add(ring);

    camera.position.z = 12;
    animate();
}

function animate() {
    requestAnimationFrame(animate);
    scene.rotation.y += 0.005;
    renderer.render(scene, camera);
}

// Charts
function initCharts() {
    const ctx1 = document.getElementById('phase-chart').getContext('2d');
    phaseChart = new Chart(ctx1, {
        type: 'radar',
        data: {
            labels: Array.from({length: 12}, (_, i) => `${i*30}°`),
            datasets: [{
                label: 'Population Distribution',
                data: Array(12).fill(0),
                borderColor: '#00f2ff',
                backgroundColor: 'rgba(0, 242, 255, 0.2)'
            }]
        },
        options: {
            scales: { r: { grid: { color: 'rgba(255,255,255,0.1)' }, angleLines: { color: 'rgba(255,255,255,0.1)' }, ticks: { display: false } } },
            plugins: { legend: { display: false } }
        }
    });

    const ctx2 = document.getElementById('mitigation-chart').getContext('2d');
    mitigationChart = new Chart(ctx2, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Order Parameter',
                data: [],
                borderColor: '#ff00c8',
                tension: 0.4,
                fill: true,
                backgroundColor: 'rgba(255, 0, 200, 0.1)'
            }]
        },
        options: {
            scales: {
                x: { display: false },
                y: { min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.05)' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

async function simStep() {
    if (!isSimulating) return;

    const params = {
        dbs_intensity: parseFloat(document.getElementById('dbs-intensity').value),
        hebbian_rate: parseFloat(document.getElementById('hebbian-rate').value),
        prime_mod: parseInt(document.getElementById('prime-mod').value),
        target_phase: parseFloat(document.getElementById('target-phase').value)
    };

    const response = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    });

    const data = await response.json();
    updateUI(data);
    setTimeout(simStep, 100);
}

function updateUI(data) {
    // Update stats
    document.getElementById('order-param').innerText = data.order_parameter.toFixed(3);
    document.getElementById('pruned-count').innerText = data.pruned_count;

    // Update 3D Neurons
    data.phases.forEach((phase, i) => {
        const colorVal = phase / (2 * Math.PI);
        neurons[i].material.color.setHSL(colorVal, 1, 0.5);
        // "Pulse" distance based on phase
        const scale = 0.8 + 0.4 * Math.sin(phase);
        neurons[i].scale.set(scale, scale, scale);
    });

    // Update Radar Chart
    const bins = Array(12).fill(0);
    data.phases.forEach(p => {
        const bin = Math.floor((p / (2 * Math.PI)) * 12);
        bins[bin]++;
    });
    phaseChart.data.datasets[0].data = bins;
    phaseChart.update('none');

    // Update Line Chart
    history.push(data.order_parameter);
    if (history.length > 50) history.shift();
    mitigationChart.data.labels = history.map((_, i) => i);
    mitigationChart.data.datasets[0].data = history;
    mitigationChart.update('none');

    // Insight text
    if (data.order_parameter > 0.9) {
        document.getElementById('insight-text').innerText = "Coherent Circadian Resonance established. Jet lag minimized.";
        document.getElementById('status-dot').style.background = "#00ff00";
    } else {
        document.getElementById('insight-text').innerText = "Hebbian coupling amplifying SCN representational energy...";
        document.getElementById('status-dot').style.background = "#ffcc00";
    }
}

// Event Listeners
document.querySelectorAll('input[type="range"]').forEach(el => {
    el.addEventListener('input', (e) => {
        document.getElementById(e.target.id + '-val').innerText = e.target.value;
    });
});

document.getElementById('toggle-sim').addEventListener('click', (e) => {
    isSimulating = !isSimulating;
    e.target.innerText = isSimulating ? "HALT MITIGATION" : "START MITIGATION";
    if (isSimulating) simStep();
});

document.getElementById('reset-sim').addEventListener('click', async () => {
    await fetch('/api/reset', { method: 'POST' });
    history = [];
    mitigationChart.data.labels = [];
    mitigationChart.data.datasets[0].data = [];
    mitigationChart.update();
});

window.addEventListener('resize', () => {
    const container = document.getElementById('brain-viz-container');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});

window.onload = () => {
    init3D();
    initCharts();
};
