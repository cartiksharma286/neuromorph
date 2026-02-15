const API_URL = 'http://localhost:5005/api';

// State
let simInterval = null;
let isSimulating = false;
let currentGrid = null;
let charts = {};
let nvqPollInterval = null;

// DOM Elements (Cached)
const els = {
    views: {},
    navBtns: document.querySelectorAll('.nav-btn'),
    trainBtn: document.getElementById('trainBtn'),
    trainingState: document.getElementById('trainingState'),
    riskValue: document.getElementById('riskValue'),
    inputs: ['inTemp', 'inHum', 'inWind', 'inBio'].map(id => document.getElementById(id)),
    simStats: { ign: document.getElementById('ignCount'), burnt: document.getElementById('burntCount') },
    nvq: {
        statusText: document.getElementById('nvqStatusText'),
        bandwidth: document.getElementById('nvqBandwidth'),
        latency: document.getElementById('nvqLatency'),
        quantum: document.getElementById('nvqQuantum'),
        nodes: document.getElementById('nvqNodes'),
        terminal: document.getElementById('nvqTerminal')
    }
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    // Setup Views
    els.navBtns.forEach(btn => {
        const viewId = btn.dataset.view;
        els.views[viewId] = document.getElementById(viewId);
        btn.addEventListener('click', () => switchView(viewId));
    });

    // Inputs
    els.inputs.forEach(input => input.addEventListener('input', livePredict));

    // Buttons
    if (els.trainBtn) els.trainBtn.addEventListener('click', trainModel);
    document.getElementById('loadTerrainBtn').addEventListener('click', loadTerrain);
    document.getElementById('startSimBtn').addEventListener('click', startSim);
    document.getElementById('stopSimBtn').addEventListener('click', stopSim);

    // Initial Load
    switchView('dashboard');
    initCharts();
    trainModel(true); // Initial training silently

    // Start Global Poller
    startNVQPoller();
});

// --- Navigation ---
function switchView(id) {
    // Update Buttons
    els.navBtns.forEach(btn => {
        if (btn.dataset.view === id) btn.classList.add('active');
        else btn.classList.remove('active');
    });

    // Update Section Title
    const titles = {
        'dashboard': 'Mission Control',
        'science': 'Data Science Engine',
        'simulation': 'Geospatial Fire Sim',
        'nvqlink': 'Quantum Telemetry'
    };
    document.getElementById('section-title').innerText = titles[id] || 'EcoGeo';

    // Show View
    document.querySelectorAll('.view').forEach(el => {
        el.classList.remove('active', 'fade-in');
        if (el.id === id) {
            el.classList.add('active', 'fade-in');
        }
    });

    // Chart Resizes
    if (id === 'dashboard' && charts.main) setTimeout(() => charts.main.resize(), 50);
    if (id === 'science' && charts.features) setTimeout(() => charts.features.resize(), 50);
}

// --- Data Science ---
async function trainModel(silent = false) {
    if (!silent) {
        els.trainBtn.disabled = true;
        els.trainBtn.querySelector('.btn-text').innerText = 'Training...';
        els.trainingState.innerText = 'Optimizing SGD...';
    }

    try {
        const res = await fetch(`${API_URL}/model/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ samples: 500 }) // Fast batch
        });
        const data = await res.json();

        if (!silent) {
            els.trainingState.innerText = `Ready (Acc: ${(data.accuracy * 100).toFixed(1)}%)`;
            els.trainBtn.disabled = false;
            els.trainBtn.querySelector('.btn-text').innerText = 'Retrain Model';
        }

        // Update Dashboard
        const dashAcc = document.getElementById('dashAccuracy');
        if (dashAcc) dashAcc.innerText = (data.accuracy * 100).toFixed(1) + "%";

        updateFeatureChart(data.coefficients);
        livePredict(); // Refresh prediction

    } catch (e) {
        console.error("Training Error:", e);
        if (!silent) els.trainingState.innerText = 'Error Connecting';
        els.trainBtn.disabled = false;
        els.trainBtn.querySelector('.btn-text').innerText = 'Try Again';
    }
}

async function livePredict() {
    const inputs = els.inputs.map(el => parseFloat(el.value));

    try {
        const res = await fetch(`${API_URL}/model/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: inputs })
        });
        const data = await res.json();

        const prob = data.fire_risk_probability;
        els.riskValue.innerText = (prob * 100).toFixed(1) + "%";

        // Color coding
        if (prob < 0.3) els.riskValue.style.color = '#10b981'; // Green
        else if (prob < 0.7) els.riskValue.style.color = '#f59e0b'; // Orange
        else els.riskValue.style.color = '#ef4444'; // Red

    } catch (e) { console.error(e); }
}

function updateFeatureChart(coeffs) {
    if (!charts.features) return;

    charts.features.data.datasets[0].data = coeffs;
    charts.features.update();
}

// --- Simulation ---
async function loadTerrain() {
    const loc = document.getElementById('locSelect').value;
    try {
        const res = await fetch(`${API_URL}/lidar/scan?location=${loc}`);
        const data = await res.json();

        const size = data.grid_metrics.density.length;
        currentGrid = {
            density: data.grid_metrics.density,
            moisture: data.grid_metrics.moisture,
            fire_state: Array(size).fill().map(() => Array(size).fill(0))
        };

        // Ignite Center
        const cx = Math.floor(size / 2);
        currentGrid.fire_state[cx][cx] = 1;

        renderGrid();
        updateSimStats(0, 0);

    } catch (e) { alert("Failed to load terrain: " + e); }
}

async function startSim() {
    if (isSimulating) return;
    if (!currentGrid) { await loadTerrain(); }
    isSimulating = true;
    simLoop();
}

function stopSim() {
    isSimulating = false;
}

async function simLoop() {
    if (!isSimulating) return;

    try {
        const res = await fetch(`${API_URL}/simulation/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ grid: currentGrid })
        });
        currentGrid = await res.json();

        renderGrid();
        updateSimStats(currentGrid.nt_ignitions, currentGrid.nt_burnt);

        if (isSimulating) requestAnimationFrame(simLoop);

    } catch (e) {
        console.error("Sim Step Error:", e);
        stopSim();
    }
}

function renderGrid() {
    const canvas = document.getElementById('simCanvas');
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (!currentGrid) return;

    const gridSize = currentGrid.density.length;
    const cw = w / gridSize;
    const ch = h / gridSize;

    for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
            const state = currentGrid.fire_state[y][x];
            const density = currentGrid.density[y][x];
            const moisture = currentGrid.moisture[y][x];

            if (state === 2) { // Burnt
                ctx.fillStyle = '#0f172a'; // Dark Burnt
            } else if (state === 1) { // Burning
                const flicker = Math.random() * 50;
                ctx.fillStyle = `rgb(${220 + flicker}, ${80 + (flicker / 2)}, 0)`;
            } else { // Vegetation
                // Green based on density, blue tint for moisture
                const g = 30 + (density * 100);
                const b = 20 + (moisture * 50);
                ctx.fillStyle = `rgb(10, ${g}, ${b})`;
            }
            ctx.fillRect(x * cw, y * ch, cw, ch);
        }
    }
}

function updateSimStats(ign, burnt) {
    els.simStats.ign.innerText = ign;
    els.simStats.burnt.innerText = burnt;
}

// --- NVQLink ---
function startNVQPoller() {
    setInterval(async () => {
        // Only fetch if tab is active or just generally keep alive? 
        // Let's fetch always but update UI only if visible for robustness
        try {
            const res = await fetch(`${API_URL}/nvqlink/status`);
            const data = await res.json();

            if (els.nvq.statusText) {
                els.nvq.statusText.innerText = data.status;
                els.nvq.statusText.style.color = '#10b981'; // Success Green
                els.nvq.bandwidth.innerText = data.bandwidth;
                els.nvq.latency.innerText = data.latency_ms + "ms";
                els.nvq.quantum.innerText = data.quantum_state;
                els.nvq.nodes.innerText = data.nodes_active;

                // Add log line occasionally
                if (Math.random() > 0.7) {
                    const line = document.createElement('div');
                    line.innerText = `> Packet received from Node ${Math.floor(Math.random() * 12)} [${Date.now()}]`;
                    els.nvq.terminal.appendChild(line);
                    els.nvq.terminal.scrollTop = els.nvq.terminal.scrollHeight;
                }
            }
        } catch (e) {
            if (els.nvq.statusText) {
                els.nvq.statusText.innerText = "OFFLINE";
                els.nvq.statusText.style.color = "#ef4444";
            }
        }
    }, 1500);
}

// --- Charts ---
function initCharts() {
    // Environmental Trends (Dashboard)
    const ctxMain = document.getElementById('mainChart').getContext('2d');
    const gradRisk = ctxMain.createLinearGradient(0, 0, 0, 300);
    gradRisk.addColorStop(0, 'rgba(239, 68, 68, 0.4)');
    gradRisk.addColorStop(1, 'rgba(239, 68, 68, 0)');

    charts.main = new Chart(ctxMain, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'],
            datasets: [
                {
                    label: 'Fire Risk',
                    data: [0.12, 0.15, 0.25, 0.45, 0.70, 0.88, 0.92, 0.85],
                    borderColor: '#ef4444',
                    backgroundColor: gradRisk,
                    fill: true,
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Avg Temp (°C)',
                    data: [4, 6, 11, 16, 22, 27, 31, 29],
                    type: 'bar',
                    backgroundColor: '#3b82f6',
                    yAxisID: 'y1',
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { labels: { color: '#94a3b8' } } },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
                y: { type: 'linear', display: true, position: 'left', grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#ef4444' } },
                y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#3b82f6' } },
            }
        }
    });

    // Feature Importance (Science)
    const ctxFeat = document.getElementById('featureChart').getContext('2d');
    charts.features = new Chart(ctxFeat, {
        type: 'bar',
        data: {
            labels: ['Temperature', 'Humidity', 'Wind Speed', 'Biomass'],
            datasets: [{
                label: 'Coefficient Impact',
                data: [0, 0, 0, 0], // Initial
                backgroundColor: ['#ef4444', '#06b6d4', '#f59e0b', '#10b981'],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}
