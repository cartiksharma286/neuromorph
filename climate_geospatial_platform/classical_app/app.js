const API_BASE = 'http://localhost:5001/api';

let simInterval = null;
let currentGrid = null;
let chartInstance = null;
let featureChartInstance = null;
let lastCoeffs = null;

// --- Navigation ---
function setView(id) {
    document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.menu-item').forEach(el => el.classList.remove('active'));
    document.getElementById(id).classList.add('active');

    // Find button for this view (simple matching via ID or index manual mapping)
    const items = document.querySelectorAll('.menu-item');
    if (id === 'dashboard') items[0].classList.add('active');
    if (id === 'science') items[1].classList.add('active');
    if (id === 'simulation') items[2].classList.add('active');
    if (id === 'nvqlink') items[3].classList.add('active');

    // Trigger updates
    if (id === 'dashboard') {
        setTimeout(initDashboardChart, 50);
    }
    if (id === 'science' && lastCoeffs) {
        setTimeout(() => updateFeatureChart(lastCoeffs), 100);
    }
}

// --- Data Science ---
async function trainModel() {
    const statsBox = document.getElementById('trainingStats');
    if (statsBox) statsBox.innerHTML = "Training in progress...";

    try {
        const res = await fetch(`${API_BASE}/model/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ samples: 2000 })
        });
        const data = await res.json();

        lastCoeffs = data.coefficients; // Store for re-render

        if (statsBox) {
            statsBox.innerHTML = `
                <strong>${data.model_type}</strong><br>
                Accuracy: ${(data.accuracy * 100).toFixed(1)}%<br>
                Time: ${(data.training_time * 1000).toFixed(0)}ms
            `;
        }

        if (document.getElementById('dashAccuracy')) {
            document.getElementById('dashAccuracy').innerText = (data.accuracy * 100).toFixed(1) + "%";
        }

        updateFeatureChart(data.coefficients);
        livePredict();

    } catch (e) {
        if (statsBox) statsBox.innerHTML = "Error Training Model";
        console.error(e);
    }
}

async function livePredict() {
    const inTemp = document.getElementById('inTemp');
    if (!inTemp) return;

    const inputs = [
        parseFloat(inTemp.value),
        parseFloat(document.getElementById('inHum').value),
        parseFloat(document.getElementById('inWind').value),
        parseFloat(document.getElementById('inBio').value)
    ];

    try {
        const res = await fetch(`${API_BASE}/model/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: inputs })
        });
        const data = await res.json();

        const risk = data.fire_risk_probability;
        const el = document.getElementById('riskValue');
        if (el) {
            el.innerText = (risk * 100).toFixed(1) + "%";
            if (risk < 0.3) el.style.color = '#10b981';
            else if (risk < 0.7) el.style.color = '#f59e0b';
            else el.style.color = '#ef4444';
        }

    } catch (e) { console.error(e); }
}

function updateFeatureChart(coeffs) {
    const canvas = document.getElementById('featureChart');
    if (!canvas) return;

    // Safety check if Chart is loaded
    if (typeof Chart === 'undefined') return;

    if (featureChartInstance) featureChartInstance.destroy();

    featureChartInstance = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: ['Temp', 'Humidity', 'Wind', 'Biomass'],
            datasets: [{
                label: 'Coefficient Impact',
                data: coeffs,
                backgroundColor: ['#ef4444', '#3b82f6', '#f59e0b', '#10b981'],
                borderWidth: 0,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#e2e8f0',
                    bodyColor: '#e2e8f0'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

// --- Simulation ---
async function loadScan() {
    const loc = document.getElementById('locSelect').value;
    try {
        const res = await fetch(`${API_BASE}/lidar/scan?location=${loc}`);
        const data = await res.json();

        const w = 64;
        currentGrid = {
            density: data.grid_metrics.density,
            moisture: data.grid_metrics.moisture,
            fire_state: Array(w).fill().map(() => Array(w).fill(0))
        };

        const cx = Math.floor(w / 2);
        currentGrid.fire_state[cx][cx] = 1;

        renderGrid();

    } catch (e) { console.error(e); }
}

async function startSim() {
    if (simInterval) return;
    simInterval = setInterval(runSimStep, 100);
}

function stopSim() {
    if (simInterval) clearInterval(simInterval);
    simInterval = null;
}

async function runSimStep() {
    if (!currentGrid) return;
    try {
        const res = await fetch(`${API_BASE}/simulation/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ grid: currentGrid })
        });
        currentGrid = await res.json();
        renderGrid();
        const statEl = document.getElementById('simStats');
        if (statEl) statEl.innerText = `Burnt: ${currentGrid.nt_burnt} | Active: ${currentGrid.nt_ignitions}`;

    } catch (e) { console.error(e); stopSim(); }
}

function renderGrid() {
    const canvas = document.getElementById('simCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    if (!currentGrid) return;

    const gw = currentGrid.density.length;
    const cw = w / gw;
    const ch = h / gw;

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    for (let y = 0; y < gw; y++) {
        for (let x = 0; x < gw; x++) {
            const state = currentGrid.fire_state[y][x];
            const dens = currentGrid.density[y][x];

            if (state === 2) {
                // Burnt area
                ctx.fillStyle = '#1e293b';
                ctx.fillRect(x * cw, y * ch, cw, ch);
            } else if (state === 1) {
                // Burning - Dynamic
                const flicker = Math.random() * 50;
                ctx.fillStyle = `rgb(${200 + flicker}, ${50 + flicker}, 20)`;
                ctx.fillRect(x * cw, y * ch, cw, ch);
            } else {
                // Vegetation - varied green
                const val = 20 + Math.floor(dens * 100);
                const moistBlue = currentGrid.moisture[y][x] * 50;
                ctx.fillStyle = `rgb(10, ${val}, ${20 + moistBlue})`;
                ctx.fillRect(x * cw, y * ch, cw, ch);
            }
        }
    }
}

// --- NVQLink ---
async function updateNVQLink() {
    try {
        const res = await fetch(`${API_BASE}/nvqlink/status`);
        const data = await res.json();

        const statusEl = document.getElementById('nvqStatus');
        if (statusEl) {
            statusEl.innerText = data.status;
            if (data.status.includes("ONLINE")) statusEl.style.color = '#10b981';
        }

        if (document.getElementById('nvqBandwidth')) document.getElementById('nvqBandwidth').innerText = data.bandwidth;
        if (document.getElementById('nvqLatency')) document.getElementById('nvqLatency').innerText = data.latency_ms + " ms";
        if (document.getElementById('nvqQuantum')) document.getElementById('nvqQuantum').innerText = data.quantum_state;

    } catch (e) { console.error(e); }
}

// --- Init ---
function initDashboardChart() {
    const ctx = document.getElementById('mainChart');
    if (!ctx || chartInstance || typeof Chart === 'undefined') return;

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [{
                label: 'Avg Fire Risk',
                data: [0.2, 0.3, 0.45, 0.6, 0.8, 0.9],
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointBackgroundColor: '#ef4444'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
            }
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    setView('dashboard');
    trainModel();

    // Poller
    setInterval(() => {
        const nvq = document.getElementById('nvqlink');
        if (nvq && nvq.classList.contains('active')) updateNVQLink();
    }, 1000);
});
