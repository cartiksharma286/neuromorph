
const API_URL = "http://localhost:8000";

// DOM Elements
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const statusBadge = document.getElementById('system-status');
const valCost = document.getElementById('val-cost');
const valFidelity = document.getElementById('val-fidelity');
const valPerm = document.getElementById('val-perm');
const permCircle = document.getElementById('perm-progress');
const riskValue = document.getElementById('val-risk');
const logList = document.getElementById('log-list');

// Charts
const costCanvas = document.getElementById('costChart');
const costCtx = costCanvas.getContext('2d');

const fieldCanvas = document.getElementById('fieldCanvas');
const fieldCtx = fieldCanvas.getContext('2d');

const brainCanvas = document.getElementById('brainCanvas');
const brainCtx = brainCanvas.getContext('2d');

// State
let pollingInterval = null;
let costs = [];
let rotation = 0;
let geodesicPath = [];
let currentField = null;

// --- API Functions ---

async function generateField() {
    try {
        log("Dreaming acoustic pattern (GenAI + CUDA)...");
        const res = await fetch(`${API_URL}/generate_field_pattern`, { method: 'POST' });
        const data = await res.json();

        currentField = data.effective_field; // 2D array
        drawHeatmap(currentField);

        const topo = data.topology;
        const tag = document.getElementById('topology-tag');
        tag.textContent = `Topology: ${topo.classification} (Charge: ${topo.charge})`;
        if (topo.is_topologically_protected) {
            tag.style.color = "#76b900";
        } else {
            tag.style.color = "#c5c6c7";
        }

        log(`Field Generated. Jones Poly: ${topo.jones_polynomial}`);

    } catch (e) {
        log("GenAI Error: " + e);
    }
}

async function startTreatment() {
    try {
        await fetch(`${API_URL}/start_treatment`, { method: 'POST' });
        log("Treatment Initialized. Quantum Engine Starting...");
        statusBadge.textContent = "SYSTEM ACTIVE";
        statusBadge.style.color = "#ff4d4d"; // Red alert style for active radiation
        statusBadge.style.borderColor = "#ff4d4d";
        btnStart.disabled = true;
        btnStop.disabled = false;

        // Start Polling
        pollingInterval = setInterval(pollStatus, 1000);

        // Request Geodesic
        fetchGeodesic();

    } catch (e) {
        log("Error starting treatment: " + e);
    }
}

async function stopTreatment() {
    try {
        await fetch(`${API_URL}/stop_treatment`, { method: 'POST' });
        log("Treatment Stopped.");
        statusBadge.textContent = "SYSTEM IDLE";
        statusBadge.style.color = "#76b900";
        statusBadge.style.borderColor = "#76b900";
        btnStart.disabled = false;
        btnStop.disabled = true;

        clearInterval(pollingInterval);
    } catch (e) {
        log("Error stopping: " + e);
    }
}

async function pollStatus() {
    try {
        const res = await fetch(`${API_URL}/get_status`);
        const data = await res.json();

        if (data.status === "active") {
            updateUI(data);
        }
    } catch (e) {
        console.error(e);
    }
}

async function fetchGeodesic() {
    // Mock locations for demo
    const u1 = 0.5, v1 = 0.5;
    const u2 = 2.0, v2 = 2.0;

    try {
        const res = await fetch(`${API_URL}/get_geodesic_path?start_u=${u1}&start_v=${v1}&end_u=${u2}&end_v=${v2}`);
        const data = await res.json();
        geodesicPath = data.path; // [[u,v], ...]
        log("Geodesic Path Optimized. Measure Weight: " + data.measure_weight);
        document.getElementById('geo-status').textContent = "Path Locked (Measure: " + data.measure_weight + ")";
    } catch (e) {
        console.error(e);
    }
}

// --- UI Updates ---

function updateUI(data) {
    // Stats
    const cost = data.optimization_cost.toFixed(4);
    valCost.textContent = cost;
    valFidelity.textContent = (1.0 - data.optimization_cost).toFixed(4); // Approx

    // Charts
    costs.push(data.optimization_cost);
    if (costs.length > 20) costs.shift();
    drawCostChart();

    // Risk
    riskValue.textContent = data.risk_var_95.toFixed(2) + " (Risk)";

    // Permeability
    const permPct = Math.round(data.permeability * 100);
    valPerm.textContent = permPct + "%";
    permCircle.style.background = `conic-gradient(#76b900 ${permPct * 3.6}deg, rgba(255,255,255,0.1) 0deg)`;

    // Params
    const p = data.current_optimal_params;
    updateBar('bar-freq', 'txt-freq', p.frequency, 2.0, "MHz");
    updateBar('bar-int', 'txt-int', p.intensity, 10.0, "W/cm²");
    updateBar('bar-dur', 'txt-dur', p.duration, 200.0, "ms");
}

function updateBar(barId, txtId, val, max, unit) {
    const pct = Math.min(100, (val / max) * 100);
    document.getElementById(barId).style.width = pct + "%";
    document.getElementById(txtId).textContent = val.toFixed(2) + " " + unit;
}

function log(msg) {
    const li = document.createElement('li');
    li.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logList.prepend(li); // Newest on top
}

// --- Canvas Drawing ---

function drawCostChart() {
    const w = costCanvas.width;
    const h = costCanvas.height;
    costCtx.clearRect(0, 0, w, h);

    costCtx.strokeStyle = "#45a29e";
    costCtx.lineWidth = 2;
    costCtx.beginPath();

    const step = w / 20;

    costs.forEach((val, i) => {
        const x = i * step;
        const y = h - (val * h); // utilizing 0-1 scale approx
        if (i === 0) costCtx.moveTo(x, y);
        else costCtx.lineTo(x, y);
    });

    costCtx.stroke();
}

function drawBrain() {
    const w = brainCanvas.width;
    const h = brainCanvas.height;
    brainCtx.clearRect(0, 0, w, h);

    // Simple wireframe sphere application
    const cx = w / 2;
    const cy = h / 2;
    const r = 100;

    rotation += 0.01;

    brainCtx.save();
    brainCtx.translate(cx, cy);

    // Draw "Meridians"
    brainCtx.strokeStyle = "rgba(118, 185, 0, 0.3)";
    brainCtx.lineWidth = 1;

    for (let i = 0; i < 8; i++) {
        brainCtx.beginPath();
        brainCtx.ellipse(0, 0, r, r * Math.sin(rotation + i), 0, 0, Math.PI * 2);
        brainCtx.stroke();
    }

    // Draw Geodesic Path if exists
    if (geodesicPath.length > 0) {
        brainCtx.strokeStyle = "#ff4d4d";
        brainCtx.lineWidth = 3;
        brainCtx.beginPath();

        geodesicPath.forEach((pt, i) => {
            // Map u,v to simple sphere coords for viz
            const u = pt[0], v = pt[1];
            // Simple projection mapping
            const x = r * Math.sin(u) * Math.cos(v + rotation);
            const y = r * Math.sin(u) * Math.sin(v + rotation);
            // Ignore z for 2d projection simple
            if (i === 0) brainCtx.moveTo(x, y);
            else brainCtx.lineTo(x, y);
        });

        brainCtx.stroke();
    }

    brainCtx.restore();

    requestAnimationFrame(drawBrain);
}

// Listeners
btnStart.addEventListener('click', startTreatment);
btnStop.addEventListener('click', stopTreatment);
document.getElementById('btn-dream').addEventListener('click', generateField);

function drawHeatmap(field2D) {
    const w = fieldCanvas.width;
    const h = fieldCanvas.height;
    const imgData = fieldCtx.createImageData(w, h);

    // Simple scaling assuming 128x128 field roughly mapped to canvas
    const rows = field2D.length;
    const cols = field2D[0].length;

    for (let i = 0; i < w * h * 4; i += 4) {
        // Map pixel index to field index
        const px_idx = i / 4;
        const x = px_idx % w;
        const y = Math.floor(px_idx / w);

        const map_x = Math.floor((x / w) * cols);
        const map_y = Math.floor((y / h) * rows);

        const val = field2D[map_y][map_x];

        // Color map: low=blue, high=red
        // Normalize val ~ -0.5 to 1.5 roughly
        const intensity = Math.max(0, Math.min(1, (val + 0.5) / 2.0));

        imgData.data[i] = intensity * 255;     // R
        imgData.data[i + 1] = 0;                 // G
        imgData.data[i + 2] = (1 - intensity) * 255; // B
        imgData.data[i + 3] = 255;               // Alpha
    }

    fieldCtx.putImageData(imgData, 0, 0);
}

// Init
drawBrain();
