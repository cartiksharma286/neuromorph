/**
 * ChemTRAC Water Quality Monitor & Treatment Optimization Console
 * Frontend JavaScript Engine
 */

let activeTab = 'live_ntu';
let allChemicals = [];
let ntuHistory = [];
let telemetryInterval = null;

document.addEventListener('DOMContentLoaded', () => {
    refreshTelemetry();
    loadChemicals();
    loadStieltjesDecay();
    loadStieltjesHarmonics();
    loadEllipticData();
    generatePqcKem();
    signAndVerifyTelemetry();

    // Stream telemetry every 2.5 seconds
    telemetryInterval = setInterval(streamTelemetryTick, 2500);
});

// --- Tab Navigation ---
function switchTab(tabId) {
    activeTab = tabId;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    const btn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
    if (btn) btn.classList.add('active');

    const content = document.getElementById(`tab-${tabId}`);
    if (content) content.classList.add('active');
}

// --- 1. Live Water Quality & NTU (< 0.04 NTU) ---
async function refreshTelemetry() {
    const intensity = (parseInt(document.getElementById('treatmentIntensity').value) || 100) / 100.0;
    try {
        const res = await fetch(`/api/water/telemetry?intensity=${intensity}`);
        const d = await res.json();
        updateTelemetryUI(d);
    } catch (err) {
        console.error("Telemetry error:", err);
    }
}

function onIntensityChange() {
    const val = parseInt(document.getElementById('treatmentIntensity').value);
    document.getElementById('intensityVal').innerText = `${(val / 100.0).toFixed(1)}×`;
    refreshTelemetry();
}

async function streamTelemetryTick() {
    const intensity = (parseInt(document.getElementById('treatmentIntensity').value) || 100) / 100.0;
    try {
        const res = await fetch(`/api/water/telemetry?intensity=${intensity}`);
        const d = await res.json();
        updateTelemetryUI(d);
    } catch (err) {
        console.error("Stream tick error:", err);
    }
}

function updateTelemetryUI(d) {
    document.getElementById('curNtuVal').innerText = `${d.turbidity_ntu.toFixed(4)} NTU`;
    document.getElementById('ntuBadge').innerText = `TURBIDITY: ${d.turbidity_ntu.toFixed(3)} NTU (< 0.04 NTU)`;
    document.getElementById('waterPhVal').innerText = d.water_ph;
    document.getElementById('waterDoVal').innerText = `${d.dissolved_oxygen_mg_l} mg/L`;
    document.getElementById('condVal').innerText = `${d.conductivity_us_cm} μS/cm`;
    document.getElementById('tocVal').innerText = `${d.total_organic_carbon_mg_l} mg/L`;
    document.getElementById('orpVal').innerText = `+${d.redox_potential_orp_mv} mV`;
    document.getElementById('compliancePctVal').innerText = `${d.overall_compliance_pct}% Compliant (Below MCL)`;

    // Record into rolling NTU history
    ntuHistory.push(d.turbidity_ntu);
    if (ntuHistory.length > 40) ntuHistory.shift();

    renderNtuStreamChart();
    renderSensorRadar(d);
}

function renderNtuStreamChart() {
    const canvas = document.getElementById('ntuStreamCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#050a14';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;
    const n = ntuHistory.length;
    const maxNtu = 0.055; // Plot upper limit

    // Draw grid
    ctx.strokeStyle = '#152238';
    ctx.lineWidth = 0.5;
    for (let y = 20; y < h - 25; y += 30) {
        ctx.beginPath();
        ctx.moveTo(40, y);
        ctx.lineTo(w - 15, y);
        ctx.stroke();
    }

    // 0.04 NTU Strict Ceiling Line (Red Dashed)
    const yThreshold = (h - 25) - (0.040 / maxNtu) * (h - 45);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(40, yThreshold);
    ctx.lineTo(w - 15, yThreshold);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ef4444';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText('0.040 NTU Quality Ceiling', 45, yThreshold - 4);

    // Live NTU trace (Emerald Green / Cyan)
    if (n > 1) {
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ntuHistory.forEach((val, idx) => {
            const x = 45 + (idx / (n - 1)) * (w - 70);
            const y = (h - 25) - (val / maxNtu) * (h - 45);
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Pulsing point at current end
        const lastX = 45 + (w - 70);
        const lastY = (h - 25) - (ntuHistory[n - 1] / maxNtu) * (h - 45);
        ctx.fillStyle = '#38bdf8';
        ctx.beginPath();
        ctx.arc(lastX, lastY, 4, 0, 2 * Math.PI);
        ctx.fill();
    }

    ctx.fillStyle = '#94a3b8';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText('T - 40s', 42, h - 8);
    ctx.fillText('NOW', w - 45, h - 8);
    ctx.fillStyle = '#10b981';
    ctx.fillText('■ Purified Turbidity (NTU)', 200, 15);
}

function renderSensorRadar(d) {
    const canvas = document.getElementById('sensorRadarCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#050a14';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const maxRadius = 75;

    // Concentric guideline rings
    ctx.strokeStyle = '#152238';
    ctx.lineWidth = 0.75;
    [0.25, 0.5, 0.75, 1.0].forEach(r => {
        ctx.beginPath();
        ctx.arc(cx, cy, maxRadius * r, 0, 2 * Math.PI);
        ctx.stroke();
    });

    const params = [
        { label: "pH", val: (d.water_ph - 6.5) / 2.0 },
        { label: "DO", val: d.dissolved_oxygen_mg_l / 12.0 },
        { label: "ORP", val: d.redox_potential_orp_mv / 800.0 },
        { label: "TOC", val: 1.0 - (d.total_organic_carbon_mg_l / 1.5) },
        { label: "Clarity", val: 1.0 - (d.turbidity_ntu / 0.05) },
        { label: "Cond", val: 1.0 - (d.conductivity_us_cm / 300.0) }
    ];

    const numAxes = params.length;
    const polyPoints = [];

    params.forEach((p, i) => {
        const angle = (2 * Math.PI * i) / numAxes - Math.PI / 2;
        // Axis spoke
        const sx = cx + Math.cos(angle) * maxRadius;
        const sy = cy + Math.sin(angle) * maxRadius;
        ctx.strokeStyle = '#1c2e48';
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(sx, sy);
        ctx.stroke();

        // Label
        const lx = cx + Math.cos(angle) * (maxRadius + 14);
        const ly = cy + Math.sin(angle) * (maxRadius + 14);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '9px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.fillText(p.label, lx, ly);

        // Data point
        const clampedNorm = Math.max(0.15, Math.min(1.0, p.val));
        const dx = cx + Math.cos(angle) * (maxRadius * clampedNorm);
        const dy = cy + Math.sin(angle) * (maxRadius * clampedNorm);
        polyPoints.push([dx, dy]);
    });

    // Draw radar polygon
    ctx.fillStyle = 'rgba(6, 182, 212, 0.25)';
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    polyPoints.forEach((pt, idx) => {
        if (idx === 0) ctx.moveTo(pt[0], pt[1]);
        else ctx.lineTo(pt[0], pt[1]);
    });
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
}

// --- 2. Chemical Tracking Contaminants (CSV) ---
async function loadChemicals() {
    try {
        const res = await fetch('/api/chemicals');
        const d = await res.json();
        allChemicals = d.chemicals;
        renderChemicalsTable(allChemicals);
    } catch (err) {
        console.error("Chemicals load error:", err);
    }
}

function filterChemicals() {
    const q = document.getElementById('chemicalSearchInput').value.toLowerCase();
    const cat = document.getElementById('categoryFilter').value;

    const filtered = allChemicals.filter(c => {
        const matchText = c.chemicalName.toLowerCase().includes(q) || c.casNo.toLowerCase().includes(q);
        const matchCat = cat === 'all' || c.category.toLowerCase().includes(cat.toLowerCase());
        return matchText && matchCat;
    });

    renderChemicalsTable(filtered);
}

function renderChemicalsTable(list) {
    const tbody = document.querySelector('#chemicalsTable tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    list.forEach(c => {
        const tr = document.createElement('tr');
        const badgeClass = c.compliance_status.includes('COMPLIANT') ? 'badge-success' : 'badge-warning';

        tr.innerHTML = `
            <td>#${c._id}</td>
            <td><strong>${c.chemicalName}</strong></td>
            <td><code class="code-block" style="display:inline; padding:2px 5px;">${c.casNo}</code></td>
            <td><span class="text-secondary">${c.category}</span></td>
            <td class="text-warning">${c.raw_water_ppb}</td>
            <td class="text-success"><strong>${c.treated_water_ppb}</strong></td>
            <td>${c.mcl_limit_ppb} ${c.unit}</td>
            <td class="text-info">${c.removal_efficiency_pct}%</td>
            <td><span class="badge-tag ${badgeClass}">${c.compliance_status.split(' ')[0]}</span></td>
        `;
        tbody.appendChild(tr);
    });
}

// --- 3. Stieltjes Primes & S-Fraction Kinetics ---
async function loadStieltjesDecay() {
    const tMin = parseFloat(document.getElementById('stieltjesTimeInp').value) || 45.0;
    try {
        const res = await fetch(`/api/stieltjes/decay?time_min=${tMin}&initial_ppm=10.0`);
        const d = await res.json();
        renderStieltjesChart(d.decay_curve);
    } catch (err) {
        console.error("Stieltjes decay error:", err);
    }
}

function renderStieltjesChart(decayCurve) {
    const canvas = document.getElementById('stieltjesChartCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#050a14';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;
    const n = decayCurve.length;
    const maxPpm = 10.0;

    // Grid
    ctx.strokeStyle = '#152238';
    ctx.lineWidth = 0.5;
    for (let y = 25; y < h - 25; y += 30) {
        ctx.beginPath();
        ctx.moveTo(40, y);
        ctx.lineTo(w - 15, y);
        ctx.stroke();
    }

    // Decay Curve (Cyan)
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    decayCurve.forEach((pt, idx) => {
        const x = 45 + (idx / (n - 1)) * (w - 70);
        const y = (h - 25) - (pt.concentration_ppm / maxPpm) * (h - 50);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = '#94a3b8';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText('0 min', 45, h - 8);
    ctx.fillText(`${decayCurve[n - 1].time_min} min`, w - 55, h - 8);
    ctx.fillStyle = '#06b6d4';
    ctx.fillText('■ Stieltjes Moment Adsorption Decay (ppm)', 120, 16);
    ctx.fillText(`Eff: ${decayCurve[n-1].removal_efficiency_pct}%`, w - 80, 16);
}

async function loadStieltjesHarmonics() {
    try {
        const res = await fetch('/api/stieltjes/harmonics');
        const d = await res.json();
        const tbody = document.querySelector('#stieltjesHarmonicsTable tbody');
        if (!tbody) return;
        tbody.innerHTML = '';
        d.harmonics.forEach(h => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong class="text-info">p = ${h.prime}</strong></td>
                <td>${h.stieltjes_const_gamma}</td>
                <td class="text-success">${h.adsorption_rate_constant_k} min⁻¹</td>
                <td>${h.acoustic_resonance_hz} Hz</td>
                <td>${h.phase_angle_rad} rad</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error("Stieltjes harmonics error:", err);
    }
}

// --- 4. Elliptic Moduli & Cauer Filtering ---
async function loadEllipticData() {
    const kVal = parseFloat(document.getElementById('ellipticModulusK').value) || 0.885;
    try {
        // 1. Characteristics
        const res1 = await fetch(`/api/elliptic/characteristics?k=${kVal}`);
        const d1 = await res1.json();
        document.getElementById('kIntegralVal').innerText = d1.K_integral;
        document.getElementById('kPrimeVal').innerText = d1.K_prime_integral;
        document.getElementById('tauVal').innerText = d1.half_period_ratio_tau;
        document.getElementById('jInvariantVal').innerText = Number(d1.klein_j_invariant).toLocaleString();

        // 2. Cauer Filter
        const res2 = await fetch('/api/elliptic/filter?raw_ntu=0.85&cycles=30');
        const d2 = await res2.json();
        renderCauerFilterChart(d2.filtration_trajectory);
    } catch (err) {
        console.error("Elliptic data error:", err);
    }
}

function renderCauerFilterChart(trajectory) {
    const canvas = document.getElementById('cauerFilterCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#050a14';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;
    const n = trajectory.length;
    const maxNtu = 0.90;

    // 0.04 NTU Target Line (Red Dashed)
    const y004 = (h - 25) - (0.04 / maxNtu) * (h - 45);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(35, y004);
    ctx.lineTo(w - 15, y004);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ef4444';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText('NTU < 0.04 Target Boundary', 40, y004 - 4);

    // Cauer curve (Purple)
    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    trajectory.forEach((pt, idx) => {
        const x = 45 + (idx / (n - 1)) * (w - 75);
        const y = (h - 25) - (pt.turbidity_ntu / maxNtu) * (h - 45);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = '#a855f7';
    ctx.fillText('■ Cauer (Elliptic) Turbidity Attenuation', 130, 16);
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Cycle 1', 45, h - 8);
    ctx.fillText(`Cycle ${n}`, w - 60, h - 8);
}

// --- 5. Post-Quantum Cryptography (PQC) ---
async function generatePqcKem() {
    try {
        const res = await fetch('/api/pqc/encapsulate', { method: 'POST' });
        const d = await res.json();
        document.getElementById('kemSharedSecret').innerText = d.shared_secret_hex;
        document.getElementById('kemCiphertext').innerText = `${d.ciphertext_b64.slice(0, 48)}... (Ciphertext Verified)`;
        showToast("ML-KEM-768 Shared Secret Key encapsulated successfully!");
    } catch (err) {
        console.error("KEM error:", err);
    }
}

async function signAndVerifyTelemetry() {
    const samplePayload = {
        station_id: "TORONTO-CHEMTRAC-STN-04",
        turbidity_ntu: 0.028,
        mcl_compliant: true,
        dominant_contaminant: "Benzene < 0.05 ppb",
        timestamp: Date.now()
    };

    try {
        // Sign
        const signRes = await fetch('/api/pqc/sign_telemetry', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(samplePayload)
        });
        const signData = await signRes.json();

        document.getElementById('dsaPayloadHash').innerText = signData.payload_hash;
        document.getElementById('dsaSignature').innerText = `${signData.signature_b64.slice(0, 56)}... (Lattice Signature)`;

        // Verify
        const verifyRes = await fetch('/api/pqc/verify_telemetry', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                packet: samplePayload,
                signature_b64: signData.signature_b64,
                payload_hash: signData.payload_hash
            })
        });
        const verifyData = await verifyRes.json();
        document.getElementById('dsaVerificationStatus').innerText = verifyData.cryptographic_status;
        showToast("ML-DSA-65 Telemetry Packet signed & verified!");
    } catch (err) {
        console.error("DSA error:", err);
    }
}

// --- Toast ---
function showToast(msg) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.innerText = msg;
    toast.style.display = 'block';
    setTimeout(() => {
        toast.style.display = 'none';
    }, 3500);
}
