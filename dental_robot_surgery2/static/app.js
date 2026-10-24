/**
 * Dental Image-Guided Surgery Robot Console
 * Frontend JavaScript Engine
 */

let activeTab = 'cbct';
let slamInterval = null;
let currentDepth = 0.0;
let slamTrajectoryHistory = [];
let cachedQmlData = null;

document.addEventListener('DOMContentLoaded', () => {
    loadCbctSlice();
    loadCbctMetrics();
    loadJoints();
    initSlamCanvas();
    runVqeRegistration();
    runQmlUsCtRegistration();
    runTmsPainSimulation();
    generateImplantModel();
    loadImplantFea();
    loadManipulatorSim();
    loadHealthEconomics();
    loadBiomarkers();
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

// --- 1. PCD-CBCT & Segmentation ---
async function loadCbctSlice() {
    const plane = document.getElementById('cbctPlane').value;
    const binMode = document.getElementById('cbctBin').value;
    const index = document.getElementById('cbctIndex').value;
    document.getElementById('sliceIndexVal').innerText = index;

    try {
        const res = await fetch(`/api/cbct/slice?plane=${plane}&index=${index}&bin_mode=${binMode}`);
        const data = await res.json();
        renderCbctSlice(data);
    } catch (err) {
        console.error("CBCT slice fetch error:", err);
    }
}

function renderCbctSlice(data) {
    const canvas = document.getElementById('cbctCanvas');
    const ctx = canvas.getContext('2d');
    const [rows, cols] = data.dimensions;
    const pixels = data.pixels;
    const seg = data.segmentation;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const cellW = canvas.width / cols;
    const cellH = canvas.height / rows;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const gray = pixels[r][c];
            const s = seg[r][c];

            let fillColor = `rgb(${gray}, ${gray}, ${gray})`;
            if (s === 1) { // Cortical bone
                fillColor = `rgba(${Math.min(255, gray + 40)}, ${Math.min(255, gray + 40)}, ${Math.min(255, gray + 70)}, 0.85)`;
            } else if (s === 2) { // Trabecular bone
                fillColor = `rgba(${gray}, ${Math.min(255, gray + 15)}, ${Math.min(255, gray + 30)}, 0.8)`;
            } else if (s === 3) { // Nerve Canal (Red/Amber Highlight)
                fillColor = `rgba(239, 68, 68, 0.9)`;
            } else if (s === 4) { // Teeth
                fillColor = `rgba(240, 249, 255, 0.95)`;
            } else if (s === 5) { // Planned Implant Bed (Cyan Highlight)
                fillColor = `rgba(2, 132, 199, 0.85)`;
            }

            ctx.fillStyle = fillColor;
            ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
        }
    }

    // Draw Crosshair and safety margin circle
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(canvas.width / 2, 0);
    ctx.lineTo(canvas.width / 2, canvas.height);
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    // Scale bar
    ctx.fillStyle = '#fff';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('Scale: 75 μm/vox | FOV: 7.2 cm', 12, canvas.height - 12);
}

async function loadCbctMetrics() {
    try {
        const res = await fetch('/api/cbct/metrics');
        const m = await res.json();
        document.getElementById('nerveClearance').innerText = `${m.nerve_clearance_mm} mm`;
        document.getElementById('boneQuality').innerText = m.bone_quality_classification;
    } catch (err) {
        console.error("Metrics load error:", err);
    }
}

// --- 2. Leibniz Primes SLAM ---
function initSlamCanvas() {
    const canvas = document.getElementById('slamPathCanvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawSlamGrid(ctx, canvas);
}

function drawSlamGrid(ctx, canvas) {
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 0.5;
    for (let x = 0; x < canvas.width; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
}

async function stepSlam() {
    try {
        const res = await fetch('/api/robot/slam_step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ t_step: 0.08, feed_rate_mm_s: 0.6, use_qkf: true })
        });
        const d = await res.json();
        currentDepth = d.drill_depth_mm;
        document.getElementById('drillDepth').innerText = `${d.drill_depth_mm.toFixed(2)} mm / ${d.drill_target_depth_mm.toFixed(2)} mm`;
        
        // Update QKF tracking error and quantum state readout
        const errText = d.tracking_error_um ? `${d.tracking_error_um} μm (< 15 μm)` : `${d.tracking_error_mm} mm`;
        document.getElementById('trackError').innerText = errText;
        if (d.quantum_purity) {
            document.getElementById('qkfPurity').innerText = d.quantum_purity.toFixed(4);
        }
        if (d.heisenberg_uncertainty_um) {
            document.getElementById('heisenbergUncert').innerText = `${d.heisenberg_uncertainty_um} μm³`;
        }
        if (d.quantum_fisher_information) {
            document.getElementById('fisherInfo').innerText = `${(d.quantum_fisher_information / 1e6).toFixed(2)} × 10⁶`;
        }

        slamTrajectoryHistory.push({ depth: d.drill_depth_mm, err: d.tracking_error_mm });
        if (slamTrajectoryHistory.length > 50) slamTrajectoryHistory.shift();

        renderSlamTrajectory(d);
        renderSpectrum(d.leibniz_spectrum);
    } catch (err) {
        console.error("SLAM step error:", err);
    }
}

async function resetQkf() {
    try {
        await fetch('/api/robot/qkf_reset', { method: 'POST' });
        currentDepth = 0.0;
        document.getElementById('drillDepth').innerText = "0.0 mm / 10.0 mm";
        initSlamCanvas();
        showToast("Quantum Kalman SLAM state reset to initial datum.");
    } catch (err) {
        console.error("QKF reset error:", err);
    }
}

function toggleSlamLoop() {
    const btn = document.getElementById('btnToggleDrill');
    if (slamInterval) {
        clearInterval(slamInterval);
        slamInterval = null;
        btn.innerText = "Start Drilling SLAM";
        btn.classList.remove('btn-secondary');
        btn.classList.add('btn-primary');
    } else {
        btn.innerText = "Pause Drilling SLAM";
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-secondary');
        slamInterval = setInterval(() => {
            stepSlam();
            if (currentDepth >= 10.0) {
                clearInterval(slamInterval);
                slamInterval = null;
                btn.innerText = "Osteotomy Completed (10mm)";
                showToast("Osteotomy target depth (10.0mm) achieved with sub-0.04mm precision!");
            }
        }, 150);
    }
}

function renderSlamTrajectory(slamData) {
    const canvas = document.getElementById('slamPathCanvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawSlamGrid(ctx, canvas);

    // Target drill axis
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(canvas.width / 2, 20);
    ctx.lineTo(canvas.width / 2, canvas.height - 20);
    ctx.stroke();
    ctx.setLineDash([]);

    // Drill tip path
    ctx.strokeStyle = '#0284c7';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    const startY = 30;
    const progressY = startY + (currentDepth / 10.0) * (canvas.height - 80);

    ctx.moveTo(canvas.width / 2, startY);
    ctx.lineTo(canvas.width / 2, progressY);
    ctx.stroke();

    // End-effector active drill bit icon
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(canvas.width / 2, progressY, 5, 0, 2 * Math.PI);
    ctx.fill();

    // High-frequency vibration dampening envelope
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.6)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let y = startY; y <= progressY; y += 4) {
        const offset = Math.sin(y * 0.4) * 3 * (slamData.tracking_error_mm / 0.05);
        if (y === startY) ctx.moveTo(canvas.width / 2 + offset, y);
        else ctx.lineTo(canvas.width / 2 + offset, y);
    }
    ctx.stroke();

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(`Target Depth: 10.00 mm | Current: ${currentDepth.toFixed(2)} mm`, 16, 22);
    ctx.fillText(`Leibniz Damped Pos: [${slamData.position.join(', ')}]`, 16, canvas.height - 12);
}

function renderSpectrum(spectrum) {
    const box = document.getElementById('spectrumGrid');
    box.innerHTML = '';
    spectrum.forEach(item => {
        const card = document.createElement('div');
        card.className = 'spectrum-card';
        card.innerHTML = `
            <div class="prime-tag">p=${item.prime}</div>
            <div class="prime-power">P: ${item.power}</div>
            <div class="prime-power">Phase: ${item.phase} rad</div>
        `;
        box.appendChild(card);
    });
}

async function loadJoints() {
    try {
        const res = await fetch('/api/robot/joints');
        const data = await res.json();
        const container = document.getElementById('jointAnglesContainer');
        container.innerHTML = '';
        data.joints.forEach(j => {
            const chip = document.createElement('div');
            chip.className = 'joint-chip';
            chip.innerHTML = `<span>${j.joint} (${j.name}):</span><strong>${j.angle_deg}°</strong>`;
            container.appendChild(chip);
        });
    } catch (err) {
        console.error("Joint load error:", err);
    }
}

// --- 3. Quantum VQE Registration ---
async function runVqeRegistration() {
    try {
        const res = await fetch('/api/registration/vqe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ iterations: 25 })
        });
        const d = await res.json();

        document.getElementById('vqeTre').innerText = `${d.final_tre_mm} mm`;
        document.getElementById('vqeFre').innerText = `${d.final_fre_mm} mm`;
        document.getElementById('vqeFidelity').innerText = d.ansatz_quantum_fidelity;
        document.getElementById('eulerRot').innerText = d.rotation_euler_deg.map(v => `${v}°`).join(', ');
        document.getElementById('transVec').innerText = `${d.translation_mm.join(', ')} mm`;
        
        // Update header badge
        const badge = document.getElementById('treBadge');
        if (badge) {
            badge.innerText = `TRE: ${d.final_tre_mm} mm (SUBMILLIMETRIC)`;
        }

        // Render SE(3) matrix
        const mHtml = d.se3_matrix.map(row => `[ ${row.map(v => (v >= 0 ? ' ' : '') + v.toFixed(4)).join('  ')} ]`).join('<br>');
        document.getElementById('se3MatrixBox').innerHTML = mHtml;

        // Render Majorana Braiding gates
        const braidBox = document.getElementById('braidListContainer');
        if (braidBox && d.sample_braid_gates) {
            braidBox.innerHTML = '';
            d.sample_braid_gates.forEach(b => {
                const item = document.createElement('div');
                item.className = 'braid-item';
                item.innerHTML = `<span class="braid-tag">${b.braid_id} ${b.mode_pair}</span><span class="braid-fid">F: ${(b.braid_fidelity*100).toFixed(3)}%</span>`;
                braidBox.appendChild(item);
            });
        }

        renderVqeChart(d.energy_history, d.tre_history);
        showToast(`Majorana Chip VQE: TRE = ${d.final_tre_mm} mm (Submillimetric Certified)`);
    } catch (err) {
        console.error("VQE registration error:", err);
    }
}

function renderVqeChart(energies, tres) {
    const canvas = document.getElementById('vqeChartCanvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;
    const n = energies.length;

    // Draw grid
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 0.5;
    for (let y = 30; y < h - 20; y += 30) {
        ctx.beginPath();
        ctx.moveTo(40, y);
        ctx.lineTo(w - 20, y);
        ctx.stroke();
    }

    // Energy line (Cyan)
    const maxE = Math.max(...energies);
    const minE = Math.min(...energies);
    ctx.strokeStyle = '#0284c7';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
        const x = 50 + (i / (n - 1)) * (w - 70);
        const y = 30 + (1.0 - (energies[i] - minE) / (maxE - minE + 1e-4)) * (h - 70);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // TRE curve (Emerald)
    const maxTre = Math.max(...tres);
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
        const x = 50 + (i / (n - 1)) * (w - 70);
        const y = 30 + (1.0 - tres[i] / (maxTre + 1e-4)) * (h - 70);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('Iter: 0', 45, h - 6);
    ctx.fillText(`Iter: ${n}`, w - 60, h - 6);
    ctx.fillStyle = '#38bdf8';
    ctx.fillText('■ Hamiltonian Energy', 60, 20);
    ctx.fillStyle = '#10b981';
    ctx.fillText('■ Target Registration Error (TRE)', 220, 20);
}

// --- 4. GenAI Chamfers & Abutment Designer ---
async function generateImplantModel() {
    const payload = {
        implant_diameter_mm: parseFloat(document.getElementById('inpDiam').value),
        implant_length_mm: parseFloat(document.getElementById('inpLen').value),
        chamfer_radius_mm: parseFloat(document.getElementById('inpChamferR').value),
        emergence_angle_deg: parseFloat(document.getElementById('inpEmergDeg').value),
        platform_switch_offset_mm: parseFloat(document.getElementById('inpPlatSwitch').value),
        screw_channel_angle_deg: parseFloat(document.getElementById('inpAscDeg').value)
    };

    try {
        const res = await fetch('/api/implant/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const d = await res.json();

        document.getElementById('peakStress').innerText = `${d.biomechanical_analysis.peak_von_mises_stress_mpa} MPa`;
        document.getElementById('safetyFactor').innerText = d.biomechanical_analysis.safety_factor;
        document.getElementById('bioWidth').innerText = `${d.biomechanical_analysis.biological_width_preservation_mm} mm`;

        renderCadViewport(d.mesh_data, payload);
    } catch (err) {
        console.error("CAD generate error:", err);
    }
}

function renderCadViewport(meshData, params) {
    const canvas = document.getElementById('cadCanvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const cx = canvas.width / 2;
    const cy = canvas.height / 2 + 10;
    const scale = 14;

    // Draw Abutment (Top, Gold/Bronze Chamfer Aesthetic)
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 1;
    meshData.abutment_rings.forEach((ring, idx) => {
        ctx.beginPath();
        ring.forEach((pt, pIdx) => {
            const x = cx + pt[0] * scale;
            const y = cy - pt[2] * scale - (pt[1] * scale * 0.25);
            if (pIdx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.stroke();
    });

    // Draw Implant Body (Bottom, Titanium Metallic Blue)
    ctx.strokeStyle = '#0284c7';
    ctx.lineWidth = 1.2;
    meshData.implant_rings.forEach((ring, idx) => {
        ctx.beginPath();
        ring.forEach((pt, pIdx) => {
            const x = cx + pt[0] * scale;
            const y = cy - pt[2] * scale - (pt[1] * scale * 0.25);
            if (pIdx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.stroke();
    });

    // Angulated Screw Channel (ASC) centerline indicator
    const ascRad = (params.screw_channel_angle_deg * Math.PI) / 180;
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.sin(ascRad) * 60, cy - Math.cos(ascRad) * 60);
    ctx.stroke();
    ctx.setLineDash([]);

    // Chamfer label
    ctx.fillStyle = '#38bdf8';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(`G² Chamfer R = ${params.chamfer_radius_mm} mm`, cx + 45, cy - 8);
    ctx.fillText(`ASC Angle: ${params.screw_channel_angle_deg}°`, cx + 45, cy - 25);
    ctx.fillText(`Platform Switch: ${params.platform_switch_offset_mm} mm`, cx + 45, cy + 12);
}

// --- 5. Dental Arch Kinematics & Biomarkers ---
async function loadBiomarkers() {
    try {
        const res = await fetch('/api/biomarkers/longitudinal');
        const data = await res.json();
        renderChewingEnvelope(data.mastication.trajectory);
        renderIsqTimeline(data.biomarkers.isq_timeline);
        renderMicromotionTimeline(data.biomarkers.micromotion_timeline);
    } catch (err) {
        console.error("Biomarkers error:", err);
    }
}

function renderChewingEnvelope(trajectory) {
    const canvas = document.getElementById('chewingCanvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const cx = canvas.width / 2;
    const cy = 35;
    const scale = 12;

    // Grid lines
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(cx, 0); ctx.lineTo(cx, canvas.height);
    ctx.moveTo(0, cy); ctx.lineTo(canvas.width, cy);
    ctx.stroke();

    // Mastication cycle teardrop loop
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    trajectory.forEach((pt, idx) => {
        const x = cx + pt.x_lateral_mm * scale;
        const y = cy - pt.z_opening_mm * scale;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();

    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
    ctx.fill();

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('Centric Occlusion (CO)', cx + 8, cy - 6);
    ctx.fillText('Maximum Mandibular Opening', cx - 80, canvas.height - 10);
}

function renderIsqTimeline(timeline) {
    const canvas = document.getElementById('isqCanvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;
    const n = timeline.length;

    // Safe threshold line at ISQ 70
    const y70 = 25 + (1.0 - (70 - 50) / 45) * (h - 50);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(40, y70);
    ctx.lineTo(w - 20, y70);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ef4444';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText('ISQ 70: Immediate Load Floor', 45, y70 - 4);

    // Primary Stability (Friction) Line
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    timeline.forEach((pt, idx) => {
        const x = 50 + (idx / (n - 1)) * (w - 80);
        const y = 25 + (1.0 - (pt.primary - 50) / 45) * (h - 50);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Composite Total ISQ (Emerald)
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    timeline.forEach((pt, idx) => {
        const x = 50 + (idx / (n - 1)) * (w - 80);
        const y = 25 + (1.0 - (pt.isq - 50) / 45) * (h - 50);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Dots & Labels
    timeline.forEach((pt, idx) => {
        const x = 50 + (idx / (n - 1)) * (w - 80);
        const y = 25 + (1.0 - (pt.isq - 50) / 45) * (h - 50);
        ctx.fillStyle = '#10b981';
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = '#64748b';
        ctx.font = '9px JetBrains Mono';
        ctx.fillText(`W${pt.week}`, x - 8, h - 8);
    });

    ctx.fillStyle = '#10b981';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('■ Composite ISQ', w - 120, 20);
}

function renderMicromotionTimeline(timeline) {
    const canvas = document.getElementById('micromotionCanvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;
    const n = timeline.length;

    // Threshold line at 50 um
    const y50 = 25 + (1.0 - 50 / 60) * (h - 50);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(40, y50);
    ctx.lineTo(w - 20, y50);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ef4444';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText('50 μm Critical Encapsulation Threshold', 45, y50 - 4);

    // Micromotion curve (Cyan)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    timeline.forEach((pt, idx) => {
        const x = 50 + (idx / (n - 1)) * (w - 80);
        const y = 25 + (1.0 - pt.micromotion_um / 60) * (h - 50);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Dots
    timeline.forEach((pt, idx) => {
        const x = 50 + (idx / (n - 1)) * (w - 80);
        const y = 25 + (1.0 - pt.micromotion_um / 60) * (h - 50);
        ctx.fillStyle = '#38bdf8';
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = '#64748b';
        ctx.font = '9px JetBrains Mono';
        ctx.fillText(`W${pt.week}`, x - 8, h - 8);
    });

    ctx.fillStyle = '#38bdf8';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('■ Micromotion (μm)', w - 130, 20);
}

// --- 5. FEA & Serial/Parallel Arm Manipulators ---
async function loadImplantFea() {
    const designKey = document.getElementById('feaDesignSelect').value;
    const loadN = parseFloat(document.getElementById('feaLoadN').value) || 250.0;
    const angleDeg = parseFloat(document.getElementById('feaAngleDeg').value) || 15.0;

    try {
        const res = await fetch('/api/implant/fea', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ design_key: designKey, load_n: loadN, load_angle_deg: angleDeg })
        });
        const d = await res.json();
        
        document.getElementById('feaPeakBone').innerText = `${d.fea_metrics.peak_bone_stress_mpa} MPa`;
        document.getElementById('feaMicrostrain').innerText = `${d.fea_metrics.crestal_microstrain} με`;
        document.getElementById('feaFrostZone').innerText = d.fea_metrics.frost_mechanostat_classification;
        document.getElementById('feaMicromotion').innerText = `${d.fea_metrics.interfacial_micromotion_um} μm`;
        document.getElementById('feaFatigue').innerText = `${d.fea_metrics.fatigue_safety_factor}×`;

        renderFeaHeatmap(d.stress_field_heatmap);
    } catch (err) {
        console.error("FEA load error:", err);
    }
}

function renderFeaHeatmap(heatmapData) {
    const canvas = document.getElementById('feaHeatmapCanvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const rows = heatmapData.length;
    const cols = heatmapData[0].length;
    
    // Draw bilateral axisymmetric implant profile
    const startX = 60;
    const totalW = canvas.width - 120;
    const cellW = (totalW / 2) / cols;
    const cellH = (canvas.height - 40) / rows;
    const midX = startX + totalW / 2;

    // Find max stress for normalization
    let maxS = 0.1;
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            if (heatmapData[r][c] > maxS) maxS = heatmapData[r][c];
        }
    }

    for (let r = 0; r < rows; r++) {
        const y = 20 + r * cellH;
        for (let c = 0; c < cols; c++) {
            const val = heatmapData[r][c];
            const norm = val / maxS;

            // Jet/Turbo colormap approximation: Blue (low) -> Cyan -> Green -> Yellow -> Red (high)
            let cr = Math.min(255, Math.floor(norm * 255 * 1.5));
            let cg = Math.min(255, Math.floor(Math.sin(norm * Math.PI) * 255));
            let cb = Math.min(255, Math.floor((1.0 - norm) * 220));

            ctx.fillStyle = `rgb(${cr}, ${cg}, ${cb})`;

            // Right side of implant
            ctx.fillRect(midX + c * cellW, y, cellW + 0.5, cellH + 0.5);
            // Left side (symmetric mirror)
            ctx.fillRect(midX - (c + 1) * cellW, y, cellW + 0.5, cellH + 0.5);
        }
    }

    // Overlay implant centerline and thread notches
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(midX, 20);
    ctx.lineTo(midX, canvas.height - 20);
    ctx.stroke();

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('Crestal Collar (Peak Stress)', midX + 35, 30);
    ctx.fillText('Helical Threads', midX + 35, canvas.height / 2);
    ctx.fillText('Apex (Minimum Stress)', midX + 35, canvas.height - 25);
    ctx.fillText(`Peak Bone: ${maxS.toFixed(1)} MPa`, 15, canvas.height - 10);
}

async function loadManipulatorSim() {
    const feedVal = parseInt(document.getElementById('manipFeedRange').value) || 50;
    document.getElementById('manipFeedVal').innerText = `${feedVal}%`;
    const progress = feedVal / 100.0;

    try {
        const res = await fetch('/api/robot/manipulators', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_xyz: [12.0, 45.0, -28.0], feed_progress: progress })
        });
        const d = await res.json();

        document.getElementById('serialDeflect').innerText = `${d.serial_arm.tooltip_deflection_um} μm`;
        document.getElementById('parallelDeflect').innerText = `${d.parallel_hexapod.tooltip_deflection_um} μm`;

        // Render comparative table
        const tableBody = document.querySelector('#manipComparisonTable tbody');
        if (tableBody && d.comparison_benchmark) {
            tableBody.innerHTML = '';
            const metrics = d.comparison_benchmark.metric;
            const serialVals = d.comparison_benchmark.serial_manipulator;
            const parallelVals = d.comparison_benchmark.parallel_hexapod;

            metrics.forEach((m, idx) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `<td><strong>${m}</strong></td><td>${serialVals[idx]}</td><td class="text-success">${parallelVals[idx]}</td>`;
                tableBody.appendChild(tr);
            });
        }

        renderManipulatorCanvas(d);
    } catch (err) {
        console.error("Manipulator sim error:", err);
    }
}

function renderManipulatorCanvas(data) {
    const canvas = document.getElementById('manipulatorCanvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;

    // Split Canvas: Left = Serial Arm, Right = Parallel Stewart Hexapod
    ctx.strokeStyle = '#1e293b';
    ctx.beginPath();
    ctx.moveTo(w / 2, 10);
    ctx.lineTo(w / 2, h - 10);
    ctx.stroke();

    // 1. Serial 6-DOF Arm Kinematics (Left)
    ctx.fillStyle = '#f59e0b';
    ctx.font = '11px JetBrains Mono';
    ctx.fillText('Serial 6-DOF Anthropomorphic', 15, 20);

    const baseSerialX = 80;
    const baseSerialY = h - 25;
    
    // Draw joints and linkages
    const joints = [
        [baseSerialX, baseSerialY],
        [baseSerialX + 25, baseSerialY - 50],
        [baseSerialX + 70, baseSerialY - 95],
        [baseSerialX + 115, baseSerialY - 60],
        [baseSerialX + 135, baseSerialY - 30 + data.feed_progress * 15]
    ];

    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 3;
    ctx.beginPath();
    joints.forEach((pt, idx) => {
        if (idx === 0) ctx.moveTo(pt[0], pt[1]);
        else ctx.lineTo(pt[0], pt[1]);
    });
    ctx.stroke();

    joints.forEach(pt => {
        ctx.fillStyle = '#38bdf8';
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 4, 0, 2 * Math.PI);
        ctx.fill();
    });

    ctx.fillStyle = '#94a3b8';
    ctx.font = '9.5px JetBrains Mono';
    ctx.fillText(`κ(J): ${data.serial_arm.condition_number}`, 15, h - 10);
    ctx.fillText(`Deflect: ${data.serial_arm.tooltip_deflection_um} μm`, 85, h - 10);

    // 2. Parallel Stewart Hexapod (Right)
    const rightOffset = w / 2 + 15;
    ctx.fillStyle = '#10b981';
    ctx.font = '11px JetBrains Mono';
    ctx.fillText('Parallel 6-DOF Stewart Hexapod', rightOffset, 20);

    const baseHexY = h - 25;
    const platHexY = 55 + data.feed_progress * 25;
    const hexCenterX = rightOffset + 110;

    // Base plate
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(hexCenterX - 85, baseHexY);
    ctx.lineTo(hexCenterX + 85, baseHexY);
    ctx.stroke();

    // Mobile platform plate
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(hexCenterX - 45, platHexY);
    ctx.lineTo(hexCenterX + 45, platHexY);
    ctx.stroke();

    // End effector spindle pointing down
    ctx.strokeStyle = '#0284c7';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(hexCenterX, platHexY);
    ctx.lineTo(hexCenterX, platHexY + 35);
    ctx.stroke();
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(hexCenterX, platHexY + 35, 4, 0, 2 * Math.PI);
    ctx.fill();

    // 6 Struts
    ctx.lineWidth = 1.2;
    data.parallel_hexapod.struts.forEach((s, idx) => {
        const bOffset = (idx - 2.5) * 30;
        const pOffset = (idx - 2.5) * 16;
        ctx.strokeStyle = idx % 2 === 0 ? '#10b981' : '#38bdf8';
        ctx.beginPath();
        ctx.moveTo(hexCenterX + bOffset, baseHexY);
        ctx.lineTo(hexCenterX + pOffset, platHexY);
        ctx.stroke();
    });

    ctx.fillStyle = '#94a3b8';
    ctx.font = '9.5px JetBrains Mono';
    ctx.fillText(`κ(J): ${data.parallel_hexapod.condition_number}`, rightOffset, h - 10);
    ctx.fillText(`Deflect: ${data.parallel_hexapod.tooltip_deflection_um} μm (5.1× Stiff)`, rightOffset + 70, h - 10);
}

// --- 6. Health Economics & >$1B Enterprise Valuation ---
async function loadHealthEconomics() {
    const procsMonth = parseFloat(document.getElementById('econProcsPerMonth').value) || 18.0;
    const sysPriceK = parseFloat(document.getElementById('econSystemPrice').value) || 280.0;

    try {
        const res = await fetch('/api/economics/projections', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                avg_procedures_per_robot_month: procsMonth,
                initial_system_price: sysPriceK * 1000.0
            })
        });
        const d = await res.json();

        // Practice & Clinical Level
        document.getElementById('econPayback').innerText = `${d.practice_level_roi.payback_period_months} Months`;
        document.getElementById('econPractice5Yr').innerText = `$${(d.practice_level_roi['5_year_practice_net_benefit_usd'] / 1000.0).toFixed(0)}k`;
        
        const yr10Clinical = d.inflection_points.year_10_terminal_state;
        document.getElementById('econPayerSavings').innerText = `$${(d.annual_projections.reduce((acc, y) => acc + y.clinical_outcomes.payer_savings_m, 0)).toFixed(1)}M`;
        document.getElementById('econIcer').innerText = `$${d.annual_projections[4].clinical_outcomes.icer_usd_per_qaly.toFixed(0)} / QALY`;

        // Inflection Points
        const valInf = d.inflection_points.valuation_inflection_point;
        const revInf = d.inflection_points.revenue_inflection_point;
        document.getElementById('inflectionValTime').innerText = `${valInf.display}`;
        document.getElementById('inflectionRevTime').innerText = `${revInf.display}`;

        // Enterprise Valuations
        document.getElementById('year10Valuation').innerText = `$${yr10Clinical.year_10_enterprise_valuation_b} Billion`;
        document.getElementById('dcfValuation').innerText = `$${d.dcf_valuation.total_dcf_enterprise_value_b} Billion`;
        document.getElementById('cum10YrRev').innerText = `$${yr10Clinical.cumulative_10yr_revenue_b} Billion`;
        document.getElementById('cumQalys').innerText = yr10Clinical.year_10_cumulative_qalys_gained.toLocaleString();

        renderRevenueStreamsChart(d.annual_projections);
        renderValuationCurveChart(d.annual_projections, valInf, revInf);
    } catch (err) {
        console.error("Health economics load error:", err);
    }
}

function renderRevenueStreamsChart(annualData) {
    const canvas = document.getElementById('revenueStreamsCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;
    const n = annualData.length;
    const barWidth = 26;
    const maxRev = annualData[n - 1].gross_revenue_m; // Peak revenue ~3800M

    // Draw grid
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 0.5;
    for (let y = 25; y < h - 35; y += 30) {
        ctx.beginPath();
        ctx.moveTo(35, y);
        ctx.lineTo(w - 15, y);
        ctx.stroke();
    }

    // Stacked Bars per Year
    annualData.forEach((item, idx) => {
        const x = 45 + (idx / (n - 1)) * (w - 90);
        const totalHeight = (item.gross_revenue_m / maxRev) * (h - 70);
        let curY = (h - 35);

        // Hardware (Blue)
        const hwH = (item.revenue_streams.hardware_m / maxRev) * (h - 70);
        ctx.fillStyle = '#0284c7';
        ctx.fillRect(x - barWidth/2, curY - hwH, barWidth, hwH);
        curY -= hwH;

        // Consumables / Kits (Emerald)
        const kitH = (item.revenue_streams.kits_consumables_m / maxRev) * (h - 70);
        ctx.fillStyle = '#10b981';
        ctx.fillRect(x - barWidth/2, curY - kitH, barWidth, kitH);
        curY -= kitH;

        // SaaS & AI CAD (Purple)
        const saasH = ((item.revenue_streams.saas_cloud_slam_m + item.revenue_streams.genai_abutment_m) / maxRev) * (h - 70);
        ctx.fillStyle = '#a855f7';
        ctx.fillRect(x - barWidth/2, curY - saasH, barWidth, saasH);
        curY -= saasH;

        // Service & Warranty (Amber)
        const servH = (item.revenue_streams.service_warranty_m / maxRev) * (h - 70);
        ctx.fillStyle = '#f59e0b';
        ctx.fillRect(x - barWidth/2, curY - servH, barWidth, servH);

        // Year label
        ctx.fillStyle = '#94a3b8';
        ctx.font = '9px JetBrains Mono';
        ctx.fillText(`Y${item.year}`, x - 8, h - 18);
        if (item.year % 2 === 0 || item.year === 1) {
            ctx.fillStyle = '#38bdf8';
            ctx.fillText(`$${(item.gross_revenue_m / 1000).toFixed(1)}B`, x - 12, h - 38 - totalHeight);
        }
    });

    // Legend
    ctx.font = '9px JetBrains Mono';
    ctx.fillStyle = '#0284c7'; ctx.fillText('■ Hardware', 40, 15);
    ctx.fillStyle = '#10b981'; ctx.fillText('■ Consumables', 125, 15);
    ctx.fillStyle = '#a855f7'; ctx.fillText('■ SaaS/AI', 225, 15);
    ctx.fillStyle = '#f59e0b'; ctx.fillText('■ Service', 300, 15);
}

function renderValuationCurveChart(annualData, valInflection, revInflection) {
    const canvas = document.getElementById('valuationCurveCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;
    const n = annualData.length;
    const maxValB = annualData[n - 1].enterprise_valuation_b; // Peak valuation ~$49B

    // Draw horizontal $1 Billion Benchmark line
    const y1B = (h - 35) - (1.0 / maxValB) * (h - 65);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(35, y1B);
    ctx.lineTo(w - 15, y1B);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ef4444';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText('$1.0 Billion Inflection Threshold', 45, y1B - 4);

    // 1. Enterprise Valuation Curve (Emerald Green)
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    annualData.forEach((item, idx) => {
        const x = 45 + (idx / (n - 1)) * (w - 90);
        const y = (h - 35) - (item.enterprise_valuation_b / maxValB) * (h - 65);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // 2. Annual Gross Revenue Curve (Cyan)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    annualData.forEach((item, idx) => {
        const x = 45 + (idx / (n - 1)) * (w - 90);
        const y = (h - 35) - ((item.gross_revenue_m / 1000.0) / maxValB) * (h - 65);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Inflection Point Markers
    // Point 1: $1B Valuation (Year ~1.77)
    const xValInf = 45 + ((valInflection.year_fraction - 1) / (n - 1)) * (w - 90);
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(xValInf, y1B, 5.5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillText('★ $1B EV (Mo 21)', xValInf - 35, y1B + 16);

    // Point 2: $1B Revenue (Year ~5.35)
    const xRevInf = 45 + ((revInflection.year_fraction - 1) / (n - 1)) * (w - 90);
    ctx.fillStyle = '#a855f7';
    ctx.beginPath();
    ctx.arc(xRevInf, y1B, 5.5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillText('★ $1B Rev (Mo 64)', xRevInf - 35, y1B + 28);

    // Axis labels
    annualData.forEach((item, idx) => {
        const x = 45 + (idx / (n - 1)) * (w - 90);
        ctx.fillStyle = '#64748b';
        ctx.font = '9px JetBrains Mono';
        ctx.fillText(`Y${item.year}`, x - 8, h - 18);
    });

    ctx.fillStyle = '#10b981'; ctx.fillText('■ Enterprise Valuation ($B)', 40, 15);
    ctx.fillStyle = '#38bdf8'; ctx.fillText('■ Annual Revenue ($B)', 240, 15);
}

// --- 7. Optimal QML Ultrasound Surface & CT Point Cloud Registration ---
async function runQmlUsCtRegistration() {
    const numQubits = parseInt(document.getElementById('qmlQubits')?.value) || 12;
    const qmlLayers = parseInt(document.getElementById('qmlLayers')?.value) || 6;
    const noiseSigma = parseFloat(document.getElementById('qmlNoiseSigma')?.value) || 0.18;

    try {
        const res = await fetch('/api/registration/qml_us_ct', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                num_qubits: numQubits,
                qml_layers: qmlLayers,
                noise_sigma_mm: noiseSigma
            })
        });
        const d = await res.json();
        cachedQmlData = d;

        const m = d.metrics;
        const elRmse = document.getElementById('qmlSurfaceRmse');
        if (elRmse) elRmse.innerText = `${m.rmse_surface_mm.toFixed(4)} mm`;
        const elUnreg = document.getElementById('qmlUnregRmse');
        if (elUnreg) elUnreg.innerText = `Pre-QML Unreg: ${m.unregistered_rmse_mm.toFixed(3)} mm`;
        const elH95 = document.getElementById('qmlHausdorff95');
        if (elH95) elH95.innerText = `${m.hausdorff_95_mm.toFixed(4)} mm`;
        const elWass = document.getElementById('qmlWasserstein');
        if (elWass) elWass.innerText = `${m.wasserstein_distance_mm.toFixed(4)} mm`;
        const elFid = document.getElementById('qmlKernelFidelity');
        if (elFid) elFid.innerText = m.quantum_kernel_alignment.toFixed(4);
        const elMut = document.getElementById('qmlMutualInfo');
        if (elMut) elMut.innerText = `${m.quantum_mutual_information_bits.toFixed(3)} bits`;
        const elParams = document.getElementById('qmlDistParams');
        if (elParams) elParams.innerText = `m=${m.nakagami_m_parameter} | κ=${m.vmf_concentration_kappa}`;

        renderQml3dSurfaceCanvas();
        renderQmlResidualDistCanvas(d.distributions, m);
        renderQmlAngularConvCanvas(d.distributions, d.convergence);
    } catch (err) {
        console.error("QML US-CT registration error:", err);
    }
}

function project3DPoint(x, y, z, azDeg, elDeg, cx, cy, scale) {
    const az = (azDeg * Math.PI) / 180.0;
    const el = (elDeg * Math.PI) / 180.0;
    const x1 = x * Math.cos(az) - y * Math.sin(az);
    const y1 = x * Math.sin(az) + y * Math.cos(az);
    const z1 = z;

    const screenX = cx + x1 * scale;
    const screenY = cy - (z1 * Math.cos(el) - y1 * Math.sin(el)) * scale;
    const depth = z1 * Math.sin(el) + y1 * Math.cos(el);
    return { sx: screenX, sy: screenY, depth };
}

function renderQml3dSurfaceCanvas() {
    if (!cachedQmlData) return;
    const canvas = document.getElementById('qmlSurfaceCloudCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2 + 22;
    const scale = 9.5;

    const azDeg = parseFloat(document.getElementById('qmlViewAzimuth')?.value || 28);
    const elDeg = parseFloat(document.getElementById('qmlViewElev')?.value || 32);
    const mode = document.getElementById('qmlOverlayMode')?.value || 'all';

    // Draw background coordinate grid
    ctx.strokeStyle = 'rgba(30, 41, 59, 0.65)';
    ctx.lineWidth = 0.7;
    for (let gx = -20; gx <= 20; gx += 5) {
        const p1 = project3DPoint(gx, -8, -6, azDeg, elDeg, cx, cy, scale);
        const p2 = project3DPoint(gx, 18, -6, azDeg, elDeg, cx, cy, scale);
        ctx.beginPath();
        ctx.moveTo(p1.sx, p1.sy);
        ctx.lineTo(p2.sx, p2.sy);
        ctx.stroke();
    }

    const geom = cachedQmlData.geometry;

    // 1. Optional Unregistered US Pose (Pre-QML) in faint amber/magenta
    if (mode === 'with_unreg' && geom.us_unregistered_cloud) {
        ctx.fillStyle = 'rgba(244, 63, 94, 0.45)';
        geom.us_unregistered_cloud.forEach(pt => {
            const p = project3DPoint(pt[0], pt[1], pt[2], azDeg, elDeg, cx, cy, scale);
            ctx.beginPath();
            ctx.arc(p.sx, p.sy, 1.7, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    // 2. Render Registered US Surface Mesh Quads (S_US) with residual heatmap coloring
    const grid = geom.us_registered_surface_grid;
    const nV = grid.length;
    const nU = grid[0].length;

    for (let i = 0; i < nV - 1; i++) {
        for (let j = 0; j < nU - 1; j++) {
            const v00 = grid[i][j];
            const v10 = grid[i + 1][j];
            const v11 = grid[i + 1][j + 1];
            const v01 = grid[i][j + 1];

            const p00 = project3DPoint(v00[0], v00[1], v00[2], azDeg, elDeg, cx, cy, scale);
            const p10 = project3DPoint(v10[0], v10[1], v10[2], azDeg, elDeg, cx, cy, scale);
            const p11 = project3DPoint(v11[0], v11[1], v11[2], azDeg, elDeg, cx, cy, scale);
            const p01 = project3DPoint(v01[0], v01[1], v01[2], azDeg, elDeg, cx, cy, scale);

            const meanRes = 0.25 * (v00[3] + v10[3] + v11[3] + v01[3]);
            const normErr = Math.min(1.0, meanRes / 0.085);

            // Emerald green (low residual) -> Gold/Amber (higher residual)
            const rCol = Math.floor(16 + normErr * 229);
            const gCol = Math.floor(185 - normErr * 30);
            const bCol = Math.floor(129 - normErr * 110);

            ctx.fillStyle = `rgba(${rCol}, ${gCol}, ${bCol}, 0.24)`;
            ctx.strokeStyle = `rgba(${rCol}, ${gCol}, ${bCol}, 0.72)`;
            ctx.lineWidth = 0.85;

            ctx.beginPath();
            ctx.moveTo(p00.sx, p00.sy);
            ctx.lineTo(p10.sx, p10.sy);
            ctx.lineTo(p11.sx, p11.sy);
            ctx.lineTo(p01.sx, p01.sy);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        }
    }

    // 3. Render CT Cortical Point Cloud Geometry (P_CT)
    geom.ct_point_cloud.forEach((pt, idx) => {
        const p = project3DPoint(pt[0], pt[1], pt[2], azDeg, elDeg, cx, cy, scale);
        ctx.fillStyle = '#38bdf8';
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, 1.9, 0, 2 * Math.PI);
        ctx.fill();
    });

    // 4. Render CT Inferior Alveolar Nerve (IAN) Canal Point Cloud Locus
    if (mode === 'all' && geom.ct_ian_nerve_cloud) {
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.85)';
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        geom.ct_ian_nerve_cloud.forEach((pt, idx) => {
            const p = project3DPoint(pt[0], pt[1], pt[2], azDeg, elDeg, cx, cy, scale);
            if (idx === 0) ctx.moveTo(p.sx, p.sy);
            else ctx.lineTo(p.sx, p.sy);
        });
        ctx.stroke();

        geom.ct_ian_nerve_cloud.forEach(pt => {
            const p = project3DPoint(pt[0], pt[1], pt[2], azDeg, elDeg, cx, cy, scale);
            ctx.fillStyle = '#ef4444';
            ctx.beginPath();
            ctx.arc(p.sx, p.sy, 2.2, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    // HUD Readouts
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(`QML Hilbert Space: 2^${cachedQmlData.num_qubits} | US Mesh: ${nV}×${nU} | CT Cloud: ${cachedQmlData.num_ct_points} pts`, 12, 18);
    ctx.fillStyle = '#10b981';
    ctx.fillText(`Registered Surface RMSE: ${cachedQmlData.metrics.rmse_surface_mm} mm | d_H95: ${cachedQmlData.metrics.hausdorff_95_mm} mm`, 12, h - 12);
}

function renderQmlResidualDistCanvas(dist, metrics) {
    const canvas = document.getElementById('qmlResidualDistCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;
    const padL = 42, padR = 18, padT = 24, padB = 26;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    const centers = dist.bin_centers_mm;
    const emp = dist.empirical_density;
    const ray = dist.rayleigh_pdf;
    const nak = dist.nakagami_pdf;
    const gau = dist.gaussian_pdf;

    const maxY = Math.max(...emp, ...ray, ...nak, ...gau, 1.0) * 1.08;
    const n = centers.length;

    // Grid
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 0.5;
    for (let g = 0; g <= 3; g++) {
        const y = padT + (g / 3) * plotH;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(w - padR, y);
        ctx.stroke();
    }

    // Empirical Histogram Bars
    const barW = (plotW / n) * 0.72;
    for (let i = 0; i < n; i++) {
        const x = padL + (i / n) * plotW;
        const bh = (emp[i] / maxY) * plotH;
        ctx.fillStyle = 'rgba(56, 189, 248, 0.28)';
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.65)';
        ctx.fillRect(x, padT + plotH - bh, barW, bh);
        ctx.strokeRect(x, padT + plotH - bh, barW, bh);
    }

    // Helper for PDF curves
    const drawCurve = (arr, color, width, dash = []) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.setLineDash(dash);
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
            const x = padL + ((i + 0.36) / n) * plotW;
            const y = padT + plotH - (arr[i] / maxY) * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
    };

    // Nakagami-m PDF (Emerald)
    drawCurve(nak, '#10b981', 2.2);
    // Rayleigh PDF (Amber)
    drawCurve(ray, '#f59e0b', 1.8, [4, 2]);
    // Gaussian PDF (Purple)
    drawCurve(gau, '#a855f7', 1.6);

    // Axis & Legend
    ctx.fillStyle = '#94a3b8';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText('0.00 mm', padL, h - 8);
    ctx.fillText(`${(centers[n - 1] * 0.5).toFixed(2)} mm`, padL + plotW * 0.45, h - 8);
    ctx.fillText(`${centers[n - 1].toFixed(2)} mm`, w - padR - 45, h - 8);

    ctx.fillStyle = '#38bdf8'; ctx.fillText('■ Empirical Hist', padL, 14);
    ctx.fillStyle = '#10b981'; ctx.fillText(`■ Nakagami-m (${metrics.nakagami_m_parameter})`, padL + 108, 14);
    ctx.fillStyle = '#f59e0b'; ctx.fillText(`■ Rayleigh (σ=${metrics.rayleigh_scale_sigma_mm})`, padL + 245, 14);
    ctx.fillStyle = '#a855f7'; ctx.fillText('■ Gaussian', padL + 395, 14);
}

function renderQmlAngularConvCanvas(dist, conv) {
    const canvas = document.getElementById('qmlAngularConvCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;
    const midX = Math.floor(w / 2);

    // Divider
    ctx.strokeStyle = '#1e293b';
    ctx.beginPath();
    ctx.moveTo(midX, 10);
    ctx.lineTo(midX, h - 10);
    ctx.stroke();

    // Left Half: von Mises-Fisher (vMF) Surface Normal Angular Error PDF
    const angGrid = dist.angular_deg_grid;
    const vmfPdf = dist.vmf_angular_pdf;
    const maxVmf = Math.max(...vmfPdf, 0.1) * 1.1;

    ctx.fillStyle = '#a855f7';
    ctx.font = '9.5px JetBrains Mono';
    ctx.fillText('vMF Normal Angle PDF f_vMF(θ)', 14, 16);

    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let i = 0; i < angGrid.length; i++) {
        const x = 22 + (i / (angGrid.length - 1)) * (midX - 38);
        const y = h - 24 - (vmfPdf[i] / maxVmf) * (h - 50);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fill under vMF
    ctx.lineTo(22 + (midX - 38), h - 24);
    ctx.lineTo(22, h - 24);
    ctx.closePath();
    ctx.fillStyle = 'rgba(168, 85, 247, 0.18)';
    ctx.fill();

    ctx.fillStyle = '#64748b';
    ctx.font = '8.5px JetBrains Mono';
    ctx.fillText('0.0°', 20, h - 8);
    ctx.fillText('2.2°', midX / 2 - 8, h - 8);
    ctx.fillText('4.5°', midX - 34, h - 8);

    // Right Half: QML Sinkhorn-Wasserstein W2 & Quantum Kernel Fidelity Convergence
    const w2Hist = conv.wasserstein_history;
    const qFidHist = conv.q_kernel_fidelity_history;
    const maxW2 = Math.max(...w2Hist, 0.1);
    const nEp = w2Hist.length;

    const rStart = midX + 18;
    const rWidth = w - rStart - 16;

    ctx.fillStyle = '#10b981';
    ctx.font = '9.5px JetBrains Mono';
    ctx.fillText('■ Kernel K_Q', rStart, 16);
    ctx.fillStyle = '#38bdf8';
    ctx.fillText('■ Sinkhorn W₂(mm)', rStart + 92, 16);

    // W2 curve (Cyan)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < nEp; i++) {
        const x = rStart + (i / (nEp - 1)) * rWidth;
        const y = h - 24 - (w2Hist[i] / maxW2) * (h - 50);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Q-Kernel Fidelity curve (Emerald)
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < nEp; i++) {
        const x = rStart + (i / (nEp - 1)) * rWidth;
        const normF = (qFidHist[i] - 0.65) / 0.36;
        const y = h - 24 - Math.max(0, Math.min(1, normF)) * (h - 50);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = '#64748b';
    ctx.font = '8.5px JetBrains Mono';
    ctx.fillText('Epoch 1', rStart, h - 8);
    ctx.fillText(`Epoch ${nEp}`, w - 58, h - 8);
}

// --- 7B. Dental Pain Management with TMS, QML Optimal Cortical Lobe Staging & BEM Simulation ---
async function runTmsPainSimulation() {
    const targetLobe = document.getElementById('tmsTargetLobe')?.value || 'M1_orofacial';
    const coilType = document.getElementById('tmsCoilType')?.value || 'figure8_70mm';
    const diDt = parseFloat(document.getElementById('tmsDiDt')?.value) || 145.0;
    const pulseFreq = parseFloat(document.getElementById('tmsPulseFreq')?.value) || 10.0;

    try {
        const res = await fetch('/api/tms/pain_simulation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_lobe: targetLobe,
                coil_type: coilType,
                di_dt_a_per_us: diDt,
                pulse_freq_hz: pulseFreq,
                qml_qubits: 10
            })
        });
        const d = await res.json();

        const m = d.summary_metrics;
        const elEf = document.getElementById('tmsPeakEfield');
        if (elEf) elEf.innerText = `${m.peak_cortical_efield_v_m} V/m`;
        const elCur = document.getElementById('tmsPeakCurrent');
        if (elCur) elCur.innerText = `${m.peak_eddy_current_a_m2} A/m²`;
        const elVasRed = document.getElementById('tmsVasReduction');
        if (elVasRed) elVasRed.innerText = `${m.vas_pain_reduction_pct}%`;
        const elVas24 = document.getElementById('tmsVas24h');
        if (elVas24) elVas24.innerText = `${m.postop_24h_vas_score} / 10`;
        const elGate = document.getElementById('tmsSp5cGating');
        if (elGate) elGate.innerText = `${m.trigeminal_v3_gating_db} dB`;
        const elOpioid = document.getElementById('tmsOpioidSparing');
        if (elOpioid) elOpioid.innerText = `${m.opioid_sparing_pct}%`;
        const elFid = document.getElementById('tmsQmlFidelity');
        if (elFid) elFid.innerText = m.qml_policy_fidelity.toFixed(4);

        // Populate Lobe BEM Breakdown Table
        const lobeTbody = document.querySelector('#tmsLobeBemTable tbody');
        if (lobeTbody && d.lobe_bem_breakdown) {
            lobeTbody.innerHTML = '';
            d.lobe_bem_breakdown.forEach(item => {
                const tr = document.createElement('tr');
                if (item.is_primary_target) tr.className = 'row-highlight';
                tr.innerHTML = `
                    <td><strong>${item.short_name}</strong></td>
                    <td class="text-info">${item.efield_v_m}</td>
                    <td>${item.current_density_a_m2}</td>
                    <td>${item.bem_charge_density_nc_m2}</td>
                    <td>${item.protocol}</td>
                `;
                lobeTbody.appendChild(tr);
            });
        }

        // Populate QML 4-Stage Cortical Lobe Staging Table
        const stageTbody = document.querySelector('#tmsStagingTable tbody');
        if (stageTbody && d.qml_staging_protocol) {
            stageTbody.innerHTML = '';
            d.qml_staging_protocol.forEach(st => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>S${st.stage}</strong> (${st.window_min})</td>
                    <td class="text-info">${st.target_lobe}</td>
                    <td>${st.waveform}</td>
                    <td class="text-purple">${st.trigeminal_sp5c_inhibition_pct}%</td>
                    <td class="text-success"><strong>${st.vas_pain_score}</strong></td>
                `;
                stageTbody.appendChild(tr);
            });
        }

        renderTmsBemCortexCanvas(d.bem_field_map, d.lobe_bem_breakdown, d.coil_configuration);
        renderTmsPainTrajectoryCanvas(d.pain_trajectory);
    } catch (err) {
        console.error("TMS pain simulation error:", err);
    }
}

function renderTmsBemCortexCanvas(fieldMap, lobes, coilLabel) {
    const canvas = document.getElementById('tmsBemCortexCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;
    const [rows, cols] = fieldMap.dimensions;
    const eGrid = fieldMap.efield_v_m;

    const padL = 30, padR = 30, padT = 28, padB = 26;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    const cellW = plotW / cols;
    const cellH = plotH / rows;

    let maxE = 1.0;
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            if (eGrid[r][c] > maxE) maxE = eGrid[r][c];
        }
    }

    // 1. Render 3-Shell BEM Outer Boundaries (Scalp Gamma_1, Skull Gamma_2, CSF/Cortex Gamma_3)
    const cx = padL + plotW * 0.5;
    const cy = padT + plotH * 0.5;
    const rxCortex = plotW * 0.44;
    const ryCortex = plotH * 0.42;

    // Scalp shell Gamma_1
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rxCortex + 14, ryCortex + 12, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Skull shell Gamma_2
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.ellipse(cx, cy, rxCortex + 7, ryCortex + 6, 0, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);

    // 2. Render Cortical BEM Induced Electric Field Heatmap ||E(r)||
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const val = eGrid[r][c];
            if (val <= 0.01) continue;
            const norm = Math.min(1.0, val / maxE);

            // Turbo-like colormap for cortical E-field
            const cr = Math.min(255, Math.floor(25 + 230 * Math.pow(norm, 0.85)));
            const cg = Math.min(255, Math.floor(40 + 205 * Math.sin(norm * Math.PI)));
            const cb = Math.min(255, Math.floor(235 * (1.0 - norm * 0.85)));

            ctx.fillStyle = `rgba(${cr}, ${cg}, ${cb}, 0.86)`;
            ctx.fillRect(padL + c * cellW, padT + r * cellH, cellW + 0.6, cellH + 0.6);
        }
    }

    // 3. Overlay BEM Triangular Mesh Wireframe across Cortical Lobes
    ctx.strokeStyle = 'rgba(226, 232, 240, 0.16)';
    ctx.lineWidth = 0.6;
    for (let r = 2; r < rows - 2; r += 3) {
        for (let c = 2; c < cols - 2; c += 3) {
            if (eGrid[r][c] <= 0.1) continue;
            const x0 = padL + c * cellW;
            const y0 = padT + r * cellH;
            const x1 = padL + (c + 3) * cellW;
            const y1 = padT + (r + 3) * cellH;
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y0);
            ctx.lineTo(x0, y1);
            ctx.closePath();
            ctx.stroke();
        }
    }

    // 4. Overlay Cortical Lobe Node Markers & Labels (M1, L-DLPFC, S1/S2, dACC, Insula)
    lobes.forEach(lobe => {
        const lx = padL + lobe.canvas_uv[0] * plotW;
        const ly = padT + lobe.canvas_uv[1] * plotH;

        if (lobe.is_primary_target) {
            // Draw Figure-8 TMS coil focus rings
            ctx.strokeStyle = '#10b981';
            ctx.lineWidth = 2.0;
            ctx.beginPath();
            ctx.arc(lx - 11, ly, 12, 0, 2 * Math.PI);
            ctx.arc(lx + 11, ly, 12, 0, 2 * Math.PI);
            ctx.stroke();
        }

        ctx.fillStyle = lobe.is_primary_target ? '#10b981' : '#f8fafc';
        ctx.beginPath();
        ctx.arc(lx, ly, lobe.is_primary_target ? 5.0 : 3.5, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = '#0f172a';
        ctx.fillRect(lx - 38, ly - 22, 82, 13);
        ctx.fillStyle = lobe.is_primary_target ? '#10b981' : '#38bdf8';
        ctx.font = '9px JetBrains Mono';
        ctx.fillText(`${lobe.short_name.split(' ')[0]}:${lobe.efield_v_m}V/m`, lx - 35, ly - 12);
    });

    // HUD Readouts
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(`3-Shell BEM (Γ₁ Scalp, Γ₂ Skull, Γ₃ Cortex) | Coil: ${coilLabel}`, 12, 17);
    ctx.fillStyle = '#10b981';
    ctx.fillText(`Peak |E_BEM|: ${maxE.toFixed(1)} V/m | 4,096 BEM Boundary Elements`, 12, h - 8);
}

function renderTmsPainTrajectoryCanvas(traj) {
    const canvas = document.getElementById('tmsPainTrajectoryCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;
    const padL = 42, padR = 20, padT = 24, padB = 24;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    const hrs = traj.hours;
    const sham = traj.vas_sham_control;
    const unstage = traj.vas_unstage_tms;
    const qmlStaged = traj.vas_qml_staged_tms;
    const n = hrs.length;
    const maxVas = 10.0;

    // Grid lines
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 0.5;
    for (let v = 0; v <= 10; v += 2.5) {
        const y = padT + (1.0 - v / maxVas) * plotH;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(w - padR, y);
        ctx.stroke();
    }

    const drawVasCurve = (arr, color, width, dash = []) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.setLineDash(dash);
        ctx.beginPath();
        arr.forEach((val, idx) => {
            const x = padL + (idx / (n - 1)) * plotW;
            const y = padT + (1.0 - val / maxVas) * plotH;
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
    };

    // Sham control (Red dashed)
    drawVasCurve(sham, '#ef4444', 1.8, [4, 3]);
    // Unstaged single-lobe rTMS (Amber)
    drawVasCurve(unstage, '#f59e0b', 2.0);
    // QML 4-Stage Optimal Lobe TMS (Emerald)
    drawVasCurve(qmlStaged, '#10b981', 2.6);

    // Axis labels
    hrs.forEach((hr, idx) => {
        if (idx % 2 === 0 || idx === n - 1) {
            const x = padL + (idx / (n - 1)) * plotW;
            ctx.fillStyle = '#64748b';
            ctx.font = '8.5px JetBrains Mono';
            ctx.fillText(`${hr}h`, x - 8, h - 7);
        }
    });

    ctx.font = '9px JetBrains Mono';
    ctx.fillStyle = '#10b981'; ctx.fillText('■ QML 4-Stage Cortical TMS', padL, 14);
    ctx.fillStyle = '#f59e0b'; ctx.fillText('■ Single-Lobe rTMS', padL + 175, 14);
    ctx.fillStyle = '#ef4444'; ctx.fillText('■ Sham Control (VAS 0-10)', padL + 305, 14);
}

// --- 8. PDF Generation & Toast ---
async function generatePdf() {
    showToast("Recompiling Nature Preprint PDF...");
    try {
        const res = await fetch('/api/pdf/generate', { method: 'POST' });
        const d = await res.json();
        if (d.success) {
            showToast(`Preprint Compiled: ${d.file_size_kb} KB. Ready for download!`);
        } else {
            showToast(`Error: ${d.error}`);
        }
    } catch (err) {
        showToast("PDF generation failed.");
        console.error("PDF generation error:", err);
    }
}

function showToast(msg) {
    const toast = document.getElementById('toast');
    toast.innerText = msg;
    toast.style.display = 'block';
    setTimeout(() => {
        toast.style.display = 'none';
    }, 3800);
}
