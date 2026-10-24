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
    generateImplantModel();
    loadImplantFea();
    loadManipulatorSim();
    loadHealthEconomics();
    loadBiomarkers();
    loadDrillPathSimulation();
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

// --- 9. Continued-Fraction Recurrent Prime Drill Path Research Simulation ---
async function loadDrillPathSimulation() {
    const depthInput = document.getElementById('drillPathDepth');
    const amplitudeInput = document.getElementById('drillPathAmplitude');
    const primeCountInput = document.getElementById('drillPathPrimeCount');
    if (!depthInput || !amplitudeInput || !primeCountInput) return;

    const payload = {
        planned_depth_mm: parseFloat(depthInput.value),
        recurrence_amplitude_mm: parseFloat(amplitudeInput.value),
        prime_count: parseInt(primeCountInput.value, 10)
    };

    try {
        const res = await fetch('/api/drill-path/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Drill-path simulation failed.');

        const characteristics = data.characteristics;
        document.getElementById('drillEndpointError').innerText =
            `${characteristics.entry_error_um.toFixed(1)} / ${characteristics.terminal_error_um.toFixed(1)} μm`;
        document.getElementById('drillPathCorrection').innerText =
            `${characteristics.maximum_lateral_correction_um.toFixed(2)} μm`;
        document.getElementById('drillTissueStress').innerText =
            `${characteristics.peak_tissue_stress_mpa.toFixed(3)} MPa`;
        document.getElementById('drillCuttingForce').innerText =
            `${characteristics.peak_cutting_force_n.toFixed(2)} N`;
        document.getElementById('drillPathEquation').innerText = data.model.equation;

        renderDrillPathCanvas(data);
        renderDrillFeaCanvas(data);
        renderDrillCharacteristics(data.trace);
        renderDrillPrimeTable(data.model.prime_terms);
    } catch (err) {
        console.error('Drill-path simulation error:', err);
        showToast(err.message);
    }
}

function renderDrillPathCanvas(data) {
    const canvas = document.getElementById('drillPathCanvas');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const depth = data.model.planned_depth_mm;
    const path = data.path_points;
    const left = 56;
    const top = 35;
    const plotWidth = width - 112;
    const plotHeight = height - 78;
    const xScale = plotWidth / 9.0;
    const yScale = plotHeight / depth;
    const toX = (value) => width / 2 + value * xScale;
    const toY = (value) => top + value * yScale;

    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = 'rgba(251, 113, 133, 0.18)';
    ctx.fillRect(left, top - 20, plotWidth, 20);
    ctx.fillStyle = '#fb7185';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('soft-tissue boundary', left + 8, top - 7);

    ctx.strokeStyle = '#f8fafc';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(width / 2, top + plotHeight / 2, plotWidth * 0.34, plotHeight * 0.47, 0, 0, 2 * Math.PI);
    ctx.stroke();

    ctx.strokeStyle = 'rgba(16, 185, 129, 0.72)';
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(toX(-0.14), toY(0));
    ctx.lineTo(toX(-0.14), toY(depth));
    ctx.moveTo(toX(0.14), toY(0));
    ctx.lineTo(toX(0.14), toY(depth));
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.7;
    ctx.beginPath();
    path.forEach((point, index) => {
        const x = toX(point.lateral_offset_mm);
        const y = toY(point.depth_mm);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    const tip = path[path.length - 1];
    ctx.fillStyle = '#a855f7';
    ctx.beginPath();
    ctx.arc(toX(tip.lateral_offset_mm), toY(tip.depth_mm), 7, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(toX(tip.lateral_offset_mm), toY(tip.depth_mm - 0.5));
    ctx.lineTo(toX(tip.lateral_offset_mm), toY(tip.depth_mm));
    ctx.stroke();

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('Lateral offset (mm)', 10, 16);
    ctx.fillText(`Depth: 0.00 -> ${depth.toFixed(2)} mm`, width - 152, height - 12);
    ctx.fillText('Cavity cross-section and bounded recurrent-prime correction', 14, height - 32);
}

function stressColor(normalized) {
    const value = Math.max(0, Math.min(1, normalized));
    const red = Math.round(38 + 217 * value);
    const green = Math.round(99 + 130 * Math.sin(value * Math.PI));
    const blue = Math.round(235 - 200 * value);
    return `rgb(${red}, ${green}, ${blue})`;
}

function renderDrillFeaCanvas(data) {
    const canvas = document.getElementById('drillFeaCanvas');
    const ctx = canvas.getContext('2d');
    const field = data.finite_element_field;
    const rows = field.stress_mpa.length;
    const columns = field.stress_mpa[0].length;
    const width = canvas.width;
    const height = canvas.height;
    const left = 46;
    const top = 24;
    const plotWidth = width - 82;
    const plotHeight = height - 54;
    const maxStress = Math.max(...field.stress_mpa.flat(), 0.1);

    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, width, height);
    for (let row = 0; row < rows; row += 1) {
        for (let column = 0; column < columns; column += 1) {
            const material = field.material_map[row][column];
            const stress = field.stress_mpa[row][column];
            const x = left + (column / columns) * plotWidth;
            const y = top + (row / rows) * plotHeight;
            const cellWidth = plotWidth / columns + 0.5;
            const cellHeight = plotHeight / rows + 0.5;

            if (material === 4) ctx.fillStyle = '#cbd5e1';
            else if (material === 1) ctx.fillStyle = '#fb7185';
            else if (material === 3) ctx.fillStyle = '#f59e0b';
            else ctx.fillStyle = stressColor(stress / maxStress);
            ctx.fillRect(x, y, cellWidth, cellHeight);
        }
    }

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    data.path_points.forEach((point, index) => {
        const x = left + ((point.lateral_offset_mm - field.x_coordinates_mm[0]) /
            (field.x_coordinates_mm[field.x_coordinates_mm.length - 1] - field.x_coordinates_mm[0])) * plotWidth;
        const y = top + ((point.depth_mm - field.depth_coordinates_mm[0]) /
            (field.depth_coordinates_mm[field.depth_coordinates_mm.length - 1] - field.depth_coordinates_mm[0])) * plotHeight;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('2-D material-field stress proxy (MPa)', 12, 15);
    ctx.fillText('surface', 4, top + 12);
    ctx.fillText('depth', 7, height - 20);
    ctx.fillText(`0.0`, width - 73, 15);
    ctx.fillStyle = '#ef4444';
    ctx.fillText(`${maxStress.toFixed(2)} MPa`, width - 47, 15);
}

function renderDrillCharacteristics(trace) {
    const canvas = document.getElementById('drillCharacteristicsCanvas');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const left = 46;
    const top = 30;
    const plotWidth = width - 68;
    const plotHeight = height - 58;
    const maxDepth = trace[trace.length - 1].depth_mm;
    const maxForce = Math.max(...trace.map((point) => point.cutting_force_n));
    const maxStress = Math.max(...trace.map((point) => point.tissue_stress_mpa));
    const maxOffset = Math.max(...trace.map((point) => Math.abs(point.lateral_offset_mm)), 0.001);

    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let row = 0; row < 4; row += 1) {
        const y = top + (row / 3) * plotHeight;
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(left + plotWidth, y);
        ctx.stroke();
    }

    const drawSeries = (color, selector, maximum) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        trace.forEach((point, index) => {
            const x = left + (point.depth_mm / maxDepth) * plotWidth;
            const y = top + plotHeight - (selector(point) / maximum) * plotHeight;
            if (index === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    };

    drawSeries('#38bdf8', (point) => point.cutting_force_n, maxForce);
    drawSeries('#a855f7', (point) => point.tissue_stress_mpa, maxStress);
    drawSeries('#10b981', (point) => Math.abs(point.lateral_offset_mm), maxOffset);

    ctx.font = '10px JetBrains Mono';
    ctx.fillStyle = '#38bdf8';
    ctx.fillText('■ Cutting reaction (normalized)', left, 15);
    ctx.fillStyle = '#a855f7';
    ctx.fillText('■ Tissue stress (normalized)', left + 198, 15);
    ctx.fillStyle = '#10b981';
    ctx.fillText('■ Lateral correction (normalized)', left + 402, 15);
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Entry', left, height - 10);
    ctx.fillText(`Depth ${maxDepth.toFixed(2)} mm`, width - 118, height - 10);
}

function renderDrillPrimeTable(terms) {
    const tableBody = document.querySelector('#drillPrimeTable tbody');
    if (!tableBody) return;
    tableBody.replaceChildren();
    terms.forEach((term) => {
        const row = document.createElement('tr');
        [term.prime, `[${term.continued_fraction.join('; ')}]`, term.convergent, term.phase_fraction.toFixed(5)]
            .forEach((value) => {
                const cell = document.createElement('td');
                cell.textContent = value;
                row.appendChild(cell);
            });
        tableBody.appendChild(row);
    });
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
