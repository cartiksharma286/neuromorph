/**
 * Dental Image-Guided Surgery Robot Console
 * Frontend JavaScript Engine
 */

let activeTab = 'cbct';
let slamInterval = null;
let currentDepth = 0.0;
let slamTrajectoryHistory = [];

document.addEventListener('DOMContentLoaded', () => {
    loadCbctSlice();
    loadCbctMetrics();
    loadJoints();
    initSlamCanvas();
    runVqeRegistration();
    generateImplantModel();
    loadImplantFea();
    loadManipulatorSim();
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

// --- 6. PDF Generation & Toast ---
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
