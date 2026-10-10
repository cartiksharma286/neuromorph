/**
 * Tourette Syndrome QML & Number-Theoretic Drug Discovery Client Application
 * Handles tabs, real-time VQE execution, 3D/2D proteomic backbone animation,
 * interactive Chart.js clinical & health economic distributions, and preprint management.
 */

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadTargets();
    loadCompounds();
    initFoldingCanvas();
    loadClinicalTrajectories();
    loadHealthEconomics();
    loadTMSModule();
    setupEventListeners();
});

// Tab Navigation
function initTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    const panes = document.querySelectorAll('.tab-pane');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            panes.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const targetPane = document.getElementById(tab.dataset.tab);
            if (targetPane) {
                targetPane.classList.add('active');
            }
        });
    });
}

// Global state
let currentTargets = {};
let currentCoordinates = [];
let animAngle = 0;
let isAnimating = true;

// Tab 1: Load Genomic Targets
async function loadTargets() {
    try {
        const res = await fetch('/api/targets');
        const json = await res.json();
        if (json.status === 'success') {
            currentTargets = json.targets;
            updateTargetDossier('DRD2');
            runVQEOptimization('DRD2');
        }
    } catch (err) {
        console.error('Error loading targets:', err);
    }
}

function updateTargetDossier(targetKey) {
    const target = currentTargets[targetKey];
    if (!target) return;

    const dossier = document.getElementById('target-dossier');
    dossier.innerHTML = `
        <div class="target-title">${target.name} (UniProt: ${target.uniprot} | Locus: ${target.locus})</div>
        <div class="target-desc"><b>Pathology:</b> ${target.pathology}</div>
        <div class="target-desc" style="margin-top:4px;">
            <b>Binding Pocket:</b> ${target.binding_pocket_volume_A3} Å³ &nbsp;|&nbsp; 
            <b>Reference K<sub>i</sub>:</b> ${target.reference_affinity_nM} nM &nbsp;|&nbsp; 
            <b>Elliptic Modulus <i>k</i>:</b> ${target.elliptic_modulus_k}
        </div>
    `;

    document.getElementById('elliptic-k').value = target.elliptic_modulus_k;
    document.getElementById('k-val').innerText = target.elliptic_modulus_k;
}

// Execute VQE
async function runVQEOptimization(targetKey) {
    const kVal = parseFloat(document.getElementById('elliptic-k').value);
    const pVal = parseInt(document.getElementById('galois-p').value);

    document.getElementById('vqe-status').innerText = 'Simulating QPU...';
    document.getElementById('vqe-status').className = 'badge badge-accent';

    try {
        const res = await fetch('/api/qml/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target: targetKey, num_params: 6, seed: Math.floor(Math.random() * 1000) })
        });
        const json = await res.json();
        if (json.status === 'success') {
            const vqe = json.vqe_result;
            document.getElementById('ground-energy').innerText = `${vqe.ground_state_energy_hartree.toFixed(2)} Ha`;
            document.getElementById('binding-energy').innerText = `${vqe.binding_free_energy_kcal_mol.toFixed(1)} kcal/mol`;
            document.getElementById('predicted-ki').innerText = `${vqe.predicted_ki_nM.toFixed(2)} nM`;
            document.getElementById('fidelity').innerText = `${(vqe.state_fidelity * 100).toFixed(1)}%`;

            document.getElementById('val-sn').innerText = vqe.jacobi_sn.toFixed(3);
            document.getElementById('val-cn').innerText = vqe.jacobi_cn.toFixed(3);
            document.getElementById('val-dn').innerText = vqe.jacobi_dn.toFixed(3);

            // Fetch and update folding simulation
            simulateProteomicFolding(targetKey);

            document.getElementById('vqe-status').innerText = 'VQE Converged';
            document.getElementById('vqe-status').className = 'badge badge-success';
        }
    } catch (err) {
        console.error('VQE error:', err);
        document.getElementById('vqe-status').innerText = 'Simulation Offline';
    }
}

// Simulate 3D Proteomic Folding
async function simulateProteomicFolding(targetKey) {
    try {
        const res = await fetch('/api/folding/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target: targetKey, residues: 110, compound_id: 'NM-TS-701' })
        });
        const json = await res.json();
        if (json.status === 'success') {
            currentCoordinates = json.folding.coordinates;
            document.getElementById('val-K').innerText = json.folding.complete_integral_K.toFixed(3);
            document.getElementById('val-E').innerText = json.folding.complete_integral_E.toFixed(3);
        }
    } catch (err) {
        console.error('Folding error:', err);
    }
}

// 3D Canvas Rendering
function initFoldingCanvas() {
    const canvas = document.getElementById('folding-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    function render() {
        ctx.fillStyle = '#020617';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw coordinate grid
        ctx.strokeStyle = 'rgba(30, 41, 59, 0.4)';
        ctx.lineWidth = 1;
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

        if (currentCoordinates.length > 1) {
            animAngle += 0.012;
            const cosA = Math.cos(animAngle);
            const sinA = Math.sin(animAngle);
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;
            const scale = 2.4;

            // Project 3D to 2D with rotation
            const projected = currentCoordinates.map(pt => {
                const rotX = pt[0] * cosA - pt[2] * sinA;
                const rotZ = pt[0] * sinA + pt[2] * cosA;
                const rotY = pt[1];
                const fov = 300 / (300 + rotZ * 0.4);
                return {
                    x: cx + rotX * scale * fov,
                    y: cy + rotY * scale * fov,
                    z: rotZ
                };
            });

            // Draw backbone lines
            for (let i = 0; i < projected.length - 1; i++) {
                const p1 = projected[i];
                const p2 = projected[i + 1];

                // Color gradient along residues
                const ratio = i / projected.length;
                const r = Math.floor(56 + ratio * 150);
                const g = Math.floor(189 - ratio * 60);
                const b = Math.floor(248);

                ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.stroke();
            }

            // Draw C-alpha atom beads
            for (let i = 0; i < projected.length; i += 4) {
                const p = projected[i];
                ctx.fillStyle = (i % 8 === 0) ? '#fbbf24' : '#38bdf8';
                ctx.beginPath();
                ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        requestAnimationFrame(render);
    }

    render();
}

// Tab 2: Load Compounds
async function loadCompounds() {
    try {
        const res = await fetch('/api/compounds');
        const json = await res.json();
        if (json.status === 'success') {
            renderCompoundCards(json.compounds);
        }
    } catch (err) {
        console.error('Error loading compounds:', err);
    }
}

function renderCompoundCards(compounds) {
    const container = document.getElementById('compounds-container');
    container.innerHTML = '';

    compounds.forEach(c => {
        const card = document.createElement('div');
        card.className = 'compound-card';
        card.innerHTML = `
            <div class="compound-header">
                <div>
                    <div class="compound-title">${c.code} • ${c.name}</div>
                    <div class="compound-formula">${c.formula_html}</div>
                </div>
                <span class="badge badge-accent">&Delta;G: ${c.delta_g_binding} kcal</span>
            </div>

            <div class="compound-svg-container">
                ${c.svg}
            </div>

            <div class="compound-props">
                <div class="prop-item">
                    <div class="prop-lbl">Molecular Wt</div>
                    <div class="prop-val">${c.molecular_weight}</div>
                </div>
                <div class="prop-item">
                    <div class="prop-lbl">LogP / tPSA</div>
                    <div class="prop-val">${c.logP} / ${c.tpsa} Å²</div>
                </div>
                <div class="prop-item">
                    <div class="prop-lbl">BBB logBB</div>
                    <div class="prop-val">${c.bbb_penetration_logBB > 0 ? '+' : ''}${c.bbb_penetration_logBB}</div>
                </div>
            </div>

            <div class="target-dossier" style="margin-top:0.5rem;">
                <div class="target-title">Primary Target & Affinity:</div>
                <div class="target-desc"><b>${c.primary_target}</b></div>
                <div class="target-desc" style="margin-top:3px;"><b>Recommended Dose:</b> ${c.dose_daily_mg}</div>
            </div>

            <div class="compound-mechanism">
                <b>Mechanism:</b> ${c.mechanism}
            </div>
        `;
        container.appendChild(card);
    });
}

// Tab 3: Clinical Trajectories & 4-Phase Recovery Plan
let chartYgtssTotal = null;
let chartSubscores = null;
let chartPuts = null;
let chartKaplanMeier = null;

async function loadClinicalTrajectories() {
    try {
        const res = await fetch('/api/clinical/trajectories?compound_id=NM-TS-701');
        const json = await res.json();
        if (json.status === 'success') {
            const data = json.data;
            renderClinicalCharts(data);
            renderRecoveryPlan(data.recovery_plan);
        }
    } catch (err) {
        console.error('Error loading clinical trajectories:', err);
    }
}

function renderClinicalCharts(data) {
    const months = data.months.map(m => `Month ${m}`);
    const traj = data.trajectories;

    // 1. YGTSS Total Chart
    const ctxTotal = document.getElementById('chart-ygtss-total').getContext('2d');
    if (chartYgtssTotal) chartYgtssTotal.destroy();
    chartYgtssTotal = new Chart(ctxTotal, {
        type: 'line',
        data: {
            labels: months,
            datasets: [
                {
                    label: 'NM-TS-701 (QML Conjugate)',
                    data: traj.qml_candidate.total,
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.15)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Standard of Care (Aripiprazole)',
                    data: traj.standard_of_care.total,
                    borderColor: '#f59e0b',
                    borderWidth: 2.2,
                    borderDash: [5, 5],
                    tension: 0.3
                },
                {
                    label: 'Placebo / Natural History',
                    data: traj.placebo.total,
                    borderColor: '#94a3b8',
                    borderWidth: 1.8,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { title: { display: true, text: 'YGTSS Total Score (0 - 100)', color: '#94a3b8' }, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
                x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#f8fafc', font: { size: 10 } } } }
        }
    });

    // 2. Motor vs Vocal Subscores
    const ctxSub = document.getElementById('chart-subscores').getContext('2d');
    if (chartSubscores) chartSubscores.destroy();
    chartSubscores = new Chart(ctxSub, {
        type: 'line',
        data: {
            labels: months,
            datasets: [
                {
                    label: 'Motor Tics (NM-TS-701)',
                    data: traj.qml_candidate.motor,
                    borderColor: '#10b981',
                    borderWidth: 2.5,
                    tension: 0.3
                },
                {
                    label: 'Vocal Tics (NM-TS-701)',
                    data: traj.qml_candidate.vocal,
                    borderColor: '#c084fc',
                    borderWidth: 2.5,
                    tension: 0.3
                },
                {
                    label: 'Motor Tics (Std of Care)',
                    data: traj.standard_of_care.motor,
                    borderColor: '#10b981',
                    borderDash: [4, 4],
                    borderWidth: 1.8
                },
                {
                    label: 'Vocal Tics (Std of Care)',
                    data: traj.standard_of_care.vocal,
                    borderColor: '#c084fc',
                    borderDash: [4, 4],
                    borderWidth: 1.8
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { title: { display: true, text: 'Subscore (0 - 25)', color: '#94a3b8' }, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
                x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#f8fafc', font: { size: 10 } } } }
        }
    });

    // 3. PUTS Urge Chart
    const ctxPuts = document.getElementById('chart-puts').getContext('2d');
    if (chartPuts) chartPuts.destroy();
    chartPuts = new Chart(ctxPuts, {
        type: 'line',
        data: {
            labels: months,
            datasets: [
                {
                    label: 'PUTS Urge (NM-TS-701)',
                    data: traj.qml_candidate.puts,
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    borderWidth: 2.8,
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'PUTS Urge (Std of Care)',
                    data: traj.standard_of_care.puts,
                    borderColor: '#f43f5e',
                    borderWidth: 2.0,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { title: { display: true, text: 'PUTS Urge Score (0 - 36)', color: '#94a3b8' }, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
                x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#f8fafc', font: { size: 10 } } } }
        }
    });

    // 4. Kaplan-Meier Curve
    const ctxKM = document.getElementById('chart-kaplan-meier').getContext('2d');
    if (chartKaplanMeier) chartKaplanMeier.destroy();
    const kmLabels = data.km_months.map(m => `M${m}`);
    chartKaplanMeier = new Chart(ctxKM, {
        type: 'line',
        data: {
            labels: kmLabels,
            datasets: [
                {
                    label: 'NM-TS-701 Remission Retention',
                    data: traj.qml_candidate.km_remission.map(v => v * 100),
                    borderColor: '#38bdf8',
                    borderWidth: 3,
                    stepped: true
                },
                {
                    label: 'Standard of Care Retention',
                    data: traj.standard_of_care.km_remission.map(v => v * 100),
                    borderColor: '#f59e0b',
                    borderWidth: 2,
                    stepped: true
                },
                {
                    label: 'Placebo Retention',
                    data: traj.placebo.km_remission.map(v => v * 100),
                    borderColor: '#94a3b8',
                    borderWidth: 1.5,
                    stepped: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { title: { display: true, text: 'Sustained Remission %', color: '#94a3b8' }, min: 0, max: 100, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
                x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#f8fafc', font: { size: 10 } } } }
        }
    });
}

function renderRecoveryPlan(plan) {
    const container = document.getElementById('recovery-plan-container');
    container.innerHTML = '';

    plan.forEach(p => {
        const div = document.createElement('div');
        div.className = 'protocol-card';
        div.innerHTML = `
            <div class="protocol-phase">${p.phase}</div>
            <div class="protocol-timeline">${p.timeline}</div>
            <div class="protocol-dose">${p.daily_dose}</div>
            <div class="protocol-obj"><b>Target Objective:</b> ${p.target_objective}</div>
            <div class="protocol-milestone"><b>Milestone:</b> ${p.clinical_milestone}</div>
            <div class="protocol-obj" style="margin-top:6px; font-size:0.75rem;"><b>Safety & Monitoring:</b> ${p.monitoring}</div>
        `;
        container.appendChild(div);
    });
}

// Tab 4: Health Economics
let chartCeac = null;
let chartPsa = null;
let chartQalyTime = null;
let chartCostTime = null;

async function loadHealthEconomics() {
    try {
        const res = await fetch('/api/economics/markov');
        const json = await res.json();
        if (json.status === 'success') {
            renderEconomicCharts(json.economics);
        }
    } catch (err) {
        console.error('Error loading health economics:', err);
    }
}

function renderEconomicCharts(econ) {
    // 1. CEAC Chart
    const ctxCeac = document.getElementById('chart-ceac').getContext('2d');
    if (chartCeac) chartCeac.destroy();
    chartCeac = new Chart(ctxCeac, {
        type: 'line',
        data: {
            labels: econ.wtp_thresholds.map(w => `$${Math.round(w / 1000)}k`),
            datasets: [{
                label: 'Probability Cost-Effective (NM-TS-701)',
                data: econ.ceac_probabilities.map(p => p * 100),
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.15)',
                borderWidth: 3,
                fill: true,
                tension: 0.2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { title: { display: true, text: 'Probability (%)', color: '#94a3b8' }, min: 80, max: 100, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
                x: { title: { display: true, text: 'Willingness to Pay Threshold ($/QALY)', color: '#94a3b8' }, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#f8fafc' } } }
        }
    });

    // 2. PSA Scatter Chart
    const ctxPsa = document.getElementById('chart-psa').getContext('2d');
    if (chartPsa) chartPsa.destroy();
    const scatterPoints = econ.psa_scatter.delta_qaly.map((q, i) => ({
        x: q,
        y: econ.psa_scatter.delta_cost[i]
    }));

    chartPsa = new Chart(ctxPsa, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Monte Carlo Trials (ΔCost vs ΔQALY)',
                data: scatterPoints,
                backgroundColor: 'rgba(56, 189, 248, 0.65)',
                pointRadius: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Incremental QALYs (ΔE)', color: '#94a3b8' }, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
                y: { title: { display: true, text: 'Incremental Cost ($ ΔC)', color: '#94a3b8' }, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#f8fafc' } } }
        }
    });

    // 3. QALY Time Chart
    const ctxQaly = document.getElementById('chart-qaly-time').getContext('2d');
    if (chartQalyTime) chartQalyTime.destroy();
    const months = Array.from({ length: 24 }, (_, i) => `M${i + 1}`);
    chartQalyTime = new Chart(ctxQaly, {
        type: 'line',
        data: {
            labels: months,
            datasets: [
                {
                    label: 'NM-TS-701 Cumulative QALY',
                    data: econ.qaly_timeline.qml,
                    borderColor: '#38bdf8',
                    borderWidth: 2.8,
                    tension: 0.2
                },
                {
                    label: 'Standard of Care Cumulative QALY',
                    data: econ.qaly_timeline.soc,
                    borderColor: '#f59e0b',
                    borderWidth: 2.2,
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { title: { display: true, text: 'Cumulative QALYs', color: '#94a3b8' }, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
                x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#f8fafc' } } }
        }
    });

    // 4. Cost Timeline Chart
    const ctxCost = document.getElementById('chart-cost-time').getContext('2d');
    if (chartCostTime) chartCostTime.destroy();
    chartCostTime = new Chart(ctxCost, {
        type: 'line',
        data: {
            labels: months,
            datasets: [
                {
                    label: 'NM-TS-701 Total Expenditure',
                    data: econ.cost_timeline.qml,
                    borderColor: '#10b981',
                    borderWidth: 2.8,
                    tension: 0.2
                },
                {
                    label: 'Standard of Care Total Expenditure',
                    data: econ.cost_timeline.soc,
                    borderColor: '#ef4444',
                    borderWidth: 2.2,
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { title: { display: true, text: 'Cumulative Costs ($ USD)', color: '#94a3b8' }, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
                x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#f8fafc' } } }
        }
    });
}

// Event Listeners
function setupEventListeners() {
    const targetSelect = document.getElementById('target-select');
    if (targetSelect) {
        targetSelect.addEventListener('change', (e) => {
            updateTargetDossier(e.target.value);
            runVQEOptimization(e.target.value);
        });
    }

    const sliderK = document.getElementById('elliptic-k');
    if (sliderK) {
        sliderK.addEventListener('input', (e) => {
            document.getElementById('k-val').innerText = e.target.value;
        });
    }

    const btnVqe = document.getElementById('btn-run-vqe');
    if (btnVqe) {
        btnVqe.addEventListener('click', () => {
            const targetKey = document.getElementById('target-select').value;
            runVQEOptimization(targetKey);
        });
    }

    const btnFolding = document.getElementById('btn-run-folding');
    if (btnFolding) {
        btnFolding.addEventListener('click', () => {
            const targetKey = document.getElementById('target-select').value;
            simulateProteomicFolding(targetKey);
        });
    }

    const btnRecompile = document.getElementById('btn-recompile-pdf');
    if (btnRecompile) {
        btnRecompile.addEventListener('click', async () => {
            btnRecompile.innerText = 'Recompiling...';
            try {
                const res = await fetch('/api/preprint/generate');
                const json = await res.json();
                if (json.status === 'success') {
                    btnRecompile.innerText = '✅ PDF Ready';
                    // Refresh iframe
                    const iframe = document.querySelector('.pdf-iframe');
                    if (iframe) {
                        iframe.src = '/api/preprint/view?t=' + Date.now();
                    }
                }
            } catch (err) {
                btnRecompile.innerText = '❌ Failed';
            }
            setTimeout(() => { btnRecompile.innerText = '🔄 Recompile PDF'; }, 3000);
        });
    }

    setupTMSEventListeners();
}

// --- TAB 5: TMS Cortical Simulation & Staging Logic ---

let currentTMSLobes = {};
let currentTMSPG = {};
let chartTMSEcon = null;
let chartTMSCash = null;

async function loadTMSModule() {
    try {
        const res = await fetch('/api/tms/lobes');
        const json = await res.json();
        if (json.status === 'success') {
            currentTMSLobes = json.lobes;
            currentTMSPG = json.pharmacogenomics;
            updateTMSLobeDossier('pre_sma');
            updateTMSGeneDossier('CYP2D6_EXTENSIVE');
            runTMSSimulation('pre_sma');
            loadTMSStaging('CYP2D6_EXTENSIVE');
            loadTMSRevenue(150);
            initTMSCanvas();
        }
    } catch (err) {
        console.error('Error loading TMS lobes:', err);
    }
}

function updateTMSLobeDossier(lobeKey) {
    const lobe = currentTMSLobes[lobeKey];
    if (!lobe) return;

    const dossier = document.getElementById('tms-lobe-dossier');
    dossier.innerHTML = `
        <div class="target-title">${lobe.name} (${lobe.broadmann})</div>
        <div class="target-desc"><b>Clinical Function:</b> ${lobe.role}</div>
        <div class="target-desc" style="margin-top:4px;">
            <b>Depth:</b> ${lobe.depth_mm} mm &nbsp;|&nbsp;
            <b>Conductivity &sigma;:</b> ${lobe.conductivity_s_m} S/m &nbsp;|&nbsp;
            <b>Permittivity &epsilon;<sub>r</sub>:</b> ${lobe.relative_permittivity} &nbsp;|&nbsp;
            <b>Protocol:</b> ${lobe.protocol_type}
        </div>
    `;
}

function updateTMSGeneDossier(pgKey) {
    const pg = currentTMSPG[pgKey];
    if (!pg) return;

    const dossier = document.getElementById('pg-dossier');
    dossier.innerHTML = `
        <div class="target-title"><b>Genomic Profile:</b> ${pg.gene} - ${pg.phenotype}</div>
        <div class="target-desc"><b>TMS & Drug Synergy:</b> ${pg.tms_synergy}</div>
        <div class="target-desc" style="margin-top:4px; color:var(--accent-green); font-weight:600;">
            <b>Response Efficacy Multiplier:</b> ${(pg.response_multiplier * 100).toFixed(0)}% Relative Target Rate
        </div>
    `;
}

async function runTMSSimulation(lobeKey) {
    const current = parseFloat(document.getElementById('tms-coil-current').value);
    const freq = parseFloat(document.getElementById('tms-pulse-freq').value);

    try {
        const res = await fetch('/api/tms/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lobe: lobeKey, coil_current: current, pulse_freq: freq })
        });
        const json = await res.json();
        if (json.status === 'success') {
            const exc = json.excitation;
            document.getElementById('tms-e-field').innerText = `${exc.e_induced_v_m} V/m`;
            document.getElementById('tms-depol-ratio').innerText = `${exc.depolarization_ratio}x Threshold`;
            if (document.getElementById('tms-didt')) {
                document.getElementById('tms-didt').innerText = `${exc.di_dt_a_per_us} A/µs`;
            }
            if (document.getElementById('tms-sar')) {
                document.getElementById('tms-sar').innerText = `${exc.sar_peak_w_kg} W/kg`;
            }

            if (document.getElementById('val-b-surf')) {
                document.getElementById('val-b-surf').innerText = `${exc.b_surface_tesla} T`;
            }
            if (document.getElementById('val-b-peak')) {
                document.getElementById('val-b-peak').innerText = `${exc.b_target_tesla} T`;
            }
            if (document.getElementById('val-bem-factor')) {
                document.getElementById('val-bem-factor').innerText = `${exc.bem_focusing_multiplier}x`;
            }
            if (document.getElementById('val-half-depth')) {
                document.getElementById('val-half-depth').innerText = `${exc.z_half_depth_mm} mm`;
            }

            const statusBadge = document.getElementById('tms-depol-status');
            if (exc.effective_excitation_achieved) {
                statusBadge.innerText = `Depolarization: ACTIVE (${exc.e_induced_v_m} V/m)`;
                statusBadge.className = 'badge badge-success';
            } else {
                statusBadge.innerText = `Sub-Threshold (${exc.e_induced_v_m} V/m)`;
                statusBadge.className = 'badge badge-accent';
            }

            renderTMSCanvasField(exc);
        }
    } catch (err) {
        console.error('TMS simulation error:', err);
    }
}

async function loadTMSStaging(pgKey) {
    try {
        const res = await fetch(`/api/tms/staging?profile=${pgKey}`);
        const json = await res.json();
        if (json.status === 'success') {
            const container = document.getElementById('tms-roadmap-container');
            container.innerHTML = '';

            json.staging.stages.forEach(s => {
                const card = document.createElement('div');
                card.className = 'protocol-card';
                card.innerHTML = `
                    <div class="protocol-phase">${s.stage}</div>
                    <div class="protocol-timeline">${s.timeline}</div>
                    <div class="protocol-dose">${s.tms_protocol}</div>
                    <div class="protocol-obj"><b>Adjuvant:</b> ${s.drug_adjuvant}</div>
                    <div class="protocol-obj"><b>Mechanism:</b> ${s.mechanism}</div>
                    <div class="protocol-milestone"><b>Target YGTSS Reduction:</b> ${s.target_ygtss_reduction}</div>
                    <div class="protocol-obj" style="margin-top:4px; font-size:0.75rem;"><b>Endpoint:</b> ${s.clinical_endpoint}</div>
                `;
                container.appendChild(card);
            });
        }
    } catch (err) {
        console.error('Error loading TMS staging:', err);
    }
}

async function loadTMSRevenue(patients) {
    try {
        const res = await fetch(`/api/tms/revenue?patients=${patients}`);
        const json = await res.json();
        if (json.status === 'success') {
            const econ = json.economics;
            const proj = econ.three_year_projections;

            document.getElementById('rev-y1').innerText = `$${econ.three_year_projections.gross_revenue[0].toLocaleString()}`;
            document.getElementById('profit-y1').innerText = `$${econ.three_year_projections.net_operating_profit[0].toLocaleString()}`;
            document.getElementById('payback-stat').innerText = `${econ.payback_months} Months`;
            document.getElementById('cashflow-3yr').innerText = `$${econ.three_year_projections.cumulative_cashflow[2].toLocaleString()}`;

            renderTMSRevenueCharts(proj);
        }
    } catch (err) {
        console.error('Error loading TMS revenue:', err);
    }
}

function renderTMSRevenueCharts(proj) {
    // 1. Revenue & Operating Costs Bar Chart
    const ctxRev = document.getElementById('chart-tms-revenue').getContext('2d');
    if (chartTMSEcon) chartTMSEcon.destroy();
    chartTMSEcon = new Chart(ctxRev, {
        type: 'bar',
        data: {
            labels: proj.years,
            datasets: [
                {
                    label: 'Gross Commercial Revenue ($)',
                    data: proj.gross_revenue,
                    backgroundColor: 'rgba(56, 189, 248, 0.75)',
                    borderColor: '#38bdf8',
                    borderWidth: 1.5
                },
                {
                    label: 'Net Operating Profit ($ EBITDA)',
                    data: proj.net_operating_profit,
                    backgroundColor: 'rgba(16, 185, 129, 0.8)',
                    borderColor: '#10b981',
                    borderWidth: 1.5
                },
                {
                    label: 'Operating Costs ($)',
                    data: proj.operating_costs,
                    backgroundColor: 'rgba(239, 68, 68, 0.65)',
                    borderColor: '#ef4444',
                    borderWidth: 1.5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: { display: true, text: 'USD ($)', color: '#94a3b8' },
                    grid: { color: '#1e293b' },
                    ticks: { color: '#94a3b8', callback: v => `$${v / 1e6}M` }
                },
                x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#f8fafc', font: { size: 10 } } } }
        }
    });

    // 2. Cumulative Cashflow Chart
    const ctxCash = document.getElementById('chart-tms-cashflow').getContext('2d');
    if (chartTMSCash) chartTMSCash.destroy();
    chartTMSCash = new Chart(ctxCash, {
        type: 'line',
        data: {
            labels: ['Start (CapEx)', ...proj.years],
            datasets: [{
                label: 'Cumulative Net Commercial Cashflow ($ USD)',
                data: [-95000, ...proj.cumulative_cashflow],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.15)',
                borderWidth: 3,
                fill: true,
                tension: 0.25,
                pointRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: { display: true, text: 'Cumulative Cashflow ($)', color: '#94a3b8' },
                    grid: { color: '#1e293b' },
                    ticks: { color: '#94a3b8', callback: v => `$${v / 1e6}M` }
                },
                x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#f8fafc' } } }
        }
    });
}

// 2D FEM Field Canvas Simulation
let tmsCanvasCtx = null;
let tmsAnimOffset = 0;
let lastTMSExcitation = null;

function initTMSCanvas() {
    const canvas = document.getElementById('tms-field-canvas');
    if (!canvas) return;
    tmsCanvasCtx = canvas.getContext('2d');
    animateTMSCanvas();
}

function renderTMSCanvasField(exc) {
    lastTMSExcitation = exc;
}

function getEFieldColor(eVal, alpha = 0.85) {
    if (eVal < 40) {
        const t = Math.max(0, eVal / 40);
        return `rgba(${Math.floor(2 + t * 4)}, ${Math.floor(6 + t * 176)}, ${Math.floor(23 + t * 189)}, ${alpha})`;
    } else if (eVal < 82) {
        const t = (eVal - 40) / 42;
        return `rgba(${Math.floor(6 + t * 245)}, ${Math.floor(182 + t * 9)}, ${Math.floor(212 - t * 176)}, ${alpha})`;
    } else if (eVal < 120) {
        const t = (eVal - 82) / 38;
        return `rgba(${Math.floor(251 - t * 2)}, ${Math.floor(191 - t * 76)}, ${Math.floor(36 - t * 14)}, ${alpha})`;
    } else if (eVal < 160) {
        const t = (eVal - 120) / 40;
        return `rgba(${Math.floor(249 - t * 10)}, ${Math.floor(115 - t * 47)}, ${Math.floor(22 + t * 46)}, ${alpha})`;
    } else {
        const t = Math.min(1.0, (eVal - 160) / 60);
        return `rgba(${Math.floor(239 - t * 22)}, ${Math.floor(68 + t * 2)}, ${Math.floor(68 + t * 171)}, ${alpha})`;
    }
}

function animateTMSCanvas() {
    if (!tmsCanvasCtx) return;
    const canvas = document.getElementById('tms-field-canvas');
    const ctx = tmsCanvasCtx;

    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const cx = canvas.width / 2;
    const cy = 25; // Surface of scalp
    tmsAnimOffset += 0.04;

    const eMax = lastTMSExcitation ? lastTMSExcitation.e_induced_v_m : 169.3;
    const depthVal = (lastTMSExcitation && typeof lastTMSExcitation.depth_mm === 'number') ? lastTMSExcitation.depth_mm : 18.5;
    const targetY = cy + depthVal * 4.2;

    // 1. Draw Cranial BEM Layers with realistic anatomical depth arcs
    const cranialLayers = [
        { r: 250, label: 'S₅ Deep White Matter (21mm+)', color: '#090d16', stroke: '#1e293b' },
        { r: 180, label: 'S₄ Cortical Gray Matter (16–21mm)', color: '#0f172a', stroke: '#334155' },
        { r: 110, label: 'S₃ Cerebrospinal Fluid (13–16mm)', color: '#082f49', stroke: '#0284c7' },
        { r: 80, label: 'S₂ Skull Bone (6–13mm)', color: '#1e293b', stroke: '#475569' },
        { r: 50, label: 'S₁ Scalp (0–6mm)', color: '#334155', stroke: '#64748b' }
    ];

    cranialLayers.forEach(l => {
        ctx.beginPath();
        ctx.arc(cx, cy, l.r, 0, Math.PI);
        ctx.fillStyle = l.color;
        ctx.fill();
        ctx.strokeStyle = l.stroke;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Interface depth label
        ctx.fillStyle = 'rgba(148, 163, 184, 0.45)';
        ctx.font = '9px system-ui';
        ctx.textAlign = 'right';
        ctx.fillText(l.label, cx + l.r - 8, cy + 12);
    });

    // 2. Continuous Scientific Heatmap Colormap for Induced E-Field
    const heatGrad = ctx.createRadialGradient(cx, targetY, 4, cx, targetY, 110);
    heatGrad.addColorStop(0, getEFieldColor(eMax, 0.95));
    heatGrad.addColorStop(0.3, getEFieldColor(eMax * 0.75, 0.85));
    heatGrad.addColorStop(0.55, getEFieldColor(82.0, 0.65)); // 82 V/m threshold isocontour
    heatGrad.addColorStop(0.8, getEFieldColor(40.0, 0.35));
    heatGrad.addColorStop(1.0, 'transparent');

    ctx.fillStyle = heatGrad;
    ctx.beginPath();
    ctx.ellipse(cx, targetY, 130, 85, 0, 0, Math.PI * 2);
    ctx.fill();

    // 3. Draw Isocontour Lines & Threshold Circle
    const threshRadius = Math.max(15, 65 * (eMax / 120.0));
    ctx.beginPath();
    ctx.ellipse(cx, targetY, threshRadius, threshRadius * 0.65, 0, 0, Math.PI * 2);
    ctx.strokeStyle = '#fbbf24'; // Yellow threshold contour
    ctx.lineWidth = 2.2;
    ctx.setLineDash([5, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 4. Figure-of-8 Magnetic Field Lines
    const numLoops = 6;
    for (let i = 1; i <= numLoops; i++) {
        const phase = (tmsAnimOffset * 1.5 + i * 0.8) % (numLoops * 0.8);
        const loopR = 15 + phase * 14;
        const opacity = Math.max(0.05, 0.6 - (phase / (numLoops * 0.8)) * 0.55);

        ctx.strokeStyle = `rgba(56, 189, 248, ${opacity})`;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        // Left wing loop
        ctx.arc(cx - 38, cy + loopR * 0.6, loopR * 0.5, 0, Math.PI * 2);
        // Right wing loop
        ctx.arc(cx + 38, cy + loopR * 0.6, loopR * 0.5, 0, Math.PI * 2);
        ctx.stroke();
    }

    // 5. Target Crosshairs and Numeric Field Tag
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(cx - 12, targetY);
    ctx.lineTo(cx + 12, targetY);
    ctx.moveTo(cx, targetY - 12);
    ctx.lineTo(cx, targetY + 12);
    ctx.stroke();

    // Center focal bead
    ctx.fillStyle = '#f43f5e';
    ctx.beginPath();
    ctx.arc(cx, targetY, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Focal point text readout
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 11px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText(`${lastTMSExcitation ? lastTMSExcitation.lobe_name : 'Pre-SMA'} (Depth: ${depthVal}mm)`, cx, targetY - 16);

    ctx.fillStyle = '#fbbf24';
    ctx.font = 'bold 12px system-ui';
    ctx.fillText(`||E|| = ${eMax.toFixed(1)} V/m ${eMax >= 82 ? '(⚡ Depolarization Active)' : '(Sub-threshold)'}`, cx, targetY + 22);

    // 6. Draw Figure-of-8 Copper Coil Geometry on Scalp
    ctx.fillStyle = 'rgba(56, 189, 248, 0.9)';
    ctx.fillRect(cx - 72, cy - 14, 64, 12);
    ctx.fillRect(cx + 8, cy - 14, 64, 12);
    ctx.strokeStyle = '#f8fafc';
    ctx.lineWidth = 2;
    ctx.strokeRect(cx - 72, cy - 14, 64, 12);
    ctx.strokeRect(cx + 8, cy - 14, 64, 12);

    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 9px system-ui';
    ctx.fillText('◄ COIL WING 1', cx - 40, cy - 5);
    ctx.fillText('COIL WING 2 ►', cx + 40, cy - 5);

    requestAnimationFrame(animateTMSCanvas);
}

function setupTMSEventListeners() {
    const lobeSelect = document.getElementById('tms-lobe-select');
    if (lobeSelect) {
        lobeSelect.addEventListener('change', (e) => {
            updateTMSLobeDossier(e.target.value);
            runTMSSimulation(e.target.value);
        });
    }

    const currentSlider = document.getElementById('tms-coil-current');
    if (currentSlider) {
        currentSlider.addEventListener('input', (e) => {
            document.getElementById('current-val').innerText = `${parseInt(e.target.value).toLocaleString()} A`;
        });
    }

    const freqSlider = document.getElementById('tms-pulse-freq');
    if (freqSlider) {
        freqSlider.addEventListener('input', (e) => {
            document.getElementById('freq-val').innerText = `${parseFloat(e.target.value).toFixed(1)} kHz`;
        });
    }

    const btnRunTMS = document.getElementById('btn-run-tms');
    if (btnRunTMS) {
        btnRunTMS.addEventListener('click', () => {
            const lobe = document.getElementById('tms-lobe-select').value;
            runTMSSimulation(lobe);
        });
    }

    const pgSelect = document.getElementById('pg-select');
    if (pgSelect) {
        pgSelect.addEventListener('change', (e) => {
            updateTMSGeneDossier(e.target.value);
            loadTMSStaging(e.target.value);
        });
    }

    const patientSelect = document.getElementById('patient-volume-select');
    if (patientSelect) {
        patientSelect.addEventListener('change', (e) => {
            loadTMSRevenue(parseInt(e.target.value));
        });
    }

    const btnRecompileTMS = document.getElementById('btn-recompile-tms-pdf');
    if (btnRecompileTMS) {
        btnRecompileTMS.addEventListener('click', async () => {
            btnRecompileTMS.innerText = 'Recompiling...';
            try {
                const res = await fetch('/api/tms/preprint/generate');
                const json = await res.json();
                if (json.status === 'success') {
                    btnRecompileTMS.innerText = '✅ TMS PDF Ready';
                    const iframe = document.querySelector('iframe[title="Nature TMS Preprint PDF"]');
                    if (iframe) {
                        iframe.src = '/api/tms/preprint/view?t=' + Date.now();
                    }
                }
            } catch (err) {
                btnRecompileTMS.innerText = '❌ Failed';
            }
            setTimeout(() => { btnRecompileTMS.innerText = '🔄 Recompile TMS PDF'; }, 3000);
        });
    }
}
