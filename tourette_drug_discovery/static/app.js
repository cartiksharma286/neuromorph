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
}
