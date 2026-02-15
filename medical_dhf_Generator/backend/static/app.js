const API_URL = 'http://localhost:8000';

const views = {
    dashboard: `
        <div class="dashboard-grid">
            <div class="glass-panel stat-card">
                <div class="stat-label">Total Devices</div>
                <div class="stat-value">12</div>
                <div style="color: var(--success)">+2 this month</div>
            </div>
            <div class="glass-panel stat-card">
                <div class="stat-label">Pending FDA 510(k)</div>
                <div class="stat-value" style="color: var(--warning)">3</div>
            </div>
            <div class="glass-panel stat-card">
                <div class="stat-label">Avg Risk Score</div>
                <div class="stat-value">2.4</div>
                <div style="color: var(--success)">-0.3 vs last audit</div>
            </div>
        </div>
        <div class="glass-panel" style="margin-top: 2rem; padding: 1.5rem;">
            <h3>Active Compliance Tasks</h3>
            <ul>
                <li>Update ISO 14971 Risk File for Device X</li>
                <li>Review Clinical Data for Stent Y</li>
            </ul>
        </div>
    `,
    'risk-analysis': `
        <h2 class="header-title">ISO 14971 Risk Analysis</h2>
        <div class="glass-panel" style="padding: 2rem;">
            <div style="display: flex; gap: 2rem;">
                <div style="flex: 1;">
                    <h3>Risk Matrix (Severity vs Probability)</h3>
                    <div class="risk-matrix" id="risk-matrix-container"></div>
                </div>
                <div style="width: 300px;">
                    <h3>Selected Risk</h3>
                    <div id="risk-details" class="glass-panel" style="padding: 1rem; min-height: 200px;">
                        Select a cell to view details.
                    </div>
                </div>
            </div>
        </div>
    `,
    'mr-opt': `
        <h2 class="header-title">Quantum Pulse Sequence Optimization (VQE)</h2>
        <div class="glass-panel" style="padding: 1.5rem; margin-bottom: 2rem;">
            <h3>Optimization Parameters</h3>
            <div style="display: flex; gap: 1rem; align-items: end;">
                <div>
                    <label style="color: var(--text-muted); font-size: 0.9rem;">Target Flip Angle (rad)</label>
                    <input type="number" id="flip-angle" value="1.57" step="0.1" style="background: rgba(255,255,255,0.1); border: 1px solid var(--glass-border); color: white; padding: 0.5rem; border-radius: 4px; display: block;">
                </div>
                <div>
                    <label style="color: var(--text-muted); font-size: 0.9rem;">Duration (ms)</label>
                    <input type="number" id="duration" value="5.0" step="0.5" style="background: rgba(255,255,255,0.1); border: 1px solid var(--glass-border); color: white; padding: 0.5rem; border-radius: 4px; display: block;">
                </div>
                <button class="btn" onclick="runQuantumOptimization()" style="background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);">
                    🚀 Run Quantum VQE
                </button>
            </div>
            <div id="optimization-status" style="margin-top: 1rem; color: var(--text-muted);">Ready to calibrate.</div>
        </div>

        <div class="dashboard-grid">
            <div class="glass-panel stat-card">
                <h3>Pulse Waveform (RF Amplitude)</h3>
                <canvas id="pulseChart"></canvas>
            </div>
            <div class="glass-panel stat-card">
                <h3>Optimization Metrics</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <div class="stat-label">Final Flip Angle</div>
                        <div class="stat-value" id="res-flip">-</div>
                    </div>
                    <div>
                        <div class="stat-label">Achieved SAR</div>
                        <div class="stat-value" id="res-sar">-</div>
                    </div>
                    <div>
                        <div class="stat-label">Convergence Cost</div>
                        <div class="stat-value" id="res-cost">-</div>
                    </div>
                    <div>
                        <div class="stat-label">Iterations</div>
                        <div class="stat-value" id="res-iter">-</div>
                    </div>
                </div>
            </div>
        </div>
    `,
    'stents': `
        <h2 class="header-title">Coronary Stent Risk Profile</h2>
        <div class="dashboard-grid">
            <div class="glass-panel stat-card">
                <h3>Radial Force Analysis</h3>
                <canvas id="forceChart"></canvas>
            </div>
            <div class="glass-panel stat-card">
                <h3>Fatigue Cycles</h3>
                <div class="stat-value">400M</div>
                <div class="stat-label">Safety Factor: 1.5x</div>
            </div>
        </div>
    `,
    'standards': `
        <h2 class="header-title">IEC 13485:2016 Standards</h2>
        <div class="glass-panel" style="padding: 1.5rem; margin-bottom: 2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3>Compliance Overview</h3>
                    <p style="color: var(--text-muted)">Project: NeuroStent III (Class III)</p>
                </div>
                <div class="stat-value" style="color: var(--primary)">33%</div>
            </div>
            <div style="background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; margin-top: 1rem; overflow: hidden;">
                <div style="background: var(--primary); height: 100%; width: 33%"></div>
            </div>
        </div>

        <div class="dashboard-grid" id="standards-grid">
            <!-- Injected -->
        </div>
    `,
    'iec60601': `
        <h2 class="header-title">IEC 60601-1 Safety Standards</h2>
        <div class="glass-panel" style="padding: 1.5rem; margin-bottom: 2rem;">
            <h3>Medical Electrical Equipment Safety</h3>
            <p style="color: var(--text-muted)">General requirements for basic safety and essential performance.</p>
        </div>
        <div class="dashboard-grid" id="iec60601-grid">
            <!-- Injected -->
        </div>
    `,
    'documents': `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
            <h2 class="header-title" style="margin: 0;">QMS Documentation</h2>
            <div>
                <button class="btn" onclick="createNewDoc()">+ New Document</button>
            </div>
        </div>
        
        <div class="glass-panel" style="padding: 1.5rem;">
            <h3>Document Register</h3>
            <table style="width: 100%; border-collapse: collapse; margin-top: 1rem; color: var(--text-main);">
                <thead>
                    <tr style="text-align: left; border-bottom: 1px solid var(--glass-border);">
                        <th style="padding: 1rem;">ID</th>
                        <th style="padding: 1rem;">Title</th>
                        <th style="padding: 1rem;">Ver</th>
                        <th style="padding: 1rem;">Status</th>
                        <th style="padding: 1rem;">Actions</th>
                    </tr>
                </thead>
                <tbody id="doc-list">
                    <tr><td colspan="5" style="padding:1rem;">Loading...</td></tr>
                </tbody>
            </table>
        </div>
    `
};

function navigate(viewName) {
    const mainContent = document.getElementById('main-content');

    // Update active nav
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    const activeNav = document.querySelector(`.nav-item[onclick="navigate('${viewName}')"]`);
    if (activeNav) activeNav.classList.add('active');

    // Update Content
    if (views[viewName]) {
        mainContent.innerHTML = views[viewName];
        if (viewName === 'risk-analysis') initRiskMatrix();
        // if (viewName === 'mr-opt') initMrCharts(); // Now handled via user action
        if (viewName === 'stents') initStentCharts();
        if (viewName === 'standards') initStandards();
        if (viewName === 'documents') loadDocuments();
    } else {
        mainContent.innerHTML = `<h2>Page Not Found</h2>`;
    }
}

function runQuantumOptimization() {
    const angle = parseFloat(document.getElementById('flip-angle').value);
    const duration = parseFloat(document.getElementById('duration').value);
    const status = document.getElementById('optimization-status');

    status.innerHTML = 'Initializing Quantum Circuit Simulation...';
    status.style.color = '#fbbf24'; // Yellow

    fetch('/api/optimize-pulse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_flip_angle: angle, duration_ms: duration })
    })
        .then(res => res.json())
        .then(data => {
            status.innerHTML = 'Optimization Complete.';
            status.style.color = '#34d399'; // Green

            // Update Stats
            document.getElementById('res-flip').innerText = data.final_flip_angle.toFixed(4);
            document.getElementById('res-sar').innerText = data.final_sar.toFixed(4);
            document.getElementById('res-cost').innerText = data.cost.toFixed(6);
            document.getElementById('res-iter').innerText = data.iterations;

            // Chart
            const ctx = document.getElementById('pulseChart').getContext('2d');

            // Destroy old chart if exists
            if (window.myPulseChart) window.myPulseChart.destroy();

            window.myPulseChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.time_points.map(t => t.toFixed(2)),
                    datasets: [{
                        label: 'RF Amplitude (uT)',
                        data: data.optimized_waveform,
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { labels: { color: 'white' } },
                    },
                    scales: {
                        y: {
                            ticks: { color: 'white' },
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            title: { display: true, text: 'Amplitude (uT)', color: 'white' }
                        },
                        x: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                    }
                }
            });
        })
        .catch(err => {
            console.error(err);
            status.innerHTML = 'Optimization Failed.';
            status.style.color = '#ef4444';
        });
}

function loadDocuments() {
    const list = document.getElementById('doc-list');
    if (!list) return;
    list.innerHTML = '<tr><td colspan="5" style="text-align:center; padding: 2rem;">Loading documents...</td></tr>';

    fetch('/api/documents')
        .then(res => res.json())
        .then(data => {
            list.innerHTML = '';
            if (data.length === 0) {
                list.innerHTML = '<tr><td colspan="5" style="text-align:center; padding: 2rem; color: var(--text-muted)">No documents found. Create one to get started.</td></tr>';
                return;
            }
            data.forEach(doc => {
                const tr = document.createElement('tr');
                tr.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
                let statusColor = 'var(--text-muted)';
                if (doc.approval_status === 'APPROVED') statusColor = 'var(--success)';
                if (doc.approval_status === 'DRAFT') statusColor = 'var(--warning)';
                if (doc.approval_status === 'REJECTED') statusColor = 'var(--primary)';

                tr.innerHTML = `
                    <td style="padding: 1rem;">${doc.id}</td>
                    <td style="padding: 1rem;">${doc.title}</td>
                    <td style="padding: 1rem;">${doc.version}</td>
                    <td style="padding: 1rem;"><span style="color: ${statusColor}">${doc.approval_status}</span></td>
                    <td style="padding: 1rem;">
                        ${doc.approval_status !== 'APPROVED' ?
                        `<button class="btn" style="padding: 0.5rem 1rem; font-size: 0.8rem;" onclick="approveDoc('${doc.id}')">Approve</button>` :
                        `<span style="color: var(--text-muted); font-size: 0.8rem;">Approved by ${doc.approved_by}</span>`
                    }
                        ${doc.approval_status === 'DRAFT' ?
                        `<button class="btn" style="padding: 0.5rem 1rem; font-size: 0.8rem; background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239,68,68,0.5); margin-left: 0.5rem;" onclick="rejectDoc('${doc.id}')">Reject</button>` : ''
                    }
                    </td>
                `;
                list.appendChild(tr);
            });
        })
        .catch(err => {
            console.error(err);
            list.innerHTML = '<tr><td colspan="5" style="text-align:center; padding: 2rem; color: #ef4444">Error loading documents.</td></tr>';
        });
}

function generateVAndV() {
    const btn = event.target; // Simple hack, ideally pass ID
    const originalText = btn.innerText;
    btn.innerText = "Generating...";
    btn.disabled = true;

    fetch('/api/generate-doc?device_id=NeuroStent_III&doc_type=verification_plan', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            alert("V&V Plan Generated: " + data.doc_id);
            loadDocuments();
            btn.innerText = originalText;
            btn.disabled = false;
        })
        .catch(err => {
            alert("Generation failed");
            btn.innerText = originalText;
            btn.disabled = false;
        });
}

function createNewDoc() {
    const id = prompt("Enter Document ID (e.g. SOP-005):");
    if (!id) return;
    const title = prompt("Enter Document Title:");
    if (!title) return;

    const newDoc = {
        id: id,
        title: title,
        version: "1.0",
        author: "CurrentUser",
        approval_status: "DRAFT",
        last_updated: new Date().toISOString(),
        content: "Draft content..."
    };

    fetch('/api/documents/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newDoc)
    })
        .then(res => {
            if (res.ok) {
                alert('Document Created!');
                loadDocuments();
            } else {
                alert('Error creating document.');
            }
        });
}

function approveDoc(id) {
    fetch(`/api/documents/${id}/approve?approver=Admin`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            loadDocuments();
        })
        .catch(err => alert("Error approving document"));
}

function rejectDoc(id) {
    const reason = prompt("Enter Rejection Reason:");
    if (!reason) return;

    fetch(`/api/documents/${id}/reject?reason=${encodeURIComponent(reason)}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            loadDocuments();
        })
        .catch(err => alert("Error rejecting document"));
}

function initStandards() {
    const grid = document.getElementById('standards-grid');
    if (!grid) return;
    grid.innerHTML = '<p>Loading standards...</p>';

    fetch('/api/standards/iec13485')
        .then(res => res.json())
        .then(data => {
            grid.innerHTML = '';
            data.forEach(clause => {
                const card = document.createElement('div');
                card.className = 'glass-panel stat-card';
                card.style.transition = '0.2s';
                card.style.cursor = 'pointer';
                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between;">
                        <span style="background: rgba(59, 130, 246, 0.2); color: var(--primary); padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">Clause ${clause.clause_number}</span>
                        <span style="color: var(--text-muted); font-size: 0.8rem;">${clause.parent_id ? 'Sub-clause' : 'Main'}</span>
                    </div>
                    <h3 style="margin: 0.5rem 0;">${clause.title}</h3>
                    <p style="color: var(--text-muted); font-size: 0.9rem; line-height: 1.4;">${clause.description}</p>
                    <div style="margin-top: 1rem; display: flex; gap: 0.5rem;">
                         <span style="font-size: 0.8rem; color: var(--success);">● Compliant</span>
                    </div>
                `;
                grid.appendChild(card);
            });
        })
        .catch(err => {
            console.error(err);
            grid.innerHTML = '<p>Error loading standards.</p>';
        });
}

function initRiskMatrix() {
    const container = document.getElementById('risk-matrix-container');
    const severities = ['Negligible', 'Minor', 'Serious', 'Critical', 'Catastrophic'];
    const probs = ['Improbable', 'Remote', 'Occasional', 'Probable', 'Frequent'];

    // Header
    container.appendChild(document.createElement('div')); // Empty corner
    probs.forEach(p => {
        const div = document.createElement('div');
        div.className = 'viz-label';
        div.innerText = p;
        container.appendChild(div);
    });

    for (let s = 4; s >= 0; s--) { // Render top to bottom
        const label = document.createElement('div');
        label.className = 'severity-label';
        label.innerText = severities[s];
        container.appendChild(label);

        for (let p = 0; p < 5; p++) {
            const val = (s + 1) * (p + 1);
            const cell = document.createElement('div');
            cell.className = 'risk-cell glass-panel';
            cell.innerText = val;

            // Color coding
            if (val <= 4) cell.style.background = 'rgba(16, 185, 129, 0.4)'; // Green
            else if (val <= 12) cell.style.background = 'rgba(245, 158, 11, 0.4)'; // Yellow/Orange
            else cell.style.background = 'rgba(239, 68, 68, 0.4)'; // Red

            cell.onclick = () => showRiskDetails(severities[s], probs[p], val);
            container.appendChild(cell);
        }
    }
}

function showRiskDetails(sev, prob, val) {
    const details = document.getElementById('risk-details');
    details.innerHTML = `
        <h4>${sev} / ${prob} (Score: ${val})</h4>
        <p>Example Risks:</p>
        <ul>
            ${val > 10 ? '<li>Patient death due to device failure</li>' : ''}
            ${val > 4 ? '<li>Minor injury requiring intervention</li>' : ''}
            <li>System warning logged</li>
        </ul>
        <button class="btn" style="width: 100%; margin-top: 1rem;">View Mitigations</button>
    `;
}

function initStentCharts() {
    const ctx = document.getElementById('forceChart').getContext('2d');

    // Simulate API Call
    const mockData = {
        radial_force_n: [1.3, 1.4, 1.25, 1.35],
        fatigue_cycles: [390000000, 410000000, 385000000],
        recoil_percent: [3, 4, 3.5]
    };

    fetch('/api/analyze-stent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mockData)
    })
        .then(res => res.json())
        .then(data => {
            console.log("Stent Analysis:", data);
            new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Radial Strength', 'Flexibility', 'Recoil', 'Foreshortening', 'Coverage'],
                    datasets: [{
                        label: `Current Design (Risk: ${data.fatigue_risk_prob})`,
                        data: [90, 70, 85, 60, 80],
                        backgroundColor: 'rgba(59, 130, 246, 0.2)',
                        borderColor: '#3b82f6'
                    }]
                },
                options: { responsive: true, plugins: { legend: { labels: { color: 'white' } } }, scales: { r: { ticks: { display: false }, grid: { color: 'rgba(255,255,255,0.1)' } } } }
            });
        });
}

// Init
navigate('dashboard');
