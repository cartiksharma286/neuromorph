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
        <h2 class="header-title">MR Pulse Sequence Optimization</h2>
        <div class="dashboard-grid">
            <div class="glass-panel stat-card">
                <h3>SAR Constraints</h3>
                <canvas id="sarChart"></canvas>
            </div>
            <div class="glass-panel stat-card">
                <h3>Gradient Slew Rates</h3>
                <div class="stat-value">180 T/m/s</div>
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
    'documents': `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
            <h2 class="header-title" style="margin: 0;">QMS Documentation</h2>
            <button class="btn" onclick="createNewDoc()">+ New Document</button>
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
                    <!-- Suggested Mock Data -->
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding: 1rem;">DOC-001</td>
                        <td style="padding: 1rem;">Quality Manual</td>
                        <td style="padding: 1rem;">1.0</td>
                        <td style="padding: 1rem;"><span style="color: var(--warning)">DRAFT</span></td>
                        <td style="padding: 1rem;">
                            <button class="btn" style="padding: 0.5rem 1rem; font-size: 0.8rem;" onclick="approveDoc('DOC-001')">Approve</button>
                        </td>
                    </tr>
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
        if (viewName === 'mr-opt') initMrCharts();
        if (viewName === 'stents') initStentCharts();
        if (viewName === 'standards') initStandards();
        // if (viewName === 'documents') loadDocuments(); // In real app, fetch docs
    } else {
        mainContent.innerHTML = `<h2>Page Not Found</h2>`;
    }
}

function createNewDoc() {
    alert("Create New Document Feature: Opens modal to enter Title, Author, etc.");
}

function approveDoc(id) {
    fetch(`/api/documents/${id}/approve?approver=Admin`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            alert(`Document ${id} Approved!`);
            navigate('documents'); // Refresh
        })
        .catch(err => alert("Error approving document"));
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

function initMrCharts() {
    const ctx = document.getElementById('sarChart').getContext('2d');

    // Simulate API Call
    const mockData = { sar_levels: [2.1, 2.5, 3.8, 1.9, 2.2], db_dt: [150, 160, 190, 140, 145] };

    fetch('/api/analyze-mr-pulse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mockData)
    })
        .then(res => res.json())
        .then(data => {
            console.log("MR Analysis:", data);
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['1ms', '2ms', '3ms', '4ms', '5ms'],
                    datasets: [{
                        label: 'Global SAR (W/kg)',
                        data: mockData.sar_levels,
                        borderColor: '#3b82f6',
                        tension: 0.4
                    }, {
                        label: 'Limit (4.0 W/kg)',
                        data: [4, 4, 4, 4, 4],
                        borderColor: '#ef4444',
                        borderDash: [5, 5]
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { labels: { color: 'white' } },
                        title: { display: true, text: `Risk Analysis: ${data.sar_risk_prob}`, color: 'white' }
                    },
                    scales: { y: { ticks: { color: 'white' } }, x: { ticks: { color: 'white' } } }
                }
            });
        });
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
