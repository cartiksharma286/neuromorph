let ptsdChartInstance = null;

async function runSimulation() {
    const severity = document.getElementById('severity').value;
    const duration = document.getElementById('duration').value;

    const response = await fetch('/api/simulate-ptsd', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ severity: parseFloat(severity), duration_years: parseInt(duration) })
    });

    const data = await response.json();
    renderChart(data);
    updateUI(data);
}

function renderChart(data) {
    const ctx = document.getElementById('ptsdChart').getContext('2d');
    
    if (ptsdChartInstance) {
        ptsdChartInstance.destroy();
    }

    ptsdChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.months,
            datasets: [
                {
                    label: 'Clinical Efficacy (%)',
                    data: data.efficacy,
                    borderColor: '#0072ff',
                    backgroundColor: 'rgba(0, 114, 255, 0.1)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 3,
                    pointRadius: 0
                },
                {
                    label: 'Trauma Symptom Index',
                    data: data.trauma_index,
                    borderColor: '#ff0055',
                    backgroundColor: 'rgba(255, 0, 85, 0.1)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 3,
                    pointRadius: 0,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: 'rgba(255, 255, 255, 0.6)', font: { family: 'Outfit' } }
                }
            },
            scales: {
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: 'rgba(255, 255, 255, 0.4)' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: 'rgba(255, 255, 255, 0.4)' }
                }
            }
        }
    });
}

function updateUI(data) {
    const recList = document.getElementById('rec-list');
    if (recList) {
        recList.innerHTML = '';
        data.recommendations.forEach(rec => {
            const item = document.createElement('div');
            item.className = 'rec-item';
            item.innerHTML = `
                <div class="rec-label">${rec.lobe}</div>
                <div class="rec-val">${rec.freq}</div>
            `;
            recList.appendChild(item);
        });
    }

    // Update KPIs
    if (document.getElementById('kpi-efficacy')) {
        document.getElementById('kpi-efficacy').innerText = Math.round(data.efficacy[data.efficacy.length - 1]) + '%';
    }
    if (document.getElementById('kpi-trauma')) {
        document.getElementById('kpi-trauma').innerText = Math.round(data.trauma_index[data.trauma_index.length - 1]);
    }
}

function switchMainTab(tabId) {
    document.querySelectorAll('.tab-btn-main').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

    const activeBtn = Array.from(document.querySelectorAll('.tab-btn-main')).find(btn => btn.innerText.toLowerCase().includes(tabId));
    if (activeBtn) activeBtn.classList.add('active');

    const activePane = document.getElementById(`${tabId}-tab`);
    if (activePane) activePane.classList.add('active');

    if (tabId === 'fea') {
        runFEASimulation();
    } else if (tabId === 'trillium') {
        runTrilliumProtocols();
    } else if (tabId === 'repair') {
        runNeuralRepairProtocols();
    }
}

let feaRadarChartInstance = null;
let feaBarChartInstance = null;

async function runFEASimulation() {
    const response = await fetch('/api/simulate-fea', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    });

    const data = await response.json();
    renderFEACharts(data);
    updateFEAMetrics(data);
}

function renderFEACharts(data) {
    const radarCtx = document.getElementById('feaRadarChart').getContext('2d');
    const barCtx = document.getElementById('feaBarChart').getContext('2d');

    if (feaRadarChartInstance) feaRadarChartInstance.destroy();
    if (feaBarChartInstance) feaBarChartInstance.destroy();

    feaRadarChartInstance = new Chart(radarCtx, {
        type: 'radar',
        data: {
            labels: data.lobes,
            datasets: [{
                label: 'Stress Distribution',
                data: data.field_data.map(d => d.stress),
                backgroundColor: 'rgba(0, 114, 255, 0.2)',
                borderColor: '#0072ff',
                pointBackgroundColor: '#0072ff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: { color: 'rgba(255, 255, 255, 0.6)' }
                }
            }
        }
    });

    feaBarChartInstance = new Chart(barCtx, {
        type: 'bar',
        data: {
            labels: data.lobes,
            datasets: [{
                label: 'Conductivity (S/m)',
                data: data.field_data.map(d => d.conductivity),
                backgroundColor: '#00c6ff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: 'rgba(255, 255, 255, 0.4)' } },
                x: { grid: { display: false }, ticks: { color: 'rgba(255, 255, 255, 0.4)' } }
            }
        }
    });
}

function updateFEAMetrics(data) {
    document.getElementById('fea-peak').innerText = data.peak_field;
    document.getElementById('fea-anisotropy').innerText = data.anisotropy_ratio;
}

async function runTrilliumProtocols() {
    const response = await fetch('/api/trillium-protocols');
    const data = await response.json();
    renderTrilliumProtocols(data.protocols);
}

function renderTrilliumProtocols(protocols) {
    const container = document.getElementById('trillium-timeline');
    container.innerHTML = '';
    
    protocols.forEach(p => {
        const card = document.createElement('div');
        card.style.cssText = 'background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 15px; padding: 20px; display: flex; justify-content: space-between; align-items: center;';
        
        card.innerHTML = `
            <div>
                <div style="color: var(--secondary); font-weight: 600; font-size: 12px; text-transform: uppercase; margin-bottom: 5px;">Stage ${p.stage}</div>
                <div style="font-size: 18px; font-weight: 600;">${p.name}</div>
            </div>
            <div style="display: flex; gap: 30px; text-align: center;">
                <div>
                    <div style="font-size: 16px; font-weight: 600;">${p.voltage}</div>
                    <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase;">Voltage</div>
                </div>
                <div>
                    <div style="font-size: 16px; font-weight: 600;">${p.freq}</div>
                    <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase;">Frequency</div>
                </div>
                <div>
                    <div style="font-size: 16px; font-weight: 600;">${p.pulse_width}</div>
                    <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase;">Pulse Width</div>
                </div>
            </div>
            <div style="background: rgba(0,255,0,0.1); color: #00ff00; padding: 10px 20px; border-radius: 50px; font-weight: 600;">
                Opt Score: ${p.score}
            </div>
        `;
        container.appendChild(card);
    });
}

async function runNeuralRepairProtocols() {
    const response = await fetch('/api/neural-repair-protocols');
    const data = await response.json();
    renderNeuralRepairProtocols(data.protocols);
}

function renderNeuralRepairProtocols(protocols) {
    const container = document.getElementById('repair-timeline');
    container.innerHTML = '';
    
    protocols.forEach(p => {
        const card = document.createElement('div');
        card.style.cssText = 'background: rgba(0, 255, 0, 0.05); border: 1px solid rgba(0, 255, 0, 0.2); border-radius: 15px; padding: 20px; display: flex; justify-content: space-between; align-items: center;';
        
        card.innerHTML = `
            <div>
                <div style="color: #00ff00; font-weight: 600; font-size: 12px; text-transform: uppercase; margin-bottom: 5px;">Stage ${p.stage}</div>
                <div style="font-size: 18px; font-weight: 600;">${p.name}</div>
                <div style="font-size: 12px; color: var(--text-dim); margin-top: 5px;">Target: ${p.target}</div>
            </div>
            <div style="display: flex; gap: 30px; text-align: center;">
                <div>
                    <div style="font-size: 16px; font-weight: 600;">${p.freq}</div>
                    <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase;">Frequency</div>
                </div>
                <div>
                    <div style="font-size: 16px; font-weight: 600;">${p.pulse_width}</div>
                    <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase;">Pulse Width</div>
                </div>
            </div>
            <div style="display: flex; gap: 10px;">
                <div style="background: rgba(0,198,255,0.1); color: #00c6ff; padding: 10px 15px; border-radius: 50px; font-weight: 600; font-size: 12px;">
                    Plasticity: ${p.plasticity}%
                </div>
                <div style="background: rgba(0,255,0,0.1); color: #00ff00; padding: 10px 15px; border-radius: 50px; font-weight: 600; font-size: 12px;">
                    Confidence: ${p.confidence}%
                </div>
            </div>
        `;
        container.appendChild(card);
    });
}

// Initial Run
window.onload = () => {
    runSimulation();
};
