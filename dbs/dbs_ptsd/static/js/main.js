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

// Initial Run
window.onload = () => {
    runSimulation();
};
