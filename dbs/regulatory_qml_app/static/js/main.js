let qmlChartInstance = null;
let controlChartInstance = null;
let monetizationChartInstance = null;

Chart.defaults.color = 'rgba(255, 255, 255, 0.6)';
Chart.defaults.font.family = 'Space Mono';

function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

    const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(btn => btn.getAttribute('onclick').includes(tabId));
    if (activeBtn) activeBtn.classList.add('active');

    const activePane = document.getElementById(`${tabId}-tab`);
    if (activePane) activePane.classList.add('active');

    if (tabId === 'qml') fetchQMLData();
    if (tabId === 'control') fetchControlData();
    if (tabId === 'monetization') fetchMonetizationData();
}

async function fetchQMLData() {
    const response = await fetch('/api/qml-fraud-detection');
    const data = await response.json();
    
    // Update metrics
    document.getElementById('max-entanglement').innerText = Math.max(...data.entanglement).toFixed(2);
    
    const threatBox = document.getElementById('threat-status');
    if (data.threat_level === "HIGH") {
        threatBox.className = 'alert-box';
        threatBox.innerHTML = '<h3 style="margin:0; color:var(--accent);">THREAT LEVEL: HIGH (QUANTUM ANOMALY DETECTED)</h3>';
    } else {
        threatBox.className = 'alert-box safe';
        threatBox.innerHTML = '<h3 style="margin:0; color:var(--success);">THREAT LEVEL: LOW (SYSTEM NOMINAL)</h3>';
    }

    const ctx = document.getElementById('qmlRadarChart').getContext('2d');
    if (qmlChartInstance) qmlChartInstance.destroy();

    qmlChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: data.labels,
            datasets: [{
                label: 'Anomaly Probability',
                data: data.probabilities,
                backgroundColor: 'rgba(0, 210, 255, 0.2)',
                borderColor: '#00d2ff',
                pointBackgroundColor: '#ff007a'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { display: false }
                }
            }
        }
    });
}

async function fetchControlData() {
    const response = await fetch('/api/optimal-control');
    const data = await response.json();

    const ctx = document.getElementById('controlChart').getContext('2d');
    if (controlChartInstance) controlChartInstance.destroy();

    controlChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.time,
            datasets: [
                {
                    label: 'Fraud Volume Evolution',
                    data: data.fraud_volume,
                    borderColor: '#ff007a',
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 0
                },
                {
                    label: 'Regulatory Intervention Cost',
                    data: data.intervention_cost,
                    borderColor: '#00d2ff',
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                annotation: {
                    annotations: {
                        line1: {
                            type: 'line',
                            xMin: data.optimal_point,
                            xMax: data.optimal_point,
                            borderColor: '#00ff88',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            label: {
                                display: true,
                                content: 'Optimal Intervention Point (u*)',
                                position: 'start'
                            }
                        }
                    }
                }
            },
            scales: {
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                x: { grid: { display: false } }
            }
        }
    });
}

async function fetchMonetizationData() {
    const response = await fetch('/api/cost-monetization');
    const data = await response.json();

    document.getElementById('total-recovered').innerText = '$' + data.total_recovered.toLocaleString();
    const avgRoi = data.roi.reduce((a, b) => a + b, 0) / data.roi.length;
    document.getElementById('avg-roi').innerText = avgRoi.toFixed(1) + '%';

    const ctx = document.getElementById('monetizationChart').getContext('2d');
    if (monetizationChartInstance) monetizationChartInstance.destroy();

    monetizationChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.months,
            datasets: [
                {
                    label: 'Funds Recovered ($)',
                    data: data.funds_recovered,
                    backgroundColor: '#00ff88',
                    borderRadius: 4
                },
                {
                    label: 'Compliance Cost ($)',
                    data: data.compliance_cost,
                    backgroundColor: 'rgba(255, 255, 255, 0.2)',
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                x: { grid: { display: false } }
            }
        }
    });
}

// Initial Load
window.onload = () => {
    fetchQMLData();
    // Simulate real-time monitoring
    setInterval(() => {
        if (document.getElementById('qml-tab').classList.contains('active')) {
            fetchQMLData();
        }
    }, 5000);
};
