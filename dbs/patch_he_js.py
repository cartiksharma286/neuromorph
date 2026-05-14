import re

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'r') as f:
    js_content = f.read()

he_js = """
// Health Economics: MS
function simulateHEMS() {
    const out = document.getElementById('he-ms-output');
    if(!out) return;
    out.innerText = "Calculating Cost-Utility for MS Intervention...\\n";
    out.innerText += "Mapping Quality-Adjusted Life Years (QALYs)...\\n";
    setTimeout(() => {
        out.innerText += "Applying Evidence-Based Outcomes Framework...\\n";
        renderHEMSChart();
        out.innerText += "Markov Decision Process Complete.\\n";
    }, 1500);
}

function renderHEMSChart() {
    const ctx = document.getElementById('he-ms-chart');
    if(!ctx) return;
    if (window.heMsChartInstance) window.heMsChartInstance.destroy();
    window.heMsChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Standard Care', 'DBS QML Mitigated'],
            datasets: [{
                label: 'Cumulative Costs ($)',
                data: [450000, 320000],
                backgroundColor: 'rgba(255, 99, 132, 0.5)'
            }, {
                label: 'Lifetime QALYs',
                data: [12.5, 15.0],
                backgroundColor: 'rgba(54, 162, 235, 0.5)'
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

// Health Economics: Huntington's
function simulateHEHuntington() {
    const out = document.getElementById('he-huntington-output');
    if(!out) return;
    out.innerText = "Calculating Cost-Utility for Huntington's Intervention...\\n";
    out.innerText += "Evaluating Caregiver Burden Metrics...\\n";
    setTimeout(() => {
        out.innerText += "Applying Incremental Cost-Effectiveness Ratio (ICER)...\\n";
        renderHEHuntingtonChart();
        out.innerText += "SPOC Economic Valuation Complete.\\n";
    }, 1500);
}

function renderHEHuntingtonChart() {
    const ctx = document.getElementById('he-huntington-chart');
    if(!ctx) return;
    if (window.heHuntChartInstance) window.heHuntChartInstance.destroy();
    window.heHuntChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Standard Care', 'SPOC DBS Array'],
            datasets: [{
                label: 'Cumulative Costs ($)',
                data: [680000, 410000],
                backgroundColor: 'rgba(255, 159, 64, 0.5)'
            }, {
                label: 'Lifetime QALYs',
                data: [9.2, 12.3],
                backgroundColor: 'rgba(75, 192, 192, 0.5)'
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

// Health Economics: Alexander's
function simulateHEAlexander() {
    const out = document.getElementById('he-alexander-output');
    if(!out) return;
    out.innerText = "Analyzing Value-Based Care for Rosenthal Ablation...\\n";
    out.innerText += "Projecting Hospitalization Reductions...\\n";
    setTimeout(() => {
        out.innerText += "Calculating QALYs and Resource Utilization...\\n";
        renderHEAlexanderChart();
        out.innerText += "Economic Validation Complete.\\n";
    }, 1500);
}

function renderHEAlexanderChart() {
    const ctx = document.getElementById('he-alexander-chart');
    if(!ctx) return;
    if (window.heAlexChartInstance) window.heAlexChartInstance.destroy();
    window.heAlexChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Standard Care', 'CF Adaptive Ablation'],
            datasets: [{
                label: 'Cumulative Costs ($)',
                data: [850000, 520000],
                backgroundColor: 'rgba(153, 102, 255, 0.5)'
            }, {
                label: 'Lifetime QALYs',
                data: [8.5, 12.5],
                backgroundColor: 'rgba(255, 206, 86, 0.5)'
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}
"""

js_content += he_js

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'w') as f:
    f.write(js_content)
