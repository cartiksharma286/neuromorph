with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'r') as f:
    js_content = f.read()

new_js = """
// Cortical Simulation
function simulateCortical() {
    const out = document.getElementById('cortical-sim-output');
    if(!out) return;
    out.innerText = "Initializing Cortical Network Engine...\\n";
    out.innerText += "Mapping M1/S1 Projection Costs...\\n";
    setTimeout(() => {
        out.innerText += "Applying Neurodynamic Equilibration...\\n";
        renderCorticalChart();
        out.innerText += "Cortical Projection Complete.\\n";
    }, 1500);
}

function renderCorticalChart() {
    const ctx = document.getElementById('cortical-sim-chart');
    if(!ctx) return;
    if (window.corticalChartInstance) window.corticalChartInstance.destroy();
    window.corticalChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['0ms', '10ms', '20ms', '30ms', '40ms', '50ms'],
            datasets: [{
                label: 'Projection Energy Cost (mWh)',
                data: [5, 15, 45, 60, 42, 25],
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                fill: true,
                tension: 0.4
            }, {
                label: 'Layer V Firing Rate (Hz)',
                data: [12, 18, 30, 25, 10, 8],
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

// Net Market Valuation
function simulateMarketValuation() {
    const out = document.getElementById('market-valuation-output');
    if(!out) return;
    out.innerText = "Accessing 10-Year Economic Outlook Engine (2026-2036)...\\n";
    out.innerText += "Calculating Net Present Value (NPV) & Discount Rates...\\n";
    setTimeout(() => {
        out.innerText += "Extrapolating Neuromodulation Trajectory...\\n";
        renderMarketValuationChart();
        out.innerText += "Valuation Projection Complete.\\n";
    }, 1500);
}

function renderMarketValuationChart() {
    const ctx = document.getElementById('market-valuation-chart');
    if(!ctx) return;
    if (window.marketValChartInstance) window.marketValChartInstance.destroy();
    window.marketValChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['2026', '2028', '2030', '2032', '2034', '2036'],
            datasets: [{
                type: 'bar',
                label: 'Annual Projection Costs ($M)',
                data: [250, 310, 420, 580, 750, 960],
                backgroundColor: 'rgba(255, 159, 64, 0.6)'
            }, {
                type: 'line',
                label: 'Net Market Valuation ($B)',
                data: [2.1, 3.5, 5.2, 7.6, 9.8, 12.4],
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 3,
                tension: 0.3
            }]
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Projection Costs ($M) / Market ($B)'
                    }
                }
            }
        }
    });
}
"""

js_content += new_js

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'w') as f:
    f.write(js_content)
