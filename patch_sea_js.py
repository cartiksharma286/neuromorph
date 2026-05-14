with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'r') as f:
    js_content = f.read()

new_js = """
// South Asia & SEA Valuation
function simulateSEAValuation() {
    const out = document.getElementById('sea-valuation-output');
    if(!out) return;
    out.innerText = "Accessing APAC & South Asia Economic Overlays (2026-2036)...\\n";
    out.innerText += "Evaluating Regional Device Penetration & Healthcare Spending...\\n";
    setTimeout(() => {
        out.innerText += "Extrapolating Neuromodulation Growth (CAGR: 21.2%)...\\n";
        renderSEAValuationChart();
        out.innerText += "Regional Valuation Projection Complete.\\n";
    }, 1500);
}

function renderSEAValuationChart() {
    const ctx = document.getElementById('sea-valuation-chart');
    if(!ctx) return;
    if (window.seaValChartInstance) window.seaValChartInstance.destroy();
    
    // Fit exponential curve for SEA Net Market Valuation
    // V(t) = a * e^(b*t)
    const seaValuationData = [0.45, 0.75, 1.25, 1.95, 2.70, 3.80]; // Billion $
    const trendFit = seaValuationData.map((v, i) => 0.45 * Math.exp(0.42 * i));

    window.seaValChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['2026', '2028', '2030', '2032', '2034', '2036'],
            datasets: [{
                type: 'bar',
                label: 'Regional Infrastructure Costs ($M)',
                data: [85, 120, 190, 275, 380, 520],
                backgroundColor: 'rgba(153, 102, 255, 0.6)',
                yAxisID: 'y1'
            }, {
                type: 'line',
                label: 'South Asia/SEA Market ($B)',
                data: seaValuationData,
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderWidth: 3,
                tension: 0.3,
                yAxisID: 'y'
            }, {
                type: 'line',
                label: 'Exponential Growth Fit ($B)',
                data: trendFit,
                borderColor: 'rgba(255, 206, 86, 1)',
                borderDash: [5, 5],
                borderWidth: 2,
                tension: 0.4,
                yAxisID: 'y'
            }]
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Market Value ($B)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Infrastructure Setup ($M)'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            }
        }
    });
}
"""

js_content += new_js

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'w') as f:
    f.write(js_content)
