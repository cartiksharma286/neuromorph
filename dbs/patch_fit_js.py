with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'r') as f:
    js_content = f.read()

import re

old_chart_js = """function renderMarketValuationChart() {
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
}"""

new_chart_js = """function renderMarketValuationChart() {
    const ctx = document.getElementById('market-valuation-chart');
    if(!ctx) return;
    if (window.marketValChartInstance) window.marketValChartInstance.destroy();
    
    // Fit exponential curve for Net Market Valuation
    // V(t) = a * e^(b*t)
    const valuationData = [2.1, 3.5, 5.2, 7.6, 9.8, 12.4];
    const trendFit = valuationData.map((v, i) => 2.0 * Math.exp(0.36 * i));

    window.marketValChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['2026', '2028', '2030', '2032', '2034', '2036'],
            datasets: [{
                type: 'bar',
                label: 'Annual Projection Costs ($M)',
                data: [250, 310, 420, 580, 750, 960],
                backgroundColor: 'rgba(255, 159, 64, 0.6)',
                yAxisID: 'y1'
            }, {
                type: 'line',
                label: 'Net Market Valuation Base ($B)',
                data: valuationData,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderWidth: 3,
                tension: 0.3,
                yAxisID: 'y'
            }, {
                type: 'line',
                label: 'Market Valuation Exponential Fit ($B)',
                data: trendFit,
                borderColor: 'rgba(255, 99, 132, 1)',
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
                        text: 'Projection Costs ($M)'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            }
        }
    });
}"""

js_content = js_content.replace(old_chart_js, new_chart_js)

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'w') as f:
    f.write(js_content)
