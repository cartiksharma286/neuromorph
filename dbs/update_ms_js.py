with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'r') as f:
    js_content = f.read()

import re

# Update the MS Chart function if it exists to be more accurate to the request
if "function renderMSChart()" in js_content:
    js_content = re.sub(
        r"function renderMSChart\(\) \{[\s\S]*?\}\s*?(?=\n\n|\Z)",
        r"""function renderMSChart() {
    const ctx = document.getElementById('ms-chart');
    if(!ctx) return;
    
    if (window.msChartInstance) {
        window.msChartInstance.destroy();
    }
    
    window.msChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12 (Months)'],
            datasets: [{
                label: 'MS Plaque Density (Quantum Optimized Mitigation)',
                data: [100, 85, 60, 42, 30, 20, 14, 9, 6, 4, 3, 2, 1],
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                fill: true,
                tension: 0.4
            }, {
                label: 'Neural Recovery %',
                data: [0, 15, 30, 48, 62, 74, 82, 88, 92, 95, 97, 98, 99],
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'QML-DBS Accelerated MS Neural Recovery'
                }
            }
        }
    });
}""",
        js_content
    )

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'w') as f:
    f.write(js_content)
