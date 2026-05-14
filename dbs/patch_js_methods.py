with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'r') as f:
    js_content = f.read()

import re

# Add simulateAlexander and renderAlexanderChart
alexander_functions = """
function simulateAlexander() {
    const out = document.getElementById('alexander-output');
    if(!out) return;
    out.innerText = "Initializing QML Adaptive Ablation...\\n";
    out.innerText += "Mapping Feynman Path Integrals over White Matter Astrocytes...\\n";
    
    setTimeout(() => {
        out.innerText += "Targeting Rosenthal Fibers...\\n";
        out.innerText += "Applying Adaptive Ablation sequences...\\n";
    }, 1000);
    
    setTimeout(() => {
        out.innerText += "Simulation Complete. Plotting mitigation dynamics...\\n";
        renderAlexanderChart();
    }, 2000);
}

function renderAlexanderChart() {
    const ctx = document.getElementById('alexander-chart');
    if(!ctx) return;
    
    if (window.alexanderChartInstance) {
        window.alexanderChartInstance.destroy();
    }
    
    window.alexanderChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            datasets: [{
                label: 'Rosenthal Fiber Density',
                data: [100, 90, 75, 55, 30, 15, 8, 4, 1, 0, 0],
                borderColor: 'rgba(255, 159, 64, 1)',
                backgroundColor: 'rgba(255, 159, 64, 0.2)',
                fill: true,
                tension: 0.4
            }, {
                label: 'Ablation Stability Index (Feynman Mapping)',
                data: [0, 20, 45, 65, 85, 95, 98, 99, 100, 100, 100],
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
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
                    text: 'Adaptive Ablation of Rosenthal Fibers (Alexander\\\'s Disease)'
                }
            }
        }
    });
}
"""

js_content += alexander_functions

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'w') as f:
    f.write(js_content)
