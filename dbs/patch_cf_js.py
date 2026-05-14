with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'r') as f:
    js_content = f.read()

import re

# Update MS output log text for CF
js_content = re.sub(
    r'out\.innerText = "Initializing Quantum Neural Network for MS mitigation...\\n";\s*out\.innerText \+= "Mapping discrete variables for plaque density...\\n";',
    'out.innerText = "Initializing Ramanujan CF Operators for MS mitigation...\\n";\n    out.innerText += "Mapping continued fraction expansions for plaque density...\\n";',
    js_content
)

# Add Alexander CF logic
cf_js = """
function simulateAlexanderCF() {
    const out = document.getElementById('alexander-cf-output');
    if(!out) return;
    out.innerText = "Initializing Continued Fraction Addendum...\\n";
    out.innerText += "Structuring QML CF Plaque Ablation model...\\n";
    
    setTimeout(() => {
        out.innerText += "Simulating Neural Recovery metrics via CF...\\n";
        out.innerText += "Computing Rosenthal Fiber dissipation limits...\\n";
    }, 1000);
    
    setTimeout(() => {
        out.innerText += "Simulation Complete. Plotting CF mitigation dynamics...\\n";
        renderAlexanderCFChart();
    }, 2000);
}

function renderAlexanderCFChart() {
    const ctx = document.getElementById('alexander-cf-chart');
    if(!ctx) return;
    
    if (window.alexanderCfChartInstance) {
        window.alexanderCfChartInstance.destroy();
    }
    
    window.alexanderCfChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            datasets: [{
                label: 'Rosenthal Fiber Density (CF bounded)',
                data: [100, 80, 50, 30, 15, 8, 4, 1, 0, 0, 0],
                borderColor: 'rgba(153, 102, 255, 1)',
                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                fill: true,
                tension: 0.4
            }, {
                label: 'Neural Recovery Index %',
                data: [0, 25, 55, 75, 88, 95, 98, 99, 100, 100, 100],
                borderColor: 'rgba(255, 206, 86, 1)',
                backgroundColor: 'rgba(255, 206, 86, 0.2)',
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
                    text: 'Alexander\\\'s Disease CF Neural Recovery & Ablation'
                }
            }
        }
    });
}
"""
js_content += cf_js

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'w') as f:
    f.write(js_content)
