with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'r') as f:
    js_content = f.read()

huntington_js = """
function simulateHuntington() {
    const out = document.getElementById('huntington-output');
    if(!out) return;
    out.innerText = "Initializing Cortical Simulation for Huntington's Disease...\\n";
    out.innerText += "Applying Statistical Parametric Optimization Circuitry...\\n";
    
    setTimeout(() => {
        out.innerText += "Evaluating Cortical Repair Thresholds...\\n";
        out.innerText += "Generating Electrical Specifications...\\n";
    }, 1000);
    
    setTimeout(() => {
        out.innerText += "Simulation Complete. Plotting Interventional Repair Matrix...\\n";
        renderHuntingtonChart();
    }, 2000);
}

function renderHuntingtonChart() {
    const ctx = document.getElementById('huntington-chart');
    if(!ctx) return;
    
    if (window.huntingtonChartInstance) {
        window.huntingtonChartInstance.destroy();
    }
    
    window.huntingtonChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Weeks 0', 'W 4', 'W 8', 'W 12', 'W 16', 'W 20', 'W 24'],
            datasets: [{
                label: 'Motor Function Degeneration',
                data: [100, 95, 80, 50, 25, 10, 5],
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                fill: true,
                tension: 0.4
            }, {
                label: 'Interventional Repair Signal',
                data: [0, 10, 35, 65, 85, 95, 100],
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
                    text: 'Statistical Parametric Optimization Circuitry (Huntington\\\'s)'
                }
            }
        }
    });
}
"""
js_content += huntington_js

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'w') as f:
    f.write(js_content)
