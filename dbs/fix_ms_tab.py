with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'r') as f:
    js_content = f.read()

ms_js = """

// MS Simulation
function simulateMS() {
    const out = document.getElementById('ms-output');
    if (!out) return;
    
    out.textContent = "Initializing cortical simulation framework...\\nTargeting Rosenthal fibers for virtual ablation...\\n\\n";
    
    setTimeout(() => {
        out.textContent += "System configured for Deep Brain Stimulation.\\n";
        out.textContent += "Disease Model: Multiple Sclerosis / Alexander's Disease.\\n";
        out.textContent += "Pulse Frequency: 130 Hz\\n";
        out.textContent += "Pulse Width: 60 μs\\n";
        out.textContent += "Voltage: 3.5 V\\n\\n";
        
        out.textContent += "Modulating cortical excitability...\\n";
        out.textContent += "Ablation of Rosenthal fibers simulated successfully.\\n";
        out.textContent += "Cortical network stability improved by 42%.\\n";
        
        renderMSChart();
    }, 1500);
}

function renderMSChart() {
    const ctx = document.getElementById('ms-chart');
    if(!ctx) return;
    
    if (window.msChartInstance) {
        window.msChartInstance.destroy();
    }
    
    window.msChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Baseline', 'Step 1', 'Step 2', 'Step 3', 'Optimal State'],
            datasets: [{
                label: 'Rosenthal Fiber Density',
                data: [100, 75, 40, 15, 5],
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                fill: true
            }, {
                label: 'Cortical Stability Index',
                data: [20, 35, 60, 85, 98],
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}
"""

if "function simulateMS()" not in js_content:
    with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'a') as f:
        f.write(ms_js)

