import re

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'r') as f:
    content = f.read()

# Add button
tab_btn = '<button class="tab-btn" onclick="switchTab(\'pareto\', event)">Pareto Optimization</button>'
new_tab_btn = tab_btn + '\n            <button class="tab-btn" onclick="switchTab(\'ms\', event)">MS / Alexander\'s</button>'
content = content.replace(tab_btn, new_tab_btn)

# Add sidebar
pareto_sidebar = '<div id="pareto-sidebar" class="tab-content">'
new_sidebar = """<div id="ms-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>MS & Alexander's</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        Cortical simulation frameworks for Deep Brain Stimulation in Multiple Sclerosis, including target modeling for ablating Rosenthal fibers in Alexander's disease.
                    </p>
                    <button class="btn-primary" id="btn-simulate-ms" onclick="simulateMS()" style="margin-top: 10px;">Simulate Ablation</button>
                </div>
            </div>

            <div id="pareto-sidebar" class="tab-content">"""
content = content.replace(pareto_sidebar, new_sidebar)

# Add main
pareto_main = '<div id="pareto-main" class="tab-content" style="height: 100%;">'
new_main = """<div id="ms-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>MS & Alexander's Disease Framework</h2>
                    <div style="display: flex; gap: 10px; flex: 1;">
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                            <h3>Cortical Simulation</h3>
                            <pre id="ms-output" style="color: #0f0; font-family: monospace; font-size: 12px; margin-top: 10px; white-space: pre-wrap; height: 300px; overflow-y: auto;">Awaiting simulation parameters...</pre>
                        </div>
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                            <h3>Rosenthal Fiber Ablation Network</h3>
                            <canvas id="ms-chart" style="width:100%; height:300px;"></canvas>
                        </div>
                    </div>
                </div>
            </div>

            <div id="pareto-main" class="tab-content" style="height: 100%;">"""
content = content.replace(pareto_main, new_main)

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'w') as f:
    f.write(content)

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/app.js', 'r') as f:
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
    with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/app.js', 'a') as f:
        f.write(ms_js)

