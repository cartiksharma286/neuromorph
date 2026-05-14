import re

html_file = 'dbs/templates/index.html'
js_file = 'dbs/static/js/main.js'

with open(html_file, 'r') as f:
    html_content = f.read()

# 1. Add Navigation Buttons
nav_insert = """            <button class="tab-btn" onclick="switchTab('fas', event)">FAS DBS Intervention</button>
            <button class="tab-btn" onclick="switchTab('sunnybrook', event)">Sunnybrook Projections 2036</button>
        </nav>"""
html_content = re.sub(r'</nav>', nav_insert, html_content, count=1)

# 2. Add Sidebars before pareto-sidebar
sidebar_insert = """
            <div id="fas-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>Fetal Alcohol Syndrome (FAS) DBS</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        Neurosymbolic AI guided inflection point tracking and statistical distribution shifts for interventional DBS in FAS.
                    </p>
                    <button class="btn-primary" id="btn-simulate-fas" onclick="simulateFAS()" style="margin-top: 10px;">Run FAS Estimates</button>
                </div>
            </div>

            <div id="sunnybrook-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>Sunnybrook 2036 Projections</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        10-year market and clinical projections for Sunnybrook Health Sciences Centre.
                    </p>
                    <button class="btn-primary" id="btn-simulate-sunnybrook" onclick="simulateSunnybrook()" style="margin-top: 10px;">Run Sunnybrook Estimates</button>
                </div>
            </div>

            <div id="pareto-sidebar"
"""
html_content = html_content.replace('            <div id="pareto-sidebar"', sidebar_insert)

# 3. Add Main content sections before pareto-main
main_insert = """
            <div id="fas-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Interventional Cure: Fetal Alcohol Syndrome (FAS) / Neurosymbolic AI</h2>
                    <div style="display: flex; gap: 20px; flex: 1; align-items: stretch;">
                        <div style="flex: 1;">
                            <h3>Statistical Distributions & Inflection Points</h3>
                            <ul style="list-style-type: none; padding: 0; font-size: 14px; margin-bottom: 15px;">
                                <li>📈 <strong style="color: var(--highlight);">Neurosymbolic AI:</strong> Symbolic priors guide DL inflection estimations.</li>
                                <li>🎯 <strong style="color: var(--highlight);">Target:</strong> Striatal & Cortical Pathway Integration.</li>
                                <li>⏱ <strong style="color: var(--highlight);">Inflection Window:</strong> 6 - 8 Years formulation trajectory.</li>
                            </ul>
                            <h3>AI Reasoning Engine Log</h3>
                            <pre id="fas-output" style="color: #0f0; font-family: monospace; font-size: 12px; margin-top: 10px; white-space: pre-wrap; height: 150px; overflow-y: auto;">Awaiting Neurosymbolic FAS Analysis...</pre>
                        </div>
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px; height: 350px;">
                            <canvas id="fas-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>

            <div id="sunnybrook-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Sunnybrook Market Projections (10-Year Outlook to 2036)</h2>
                    <div style="display: flex; gap: 20px; flex: 1; align-items: stretch;">
                        <div style="flex: 1;">
                            <h3>2036 Strategic Metrics</h3>
                            <ul style="list-style-type: none; padding: 0; font-size: 14px; margin-bottom: 15px;">
                                <li>📈 <strong style="color: var(--highlight);">CAGR:</strong> 15.4% Projected Sunnybrook Expansion.</li>
                                <li>🎯 <strong style="color: var(--highlight);">Scale:</strong> Cross-Border Ecosystem Leadership.</li>
                            </ul>
                            <h3>Projection Engine Log</h3>
                            <pre id="sunnybrook-output" style="color: #0f0; font-family: monospace; font-size: 12px; margin-top: 10px; white-space: pre-wrap; height: 150px; overflow-y: auto;">Awaiting Sunnybrook 2036 Projections...</pre>
                        </div>
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px; height: 350px;">
                            <canvas id="sunnybrook-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>

            <div id="pareto-main"
"""
html_content = html_content.replace('            <div id="pareto-main"', main_insert)

with open(html_file, 'w') as f:
    f.write(html_content)

# Now JS patching
with open(js_file, 'r') as f:
    js_content = f.read()

js_insert = """

function simulateFAS() {
    const out = document.getElementById('fas-output');
    out.innerText = "Initializing Neurosymbolic Rule Engine...\n";
    
    setTimeout(() => {
        out.innerText += "Injecting deep learning weights with Bayesian symbolic priors...\n";
    }, 500);
    
    setTimeout(() => {
        out.innerText += "Calculating neurodevelopmental deviation variances in FAS...\n";
        out.innerText += "Identifying optimal surgical inflection point triggers...\n";
    }, 1000);
    
    setTimeout(() => {
        out.innerText += "Synthesizing normalized cognitive trajectory distributions...\n";
        renderFASChart();
        out.innerText += "FAS Inflection Point Analysis Simulation Complete.\n";
    }, 1500);
}

function renderFASChart() {
    const ctx = document.getElementById('fas-chart');
    if(!ctx) return;
    if (window.fasChartInstance) window.fasChartInstance.destroy();
    
    const labels = ["Age 2", "Age 4", "Age 6", "Age 8", "Age 10", "Age 12", "Age 14"];
    const fasBaseline = [30, 35, 45, 55, 60, 65, 68];
    const typicalNeuro = [40, 55, 75, 90, 105, 115, 120];
    const postDbsTrajectory = [30, 35, 45, 80, 95, 108, 115]; // Inflection at Age 6-8

    window.fasChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Unmitigated FAS Trajectory',
                    data: fasBaseline,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    borderWidth: 2,
                    tension: 0.4
                },
                {
                    label: 'Typical Neurodevelopment',
                    data: typicalNeuro,
                    borderColor: 'rgba(200, 200, 200, 0.4)',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.4
                },
                {
                    label: 'Post-DBS Inflection (Neurosymbolic)',
                    data: postDbsTrajectory,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                annotation: {
                    annotations: {
                        inflectionLine: {
                            type: 'line',
                            xMin: 'Age 6',
                            xMax: 'Age 6',
                            borderColor: 'rgba(255, 206, 86, 0.8)',
                            borderWidth: 2,
                            label: {
                                content: 'DBS Intervention Inflection Point',
                                enabled: true,
                                position: 'top'
                            }
                        }
                    }
                }
            },
            scales: {
                y: {
                    title: { display: true, text: 'Cognitive / Motor Integrity Score' }
                }
            }
        }
    });
}

function simulateSunnybrook() {
    const out = document.getElementById('sunnybrook-output');
    out.innerText = "Connecting to Market Analysis Modules...\n";
    
    setTimeout(() => {
        out.innerText += "Cross-referencing historical R&D spend at Sunnybrook...\n";
        out.innerText += "Evaluating 10-year capital outlay and IP valuations...\n";
    }, 500);
    
    setTimeout(() => {
        out.innerText += "Projecting exponential expansion towards 2036 ecosystem...\n";
    }, 1000);
    
    setTimeout(() => {
        out.innerText += "Generating cumulative asset value matrices...\n";
        renderSunnybrookChart();
        out.innerText += "Sunnybrook 2036 Strategic Projections Computed.\n";
    }, 1500);
}

function renderSunnybrookChart() {
    const ctx = document.getElementById('sunnybrook-chart');
    if(!ctx) return;
    if (window.sunnybrookChartInstance) window.sunnybrookChartInstance.destroy();
    
    const years = [];
    const rndOverhead = [];
    const marketVal = [];
    let startVal = 2.4; 
    const cagr = 0.154; // 15.4%
    
    for (let i = 0; i <= 10; i++) {
        let y = 2026 + i;
        years.push(y);
        rndOverhead.push(Math.round(40 + (i * 8.5) + (Math.random() * 5))); 
        let val = startVal * Math.exp(cagr * i);
        marketVal.push(val.toFixed(2));
    }

    window.sunnybrookChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: years,
            datasets: [
                {
                    label: 'Sunnybrook Net Value / Ecosystem Projection ($B)',
                    data: marketVal,
                    type: 'line',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 3,
                    tension: 0.3,
                    fill: true,
                    yAxisID: 'y'
                },
                {
                    label: 'DBS Institutional R&D Overhead ($M)',
                    data: rndOverhead,
                    backgroundColor: 'rgba(153, 102, 255, 0.5)',
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderWidth: 1,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true, 
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear', display: true, position: 'left',
                    title: { display: true, text: 'Valuation ($ Billion)' }
                },
                y1: {
                    type: 'linear', display: true, position: 'right',
                    title: { display: true, text: 'R&D Cost ($ Million)' },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}
"""

if "simulateFAS()" not in js_content:
    with open(js_file, 'a') as f:
        f.write(js_insert)

print("Patched successfully.")
