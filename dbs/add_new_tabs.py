import re

html_file = "/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html"
with open(html_file, "r") as f:
    html = f.read()

# Add to Nav Bar
nav_replacement = """        <nav class="tab-bar">
            <button class="tab-btn active" onclick="switchTab('planner', event)">Treatment Planner</button>
            <button class="tab-btn" onclick="switchTab('conductivity', event)">Conductivity Analysis</button>
            <button class="tab-btn" onclick="switchTab('dementia', event)">Dementia Opt. Protocol</button>
            <button class="tab-btn" onclick="switchTab('fea', event)">Cortical FEA Manifolds</button>
        </nav>"""
html = re.sub(r'<nav class="tab-bar">.*?</nav>', nav_replacement, html, flags=re.DOTALL)


# Add Sidebar content for new tabs
sidebar_addition = """
            <div id="dementia-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>Protocol Config</h2>
                    <div class="control-group">
                        <label>Dementia Decline Rate (\beta)</label>
                        <input type="range" id="dementia-decline-range" min="0.01" max="0.1" step="0.01" value="0.05">
                    </div>
                    <div class="control-group">
                        <label>Prompt / Modality</label>
                        <select id="dementia-prompt" style="background: rgba(0,0,0,0.5); color:white; border:none; padding:5px; border-radius:4px; width:100%;">
                            <option value="baseline">Baseline Control</option>
                            <option value="plasticity">Hebbian Plasticity Focus</option>
                            <option value="aggressive">Aggressive Pruning</option>
                        </select>
                    </div>
                    <button class="btn-primary" id="btn-generate-dementia" onclick="updateDementiaChart()" style="margin-top: 10px;">Run Optimization</button>
                </div>
            </div>

            <div id="fea-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>Cortical FEA Modifiers</h2>
                    <div class="control-group">
                        <label>Density Nodes Allocation</label>
                        <input type="range" id="fea-nodes" min="100" max="5000" step="100" value="1000">
                    </div>
                    <div class="control-group">
                        <label>Manifold Relaxation</label>
                        <input type="range" id="fea-relax" min="0" max="1" step="0.1" value="0.5">
                    </div>
                    <button class="btn-primary" id="btn-fea-sim" onclick="initLargerFEA()" style="margin-top: 10px;">Regenerate Mesh</button>
                </div>
            </div>
"""
# Insert before </aside> for left sidebar
html = html.replace('        </aside>\n\n        <main class="main-view">', sidebar_addition + '        </aside>\n\n        <main class="main-view">')


# Add Main content for new tabs
main_addition = """
            <div id="dementia-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Optimal Protocol & Dementia Progress Visualization (60 Months)</h2>
                    <div style="flex-grow: 1; position: relative;">
                        <!-- Using Chart.js script injection dynamically via JS -->
                        <canvas id="dementia-chart" style="width:100%; height:100%;"></canvas>
                    </div>
                </div>
            </div>

            <div id="fea-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Finite Element Analysis - Cortical Surface Manifolds</h2>
                    <div id="fea-large-container" style="flex-grow: 1; border-radius: 8px; overflow: hidden; position: relative;">
                        <!-- Large Three.js canvas goes here -->
                    </div>
                </div>
            </div>
"""
# Insert before </main>
html = html.replace('        </main>\n\n        <aside class="sidebar-right">', main_addition + '        </main>\n\n        <aside class="sidebar-right">')

# Add Chart.js to Scripts
script_addition = """    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>"""
html = html.replace('<script src="{{ url_for(\'static\', filename=\'js/main.js\') }}"></script>', script_addition)


with open(html_file, "w") as f:
    f.write(html)
print("Updated index.html")
