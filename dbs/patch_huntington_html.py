import re

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'r') as f:
    html = f.read()

# Update Nav
nav_patch = r"""<button class="tab-btn" onclick="switchTab('alexander-cf', event)">Alexander CF Addendum</button>
            <button class="tab-btn" onclick="switchTab('huntington', event)">Huntington's Disease</button>"""
html = re.sub(r'<button class="tab-btn" onclick="switchTab\(\'alexander-cf\', event\)">Alexander CF Addendum</button>', nav_patch, html)

# Update Sidebar
alexander_cf_sidebar_re = r'<div id="alexander-cf-sidebar" class="tab-content">.*?</div>\s*</div>'
alexander_cf_match = re.search(alexander_cf_sidebar_re, html, flags=re.DOTALL).group(0)

huntington_sidebar = """
            <div id="huntington-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>Huntington's Disease</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        Cortical simulations with electrical target specs utilizing statistical parametric optimization circuitry for interventional repair.
                    </p>
                    <button class="btn-primary" id="btn-simulate-huntington" onclick="simulateHuntington()" style="margin-top: 10px;">Run Interventional Simulation</button>
                </div>
            </div>"""

html = html.replace(alexander_cf_match, alexander_cf_match + huntington_sidebar)

# Update Main Panel
alexander_cf_main_re = r'<div id="alexander-cf-main" class="tab-content" style="height: 100%;">.*?</div>\s*</div>\s*</div>\s*</div>'
alexander_cf_main_match = re.search(alexander_cf_main_re, html, flags=re.DOTALL).group(0)

huntington_main = """
            <div id="huntington-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Huntington's Disease Cortical Repair Simulation</h2>
                    <div style="display: flex; gap: 10px; flex: 1;">
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                            <h3>Statistical Parametric Optimization Circuitry</h3>
                            <ul style="font-size: 12px; color: #add8e6; margin-top: 10px; list-style-type: square; margin-left: 20px;">
                                <li><strong>Target Structure:</strong> Motor Cortex & Striatum</li>
                                <li><strong>Circuitry:</strong> Statistical Parametric Optimization</li>
                                <li><strong>Intervention:</strong> Cortical Repair Modulation</li>
                            </ul>
                            <h3>Optimization Log</h3>
                            <pre id="huntington-output" style="color: #0f0; font-family: monospace; font-size: 12px; margin-top: 10px; white-space: pre-wrap; height: 150px; overflow-y: auto;">Awaiting simulation...</pre>
                        </div>
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px; height: 350px;">
                            <canvas id="huntington-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>"""

html = html.replace(alexander_cf_main_match, alexander_cf_main_match + huntington_main)

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'w') as f:
    f.write(html)
