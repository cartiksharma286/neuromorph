import re

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'r') as f:
    html = f.read()

# Update Nav
nav_patch = r"""<button class="tab-btn" onclick="switchTab('ms', event)">MS Ablation</button>
            <button class="tab-btn" onclick="switchTab('alexander', event)">Alexander's Disease</button>
            <button class="tab-btn" onclick="switchTab('alexander-cf', event)">Alexander CF Addendum</button>"""
html = re.sub(r'<button class="tab-btn" onclick="switchTab\(\'ms\', event\)">MS Ablation</button>\s*<button class="tab-btn" onclick="switchTab\(\'alexander\', event\)">Alexander\'s Disease</button>', nav_patch, html)

# Update Sidebars
ms_sidebar_re = r'<div id="ms-sidebar" class="tab-content">.*?</div>\s*</div>'
ms_sidebar_patch = """<div id="ms-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>MS Plaque Ablation (CF)</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        Simulate Neural Recovery and Plaque Density Mitigation for Multiple Sclerosis utilizing Continued Fractions and Ramanujan Operators.
                    </p>
                    <button class="btn-primary" id="btn-simulate-ms" onclick="simulateMS()" style="margin-top: 10px;">Run Ramanujan CF Mitigation</button>
                </div>
            </div>"""
html = re.sub(ms_sidebar_re, ms_sidebar_patch, html, flags=re.DOTALL)

alexander_sidebar_re = r'<div id="alexander-sidebar" class="tab-content">.*?</div>\s*</div>'
alexander_sidebar_match = re.search(alexander_sidebar_re, html, flags=re.DOTALL).group(0)

alexander_cf_sidebar = """
            <div id="alexander-cf-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>Alexander's Disease CF Addendum</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        Simulation of Neural Recovery and Rosenthal fiber plaque ablation evaluated with Continued Fractions.
                    </p>
                    <button class="btn-primary" id="btn-simulate-alexander-cf" onclick="simulateAlexanderCF()" style="margin-top: 10px;">Run CF Ablation</button>
                </div>
            </div>"""

html = html.replace(alexander_sidebar_match, alexander_sidebar_match + alexander_cf_sidebar)


# Update Main
ms_main_re = r'<div id="ms-main" class="tab-content" style="height: 100%;">.*?</div>\s*</div>\s*</div>\s*</div>'
ms_main_patch = """<div id="ms-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Multiple Sclerosis Plaque Mitigation (Ramanujan CF)</h2>
                    <div style="display: flex; gap: 10px; flex: 1;">
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                            <h3>Quantum Machine Learning & CF Specifications</h3>
                            <ul style="font-size: 12px; color: #add8e6; margin-top: 10px; list-style-type: square; margin-left: 20px;">
                                <li><strong>Target Structure:</strong> Thalamus & Basal Ganglia</li>
                                <li><strong>Frequency Range:</strong> 130 - 180 Hz (Dynamic CF Tuning)</li>
                                <li><strong>Algorithm:</strong> Ramanujan CF Operators</li>
                            </ul>
                            <h3>Mitigation Log</h3>
                            <pre id="ms-output" style="color: #0f0; font-family: monospace; font-size: 12px; margin-top: 10px; white-space: pre-wrap; height: 150px; overflow-y: auto;">Awaiting simulation...</pre>
                        </div>
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px; height: 350px;">
                            <canvas id="ms-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>"""
html = re.sub(ms_main_re, ms_main_patch, html, flags=re.DOTALL)

alexander_main_re = r'<div id="alexander-main" class="tab-content" style="height: 100%;">.*?</div>\s*</div>\s*</div>\s*</div>'
alexander_main_match = re.search(alexander_main_re, html, flags=re.DOTALL).group(0)

alexander_cf_main = """
            <div id="alexander-cf-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Alexander's Disease: Continued Fraction Addendum</h2>
                    <div style="display: flex; gap: 10px; flex: 1;">
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                            <h3>CF Neural Recovery Specifications</h3>
                            <ul style="font-size: 12px; color: #add8e6; margin-top: 10px; list-style-type: square; margin-left: 20px;">
                                <li><strong>Modeling constraint:</strong> Continued Fractions</li>
                                <li><strong>Target Structure:</strong> White Matter Astrocytes</li>
                                <li><strong>Framework:</strong> QML CF Plaque Ablation</li>
                            </ul>
                            <h3>Ablation Log</h3>
                            <pre id="alexander-cf-output" style="color: #0f0; font-family: monospace; font-size: 12px; margin-top: 10px; white-space: pre-wrap; height: 150px; overflow-y: auto;">Awaiting simulation...</pre>
                        </div>
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px; height: 350px;">
                            <canvas id="alexander-cf-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>"""

html = html.replace(alexander_main_match, alexander_main_match + alexander_cf_main)

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'w') as f:
    f.write(html)
