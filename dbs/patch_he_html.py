import re

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'r') as f:
    html = f.read()

# Update Nav
nav_patch = r"""<button class="tab-btn" onclick="switchTab('huntington', event)">Huntington's Disease</button>
            <button class="tab-btn" onclick="switchTab('he-ms', event)">HE: MS</button>
            <button class="tab-btn" onclick="switchTab('he-huntington', event)">HE: Huntington's</button>
            <button class="tab-btn" onclick="switchTab('he-alexander', event)">HE: Alexander's</button>"""
html = re.sub(r'<button class="tab-btn" onclick="switchTab\(\'huntington\', event\)">Huntington\'s Disease</button>', nav_patch, html)

# Update Sidebars
huntington_sidebar_re = r'<div id="huntington-sidebar" class="tab-content">.*?</div>\s*</div>'
huntington_sidebar_match = re.search(huntington_sidebar_re, html, flags=re.DOTALL).group(0)

he_sidebars = """
            <div id="he-ms-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>MS Health Economics</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        Evaluate health economics, evidence-based outcomes, and cost-utility of DBS for Multiple Sclerosis.
                    </p>
                    <button class="btn-primary" id="btn-simulate-he-ms" onclick="simulateHEMS()" style="margin-top: 10px;">Run HE Analysis</button>
                </div>
            </div>
            
            <div id="he-huntington-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>Huntington's HE</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        Analyze health economics and evidence-based outcomes for interventional DBS in Huntington's disease.
                    </p>
                    <button class="btn-primary" id="btn-simulate-he-huntington" onclick="simulateHEHuntington()" style="margin-top: 10px;">Run HE Analysis</button>
                </div>
            </div>
            
            <div id="he-alexander-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>Alexander's HE</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        Analyze cost-effectiveness and health outcomes for Rosenthal fiber ablation in Alexander's disease.
                    </p>
                    <button class="btn-primary" id="btn-simulate-he-alexander" onclick="simulateHEAlexander()" style="margin-top: 10px;">Run HE Analysis</button>
                </div>
            </div>"""

html = html.replace(huntington_sidebar_match, huntington_sidebar_match + he_sidebars)

# Update Main Panel
huntington_main_re = r'<div id="huntington-main" class="tab-content" style="height: 100%;">.*?</div>\s*</div>\s*</div>\s*</div>'
huntington_main_match = re.search(huntington_main_re, html, flags=re.DOTALL).group(0)

he_mains = """
            <div id="he-ms-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Health Economics: Multiple Sclerosis DBS</h2>
                    <div style="display: flex; gap: 10px; flex: 1;">
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                            <h3>Evidence-Based Outcomes & Cost Utility</h3>
                            <ul style="font-size: 12px; color: #add8e6; margin-top: 10px; list-style-type: square; margin-left: 20px;">
                                <li><strong>QALY Gain:</strong> +2.5 Years Expected</li>
                                <li><strong>ICER:</strong> $45,000 / QALY</li>
                                <li><strong>Healthcare Savings:</strong> Reduced hospitalization by 40%</li>
                            </ul>
                            <h3>Economic Log</h3>
                            <pre id="he-ms-output" style="color: #0f0; font-family: monospace; font-size: 12px; margin-top: 10px; white-space: pre-wrap; height: 150px; overflow-y: auto;">Awaiting HE calculation...</pre>
                        </div>
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px; height: 350px;">
                            <canvas id="he-ms-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div id="he-huntington-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Health Economics: Huntington's DBS</h2>
                    <div style="display: flex; gap: 10px; flex: 1;">
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                            <h3>Interventional Outcomes & Value</h3>
                            <ul style="font-size: 12px; color: #add8e6; margin-top: 10px; list-style-type: square; margin-left: 20px;">
                                <li><strong>QALY Gain:</strong> +3.1 Years Expected</li>
                                <li><strong>ICER:</strong> $38,000 / QALY</li>
                                <li><strong>Caregiver Burden:</strong> Reduced by 60%</li>
                            </ul>
                            <h3>Economic Log</h3>
                            <pre id="he-huntington-output" style="color: #0f0; font-family: monospace; font-size: 12px; margin-top: 10px; white-space: pre-wrap; height: 150px; overflow-y: auto;">Awaiting HE calculation...</pre>
                        </div>
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px; height: 350px;">
                            <canvas id="he-huntington-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div id="he-alexander-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Health Economics: Alexander's Disease DBS</h2>
                    <div style="display: flex; gap: 10px; flex: 1;">
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                            <h3>Ablation Outcomes & Utility Mapping</h3>
                            <ul style="font-size: 12px; color: #add8e6; margin-top: 10px; list-style-type: square; margin-left: 20px;">
                                <li><strong>QALY Gain:</strong> +4.0 Years Expected</li>
                                <li><strong>ICER:</strong> $41,500 / QALY</li>
                                <li><strong>Symptom Control:</strong> 75% reduction in seizures</li>
                            </ul>
                            <h3>Economic Log</h3>
                            <pre id="he-alexander-output" style="color: #0f0; font-family: monospace; font-size: 12px; margin-top: 10px; white-space: pre-wrap; height: 150px; overflow-y: auto;">Awaiting HE calculation...</pre>
                        </div>
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px; height: 350px;">
                            <canvas id="he-alexander-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>"""

html = html.replace(huntington_main_match, huntington_main_match + he_mains)

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'w') as f:
    f.write(html)
