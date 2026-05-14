import re

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'r') as f:
    html = f.read()

# Update Nav
nav_patch = r"""<button class="tab-btn" onclick="switchTab('he-alexander', event)">HE: Alexander's</button>
            <button class="tab-btn" onclick="switchTab('cortical-sim', event)">Cortical Simulation</button>
            <button class="tab-btn" onclick="switchTab('market-valuation', event)">Net Market Valuation (2026-2036)</button>"""
html = html.replace('''<button class="tab-btn" onclick="switchTab('he-alexander', event)">HE: Alexander's</button>''', nav_patch)

# Update Sidebars
he_alexander_sidebar_re = r'<div id="he-alexander-sidebar" class="tab-content">.*?</div>\s*</div>'
match = re.search(he_alexander_sidebar_re, html, flags=re.DOTALL)
if match:
    he_alexander_sidebar_match = match.group(0)

    new_sidebars = """
            <div id="cortical-sim-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>Cortical Simulation</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        Simulate cortical networks, pyramidal neuron firing rates, and macro-scale projection costs.
                    </p>
                    <button class="btn-primary" id="btn-simulate-cortical" onclick="simulateCortical()" style="margin-top: 10px;">Run Cortical Sim</button>
                </div>
            </div>
            
            <div id="market-valuation-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>Market Valuation</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        Calculate projection costs and net market valuation over a 10-year period to 2036.
                    </p>
                    <button class="btn-primary" id="btn-simulate-market" onclick="simulateMarketValuation()" style="margin-top: 10px;">Project 10-Year Valuation</button>
                </div>
            </div>"""
    html = html.replace(he_alexander_sidebar_match, he_alexander_sidebar_match + new_sidebars)

# Update Main Panel
he_alexander_main_re = r'<div id="he-alexander-main" class="tab-content" style="height: 100%;">.*?</div>\s*</div>\s*</div>\s*</div>'
match_main = re.search(he_alexander_main_re, html, flags=re.DOTALL)
if match_main:
    he_alexander_main_match = match_main.group(0)
    
    new_mains = """
            <div id="cortical-sim-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Cortical Dynamics & Projection Costs</h2>
                    <div style="display: flex; gap: 10px; flex: 1;">
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                            <h3>Network & Projection Metrics</h3>
                            <ul style="font-size: 12px; color: #add8e6; margin-top: 10px; list-style-type: square; margin-left: 20px;">
                                <li><strong>Active Synapses:</strong> 1.2 x 10^9</li>
                                <li><strong>Global Projection Energy:</strong> 45.2 mWh</li>
                                <li><strong>Cortical Coherence:</strong> High Synchronization (Gamma band)</li>
                            </ul>
                            <h3>Simulation Log</h3>
                            <pre id="cortical-sim-output" style="color: #0f0; font-family: monospace; font-size: 12px; margin-top: 10px; white-space: pre-wrap; height: 150px; overflow-y: auto;">Awaiting cortical data...</pre>
                        </div>
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px; height: 350px;">
                            <canvas id="cortical-sim-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div id="market-valuation-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Net Market Valuation (2026-2036)</h2>
                    <div style="display: flex; gap: 10px; flex: 1;">
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                            <h3>10-Year Valuation & Projection Costs</h3>
                            <ul style="font-size: 12px; color: #add8e6; margin-top: 10px; list-style-type: square; margin-left: 20px;">
                                <li><strong>CAGR:</strong> 18.5% Expected</li>
                                <li><strong>Forecast (2036):</strong> $12.4 Billion</li>
                                <li><strong>R&D / Projection Costs:</strong> Normalized at 15% ARR</li>
                            </ul>
                            <h3>Economic Engine Log</h3>
                            <pre id="market-valuation-output" style="color: #0f0; font-family: monospace; font-size: 12px; margin-top: 10px; white-space: pre-wrap; height: 150px; overflow-y: auto;">Awaiting 10-year projection data...</pre>
                        </div>
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px; height: 350px;">
                            <canvas id="market-valuation-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>"""
    html = html.replace(he_alexander_main_match, he_alexander_main_match + new_mains)

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'w') as f:
    f.write(html)
