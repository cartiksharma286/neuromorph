import re

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'r') as f:
    html = f.read()

# Update Nav
nav_old = r"""<button class="tab-btn" onclick="switchTab('sea-valuation', event)">South Asia/SEA Market (10 YR)</button>"""
nav_new = nav_old + """
            <button class="tab-btn" onclick="switchTab('india-valuation', event)">India Projections (10 YR)</button>
            <button class="tab-btn" onclick="switchTab('america-valuation', event)">America Projections (10 YR)</button>"""
html = html.replace(nav_old, nav_new)

# Update Sidebars
sidebar_old_re = r'<div id="sea-valuation-sidebar" class="tab-content">.*?</div>\s*</div>'
match = re.search(sidebar_old_re, html, flags=re.DOTALL)
if match:
    sidebar_old = match.group(0)
    sidebar_new = sidebar_old + """
            <div id="india-valuation-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>India DBS Projections</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        10-year localized estimates for Deep Brain Stimulation growth, adoption, and scaling in India.
                    </p>
                    <button class="btn-primary" id="btn-simulate-india" onclick="simulateIndiaValuation()" style="margin-top: 10px;">Run India Estimates</button>
                </div>
            </div>
            
            <div id="america-valuation-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>America DBS Projections</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        10-year estimates for DBS market volume, technological maturity, and costs in the Americas.
                    </p>
                    <button class="btn-primary" id="btn-simulate-america" onclick="simulateAmericaValuation()" style="margin-top: 10px;">Run America Estimates</button>
                </div>
            </div>"""
    html = html.replace(sidebar_old, sidebar_new)

# Update Main Panel
main_old_re = r'<div id="sea-valuation-main" class="tab-content" style="height: 100%;">.*?</div>\s*</div>\s*</div>\s*</div>'
match_main = re.search(main_old_re, html, flags=re.DOTALL)
if match_main:
    main_old = match_main.group(0)
    main_new = main_old + """
            <div id="india-valuation-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Regional DBS Valuation: India (2026-2036)</h2>
                    <div style="display: flex; gap: 10px; flex: 1;">
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                            <h3>India Market Trajectory & Scaling</h3>
                            <ul style="font-size: 12px; color: #add8e6; margin-top: 10px; list-style-type: square; margin-left: 20px;">
                                <li><strong>CAGR (Regional):</strong> 24.5% Expected (Explosive Growth)</li>
                                <li><strong>Forecast (2036):</strong> $2.8 Billion</li>
                                <li><strong>Primary Adoption Driver:</strong> Local manufacturing & affordable care policies</li>
                            </ul>
                            <h3>Economic Engine Log</h3>
                            <pre id="india-valuation-output" style="color: #0f0; font-family: monospace; font-size: 12px; margin-top: 10px; white-space: pre-wrap; height: 150px; overflow-y: auto;">Awaiting India regional projection data...</pre>
                        </div>
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px; height: 350px;">
                            <canvas id="india-valuation-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div id="america-valuation-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Regional DBS Valuation: America (2026-2036)</h2>
                    <div style="display: flex; gap: 10px; flex: 1;">
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                            <h3>Americas Market Maturity Map</h3>
                            <ul style="font-size: 12px; color: #add8e6; margin-top: 10px; list-style-type: square; margin-left: 20px;">
                                <li><strong>CAGR:</strong> 12.1% Expected (Mature Scaling)</li>
                                <li><strong>Forecast (2036):</strong> $8.5 Billion</li>
                                <li><strong>Next-Gen Catalyst:</strong> Closed-loop adaptive stimulation arrays</li>
                            </ul>
                            <h3>Economic Engine Log</h3>
                            <pre id="america-valuation-output" style="color: #0f0; font-family: monospace; font-size: 12px; margin-top: 10px; white-space: pre-wrap; height: 150px; overflow-y: auto;">Awaiting America regional projection data...</pre>
                        </div>
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px; height: 350px;">
                            <canvas id="america-valuation-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>"""
    html = html.replace(main_old, main_new)

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'w') as f:
    f.write(html)
