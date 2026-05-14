import re

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'r') as f:
    html = f.read()

# Update Nav
nav_patch = r"""<button class="tab-btn" onclick="switchTab('market-valuation', event)">Net Market Valuation (2026-2036)</button>
            <button class="tab-btn" onclick="switchTab('sea-valuation', event)">South Asia/SEA Market (10 YR)</button>"""
html = html.replace('''<button class="tab-btn" onclick="switchTab('market-valuation', event)">Net Market Valuation (2026-2036)</button>''', nav_patch)

# Update Sidebars
market_val_sidebar_re = r'<div id="market-valuation-sidebar" class="tab-content">.*?</div>\s*</div>'
match = re.search(market_val_sidebar_re, html, flags=re.DOTALL)
if match:
    market_val_sidebar_match = match.group(0)

    new_sidebar = """
            <div id="sea-valuation-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>South Asia & SEA Market</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        Calculate Deep Brain Stimulation market trajectory across South Asia and SEA over the next 10 years.
                    </p>
                    <button class="btn-primary" id="btn-simulate-sea" onclick="simulateSEAValuation()" style="margin-top: 10px;">Run Regional Projection</button>
                </div>
            </div>"""
    html = html.replace(market_val_sidebar_match, market_val_sidebar_match + new_sidebar)

# Update Main Panel
market_val_main_re = r'<div id="market-valuation-main" class="tab-content" style="height: 100%;">.*?</div>\s*</div>\s*</div>\s*</div>'
match_main = re.search(market_val_main_re, html, flags=re.DOTALL)
if match_main:
    market_val_main_match = match_main.group(0)
    
    new_main = """
            <div id="sea-valuation-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Regional DBS Valuation: South Asia & SEA (2026-2036)</h2>
                    <div style="display: flex; gap: 10px; flex: 1;">
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                            <h3>Emerging Market Forecasting</h3>
                            <ul style="font-size: 12px; color: #add8e6; margin-top: 10px; list-style-type: square; margin-left: 20px;">
                                <li><strong>CAGR (Regional):</strong> 21.2% Expected (High Growth)</li>
                                <li><strong>Forecast (2036):</strong> $3.8 Billion</li>
                                <li><strong>Accessibility Factor:</strong> Improving clinical infrastructure</li>
                            </ul>
                            <h3>Economic Engine Log</h3>
                            <pre id="sea-valuation-output" style="color: #0f0; font-family: monospace; font-size: 12px; margin-top: 10px; white-space: pre-wrap; height: 150px; overflow-y: auto;">Awaiting regional 10-year projection data...</pre>
                        </div>
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px; height: 350px;">
                            <canvas id="sea-valuation-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>"""
    html = html.replace(market_val_main_match, market_val_main_match + new_main)

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'w') as f:
    f.write(html)
