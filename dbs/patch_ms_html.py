with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'r') as f:
    html = f.read()

# Make sure the sidebar name matches the prompt better if needed
new_sidebar = """<div id="ms-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>MS Plaque Mitigation</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        Simulate Neural Recovery and Plaque Density Mitigation for Multiple Sclerosis mapped through Quantum Machine Learning target modeling.
                    </p>
                    <button class="btn-primary" id="btn-simulate-ms" onclick="simulateMS()" style="margin-top: 10px;">Run QML Mitigation Model</button>
                </div>
            </div>"""

import re
html = re.sub(r'<div id="ms-sidebar" class="tab-content">.*?</div>\s*</div>', new_sidebar, html, flags=re.DOTALL)


with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'w') as f:
    f.write(html)
