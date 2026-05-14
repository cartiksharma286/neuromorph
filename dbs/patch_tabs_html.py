with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'r') as f:
    html = f.read()

import re

# Update tabs navigation
nav_patch = """<button class="tab-btn" onclick="switchTab('ms', event)">MS Ablation</button>
            <button class="tab-btn" onclick="switchTab('alexander', event)">Alexander's Disease</button>"""
html = re.sub(r'<button class="tab-btn" onclick="switchTab\(\'ms\', event\)">MS / Alexander\'s</button>', nav_patch, html)

# Modify MS Sidebar and add Alexander sidebar
ms_sidebar_patch = """<div id="ms-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>MS Plaque Ablation</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        Simulate Neural Recovery and Plaque Density Mitigation for Multiple Sclerosis.
                    </p>
                    <button class="btn-primary" id="btn-simulate-ms" onclick="simulateMS()" style="margin-top: 10px;">Run Mitigation Model</button>
                </div>
            </div>

            <div id="alexander-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>Alexander's Disease Mitigation</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        Rosenthal fiber mitigation mapping via Quantum Machine Learning and Feynman Path Integrals in an adaptive ablation sense.
                    </p>
                    <button class="btn-primary" id="btn-simulate-alexander" onclick="simulateAlexander()" style="margin-top: 10px;">Run QML Adaptive Ablation</button>
                </div>
            </div>"""
html = re.sub(r'<div id="ms-sidebar" class="tab-content">.*?</div>\s*</div>', ms_sidebar_patch, html, flags=re.DOTALL)

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'w') as f:
    f.write(html)
