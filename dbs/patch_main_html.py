with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'r') as f:
    html = f.read()

import re

# We will search for id="ms-main" and replace it and add alexander-main
main_patch = """<div id="ms-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Multiple Sclerosis Plaque Mitigation</h2>
                    <div style="display: flex; gap: 10px; flex: 1;">
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                            <h3>Quantum Machine Learning DBS Specifications</h3>
                            <ul style="font-size: 12px; color: #add8e6; margin-top: 10px; list-style-type: square; margin-left: 20px;">
                                <li><strong>Target Structure:</strong> Thalamus & Basal Ganglia</li>
                                <li><strong>Frequency Range:</strong> 130 - 180 Hz (Dynamic QML Tuning)</li>
                                <li><strong>Pulse Width:</strong> 60 - 90 μs</li>
                                <li><strong>Algorithm:</strong> Quantum Neural Network Mitigation</li>
                            </ul>
                            <h3>Mitigation Log</h3>
                            <pre id="ms-output" style="color: #0f0; font-family: monospace; font-size: 12px; margin-top: 10px; white-space: pre-wrap; height: 150px; overflow-y: auto;">Awaiting simulation...</pre>
                        </div>
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px; height: 350px;">
                            <canvas id="ms-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>

            <div id="alexander-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Alexander's Disease: Rosenthal Fiber Mitigation</h2>
                    <div style="display: flex; gap: 10px; flex: 1;">
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                            <h3>Adaptive Ablation Specifications</h3>
                            <ul style="font-size: 12px; color: #add8e6; margin-top: 10px; list-style-type: square; margin-left: 20px;">
                                <li><strong>Modeling constraint:</strong> Feynman Path Integrals</li>
                                <li><strong>Target Structure:</strong> White Matter Astrocytes</li>
                                <li><strong>Framework:</strong> QML Adaptive Ablation</li>
                            </ul>
                            <h3>Ablation Log</h3>
                            <pre id="alexander-output" style="color: #0f0; font-family: monospace; font-size: 12px; margin-top: 10px; white-space: pre-wrap; height: 150px; overflow-y: auto;">Awaiting simulation...</pre>
                        </div>
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px; height: 350px;">
                            <canvas id="alexander-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>"""

html = re.sub(r'<div id="ms-main" class="tab-content" style="height: 100%;">.*?</div>\s*</div>\s*</div>\s*</div>\s*</div>', main_patch, html, flags=re.DOTALL)


with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'w') as f:
    f.write(html)
