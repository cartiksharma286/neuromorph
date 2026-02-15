
import os

# New Dementia Tab Button to inject
new_tab = """            <button class="nav-tab" data-module="dementia">
                <span class="tab-icon">💡</span>
                Dementia Care
            </button>"""

# New Dementia View Module to inject
new_view = """    <!-- Dementia Treatment View -->
    <div id="dementiaModule" class="module" style="display:none;">
        <div class="module-header">
            <h2>Dementia & Alzheimer's Treatment</h2>
            <p>Cognitive preservation through cholinergic nucleus basalis stimulation</p>
        </div>

        <div class="grid-2">
            <div class="card">
                <h3>Cognitive Scores</h3>
                <div class="grid-2" style="margin-bottom:10px;">
                   <div class="metric-card"><div class="metric-label">MMSE</div><div class="metric-value" id="metricMMSE">--</div><div class="metric-bar"><div class="metric-fill" id="barMMSE"></div></div></div>
                   <div class="metric-card"><div class="metric-label">MoCA</div><div class="metric-value" id="metricMoCA">--</div><div class="metric-bar"><div class="metric-fill" id="barMoCA"></div></div></div>
                </div>
                <div class="grid-2">
                   <div class="metric-card"><div class="metric-label">Memory Enc.</div><div class="metric-value" id="metricMemoryEncoding">--</div><div class="metric-bar"><div class="metric-fill" id="barMemoryEncoding"></div></div></div>
                   <div class="metric-card"><div class="metric-label">Memory Ret.</div><div class="metric-value" id="metricMemoryRetrieval">--</div><div class="metric-bar"><div class="metric-fill" id="barMemoryRetrieval"></div></div></div>
                </div>
            </div>

            <div class="card">
                <h3>Neuropathology Biomarkers</h3>
                <div id="dementiaBiomarkers" class="biomarkers-grid" style="grid-template-columns: 1fr 1fr;"></div>
            </div>
        </div>

        <div class="grid-2">
            <div class="card">
                <h3>Simulation Controls</h3>
                <div class="parameter-controls">
                    <label>Target Region</label>
                    <select id="dementiaTargetRegion">
                        <option value="nucleus_basalis">Nucleus Basalis</option>
                        <option value="fornix">Fornix</option>
                    </select>
                    
                    <label>Amplitude (mA)</label>
                    <input type="number" id="dementiaAmplitude" value="3.0" step="0.1">
                    
                    <label>Frequency (Hz)</label>
                    <input type="number" id="dementiaFrequency" value="20">
                    
                    <label>Pulse Width (μs)</label>
                    <input type="number" id="dementiaPulseWidth" value="90">
                </div>
                <button id="simulateDementiaBtn" class="btn-primary" style="margin-top:20px;">Run Cholinergic Simulation</button>
            </div>

            <div class="card">
                <h3>Neural Activity Response</h3>
                <canvas id="dementiaActivityCanvas" style="width:100%; height:300px;"></canvas>
            </div>
        </div>

        <div class="card">
            <h3>Long-term Prognosis (6-Month Predictor)</h3>
            <div style="margin-bottom:15px;">
                <label style="color:#aaa;">Projected Duration (Months):</label>
                <input type="number" id="treatmentMonths" value="6" style="background:#333; border:1px solid #555; colr:#fff; padding:5px; width:60px;">
                <button id="predictDementiaBtn" class="btn-secondary" style="display:inline-block; margin-left:10px;">Predict Outcome</button>
            </div>
            <canvas id="dementiaPredictionCanvas" style="width:100%; height:300px;"></canvas>
        </div>
    </div>
"""

path = 'index.html'
with open(path, 'r') as f:
    content = f.read()

# 1. Inject Tab if not present
if 'data-module="dementia"' not in content:
    # Insert before ASD tab
    tab_marker = '<button class="nav-tab" data-module="asd">'
    content = content.replace(tab_marker, new_tab + "\n            " + tab_marker)
    print("Injected Dementia Tab")

# 2. Inject View if not present
if 'id="dementiaModule"' not in content:
    # Insert before ASD Module
    view_marker = '<div id="asdModule"'
    content = content.replace(view_marker, new_view + "\n    " + view_marker)
    print("Injected Dementia View")

with open(path, 'w') as f:
    f.write(content)
print("Update Complete")
