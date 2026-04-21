html_path = '/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html'
with open(html_path, 'r') as f:
    html = f.read()

# 1. Add Nav BTN
nav_target = """<button class="tab-btn" onclick="switchTab('dementia', event)">Dementia Opt. Protocol</button>"""
nav_new = nav_target + """\n            <button class="tab-btn" onclick="switchTab('stage', event)">Stage-Gated Protocol</button>"""
html = html.replace(nav_target, nav_new)

# 2. Add Sidebar
sidebar_target = """<div id="fea-sidebar" class="tab-content">"""
sidebar_new = """<div id="stage-sidebar" class="tab-content">
                <div class="glass-panel">
                    <h2>Stage-Gated Protocol</h2>
                    <p style="font-size: 11px; margin-bottom: 10px; color: var(--text-dim);">
                        Utilizing Queueing Models (M/M/1) to balance tau protein aggregation rates against glymphatic clearance.
                    </p>
                    <button class="btn-primary" id="btn-fetch-stages" onclick="fetchStageProtocol()">Optimize Stages</button>
                </div>
            </div>\n\n            """ + sidebar_target

html = html.replace(sidebar_target, sidebar_new)

# 3. Add Main Content
main_target = """<div id="fea-main" class="tab-content" style="height: 100%;">"""
main_new = """<div id="stage-main" class="tab-content" style="height: 100%;">
                <div class="glass-panel" style="height: 100%; display: flex; flex-direction: column;">
                    <h2>Clinical DBS Stage-Gated Protocol for Dementia Care</h2>
                    <div id="stage-protocol-container" style="flex-grow: 1; overflow-y: auto; padding: 10px; display: grid; gap: 15px; grid-template-columns: 1fr;">
                        <div style="color: var(--text-dim); text-align: center; margin-top: 50px;">
                            Click "Optimize Stages" to generate queueing progression protocol.
                        </div>
                    </div>
                </div>
            </div>\n\n            """ + main_target

html = html.replace(main_target, main_new)

with open(html_path, 'w') as f:
    f.write(html)
print("html patched")
