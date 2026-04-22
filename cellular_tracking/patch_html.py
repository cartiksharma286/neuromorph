import re

html_path = 'index.html'
with open(html_path, 'r') as f:
    html = f.read()

nav_insert = """
                <button class="nav-item" data-view="microfluidics" onclick="document.querySelectorAll('.view-content').forEach(el=>el.classList.add('hidden')); document.getElementById('microfluidics-view').classList.remove('hidden'); document.querySelectorAll('.nav-item').forEach(el=>el.classList.remove('active')); this.classList.add('active');">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10" />
                        <path d="M12 2A15.3 15.3 0 0 1 17 12a15.3 15.3 0 0 1-5 10A15.3 15.3 0 0 1 7 12 15.3 15.3 0 0 1 12 2z" />
                        <path d="M2 12h20" />
                    </svg>
                    <span>Pathology CFD</span>
                </button>

                <button class="nav-item" data-view="optical" onclick="document.querySelectorAll('.view-content').forEach(el=>el.classList.add('hidden')); document.getElementById('optical-view').classList.remove('hidden'); document.querySelectorAll('.nav-item').forEach(el=>el.classList.remove('active')); this.classList.add('active');">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2v20" />
                        <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
                        <circle cx="12" cy="12" r="3" />
                    </svg>
                    <span>Optical Photonics</span>
                </button>
            </div>
"""

views_insert = """
            <!-- Microfluidics CFD View -->
            <div id="microfluidics-view" class="view-content hidden">
                <div class="panel">
                    <div class="panel-header">
                        <h3>Microfluidics Computational Fluid Dynamics</h3>
                        <p>Simulate cellular trajectory drift within a finite capillary micro-chip</p>
                    </div>
                    <div class="panel-body" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div class="control-group">
                            <label>Inlet Pressure (Pa)</label>
                            <input type="range" id="cfd-pressure" min="10" max="500" value="100" />
                            <span id="cfd-pressure-val">100</span>
                        </div>
                        <div class="control-group">
                            <label>Viscosity (Poisuille)</label>
                            <input type="range" id="cfd-viscosity" min="1" max="100" value="10" />
                            <span id="cfd-viscosity-val">10</span>
                        </div>
                        <button class="btn btn-primary" style="grid-column: span 2;" onclick="runCFDSimulation()">Run Navier-Stokes CFD</button>
                    </div>
                    <div class="visualization-container" style="margin-top: 20px; height: 300px; background: #0a0a0f; border-radius: 8px; position: relative;">
                        <canvas id="cfdCanvas" width="800" height="300" style="width: 100%; height: 100%;"></canvas>
                        <div id="cfd-overlay" style="position: absolute; top: 10px; left: 10px; color: var(--accent-cyan); font-family: monospace;">Awaiting Simulation...</div>
                    </div>
                </div>
            </div>

            <!-- Optical Photonics View -->
            <div id="optical-view" class="view-content hidden">
                <div class="panel">
                    <div class="panel-header">
                        <h3>Optical Microscopy (Photonics)</h3>
                        <p>Evaluate diffractive refractive indexes for cell morphology categorization</p>
                    </div>
                    <div class="panel-body" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div class="control-group">
                            <label>Wavelength (nm)</label>
                            <input type="range" id="optical-wave" min="400" max="700" value="550" />
                            <span id="optical-wave-val" style="color:var(--accent-pink);">550</span>
                        </div>
                        <div class="control-group">
                            <label>Refractive Index (n)</label>
                            <input type="range" id="optical-refract" min="10" max="25" value="13" />
                            <span id="optical-refract-val">1.3</span>
                        </div>
                        <button class="btn btn-primary" style="grid-column: span 2;" onclick="runPhotonics()">Analyze Optical Signatures</button>
                    </div>
                    <div class="visualization-container" style="margin-top: 20px; height: 300px; background: #0a0a0f; border-radius: 8px; position: relative;">
                        <canvas id="opticalCanvas" width="800" height="300" style="width: 100%; height: 100%;"></canvas>
                        <div id="optical-overlay" style="position: absolute; top: 10px; left: 10px; color: var(--accent-pink); font-family: monospace;">Awaiting Signatures...</div>
                    </div>
                </div>
            </div>
</main>
"""

# Hard reset and replace structure
if 'data-view="microfluidics"' not in html:
    html = html.replace('</div>\n\n            <div class="sidebar-footer">', nav_insert + '\n            <div class="sidebar-footer">')
    html = html.replace('</main>', views_insert)
    
    with open(html_path, 'w') as f:
        f.write(html)
    print("HTML Patched!")
else:
    print("HTML already patched. Enabling javascript listeners.")
