js_path = '/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js'
with open(js_path, 'r') as f:
    js = f.read()

# Add the fetch method for the new stage-gated protocol using the backend.
new_func = """

// Stage-Gated Dementia Protocol (Queueing Theory)
async function fetchStageProtocol() {
    try {
        const response = await fetch('/api/stage-gated-protocol');
        const data = await response.json();
        
        const container = document.getElementById('stage-protocol-container');
        container.innerHTML = data.protocol.map(stage => `
            <div class="stat-card" style="border-left: 3px solid var(--accent-cyan); display: flex; flex-direction: column; gap: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin:0; color: #00f2ff;">${stage.name}</h3>
                    <span style="font-size:10px; padding: 3px 8px; background: rgba(255,0,200,0.2); border-radius: 12px; color: var(--accent-pink);">
                        Stage ${stage.stage}
                    </span>
                </div>
                <p style="font-size: 11px; margin: 0; color: var(--text-dim);">${stage.desc}</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 5px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
                    <div>
                        <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase;">Electrical Protocol</div>
                        <ul style="margin: 5px 0 0 15px; font-size: 11px; color: #fff;">
                            <li>Voltage: <span style="color:var(--accent-cyan);">${stage.electrical.voltage_v} V</span></li>
                            <li>Frequency: <span style="color:var(--accent-cyan);">${stage.electrical.frequency_hz} Hz</span></li>
                            <li>Pulse Width: <span style="color:var(--accent-cyan);">${stage.electrical.pulse_width_us} µs</span></li>
                            <li>Target: <span style="color:var(--accent-cyan);">${stage.electrical.target}</span></li>
                        </ul>
                    </div>
                    <div>
                        <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase;">Molecular Queueing (M/M/1)</div>
                        <ul style="margin: 5px 0 0 15px; font-size: 11px; color: #fff;">
                            <li>Tau Aggregation Rate (λ): <span style="color:var(--accent-pink);">${stage.queueing.lambda_arrival} /yr</span></li>
                            <li>Glymphatic Clearance (μ): <span style="color:var(--accent-pink);">${stage.queueing.mu_clearance} /yr</span></li>
                            <li>System Utilization (ρ): <span style="color:var(--accent-pink);">${stage.queueing.rho_utilization}</span></li>
                            <li>Queue Length (Lq): <span style="color:var(--accent-pink);">${stage.queueing.l_q}</span></li>
                        </ul>
                    </div>
                </div>
            </div>
        `).join('');
    } catch(err) {
        console.error(err);
    }
}
"""

js = js + new_func
with open(js_path, 'w') as f:
    f.write(js)
print("js patched")
