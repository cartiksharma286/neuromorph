"""
Basic web UI for Huawei Optical Networking App (placeholder for branding).
"""
from flask import Flask, render_template_string, request
from optical_switch import OpticalSwitch
from network_simulation import OpticalNetworkSimulator
from router_config import QuantumRouterConfig

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    tab = request.form.get("tab", "switch")
    # Optical Switch Tab
    if tab == "switch" and request.method == "POST":
        try:
            n_core = float(request.form.get("n_core", 1.5))
            n_clad = float(request.form.get("n_clad", 1.33))
            wavelength = float(request.form.get("wavelength", 1550))
            length = float(request.form.get("length", 10))
            order = int(request.form.get("order", 3))
            fea_nodes = int(request.form.get("fea_nodes", 10))
            run_fea = request.form.get("run_fea", "off") == "on"
            switch = OpticalSwitch(n_core, n_clad)
            result = {
                "critical_angle": switch.critical_angle(),
                "path_integral": switch.simulate_path_integral(wavelength, length),
                "continued_fraction": switch.continued_fraction_loss(order)
            }
            if run_fea:
                fea_result = switch.finite_element_simulation(nodes=fea_nodes, order=order)
                result["fea"] = fea_result
        except Exception as e:
            result = {"error": str(e)}
    # Network Simulation Tab
    elif tab == "network" and request.method == "POST":
        try:
            bandwidth = float(request.form.get("bandwidth", 800))
            switch_count = int(request.form.get("switch_count", 1))
            switches = [OpticalSwitch(1.5, 1.33)] * switch_count
            sim = OpticalNetworkSimulator(switches, bandwidth)
            result = sim.simulate()
        except Exception as e:
            result = {"error": str(e)}
    # Router Config Tab
    elif tab == "router" and request.method == "POST":
        try:
            topology = request.form.get("topology", "mesh")
            entanglement = request.form.get("entanglement", "on") == "on"
            router = QuantumRouterConfig(topology, {"entanglement": entanglement})
            result = {"config": router.configure()}
        except Exception as e:
            result = {"error": str(e)}
    return render_template_string('''
    <html>
    <head>
        <title>Huawei Optical Network App</title>
        <style>
            body { font-family:sans-serif; text-align:center; }
            .tabs { margin: 20px auto; display: flex; justify-content: center; }
            .tab { padding: 10px 30px; border: 1px solid #ccc; border-bottom: none; cursor: pointer; background: #f8f8f8; }
            .tab.selected { background: #fff; font-weight: bold; border-top: 2px solid #c00; }
            .tab-content { border: 1px solid #ccc; padding: 20px; max-width: 600px; margin: 0 auto; background: #fff; }
            .result { background: #f0f0f0; margin: 10px auto; padding: 10px; border-radius: 6px; max-width: 500px; }
        </style>
    </head>
    <body>
        <img src="/static/huawei_logo.png" alt="Huawei Logo" height="80"/><br/>
        <h1>Huawei Optical Networking Simulator</h1>
        <div class="tabs">
            <form method="post" style="display:inline;">
                <input type="hidden" name="tab" value="switch"/>
                <button class="tab {{'selected' if tab=='switch' else ''}}" type="submit">Optical Switch</button>
            </form>
            <form method="post" style="display:inline;">
                <input type="hidden" name="tab" value="network"/>
                <button class="tab {{'selected' if tab=='network' else ''}}" type="submit">Network Simulation</button>
            </form>
            <form method="post" style="display:inline;">
                <input type="hidden" name="tab" value="router"/>
                <button class="tab {{'selected' if tab=='router' else ''}}" type="submit">Router Config</button>
            </form>
        </div>
        <div class="tab-content">
            {% if tab=='switch' %}
            <form method="post">
                <input type="hidden" name="tab" value="switch"/>
                <label>Core Index: <input name="n_core" value="1.5" type="number" step="0.01"/></label><br/>
                <label>Cladding Index: <input name="n_clad" value="1.33" type="number" step="0.01"/></label><br/>
                <label>Wavelength (nm): <input name="wavelength" value="1550" type="number" step="1"/></label><br/>
                <label>Length (mm): <input name="length" value="10" type="number" step="0.1"/></label><br/>
                <label>Continued Fraction Order: <input name="order" value="3" type="number" min="1" max="10"/></label><br/>
                <label>FEA Nodes: <input name="fea_nodes" value="10" type="number" min="2" max="100"/></label><br/>
                <label>Run FEA: <input name="run_fea" type="checkbox"/></label><br/>
                <button type="submit">Simulate Optical Switch</button>
            </form>
            {% endif %}
            {% if tab=='network' %}
            <form method="post">
                <input type="hidden" name="tab" value="network"/>
                <label>Bandwidth (GHz): <input name="bandwidth" value="800" type="number" step="1"/></label><br/>
                <label>Switch Count: <input name="switch_count" value="1" type="number" min="1" max="10"/></label><br/>
                <button type="submit">Simulate Network</button>
            </form>
            {% endif %}
            {% if tab=='router' %}
            <form method="post">
                <input type="hidden" name="tab" value="router"/>
                <label>Topology: <input name="topology" value="mesh" type="text"/></label><br/>
                <label>Quantum Entanglement: <input name="entanglement" type="checkbox" checked/></label><br/>
                <button type="submit">Configure Router</button>
            </form>
            {% endif %}
            {% if result %}
            <div class="result">
                <b>Result:</b><br/>
                <pre>{{result}}</pre>
                {% if result.fea %}
                <hr/>
                <b>Finite Element Analysis (FEA) Results:</b><br/>
                Nodes: {{result.fea.nodes}}<br/>
                Loss (continued fraction): {{result.fea.loss}}<br/>
                Field (first 5 nodes): {{result.fea.field[:5]}}<br/>
                Field with Loss (first 5 nodes): {{result.fea.field_loss[:5]}}<br/>
                {% endif %}
            </div>
            {% endif %}
        </div>
    </body>
    </html>
    ''', tab=tab, result=result)

if __name__ == "__main__":
    app.run(debug=True, port=5002)