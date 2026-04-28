
from flask import Flask, render_template_string, request
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server
import matplotlib.pyplot as plt
import io
import base64


from clinical_protocols import get_protocols

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DBS for Jet Lag & Sleep Apnea</title>
    <style>
        .tab { display: inline-block; margin-right: 20px; padding: 10px; background: #eee; border-radius: 5px 5px 0 0; }
        .tab.active { background: #fff; border-bottom: 2px solid #fff; }
        .tab-content { border: 1px solid #eee; padding: 20px; border-radius: 0 5px 5px 5px; background: #fff; }
    </style>
</head>
<body>
    <h1>Deep Brain Stimulation for Jet Lag & Sleep Apnea</h1>
    <div>
        <a href="/?tab=simulate" class="tab {% if tab == 'simulate' %}active{% endif %}">Simulation</a>
        <a href="/?tab=protocols" class="tab {% if tab == 'protocols' %}active{% endif %}">Clinical Protocols</a>
    </div>
    <div class="tab-content">
    {% if tab == 'simulate' %}
        <form method="post">
            <label>Time Steps: <input type="number" name="time_steps" value="{{ time_steps }}" min="50" max="500"></label><br>
            <label>Neurons: <input type="number" name="neurons" value="{{ neurons }}" min="10" max="200"></label><br>
            <label>DBS Intensity: <input type="number" step="0.01" name="dbs_intensity" value="{{ dbs_intensity }}" min="0.0" max="2.0"></label><br>
            <label>Sleep Apnea Factor: <input type="number" step="0.01" name="apnea_factor" value="{{ apnea_factor }}" min="0.0" max="1.0"></label><br>
            <label>FEA Grid Size: <input type="number" name="fea_grid" value="{{ fea_grid }}" min="5" max="50"></label><br>
            <input type="submit" value="Simulate">
        </form>
        {% if plot_url %}
            <h2>Simulation Result</h2>
            <img src="data:image/png;base64,{{ plot_url }}"/>
            <p>Simulation complete. The heatmap shows neuronal activity over time with DBS interventions for jet lag and sleep apnea repair.</p>
            <h3>FEA DBS Field</h3>
            <img src="data:image/png;base64,{{ fea_url }}"/>
            <p>FEA simulation: DBS field distribution in neural grid.</p>
        {% endif %}
    {% elif tab == 'protocols' %}
        <h2>Clinical Treatment Protocols for Jet Lag</h2>
        {% for protocol in protocols %}
            <div style="margin-bottom: 20px;">
                <strong>{{ protocol.name }}</strong><br>
                <em>{{ protocol.description }}</em>
                <ul>
                {% for step in protocol.steps %}
                    <li>{{ step }}</li>
                {% endfor %}
                </ul>
            </div>
        {% endfor %}
    {% endif %}
    </div>
</body>
</html>
'''


@app.route('/', methods=['GET', 'POST'])
def index():
    import urllib.parse
    tab = request.args.get('tab', 'simulate')
    plot_url = None
    fea_url = None
    protocols = get_protocols() if tab == 'protocols' else None
    # Defaults

    def safe_int(val, default, minv=None, maxv=None):
        try:
            v = int(val)
            if minv is not None: v = max(v, minv)
            if maxv is not None: v = min(v, maxv)
            return v
        except Exception:
            return default

    def safe_float(val, default, minv=None, maxv=None):
        try:
            v = float(val)
            if minv is not None: v = max(v, minv)
            if maxv is not None: v = min(v, maxv)
            return v
        except Exception:
            return default

    time_steps = safe_int(request.form.get('time_steps', 200), 200, 50, 500)
    neurons = safe_int(request.form.get('neurons', 50), 50, 10, 200)
    dbs_intensity = safe_float(request.form.get('dbs_intensity', 1.0), 1.0, 0.0, 2.0)
    apnea_factor = safe_float(request.form.get('apnea_factor', 0.5), 0.5, 0.0, 1.0)
    fea_grid = safe_int(request.form.get('fea_grid', 20), 20, 5, 50)

    if tab == 'simulate' and request.method == 'POST':
        np.random.seed(42)
        activity = np.random.rand(neurons)
        weights = np.random.rand(neurons, neurons) * 0.1
        activity_history = [activity.copy()]
        apnea_events = np.random.binomial(1, apnea_factor, time_steps)
        dt = 0.1  # continuous time step
        for t in range(time_steps):
            # DBS effect (continuous time scale)
            activity += dt * dbs_intensity * np.dot(weights, activity)
            # Sleep apnea event: sudden drop in activity
            if apnea_events[t]:
                activity *= 0.7
            # Normalize
            activity = np.clip(activity, 0, 1)
            activity_history.append(activity.copy())
        activity_history = np.array(activity_history)
        # Plot main simulation
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(activity_history.T, aspect='auto', cmap='inferno', interpolation='nearest')
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Neuron Index")
        ax.set_title("Activity Heatmap (DBS + Apnea)")
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        plot_url = base64.b64encode(buf.read()).decode('utf8')
        # FEA-like simulation: DBS field in a 2D grid
        grid = np.zeros((fea_grid, fea_grid))
        center = fea_grid // 2
        for i in range(fea_grid):
            for j in range(fea_grid):
                dist = np.sqrt((i-center)**2 + (j-center)**2)
                grid[i, j] = dbs_intensity * np.exp(-dist/3)
        fig2, ax2 = plt.subplots(figsize=(4,4))
        c = ax2.imshow(grid, cmap='Blues', interpolation='bilinear')
        ax2.set_title('DBS Field (FEA Approx)')
        # Fix: create colorbar on the same figure as ax2
        fig2.colorbar(c, ax=ax2, label='Field Strength')
        plt.tight_layout()
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        plt.close(fig2)
        buf2.seek(0)
        fea_url = base64.b64encode(buf2.read()).decode('utf8')
    return render_template_string(
        HTML,
        plot_url=plot_url,
        fea_url=fea_url,
        tab=tab,
        protocols=protocols,
        time_steps=time_steps,
        neurons=neurons,
        dbs_intensity=dbs_intensity,
        apnea_factor=apnea_factor,
        fea_grid=fea_grid
    )

if __name__ == '__main__':
    import sys
    port = 5000
    for arg in sys.argv:
        if arg.startswith('--port='):
            try:
                port = int(arg.split('=')[1])
            except Exception:
                pass
    app.run(debug=True, port=port)
