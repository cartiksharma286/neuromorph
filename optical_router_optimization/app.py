import os
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate')
def simulate():
    # Performance benchmarks: Base X3 Pro (Wi-Fi 6 Plus, ~3000 Mbps, 5GHz)
    # Target Optimal Design: OptiMesh 8G (Optical Switches + 8GHz Relay, ~10000 Mbps)
    
    time_bins = np.linspace(0, 10, 1000)
    
    # Statistical Distributions for Wave Packets
    # Base packet arrival (Poisson-like envelope modeling heavy traffic)
    base_traffic = np.random.poisson(lam=50, size=1000)
    
    # 5GHz Wave Packet Dispersion (Standard fading model using Gaussian decay)
    base_dispersion = np.exp(-0.5 * ((time_bins - 5.5) / 1.5)**2) * base_traffic
    
    # OptiMesh 8G Wave Packet Dispersion
    # The optical switch limits node-to-node relay loss and 8GHz band tightens dispersion
    opt_traffic = np.random.poisson(lam=120, size=1000)
    opt_dispersion = np.exp(-0.5 * ((time_bins - 5.0) / 0.5)**2) * opt_traffic
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Wave Packet Distributions
    ax1.plot(time_bins, base_dispersion, label='Huawei Mesh X3 Pro (5GHz)', color='#ef4444', alpha=0.7)
    ax1.plot(time_bins, opt_dispersion, label='OptiMesh 8G (Optical Relay)', color='#06b6d4', alpha=0.9, lw=2)
    ax1.set_title("Wave Packet Statistical Distribution (Dispersion)")
    ax1.set_xlabel("Time Relay Inter-Arrival (ns)")
    ax1.set_ylabel("Packet Amplitude / Density")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    ax1.set_facecolor('#0f172a')
    
    # Plot 2: Optical Switching Network Optimization vs Distance
    distances = np.linspace(0, 120, 100) # distance in meters
    # Standard router drops sharply through physical barriers
    base_throughput = 3000 * np.exp(-distances / 25) 
    # Optic-relay design maintains coherent signal stability much longer
    optical_throughput = 10000 * np.exp(-distances / 90)
    
    ax2.plot(distances, base_throughput, label='Base X3 Pro', color='#ef4444', lw=2)
    ax2.plot(distances, optical_throughput, label='OptiMesh 8G', color='#06b6d4', lw=2)
    ax2.set_title("Optical Network Communications vs Signal Attenuation")
    ax2.set_xlabel("Distance from Main Node (m)")
    ax2.set_ylabel("Sustained Throughput (Mbps)")
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    ax2.set_facecolor('#0f172a')
    
    # Theming
    fig.patch.set_facecolor('#0f172a')
    for ax in [ax1, ax2]:
        ax.tick_params(colors='#e2e8f0')
        ax.xaxis.label.set_color('#e2e8f0')
        ax.yaxis.label.set_color('#e2e8f0')
        ax.title.set_color('#bae6fd')
        for spine in ax.spines.values():
            spine.set_color('#334155')
            
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return jsonify({
        'metrics': {
            'x3_pro_latency': '4.5 ms',
            'opti_latency': '0.7 ms',
            'x3_pro_throughput': '3000 Mbps',
            'opti_throughput': '10000 Mbps',
            'switch_efficiency': '99.9%',
            'relay_amplification': '+330%',
            'frequency_band': '8.0 GHz + P-Optical'
        },
        'plot': 'data:image/png;base64,' + img_b64
    })

if __name__ == '__main__':
    port = int(os.environ.get('FLASK_RUN_PORT', 5070))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)