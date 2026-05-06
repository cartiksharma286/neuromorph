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
    # Performance benchmarks: Huawei 10G Mesh (BE3 Pro / OptiXstar Series, ~10000 Mbps, 5GHz/6GHz)
    # Target Optimal Design: QuantumMesh 10G (Quantum Tunneling + Optical Switches + 10GHz Relay, ~12000+ Mbps)
    
    time_bins = np.linspace(0, 10, 1000)
    
    # Statistical Distributions for Wave Packets
    # Base packet arrival (Poisson-like envelope modeling heavy traffic on Huawei 10G Mesh)
    base_traffic = np.random.poisson(lam=80, size=1000)
    
    # 6GHz Wave Packet Dispersion (Standard fading model using Gaussian decay for Huawei 10G)
    base_dispersion = np.exp(-0.5 * ((time_bins - 5.5) / 1.0)**2) * base_traffic
    
    # QuantumMesh 10G Wave Packet Dispersion (Quantum tunneling probability distributions)
    # Tunnels drastically reduce wave dispersion over time
    opt_traffic = np.random.poisson(lam=180, size=1000)
    opt_dispersion = np.exp(-0.5 * ((time_bins - 4.5) / 0.2)**2) * opt_traffic
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Wave Packet Distributions
    ax1.plot(time_bins, base_dispersion, label='Huawei 10G Mesh (6GHz)', color='#ef4444', alpha=0.7)
    ax1.plot(time_bins, opt_dispersion, label='QuantumMesh 10G (Tunneling)', color='#a855f7', alpha=0.9, lw=2)
    ax1.set_title("Wave Packet Distribution (Dispersion via Tunneling)")
    ax1.set_xlabel("Time Relay Inter-Arrival (ns)")
    ax1.set_ylabel("Packet Amplitude / Density")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    ax1.set_facecolor('#0f172a')
    
    # Plot 2: Optical Switching Network Optimization vs Distance
    distances = np.linspace(0, 120, 100) # distance in meters
    # Huawei 10G Mesh standard attenuation (handles distances better than old Wi-Fi 6, but still drops)
    base_throughput = 10000 * np.exp(-distances / 40) 
    # Quantum tunneling maintains non-local signal cohesion practically ignoring initial barrier attenuation
    optical_throughput = 12500 * np.exp(-(distances / 100)**1.5)
    
    ax2.plot(distances, base_throughput, label='Huawei 10G Mesh', color='#ef4444', lw=2)
    ax2.plot(distances, optical_throughput, label='QuantumMesh 10G', color='#a855f7', lw=2)
    ax2.set_title("Quantum Tunneling Communications vs Signal Attenuation")
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
            'x3_pro_latency': '2.1 ms (Huawei 10G)',
            'opti_latency': '0.08 ms (Quantum)',
            'x3_pro_throughput': '10000 Mbps',
            'opti_throughput': '12500 Mbps',
            'switch_efficiency': '99.999% (Tunneling)',
            'relay_amplification': '+450%',
            'frequency_band': '10.0 GHz + Quantum Tunnels'
        },
        'plot': 'data:image/png;base64,' + img_b64
    })

if __name__ == '__main__':
    port = int(os.environ.get('FLASK_RUN_PORT', 5070))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)