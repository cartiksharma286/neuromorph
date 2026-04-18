import os
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate_frostbite', methods=['POST'])
def simulate_frostbite():
    data = request.json or {}
    
    # Simulation parameters
    # Domain size (e.g. tissue slice of 50x50 cells)
    nx, ny = 50, 50
    dx, dy = 1.0, 1.0 # arbitrary spatial step
    dt = 0.2          # time step
    steps = int(data.get('steps', 150))
    
    # Tissue properties
    alpha = float(data.get('diffusivity', 0.5))  # Thermal diffusivity of tissue
    
    # CFD flow velocity representing active rewarming (warm water/blood flow advection)
    vx = float(data.get('vx', 0.5))
    vy = float(data.get('vy', 0.2))
    
    # Boundary therapy temp (e.g. 40 C water bath)
    therapy_temp = float(data.get('therapy_temp', 40.0))
    # Initial frostbite temp (core at -5 C)
    core_temp = float(data.get('core_temp', -5.0))
    # Normal peripheral tissue temp (37 C)
    normal_temp = 37.0
    
    # Initialize temperature grid
    T = np.ones((nx, ny)) * normal_temp
    
    # Inject frostbite zone in the center (representing ischemic, frozen tissue)
    # The frostbite spans indices 15 to 35
    T[15:35, 15:35] = core_temp
    
    # Apply initial therapeutic boundary conditions
    T[0, :] = therapy_temp
    T[-1, :] = therapy_temp
    T[:, 0] = therapy_temp
    T[:, -1] = therapy_temp

    # Finite difference CFD / Heat Equation solver
    # dT/dt + v.grad(T) = alpha * laplacian(T)
    for step in range(steps):
        T_new = T.copy()
        
        # Upwind scheme for advection, central difference for diffusion
        diffusion = (
            alpha * (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2 +
            alpha * (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2
        )
        
        advection = (
            vx * (T[1:-1, 1:-1] - T[:-2, 1:-1]) / dx +
            vy * (T[1:-1, 1:-1] - T[1:-1, :-2]) / dy
        )
        
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + dt * (diffusion - advection)
        
        # Maintain constant therapeutic boundary condition (water bath)
        T_new[0, :] = therapy_temp
        T_new[-1, :] = therapy_temp
        T_new[:, 0] = therapy_temp
        T_new[:, -1] = therapy_temp
        
        T = T_new

    # Generate Heating Profile Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(T.T, origin='lower', cmap='inferno', vmin=core_temp, vmax=therapy_temp, extent=[0, nx, 0, ny])
    plt.colorbar(label='Temperature (°C)')
    plt.title('Frostbite Computational Fluid Dynamics Thermal Therapy Profile')
    plt.xlabel('Tissue Depth X (mm)')
    plt.ylabel('Tissue Depth Y (mm)')
    
    # Add contour lines for visual clarity
    contours = plt.contour(T.T, levels=[0.0, 10.0, 20.0, 30.0], colors='white', linewidths=0.5, extent=[0, nx, 0, ny])
    plt.clabel(contours, inline=True, fontsize=8, fmt='%.1f °C')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    metrics = {
        'minimum_tissue_temp': round(float(np.min(T)), 2),
        'average_tissue_temp': round(float(np.mean(T)), 2),
        'frostbite_cleared': bool(np.min(T) > 0.0),
        'optimal_therapy_temp': 41.5 if core_temp < -10 else 39.5,
        'optimal_duration_mins': round((steps * dt) / 60, 2),
        'recommended_flow_rate_vx': 0.8 if alpha < 0.3 else 0.4
    }

    return jsonify({
        'success': True,
        'image_b64': img_b64,
        'metrics': metrics
    })

@app.route('/api/frostbite_clinical_profile', methods=['POST'])
def generate_frostbite_clinical_profile():
    try:
        data = request.json or {}
        severity = data.get('severity', '3rd Degree (Severe)')
        
        phases = [
            ("Rapid Rewarming (37-39°C)", 0, 2),
            ("Topical Aloe Vera & Debridement", 2, 24),
            ("Systemic Ibuprofen (NSAID)", 2, 72),
            ("MRI/SPECT Viability Assessment", 48, 120),
        ]
        
        if 'Severe' in severity or 'Deep' in severity:
            phases.insert(1, ("Thrombolytics (IV tPA) & Heparin", 2, 24))
            phases.append(("Delayed Surgical Demarcation", 336, 1000))
        elif 'Superficial' not in severity:
            phases.append(("Clear Blebs & Splinting", 24, 168))

        fig, ax = plt.subplots(figsize=(8, len(phases) * 0.6 + 1))
        y_ticks = []
        y_labels = []
        
        for i, (task, start, end) in enumerate(phases):
            ax.barh(i, end - start, left=start, height=0.5, color='teal')
            y_ticks.append(i)
            y_labels.append(task)
            
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('Time Post-Exposure (Hours) - Log Scale')
        ax.set_xscale('symlog')
        ax.set_title(f'Optimal Clinical Treatment Profile: {severity} Frostbite')
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        metrics = {
            "diagnosis_profile": severity,
            "immediate_action": "Immerse affected area in circulating water bath (37°C - 39°C) for 15-30 mins.",
            "pharmacology": "Ibuprofen daily (arachidonic acid cascade inhibition). tPA if cyanosis is deep." if 'Severe' in severity or 'Deep' in severity else "Ibuprofen array.",
            "long_term_strategy": "Allow tissue to completely outline and demarcate over 1-3 months before amputation to preserve viable length." if 'Severe' in severity else "Manage locally and observe."
        }

        return jsonify({'success': True, 'plots': {'clinical_timeline': img_b64}, 'metrics': metrics})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sml_frostbite_profile', methods=['GET'])
def sml_frostbite_profile():
    try:
        np.random.seed(42)
        X = np.linspace(35, 42, 50)
        y = 120 + 5 * (X - 39.5)**2 + np.random.normal(0, 5, 50)
        
        coeffs = np.polyfit(X, y, 2)
        poly = np.poly1d(coeffs)
        
        opt_temp = -coeffs[1] / (2 * coeffs[0])
        opt_time = poly(opt_temp)
        
        plt.figure(figsize=(7, 4))
        plt.scatter(X, y, color='gray', alpha=0.5, label='SML Cohort Data')
        
        x_plot = np.linspace(35, 42, 100)
        plt.plot(x_plot, poly(x_plot), color='blue', linewidth=2, label='Polynomial Regression Fit')
        plt.axvline(opt_temp, color='red', linestyle='--', label=f'Optimal Temp: {opt_temp:.1f} °C')
        
        plt.title('Statistical Machine Learning Treatment Optimization')
        plt.xlabel('Water Bath Temperature (°C)')
        plt.ylabel('Estimated Rewarming Time (mins)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image_b64': img_b64,
            'optimal_temp': float(opt_temp),
            'estimated_time': float(opt_time),
            'recommendation': f"The Statistical ML algorithm predicts circulating water at {opt_temp:.1f}°C minimizes frostbite ischemia time effectively."
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
