import json

app_file = 'app.py'

with open(app_file, 'r') as f:
    text = f.read()

# Add the new endpoint right before the first import numpy or before dementia_staging
new_route = """
@app.route('/api/stage-gated-protocol', methods=['GET'])
def stage_gated_protocol():
    # Generate stages using queueing theory (arrival rate of tau proteins, vs clearance rate)
    # Stage 1: Mild Cognitive Impairment, Stage 2: Moderate, Stage 3: Severe
    # M/M/1 Queue parameters
    lambda_arrival = [1.2, 2.5, 4.0] # Tau/A-beta aggregation rate
    mu_clearance = [1.5, 2.0, 1.8]   # Glymphatic clearance (enhanced by DBS)
    
    stages = []
    
    names = ["1. Early (MCI)", "2. Moderate Dementia", "3. Advanced Dementia"]
    voltages = [2.0, 3.5, 4.5]
    freqs = [130, 145, 185]
    pw = [60, 90, 120]
    
    for i in range(3):
        lam = lambda_arrival[i]
        mu = mu_clearance[i]
        rho = lam / mu if mu > 0 else float('inf') # traffic intensity
        
        # Queue metrics
        l_q = (lam**2) / (mu * (mu - lam)) if mu > lam else 999.9  # Unstable
        w_q = lam / (mu * (mu - lam)) if mu > lam else 99.9 # Unstable
        
        stages.append({
            "stage": i+1,
            "name": names[i],
            "electrical": {
                "voltage_v": voltages[i],
                "frequency_hz": freqs[i],
                "pulse_width_us": pw[i],
                "target": "Fornix / NBM"
            },
            "queueing": {
                "lambda_arrival": lam,
                "mu_clearance": mu,
                "rho_utilization": round(rho, 3),
                "l_q": round(l_q, 3) if isinstance(l_q, float) else l_q,
                "w_q_years": round(w_q, 3) if isinstance(w_q, float) else w_q
            },
            "desc": f"Stage {i+1} optimal DBS. emphasizes {'neuroprotection' if i==0 else 'network stabilization' if i==1 else 'symptom management'} given molecular aggregation queue profile."
        })
        
    return jsonify({"protocol": stages})

"""

target = "@app.route('/api/dementia-staging', methods=['POST'])"
if target in text:
    new_text = text.replace(target, new_route + target)
    with open(app_file, 'w') as f:
        f.write(new_text)
    print("app patched")
else:
    print("target not found")
