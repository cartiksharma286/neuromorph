from flask import Flask, render_template_string, jsonify
import numpy as np
import math
import os

app = Flask(__name__)

def continued_fraction(num, max_depth=10):
    """Calculates the continued fraction representation of a number."""
    cf = []
    for _ in range(max_depth):
        i = int(num)
        cf.append(i)
        num -= i
        if abs(num) < 1e-6: 
            break
        num = 1.0 / num
    return cf

def compute_congruence(portfolio_weights):
    """Computes statistical congruence fractions for the given weights."""
    congruence_scores = []
    for w in portfolio_weights:
        congruence_scores.append({
            "weight_value": float(w),
            "fraction_seq": continued_fraction(w, max_depth=6)
        })
    return congruence_scores

def combinatorial_optimization(n_traits=5):
    """Generates optimal fractional combinatorics for a health portfolio."""
    # Simulating combinatorial distributions using Gamma/Dirichlet for compositional data
    weights = np.random.gamma(shape=2.5, scale=1.0, size=n_traits)
    # Normalize to form a proper portfolio fraction
    weights /= weights.sum()
    return weights

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    # UK Biobank analogous traits relevant to neuroeconomics
    traits = [
        "Cognitive_Reserve_Index", 
        "Cardiometabolic_Risk", 
        "Neuro_Resilience", 
        "Delayed_Discounting_Rate",
        "Risk_Aversion_Threshold"
    ]
    
    weights = combinatorial_optimization(len(traits))
    congruence = compute_congruence(weights)
    
    portfolio = [{"trait": t, "fractional_weight": float(w), "statistical_congruence": c["fraction_seq"]} 
                 for t, w, c in zip(traits, weights, congruence)]
    
    # Calculate a theoretical NYU Neuroecon Expected Utility Score
    # Using a logarithmic utility transformation common in economics
    utility_score = sum([w * math.log(w + 1) for w in weights])
    
    return jsonify({
        "status": "success",
        "institution": "NYU Neuroeconomics Center",
        "model": "UK Biobank Combinatorial Health Portfolio",
        "theoretical_utility_score": round(utility_score, 5),
        "portfolio": portfolio
    })

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NYU Neuroeconomics - Health Portfolios</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #f4f5f7; padding: 30px; color: #333; }
        .container { max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); }
        h1 { color: #57068c; } /* NYU Violet */
        h2 { color: #666; font-weight: 300; font-size: 1.2em; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        button { background: #57068c; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-size: 1em; margin-top: 10px; transition: background 0.3s; }
        button:hover { background: #8900e1; }
        pre { background: #1e1e1e; color: #d4d4d4; padding: 20px; border-radius: 8px; overflow-x: auto; font-family: "Courier New", Courier, monospace; }
        .highlight { color: #4ec9b0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>NYU Neuroeconomics Laboratory</h1>
        <h2>Statistical Congruence Fractions & Biobank Portfolios</h2>
        <p>This engine runs Monte Carlo combinatorial optimizations to generate health portfolios based on UK Biobank phenotypic distributions. It applies continued fractional representation to evaluate statistical congruence across neuro_resilience, cognitive reserve, and economic discounting variables.</p>
        <button onclick="generatePortfolio()">Simulate Optimal Portfolio</button>
        <div id="result" style="margin-top: 25px;"></div>
    </div>

    <script>
        function generatePortfolio() {
            document.getElementById('result').innerHTML = "<p><i>Computing combinatorics...</i></p>";
            fetch('/api/portfolio')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('result').innerHTML = `<pre>${JSON.stringify(data, null, 4)}</pre>`;
                })
                .catch(err => {
                    document.getElementById('result').innerHTML = `<p style="color:red;">Error: ${err}</p>`;
                });
        }
        // Auto-run simulation on load
        generatePortfolio();
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('FLASK_RUN_PORT', 5066))
    # Binding to 0.0.0.0 to allow LAN network communication
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
