#!/usr/bin/env python3
"""Research-only mathematical visualization of hypothetical HIV cure concepts.

This software is not a clinical decision tool and does not predict patient outcomes.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template_string, request, send_from_directory

app = Flask(__name__)
OUTPUT_DIR = Path(__file__).parent / "hiv_research_outputs"

VARIANTS = {
    "High-fidelity editor concept": {"activity": 0.56, "specificity": 0.93},
    "Transient inhibitor-gated concept": {"activity": 0.48, "specificity": 0.96},
    "Delivery-limited combination concept": {"activity": 0.39, "specificity": 0.98},
}


def continued_fraction_factor(depth: int) -> float:
    """Bounded convergence modifier derived from a finite continued fraction."""
    value = 1.0
    for term in range(max(depth, 1), 0, -1):
        value = term + 1.0 / value
    return min(1.08, 0.92 + 0.03 * (value / (depth + 1)))


def calculate_projection(variant: str, sequencing_quality: float, conjugate_stability: float,
                         horizon_years: int, cf_depth: int) -> dict:
    settings = VARIANTS.get(variant, VARIANTS["High-fidelity editor concept"])
    years = np.linspace(0, horizon_years, 121)
    elliptic_scale = 0.90 + 0.08 * np.clip(conjugate_stability, 0, 1)
    convergence = continued_fraction_factor(cf_depth)
    base = settings["activity"] * sequencing_quality * elliptic_scale * convergence
    repair = 100 * (1 - np.exp(-base * years / max(horizon_years, 1)))
    uncertainty = 6 + 14 * (1 - settings["specificity"] * sequencing_quality)
    lower = np.maximum(0, repair - uncertainty * np.sqrt(years / max(horizon_years, 1)))
    upper = np.minimum(100, repair + uncertainty * np.sqrt(years / max(horizon_years, 1)))
    return {
        "years": years.round(2).tolist(),
        "repair": repair.round(2).tolist(),
        "lower": lower.round(2).tolist(),
        "upper": upper.round(2).tolist(),
        "summary": {
            "endpoint": round(float(repair[-1]), 1),
            "interval_low": round(float(lower[-1]), 1),
            "interval_high": round(float(upper[-1]), 1),
            "specificity": settings["specificity"],
        },
    }


def write_plots() -> list[str]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    projection = calculate_projection("High-fidelity editor concept", 0.88, 0.76, 20, 8)
    years = np.array(projection["years"])
    repair = np.array(projection["repair"])
    lower = np.array(projection["lower"])
    upper = np.array(projection["upper"])
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.fill_between(years, lower, upper, color="#e08e45", alpha=0.25, label="Illustrative uncertainty band")
    axis.plot(years, repair, color="#0a6f73", linewidth=2.5, label="Model trajectory")
    axis.set(xlabel="Long-term horizon (years)", ylabel="Illustrative repair index (%)", ylim=(0, 100),
             title="Hypothetical repair-index trajectory")
    axis.legend(frameon=False)
    figure.tight_layout()
    trajectory_path = OUTPUT_DIR / "long_term_repair_horizon.png"
    figure.savefig(trajectory_path, dpi=180)
    plt.close(figure)

    rng = np.random.default_rng(42)
    values = rng.beta(8, 5, 3000) * 100
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(values, bins=34, density=True, color="#0a6f73", alpha=0.85)
    axis.set(xlabel="Conjugate-associated response index (%)", ylabel="Density",
             title="Illustrative beta distribution for conjugate variability")
    figure.tight_layout()
    distribution_path = OUTPUT_DIR / "conjugate_distribution.png"
    figure.savefig(distribution_path, dpi=180)
    plt.close(figure)
    return [trajectory_path.name, distribution_path.name]


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>HIV Repair Research Simulator</title><script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
:root{--ink:#17323a;--teal:#0a6f73;--orange:#d36a2d;--paper:#f8f5ee;--line:#c8d3cf}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:'DM Mono',monospace}.shell{max-width:1180px;margin:auto;padding:34px 22px 50px}header{border-bottom:2px solid var(--ink);padding-bottom:24px;margin-bottom:26px}.eyebrow{color:var(--orange);font-size:12px;text-transform:uppercase;letter-spacing:1px}h1,h2{font-family:Fraunces,serif;margin:8px 0}h1{font-size:clamp(30px,5vw,52px);line-height:1.02}h2{font-size:24px}.notice{background:#fff0dd;border-left:5px solid var(--orange);padding:14px 16px;line-height:1.55;font-size:13px;margin:20px 0}.layout{display:grid;grid-template-columns:320px 1fr;gap:26px}.panel{border:1px solid var(--line);padding:18px;background:#fffdf8}label{display:block;font-size:12px;margin:16px 0 6px}select,input{width:100%;font:inherit;border:1px solid var(--ink);background:#fffdf8;padding:10px}button{margin-top:22px;width:100%;padding:12px;background:var(--teal);color:white;border:0;font:inherit;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:12px 0 22px}.metric{border-top:3px solid var(--orange);padding:10px 0}.metric b{font-family:Fraunces,serif;font-size:28px;display:block}.metric span{font-size:11px}canvas{max-height:350px}.math{font-family:Georgia,serif;font-size:16px;line-height:1.8;border-top:1px solid var(--line);margin-top:25px;padding-top:15px}@media(max-width:760px){.layout{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}}
</style></head><body><main class="shell"><header><div class="eyebrow">Exploratory computational biology</div><h1>HIV Repair Research Simulator</h1><div>Finite mathematical scenarios for sequencing, hypothetical editor/inhibitor concepts, and biomolecular conjugate variability.</div></header>
<div class="notice"><b>Research-only simulation.</b> Values are synthetic model outputs, not clinical predictions, treatment advice, or evidence of an HIV cure. No experiment design, dosing, or sequence-level intervention guidance is provided.</div>
<section class="layout"><form class="panel" id="controls"><h2>Scenario inputs</h2><label>Conceptual variant</label><select id="variant">{% for option in variants %}<option>{{option}}</option>{% endfor %}</select><label>Sequencing confidence (0.5-1.0)</label><input id="sequencing" type="number" value="0.88" min="0.5" max="1" step="0.01"><label>Conjugate stability (0-1)</label><input id="stability" type="number" value="0.76" min="0" max="1" step="0.01"><label>Long-term horizon (years)</label><input id="horizon" type="number" value="20" min="1" max="50"><label>Continued-fraction depth</label><input id="depth" type="number" value="8" min="1" max="30"><button>Update illustrative projection</button></form>
<div><div class="metrics"><div class="metric"><b id="endpoint">--</b><span>endpoint index</span></div><div class="metric"><b id="interval">--</b><span>illustrative interval</span></div><div class="metric"><b id="specificity">--</b><span>concept specificity</span></div></div><div class="panel"><canvas id="chart"></canvas><div class="math"><b>Finite model notation</b><br>R(t) = 100[1 - exp(-&alpha;t / H)]<br>&alpha; = A &middot; q &middot; E(k) &middot; C<sub>n</sub><br>C<sub>n</sub> = a<sub>1</sub> + 1/(a<sub>2</sub> + ... + 1/a<sub>n</sub>)</div></div></div></section></main>
<script>let chart;async function update(e){if(e)e.preventDefault();const p=new URLSearchParams({variant:variant.value,sequencing:sequencing.value,stability:stability.value,horizon:horizon.value,depth:depth.value});const r=await fetch('/api/projection?'+p);const d=await r.json();endpoint.textContent=d.summary.endpoint+'%';interval.textContent=d.summary.interval_low+'-'+d.summary.interval_high+'%';specificity.textContent=(d.summary.specificity*100).toFixed(0)+'%';if(chart)chart.destroy();chart=new Chart(document.getElementById('chart'),{type:'line',data:{labels:d.years,datasets:[{label:'Upper bound',data:d.upper,borderColor:'transparent',backgroundColor:'rgba(224,142,69,.25)',fill:'+1',pointRadius:0},{label:'Lower bound',data:d.lower,borderColor:'transparent',backgroundColor:'rgba(224,142,69,.25)',fill:false,pointRadius:0},{label:'Illustrative repair index',data:d.repair,borderColor:'#0a6f73',borderWidth:3,pointRadius:0}]},options:{responsive:true,plugins:{legend:{labels:{font:{family:'DM Mono'}}}},scales:{x:{title:{display:true,text:'Years'}},y:{min:0,max:100,title:{display:true,text:'Index (%)'}}}}})}controls.addEventListener('submit',update);update();</script></body></html>"""


@app.route("/")
def index():
    return render_template_string(PAGE, variants=VARIANTS)


@app.route("/api/projection")
def projection():
    return jsonify(calculate_projection(
        request.args.get("variant", "High-fidelity editor concept"),
        float(request.args.get("sequencing", 0.88)), float(request.args.get("stability", 0.76)),
        int(request.args.get("horizon", 20)), int(request.args.get("depth", 8)),
    ))


@app.route("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUTPUT_DIR, name)


if __name__ == "__main__":
    write_plots()
    app.run(host="127.0.0.1", port=5057, debug=False)