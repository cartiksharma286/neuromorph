// Vector Field | Infectious Disease Convergence & Health Economics Lab
const defaults = {
  // Disease Progression
  transmission: 1.35,
  intervention: 0.48,
  diffusion: 0.42,
  phase_shift: 1.20,
  repair_capacity: 0.58,
  // Health Economics
  cohort_size: 250000,
  horizon_years: 5,
  intervention_cost: 350,
  hospitalization_cost: 18500,
  reimbursement_rate: 0.65,
  learning_regularizer: 0.25,
  bioremediation_adoption: 0.45,
};

let currentTab = 'progression';
let cachedData = null;

const colors = {
  susceptible: '#48d2b5',
  exposed: '#edc965',
  infectious: '#ee8470',
  recovered: '#75b9e8',
  teal: '#48d2b5',
  yellow: '#edc965',
  coral: '#ee8470',
  blue: '#75b9e8',
  purple: '#b388ff',
  line: '#38565a',
  text: '#8aa8a6',
  band: 'rgba(117, 185, 232, 0.22)',
};

function payload() {
  const data = {};
  for (const key of Object.keys(defaults)) {
    const el = document.getElementById(key);
    if (el) data[key] = +el.value;
  }
  return data;
}

function updateOutputs() {
  const p = payload();
  for (const [key, value] of Object.entries(p)) {
    const out = document.getElementById(`${key}-output`);
    if (!out) continue;
    if (['intervention', 'repair_capacity', 'reimbursement_rate', 'bioremediation_adoption'].includes(key)) {
      out.textContent = `${Math.round(value * 100)}%`;
    } else if (key === 'cohort_size') {
      out.textContent = Number(value).toLocaleString();
    } else if (key === 'horizon_years') {
      out.textContent = `${value} yrs`;
    } else if (['intervention_cost', 'hospitalization_cost'].includes(key)) {
      out.textContent = `$${Number(value).toLocaleString()}`;
    } else {
      out.textContent = value.toFixed(2);
    }
  }
}

function rows(items, renderFn) {
  return items.map(renderFn).join('');
}

function drawLineChart(id, sets, { max = 100, min = 0, yUnit = '', xLabels = [] } = {}) {
  const canvas = document.getElementById(id);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const box = canvas.getBoundingClientRect();
  if (box.width === 0 || box.height === 0) return;
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, box.width);
  const height = Math.max(1, box.height);
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);

  const p = { l: 40, r: 12, t: 10, b: 24 };
  ctx.strokeStyle = colors.line;
  ctx.lineWidth = 1;

  for (let j = 0; j < 4; j++) {
    const y = p.t + ((height - p.t - p.b) * j) / 3;
    ctx.beginPath();
    ctx.moveTo(p.l, y);
    ctx.lineTo(width - p.r, y);
    ctx.stroke();
  }

  ctx.fillStyle = colors.text;
  ctx.font = '9px ui-monospace';
  ctx.textAlign = 'right';
  ctx.fillText(`${min}${yUnit}`, p.l - 6, height - p.b);
  ctx.fillText(`${max}${yUnit}`, p.l - 6, p.t + 8);

  const range = Math.max(1e-5, max - min);

  for (const set of sets) {
    if (!set.values || set.values.length < 2) continue;
    ctx.strokeStyle = set.color;
    ctx.lineWidth = set.lineWidth || 2;
    if (set.dashed) ctx.setLineDash([4, 3]);
    else ctx.setLineDash([]);

    ctx.beginPath();
    set.values.forEach((val, idx) => {
      const x = p.l + ((width - p.l - p.r) * idx) / (set.values.length - 1);
      const y = height - p.b - ((val - min) / range) * (height - p.t - p.b);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  if (xLabels.length > 0) {
    ctx.textAlign = 'center';
    xLabels.forEach((lbl, idx) => {
      const x = p.l + ((width - p.l - p.r) * idx) / (xLabels.length - 1);
      ctx.fillText(lbl, x, height - 6);
    });
  }
}

function drawConfidenceBandChart(id, proj) {
  const canvas = document.getElementById(id);
  if (!canvas || !proj || proj.length === 0) return;
  const ctx = canvas.getContext('2d');
  const box = canvas.getBoundingClientRect();
  if (box.width === 0 || box.height === 0) return;
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, box.width);
  const height = Math.max(1, box.height);
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);

  const p = { l: 48, r: 16, t: 14, b: 26 };
  const maxVal = Math.max(...proj.map(x => x.gross_upper_m)) * 1.12;

  ctx.strokeStyle = colors.line;
  ctx.lineWidth = 1;
  for (let j = 0; j < 4; j++) {
    const y = p.t + ((height - p.t - p.b) * j) / 3;
    ctx.beginPath();
    ctx.moveTo(p.l, y);
    ctx.lineTo(width - p.r, y);
    ctx.stroke();
  }

  ctx.fillStyle = colors.text;
  ctx.font = '9px ui-monospace';
  ctx.textAlign = 'right';
  ctx.fillText('$0M', p.l - 6, height - p.b);
  ctx.fillText(`$${maxVal.toFixed(1)}M`, p.l - 6, p.t + 8);

  ctx.fillStyle = colors.band;
  ctx.beginPath();
  proj.forEach((pt, idx) => {
    const x = p.l + ((width - p.l - p.r) * idx) / (proj.length - 1);
    const yUpper = height - p.b - (pt.gross_upper_m / maxVal) * (height - p.t - p.b);
    if (idx === 0) ctx.moveTo(x, yUpper);
    else ctx.lineTo(x, yUpper);
  });
  for (let idx = proj.length - 1; idx >= 0; idx--) {
    const pt = proj[idx];
    const x = p.l + ((width - p.l - p.r) * idx) / (proj.length - 1);
    const yLower = height - p.b - (pt.gross_lower_m / maxVal) * (height - p.t - p.b);
    ctx.lineTo(x, yLower);
  }
  ctx.closePath();
  ctx.fill();

  const drawLine = (key, color, dashed = false) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    if (dashed) ctx.setLineDash([4, 3]);
    else ctx.setLineDash([]);
    ctx.beginPath();
    proj.forEach((pt, idx) => {
      const x = p.l + ((width - p.l - p.r) * idx) / (proj.length - 1);
      const y = height - p.b - (pt[key] / maxVal) * (height - p.t - p.b);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  };

  drawLine('gross_revenue_m', colors.teal);
  drawLine('net_margin_m', colors.blue);
  drawLine('shared_savings_m', colors.yellow, true);

  ctx.fillStyle = colors.text;
  ctx.textAlign = 'center';
  proj.forEach((pt, idx) => {
    const x = p.l + ((width - p.l - p.r) * idx) / (proj.length - 1);
    ctx.fillText(`Yr ${pt.year}`, x, height - 6);
  });
}

function drawPayerMixChart(id, payerMix) {
  const canvas = document.getElementById(id);
  if (!canvas || !payerMix) return;
  const ctx = canvas.getContext('2d');
  const box = canvas.getBoundingClientRect();
  if (box.width === 0 || box.height === 0) return;
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, box.width);
  const height = Math.max(1, box.height);
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);

  const p = { l: 20, r: 20, t: 15, b: 20 };
  const palette = [colors.teal, colors.blue, colors.yellow, colors.purple];
  const barHeight = 22;
  const spacing = 18;

  payerMix.forEach((item, idx) => {
    const y = p.t + idx * (barHeight + spacing);
    const maxBarW = Math.max(20, width - p.l - p.r - 80);
    const barW = Math.max(4, (item.share_pct / 50) * maxBarW);

    ctx.fillStyle = colors.text;
    ctx.font = '9px ui-monospace';
    ctx.textAlign = 'left';
    ctx.fillText(item.segment, p.l, y - 4);

    ctx.textAlign = 'right';
    ctx.fillText(`${item.share_pct}% ($${item.revenue_m}M)`, width - p.r, y + 14);

    ctx.fillStyle = 'rgba(35,66,71,0.5)';
    ctx.fillRect(p.l, y, maxBarW, barHeight);

    ctx.fillStyle = palette[idx % palette.length];
    ctx.fillRect(p.l, y, barW, barHeight);
  });
}

function drawLearningCurveChart(id, points) {
  const canvas = document.getElementById(id);
  if (!canvas || !points || points.length === 0) return;
  const ctx = canvas.getContext('2d');
  const box = canvas.getBoundingClientRect();
  if (box.width === 0 || box.height === 0) return;
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, box.width);
  const height = Math.max(1, box.height);
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);

  const p = { l: 45, r: 16, t: 12, b: 24 };
  const costs = points.map(x => x.unit_cost);
  const minCost = Math.min(...costs) * 0.85;
  const maxCost = Math.max(...costs) * 1.12;
  const range = Math.max(1, maxCost - minCost);

  ctx.strokeStyle = colors.line;
  ctx.lineWidth = 1;
  for (let j = 0; j < 4; j++) {
    const y = p.t + ((height - p.t - p.b) * j) / 3;
    ctx.beginPath();
    ctx.moveTo(p.l, y);
    ctx.lineTo(width - p.r, y);
    ctx.stroke();
  }

  ctx.fillStyle = colors.text;
  ctx.font = '9px ui-monospace';
  ctx.textAlign = 'right';
  ctx.fillText(`$${Math.round(minCost)}`, p.l - 6, height - p.b);
  ctx.fillText(`$${Math.round(maxCost)}`, p.l - 6, p.t + 8);

  ctx.strokeStyle = colors.coral;
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  points.forEach((pt, idx) => {
    const x = p.l + ((width - p.l - p.r) * idx) / (points.length - 1);
    const y = height - p.b - ((pt.unit_cost - minCost) / range) * (height - p.t - p.b);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = colors.coral;
  points.forEach((pt, idx) => {
    const x = p.l + ((width - p.l - p.r) * idx) / (points.length - 1);
    const y = height - p.b - ((pt.unit_cost - minCost) / range) * (height - p.t - p.b);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = colors.text;
  ctx.textAlign = 'center';
  points.forEach((pt, idx) => {
    const x = p.l + ((width - p.l - p.r) * idx) / (points.length - 1);
    ctx.fillText(`${pt.volume_k}k`, x, height - 6);
  });
}

function render(data) {
  cachedData = data;
  const h = data.headline;

  document.getElementById('effective-r').textContent = h.effective_r.toFixed(2);
  document.getElementById('peak-infectious').textContent = `${h.peak_infectious_pct.toFixed(1)}%`;
  document.getElementById('residual').textContent = h.terminal_residual.toFixed(3);
  document.getElementById('readiness').textContent = `${h.governance_readiness}%`;

  if (currentTab === 'progression') {
    drawLineChart('trajectory-chart', [
      { color: colors.susceptible, values: data.trajectory.map(x => x.susceptible) },
      { color: colors.exposed, values: data.trajectory.map(x => x.exposed) },
      { color: colors.infectious, values: data.trajectory.map(x => x.infectious) },
      { color: colors.recovered, values: data.trajectory.map(x => x.recovered) },
    ]);
    drawLineChart('convergence-chart', [{ color: colors.yellow, values: data.convergence.map(x => x.residual) }], { max: 0.8 });
    drawLineChart('phase-chart', [
      { color: colors.blue, values: data.phase_sifting.map(x => (x.amplitude + 1) / 2) },
      { color: colors.teal, values: data.phase_sifting.map(x => x.sift) },
    ], { max: 1 });

    const math = data.finite_math;
    document.getElementById('elliptic').textContent = `K(k) = ${math.elliptic_projection_k}`;
    document.getElementById('math-content').innerHTML = `
      <div class="row"><span>Pigeonhole occupancy</span><b>${math.regional_occupancy.join(' · ')}</b></div>
      <div class="row"><span>Capacity collisions</span><b>${math.overflow_collisions}</b></div>
      <div class="row"><span>Prime signatures</span><b>${math.leibniz_prime_signatures.map(x => x.prime).join(', ')}</b></div>
    `;
    document.getElementById('transitions').innerHTML = rows(data.state_transitions, x => `
      <div class="row"><span>${x.from} → ${x.to}</span><b>${x.driver}</b></div>
    `);
    document.getElementById('governance').innerHTML = rows(data.governance, x => `
      <div class="row"><span>${x.name}</span><b class="tag ${x.state.toLowerCase()}">${x.state} ${x.score}</b></div>
    `);
  }

  if (data.health_economics) {
    const econ = data.health_economics;
    const eh = econ.headline;
    document.getElementById('econ-gross-rev').textContent = `$${eh.cumulative_gross_revenue_m.toFixed(1)}M`;
    document.getElementById('econ-npv').textContent = `$${eh.net_present_value_m.toFixed(1)}M`;
    document.getElementById('econ-r2').textContent = `${(eh.statistical_r2 * 100).toFixed(1)}%`;
    document.getElementById('econ-icer').textContent = `$${eh.icer_per_qaly.toLocaleString()}`;
    document.getElementById('econ-horizon-span').textContent = `${econ.parameters.horizon_years}-Year Ridge Estimator (λ=${econ.parameters.learning_regularizer})`;

    if (currentTab === 'economics') {
      drawConfidenceBandChart('revenue-chart', econ.annual_projections);
      drawPayerMixChart('payer-chart', econ.payer_mix);
      drawLearningCurveChart('learning-chart', econ.learning_curve);

      const sm = econ.statistical_model;
      document.getElementById('econ-cond-num').textContent = `Cond: ${sm.condition_number}`;
      document.getElementById('econ-model-specs').innerHTML = `
        <div class="row"><span>Estimator Type</span><b>Ridge (L2 Closed Form)</b></div>
        <div class="row"><span>Penalty λ</span><b>${sm.ridge_lambda}</b></div>
        <div class="row"><span>Training RMSE</span><b>${sm.rmse}</b></div>
        <div class="row"><span>Learned Weights</span><b>[${sm.weights.join(', ')}]</b></div>
        <div class="row"><span>Features</span><b style="font-size:8.5px;">${sm.features.join(' · ')}</b></div>
      `;

      const ledgerRows = econ.annual_projections.map(r => `
        <tr>
          <td><b>Yr ${r.year}</b></td>
          <td>${(r.enrolled / 1000).toFixed(0)}k</td>
          <td style="color:${colors.teal}; font-weight:bold;">$${r.gross_revenue_m.toFixed(1)}M</td>
          <td style="color:${colors.blue};">$${r.gross_lower_m.toFixed(1)}–${r.gross_upper_m.toFixed(1)}M</td>
          <td style="color:${colors.blue};">$${r.net_margin_m.toFixed(1)}M</td>
          <td style="color:${colors.yellow};">$${r.shared_savings_m.toFixed(1)}M</td>
          <td>${r.admissions_averted}</td>
          <td>${r.qalys_gained.toFixed(0)}</td>
        </tr>
      `).join('');

      document.getElementById('econ-ledger').innerHTML = `
        <table class="econ-table">
          <thead>
            <tr><th>Yr</th><th>Enrolled</th><th>Gross</th><th>95% CI</th><th>Net</th><th>Savings</th><th>Averted</th><th>QALY</th></tr>
          </thead>
          <tbody>${ledgerRows}</tbody>
        </table>
      `;

      document.getElementById('econ-governance').innerHTML = `
        <div class="row"><span>Capitation Risk Readiness</span><b class="tag ready">READY 88</b></div>
        <div class="row"><span>CMS Shared-Savings Audit</span><b class="tag active">ACTIVE 94</b></div>
        <div class="row"><span>Equity-Weighted Pricing Tier</span><b class="tag active">ACTIVE 91</b></div>
        <div class="row"><span>Bioremediation Telemetry</span><b class="tag monitor">MONITOR 76</b></div>
      `;
    }
  }
}

async function run() {
  updateOutputs();
  try {
    const response = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload()),
    });
    if (response.ok) {
      render(await response.json());
    }
  } catch (err) {
    console.error('Simulation run failed:', err);
  }
}

function switchTab(tab) {
  currentTab = tab;
  const btnProg = document.getElementById('tab-btn-progression');
  const btnEcon = document.getElementById('tab-btn-economics');
  const ctrlProg = document.getElementById('controls-progression');
  const ctrlEcon = document.getElementById('controls-economics');
  const viewProg = document.getElementById('view-progression');
  const viewEcon = document.getElementById('view-economics');

  const eyebrow = document.getElementById('hero-eyebrow');
  const title = document.getElementById('hero-title');
  const desc = document.getElementById('hero-desc');
  const ctrlTitle = document.getElementById('controls-title');

  if (tab === 'progression') {
    btnProg.classList.add('active');
    btnEcon.classList.remove('active');
    ctrlProg.classList.add('active');
    ctrlEcon.classList.remove('active');
    viewProg.classList.add('active');
    viewEcon.classList.remove('active');

    eyebrow.textContent = 'FINITE MATHEMATICS / POPULATION STATE MODELS';
    title.innerHTML = 'Progression signals, <em>made inspectable.</em>';
    desc.textContent = 'Scenario exploration using combinatoric capacity checks, Leibniz recurrent-prime signatures, elliptic projections, interferometric diffusion and phase sifting.';
    ctrlTitle.textContent = 'Scenario controls';
  } else {
    btnEcon.classList.add('active');
    btnProg.classList.remove('active');
    ctrlEcon.classList.add('active');
    ctrlProg.classList.remove('active');
    viewEcon.classList.add('active');
    viewProg.classList.remove('active');

    eyebrow.textContent = 'STATISTICAL LEARNING / VALUE-BASED HEALTH ECONOMICS';
    title.innerHTML = 'Long-term revenue, <em>risk-quantified.</em>';
    desc.textContent = 'Value-based healthcare cash flow modeling, regularized ridge parameter estimation with 95% predictive confidence intervals, and Wright\'s law operational learning curves.';
    ctrlTitle.textContent = 'Economic parameters';
  }

  if (cachedData) {
    setTimeout(() => render(cachedData), 30);
  }
}

document.getElementById('tab-btn-progression').addEventListener('click', () => switchTab('progression'));
document.getElementById('tab-btn-economics').addEventListener('click', () => switchTab('economics'));

for (const key of Object.keys(defaults)) {
  const el = document.getElementById(key);
  if (el) el.addEventListener('input', run);
}

document.getElementById('run').addEventListener('click', run);
document.getElementById('reset').addEventListener('click', () => {
  for (const [key, value] of Object.entries(defaults)) {
    const el = document.getElementById(key);
    if (el) el.value = value;
  }
  run();
});

window.addEventListener('resize', () => {
  if (cachedData) render(cachedData);
});

run();

