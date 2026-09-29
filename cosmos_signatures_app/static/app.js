const controlIds = [
  "mass", "spin", "inclination", "redshift", "modulus", "coupling",
  "ergodic_mixing", "combinatorial_budget", "game_weight",
  "hifu_freq", "hifu_pressure", "hifu_skull_comp", "hifu_cycles",
  "sr_energy", "sr_fork_gev", "sr_liv_coupling", "cbct_kvp", "cbct_cone_angle", "cf_order",
  "bb_prime_modulus", "bb_inflation_efolds", "bb_pbh_mass_log", "bb_pbh_fraction", "bb_quantum_coherence"
];
const defaults = {
  mass: 10, spin: 0.72, inclination: 55, redshift: 0.8, modulus: 0.68, coupling: 0.35,
  ergodic_mixing: 0.65, combinatorial_budget: 45, game_weight: 0.50,
  hifu_freq: 1.25, hifu_pressure: 0.85, hifu_skull_comp: 0.75, hifu_cycles: 100,
  sr_energy: 45.0, sr_fork_gev: 50.0, sr_liv_coupling: 0.35, cbct_kvp: 120, cbct_cone_angle: 14, cf_order: 6,
  bb_prime_modulus: 17, bb_inflation_efolds: 60, bb_pbh_mass_log: 11.0, bb_pbh_fraction: 0.04, bb_quantum_coherence: 0.65
};
const canvases = {
  hole: document.querySelector("#black-hole"),
  fork: document.querySelector("#fork-chart"),
  modular: document.querySelector("#modular-chart"),
  invDensity: document.querySelector("#invariant-density-chart"),
  subrings: document.querySelector("#subrings-chart"),
  pareto: document.querySelector("#pareto-chart"),
  uvCoverage: document.querySelector("#uv-coverage-chart"),
  reconImage: document.querySelector("#reconstructed-image-chart"),
  reconCross: document.querySelector("#reconstruction-cross-section-chart"),
  hifuFocus: document.querySelector("#hifu-focus-chart"),
  hifuSpectrum: document.querySelector("#hifu-spectrum-chart"),
  hifuClearance: document.querySelector("#hifu-clearance-chart"),
  srFork: document.querySelector("#sr-fork-chart"),
  cbctProfile: document.querySelector("#cbct-profile-chart"),
  maxwellJuttner: document.querySelector("#maxwell-juttner-chart"),
  stieltjesConvergence: document.querySelector("#stieltjes-convergence-chart"),
  bbCosmic: document.querySelector("#bb-cosmic-evolution-chart"),
  bbHawking: document.querySelector("#bb-hawking-radiation-chart"),
  bbPrimeModuli: document.querySelector("#bb-prime-moduli-chart"),
  bbQuantum: document.querySelector("#bb-quantum-spectrum-chart"),
};
let latestModel = null;
let requestToken = 0;
let activeView = "hole";

function readParameters() {
  return Object.fromEntries(controlIds.map((id) => [id, Number(document.getElementById(id).value)]));
}

function updateLabels(parameters) {
  document.querySelector("#mass-value").textContent = `${parameters.mass} M☉`;
  document.querySelector("#spin-value").textContent = parameters.spin.toFixed(2);
  document.querySelector("#inclination-value").textContent = `${parameters.inclination}°`;
  document.querySelector("#redshift-value").textContent = parameters.redshift.toFixed(1);
  document.querySelector("#modulus-value").textContent = parameters.modulus.toFixed(2);
  document.querySelector("#coupling-value").textContent = parameters.coupling.toFixed(2);
  if (document.querySelector("#ergodic_mixing-value")) {
    document.querySelector("#ergodic_mixing-value").textContent = parameters.ergodic_mixing.toFixed(2);
  }
  if (document.querySelector("#combinatorial_budget-value")) {
    document.querySelector("#combinatorial_budget-value").textContent = String(Math.round(parameters.combinatorial_budget));
  }
  if (document.querySelector("#game_weight-value")) {
    document.querySelector("#game_weight-value").textContent = parameters.game_weight.toFixed(2);
  }
  if (document.querySelector("#hifu_freq-value")) {
    document.querySelector("#hifu_freq-value").textContent = `${parameters.hifu_freq.toFixed(2)} MHz`;
  }
  if (document.querySelector("#hifu_pressure-value")) {
    document.querySelector("#hifu_pressure-value").textContent = `${parameters.hifu_pressure.toFixed(2)} MPa`;
  }
  if (document.querySelector("#hifu_skull_comp-value")) {
    document.querySelector("#hifu_skull_comp-value").textContent = parameters.hifu_skull_comp.toFixed(2);
  }
  if (document.querySelector("#hifu_cycles-value")) {
    document.querySelector("#hifu_cycles-value").textContent = String(Math.round(parameters.hifu_cycles));
  }
  if (document.querySelector("#sr_energy-value")) {
    document.querySelector("#sr_energy-value").textContent = `${parameters.sr_energy.toFixed(1)} GeV`;
  }
  if (document.querySelector("#sr_fork_gev-value")) {
    document.querySelector("#sr_fork_gev-value").textContent = `${parameters.sr_fork_gev.toFixed(1)} GeV`;
  }
  if (document.querySelector("#sr_liv_coupling-value")) {
    document.querySelector("#sr_liv_coupling-value").textContent = parameters.sr_liv_coupling.toFixed(2);
  }
  if (document.querySelector("#cbct_kvp-value")) {
    document.querySelector("#cbct_kvp-value").textContent = `${Math.round(parameters.cbct_kvp)} kVp`;
  }
  if (document.querySelector("#cbct_cone_angle-value")) {
    document.querySelector("#cbct_cone_angle-value").textContent = `${parameters.cbct_cone_angle.toFixed(1)}°`;
  }
  if (document.querySelector("#cf_order-value")) {
    document.querySelector("#cf_order-value").textContent = String(Math.round(parameters.cf_order));
  }
  if (document.querySelector("#bb_prime_modulus-value")) {
    document.querySelector("#bb_prime_modulus-value").textContent = `p = ${parameters.bb_prime_modulus}`;
  }
  if (document.querySelector("#bb_inflation_efolds-value")) {
    document.querySelector("#bb_inflation_efolds-value").textContent = `${Math.round(parameters.bb_inflation_efolds)} N`;
  }
  if (document.querySelector("#bb_pbh_mass_log-value")) {
    document.querySelector("#bb_pbh_mass_log-value").textContent = `10^${parameters.bb_pbh_mass_log.toFixed(1)} g`;
  }
  if (document.querySelector("#bb_pbh_fraction-value")) {
    document.querySelector("#bb_pbh_fraction-value").textContent = parameters.bb_pbh_fraction.toFixed(3);
  }
  if (document.querySelector("#bb_quantum_coherence-value")) {
    document.querySelector("#bb_quantum_coherence-value").textContent = parameters.bb_quantum_coherence.toFixed(2);
  }
}

function fitCanvas(canvas, ratio = 2) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return null;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const height = Math.max(170, rect.width / ratio);
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: rect.width, height };
}

function drawBlackHole(model) {
  const fitted = fitCanvas(canvases.hole, 1.8);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  const { spin, inclination_deg: inclination } = model.parameters;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#060b0b";
  ctx.fillRect(0, 0, width, height);

  for (let index = 0; index < 190; index += 1) {
    const x = ((index * 7919) % 1009) / 1009 * width;
    const y = ((index * 3571 + 137) % 1013) / 1013 * height;
    const alpha = 0.18 + ((index * 17) % 63) / 100;
    ctx.fillStyle = `rgba(213, 230, 213, ${alpha})`;
    ctx.fillRect(x, y, index % 13 === 0 ? 1.7 : 1, index % 13 === 0 ? 1.7 : 1);
  }

  const centerX = width * (0.51 - 0.055 * spin * Math.sin(inclination * Math.PI / 180));
  const centerY = height * 0.51;
  const perspective = 0.28 + 0.62 * Math.cos(inclination * Math.PI / 180);
  const diskRx = Math.min(width * 0.39, height * 0.70);
  const diskRy = diskRx * perspective;

  ctx.save();
  ctx.translate(centerX, centerY);
  ctx.rotate(-0.11 - spin * 0.09);
  ctx.scale(1, Math.max(0.3, perspective));
  for (let ring = 0; ring < 36; ring += 1) {
    const fraction = 1 - ring / 44;
    const rx = diskRx * fraction;
    const ry = diskRx * (0.18 + 0.055 * Math.sin(ring * 1.7)) * fraction;
    const start = 0.06 + ring * 0.22;
    const end = start + Math.PI * (1.08 + (ring % 4) * 0.18);
    const hue = ring % 3 === 0 ? "242, 133, 77" : ring % 3 === 1 ? "250, 185, 111" : "116, 199, 194";
    ctx.beginPath();
    ctx.ellipse(0, 0, rx, ry, 0, start, end);
    ctx.strokeStyle = `rgba(${hue}, ${0.13 + (1 - fraction) * 0.40})`;
    ctx.lineWidth = ring % 6 === 0 ? 2.2 : 1.1;
    ctx.shadowColor = `rgba(${hue}, .65)`;
    ctx.shadowBlur = ring % 5 === 0 ? 12 : 4;
    ctx.stroke();
  }
  ctx.restore();

  const horizon = Math.min(width, height) * (0.135 + 0.018 * spin);
  const ringGlow = ctx.createRadialGradient(centerX, centerY, horizon * 0.78, centerX, centerY, horizon * 1.28);
  ringGlow.addColorStop(0, "rgba(0,0,0,0)");
  ringGlow.addColorStop(0.68, "rgba(238, 157, 92, .52)");
  ringGlow.addColorStop(0.82, "rgba(139, 216, 208, .92)");
  ringGlow.addColorStop(1, "rgba(242, 139, 84, 0)");
  ctx.fillStyle = ringGlow;
  ctx.beginPath();
  ctx.ellipse(centerX, centerY, horizon * 1.45, horizon * 1.45 * perspective, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#010303";
  ctx.beginPath();
  ctx.ellipse(centerX, centerY, horizon, horizon * perspective, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(255, 206, 152, .82)";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.ellipse(centerX, centerY, horizon * 1.09, horizon * 1.09 * perspective, 0, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = "rgba(218, 231, 219, .72)";
  ctx.font = "10px SFMono-Regular, Consolas, monospace";
  ctx.fillText(`a* ${spin.toFixed(2)}   i ${inclination.toFixed(0)}°`, 16, height - 15);
  ctx.textAlign = "right";
  ctx.fillStyle = "rgba(150, 176, 158, .68)";
  ctx.fillText("ILLUSTRATIVE / NOT RAY-TRACED", width - 14, height - 15);
  ctx.textAlign = "left";
}

function plotBase(canvas, ratio = 2.05) {
  const fitted = fitCanvas(canvas, ratio);
  if (!fitted) return null;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);
  const margin = { left: 52, right: 22, top: 22, bottom: 38 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  ctx.strokeStyle = "rgba(166, 190, 170, .13)";
  ctx.lineWidth = 1;
  ctx.font = "10px SFMono-Regular, Consolas, monospace";
  ctx.fillStyle = "#829386";
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = plot.y + plot.h * tick / 4;
    ctx.beginPath();
    ctx.moveTo(plot.x, y);
    ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }
  for (let tick = 0; tick <= 4; tick += 1) {
    const x = plot.x + plot.w * tick / 4;
    ctx.beginPath();
    ctx.moveTo(x, plot.y);
    ctx.lineTo(x, plot.y + plot.h);
    ctx.stroke();
    ctx.textAlign = "center";
    ctx.fillText(String(tick * 2), x, height - 13);
  }
  ctx.textAlign = "left";
  ctx.fillText("z", plot.x + plot.w + 7, height - 13);
  return { ctx, width, height, plot };
}

function drawFork(model) {
  const fitted = plotBase(canvases.fork, 2.05);
  if (!fitted) return;
  const { ctx, plot } = fitted;
  const { toy_coupling: coupling, elliptic_modulus: modulus } = model.parameters;
  const amplitude = coupling * (2 * modulus - 1) * 0.12;
  const centerY = plot.y + plot.h / 2;
  ctx.strokeStyle = "#80c9cb";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(plot.x, centerY);
  ctx.lineTo(plot.x + plot.w, centerY);
  ctx.stroke();
  ctx.strokeStyle = "#f28b54";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  for (let step = 0; step <= 120; step += 1) {
    const z = 8 * step / 120;
    const residual = amplitude * (1 - Math.exp(-z));
    const x = plot.x + plot.w * z / 8;
    const y = centerY - residual / 0.15 * (plot.h * 0.43);
    if (step === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#8b9b8e";
  ctx.font = "10px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText("0.15", plot.x - 9, plot.y + 5);
  ctx.fillText("0", plot.x - 9, centerY + 3);
  ctx.fillText("−0.15", plot.x - 9, plot.y + plot.h + 3);
  ctx.textAlign = "left";
  const z = model.parameters.redshift;
  const selected = amplitude * (1 - Math.exp(-z));
  const x = plot.x + plot.w * z / 8;
  const y = centerY - selected / 0.15 * (plot.h * 0.43);
  ctx.fillStyle = "#ffe2cf";
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.fill();
  document.querySelector("#delta-mu").textContent = `${selected >= 0 ? "+" : ""}${selected.toFixed(4)} mag`;
}

function drawModular(model) {
  const fitted = plotBase(canvases.modular, 2.05);
  if (!fitted) return;
  const { ctx, plot } = fitted;
  const values = model.ramanujan_delta_harmonic;
  const centerY = plot.y + plot.h / 2;
  ctx.strokeStyle = "#a8d8bb";
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = plot.x + plot.w * index / (values.length - 1);
    const y = centerY - value * plot.h * 0.42;
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#829386";
  ctx.font = "10px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText("+1", plot.x - 9, plot.y + 5);
  ctx.fillText("0", plot.x - 9, centerY + 3);
  ctx.fillText("−1", plot.x - 9, plot.y + plot.h + 3);
}

function drawInvariantDensity(model) {
  if (!canvases.invDensity) return;
  const fitted = fitCanvas(canvases.invDensity, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 42, right: 15, top: 18, bottom: 30 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };

  const { flux_bins: bins, density_invariant: pInv, density_turbulent: pTurb } = model.statistical_invariance;
  const maxP = Math.max(Math.max(...pInv), Math.max(...pTurb), 0.1);

  // Draw grid
  ctx.strokeStyle = "rgba(166, 190, 170, .12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath();
    ctx.moveTo(plot.x, y);
    ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }

  // Draw Invariant Distribution (Cyan)
  ctx.strokeStyle = "#80c9cb";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  pInv.forEach((val, idx) => {
    const x = plot.x + plot.w * idx / (pInv.length - 1);
    const y = plot.y + plot.h * (1.0 - val / maxP);
    if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw Turbulent Snapshot (Orange)
  ctx.strokeStyle = "#f28b54";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  pTurb.forEach((val, idx) => {
    const x = plot.x + plot.w * idx / (pTurb.length - 1);
    const y = plot.y + plot.h * (1.0 - val / maxP);
    if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Axes labels
  ctx.fillStyle = "#829386";
  ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText(maxP.toFixed(1), plot.x - 6, plot.y + 6);
  ctx.fillText("0", plot.x - 6, plot.y + plot.h + 2);
  ctx.textAlign = "center";
  ctx.fillText("Flux Intensity I / I₀", plot.x + plot.w / 2, height - 8);
}

function drawSubrings(model) {
  if (!canvases.subrings) return;
  const fitted = fitCanvas(canvases.subrings, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 42, right: 15, top: 18, bottom: 30 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };

  const { radii_rg: radii, intensity_invariant: iInv, intensity_snapshot: iTurb } = model.statistical_invariance.radial_profile;
  const maxI = Math.max(Math.max(...iInv), Math.max(...iTurb), 1.0);

  // Grid
  ctx.strokeStyle = "rgba(166, 190, 170, .12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath();
    ctx.moveTo(plot.x, y);
    ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }

  // Time-averaged Invariant Profile (Cyan)
  ctx.strokeStyle = "#80c9cb";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  iInv.forEach((val, idx) => {
    const x = plot.x + plot.w * idx / (iInv.length - 1);
    const y = plot.y + plot.h * (1.0 - val / maxI);
    if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Snapshot (Orange)
  ctx.strokeStyle = "#f28b54";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  iTurb.forEach((val, idx) => {
    const x = plot.x + plot.w * idx / (iTurb.length - 1);
    const y = plot.y + plot.h * (1.0 - val / maxI);
    if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#829386";
  ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText(maxI.toFixed(1), plot.x - 6, plot.y + 6);
  ctx.fillText("0", plot.x - 6, plot.y + plot.h + 2);
  ctx.textAlign = "center";
  ctx.fillText("Radius r / r_g", plot.x + plot.w / 2, height - 8);
}

function drawParetoGame(model) {
  if (!canvases.pareto) return;
  const fitted = fitCanvas(canvases.pareto, 2.2);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 52, right: 25, top: 22, bottom: 38 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };

  const frontier = model.ergodic_game.pareto_frontier;
  const nash = model.ergodic_game.nash_equilibrium;

  const maxH = 1.15;
  const maxF = 1.10;

  // Grid
  ctx.strokeStyle = "rgba(166, 190, 170, .13)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = plot.y + plot.h * i / 4;
    ctx.beginPath();
    ctx.moveTo(plot.x, y);
    ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
    const x = plot.x + plot.w * i / 4;
    ctx.beginPath();
    ctx.moveTo(x, plot.y);
    ctx.lineTo(x, plot.y + plot.h);
    ctx.stroke();
  }

  // Pareto Frontier line (Cyan)
  ctx.strokeStyle = "#80c9cb";
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  frontier.forEach((pt, idx) => {
    const x = plot.x + plot.w * (pt.entropy / maxH);
    const y = plot.y + plot.h * (1.0 - pt.resolution / maxF);
    if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Nash equilibrium point (Signal orange)
  const nx = plot.x + plot.w * (nash.entropy / maxH);
  const ny = plot.y + plot.h * (1.0 - nash.resolution / maxF);
  ctx.fillStyle = "#f28b54";
  ctx.beginPath();
  ctx.arc(nx, ny, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#ffe2cf";
  ctx.lineWidth = 2;
  ctx.stroke();

  // Axis text
  ctx.fillStyle = "#829386";
  ctx.font = "10px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText("1.0", plot.x - 8, plot.y + plot.h * 0.1);
  ctx.fillText("0.5", plot.x - 8, plot.y + plot.h * 0.55);
  ctx.fillText("0", plot.x - 8, plot.y + plot.h + 2);
  ctx.textAlign = "center";
  ctx.fillText("Information Entropy H(S) [nats]", plot.x + plot.w / 2, height - 12);
  ctx.textAlign = "left";
  ctx.fillText("Fisher Resolution F", plot.x + 8, plot.y + 14);
}

function drawUvCoverage(model) {
  if (!canvases.uvCoverage) return;
  const fitted = fitCanvas(canvases.uvCoverage, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const cx = width / 2;
  const cy = height / 2;
  const maxR = Math.min(width, height) * 0.44;

  // Grid rings
  ctx.strokeStyle = "rgba(166, 190, 170, .14)";
  ctx.lineWidth = 1;
  [0.33, 0.66, 1.0].forEach((ratio) => {
    ctx.beginPath();
    ctx.arc(cx, cy, maxR * ratio, 0, Math.PI * 2);
    ctx.stroke();
  });
  ctx.beginPath();
  ctx.moveTo(cx - maxR, cy); ctx.lineTo(cx + maxR, cy);
  ctx.moveTo(cx, cy - maxR); ctx.lineTo(cx, cy + maxR);
  ctx.stroke();

  // Baseline points
  const points = model.image_reconstruction.uv_coverage;
  ctx.fillStyle = "rgba(128, 201, 203, .85)";
  points.forEach((pt) => {
    const px = cx + (pt.u / 100.0) * maxR;
    const py = cy - (pt.v / 100.0) * maxR;
    ctx.fillRect(px - 1, py - 1, 2.2, 2.2);
    // Conjugate symmetry
    const cpx = cx - (pt.u / 100.0) * maxR;
    const cpy = cy + (pt.v / 100.0) * maxR;
    ctx.fillRect(cpx - 1, cpy - 1, 2.2, 2.2);
  });

  ctx.fillStyle = "#829386";
  ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.fillText("+u", cx + maxR - 12, cy - 4);
  ctx.fillText("+v", cx + 4, cy - maxR + 10);
  ctx.fillText("Baseline spatial freq (Gλ)", 10, height - 8);
}

function drawReconstructedImage(model) {
  if (!canvases.reconImage) return;
  const fitted = fitCanvas(canvases.reconImage, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const halfW = width / 2;
  const cy = height / 2;
  const cx1 = halfW * 0.5;
  const cx2 = halfW * 1.5;
  const baseR = Math.min(halfW, height) * 0.28;

  // Divider
  ctx.strokeStyle = "rgba(166, 190, 170, .2)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(halfW, 10); ctx.lineTo(halfW, height - 10);
  ctx.stroke();

  // LEFT: Dirty Image (Blurred, aliased sidelobes)
  for (let r = baseR * 1.5; r > 5; r -= 3) {
    const aliased = Math.sin(r * 0.4) * 0.3 + 0.5;
    ctx.strokeStyle = `rgba(242, 139, 84, ${Math.max(0, aliased * 0.22)})`;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(cx1, cy, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.fillStyle = "#0c130f";
  ctx.beginPath();
  ctx.arc(cx1, cy, baseR * 0.75, 0, Math.PI * 2);
  ctx.fill();

  // RIGHT: Continued-Fraction Padé Restored Image (Sharp ring, suppressed sidelobes)
  const grad = ctx.createRadialGradient(cx2, cy, baseR * 0.7, cx2, cy, baseR * 1.3);
  grad.addColorStop(0, "rgba(128, 201, 203, 0)");
  grad.addColorStop(0.35, "rgba(255, 230, 180, 0.95)");
  grad.addColorStop(0.55, "rgba(128, 201, 203, 0.85)");
  grad.addColorStop(1, "rgba(17, 28, 22, 0)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx2, cy, baseR * 1.4, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#090d0b";
  ctx.beginPath();
  ctx.arc(cx2, cy, baseR * 0.82, 0, Math.PI * 2);
  ctx.fill();

  // Labels
  ctx.fillStyle = "#f28b54";
  ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "center";
  ctx.fillText("DIRTY APERTURE", cx1, height - 10);
  ctx.fillStyle = "#80c9cb";
  ctx.fillText("CF-PADÉ RESTORED", cx2, height - 10);
}

function drawReconstructionCrossSection(model) {
  if (!canvases.reconCross) return;
  const fitted = fitCanvas(canvases.reconCross, 2.2);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 20, top: 18, bottom: 32 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const data = model.image_reconstruction.cross_section;
  const maxVal = 2.6;

  // Grid
  ctx.strokeStyle = "rgba(166, 190, 170, .12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath();
    ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }

  // 1. Ground Truth (Green #4ade80)
  ctx.strokeStyle = "#4ade80";
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  data.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (data.length - 1);
    const y = plot.y + plot.h * (1.0 - pt.ground_truth / maxVal);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // 2. Dirty Image (Orange #f28b54)
  ctx.strokeStyle = "#f28b54";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  data.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (data.length - 1);
    const y = plot.y + plot.h * (1.0 - pt.dirty_image / maxVal);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // 3. Continued Fraction Padé Filter (Cyan #80c9cb)
  ctx.strokeStyle = "#80c9cb";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  data.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (data.length - 1);
    const y = plot.y + plot.h * (1.0 - pt.cf_reconstruction / maxVal);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Axes
  ctx.fillStyle = "#829386";
  ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText(maxVal.toFixed(1), plot.x - 6, plot.y + 6);
  ctx.fillText("0", plot.x - 6, plot.y + plot.h + 2);
  ctx.textAlign = "center";
  ctx.fillText("Radial Horizon Coordinate r / r_g", plot.x + plot.w / 2, height - 8);
}

function drawHifuFocus(model) {
  if (!canvases.hifuFocus) return;
  const fitted = fitCanvas(canvases.hifuFocus, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 20, top: 18, bottom: 32 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const field = model.hifu_dementia.focus_field;
  const maxP = Math.max(...field.map(f => f.pressure_mpa), 1.2);

  // Grid
  ctx.strokeStyle = "rgba(166, 190, 170, .12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath();
    ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }

  // Focal field curve (Cyan)
  ctx.strokeStyle = "#80c9cb";
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  field.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (field.length - 1);
    const y = plot.y + plot.h * (1.0 - pt.pressure_mpa / maxP);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#829386";
  ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText(`${maxP.toFixed(2)} MPa`, plot.x - 6, plot.y + 6);
  ctx.fillText("0", plot.x - 6, plot.y + plot.h + 2);
  ctx.textAlign = "center";
  ctx.fillText("Lateral Focal Position (mm) across Hippocampus", plot.x + plot.w / 2, height - 8);
}

function drawHifuSpectrum(model) {
  if (!canvases.hifuSpectrum) return;
  const fitted = fitCanvas(canvases.hifuSpectrum, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 20, top: 18, bottom: 32 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const spec = model.hifu_dementia.cavitation_spectrum;
  const maxAmp = 65.0;
  const minAmp = 0.0;

  // Grid
  ctx.strokeStyle = "rgba(166, 190, 170, .12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath();
    ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }

  // RF Cavitation curve (Signal orange / Cyan)
  ctx.strokeStyle = "#2dd4bf";
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  spec.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (spec.length - 1);
    const norm = Math.max(0, Math.min(1.0, (pt.amplitude_db - minAmp) / (maxAmp - minAmp)));
    const y = plot.y + plot.h * (1.0 - norm);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#829386";
  ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText("60 dB", plot.x - 6, plot.y + 6);
  ctx.fillText("0 dB", plot.x - 6, plot.y + plot.h + 2);
  ctx.textAlign = "center";
  ctx.fillText("Acoustic Frequency (MHz) [f₀/2 subharmonic marker]", plot.x + plot.w / 2, height - 8);
}

function drawHifuClearance(model) {
  if (!canvases.hifuClearance) return;
  const fitted = fitCanvas(canvases.hifuClearance, 2.2);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 20, top: 18, bottom: 32 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const sessions = model.hifu_dementia.session_trajectory;

  // Grid
  ctx.strokeStyle = "rgba(166, 190, 170, .12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = plot.y + plot.h * i / 4;
    ctx.beginPath();
    ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }

  // 1. Amyloid-beta burden % (Orange #f28b54)
  ctx.strokeStyle = "#f28b54";
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  sessions.forEach((s, i) => {
    const x = plot.x + plot.w * i / (sessions.length - 1);
    const y = plot.y + plot.h * (1.0 - s.amyloid_burden_pct / 100.0);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // 2. Tau burden % (Rose #f43f5e)
  ctx.strokeStyle = "#f43f5e";
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  sessions.forEach((s, i) => {
    const x = plot.x + plot.w * i / (sessions.length - 1);
    const y = plot.y + plot.h * (1.0 - s.tau_burden_pct / 100.0);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // 3. Cognitive Retention Score (Cyan #80c9cb)
  ctx.strokeStyle = "#80c9cb";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  sessions.forEach((s, i) => {
    const x = plot.x + plot.w * i / (sessions.length - 1);
    const y = plot.y + plot.h * (1.0 - s.cognitive_retention_score / 100.0);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#829386";
  ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText("100%", plot.x - 6, plot.y + 6);
  ctx.fillText("0%", plot.x - 6, plot.y + plot.h + 2);
  ctx.textAlign = "center";
  ctx.fillText("Weekly Treatment Sessions (Targeted Transcranial BBB Openings)", plot.x + plot.w / 2, height - 8);
}

function drawSrFork(model) {
  if (!canvases.srFork || !model.special_relativity_conebeam) return;
  const fitted = fitCanvas(canvases.srFork, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 18, top: 18, bottom: 32 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const pts = model.special_relativity_conebeam.fork_in_the_road.spectrum_points;
  const eFork = model.special_relativity_conebeam.fork_in_the_road.fork_threshold_gev;
  const minBeta = 0.70;
  const maxBeta = 1.0005;

  // Grid
  ctx.strokeStyle = "rgba(166, 190, 170, .12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath();
    ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }

  // Vertical Fork in the Road line at E_fork
  const forkFrac = Math.max(0, Math.min(1.0, (eFork - 0.5) / (500.0 - 0.5)));
  const xFork = plot.x + plot.w * forkFrac;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.28)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(xFork, plot.y); ctx.lineTo(xFork, plot.y + plot.h);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#829386";
  ctx.font = "8px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "center";
  ctx.fillText(`FORK: ${eFork} GeV`, xFork, plot.y - 4);

  // 1. Einstein SR baseline (Mint #a8d8bb)
  ctx.strokeStyle = "#a8d8bb";
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  pts.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (pts.length - 1);
    const norm = Math.max(0, Math.min(1.0, (pt.beta_einstein - minBeta) / (maxBeta - minBeta)));
    const y = plot.y + plot.h * (1.0 - norm);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // 2. Superluminal branch (Sky-blue #38bdf8)
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  pts.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (pts.length - 1);
    const norm = Math.max(0, Math.min(1.0, (pt.beta_superluminal - minBeta) / (maxBeta - minBeta)));
    const y = plot.y + plot.h * (1.0 - norm);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // 3. Subluminal LIV branch (Orange #f28b54)
  ctx.strokeStyle = "#f28b54";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  pts.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (pts.length - 1);
    const norm = Math.max(0, Math.min(1.0, (pt.beta_subluminal - minBeta) / (maxBeta - minBeta)));
    const y = plot.y + plot.h * (1.0 - norm);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Mark current operating point
  const curE = model.special_relativity_conebeam.kinematics.kinetic_energy_gev;
  const curFrac = Math.max(0, Math.min(1.0, (curE - 0.5) / (500.0 - 0.5)));
  const curX = plot.x + plot.w * curFrac;
  const curBeta = model.special_relativity_conebeam.kinematics.beta;
  const curY = plot.y + plot.h * (1.0 - Math.max(0, Math.min(1.0, (curBeta - minBeta) / (maxBeta - minBeta))));
  ctx.fillStyle = "#ffedd5";
  ctx.beginPath();
  ctx.arc(curX, curY, 4.5, 0, Math.PI * 2);
  ctx.fill();

  // Labels
  ctx.fillStyle = "#829386";
  ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText("1.000 c", plot.x - 6, plot.y + 6);
  ctx.fillText("0.700 c", plot.x - 6, plot.y + plot.h + 2);
  ctx.textAlign = "center";
  ctx.fillText("Particle Kinetic Energy (GeV) [0.5 to 500 GeV]", plot.x + plot.w / 2, height - 8);
}

function drawCbctProfile(model) {
  if (!canvases.cbctProfile || !model.special_relativity_conebeam) return;
  const fitted = fitCanvas(canvases.cbctProfile, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 18, top: 18, bottom: 32 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const data = model.special_relativity_conebeam.conebeam_ct.spatial_line_profile;
  const minHU = -250.0;
  const maxHU = 1100.0;

  // Grid
  ctx.strokeStyle = "rgba(166, 190, 170, .12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath();
    ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }

  // 1. Ground Truth (Green #4ade80)
  ctx.strokeStyle = "#4ade80";
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  data.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (data.length - 1);
    const norm = Math.max(0, Math.min(1.0, (pt.ground_truth_hu - minHU) / (maxHU - minHU)));
    const y = plot.y + plot.h * (1.0 - norm);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // 2. Uncorrected Cupping Artifact (Orange #f28b54)
  ctx.strokeStyle = "#f28b54";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  data.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (data.length - 1);
    const norm = Math.max(0, Math.min(1.0, (pt.uncorrected_hu - minHU) / (maxHU - minHU)));
    const y = plot.y + plot.h * (1.0 - norm);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // 3. Stieltjes Continued Fraction Restored (Cyan #80c9cb)
  ctx.strokeStyle = "#80c9cb";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  data.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (data.length - 1);
    const norm = Math.max(0, Math.min(1.0, (pt.stieltjes_restored_hu - minHU) / (maxHU - minHU)));
    const y = plot.y + plot.h * (1.0 - norm);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Labels
  ctx.fillStyle = "#829386";
  ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText("1000 HU", plot.x - 6, plot.y + 6);
  ctx.fillText("0 HU", plot.x - 6, plot.y + plot.h * (1.0 - (0 - minHU) / (maxHU - minHU)));
  ctx.fillText("-200 HU", plot.x - 6, plot.y + plot.h + 2);
  ctx.textAlign = "center";
  ctx.fillText("Detector Spatial Channel Position across Phantom (mm)", plot.x + plot.w / 2, height - 8);
}

function drawMaxwellJuttner(model) {
  if (!canvases.maxwellJuttner || !model.special_relativity_conebeam) return;
  const fitted = fitCanvas(canvases.maxwellJuttner, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 18, top: 18, bottom: 32 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const pdf = model.special_relativity_conebeam.statistical_distribution.maxwell_juttner_pdf;
  const maxP = Math.max(...pdf.map(p => p.probability_density), 0.05);

  // Grid
  ctx.strokeStyle = "rgba(166, 190, 170, .12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath();
    ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }

  // Shaded area under curve
  ctx.fillStyle = "rgba(128, 201, 203, 0.18)";
  ctx.beginPath();
  ctx.moveTo(plot.x, plot.y + plot.h);
  pdf.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (pdf.length - 1);
    const y = plot.y + plot.h * (1.0 - pt.probability_density / maxP);
    ctx.lineTo(x, y);
  });
  ctx.lineTo(plot.x + plot.w, plot.y + plot.h);
  ctx.closePath();
  ctx.fill();

  // Curve
  ctx.strokeStyle = "#80c9cb";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  pdf.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (pdf.length - 1);
    const y = plot.y + plot.h * (1.0 - pt.probability_density / maxP);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Mean gamma vertical dashed line
  const meanGamma = model.special_relativity_conebeam.statistical_distribution.mean_gamma;
  const gMin = pdf[0].gamma;
  const gMax = pdf[pdf.length - 1].gamma;
  const mFrac = Math.max(0, Math.min(1.0, (meanGamma - gMin) / (gMax - gMin)));
  const mX = plot.x + plot.w * mFrac;
  ctx.strokeStyle = "#f28b54";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(mX, plot.y); ctx.lineTo(mX, plot.y + plot.h);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#f28b54";
  ctx.font = "8px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "center";
  ctx.fillText(`⟨γ⟩ = ${meanGamma}`, mX, plot.y - 4);

  // Labels
  ctx.fillStyle = "#829386";
  ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText(maxP.toFixed(2), plot.x - 6, plot.y + 6);
  ctx.fillText("0", plot.x - 6, plot.y + plot.h + 2);
  ctx.textAlign = "center";
  ctx.fillText("Lorentz Factor γ (Dimensionless)", plot.x + plot.w / 2, height - 8);
}

function drawStieltjesConvergence(model) {
  if (!canvases.stieltjesConvergence || !model.special_relativity_conebeam) return;
  const fitted = fitCanvas(canvases.stieltjesConvergence, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 18, top: 18, bottom: 32 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const series = model.special_relativity_conebeam.statistical_distribution.cf_convergence;

  // Log error range: 0 down to -6
  const logMax = 0.0;
  const logMin = -6.0;

  // Grid
  ctx.strokeStyle = "rgba(166, 190, 170, .12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath();
    ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }

  // 1. Taylor polynomial error (Orange #f28b54)
  ctx.strokeStyle = "#f28b54";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  series.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (series.length - 1);
    const logErr = Math.log10(Math.max(1e-7, pt.polynomial_taylor_error));
    const norm = Math.max(0, Math.min(1.0, (logErr - logMin) / (logMax - logMin)));
    const y = plot.y + plot.h * (1.0 - norm);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // 2. Stieltjes continued fraction error (Cyan #80c9cb)
  ctx.strokeStyle = "#80c9cb";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  series.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (series.length - 1);
    const logErr = Math.log10(Math.max(1e-7, pt.stieltjes_cf_error));
    const norm = Math.max(0, Math.min(1.0, (logErr - logMin) / (logMax - logMin)));
    const y = plot.y + plot.h * (1.0 - norm);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Dots for Stieltjes
  ctx.fillStyle = "#80c9cb";
  series.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (series.length - 1);
    const logErr = Math.log10(Math.max(1e-7, pt.stieltjes_cf_error));
    const norm = Math.max(0, Math.min(1.0, (logErr - logMin) / (logMax - logMin)));
    const y = plot.y + plot.h * (1.0 - norm);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  });

  // Labels
  ctx.fillStyle = "#829386";
  ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText("10⁰", plot.x - 6, plot.y + 6);
  ctx.fillText("10⁻³", plot.x - 6, plot.y + plot.h / 2 + 3);
  ctx.fillText("10⁻⁶", plot.x - 6, plot.y + plot.h + 2);
  ctx.textAlign = "center";
  ctx.fillText("Approximation Order (Degree k = 1 to 10)", plot.x + plot.w / 2, height - 8);
}

function drawBbCosmic(model) {
  if (!canvases.bbCosmic || !model.big_bang) return;
  const fitted = fitCanvas(canvases.bbCosmic, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 18, top: 18, bottom: 32 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const timeline = model.big_bang.cosmic_timeline;

  // Grid
  ctx.strokeStyle = "rgba(166, 190, 170, .12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath();
    ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }

  // 1. Log scale factor log_a (-35 to 10) (Mint #a8d8bb)
  const minA = -35.0, maxA = 10.0;
  ctx.strokeStyle = "#a8d8bb";
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  timeline.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (timeline.length - 1);
    const norm = Math.max(0, Math.min(1.0, (pt.log_a - minA) / (maxA - minA)));
    const y = plot.y + plot.h * (1.0 - norm);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // 2. Log radiation temp log_temp_k (2 to 32 K) (Orange #f28b54)
  const minT = 2.0, maxT = 32.0;
  ctx.strokeStyle = "#f28b54";
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  timeline.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (timeline.length - 1);
    const norm = Math.max(0, Math.min(1.0, (pt.log_temp_k - minT) / (maxT - minT)));
    const y = plot.y + plot.h * (1.0 - norm);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Labels
  ctx.fillStyle = "#829386";
  ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText("10³² K", plot.x - 6, plot.y + 6);
  ctx.fillText("3000 K", plot.x - 6, plot.y + plot.h - 4);
  ctx.textAlign = "center";
  ctx.fillText("Cosmic Time log₁₀(t / s) [10⁻⁴³ s Planck to 10¹³ s Recombination]", plot.x + plot.w / 2, height - 8);
}

function drawBbHawking(model) {
  if (!canvases.bbHawking || !model.big_bang) return;
  const fitted = fitCanvas(canvases.bbHawking, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 18, top: 18, bottom: 32 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const timeline = model.big_bang.cosmic_timeline;

  // Grid
  ctx.strokeStyle = "rgba(166, 190, 170, .12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath();
    ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }

  // 1. PBH Mass fraction M(t)/M0 (Sky blue #38bdf8)
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  timeline.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (timeline.length - 1);
    const y = plot.y + plot.h * (1.0 - pt.pbh_mass_frac);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // 2. Hawking burst power log10 P(t) (Rose #f43f5e)
  const maxLogP = 30.0;
  ctx.strokeStyle = "#f43f5e";
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  timeline.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (timeline.length - 1);
    const norm = Math.max(0, Math.min(1.0, pt.p_hawking_log_w / maxLogP));
    const y = plot.y + plot.h * (1.0 - norm);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Labels
  ctx.fillStyle = "#829386";
  ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText("1.0 M₀", plot.x - 6, plot.y + 6);
  ctx.fillText("0 M₀", plot.x - 6, plot.y + plot.h + 2);
  ctx.textAlign = "center";
  ctx.fillText("Cosmic Evolution toward Final PBH Evaporation Epoch", plot.x + plot.w / 2, height - 8);
}

function drawBbPrimeModuli(model) {
  if (!canvases.bbPrimeModuli || !model.big_bang) return;
  const fitted = fitCanvas(canvases.bbPrimeModuli, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 18, top: 18, bottom: 32 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const lattice = model.big_bang.prime_moduli.lattice;
  const p = model.big_bang.prime_moduli.prime_p;

  // Grid
  ctx.strokeStyle = "rgba(166, 190, 170, .12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath();
    ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }

  // Horizontal singularity boundary line (Yellow #fcd34d) at p - 1
  const yBound = plot.y + plot.h * (1.0 - (p - 1) / p);
  ctx.strokeStyle = "rgba(252, 211, 77, 0.45)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(plot.x, yBound); ctx.lineTo(plot.x + plot.w, yBound);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#fcd34d";
  ctx.font = "8px SFMono-Regular, Consolas, monospace";
  ctx.fillText(`BOUND: p − 1 = ${p - 1}`, plot.x + plot.w - 80, yBound - 4);

  // Bars for modular curvature K_k mod p
  const barW = Math.max(3, (plot.w / lattice.length) * 0.65);
  lattice.forEach((pt, i) => {
    const x = plot.x + (plot.w / lattice.length) * (i + 0.5) - barW / 2;
    const barH = plot.h * (pt.curvature_mod_p / p);
    const y = plot.y + plot.h - barH;
    ctx.fillStyle = pt.legendre >= 0 ? "rgba(168, 216, 187, 0.85)" : "rgba(242, 139, 84, 0.85)";
    ctx.fillRect(x, y, barW, barH);
  });

  // Gauss sum component line
  ctx.strokeStyle = "#80c9cb";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  lattice.forEach((pt, i) => {
    const x = plot.x + (plot.w / lattice.length) * (i + 0.5);
    const norm = (pt.gauss_re + 1.0) / 2.0;
    const y = plot.y + plot.h * (1.0 - norm);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Labels
  ctx.fillStyle = "#829386";
  ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText(`p = ${p}`, plot.x - 6, plot.y + 6);
  ctx.fillText("0", plot.x - 6, plot.y + plot.h + 2);
  ctx.textAlign = "center";
  ctx.fillText(`Discrete Element k ∈ ℤ_${p} [Galois Curvature Lattice]`, plot.x + plot.w / 2, height - 8);
}

function drawBbQuantum(model) {
  if (!canvases.bbQuantum || !model.big_bang) return;
  const fitted = fitCanvas(canvases.bbQuantum, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111c16";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 18, top: 18, bottom: 32 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const wf = model.big_bang.quantum_signatures.wavefunction;
  const spec = model.big_bang.quantum_signatures.hawking_spectrum;

  // Grid
  ctx.strokeStyle = "rgba(166, 190, 170, .12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath();
    ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
  }

  // 1. Wheeler-DeWitt |Psi(a)|^2 (Mint #a8d8bb with shaded area)
  const maxPsi = Math.max(...wf.map(w => w.psi_squared), 0.05);
  ctx.fillStyle = "rgba(168, 216, 187, 0.16)";
  ctx.beginPath();
  ctx.moveTo(plot.x, plot.y + plot.h);
  wf.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (wf.length - 1);
    const y = plot.y + plot.h * (1.0 - pt.psi_squared / maxPsi);
    ctx.lineTo(x, y);
  });
  ctx.lineTo(plot.x + plot.w, plot.y + plot.h);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = "#a8d8bb";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  wf.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (wf.length - 1);
    const y = plot.y + plot.h * (1.0 - pt.psi_squared / maxPsi);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // 2. Hawking particle burst emission spectrum dN/dE (Orange #fb923c)
  ctx.strokeStyle = "#fb923c";
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  spec.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (spec.length - 1);
    const y = plot.y + plot.h * (1.0 - pt.emission_rate);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Labels
  ctx.fillStyle = "#829386";
  ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText("1.0", plot.x - 6, plot.y + 6);
  ctx.fillText("0.0", plot.x - 6, plot.y + plot.h + 2);
  ctx.textAlign = "center";
  ctx.fillText("Quantum Minisuperspace Amplitude & Hawking Emission Modes", plot.x + plot.w / 2, height - 8);
}

function renderMetrics(model) {
  document.querySelector("#gr-shadow").textContent = `${model.shadow_radius_gr_rg.toFixed(3)} r_g`;
  document.querySelector("#toy-shadow").textContent = `${model.shadow_radius_modular_toy_rg.toFixed(3)} r_g`;
  document.querySelector("#elliptic-nome").textContent = model.elliptic_nome.toFixed(5);
  document.querySelector("#hawking-temperature").textContent = `${model.hawking_temperature_k.toExponential(2)} K`;
  document.querySelector("#entropy").textContent = model.bekenstein_hawking_entropy_over_kb.toExponential(2);

  if (document.querySelector("#inv-lyapunov")) {
    document.querySelector("#inv-lyapunov").textContent = model.statistical_invariance.lyapunov_gamma.toFixed(3);
    document.querySelector("#inv-subring-ratio").textContent = model.statistical_invariance.subring_scale_decay.toFixed(4);
    document.querySelector("#inv-wasserstein").textContent = model.statistical_invariance.wasserstein_w1.toFixed(4);
    document.querySelector("#inv-haar-moment").textContent = model.statistical_invariance.haar_azimuthal_moments.mu_1.toFixed(3);
    document.querySelector("#inv-nash-value").textContent = model.ergodic_game.nash_equilibrium.payoff_value.toFixed(3);
  }

  if (document.querySelector("#recon-snr-gain") && model.image_reconstruction) {
    document.querySelector("#recon-snr-gain").textContent = `${model.image_reconstruction.metrics.snr_gain_db.toFixed(1)} dB`;
    document.querySelector("#recon-ssim").textContent = model.image_reconstruction.metrics.ssim.toFixed(3);
    document.querySelector("#recon-sidelobe-sup").textContent = `${model.image_reconstruction.metrics.sidelobe_suppression_db.toFixed(1)} dB`;
    document.querySelector("#recon-resolution").textContent = `${model.image_reconstruction.metrics.angular_resolution_uas.toFixed(1)} µas`;
    const terms = model.image_reconstruction.cf_gamma_terms.slice(0, 4).join(", ");
    document.querySelector("#recon-cf-order").textContent = `[${terms}, ...]`;
  }

  if (document.querySelector("#hifu-focal-pressure") && model.hifu_dementia) {
    document.querySelector("#hifu-focal-pressure").textContent = `${model.hifu_dementia.focal_pressure_mpa.toFixed(2)} MPa`;
    document.querySelector("#hifu-mechanical-index").textContent = model.hifu_dementia.mechanical_index.toFixed(2);
    document.querySelector("#hifu-bbb-ratio").textContent = `${model.hifu_dementia.bbb_permeability_ratio.toFixed(1)}x`;
    const lastSession = model.hifu_dementia.session_trajectory[model.hifu_dementia.session_trajectory.length - 1];
    document.querySelector("#hifu-amyloid-clearance").textContent = `${(100.0 - lastSession.amyloid_burden_pct).toFixed(1)}%`;
    document.querySelector("#hifu-regime-status").textContent = model.hifu_dementia.safety_status;
  }

  if (document.querySelector("#sr-gamma") && model.special_relativity_conebeam) {
    const sr = model.special_relativity_conebeam;
    document.querySelector("#sr-gamma").textContent = sr.kinematics.lorentz_gamma.toFixed(2);
    document.querySelector("#sr-speed-deficit").textContent = `${sr.kinematics.speed_deficit_ms.toLocaleString(undefined, { maximumFractionDigits: 1 })} m/s`;
    document.querySelector("#sr-fork-threshold").textContent = `${sr.fork_in_the_road.fork_threshold_gev.toFixed(1)} GeV`;
    document.querySelector("#cbct-hu-error").textContent = `${sr.conebeam_ct.max_cupping_artifact_hu.toFixed(1)} HU`;
    document.querySelector("#sr-cf-order").textContent = `Padé [${sr.conebeam_ct.cf_order}/${sr.conebeam_ct.cf_order}]`;
    document.querySelector("#sr-bifurcation-delta").textContent = `${sr.fork_in_the_road.bifurcation_split_ms.toFixed(3)} m/s`;
  }

  if (document.querySelector("#bb-planck-density") && model.big_bang) {
    const bb = model.big_bang;
    document.querySelector("#bb-planck-density").textContent = bb.metrics.planck_finite_curvature;
    document.querySelector("#bb-hawking-temp").textContent = bb.metrics.hawking_burst_temp_gev;
    document.querySelector("#bb-pbh-lifetime").textContent = bb.metrics.pbh_lifetime_s;
    document.querySelector("#bb-inflation-expansion").textContent = bb.metrics.inflation_expansion;
    document.querySelector("#bb-recombination-z").textContent = bb.metrics.recombination_z;
    document.querySelector("#bb-entropy-bound").textContent = bb.metrics.quantum_entropy;
  }

  drawBlackHole(model);
  if (activeView === "fork") drawFork(model);
  if (activeView === "modular") drawModular(model);
  if (activeView === "invariant") {
    drawInvariantDensity(model);
    drawSubrings(model);
    drawParetoGame(model);
  }
  if (activeView === "reconstruction") {
    drawUvCoverage(model);
    drawReconstructedImage(model);
    drawReconstructionCrossSection(model);
  }
  if (activeView === "hifu") {
    drawHifuFocus(model);
    drawHifuSpectrum(model);
    drawHifuClearance(model);
  }
  if (activeView === "relativity") {
    drawSrFork(model);
    drawCbctProfile(model);
    drawMaxwellJuttner(model);
    drawStieltjesConvergence(model);
  }
  if (activeView === "bigbang") {
    drawBbCosmic(model);
    drawBbHawking(model);
    drawBbPrimeModuli(model);
    drawBbQuantum(model);
  }
}

async function refreshModel() {
  const parameters = readParameters();
  updateLabels(parameters);
  const currentToken = ++requestToken;
  try {
    const response = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(parameters),
    });
    if (!response.ok) throw new Error(`Simulation returned ${response.status}`);
    const model = await response.json();
    if (currentToken !== requestToken) return;
    latestModel = model;
    renderMetrics(model);
  } catch (error) {
    document.querySelector(".live-state").lastChild.textContent = " OFFLINE";
    console.error(error);
  }
}

function switchView(view) {
  activeView = view;
  document.querySelectorAll(".view-tab").forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".workspace[role='tabpanel']").forEach((panel) => {
    panel.hidden = panel.id !== `view-${view}`;
  });
  if (latestModel && view === "fork") drawFork(latestModel);
  if (latestModel && view === "modular") drawModular(latestModel);
  if (latestModel && view === "invariant") {
    drawInvariantDensity(latestModel);
    drawSubrings(latestModel);
    drawParetoGame(latestModel);
  }
  if (latestModel && view === "reconstruction") {
    drawUvCoverage(latestModel);
    drawReconstructedImage(latestModel);
    drawReconstructionCrossSection(latestModel);
  }
  if (latestModel && view === "hifu") {
    drawHifuFocus(latestModel);
    drawHifuSpectrum(latestModel);
    drawHifuClearance(latestModel);
  }
  if (latestModel && view === "relativity") {
    drawSrFork(latestModel);
    drawCbctProfile(latestModel);
    drawMaxwellJuttner(latestModel);
    drawStieltjesConvergence(latestModel);
  }
  if (latestModel && view === "bigbang") {
    drawBbCosmic(latestModel);
    drawBbHawking(latestModel);
    drawBbPrimeModuli(latestModel);
    drawBbQuantum(latestModel);
  }
}

document.querySelectorAll(".view-tab").forEach((button) => {
  button.addEventListener("click", () => switchView(button.dataset.view));
});
controlIds.forEach((id) => document.getElementById(id).addEventListener("input", refreshModel));
document.querySelector("#reset-button").addEventListener("click", () => {
  controlIds.forEach((id) => { document.getElementById(id).value = defaults[id]; });
  refreshModel();
});
document.querySelector("#export-button").addEventListener("click", () => {
  canvases.hole.toBlob((blob) => {
    if (!blob) return;
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "horizon-atlas-black-hole-schematic.png";
    link.click();
    URL.revokeObjectURL(link.href);
  }, "image/png");
});
window.addEventListener("resize", () => {
  if (!latestModel) return;
  renderMetrics(latestModel);
});
refreshModel();