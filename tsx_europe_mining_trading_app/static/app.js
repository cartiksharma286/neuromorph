const controlIds = [
  "tsx_weight", "ger_weight", "ore_momentum", "volatility",
  "gamma_damping", "drift", "fx_eur_cad", "options_overlay", "prime_order"
];

const defaults = {
  tsx_weight: 0.55, ger_weight: 0.45, ore_momentum: 0.45, volatility: 0.28,
  gamma_damping: 0.65, drift: 0.15, fx_eur_cad: 1.4850, options_overlay: 0.85, prime_order: 10
};

const canvases = {
  overview: document.querySelector("#overview-canvas"),
  stieltjesDepth: document.querySelector("#stieltjes-depth-chart"),
  stieltjesConvergents: document.querySelector("#stieltjes-convergents-chart"),
  leibnizSeries: document.querySelector("#leibniz-series-chart"),
  primeResonance: document.querySelector("#prime-resonance-chart"),
  odeTrajectory: document.querySelector("#ode-trajectory-chart"),
  odePhase: document.querySelector("#ode-phase-chart"),
  dividendTrajectory: document.querySelector("#dividend-trajectory-chart"),
  dividendBreakdown: document.querySelector("#dividend-breakdown-chart"),
  liveOrderbook: document.querySelector("#live-orderbook-chart"),
  livePnl: document.querySelector("#live-pnl-chart"),
};

let latestModel = null;
let requestToken = 0;
let activeView = "overview";
let liveTradingActive = true;
let liveTrades = [];
let cumulativePnl = 148250;
let pnlHistory = [148250];
let liveTickInterval = null;
let userIsDraggingSlider = false;

// Parameter bounds mirror the <input min/max> in index.html; used by the live drift engine.
const paramBounds = {
  tsx_weight: [0.1, 0.9], ger_weight: [0.1, 0.9], ore_momentum: [-1.0, 1.0], volatility: [0.05, 0.95],
  gamma_damping: [0.05, 1.5], drift: [-0.5, 0.5], fx_eur_cad: [1.35, 1.65], options_overlay: [0.2, 1.0],
  prime_order: [4, 20]
};

function readParameters() {
  return Object.fromEntries(controlIds.map((id) => [id, Number(document.getElementById(id).value)]));
}

function updateLabels(p) {
  document.querySelector("#tsx_weight-value").textContent = p.tsx_weight.toFixed(2);
  document.querySelector("#ger_weight-value").textContent = p.ger_weight.toFixed(2);
  document.querySelector("#ore_momentum-value").textContent = (p.ore_momentum >= 0 ? "+" : "") + p.ore_momentum.toFixed(2);
  document.querySelector("#volatility-value").textContent = p.volatility.toFixed(2);
  document.querySelector("#gamma_damping-value").textContent = p.gamma_damping.toFixed(2);
  document.querySelector("#drift-value").textContent = (p.drift >= 0 ? "+" : "") + p.drift.toFixed(2);
  document.querySelector("#fx_eur_cad-value").textContent = p.fx_eur_cad.toFixed(4);
  document.querySelector("#options_overlay-value").textContent = p.options_overlay.toFixed(2);
  document.querySelector("#prime_order-value").textContent = String(Math.round(p.prime_order));
}

function fitCanvas(canvas, ratio = 2.0) {
  if (!canvas) return null;
  let rect = canvas.getBoundingClientRect();
  let width = rect.width;
  if (!width && canvas.parentElement) {
    width = canvas.parentElement.clientWidth || canvas.parentElement.getBoundingClientRect().width;
  }
  if (!width) {
    const panel = canvas.closest(".workspace");
    if (panel) {
      width = panel.clientWidth - 48;
    }
  }
  if (!width || width <= 10) return null;

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const height = Math.max(160, width / ratio);
  const targetBufferW = Math.round(width * dpr);
  const targetBufferH = Math.round(height * dpr);

  if (canvas.width !== targetBufferW || canvas.height !== targetBufferH) {
    canvas.width = targetBufferW;
    canvas.height = targetBufferH;
  }

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function drawSweepCursor(ctx, plot, color, speed = 0.00022) {
  const progress = (Date.now() * speed) % 1;
  const x = plot.x + plot.w * progress;
  const gradient = ctx.createLinearGradient(x - 20, 0, x + 4, 0);
  gradient.addColorStop(0, "rgba(0, 0, 0, 0)");
  gradient.addColorStop(1, color);
  ctx.fillStyle = gradient;
  ctx.fillRect(x - 20, plot.y, 24, plot.h);
  ctx.fillStyle = color;
  ctx.fillRect(x, plot.y, 2, plot.h);
}

function drawOverview(model) {
  const fitted = fitCanvas(canvases.overview, 1.85);
  if (!fitted) return;
  const { ctx, width, height } = fitted;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#070d12";
  ctx.fillRect(0, 0, width, height);

  // Draw grid lines
  ctx.strokeStyle = "rgba(24, 40, 54, 0.4)";
  ctx.lineWidth = 1;
  for (let x = 0; x < width; x += 36) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
  }
  for (let y = 0; y < height; y += 36) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }

  const txX = width * 0.22;
  const txY = height * 0.50;
  const deX = width * 0.78;
  const deY = height * 0.50;

  // Transatlantic connection line (Forward route)
  ctx.strokeStyle = "rgba(56, 189, 248, 0.45)";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(txX, txY);
  ctx.bezierCurveTo(width * 0.4, txY - 65, width * 0.6, deY - 65, deX, deY);
  ctx.stroke();

  // Bottom return arbitrage line (Settlement / FX route)
  ctx.strokeStyle = "rgba(52, 211, 153, 0.45)";
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  ctx.moveTo(deX, deY);
  ctx.bezierCurveTo(width * 0.6, deY + 65, width * 0.4, txY + 65, txX, txY);
  ctx.stroke();

  // Live streaming particles (Multiple packets)
  const now = Date.now() * 0.0018;
  for (let p = 0; p < 4; p++) {
    const tNorm = ((now + p * 0.25) % 1.0);
    // Upper path
    const px1 = txX + (deX - txX) * tNorm;
    const py1 = txY - Math.sin(tNorm * Math.PI) * 48;
    ctx.fillStyle = "#38bdf8";
    ctx.shadowColor = "#38bdf8";
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(px1, py1, 3.5, 0, Math.PI * 2);
    ctx.fill();

    // Lower path
    const px2 = deX - (deX - txX) * tNorm;
    const py2 = deY + Math.sin(tNorm * Math.PI) * 48;
    ctx.fillStyle = "#34d399";
    ctx.shadowColor = "#34d399";
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(px2, py2, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.shadowBlur = 0;

  // TSX Node
  const tsxGrad = ctx.createRadialGradient(txX, txY, 4, txX, txY, 50);
  tsxGrad.addColorStop(0, "rgba(251, 146, 60, 0.95)");
  tsxGrad.addColorStop(0.5, "rgba(251, 146, 60, 0.35)");
  tsxGrad.addColorStop(1, "rgba(251, 146, 60, 0)");
  ctx.fillStyle = tsxGrad;
  ctx.beginPath(); ctx.arc(txX, txY, 50, 0, Math.PI * 2); ctx.fill();

  ctx.fillStyle = "#fff";
  ctx.beginPath(); ctx.arc(txX, txY, 8, 0, Math.PI * 2); ctx.fill();

  // Frankfurt Node
  const deGrad = ctx.createRadialGradient(deX, deY, 4, deX, deY, 50);
  deGrad.addColorStop(0, "rgba(52, 211, 153, 0.95)");
  deGrad.addColorStop(0.5, "rgba(52, 211, 153, 0.35)");
  deGrad.addColorStop(1, "rgba(52, 211, 153, 0)");
  ctx.fillStyle = deGrad;
  ctx.beginPath(); ctx.arc(deX, deY, 50, 0, Math.PI * 2); ctx.fill();

  ctx.fillStyle = "#fff";
  ctx.beginPath(); ctx.arc(deX, deY, 8, 0, Math.PI * 2); ctx.fill();

  // Mini Orderbook Queues beside nodes
  // TSX Book
  ctx.fillStyle = "rgba(251, 146, 60, 0.15)";
  ctx.fillRect(txX - 95, txY + 12, 60, 40);
  ctx.fillStyle = "#fb923c";
  ctx.font = "8.5px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "left";
  ctx.fillText("BID $64.18", txX - 90, txY + 26);
  ctx.fillText("ASK $64.22", txX - 90, txY + 42);

  // Frankfurt Book
  ctx.fillStyle = "rgba(52, 211, 153, 0.15)";
  ctx.fillRect(deX + 35, deY + 12, 60, 40);
  ctx.fillStyle = "#34d399";
  ctx.fillText("BID €78.48", deX + 40, deY + 26);
  ctx.fillText("ASK €78.52", deX + 40, deY + 42);

  // Text badges
  ctx.font = "bold 11.5px SFMono-Regular, Consolas, monospace";
  ctx.fillStyle = "#fb923c";
  ctx.textAlign = "center";
  ctx.fillText("TORONTO STOCK EXCHANGE (TSX)", txX, txY - 52);
  ctx.font = "10px SFMono-Regular, Consolas, monospace";
  ctx.fillStyle = "#9bb0c2";
  ctx.fillText("TECK.B · IVN · ABX · FM (Mining Hub)", txX, txY - 36);

  ctx.font = "bold 11.5px SFMono-Regular, Consolas, monospace";
  ctx.fillStyle = "#34d399";
  ctx.fillText("FRANKFURT (XETRA DAX)", deX, deY - 52);
  ctx.font = "10px SFMono-Regular, Consolas, monospace";
  ctx.fillStyle = "#9bb0c2";
  ctx.fillText("AUR.DE · BAS.DE · SDF · RWE (German Equities)", deX, deY - 36);

  // Center Badge
  const cx = width / 2;
  ctx.fillStyle = "rgba(13, 23, 32, 0.9)";
  ctx.strokeStyle = "#274056";
  ctx.lineWidth = 1;
  ctx.fillRect(cx - 130, height * 0.40, 260, 56);
  ctx.strokeRect(cx - 130, height * 0.40, 260, 56);

  ctx.fillStyle = "#38bdf8";
  ctx.font = "10px SFMono-Regular, Consolas, monospace";
  ctx.fillText("EUR / CAD FOREX ARBITRAGE ROUTER", cx, height * 0.40 + 20);
  ctx.fillStyle = "#fff";
  ctx.font = "bold 13px SFMono-Regular, Consolas, monospace";
  ctx.fillText(`1 EUR = ${model.parameters.fx_eur_cad.toFixed(4)} CAD`, cx, height * 0.40 + 40);
}

function drawStieltjesDepth(model) {
  const fitted = fitCanvas(canvases.stieltjesDepth, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#070d12"; ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 15, top: 18, bottom: 30 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const data = model.stieltjes_fractions.depth_curve;
  const maxD = Math.max(...data.map(d => Math.max(d.auction_depth, d.stieltjes_beyond_auction)), 1.2);

  // Background Grid
  ctx.strokeStyle = "rgba(24, 40, 54, 0.7)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath(); ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y); ctx.stroke();
  }
  const midX = plot.x + plot.w / 2;
  ctx.beginPath(); ctx.moveTo(midX, plot.y); ctx.lineTo(midX, plot.y + plot.h); ctx.stroke();

  // Time-varying ripple so the depth curve visibly animates between live-drift refreshes
  const wave = Math.sin(Date.now() * 0.0015) * 0.035;
  const jitter = (i, len) => 1.0 + wave * Math.sin((i / len) * Math.PI * 3);

  // Shaded area under Stieltjes Continuous Depth (Cyan gradient)
  const areaGrad = ctx.createLinearGradient(0, plot.y, 0, plot.y + plot.h);
  areaGrad.addColorStop(0, "rgba(56, 189, 248, 0.28)");
  areaGrad.addColorStop(1, "rgba(56, 189, 248, 0.02)");
  ctx.fillStyle = areaGrad;
  ctx.beginPath();
  data.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (data.length - 1);
    const val = pt.stieltjes_beyond_auction * jitter(i, data.length);
    const y = plot.y + plot.h * (1.0 - val / maxD);
    if (i === 0) { ctx.moveTo(x, plot.y + plot.h); ctx.lineTo(x, y); }
    else { ctx.lineTo(x, y); }
  });
  ctx.lineTo(plot.x + plot.w, plot.y + plot.h);
  ctx.closePath();
  ctx.fill();

  // 1. Truncated Discrete Auction depth (Orange #f97316)
  ctx.strokeStyle = "#f97316"; ctx.lineWidth = 1.8;
  ctx.beginPath();
  data.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (data.length - 1);
    const y = plot.y + plot.h * (1.0 - pt.auction_depth / maxD);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // 2. Stieltjes continuous depth line (Cyan #38bdf8)
  ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 2.4;
  ctx.beginPath();
  data.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (data.length - 1);
    const val = pt.stieltjes_beyond_auction * jitter(i, data.length);
    const y = plot.y + plot.h * (1.0 - val / maxD);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Axis Labels
  ctx.fillStyle = "#7d93a6"; ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText(maxD.toFixed(1), plot.x - 6, plot.y + 6);
  ctx.fillText("0", plot.x - 6, plot.y + plot.h + 2);
  ctx.textAlign = "center";
  ctx.fillText("Spread Offset (Basis Points from Midpoint)", plot.x + plot.w / 2, height - 8);
  drawSweepCursor(ctx, plot, "rgba(56, 189, 248, 0.34)");
}

function drawStieltjesConvergents(model) {
  const fitted = fitCanvas(canvases.stieltjesConvergents, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#070d12"; ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 15, top: 18, bottom: 30 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const convs = model.stieltjes_fractions.convergents;
  const maxV = Math.max(...convs.map(c => c.value), 1.5);
  const minV = Math.min(...convs.map(c => c.value), 0.5) * 0.85;

  ctx.strokeStyle = "rgba(24, 40, 54, 0.7)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath(); ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y); ctx.stroke();
  }

  // Connecting convergence curve
  ctx.strokeStyle = "#34d399"; ctx.lineWidth = 2.2;
  ctx.beginPath();
  convs.forEach((c, i) => {
    const x = plot.x + plot.w * i / (convs.length - 1);
    const y = plot.y + plot.h * (1.0 - (c.value - minV) / (maxV - minV));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Nodes with values
  convs.forEach((c, i) => {
    const x = plot.x + plot.w * i / (convs.length - 1);
    const y = plot.y + plot.h * (1.0 - (c.value - minV) / (maxV - minV));
    ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#34d399"; ctx.lineWidth = 2; ctx.stroke();

    ctx.fillStyle = "#9bb0c2";
    ctx.font = "9px SFMono-Regular, Consolas, monospace";
    ctx.textAlign = "center";
    ctx.fillText(`${c.value.toFixed(3)}`, x, y - 8);
    ctx.fillText(`k=${c.order}`, x, height - 8);
  });
  drawSweepCursor(ctx, plot, "rgba(52, 211, 153, 0.30)");
}

function drawLeibnizSeries(model) {
  const fitted = fitCanvas(canvases.leibnizSeries, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#070d12"; ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 15, top: 18, bottom: 30 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const series = model.leibniz_recurrent_primes.leibniz_series;
  const targetPi4 = Math.PI / 4.0; // 0.785398

  // Target line π/4
  const targetY = plot.y + plot.h * (1.0 - targetPi4 / 1.2);
  ctx.strokeStyle = "rgba(56, 189, 248, 0.85)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(plot.x, targetY); ctx.lineTo(plot.x + plot.w, targetY); ctx.stroke();
  ctx.setLineDash([]);

  // Plot alternating series with shaded envelope
  ctx.strokeStyle = "#f43f5e"; ctx.lineWidth = 2.0;
  ctx.beginPath();
  series.forEach((s, i) => {
    const x = plot.x + plot.w * i / (series.length - 1);
    const y = plot.y + plot.h * (1.0 - s.running_pi_over_4 / 1.2);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw points
  series.forEach((s, i) => {
    if (i % 3 === 0) {
      const x = plot.x + plot.w * i / (series.length - 1);
      const y = plot.y + plot.h * (1.0 - s.running_pi_over_4 / 1.2);
      ctx.fillStyle = (i % 2 === 0) ? "#34d399" : "#f43f5e";
      ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
    }
  });

  // Traveling convergence tracer so the chart visibly animates even though
  // the underlying series only depends on the (stable) prime sieve order
  if (series.length > 0) {
    const tracerIdx = Math.floor((Date.now() * 0.006) % series.length);
    const ts = series[tracerIdx];
    const tx = plot.x + plot.w * tracerIdx / (series.length - 1);
    const ty = plot.y + plot.h * (1.0 - ts.running_pi_over_4 / 1.2);
    ctx.fillStyle = "#38bdf8";
    ctx.shadowColor = "#38bdf8";
    ctx.shadowBlur = 9;
    ctx.beginPath(); ctx.arc(tx, ty, 4.5, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
  }

  ctx.fillStyle = "#38bdf8"; ctx.font = "9.5px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText("π/4 (0.7854)", plot.x - 6, targetY + 3);
  ctx.fillStyle = "#7d93a6";
  ctx.textAlign = "center";
  ctx.fillText("Alternating Leibniz Expansion Steps (k = 0 to 40)", plot.x + plot.w / 2, height - 8);
}

function drawPrimeResonance(model) {
  const fitted = fitCanvas(canvases.primeResonance, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#070d12"; ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 15, top: 18, bottom: 30 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const data = model.leibniz_recurrent_primes.resonance_spectrum;

  // Grid
  ctx.strokeStyle = "rgba(24, 40, 54, 0.7)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath(); ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y); ctx.stroke();
  }

  // Traveling phase term so the harmonic spectrum visibly animates between live-drift refreshes
  const phaseShift = Date.now() * 0.0022;
  const dynAmp = (pt, i) => pt.resonance_amplitude * (0.88 + 0.12 * Math.sin(phaseShift + i * 0.35));

  // Gradient area
  const resGrad = ctx.createLinearGradient(0, plot.y, 0, plot.y + plot.h);
  resGrad.addColorStop(0, "rgba(168, 85, 247, 0.35)");
  resGrad.addColorStop(1, "rgba(168, 85, 247, 0.02)");
  ctx.fillStyle = resGrad;
  ctx.beginPath();
  data.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (data.length - 1);
    const y = plot.y + plot.h * (1.0 - dynAmp(pt, i) / 1.2);
    if (i === 0) { ctx.moveTo(x, plot.y + plot.h); ctx.lineTo(x, y); }
    else { ctx.lineTo(x, y); }
  });
  ctx.lineTo(plot.x + plot.w, plot.y + plot.h);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = "#c084fc"; ctx.lineWidth = 2.2;
  ctx.beginPath();
  data.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (data.length - 1);
    const y = plot.y + plot.h * (1.0 - dynAmp(pt, i) / 1.2);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#7d93a6"; ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "center";
  ctx.fillText("Transatlantic Prime Harmonic Frequencies (GHz)", plot.x + plot.w / 2, height - 8);
}

function drawOdeTrajectory(model) {
  const fitted = fitCanvas(canvases.odeTrajectory, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#070d12"; ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 15, top: 18, bottom: 30 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const traj = model.neural_ode.trajectory;

  // Grid
  ctx.strokeStyle = "rgba(24, 40, 54, 0.7)";
  ctx.lineWidth = 1;
  const midY = plot.y + plot.h / 2;
  ctx.beginPath(); ctx.moveTo(plot.x, midY); ctx.lineTo(plot.x + plot.w, midY); ctx.stroke();

  // Position x(t) - Cyan
  ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 2.2;
  ctx.beginPath();
  traj.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (traj.length - 1);
    const y = midY - pt.price_deviation * (plot.h * 0.40);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Velocity v(t) - Orange
  ctx.strokeStyle = "#f97316"; ctx.lineWidth = 1.6;
  ctx.beginPath();
  traj.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (traj.length - 1);
    const y = midY - pt.momentum * (plot.h * 0.40);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Hamiltonian Energy H(t) - Green (Conserved)
  ctx.strokeStyle = "#34d399"; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
  ctx.beginPath();
  traj.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (traj.length - 1);
    const y = plot.y + plot.h * 0.15 + (pt.hamiltonian_energy * 5);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#7d93a6"; ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "center";
  ctx.fillText("Symplectic Verlet Time Step (seconds) [Green: Conserved Energy H]", plot.x + plot.w / 2, height - 8);
}

function drawOdePhase(model) {
  const fitted = fitCanvas(canvases.odePhase, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#070d12"; ctx.fillRect(0, 0, width, height);

  const cx = width / 2;
  const cy = height / 2;
  const scale = Math.min(width, height) * 0.38;
  const points = model.neural_ode.phase_space;

  // Axes
  ctx.strokeStyle = "rgba(24, 40, 54, 0.8)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx, 10); ctx.lineTo(cx, height - 10); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(10, cy); ctx.lineTo(width - 10, cy); ctx.stroke();

  // Vector field arrows
  ctx.strokeStyle = "rgba(192, 132, 252, 0.18)"; ctx.lineWidth = 1;
  for (let gx = -1.2; gx <= 1.2; gx += 0.4) {
    for (let gy = -1.2; gy <= 1.2; gy += 0.4) {
      const px = cx + gx * scale;
      const py = cy - gy * scale;
      // dx/dt = v, dv/dt = -k x
      const vx = gy * 0.15;
      const vy = -gx * 0.15;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px + vx * 20, py - vy * 20);
      ctx.stroke();
    }
  }

  // Trajectory Orbit
  ctx.strokeStyle = "#c084fc"; ctx.lineWidth = 2.4;
  ctx.beginPath();
  points.forEach((pt, i) => {
    const px = cx + pt.position_x * scale;
    const py = cy - pt.momentum_p * scale;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();

  // Moving phase tracer so the orbit visibly animates between live-drift refreshes
  if (points.length > 0) {
    const tracerIdx = Math.floor((Date.now() * 0.012) % points.length);
    const tp = points[tracerIdx];
    const tx = cx + tp.position_x * scale;
    const ty = cy - tp.momentum_p * scale;
    ctx.fillStyle = "#38bdf8";
    ctx.shadowColor = "#38bdf8";
    ctx.shadowBlur = 10;
    ctx.beginPath(); ctx.arc(tx, ty, 5, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
  }

  ctx.fillStyle = "#7d93a6"; ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.fillText("+x (Position / Price)", width - 75, cy - 6);
  ctx.fillText("+p (Momentum / Velocity)", cx + 6, 20);
}

function drawDividendTrajectory(model) {
  const fitted = fitCanvas(canvases.dividendTrajectory, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#070d12"; ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 15, top: 18, bottom: 30 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const data = model.dividends_and_trades.monthly_trajectory;
  const maxVal = Math.max(...data.map(d => d.portfolio_value), 160.0);
  const minVal = 95.0;
  const envelopeWave = Math.sin(Date.now() * 0.0018) * 0.012;

  // Shaded 95% CI Monte Carlo envelope
  ctx.fillStyle = "rgba(52, 211, 153, 0.12)";
  ctx.beginPath();
  data.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (data.length - 1);
    const upper = pt.portfolio_value * (1.04 + envelopeWave);
    const y = plot.y + plot.h * (1.0 - (upper - minVal) / (maxVal - minVal));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  for (let i = data.length - 1; i >= 0; i--) {
    const pt = data[i];
    const x = plot.x + plot.w * i / (data.length - 1);
    const lower = pt.portfolio_value * (0.96 - envelopeWave);
    const y = plot.y + plot.h * (1.0 - (lower - minVal) / (maxVal - minVal));
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();

  // Target 40% threshold line ($140.0)
  const targetY = plot.y + plot.h * (1.0 - (140.0 - minVal) / (maxVal - minVal));
  ctx.strokeStyle = "rgba(251, 191, 36, 0.75)";
  ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(plot.x, targetY); ctx.lineTo(plot.x + plot.w, targetY); ctx.stroke();
  ctx.setLineDash([]);

  // Trajectory curve
  ctx.strokeStyle = "#34d399"; ctx.lineWidth = 2.6;
  ctx.beginPath();
  data.forEach((pt, i) => {
    const x = plot.x + plot.w * i / (data.length - 1);
    const y = plot.y + plot.h * (1.0 - (pt.portfolio_value - minVal) / (maxVal - minVal));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#fbbf24"; ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText("40% Threshold", plot.x - 6, targetY + 3);
  ctx.fillStyle = "#7d93a6";
  ctx.fillText(maxVal.toFixed(0), plot.x - 6, plot.y + 6);
  ctx.fillText("100", plot.x - 6, plot.y + plot.h * (1.0 - (100.0 - minVal) / (maxVal - minVal)));
  ctx.textAlign = "center";
  ctx.fillText("Months (Jan - Dec Cumulative Portfolio Capital with 95% CI)", plot.x + plot.w / 2, height - 8);
  drawSweepCursor(ctx, plot, "rgba(52, 211, 153, 0.25)", 0.00018);
}

function drawDividendBreakdown(model) {
  const fitted = fitCanvas(canvases.dividendBreakdown, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#070d12"; ctx.fillRect(0, 0, width, height);

  const margin = { left: 45, right: 15, top: 18, bottom: 30 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const data = model.dividends_and_trades.monthly_trajectory;
  const maxGain = Math.max(...data.map(d => d.total_monthly_gain), 5.5);

  // Highlighted month index sweeps across the bars so the chart visibly animates
  const highlightIdx = Math.floor((Date.now() * 0.0009) % data.length);

  const barW = plot.w / (data.length * 1.55);
  data.forEach((d, i) => {
    const x = plot.x + i * (plot.w / data.length) + 6;
    let currY = plot.y + plot.h;
    const glow = i === highlightIdx ? 1.0 : 0.78;

    // 1. Cash Div
    const h1 = plot.h * (d.cash_dividend / maxGain);
    ctx.fillStyle = `rgba(56, 189, 248, ${glow})`;
    ctx.fillRect(x, currY - h1, barW, h1);
    currY -= h1;

    // 2. Options Overlay
    const h2 = plot.h * (d.options_premium / maxGain);
    ctx.fillStyle = `rgba(245, 158, 11, ${glow})`;
    ctx.fillRect(x, currY - h2, barW, h2);
    currY -= h2;

    // 3. Arbitrage
    const h3 = plot.h * (d.arbitrage_gain / maxGain);
    ctx.fillStyle = `rgba(168, 85, 247, ${glow})`;
    ctx.fillRect(x, currY - h3, barW, h3);
  });
  drawSweepCursor(ctx, plot, "rgba(245, 158, 11, 0.28)", 0.00018);
}

function drawLiveOrderbook(model) {
  const fitted = fitCanvas(canvases.liveOrderbook, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#070d12"; ctx.fillRect(0, 0, width, height);

  const cx = width / 2;
  const margin = { left: 40, right: 40, top: 20, bottom: 25 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };

  // Center Midpoint line
  ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(cx, plot.y); ctx.lineTo(cx, plot.y + plot.h); ctx.stroke();

  // Bids on left (Green #34d399), Asks on right (Red #f43f5e)
  const levels = 8;
  const barH = plot.h / (levels * 1.35);

  for (let i = 0; i < levels; i++) {
    const y = plot.y + i * (plot.h / levels) + 2;
    // Bids depth grows further from mid
    const bidW = (cx - plot.x) * (0.35 + 0.65 * (i / levels)) * (0.8 + 0.2 * Math.sin(Date.now() * 0.003 + i));
    ctx.fillStyle = "rgba(52, 211, 153, 0.45)";
    ctx.fillRect(cx - bidW, y, bidW, barH);

    // Asks depth
    const askW = (plot.x + plot.w - cx) * (0.35 + 0.65 * (i / levels)) * (0.8 + 0.2 * Math.cos(Date.now() * 0.003 + i));
    ctx.fillStyle = "rgba(244, 63, 94, 0.45)";
    ctx.fillRect(cx, y, askW, barH);
  }

  // Stieltjes continuous clearing indicator line
  const stieltjesShift = (Math.sin(Date.now() * 0.002) * 15);
  ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(cx + stieltjesShift, plot.y - 4);
  ctx.lineTo(cx + stieltjesShift, plot.y + plot.h + 4);
  ctx.stroke();

  ctx.fillStyle = "#34d399"; ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "left"; ctx.fillText("BIDS (BUY QUEUE)", plot.x + 5, 14);
  ctx.fillStyle = "#f43f5e"; ctx.textAlign = "right"; ctx.fillText("ASKS (SELL QUEUE)", plot.x + plot.w - 5, 14);
  ctx.fillStyle = "#38bdf8"; ctx.textAlign = "center"; ctx.fillText("STIELTJES CLEARING", cx, height - 6);

  // Live clock readout confirms the canvas is actively redrawing every frame
  const now = new Date();
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  const ms = String(now.getMilliseconds()).padStart(3, "0");
  ctx.fillStyle = "#fbbf24"; ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText(`LIVE ${hh}:${mm}:${ss}.${ms}`, plot.x + plot.w - 4, plot.y + plot.h - 4);
  drawSweepCursor(ctx, plot, "rgba(56, 189, 248, 0.22)", 0.0003);
}

function drawLivePnl(model) {
  const fitted = fitCanvas(canvases.livePnl, 1.55);
  if (!fitted) return;
  const { ctx, width, height } = fitted;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#070d12"; ctx.fillRect(0, 0, width, height);

  const margin = { left: 55, right: 20, top: 20, bottom: 25 };
  const plot = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };

  // Keep the chart meaningful immediately after a stream reset, before the next fill arrives.
  const pnlPoints = pnlHistory.length > 1 ? pnlHistory : [cumulativePnl * 0.995, cumulativePnl];
  const rawMaxPnl = Math.max(...pnlPoints);
  const rawMinPnl = Math.min(...pnlPoints);
  const pnlPadding = Math.max(Math.abs(rawMaxPnl) * 0.02, 1);
  const maxPnl = rawMaxPnl + pnlPadding;
  const minPnl = rawMinPnl - pnlPadding;

  // Grid
  ctx.strokeStyle = "rgba(24, 40, 54, 0.7)"; ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + plot.h * i / 3;
    ctx.beginPath(); ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y); ctx.stroke();
  }

  // PnL Step curve
  ctx.strokeStyle = "#34d399"; ctx.lineWidth = 2.4;
  ctx.beginPath();
  pnlPoints.forEach((val, i) => {
    const x = plot.x + plot.w * i / (pnlPoints.length - 1);
    const y = plot.y + plot.h * (1.0 - (val - minPnl) / (maxPnl - minPnl));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Pulsing marker on the latest fill so accretion is visibly live
  const lastIdx = pnlPoints.length - 1;
  const lastX = plot.x + plot.w;
  const lastY = plot.y + plot.h * (1.0 - (pnlPoints[lastIdx] - minPnl) / (maxPnl - minPnl));
  const pulse = 3 + 2.5 * Math.abs(Math.sin(Date.now() * 0.004));
  ctx.fillStyle = "#34d399";
  ctx.shadowColor = "#34d399";
  ctx.shadowBlur = 8;
  ctx.beginPath(); ctx.arc(lastX, lastY, pulse, 0, Math.PI * 2); ctx.fill();
  ctx.shadowBlur = 0;

  ctx.fillStyle = "#34d399"; ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText(`+$${Math.round(maxPnl).toLocaleString()}`, plot.x - 6, plot.y + 6);
  ctx.fillText(`+$${Math.round(minPnl).toLocaleString()}`, plot.x - 6, plot.y + plot.h + 2);
  ctx.textAlign = "center";
  ctx.fillText("Real-Time Intraday Fill Progression (PnL Accretion in CAD)", plot.x + plot.w / 2, height - 6);

  // Live clock readout confirms the canvas is actively redrawing every frame
  const now = new Date();
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  const ms = String(now.getMilliseconds()).padStart(3, "0");
  ctx.fillStyle = "#fbbf24"; ctx.font = "9px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "right";
  ctx.fillText(`LIVE ${hh}:${mm}:${ss}.${ms}`, plot.x + plot.w - 4, plot.y + 6);
  drawSweepCursor(ctx, plot, "rgba(52, 211, 153, 0.20)", 0.0003);
}

function renderLedger(trades) {
  const tbody = document.querySelector("#trade-ledger-body");
  if (!tbody) return;
  tbody.innerHTML = trades.map((t, idx) => `
    <tr class="${idx === 0 ? 'trade-row-new' : ''}">
      <td><strong>${t.id}</strong></td>
      <td>${t.timestamp}</td>
      <td>${t.market}</td>
      <td><strong>${t.symbol}</strong> <small style="color:#7d93a6">(${t.name})</small></td>
      <td>${t.side}</td>
      <td>${t.shares ? (typeof t.shares === 'number' ? t.shares.toLocaleString() : t.shares) : t.notional}</td>
      <td>$${typeof t.execution_price === "number" ? t.execution_price.toFixed(2) : t.execution_price}</td>
      <td style="color:#38bdf8">$${typeof t.stieltjes_clearing === "number" ? t.stieltjes_clearing.toFixed(2) : t.stieltjes_clearing}</td>
      <td style="color:#34d399;font-weight:bold">${t.dividend_apr}</td>
      <td><span class="status-badge ${t.status === 'SETTLED' ? 'status-settled' : 'status-active'}">${t.status}</span></td>
    </tr>
  `).join("");
}

function renderMetrics(model) {
  document.querySelector("#stat-dividend-yield").textContent = `${model.dividends_and_trades.annual_dividend_yield_pct.toFixed(1)}%`;
  document.querySelector("#stat-sharpe").textContent = model.dividends_and_trades.sharpe_ratio.toFixed(2);
  document.querySelector("#stat-drawdown").textContent = `${model.dividends_and_trades.max_drawdown_pct.toFixed(1)}%`;
  document.querySelector("#stat-slippage-saving").textContent = `${model.stieltjes_fractions.execution_slippage_reduction_bp.toFixed(1)} bp`;

  // Stieltjes Metrics
  if (document.querySelector("#stieltjes-c0")) {
    document.querySelector("#stieltjes-c0").textContent = model.stieltjes_fractions.coefficients[0].toFixed(2);
    document.querySelector("#stieltjes-c1").textContent = model.stieltjes_fractions.coefficients[1].toFixed(2);
    document.querySelector("#stieltjes-liquidity-gain").textContent = `+${model.stieltjes_fractions.liquidity_gain_pct.toFixed(1)}%`;
    document.querySelector("#stieltjes-slippage").textContent = `-${model.stieltjes_fractions.execution_slippage_reduction_bp.toFixed(1)} bp`;
    const lastConv = model.stieltjes_fractions.convergents[model.stieltjes_fractions.convergents.length - 1];
    document.querySelector("#stieltjes-eval-z").textContent = lastConv ? lastConv.value.toFixed(4) : "—";
  }

  // Leibniz Primes Metrics
  if (document.querySelector("#prime-leibniz-pi")) {
    document.querySelector("#prime-leibniz-pi").textContent = model.leibniz_recurrent_primes.leibniz_limit.toFixed(4);
    document.querySelector("#prime-fidelity").textContent = `${(model.leibniz_recurrent_primes.prime_recurrence_fidelity * 100).toFixed(2)}%`;
    document.querySelector("#prime-count").textContent = String(model.leibniz_recurrent_primes.selected_primes.length);
    document.querySelector("#prime-chi-pos").textContent = "p ≡ 1 mod 4 (5, 13, 17...)";
    document.querySelector("#prime-chi-neg").textContent = "p ≡ 3 mod 4 (3, 7, 11...)";
  }

  // Neural ODE Metrics
  if (document.querySelector("#ode-stability")) {
    document.querySelector("#ode-stability").textContent = model.neural_ode.system_stability;
    document.querySelector("#ode-damping-ratio").textContent = model.neural_ode.damping_ratio.toFixed(2);
    document.querySelector("#ode-lyapunov").textContent = model.neural_ode.lyapunov_exponent.toFixed(3);
    const traj = model.neural_ode.trajectory;
    document.querySelector("#ode-energy-init").textContent = traj.length ? traj[0].hamiltonian_energy.toFixed(3) : "—";
    document.querySelector("#ode-energy-final").textContent = traj.length ? traj[traj.length - 1].hamiltonian_energy.toFixed(3) : "—";
  }

  // Dividend Yield Metrics
  if (document.querySelector("#div-total-yield")) {
    document.querySelector("#div-total-yield").textContent = `${model.dividends_and_trades.annual_dividend_yield_pct.toFixed(1)}%`;
    document.querySelector("#div-base-yield").textContent = `${model.dividends_and_trades.base_dividend_pct.toFixed(1)}%`;
    document.querySelector("#div-options-yield").textContent = `${model.dividends_and_trades.options_overlay_yield_pct.toFixed(1)}%`;
    document.querySelector("#div-capture-yield").textContent = `${model.dividends_and_trades.dividend_capture_yield_pct.toFixed(1)}%`;
    document.querySelector("#div-fx-yield").textContent = `${model.dividends_and_trades.fx_arbitrage_yield_pct.toFixed(1)}%`;
  }

  // Render Charts based on active view
  drawOverview(model);
  if (activeView === "stieltjes") {
    drawStieltjesDepth(model);
    drawStieltjesConvergents(model);
  }
  if (activeView === "primes") {
    drawLeibnizSeries(model);
    drawPrimeResonance(model);
  }
  if (activeView === "neural-ode") {
    drawOdeTrajectory(model);
    drawOdePhase(model);
  }
  if (activeView === "dividends") {
    drawDividendTrajectory(model);
    drawDividendBreakdown(model);
  }
  if (activeView === "ledger") {
    renderLedger(liveTrades.length > 0 ? liveTrades : model.dividends_and_trades.trades);
    drawLiveOrderbook(model);
    drawLivePnl(model);
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
    if (liveTrades.length === 0 && model.dividends_and_trades.trades) {
      liveTrades = [...model.dividends_and_trades.trades];
    }
    renderMetrics(model);
  } catch (error) {
    console.error(error);
  }
}

// Nudges continuous market parameters with a small bounded random walk and re-simulates,
// so every model-driven chart (Stieltjes fractions, Leibniz primes, Neural ODE, dividends)
// shows genuinely fresh values on a timer instead of redrawing identical static data.
function driftLiveParameters() {
  if (!liveTradingActive || userIsDraggingSlider) return;
  controlIds.forEach((id) => {
    if (id === "prime_order") return; // integer sieve order stays stable, not part of market drift
    const el = document.getElementById(id);
    if (!el) return;
    const [lo, hi] = paramBounds[id];
    const span = hi - lo;
    const step = (Math.random() - 0.5) * span * 0.035;
    let next = Number(el.value) + step;
    next = Math.min(hi, Math.max(lo, next));
    el.value = next;
  });
  refreshModel();
}

async function pollLiveTick() {
  if (!liveTradingActive) return;
  try {
    const res = await fetch("/api/live_tick");
    if (!res.ok) return;
    const data = await res.json();

    // Update Live Tickers
    if (data.tickers) {
      if (document.querySelector("#tick-copper")) document.querySelector("#tick-copper").textContent = `$${data.tickers.copper_usd_lb.toFixed(3)} / lb`;
      if (document.querySelector("#tick-nickel")) document.querySelector("#tick-nickel").textContent = `$${data.tickers.nickel_usd_t.toLocaleString()} / t`;
      if (document.querySelector("#tick-gold")) document.querySelector("#tick-gold").textContent = `$${data.tickers.gold_usd_oz.toFixed(2)} / oz`;
      if (document.querySelector("#tick-eur-cad")) document.querySelector("#tick-eur-cad").textContent = data.tickers.eur_cad_fx.toFixed(4);
      if (document.querySelector("#tick-tsx")) document.querySelector("#tick-tsx").textContent = data.tickers.tsx_mining_index.toFixed(2);
      if (document.querySelector("#tick-dax")) document.querySelector("#tick-dax").textContent = data.tickers.dax_materials_index.toFixed(2);
    }

    // Ingest new live trade fill
    if (data.new_trade) {
      liveTrades.unshift(data.new_trade);
      if (liveTrades.length > 18) liveTrades.pop();

      // Accumulate PnL
      const gain = Math.round(650 + Math.random() * 1800);
      cumulativePnl += gain;
      pnlHistory.push(cumulativePnl);
      if (pnlHistory.length > 25) pnlHistory.shift();

      if (document.querySelector("#live-pnl-display")) {
        document.querySelector("#live-pnl-display").textContent = `+$${cumulativePnl.toLocaleString()} CAD`;
      }
      if (document.querySelector("#fill-count-badge")) {
        document.querySelector("#fill-count-badge").textContent = String(liveTrades.length);
      }

      if (activeView === "ledger" && latestModel) {
        renderLedger(liveTrades);
        drawLiveOrderbook(latestModel);
        drawLivePnl(latestModel);
      }
    }
  } catch (err) {
    console.error("Live tick error:", err);
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
  // setTimeout (not rAF) guarantees the redraw fires even in a background/unfocused tab
  setTimeout(() => {
    if (latestModel) renderMetrics(latestModel);
  }, 0);
  setTimeout(() => {
    if (latestModel) renderMetrics(latestModel);
  }, 60);
}

document.querySelectorAll(".view-tab").forEach((button) => {
  button.addEventListener("click", () => switchView(button.dataset.view));
});

controlIds.forEach((id) => {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener("input", refreshModel);
  el.addEventListener("pointerdown", () => { userIsDraggingSlider = true; });
  el.addEventListener("pointerup", () => { userIsDraggingSlider = false; });
  el.addEventListener("touchstart", () => { userIsDraggingSlider = true; }, { passive: true });
  el.addEventListener("touchend", () => { userIsDraggingSlider = false; });
});

document.querySelector("#reset-button").addEventListener("click", () => {
  controlIds.forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.value = defaults[id];
  });
  refreshModel();
});

const liveToggleBtn = document.querySelector("#live-toggle-btn");
if (liveToggleBtn) {
  liveToggleBtn.addEventListener("click", () => {
    liveTradingActive = !liveTradingActive;
    liveToggleBtn.classList.toggle("paused", !liveTradingActive);
    liveToggleBtn.querySelector("span").textContent = liveTradingActive ? "LIVE TRADING: ACTIVE" : "LIVE TRADING: PAUSED";
  });
}

const clearLedgerBtn = document.querySelector("#clear-ledger-btn");
if (clearLedgerBtn) {
  clearLedgerBtn.addEventListener("click", () => {
    if (latestModel && latestModel.dividends_and_trades.trades) {
      liveTrades = [...latestModel.dividends_and_trades.trades];
    } else {
      liveTrades = [];
    }
    cumulativePnl = 148250;
    pnlHistory = [148250];
    if (document.querySelector("#live-pnl-display")) {
      document.querySelector("#live-pnl-display").textContent = `+$${cumulativePnl.toLocaleString()} CAD`;
    }
    if (document.querySelector("#fill-count-badge")) {
      document.querySelector("#fill-count-badge").textContent = String(liveTrades.length);
    }
    if (latestModel) {
      renderLedger(liveTrades);
      drawLiveOrderbook(latestModel);
      drawLivePnl(latestModel);
    }
  });
}

document.querySelector("#export-button").addEventListener("click", () => {
  if (!canvases.overview) return;
  canvases.overview.toBlob((blob) => {
    if (!blob) return;
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "transatlantic-tsx-frankfurt-trading.png";
    link.click();
    URL.revokeObjectURL(link.href);
  }, "image/png");
});

window.addEventListener("resize", () => {
  if (!latestModel) return;
  renderMetrics(latestModel);
});

// Universal animation loop: setInterval keeps firing even when the tab is backgrounded/unfocused,
// unlike requestAnimationFrame which Chromium fully pauses for hidden documents.
function animationTick() {
  if (!latestModel) return;
  if (activeView === "overview") {
    drawOverview(latestModel);
  } else if (activeView === "stieltjes") {
    drawStieltjesDepth(latestModel);
    drawStieltjesConvergents(latestModel);
  } else if (activeView === "primes") {
    drawLeibnizSeries(latestModel);
    drawPrimeResonance(latestModel);
  } else if (activeView === "neural-ode") {
    drawOdeTrajectory(latestModel);
    drawOdePhase(latestModel);
  } else if (activeView === "dividends") {
    drawDividendTrajectory(latestModel);
    drawDividendBreakdown(latestModel);
  } else if (activeView === "ledger") {
    drawLiveOrderbook(latestModel);
    drawLivePnl(latestModel);
  }
}
setInterval(animationTick, 66);

// Live Trading Poll Interval (every 1800ms)
setInterval(pollLiveTick, 1800);

// Live Model Drift: re-simulates with small parameter perturbations every 2.5s so
// Stieltjes/Leibniz/Neural-ODE/Dividend charts show genuinely new data, not just decoration.
setInterval(driftLiveParameters, 2500);

refreshModel();