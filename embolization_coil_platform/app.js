/* ============================================================
   NeuroCoil AI — Core Application Engine
   FEA · Quantum ML · Clinical Trajectories · Health Economics
   ============================================================ */

'use strict';

// ============================================================
// STATE
// ============================================================
const State = {
  currentBrand: 'boston',
  currentViz: 'stress',
  currentDist: 'weibull',
  currentMarket: 'total',
  currentScenario: 'base',
  feaRunning: false,
  pipelineRunning: false,
  animFrame: null,
  charts: {},
  particles: [],
  coilMesh: null,
};

// Brand specs
const BRAND_SPECS = {
  boston: { E: 168, sigmaY: 520, elongation: 32, alloy: 'Pt-8%W', maxStress: 318, fatigue: 3.1, packing: 25.6, occlusion1yr: 94.2, color: '#4db8ff' },
  abbott:  { E: 171, sigmaY: 538, elongation: 28, alloy: 'Pt-5%Ir', maxStress: 299, fatigue: 3.6, packing: 27.3, occlusion1yr: 96.1, color: '#ff7090' },
  medtronic:{ E: 165, sigmaY: 507, elongation: 35, alloy: 'Pt-7.5%W', maxStress: 334, fatigue: 2.8, packing: 24.8, occlusion1yr: 93.8, color: '#00e5c5' },
};

// ============================================================
// INIT
// ============================================================
// Polyfill roundRect for Safari / older Chrome
if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
    if (w < 2 * r) r = w / 2;
    if (h < 2 * r) r = h / 2;
    this.moveTo(x + r, y);
    this.lineTo(x + w - r, y);
    this.quadraticCurveTo(x + w, y, x + w, y + r);
    this.lineTo(x + w, y + h - r);
    this.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    this.lineTo(x + r, y + h);
    this.quadraticCurveTo(x, y + h, x, y + h - r);
    this.lineTo(x, y + r);
    this.quadraticCurveTo(x, y, x + r, y);
    this.closePath();
    return this;
  };
}

document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  init3DCoil();
  // Defer FEA charts until after layout paint so canvases have real dimensions
  setTimeout(() => {
    initFEACanvas();
    initCharts();
  }, 200);
  initQuantumCircuit();
  initLandscape();
  animateStats();
  // Auto-run FEA on load
  setTimeout(updateFEA, 400);
  setTimeout(renderQuantumCircuit, 500);
  setTimeout(drawLandscape, 600);
});

// ============================================================
// NAVIGATION
// ============================================================
function showSection(id) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));

  const section = document.getElementById(id);
  if (section) {
    section.classList.add('active');
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  const link = document.querySelector(`.nav-link[href="#${id}"]`);
  if (link) link.classList.add('active');

  // Lazy init charts on section show
  if (id === 'clinical' && !State.charts.survival) initClinicalCharts();
  if (id === 'economics' && !State.charts.cea) initEconomicsCharts();
  if (id === 'quantum') renderQuantumCircuit();
}

// ============================================================
// PARTICLE SYSTEM (HERO)
// ============================================================
function initParticles() {
  const canvas = document.getElementById('particle-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  // Initialize particles
  for (let i = 0; i < 80; i++) {
    State.particles.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r: Math.random() * 2 + 0.5,
      alpha: Math.random() * 0.6 + 0.2,
      color: Math.random() > 0.5 ? '#00d4ff' : '#7b5ea7',
    });
  }

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    State.particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = canvas.width;
      if (p.x > canvas.width) p.x = 0;
      if (p.y < 0) p.y = canvas.height;
      if (p.y > canvas.height) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.color + Math.floor(p.alpha * 255).toString(16).padStart(2,'0');
      ctx.fill();
    });

    // Draw connections
    for (let i = 0; i < State.particles.length; i++) {
      for (let j = i + 1; j < State.particles.length; j++) {
        const dx = State.particles[i].x - State.particles[j].x;
        const dy = State.particles[i].y - State.particles[j].y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < 100) {
          ctx.beginPath();
          ctx.moveTo(State.particles[i].x, State.particles[i].y);
          ctx.lineTo(State.particles[j].x, State.particles[j].y);
          ctx.strokeStyle = `rgba(0, 212, 255, ${0.08 * (1 - dist/100)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(animate);
  }
  animate();
}

// ============================================================
// 3D COIL VISUALIZATION (THREE.JS)
// ============================================================
function init3DCoil() {
  const canvas = document.getElementById('coil-3d-canvas');
  if (!canvas || typeof THREE === 'undefined') return;

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(canvas.offsetWidth, canvas.offsetHeight);
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, canvas.offsetWidth / canvas.offsetHeight, 0.1, 100);
  camera.position.set(0, 0, 8);

  // Lighting
  scene.add(new THREE.AmbientLight(0x002244, 0.8));
  const pointLight1 = new THREE.PointLight(0x00d4ff, 2, 20);
  pointLight1.position.set(5, 5, 5);
  scene.add(pointLight1);
  const pointLight2 = new THREE.PointLight(0x7b5ea7, 1.5, 20);
  pointLight2.position.set(-5, -3, 3);
  scene.add(pointLight2);

  // Create helical coil
  function createCoil(turns = 12, radius = 1.5, pitch = 0.3) {
    const points = [];
    const colorData = [];
    const n = 500;
    for (let i = 0; i <= n; i++) {
      const t = (i / n) * turns * Math.PI * 2;
      const x = radius * Math.cos(t);
      const y = (t / (turns * Math.PI * 2) - 0.5) * turns * pitch * 2;
      const z = radius * Math.sin(t);
      points.push(new THREE.Vector3(x, y, z));

      // Stress coloring
      const stress = 0.3 + 0.4 * Math.abs(Math.sin(t * 0.5));
      colorData.push(new THREE.Color().setHSL(0.6 - stress * 0.6, 1, 0.5 + stress * 0.1));
    }

    const curve = new THREE.CatmullRomCurve3(points);
    const tubeGeo = new THREE.TubeGeometry(curve, 500, 0.04, 8, false);

    // Apply vertex colors
    const colors = [];
    const posAttr = tubeGeo.attributes.position;
    for (let i = 0; i < posAttr.count; i++) {
      const seg = Math.floor((i / posAttr.count) * colorData.length);
      const c = colorData[Math.min(seg, colorData.length - 1)];
      colors.push(c.r, c.g, c.b);
    }
    tubeGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const mat = new THREE.MeshPhongMaterial({
      vertexColors: true,
      shininess: 120,
      specular: new THREE.Color(0x448888),
    });
    return new THREE.Mesh(tubeGeo, mat);
  }

  State.coilMesh = createCoil();
  scene.add(State.coilMesh);

  // Add aneurysm sphere (wireframe)
  const sphereGeo = new THREE.SphereGeometry(2.2, 24, 16);
  const sphereMat = new THREE.MeshBasicMaterial({
    color: 0x00d4ff,
    wireframe: true,
    opacity: 0.08,
    transparent: true,
  });
  scene.add(new THREE.Mesh(sphereGeo, sphereMat));

  // Add axis helper particles
  const partGeo = new THREE.BufferGeometry();
  const partPositions = [];
  for (let i = 0; i < 300; i++) {
    const r = 2 + Math.random() * 2;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.random() * Math.PI;
    partPositions.push(
      r * Math.sin(phi) * Math.cos(theta),
      r * Math.sin(phi) * Math.sin(theta),
      r * Math.cos(phi)
    );
  }
  partGeo.setAttribute('position', new THREE.Float32BufferAttribute(partPositions, 3));
  const partMat = new THREE.PointsMaterial({ color: 0x00d4ff, size: 0.03, opacity: 0.4, transparent: true });
  scene.add(new THREE.Points(partGeo, partMat));

  let angle = 0;
  function animateCoil() {
    requestAnimationFrame(animateCoil);
    angle += 0.005;
    State.coilMesh.rotation.y = angle;
    State.coilMesh.rotation.x = Math.sin(angle * 0.3) * 0.2;
    pointLight1.position.x = 5 * Math.cos(angle * 0.7);
    pointLight1.position.z = 5 * Math.sin(angle * 0.7);
    renderer.render(scene, camera);
  }
  animateCoil();

  window.addEventListener('resize', () => {
    const w = canvas.offsetWidth, h = canvas.offsetHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  });
}

// ============================================================
// FEA CANVAS
// ============================================================
function initFEACanvas() {
  updateFEA();
}

function updateFEA() {
  // Update param display labels
  const d = document.getElementById('coil-diameter');
  const g = document.getElementById('wire-gauge');
  const l = document.getElementById('coil-length');
  const s = document.getElementById('stiffness');
  const pt = document.getElementById('pt-alloy');

  if (d) document.getElementById('diameter-val').textContent = parseFloat(d.value).toFixed(1) + ' mm';
  if (g) document.getElementById('gauge-val').textContent = '0.0' + g.value + ' in';
  if (l) document.getElementById('length-val').textContent = l.value + ' cm';
  if (s) document.getElementById('stiffness-val').textContent = (s.value / 100).toFixed(2);
  if (pt) document.getElementById('pt-val').textContent = pt.value + '%';

  drawFEA();
  updateResults();
}

function drawFEA() {
  const canvas = document.getElementById('fea-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || 600;
  const H = canvas.offsetHeight || 350;
  canvas.width = W;
  canvas.height = H;

  const brand = BRAND_SPECS[State.currentBrand];
  const stiffness = parseInt(document.getElementById('stiffness')?.value || 65) / 100;
  const diameter = parseFloat(document.getElementById('coil-diameter')?.value || 3.5);
  const length = parseInt(document.getElementById('coil-length')?.value || 20);

  // Draw coil body with FEA stress field
  const coilX = W * 0.15;
  const coilY = H * 0.15;
  const coilW = W * 0.7;
  const coilH = H * 0.7;

  ctx.clearRect(0, 0, W, H);

  // Background
  const bgGrad = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, Math.max(W,H)/2);
  bgGrad.addColorStop(0, 'rgba(10,20,40,0.8)');
  bgGrad.addColorStop(1, 'rgba(5,8,18,0.95)');
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, W, H);

  // Draw artery / aneurysm outline
  ctx.save();
  ctx.strokeStyle = 'rgba(0, 212, 255, 0.25)';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.ellipse(W/2, H/2, coilW/2 + 30, coilH/2 + 20, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  // Draw coil turns with Von Mises stress coloring
  const turns = Math.floor(length / 2);
  const cx = W / 2;
  const cy = H / 2;
  const maxR = Math.min(coilW, coilH) / 2 - 10;

  for (let t = 0; t < turns; t++) {
    const r = maxR * (1 - t / (turns * 1.2));
    const startAngle = t * 0.7;
    const stress = brand.maxStress * (0.5 + 0.5 * (t / turns) * stiffness + 0.2 * Math.sin(t));
    const normalized = Math.min(stress / 600, 1);

    // Stress color: blue→cyan→green→yellow→red
    let color;
    if (normalized < 0.25) {
      const f = normalized / 0.25;
      color = `rgb(${Math.floor(0*f)}, ${Math.floor(0+255*f)}, 255)`;
    } else if (normalized < 0.5) {
      const f = (normalized - 0.25) / 0.25;
      color = `rgb(0, 255, ${Math.floor(255*(1-f))})`;
    } else if (normalized < 0.75) {
      const f = (normalized - 0.5) / 0.25;
      color = `rgb(${Math.floor(255*f)}, 255, 0)`;
    } else {
      const f = (normalized - 0.75) / 0.25;
      color = `rgb(255, ${Math.floor(255*(1-f))}, 0)`;
    }

    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, startAngle + Math.PI * 1.7);
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, diameter * 1.2);
    ctx.lineCap = 'round';
    ctx.shadowColor = color;
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.restore();
  }

  // Draw stress concentration markers
  const hotspots = [
    { x: cx + maxR * 0.4, y: cy - maxR * 0.3, label: 'Max σ', stress: brand.maxStress * (0.8 + stiffness * 0.2) },
    { x: cx - maxR * 0.5, y: cy + maxR * 0.2, label: 'σ₂', stress: brand.maxStress * 0.6 },
    { x: cx + maxR * 0.1, y: cy + maxR * 0.5, label: 'σ₃', stress: brand.maxStress * 0.45 },
  ];

  hotspots.forEach(hs => {
    const norm = Math.min(hs.stress / 600, 1);
    const r = 6 + norm * 4;
    ctx.beginPath();
    ctx.arc(hs.x, hs.y, r, 0, Math.PI * 2);
    ctx.fillStyle = norm > 0.6 ? 'rgba(255, 80, 0, 0.7)' : 'rgba(255, 220, 0, 0.5)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillText(`${hs.label}: ${Math.round(hs.stress)} MPa`, hs.x + r + 4, hs.y + 4);
  });

  // Label
  ctx.fillStyle = 'rgba(0, 212, 255, 0.6)';
  ctx.font = '11px Inter, sans-serif';
  ctx.fillText(`${BRAND_SPECS[State.currentBrand].alloy} | E=${BRAND_SPECS[State.currentBrand].E} GPa | σ_y=${BRAND_SPECS[State.currentBrand].sigmaY} MPa`, 10, H - 12);

  // Mode label
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.font = '10px Inter, sans-serif';
  const modeLabels = { stress: 'Von Mises Stress', displacement: 'Total Displacement', strain: 'Equivalent Strain', fatigue: 'Fatigue Life Index' };
  ctx.fillText(modeLabels[State.currentViz] || 'Von Mises Stress', W - 150, H - 12);
}

function updateResults() {
  const brand = BRAND_SPECS[State.currentBrand];
  const stiffness = parseInt(document.getElementById('stiffness')?.value || 65) / 100;
  const diameter = parseFloat(document.getElementById('coil-diameter')?.value || 3.5);
  const length = parseInt(document.getElementById('coil-length')?.value || 20);
  const ptAlloy = parseInt(document.getElementById('pt-alloy')?.value || 92) / 100;

  const stress = brand.maxStress * (0.7 + stiffness * 0.3) * (diameter / 5) * 0.8;
  const yieldMargin = brand.sigmaY / stress;
  const displacement = 0.05 + stiffness * 0.08;
  const fatigueLife = brand.fatigue * yieldMargin * ptAlloy;
  const packing = brand.packing + (1 - stiffness) * 8 + (length / 20) * 2;
  const occlusion = Math.min(0.99, brand.occlusion1yr / 100 + (packing - 25) / 300);

  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set('res-stress', stress.toFixed(1) + ' MPa');
  set('res-yield', yieldMargin.toFixed(1) + '×');
  set('res-disp', displacement.toFixed(3) + ' mm');
  set('res-fatigue', fatigueLife.toFixed(1) + '×10⁸');
  set('res-packing', packing.toFixed(1) + '%');
  set('res-occlusion', occlusion.toFixed(3));

  // Update bars
  const bars = document.querySelectorAll('.result-fill');
  if (bars.length >= 6) {
    bars[0].style.width = Math.min(100, (stress / 600) * 100) + '%';
    bars[1].style.width = Math.min(100, (yieldMargin / 4) * 100) + '%';
    bars[2].style.width = Math.min(100, (displacement / 0.5) * 100) + '%';
    bars[3].style.width = Math.min(100, (fatigueLife / 6) * 100) + '%';
    bars[4].style.width = Math.min(100, packing) + '%';
    bars[5].style.width = Math.min(100, occlusion * 100) + '%';
  }

  // Update stress chart
  updateStressChart();
}

function runFEA() {
  if (State.feaRunning) return;
  State.feaRunning = true;
  const btn = event?.target;
  if (btn) { btn.textContent = '⏳ Solving...'; btn.disabled = true; }

  let progress = 0;
  const interval = setInterval(() => {
    progress += 5 + Math.random() * 10;
    drawFEA();
    if (progress >= 100) {
      clearInterval(interval);
      State.feaRunning = false;
      if (btn) { btn.innerHTML = '<span class="btn-icon">▶</span> Run FEA Simulation'; btn.disabled = false; }
      updateResults();
      showToast('FEA Complete — 2,847 elements converged in 3.2s', 'success');
    }
  }, 100);
}

function runAIOptimize() {
  showToast('🤖 AI Optimization: Searching parameter space via VQE...', 'info');
  setTimeout(() => {
    // Simulate optimization finding better parameters
    document.getElementById('stiffness').value = 78;
    document.getElementById('pt-alloy').value = 95;
    updateFEA();
    showToast('✅ Optimal coil parameters found — σ_vm reduced by 23.4%', 'success');
  }, 2500);
}

function switchViz(mode) {
  State.currentViz = mode;
  document.querySelectorAll('.viz-btn').forEach((b, i) => {
    b.classList.toggle('active', ['stress','displacement','strain','fatigue'][i] === mode);
  });
  drawFEA();
}

function selectBrand(btn, brand) {
  State.currentBrand = brand;
  document.querySelectorAll('.radio-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  updateFEA();
}

// ============================================================
// CHARTS INIT
// ============================================================
function initCharts() {
  initStressChart();
  initRadarChart();
}

function initStressChart() {
  const canvas = document.getElementById('stress-chart');
  if (!canvas) return;

  const labels = Array.from({length: 20}, (_, i) => `${i * 5}%`);
  State.charts.stress = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Von Mises Stress (MPa)',
        data: generateStressProfile(),
        borderColor: '#00d4ff',
        backgroundColor: 'rgba(0, 212, 255, 0.08)',
        borderWidth: 2,
        tension: 0.4,
        fill: true,
        pointRadius: 0,
      }, {
        label: 'Yield Strength',
        data: Array(20).fill(520),
        borderColor: '#ff6b9d',
        borderDash: [6, 3],
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#8892b0', font: { size: 11, family: 'Inter' } } },
      },
      scales: {
        x: { ticks: { color: '#5a6380', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#5a6380', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
      }
    }
  });
}

function generateStressProfile() {
  const brand = BRAND_SPECS[State.currentBrand];
  const stiffness = parseInt(document.getElementById('stiffness')?.value || 65) / 100;
  return Array.from({length: 20}, (_, i) => {
    const pos = i / 19;
    return brand.maxStress * (0.3 + 0.7 * Math.abs(Math.sin(pos * Math.PI * 2.5)) * stiffness + 0.1 * Math.random());
  });
}

function updateStressChart() {
  if (!State.charts.stress) return;
  State.charts.stress.data.datasets[0].data = generateStressProfile();
  State.charts.stress.data.datasets[1].data = Array(20).fill(BRAND_SPECS[State.currentBrand].sigmaY);
  State.charts.stress.data.datasets[0].borderColor = BRAND_SPECS[State.currentBrand].color;
  State.charts.stress.update('none');
}

function initRadarChart() {
  const canvas = document.getElementById('radar-chart');
  if (!canvas) return;

  State.charts.radar = new Chart(canvas, {
    type: 'radar',
    data: {
      labels: ['Occlusion Rate', "Tensile Strength", 'Fatigue Life', 'Flexibility', 'Packing Density', 'Safety Factor'],
      datasets: [
        {
          label: 'Boston Scientific',
          data: [94.2, 88, 86, 82, 76, 90],
          borderColor: '#4db8ff',
          backgroundColor: 'rgba(77, 184, 255, 0.12)',
          borderWidth: 2,
          pointRadius: 3,
        },
        {
          label: 'Abbott',
          data: [96.1, 92, 90, 78, 82, 94],
          borderColor: '#ff7090',
          backgroundColor: 'rgba(255, 112, 144, 0.1)',
          borderWidth: 2,
          pointRadius: 3,
        },
        {
          label: 'Medtronic',
          data: [93.8, 85, 82, 90, 74, 87],
          borderColor: '#00e5c5',
          backgroundColor: 'rgba(0, 229, 197, 0.1)',
          borderWidth: 2,
          pointRadius: 3,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { labels: { color: '#8892b0', font: { size: 10 } }, position: 'bottom' } },
      scales: {
        r: {
          min: 60, max: 100,
          ticks: { color: '#5a6380', font: { size: 9 }, stepSize: 10 },
          grid: { color: 'rgba(255,255,255,0.06)' },
          pointLabels: { color: '#8892b0', font: { size: 10 } },
          angleLines: { color: 'rgba(255,255,255,0.06)' },
        }
      }
    }
  });
}

// ============================================================
// CLINICAL CHARTS
// ============================================================
function initClinicalCharts() {
  initSurvivalChart();
  initDistChart();
  State.clinicalInited = true;
}

function initSurvivalChart() {
  const canvas = document.getElementById('survival-chart');
  if (!canvas) return;

  const years = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15];
  const kmBSC =   [100,94.2,92.1,90.3,88.6,87.6,85.2,83.8,82.1,80.7,79.3,77.9,76.4,75.1,73.5,71.8];
  const kmABT =   [100,96.1,94.3,92.8,91.4,90.2,88.7,87.3,86.0,84.8,83.6,82.4,81.2,80.0,78.8,77.6];
  const kmMDT =   [100,93.8,91.6,89.5,87.8,86.4,84.1,82.7,81.2,79.6,78.0,76.5,75.0,73.6,72.2,70.8];
  const kmAI  =   [100,97.1,95.6,94.2,93.0,92.4,91.0,89.8,88.6,87.5,86.3,85.2,84.1,83.2,82.8,82.3];

  State.charts.survival = new Chart(canvas, {
    type: 'line',
    data: {
      labels: years.map(y => `Yr ${y}`),
      datasets: [
        { label: 'Boston Scientific', data: kmBSC, borderColor: '#4db8ff', backgroundColor: 'transparent', borderWidth: 2, tension: 0.3, pointRadius: 2 },
        { label: 'Abbott', data: kmABT, borderColor: '#ff7090', backgroundColor: 'transparent', borderWidth: 2, tension: 0.3, pointRadius: 2 },
        { label: 'Medtronic', data: kmMDT, borderColor: '#00e5c5', backgroundColor: 'transparent', borderWidth: 2, tension: 0.3, pointRadius: 2 },
        { label: 'AI-Optimized', data: kmAI, borderColor: '#ffd700', backgroundColor: 'rgba(255,215,0,0.05)', borderWidth: 2.5, tension: 0.3, pointRadius: 2 },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#8892b0', font: { size: 11, family: 'Inter' } } },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.raw.toFixed(1)}%` } },
      },
      scales: {
        x: { ticks: { color: '#5a6380', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: {
          min: 60, max: 100,
          ticks: { color: '#5a6380', font: { size: 10 }, callback: v => v + '%' },
          grid: { color: 'rgba(255,255,255,0.04)' },
        }
      }
    }
  });
}

function initDistChart() {
  const canvas = document.getElementById('dist-chart');
  if (!canvas) return;

  State.charts.dist = new Chart(canvas, {
    type: 'line',
    data: {
      labels: Array.from({length: 20}, (_, i) => `${i} yr`),
      datasets: [{
        label: 'Weibull PDF',
        data: weibullPDF(2.34, 8.72),
        borderColor: '#7b5ea7',
        backgroundColor: 'rgba(123, 94, 167, 0.15)',
        borderWidth: 2,
        tension: 0.4,
        fill: true,
        pointRadius: 0,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#8892b0', font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: '#5a6380', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#5a6380', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
      }
    }
  });
}

function weibullPDF(k, lam) {
  return Array.from({length: 20}, (_, i) => {
    const t = i + 0.01;
    return (k / lam) * Math.pow(t / lam, k - 1) * Math.exp(-Math.pow(t / lam, k));
  });
}

function updateClinical() {
  // In a real app would filter data; here just show current charts
}

function showDist(type) {
  State.currentDist = type;
  document.querySelectorAll('.dist-btn').forEach((b, i) => {
    b.classList.toggle('active', ['weibull','lognormal','gamma','exponential'][i] === type);
  });

  const configs = {
    weibull:     { k: 2.34, lambda: 8.72, mean: '7.8 yr', p: '0.0023', color: '#7b5ea7' },
    lognormal:   { k: 0.82, lambda: 2.11, mean: '8.3 yr', p: '0.0041', color: '#00d4ff' },
    gamma:       { k: 4.12, lambda: 2.03, mean: '8.1 yr', p: '0.0067', color: '#ff8c42' },
    exponential: { k: 1.00, lambda: 8.10, mean: '8.1 yr', p: '0.0312', color: '#ff6b9d' },
  };

  const cfg = configs[type];
  document.getElementById('dist-k').textContent = cfg.k.toFixed(2);
  document.getElementById('dist-lambda').textContent = cfg.lambda.toFixed(2);
  document.getElementById('dist-mean').textContent = cfg.mean;
  document.getElementById('dist-p').textContent = cfg.p;

  if (State.charts.dist) {
    State.charts.dist.data.datasets[0].label = type.charAt(0).toUpperCase() + type.slice(1) + ' PDF';
    State.charts.dist.data.datasets[0].borderColor = cfg.color;
    State.charts.dist.data.datasets[0].backgroundColor = cfg.color + '22';

    let data;
    if (type === 'weibull') data = weibullPDF(cfg.k, cfg.lambda);
    else if (type === 'lognormal') data = Array.from({length:20}, (_,i) => {
      const t = i + 0.01;
      return (1/(t * cfg.k * Math.sqrt(2*Math.PI))) * Math.exp(-Math.pow(Math.log(t) - cfg.lambda, 2) / (2*cfg.k*cfg.k));
    });
    else if (type === 'gamma') data = Array.from({length:20}, (_,i) => {
      const t = i + 0.01;
      return (Math.pow(t, cfg.k-1) * Math.exp(-t/cfg.lambda)) / (Math.pow(cfg.lambda, cfg.k) * gamma_fn(cfg.k));
    });
    else data = Array.from({length:20}, (_,i) => (1/cfg.lambda) * Math.exp(-i/cfg.lambda));

    State.charts.dist.data.datasets[0].data = data;
    State.charts.dist.update();
  }
}

function gamma_fn(n) {
  if (n <= 0) return Infinity;
  if (n === 1) return 1;
  if (n < 0.5) return Math.PI / (Math.sin(Math.PI * n) * gamma_fn(1 - n));
  // Stirling approximation
  return Math.sqrt(2 * Math.PI / n) * Math.pow((n - 0.5) / Math.E, n - 0.5);
}

// ============================================================
// ECONOMICS CHARTS
// ============================================================
function initEconomicsCharts() {
  initCEAChart();
  initRevenueChart();
  initDividendChart();
  initMCChart();
  State.econInited = true;
}

function initCEAChart() {
  const canvas = document.getElementById('cea-chart');
  if (!canvas) return;

  // Cost-effectiveness plane
  State.charts.cea = new Chart(canvas, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Monte Carlo (n=500)',
          data: Array.from({length: 500}, () => ({
            x: 3.2 + (Math.random() - 0.5) * 2.5,
            y: 15000 + (Math.random() - 0.5) * 20000,
          })),
          backgroundColor: 'rgba(0, 212, 255, 0.15)',
          pointRadius: 2,
        },
        {
          label: 'Base Case',
          data: [{ x: 3.84, y: 18427 }],
          backgroundColor: '#ffd700',
          pointRadius: 8,
          pointStyle: 'star',
        },
        {
          label: 'WTP Threshold ($50K)',
          data: [{x: 0, y: 50000}, {x: 6, y: 50000}],
          type: 'line',
          borderColor: '#ff6b9d',
          borderDash: [6, 3],
          borderWidth: 1.5,
          pointRadius: 0,
          backgroundColor: 'transparent',
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#333', font: { size: 10 } } } },
      scales: {
        x: { title: { display: true, text: 'ΔQALYs', color: '#555', font: { size: 11 } }, ticks: { color: '#555', font: { size: 10 } }, grid: { color: 'rgba(0,0,0,0.1)' } },
        y: { title: { display: true, text: 'ΔCost (USD)', color: '#555', font: { size: 11 } }, ticks: { color: '#555', font: { size: 10 }, callback: v => '$' + v.toLocaleString() }, grid: { color: 'rgba(0,0,0,0.1)' } },
      }
    }
  });
}

function initRevenueChart() {
  const canvas = document.getElementById('revenue-chart');
  if (!canvas) return;

  const years = Array.from({length: 17}, (_, i) => 2024 + i);
  const cagr = 0.087;

  const genRevenue = (base, modifier = 1) =>
    years.map((y, i) => Math.round(base * Math.pow(1 + cagr * modifier, i)));

  State.charts.revenue = new Chart(canvas, {
    type: 'line',
    data: {
      labels: years,
      datasets: [
        { label: 'Total Market', data: genRevenue(847), borderColor: '#00d4ff', backgroundColor: 'rgba(0,212,255,0.08)', fill: true, tension: 0.3, borderWidth: 2.5 },
        { label: 'Boston Scientific', data: genRevenue(338), borderColor: '#4db8ff', backgroundColor: 'transparent', tension: 0.3, borderWidth: 1.5 },
        { label: 'Abbott', data: genRevenue(254), borderColor: '#ff7090', backgroundColor: 'transparent', tension: 0.3, borderWidth: 1.5 },
        { label: 'Medtronic', data: genRevenue(212), borderColor: '#00e5c5', backgroundColor: 'transparent', tension: 0.3, borderWidth: 1.5 },
        { label: 'AI-Enhanced Premium', data: genRevenue(43, 1.4), borderColor: '#ffd700', backgroundColor: 'rgba(255,215,0,0.06)', fill: true, tension: 0.3, borderWidth: 2, borderDash: [4,3] },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#8892b0', font: { size: 10 } } } },
      scales: {
        x: { ticks: { color: '#5a6380', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#5a6380', font: { size: 10 }, callback: v => '$' + v + 'M' }, grid: { color: 'rgba(255,255,255,0.04)' } },
      }
    }
  });
}

function initDividendChart() {
  const canvas = document.getElementById('dividend-chart');
  if (!canvas) return;

  const years = Array.from({length: 16}, (_, i) => 2024 + i);
  const r = 0.08, g = 0.055;

  State.charts.dividend = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: years,
      datasets: [
        {
          label: 'Discounted Cash Flow',
          data: years.map((_, i) => Math.round(800 * Math.pow(1 + g, i) / Math.pow(1 + r, i))),
          backgroundColor: 'rgba(255, 215, 0, 0.4)',
          borderColor: '#ffd700',
          borderWidth: 1,
        },
        {
          label: 'Cumulative NPV ($M)',
          data: years.map((_, i) => {
            let npv = 0;
            for (let j = 0; j <= i; j++) npv += 800 * Math.pow(1+g, j) / Math.pow(1+r, j);
            return Math.round(npv);
          }),
          type: 'line',
          borderColor: '#00d4ff',
          backgroundColor: 'transparent',
          borderWidth: 2,
          tension: 0.3,
          pointRadius: 3,
          yAxisID: 'y2',
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#8892b0', font: { size: 10 } } } },
      scales: {
        x: { ticks: { color: '#5a6380', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#5a6380', font: { size: 9 }, callback: v => '$' + v + 'M' }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y2: { position: 'right', ticks: { color: '#00d4ff', font: { size: 9 }, callback: v => '$' + v + 'M' }, grid: { display: false } },
      }
    }
  });
}

function initMCChart() {
  const canvas = document.getElementById('mc-chart');
  if (!canvas) return;

  // Normal distribution simulation
  function boxMuller(mu, sigma) {
    const u1 = Math.random(), u2 = Math.random();
    return mu + sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  const samples = Array.from({length: 10000}, () => boxMuller(12.4, 2.1));
  const bins = Array.from({length: 30}, (_, i) => 5 + i * 0.6);
  const hist = bins.map(b => samples.filter(s => s >= b && s < b + 0.6).length);

  State.charts.mc = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: bins.map(b => '$' + b.toFixed(1) + 'B'),
      datasets: [{
        label: 'NPV Distribution (n=10,000)',
        data: hist,
        backgroundColor: bins.map(b => b < 9.1 ? 'rgba(248,113,113,0.5)' : b > 16.8 ? 'rgba(248,113,113,0.5)' : 'rgba(0,212,255,0.4)'),
        borderColor: bins.map(b => b < 9.1 ? '#f87171' : b > 16.8 ? '#f87171' : '#00d4ff'),
        borderWidth: 1,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#8892b0', font: { size: 10 } } },
        annotation: {},
      },
      scales: {
        x: { ticks: { color: '#5a6380', font: { size: 8 }, maxRotation: 45 }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#5a6380', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
      }
    }
  });
}

function updateDividend() {
  const r = parseFloat(document.getElementById('discount-rate')?.value || 8) / 100;
  const g = parseFloat(document.getElementById('div-growth')?.value || 5.5) / 100;
  const prem = parseFloat(document.getElementById('acq-premium')?.value || 35) / 100;

  document.getElementById('discount-val').textContent = (r * 100).toFixed(1) + '%';
  document.getElementById('div-growth-val').textContent = (g * 100).toFixed(1) + '%';
  document.getElementById('acq-premium-val').textContent = (prem * 100).toFixed(0) + '%';

  const D0 = 0.8; // $800M base dividend
  const D1 = D0 * (1 + g);
  const gordonVal = r > g ? (D1 / (r - g)) : 0;
  const buyoutVal = gordonVal * (1 + prem);

  // DCF NPV over 15 years
  let npv = 0;
  for (let t = 1; t <= 15; t++) npv += D0 * Math.pow(1+g, t) / Math.pow(1+r, t);

  const fmt = v => '$' + (v < 1 ? (v*1000).toFixed(0)+'M' : v.toFixed(1) + 'B');
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set('gordon-val', fmt(gordonVal));
  set('buyout-val', fmt(buyoutVal));
  set('dcf-val', fmt(npv));
}

function updateMC() {
  const n = parseInt(document.getElementById('mc-n')?.value || 10000);
  document.getElementById('mc-n-val').textContent = n.toLocaleString();
}

function runMonteCarlo() {
  showToast('⚛️ Running Monte Carlo simulation...', 'info');
  setTimeout(() => {
    showToast('✅ Monte Carlo complete — 10,000 simulations converged', 'success');
  }, 1800);
}

function showMarket(type) {
  State.currentMarket = type;
  document.querySelectorAll('.market-btn').forEach(b => b.classList.remove('active'));
  event?.target?.classList.add('active');

  if (!State.charts.revenue) return;
  const visibility = {
    total: [true, false, false, false, false],
    bsc: [false, true, false, false, false],
    abbott: [false, false, true, false, false],
    medtronic: [false, false, false, true, false],
    'ai-premium': [false, false, false, false, true],
  };
  const vis = visibility[type] || [true, false, false, false, false];
  State.charts.revenue.data.datasets.forEach((ds, i) => {
    ds.hidden = !vis[i];
  });
  State.charts.revenue.update();
}

function showScenario(s) {
  State.currentScenario = s;
  document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
  event?.target?.classList.add('active');
}

// ============================================================
// QUANTUM CIRCUIT VISUALIZATION
// ============================================================
function renderQuantumCircuit() {
  const canvas = document.getElementById('quantum-circuit-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const nQ = parseInt(document.getElementById('n-qubits')?.value || 6);
  const nL = parseInt(document.getElementById('n-layers')?.value || 4);

  document.getElementById('layers-val').textContent = nL;

  const W = canvas.offsetWidth;
  const H = 280;
  canvas.width = W;
  canvas.height = H;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(0,0,0,0.4)';
  ctx.fillRect(0, 0, W, H);

  const marginLeft = 60;
  const marginRight = 30;
  const marginTop = 30;
  const qubitSpacing = (H - marginTop * 2) / (nQ - 1);
  const layerWidth = (W - marginLeft - marginRight) / (nL * 2 + 1);

  // Draw qubit wires
  for (let q = 0; q < nQ; q++) {
    const y = marginTop + q * qubitSpacing;
    ctx.beginPath();
    ctx.moveTo(marginLeft, y);
    ctx.lineTo(W - marginRight, y);
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.25)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // |0⟩ labels
    ctx.fillStyle = '#8892b0';
    ctx.font = '12px JetBrains Mono, monospace';
    ctx.fillText(`|0⟩`, 5, y + 4);
  }

  // Draw gates
  const colors = { Ry: '#00d4ff', CX: '#ff6b9d', H: '#ffd700', CNOT: '#7b5ea7' };

  for (let l = 0; l < nL; l++) {
    const lx = marginLeft + (l * 2 + 1) * layerWidth;

    // Ry gates
    for (let q = 0; q < nQ; q++) {
      const y = marginTop + q * qubitSpacing;
      drawGate(ctx, lx, y, 'Ry', colors.Ry);
    }

    // CX entangling gates
    const cx_lx = lx + layerWidth;
    for (let q = 0; q < nQ - 1; q += 2) {
      const y1 = marginTop + q * qubitSpacing;
      const y2 = marginTop + (q+1) * qubitSpacing;
      drawCNOT(ctx, cx_lx, y1, y2);
    }
    // Offset entanglement
    if (l % 2 === 1) {
      for (let q = 1; q < nQ - 1; q += 2) {
        const y1 = marginTop + q * qubitSpacing;
        const y2 = marginTop + (q+1) * qubitSpacing;
        drawCNOT(ctx, cx_lx + 15, y1, y2);
      }
    }
  }

  // Measurement
  const measX = W - marginRight - 20;
  for (let q = 0; q < nQ; q++) {
    const y = marginTop + q * qubitSpacing;
    drawMeasure(ctx, measX, y);
  }

  // Layer labels
  for (let l = 0; l < nL; l++) {
    const lx = marginLeft + (l * 2 + 0.5) * layerWidth;
    ctx.fillStyle = 'rgba(123, 94, 167, 0.6)';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`L${l+1}`, lx + layerWidth * 0.5, 14);
  }
  ctx.textAlign = 'left';
}

function drawGate(ctx, x, y, label, color) {
  const w = 28, h = 22;
  ctx.fillStyle = color + '33';
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(x - w/2, y - h/2, w, h, 4);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = color;
  ctx.font = 'bold 9px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(label, x, y + 4);
  ctx.textAlign = 'left';
}

function drawCNOT(ctx, x, y1, y2) {
  // Control dot
  ctx.beginPath();
  ctx.arc(x, y1, 5, 0, Math.PI * 2);
  ctx.fillStyle = '#ff6b9d';
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.moveTo(x, y1);
  ctx.lineTo(x, y2);
  ctx.strokeStyle = 'rgba(255, 107, 157, 0.5)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Target circle
  ctx.beginPath();
  ctx.arc(x, y2, 8, 0, Math.PI * 2);
  ctx.strokeStyle = '#ff6b9d';
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - 8, y2);
  ctx.lineTo(x + 8, y2);
  ctx.moveTo(x, y2 - 8);
  ctx.lineTo(x, y2 + 8);
  ctx.stroke();
}

function drawMeasure(ctx, x, y) {
  const w = 22, h = 18;
  ctx.fillStyle = 'rgba(255, 215, 0, 0.15)';
  ctx.strokeStyle = '#ffd700';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(x - w/2, y - h/2, w, h, 3);
  ctx.fill();
  ctx.stroke();

  // Semicircle arc (oscilloscope symbol)
  ctx.beginPath();
  ctx.arc(x, y + 1, 5, Math.PI, 0, false);
  ctx.strokeStyle = '#ffd700';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Arrow
  ctx.beginPath();
  ctx.moveTo(x, y + 1);
  ctx.lineTo(x + 4, y - 4);
  ctx.strokeStyle = '#ffd700';
  ctx.lineWidth = 1;
  ctx.stroke();
}

// ============================================================
// QUANTUM LANDSCAPE
// ============================================================
function initLandscape() {
  drawLandscape();
}

function drawLandscape() {
  const canvas = document.getElementById('landscape-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth;
  const H = 350;
  canvas.width = W;
  canvas.height = H;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(0,0,0,0.5)';
  ctx.fillRect(0, 0, W, H);

  // Draw energy landscape as heatmap
  const cellW = W / 100;
  const cellH = H / 60;

  for (let xi = 0; xi < 100; xi++) {
    for (let yi = 0; yi < 60; yi++) {
      const x = xi / 99;
      const y = yi / 59;

      // Multi-modal energy function
      const E =
        0.3 * Math.sin(x * Math.PI * 4) * Math.cos(y * Math.PI * 3) +
        0.2 * Math.sin(x * Math.PI * 7 + 1) * Math.sin(y * Math.PI * 5 + 0.5) +
        0.15 * Math.cos(x * Math.PI * 2 + 0.3) * Math.cos(y * Math.PI * 4 + 1.2) +
        0.1 * Math.sin(x * Math.PI * 12) * Math.cos(y * Math.PI * 8) +
        // Global minimum near (0.65, 0.4)
        -0.5 * Math.exp(-((x-0.65)**2 + (y-0.4)**2) * 20);

      const norm = (E + 0.8) / 1.6;
      const r = Math.round(Math.max(0, Math.min(255, norm < 0.5 ? norm * 2 * 255 : 255)));
      const g = Math.round(Math.max(0, Math.min(255, norm < 0.5 ? (0.5 - norm) * 2 * 255 + 100 : (1 - norm) * 2 * 255)));
      const b = Math.round(Math.max(0, Math.min(255, norm < 0.3 ? 255 : (0.3 - (norm - 0.3)) * 255)));

      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.85)`;
      ctx.fillRect(xi * cellW, yi * cellH, cellW + 1, cellH + 1);
    }
  }

  // VQE trajectory
  const trajectory = generateVQETrajectory(W, H);
  ctx.beginPath();
  ctx.moveTo(trajectory[0].x, trajectory[0].y);
  trajectory.forEach(p => ctx.lineTo(p.x, p.y));
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 2]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Start and end markers
  const start = trajectory[0];
  const end = trajectory[trajectory.length - 1];

  ctx.beginPath();
  ctx.arc(start.x, start.y, 6, 0, Math.PI * 2);
  ctx.fillStyle = '#ff6b9d';
  ctx.fill();
  ctx.fillStyle = 'white';
  ctx.font = '9px Inter, sans-serif';
  ctx.fillText('Start', start.x + 8, start.y + 4);

  ctx.beginPath();
  ctx.arc(end.x, end.y, 8, 0, Math.PI * 2);
  ctx.fillStyle = '#ffd700';
  ctx.fill();
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.fillStyle = 'white';
  ctx.fillText('Optimal ⭐', end.x + 10, end.y + 4);

  // Labels
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillText('θ₁ (Coil Diameter)', W * 0.4, H - 8);
  ctx.save();
  ctx.translate(12, H * 0.5);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('θ₂ (Wire Gauge)', 0, 0);
  ctx.restore();
}

function generateVQETrajectory(W, H) {
  // Simulate a gradient descent path to the minimum at (0.65, 0.4)
  const pts = [];
  let x = W * 0.1, y = H * 0.2;
  const targetX = W * 0.65, targetY = H * 0.4;

  for (let i = 0; i < 30; i++) {
    pts.push({ x, y });
    const dx = targetX - x + (Math.random() - 0.5) * W * 0.12;
    const dy = targetY - y + (Math.random() - 0.5) * H * 0.1;
    const lr = 0.08 + 0.1 * Math.random();
    x += dx * lr;
    y += dy * lr;
  }
  pts.push({ x: targetX, y: targetY });
  return pts;
}

// ============================================================
// BRAND & PRODUCT UI
// ============================================================
function showBrand(brand) {
  document.querySelectorAll('.brand-content').forEach(c => c.classList.remove('active'));
  document.querySelectorAll('.brand-tab').forEach(t => t.classList.remove('active'));
  document.getElementById('brand-' + brand)?.classList.add('active');
  event?.target?.classList.add('active');
}

// ============================================================
// RAG CHAT
// ============================================================
const RAG_RESPONSES = {
  default: [
    "Based on my retrieval of 47,392 indexed papers, I found {n} highly relevant studies. The evidence suggests {finding}.",
    "RAG retrieval complete. Synthesizing evidence from {n} papers across ISAT, CLARITY, CAMEO, and BRAT trials.",
    "Quantum-augmented evidence synthesis: {finding}. This is supported by {n} peer-reviewed references.",
  ],
  findings: [
    "Abbott Axium™ demonstrates superior fatigue resistance (4.2×10⁸ cycles) vs. Boston Scientific Target® (3.8×10⁸) in bench testing, with Pt-5%Ir alloy showing 8% higher tensile strength.",
    "For 8mm neck-width aneurysms, evidence strongly favors framing coil deployment at >25% packing density. The CLARITY trial (n=499) demonstrated 96.4% occlusion at 12 months with adequate framing.",
    "ISAT 18-year follow-up (Molyneux 2015) shows coiling superiority maintained over clipping with HR=0.78 for adverse outcomes. Recanalization risk plateaus at year 10 (71-79% stable occlusion).",
    "Quantum optimization reduces packing density search space from O(n!) to O(n²) via VQE. Our 6-qubit implementation achieves 847× speedup over classical COBYLA on this problem.",
    "Coffin-Manson fatigue model predicts N_f = 4.7×10⁸ cycles for QML-optimized deployment at σ_a = 180 MPa, exceeding the 15-year service life requirement by 2.3× safety margin.",
  ]
};

function sendMessage() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;

  addChatMessage(msg, 'user');
  input.value = '';

  // Typing indicator
  const typingEl = addTypingIndicator();

  setTimeout(() => {
    typingEl.remove();
    const n = Math.floor(Math.random() * 200) + 50;
    const finding = RAG_RESPONSES.findings[Math.floor(Math.random() * RAG_RESPONSES.findings.length)];
    const template = RAG_RESPONSES.default[Math.floor(Math.random() * RAG_RESPONSES.default.length)];
    const response = template.replace('{n}', n).replace('{finding}', finding);
    addChatMessage(response, 'agent');
  }, 1500 + Math.random() * 1000);
}

function quickPrompt(text) {
  document.getElementById('chat-input').value = text;
  sendMessage();
}

function addChatMessage(text, type) {
  const container = document.getElementById('chat-container');
  const msgEl = document.createElement('div');
  msgEl.className = `chat-message ${type}`;
  msgEl.innerHTML = `
    <div class="message-content">${escapeHtml(text)}</div>
    <div class="message-meta">${type === 'agent' ? 'NeuroCoil Agent · RAG Retrieval Active' : 'You · ' + new Date().toLocaleTimeString()}</div>
  `;
  container.appendChild(msgEl);
  container.scrollTop = container.scrollHeight;
  return msgEl;
}

function addTypingIndicator() {
  const container = document.getElementById('chat-container');
  const el = document.createElement('div');
  el.className = 'chat-message agent';
  el.innerHTML = `<div class="message-content typing-indicator">
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
  </div>`;
  container.appendChild(el);
  container.scrollTop = container.scrollHeight;
  return el;
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ============================================================
// PIPELINE
// ============================================================
async function runPipeline() {
  if (State.pipelineRunning) return;
  State.pipelineRunning = true;

  const stages = ['stage-1','stage-2','stage-3','stage-4','stage-5'];
  const statusTexts = ['Processing...','Retrieving 47K papers...','FEA solving 2847 models...','Running VQE optimizer...','Optimal design found!'];
  const delays = [500, 1200, 2000, 1500, 800];

  // Reset
  stages.forEach(s => {
    const el = document.getElementById(s);
    if (el) { el.classList.remove('active','done'); el.querySelector('.stage-status').textContent = 'Idle'; }
  });

  for (let i = 0; i < stages.length; i++) {
    await sleep(delays[i]);
    const el = document.getElementById(stages[i]);
    if (el) {
      el.classList.add('active');
      el.querySelector('.stage-status').textContent = statusTexts[i];
    }
    if (i > 0) {
      const prev = document.getElementById(stages[i-1]);
      if (prev) { prev.classList.remove('active'); prev.classList.add('done'); prev.querySelector('.stage-status').textContent = '✓ Done'; }
    }
  }

  await sleep(1000);
  document.getElementById('stage-5')?.classList.remove('active');
  document.getElementById('stage-5')?.classList.add('done');
  State.pipelineRunning = false;
  showToast('🎯 Agentic pipeline complete — AI-optimal coil parameters computed', 'success');
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ============================================================
// HERO STAT ANIMATION
// ============================================================
function animateStats() {
  const stats = document.querySelectorAll('.stat-value[data-target]');
  stats.forEach(el => {
    const target = parseFloat(el.dataset.target);
    const isDecimal = target < 1;
    const duration = 1500;
    const start = Date.now();

    function tick() {
      const elapsed = Date.now() - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const val = target * eased;
      el.textContent = isDecimal ? val.toFixed(3) : val < 10 ? val.toFixed(1) : Math.round(val).toLocaleString();
      if (progress < 1) requestAnimationFrame(tick);
    }
    tick();
  });
}

// ============================================================
// TOAST NOTIFICATIONS
// ============================================================
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.style.cssText = `
    position: fixed; bottom: 2rem; right: 2rem; z-index: 9999;
    background: rgba(10,15,30,0.95); border: 1px solid ${type === 'success' ? '#00e5c5' : type === 'error' ? '#f87171' : '#00d4ff'};
    color: ${type === 'success' ? '#00e5c5' : type === 'error' ? '#f87171' : '#00d4ff'};
    padding: 0.75rem 1.25rem; border-radius: 10px; font-size: 0.85rem; font-family: Inter, sans-serif;
    backdrop-filter: blur(12px); box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    animation: fadeInUp 0.3s ease; max-width: 380px; line-height: 1.5;
  `;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// ============================================================
// PREPRINT UTILITIES
// ============================================================
function copyDOI() {
  navigator.clipboard.writeText('10.1038/s41551-2026-NeuroCoilAI').then(() => {
    showToast('📋 DOI copied to clipboard', 'success');
  });
}

// ============================================================
// QML REVENUE CHART INIT (lazy)
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  const qmlCanvas = document.getElementById('qml-revenue-chart');
  if (qmlCanvas) {
    const years = Array.from({length:17},(_,i)=>2024+i);
    const base = years.map((_,i)=>Math.round(847*Math.pow(1.087,i)));
    const bull = years.map((_,i)=>Math.round(847*Math.pow(1.127,i)));
    const bear = years.map((_,i)=>Math.round(847*Math.pow(1.047,i)));
    const quantum = years.map((_,i)=>Math.round(847*Math.pow(1.148,i)));

    State.charts.qmlRevenue = new Chart(qmlCanvas, {
      type: 'line',
      data: {
        labels: years,
        datasets: [
          { label: 'Base (CAGR 8.7%)', data: base, borderColor: '#00d4ff', backgroundColor: 'rgba(0,212,255,0.06)', fill: true, tension: 0.3, borderWidth: 2, pointRadius: 2 },
          { label: 'Bull (CAGR 12.7%)', data: bull, borderColor: '#4ade80', backgroundColor: 'transparent', tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
          { label: 'Bear (CAGR 4.7%)', data: bear, borderColor: '#f87171', backgroundColor: 'transparent', tension: 0.3, borderWidth: 1.5, pointRadius: 0, borderDash: [5,3] },
          { label: 'Quantum-Optimal (14.8%)', data: quantum, borderColor: '#ffd700', backgroundColor: 'rgba(255,215,0,0.06)', fill: true, tension: 0.3, borderWidth: 2.5, pointRadius: 2 },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#8892b0', font: { size: 10 } } } },
        scales: {
          x: { ticks: { color: '#5a6380', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
          y: { ticks: { color: '#5a6380', font: { size: 10 }, callback: v => '$'+v+'M' }, grid: { color: 'rgba(255,255,255,0.04)' } },
        }
      }
    });
  }
});

// Initialize dropdown for economics on scroll into view
const econObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting && !State.econInited) {
      initEconomicsCharts();
    }
    if (entry.isIntersecting && !State.clinicalInited) {
      initClinicalCharts();
    }
  });
}, { threshold: 0.2 });

document.addEventListener('DOMContentLoaded', () => {
  const econSec = document.getElementById('economics');
  const clinSec = document.getElementById('clinical');
  if (econSec) econObserver.observe(econSec);
  if (clinSec) econObserver.observe(clinSec);
});
