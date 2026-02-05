const canvas = document.getElementById('circuit-canvas');
const ctx = canvas.getContext('2d');

// State
let width, height;
let nodes = [];
let links = [];
let animationId;
let isRunning = false;
let simulationInterval;
let hoveredNode = null;

// Config
const CONFIG = {
    nodeBaseRadius: 8,
    nodeMaxRadiusAdd: 10,
    linkWidthScale: 3,
    friction: 0.9,
    repulsion: -200,
    springLength: 100,
    springK: 0.05
};

// Colors based on CSS
const COLORS = {
    primary: '#00f3ff',
    secondary: '#bc13fe',
    accent: '#ffee00',
    white: '#ffffff'
};

// --- Initialization ---

function resize() {
    width = canvas.parentElement.clientWidth;
    height = canvas.parentElement.clientHeight;
    canvas.width = width;
    canvas.height = height;
}

window.addEventListener('resize', resize);
resize();

// --- API Calls ---

async function fetchCircuit() {
    try {
        const res = await fetch('/api/circuit');
        const data = await res.json();
        updateGraphData(data);
    } catch (e) {
        console.error("Failed to fetch circuit", e);
    }
}

async function evolveCircuit() {
    const res = await fetch('/api/evolve', { method: 'POST' });
    const data = await res.json();
    updateGraphData(data);
}

async function trainCircuit() {
    const res = await fetch('/api/train', { method: 'POST' });
    const data = await res.json();
    updateGraphData(data);
}

async function resetCircuit() {
    const res = await fetch('/api/reset', { method: 'POST' });
    const data = await res.json();
    // Reset positions randomly
    nodes = [];
    links = [];
    updateGraphData(data);

    // Reset Dementia UI
    document.getElementById('patient-stage').innerText = "--";
    document.getElementById('val-plasticity').innerText = "0.0";
    document.getElementById('val-density').innerText = "0.0";
    document.getElementById('val-mem-coherence').innerText = "0.0";
    document.getElementById('treatment-status').innerText = "Reset";
    document.getElementById('ai-recommendation').innerText = "--";
    document.getElementById('treatment-log').innerHTML = '<div class="log-entry placeholder">Treatment logs will appear here...</div>';

    // Re-fetch initial state
    fetchDementiaMetrics();
}

// --- Dementia Treatment Logic ---

async function fetchDementiaMetrics() {
    try {
        const res = await fetch('/api/dementia/metrics');
        const data = await res.json();
        updateDementiaUI(data);
    } catch (e) {
        console.error("Failed to fetch dementia metrics", e);
    }
}

function updateDementiaUI(data) {
    document.getElementById('patient-stage').innerText = data.stage;
    document.getElementById('val-plasticity').innerText = data.plasticity_index.toFixed(2);
    document.getElementById('val-density').innerText = data.synaptic_density.toFixed(2);
    document.getElementById('val-mem-coherence').innerText = data.memory_coherence.toFixed(2);
    document.getElementById('ai-recommendation').innerText = data.recommended_exercise;

    // Update main header state if needed
    // document.getElementById('system-status').innerText = data.stage; 
}

// --- Ethics & Consent Logic ---

async function verifyConsent() {
    const studyId = document.getElementById('study-id-input').value.trim() || "STANDARD";
    const neuroChecked = document.getElementById('neuro-check').checked;

    if (!neuroChecked) {
        alert("Neurological Clearance must be confirmed before proceeding.");
        return;
    }

    try {
        const res = await fetch('/api/ethics/consent', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ study_id: studyId })
        });
        const data = await res.json();

        if (data.status === "Consent Verified") {
            unlockTreatmentPanel(data.mode);
        }
    } catch (e) {
        console.error("Consent failed", e);
    }
}

function unlockTreatmentPanel(mode) {
    const ethicsStatus = document.getElementById('ethics-status');
    ethicsStatus.innerText = "Authorized";
    ethicsStatus.style.color = "#00ff88";
    ethicsStatus.style.borderColor = "#00ff88";

    // Hide inputs after locking in
    document.getElementById('study-id-input').disabled = true;
    document.getElementById('neuro-check').disabled = true;
    document.getElementById('btn-consent').style.display = 'none';

    const panel = document.getElementById('treatment-panel');
    panel.style.opacity = "1";
    panel.style.pointerEvents = "auto";

    document.getElementById('treatment-status').innerText = `Active (${mode})`;

    // Visual cue for custom mode
    if (mode && mode.includes("Custom")) {
        document.getElementById('treatment-status').style.color = "#ffee00";
        document.getElementById('treatment-status').style.borderColor = "#ffee00";
    }
}

async function applyTreatment() {
    const type = document.getElementById('treatment-type').value;
    const intensity = parseFloat(document.getElementById('intensity-slider').value);

    document.getElementById('treatment-status').innerText = "Simulating Treatment...";
    document.getElementById('btn-treat').disabled = true;

    try {
        const res = await fetch('/api/dementia/treat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ treatment_type: type, intensity: intensity })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Treatment Failed");
        }

        const data = await res.json();

        // Update Logs
        const logContainer = document.getElementById('treatment-log');
        if (logContainer.querySelector('.placeholder')) {
            logContainer.innerHTML = '';
        }

        // Add separator
        const sep = document.createElement('div');
        sep.className = 'log-entry';
        sep.innerText = `--- Treatment: ${type.toUpperCase()} (Int: ${intensity}) ---`;
        sep.style.color = '#ffee00';
        logContainer.appendChild(sep);

        data.treatment_logs.forEach(log => {
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerText = `> ${log}`;
            // Highlight remediations
            if (log.includes("[REMEDIATION]")) {
                entry.style.color = "#ff9900";
                entry.style.fontWeight = "bold";
            }
            logContainer.appendChild(entry);
        });

        // Update UI with new metrics
        updateDementiaUI(data.new_metrics);

        // Update Visual Graph
        updateGraphData(data.brain_state);

        document.getElementById('treatment-status').innerText = "Treatment Complete";

        logContainer.scrollTop = logContainer.scrollHeight;

    } catch (e) {
        console.error("Treatment failed", e);
        document.getElementById('treatment-status').innerText = e.message;

        // Log error in the log box too
        const logContainer = document.getElementById('treatment-log');
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerText = `[ERROR] ${e.message}`;
        entry.style.color = '#ff4444';
        logContainer.appendChild(entry);

    } finally {
        document.getElementById('btn-treat').disabled = false;
    }
}


// --- Graph Logic ---

function updateGraphData(data) {
    // Merge new data with existing nodes to preserve positions
    const newNodesMap = new Map();

    data.nodes.forEach(n => {
        const existing = nodes.find(en => en.id === n.id);
        if (existing) {
            existing.theta = n.theta;
            existing.phi = n.phi;
            existing.excitation = n.excitation;
            existing.phase = n.phase;
            newNodesMap.set(n.id, existing);
        } else {
            const newNode = {
                ...n,
                x: Math.random() * width,
                y: Math.random() * height,
                vx: 0,
                vy: 0
            };
            newNodesMap.set(n.id, newNode);
        }
    });

    nodes = Array.from(newNodesMap.values());

    // Links
    links = data.links.map(l => ({
        source: newNodesMap.get(l.source),
        target: newNodesMap.get(l.target),
        strength: l.strength,
        coherence: l.coherence
    })).filter(l => l.source && l.target);

    // Update UI Metrics
    const avgCoherence = links.reduce((acc, l) => acc + l.coherence, 0) / (links.length || 1);
    document.getElementById('coherence-metric').textContent = avgCoherence.toFixed(4);
}

function updatePhysics() {
    // Simple Force Directed Layout

    // Repulsion
    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            const dx = nodes[j].x - nodes[i].x;
            const dy = nodes[j].y - nodes[i].y;
            const distSq = dx * dx + dy * dy || 1;
            const dist = Math.sqrt(distSq);

            const force = CONFIG.repulsion / distSq;
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;

            nodes[i].vx -= fx;
            nodes[i].vy -= fy;
            nodes[j].vx += fx;
            nodes[j].vy += fy;
        }
    }

    // Springs (Links)
    links.forEach(link => {
        const dx = link.target.x - link.source.x;
        const dy = link.target.y - link.source.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;

        // target length inverse to strength (stronger = closer)
        const targetLen = CONFIG.springLength * (1.5 - link.strength);
        const displacement = dist - targetLen;

        const force = displacement * CONFIG.springK;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;

        link.source.vx += fx;
        link.source.vy += fy;
        link.target.vx -= fx;
        link.target.vy -= fy;
    });

    // Center Gravity
    const cx = width / 2;
    const cy = height / 2;
    nodes.forEach(node => {
        node.vx += (cx - node.x) * 0.001;
        node.vy += (cy - node.y) * 0.001;

        node.vx *= CONFIG.friction;
        node.vy *= CONFIG.friction;

        node.x += node.vx;
        node.y += node.vy;

        // Bounds
        if (node.x < 0) { node.x = 0; node.vx *= -1; }
        if (node.x > width) { node.x = width; node.vx *= -1; }
        if (node.y < 0) { node.y = 0; node.vy *= -1; }
        if (node.y > height) { node.y = height; node.vy *= -1; }
    });
}

// --- Rendering ---

function getPhaseColor(phase) {
    // Map phase (-PI to PI) to hue (0-360)
    // Shift so it aligns with our aesthetic
    const normalized = (phase + Math.PI) / (2 * Math.PI);
    const hue = normalized * 360;
    return `hsl(${hue}, 80%, 60%)`;
}

function draw() {
    ctx.clearRect(0, 0, width, height);

    updatePhysics();

    // Draw Links
    links.forEach(link => {
        ctx.beginPath();
        ctx.moveTo(link.source.x, link.source.y);
        ctx.lineTo(link.target.x, link.target.y);

        // Style
        const alpha = 0.1 + (link.strength * 0.5);
        ctx.strokeStyle = `rgba(0, 243, 255, ${alpha})`;
        ctx.lineWidth = 1 + link.strength * CONFIG.linkWidthScale;
        ctx.stroke();

        // Quantum "Flow" particles
        if (link.coherence > 0.1) {
            const time = Date.now() / 1000;
            // Create a moving pulse
            // Speed depends on strength
            const speed = link.strength;
            const offset = (time * speed) % 1;

            const px = link.source.x + (link.target.x - link.source.x) * offset;
            const py = link.source.y + (link.target.y - link.source.y) * offset;

            ctx.beginPath();
            ctx.arc(px, py, 2, 0, Math.PI * 2);
            ctx.fillStyle = COLORS.white;
            ctx.shadowBlur = 5;
            ctx.shadowColor = COLORS.white;
            ctx.fill();
            ctx.shadowBlur = 0;
        }
    });

    // Draw Nodes
    nodes.forEach(node => {
        const radius = CONFIG.nodeBaseRadius + (node.excitation * CONFIG.nodeMaxRadiusAdd);
        const color = getPhaseColor(node.phase);

        // Glow
        ctx.shadowBlur = 10 + (node.excitation * 20);
        ctx.shadowColor = color;

        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // Inner core
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius * 0.4, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.fill();

        ctx.shadowBlur = 0;

        // Text ID
        if (node.excitation > 0.5) {
            ctx.fillStyle = 'rgba(255,255,255,0.7)';
            ctx.font = '10px Rajdhani';
            ctx.fillText(node.id, node.x + 12, node.y - 12);
        }
    });

    // Highlight hovered
    if (hoveredNode) {
        ctx.beginPath();
        ctx.arc(hoveredNode.x, hoveredNode.y, 25, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    animationId = requestAnimationFrame(draw);
}

// --- Interaction ---

canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    hoveredNode = null;
    let closestDist = Infinity;

    nodes.forEach(node => {
        const dx = node.x - mx;
        const dy = node.y - my;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 30 && dist < closestDist) {
            closestDist = dist;
            hoveredNode = node;
        }
    });

    updateInfoPanel();
});

function updateInfoPanel() {
    const panel = document.getElementById('node-info-panel');
    if (hoveredNode) {
        panel.classList.add('visible');
        document.getElementById('info-id').textContent = hoveredNode.id;
        document.getElementById('info-exc').textContent = hoveredNode.excitation.toFixed(4);
        document.getElementById('info-phase').textContent = hoveredNode.phase.toFixed(4);

        // Visualization on Bloch sphere approximation
        // Just moving the dot based on theta/phi is enough for effect
        const dot = document.getElementById('bloch-point');
        // Map theta (0-PI) to Y (0-100%)
        // Map phi (0-2PI) to X (0-100%)
        const y = (hoveredNode.theta / Math.PI) * 100;
        const x = (hoveredNode.phi / (2 * Math.PI)) * 100;
        dot.style.top = `${y}%`;
        dot.style.left = `${x}%`;

    } else {
        panel.classList.remove('visible');
    }
}

// --- Controls ---

document.getElementById('btn-reset').addEventListener('click', resetCircuit);
document.getElementById('btn-step').addEventListener('click', evolveCircuit);
document.getElementById('btn-train').addEventListener('click', trainCircuit);

document.getElementById('intensity-slider').addEventListener('input', (e) => {
    document.getElementById('intensity-val').innerText = e.target.value;
});
document.getElementById('btn-treat').addEventListener('click', applyTreatment);
document.getElementById('btn-consent').addEventListener('click', verifyConsent);

// Prime Resonance controls
document.getElementById('prime-intensity-slider').addEventListener('input', (e) => {
    document.getElementById('prime-intensity-val').innerText = e.target.value;
});

document.getElementById('btn-prime-repair').addEventListener('click', async () => {
    const prime = parseInt(document.getElementById('prime-modulus').value);
    const intensity = parseFloat(document.getElementById('prime-intensity-slider').value);

    const btn = document.getElementById('btn-prime-repair');
    btn.disabled = true;
    btn.innerText = 'Applying Prime Resonance...';

    try {
        // Apply prime_resonance treatment
        const res = await fetch('/api/dementia/treat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                treatment_type: 'prime_resonance',
                intensity: intensity,
                prime_modulus: prime
            })
        });

        if (!res.ok) {
            throw new Error('Prime repair failed');
        }

        const data = await res.json();

        // Update logs
        const logContainer = document.getElementById('prime-log');
        logContainer.innerHTML = '';

        const header = document.createElement('div');
        header.className = 'log-entry';
        header.innerText = `--- Prime ${prime} Resonance Repair (Intensity: ${intensity}) ---`;
        header.style.color = '#ffee00';
        logContainer.appendChild(header);

        data.treatment_logs.forEach(log => {
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerText = `> ${log}`;
            logContainer.appendChild(entry);
        });

        // Get detailed stats
        const statsRes = await fetch('/api/dementia/detailed_stats');
        const stats = await statsRes.json();

        // Update metrics
        document.getElementById('val-surface-flux').innerText = stats.surface_integral_flux.toFixed(4);
        document.getElementById('val-kam-stability').innerText = stats.kam_stability_index.toFixed(4);
        document.getElementById('val-ramanujan').innerText = stats.ramanujan_congruence_ratio.toFixed(4);
        document.getElementById('prime-plasticity').innerText = stats.plasticity_index.toFixed(2);
        document.getElementById('prime-density').innerText = stats.synaptic_density.toFixed(2);
        document.getElementById('prime-coherence').innerText = stats.global_coherence.toFixed(4);

        // Update visual graph
        updateGraphData(data.brain_state);

        btn.innerText = 'Apply Prime Resonance Repair';

    } catch (e) {
        console.error('Prime repair failed', e);
        const logContainer = document.getElementById('prime-log');
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerText = `[ERROR] ${e.message}`;
        entry.style.color = '#ff4444';
        logContainer.appendChild(entry);
        btn.innerText = 'Apply Prime Resonance Repair';
    } finally {
        btn.disabled = false;
    }
});


const btnRun = document.getElementById('btn-run');
btnRun.addEventListener('click', () => {
    if (isRunning) {
        clearInterval(simulationInterval);
        isRunning = false;
        btnRun.classList.remove('active');
        btnRun.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Continuously Evolve`;
        document.getElementById('system-status').innerText = "Paused";
        document.getElementById('system-status').style.color = CONFIG.white;
    } else {
        simulationInterval = setInterval(evolveCircuit, 200); // 5fps update
        isRunning = true;
        btnRun.classList.add('active');
        btnRun.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Stop Evolution`;
        document.getElementById('system-status').innerText = "Evolving";
        document.getElementById('system-status').style.color = COLORS.primary;
    }
});

// --- HPC / ANE Stats Logic ---
async function pollANEStats() {
    try {
        const res = await fetch('/api/ane/stats');
        const data = await res.json();

        // Update Text
        document.getElementById('ane-status').innerText = data.status;
        document.getElementById('ane-cores').innerText = data.active_cores;
        document.getElementById('ane-tops').innerText = data.tops_utilization.toFixed(1);
        document.getElementById('ane-mem').innerText = data.memory_bandwidth_gbps.toFixed(1);
        document.getElementById('ane-temp').innerText = data.temperature_c.toFixed(1);

        // Update Core Grid Map
        const viz = document.getElementById('ane-core-viz');
        viz.innerHTML = ''; // efficient enough for 32 items

        data.core_map.forEach(util => {
            const cell = document.createElement('div');
            cell.className = 'core-cell';
            const alpha = util;
            cell.style.backgroundColor = `rgba(0, 255, 136, ${0.1 + alpha * 0.9})`;
            if (util > 0.5) cell.style.boxShadow = `0 0 5px rgba(0, 255, 136, ${alpha})`;
            viz.appendChild(cell);
        });

    } catch (e) {
        console.error("ANE Poll failed", e);
    }
}

// Start ANE Polling (High Frequency for Real-time feel)
setInterval(pollANEStats, 500);

// --- Tab Switching ---
const tabs = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');

tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const targetTab = tab.dataset.tab;

        // Update active states
        tabs.forEach(t => t.classList.remove('active'));
        tabContents.forEach(tc => tc.classList.remove('active'));

        tab.classList.add('active');
        document.getElementById(`${targetTab}-tab`).classList.add('active');
    });
});

// --- Combinatorial Manifold Functions ---

async function initializeManifold(pathologyType) {
    const btnId = `btn-${pathologyType}-init`;
    const btn = document.getElementById(btnId);
    btn.disabled = true;
    btn.innerText = 'Initializing...';

    try {
        const res = await fetch('/api/manifold/initialize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pathology_type: pathologyType,
                num_neurons: 100
            })
        });

        const data = await res.json();
        updateManifoldUI(pathologyType, data.baseline_topology);

        // Trigger Visualization
        document.getElementById(`${pathologyType}-viz-placeholder`).innerText = "Generating Projection...";
        const viz = await fetchManifoldVisuals(pathologyType);

        btn.innerText = 'Re-initialize';
    } catch (e) {
        console.error(`Failed to initialize ${pathologyType}`, e);
        btn.innerText = 'Error - Retry';
    } finally {
        btn.disabled = false;
    }
}

async function fetchManifoldVisuals(pathologyType) {
    try {
        const res = await fetch('/api/manifold/visualize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pathology_type: pathologyType, num_neurons: 100 })
        });
        const data = await res.json();

        const baselineImg = document.getElementById(`${pathologyType}-viz-baseline`);
        const repairedImg = document.getElementById(`${pathologyType}-viz-repaired`);

        if (data.baseline_image) {
            baselineImg.src = data.baseline_image + "?t=" + new Date().getTime();
            baselineImg.style.display = 'block';
            document.getElementById(`${pathologyType}-viz-placeholder`).style.display = 'none';
        }

        // Store repaired image URL for later use if available
        if (data.repaired_image) {
            repairedImg.dataset.src = data.repaired_image + "?t=" + new Date().getTime();
        } else {
            repairedImg.dataset.src = ""; // Clear if not available
        }

        return data;
    } catch (e) {
        console.error("Visuals failed", e);
        document.getElementById(`${pathologyType}-viz-placeholder`).innerText = "Viz Gen Failed";
    }
}

async function applyManifoldRepair(pathologyType) {
    const btnId = `btn-${pathologyType}-repair`;
    const btn = document.getElementById(btnId);
    btn.disabled = true;
    btn.innerText = 'Applying Repair...';

    // Clear previous log
    const logId = `${pathologyType}-repair-log`;
    const logEl = document.getElementById(logId);
    if (logEl) {
        logEl.style.display = 'block';
        logEl.innerHTML = `<div style="color: var(--accent); border-bottom: 1px solid rgba(255,255,255,0.1);">Neural Event Log:</div><div style="color: var(--text-dim);">Initializing repair cycles...</div>`;
    }

    try {
        const res = await fetch('/api/manifold/repair', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pathology_type: pathologyType,
                num_cycles: 5
            })
        });

        const data = await res.json();

        if (data.final_statistics) {
            updateManifoldRepairUI(pathologyType, data.final_statistics);
        }

        if (data.repair_history) {
            updateManifoldLog(pathologyType, data.repair_history);
        }

        if (data.projection_image) {
            updateManifoldProjection(pathologyType, data.projection_image);

            // Show Repaired Image in the comparison view
            const repairedImg = document.getElementById(`${pathologyType}-viz-repaired`);
            repairedImg.src = data.projection_image + "?t=" + new Date().getTime();
            repairedImg.style.display = 'block';
            document.getElementById(`${pathologyType}-viz-repaired-placeholder`).style.display = 'none';
        }

        // Update comparison
        updateComparison();

        btn.innerText = 'Apply Repair (5 cycles)';
    } catch (e) {
        console.error(`Failed to repair ${pathologyType}`, e);
        if (logEl) logEl.innerHTML += `<div style="color: red;">Error: ${e.message}</div>`;
        btn.innerText = 'Error - Retry';
    } finally {
        btn.disabled = false;
    }
}

function updateManifoldLog(pathologyType, history) {
    const logId = `${pathologyType}-repair-log`;
    const logEl = document.getElementById(logId);
    if (!logEl) return;

    logEl.style.display = 'block';
    logEl.innerHTML = `<div style="color: var(--accent); border-bottom: 1px solid rgba(255,255,255,0.1);">Neural Event Log:</div>`;

    history.forEach(h => {
        const line = document.createElement('div');
        line.style.marginTop = '4px';
        line.innerHTML = `> Cycle ${h.cycle + 1}: <span style="color: #00ff88">+${h.neurons_added} neurons</span> (Pathology: ${h.pathological_nodes_remaining})`;
        logEl.appendChild(line);
    });

    logEl.scrollTop = logEl.scrollHeight;
}

function updateManifoldUI(pathologyType, topology) {
    const prefix = pathologyType;

    // Update Betti numbers
    document.getElementById(`${prefix}-beta0`).innerText = topology.betti_numbers.beta_0;
    document.getElementById(`${prefix}-beta1`).innerText = topology.betti_numbers.beta_1;
    document.getElementById(`${prefix}-beta2`).innerText = topology.betti_numbers.beta_2;

    // Update basic stats
    document.getElementById(`${prefix}-nodes`).innerText = topology.num_nodes;
    document.getElementById(`${prefix}-edges`).innerText = topology.num_edges;
    document.getElementById(`${prefix}-pathological`).innerText =
        topology.pathological_regions.nodes.length;

    if (topology.nash_stability_index !== undefined) {
        const el = document.getElementById(`${prefix}-nash`);
        if (el) el.innerText = topology.nash_stability_index.toFixed(4);
    }
}

function updateManifoldRepairUI(pathologyType, stats) {
    const prefix = pathologyType;

    // Update repair statistics
    document.getElementById(`${prefix}-added`).innerText = stats.total_neurons_added;
    document.getElementById(`${prefix}-cycles`).innerText = stats.repair_cycles;
    document.getElementById(`${prefix}-reduction`).innerText =
        `${stats.pathology_reduction_percent.toFixed(1)}%`;

    // Update final topology
    if (stats.final_topology) {
        updateManifoldUI(pathologyType, stats.final_topology);
    }

    // Update Post Treatment Parameters
    if (stats.post_treatment_parameters) {
        const params = stats.post_treatment_parameters;
        document.getElementById(`${prefix}-advanced-metrics`).style.display = 'block';
        document.getElementById(`${prefix}-curvature`).innerText = params.curvature_homogeneity.toFixed(4);
        document.getElementById(`${prefix}-spectral`).innerText = params.spectral_gap.toFixed(4);
        document.getElementById(`${prefix}-prime-res`).innerText = params.prime_resonance_index.toFixed(4);
    }
}

function updateManifoldProjection(pathologyType, filename) {
    const container = document.getElementById(`${pathologyType}-projection-container`);
    const img = document.getElementById(`${pathologyType}-projection-img`);

    // Add timestamp to prevent caching
    img.src = `${filename}?t=${new Date().getTime()}`;
    container.style.display = 'block';
}

async function updateComparison() {
    try {
        const res = await fetch('/api/manifold/comparison');
        const data = await res.json();

        const comparisonDiv = document.getElementById('comparison-results');

        if (data.dementia && data.ptsd) {
            const dStats = data.dementia;
            const pStats = data.ptsd;

            comparisonDiv.innerHTML = `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <h4 style="color: var(--accent); margin-bottom: 10px;">🧠 Dementia Repair</h4>
                        <div style="font-size: 0.85rem;">
                            <div style="margin-bottom: 5px;">✓ Neurons Added: <strong>${dStats.total_neurons_added}</strong></div>
                            <div style="margin-bottom: 5px;">✓ Repair Cycles: <strong>${dStats.repair_cycles}</strong></div>
                            <div style="margin-bottom: 5px;">✓ Pathology Reduction: <strong style="color: var(--success);">${dStats.pathology_reduction_percent.toFixed(1)}%</strong></div>
                            <div style="margin-bottom: 5px;">✓ Nodes Resolved: <strong>${dStats.nodes_resolved}</strong> / ${dStats.initial_pathological_nodes}</div>
                            <div style="margin-bottom: 5px;">✓ Edge Improvement: <strong style="color: var(--success);">${dStats.edge_improvement_percent.toFixed(1)}%</strong></div>
                            <div style="margin-bottom: 5px;">✓ Connectivity Gain: <strong style="color: var(--success);">${dStats.connectivity_improvement_percent.toFixed(1)}%</strong></div>
                            <div style="margin-bottom: 5px;">✓ Repair Efficiency: <strong>${dStats.repair_efficiency.toFixed(2)}</strong></div>
                            <div style="margin-bottom: 5px;">✓ Network Health: <strong style="color: var(--accent);">${dStats.network_health_score.toFixed(1)}/100</strong></div>
                            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(0,255,255,0.2);">
                                <div style="font-size: 0.8rem; color: var(--text-dim);">Betti Changes:</div>
                                <div style="font-size: 0.8rem;">Δβ₀: ${dStats.betti_improvement.beta_0}, Δβ₁: ${dStats.betti_improvement.beta_1}, Δβ₂: ${dStats.betti_improvement.beta_2}</div>
                            </div>
                        </div>
                    </div>
                    <div>
                        <h4 style="color: var(--accent); margin-bottom: 10px;">⚡ PTSD Repair</h4>
                        <div style="font-size: 0.85rem;">
                            <div style="margin-bottom: 5px;">✓ Neurons Added: <strong>${pStats.total_neurons_added}</strong></div>
                            <div style="margin-bottom: 5px;">✓ Repair Cycles: <strong>${pStats.repair_cycles}</strong></div>
                            <div style="margin-bottom: 5px;">✓ Pathology Reduction: <strong style="color: var(--success);">${pStats.pathology_reduction_percent.toFixed(1)}%</strong></div>
                            <div style="margin-bottom: 5px;">✓ Nodes Resolved: <strong>${pStats.nodes_resolved}</strong> / ${pStats.initial_pathological_nodes}</div>
                            <div style="margin-bottom: 5px;">✓ Edge Improvement: <strong style="color: var(--success);">${pStats.edge_improvement_percent.toFixed(1)}%</strong></div>
                            <div style="margin-bottom: 5px;">✓ Connectivity Gain: <strong style="color: var(--success);">${pStats.connectivity_improvement_percent.toFixed(1)}%</strong></div>
                            <div style="margin-bottom: 5px;">✓ Repair Efficiency: <strong>${pStats.repair_efficiency.toFixed(2)}</strong></div>
                            <div style="margin-bottom: 5px;">✓ Network Health: <strong style="color: var(--accent);">${pStats.network_health_score.toFixed(1)}/100</strong></div>
                            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(0,255,255,0.2);">
                                <div style="font-size: 0.8rem; color: var(--text-dim);">Betti Changes:</div>
                                <div style="font-size: 0.8rem;">Δβ₀: ${pStats.betti_improvement.beta_0}, Δβ₁: ${pStats.betti_improvement.beta_1}, Δβ₂: ${pStats.betti_improvement.beta_2}</div>
                            </div>
                        </div>
                    </div>
                </div>
                <div style="margin-top: 20px; padding: 15px; background: rgba(0, 255, 255, 0.05); border-radius: 8px; border-left: 3px solid var(--accent);">
                    <strong style="color: var(--accent);">Key Findings:</strong>
                    <div style="color: var(--text-main); margin-top: 8px; font-size: 0.9rem;">
                        <div style="margin-bottom: 5px;">• <strong>Dementia</strong> shows ${dStats.pathology_reduction_percent > pStats.pathology_reduction_percent ? 'superior' : 'comparable'} pathology reduction (${dStats.pathology_reduction_percent.toFixed(1)}% vs ${pStats.pathology_reduction_percent.toFixed(1)}%)</div>
                        <div style="margin-bottom: 5px;">• <strong>Network Health</strong>: Dementia ${dStats.network_health_score.toFixed(0)}/100, PTSD ${pStats.network_health_score.toFixed(0)}/100</div>
                        <div style="margin-bottom: 5px;">• <strong>Repair Efficiency</strong>: ${dStats.repair_efficiency > pStats.repair_efficiency ? 'Dementia' : 'PTSD'} demonstrates higher efficiency (${Math.max(dStats.repair_efficiency, pStats.repair_efficiency).toFixed(2)} nodes/neuron)</div>
                        <div style="margin-bottom: 5px;">• Both pathologies show significant improvement through prime congruence-based neurogenesis</div>
                        <div>• Dementia repair focuses on connectivity restoration, while PTSD repair rebalances hyperconnected trauma networks</div>
                    </div>
                </div>
            `;
        } else if (data.dementia || data.ptsd) {
            comparisonDiv.innerHTML = '<span style="color: var(--text-dim);">Run repair on both pathologies for comparative analysis...</span>';
        }
    } catch (e) {
        console.error('Failed to update comparison', e);
    }
}

// Manifold event listeners
document.getElementById('btn-dementia-init').addEventListener('click', () => initializeManifold('dementia'));
document.getElementById('btn-dementia-repair').addEventListener('click', () => applyManifoldRepair('dementia'));
document.getElementById('btn-ptsd-init').addEventListener('click', () => initializeManifold('ptsd'));
document.getElementById('btn-ptsd-repair').addEventListener('click', () => applyManifoldRepair('ptsd'));

// Start
fetchCircuit();
fetchDementiaMetrics();
draw();
