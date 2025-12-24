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
    // Handle specific high-DPI scaling if needed, keeping it simple for now
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
        if(node.x < 0) { node.x = 0; node.vx *= -1; }
        if(node.x > width) { node.x = width; node.vx *= -1; }
        if(node.y < 0) { node.y = 0; node.vy *= -1; }
        if(node.y > height) { node.y = height; node.vy *= -1; }
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
        const dist = Math.sqrt(dx*dx + dy*dy);
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
        const x = (hoveredNode.phi / (2*Math.PI)) * 100;
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

// Start
fetchCircuit();
draw();
