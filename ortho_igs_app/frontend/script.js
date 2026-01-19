
function showSection(id) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.getElementById(id).classList.add('active');

    document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
    document.querySelector(`button[onclick="showSection('${id}')"]`).classList.add('active');
}

// === Optimization Functions ===

async function runImplantOptimization() {
    const viz = document.getElementById('fit-viz');
    const resBox = document.getElementById('fit-results');
    resBox.innerText = "Running Quantum Surface Integral Optimization...";
    viz.innerHTML = `<div style="padding:20px; color:#aaa;">Simulating Conformal Manifold Alignment...</div>`;

    try {
        const res = await fetch('/api/planning/optimize_fit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ size: 'Auto' })
        });
        const resp = await res.json();
        const data = resp.data;

        // Show result text
        resBox.innerHTML = `
            <strong>Resolution:</strong> ${data.recommended_size}<br/>
            <strong>Alignment:</strong> ${data.alignment_score}<br/>
            <span style="color:#00cc66">Fit Optimized</span>
        `;

        // Simple Graph for Convergence
        viz.innerHTML = '';
        const history = data.convergence_history;
        const maxE = Math.abs(Math.min(...history));

        let path = `M 0 150 `;
        const w = 300; // width of viz container approx
        const h = 150;

        history.forEach((e, i) => {
            const x = (i / history.length) * 100 + "%";
            // Normalise energy -100 to 0 approx
            const y = h - (Math.abs(e) / maxE) * h;
            // We can't use % easily in SVG path without viewbox, stick to HTML bars
            viz.innerHTML += `<div style="display:inline-block; width:5px; height:${(Math.abs(e) / maxE) * 100}%; background:#4da6ff; margin-right:1px; vertical-align:bottom;"></div>`;
        });

    } catch (e) {
        resBox.innerText = "Error: " + e.message;
    }
}

// === New Workflow Functions ===

// === Post-Op Analytics ===

async function loadPostOp() {
    const grid = document.getElementById('analytics-grid');
    grid.style.display = 'grid';
    grid.innerHTML = '<div style="color:#aaa;">Simulating 5000 simulations using Monte Carlo methods...</div>';

    try {
        const res = await fetch('/api/workflow/postop');
        const resp = await res.json();
        const data = resp.data;

        // Clear loading
        grid.innerHTML = `
            <div class="card">
                <h3>Flexion Statistics</h3>
                <div id="flexion-stats" class="result-box">
                    Avg: ${data.avg_flexion.toFixed(1)}°<br/>
                    StdDev: ${data.std_flexion.toFixed(1)}°
                </div>
                <div id="flexion-hist" style="height:150px; background:#111; margin-top:10px; display:flex; align-items:flex-end;"></div>
            </div>
            <div class="card">
                <h3>Extension Statistics</h3>
                <div id="extension-stats" class="result-box">
                    Avg: ${data.avg_extension.toFixed(1)}°<br/>
                    Target: 0°
                </div>
            </div>
            <div class="card">
                <h3>Cost Analysis</h3>
                <div id="cost-stats" class="result-box">
                    Avg Cost: $${data.avg_cost.toLocaleString()}<br/>
                    Procedures: ${data.total_procedures}
                </div>
                <div id="cost-hist" style="height:150px; background:#111; margin-top:10px; display:flex; align-items:flex-end;"></div>
            </div>
        `;

        // Render Histogram (Flexion)
        const fHist = document.getElementById('flexion-hist');
        const maxF = Math.max(...data.flexion_bins);
        data.flexion_bins.forEach(val => {
            const h = (val / maxF) * 100;
            fHist.innerHTML += `<div style="flex:1; background:#4da6ff; margin:0 1px; height:${h}%"></div>`;
        });

        // Render Histogram (Cost)
        const cHist = document.getElementById('cost-hist');
        const maxC = Math.max(...data.cost_bins);
        data.cost_bins.forEach(val => {
            const h = (val / maxC) * 100;
            cHist.innerHTML += `<div style="flex:1; background:#ff9933; margin:0 1px; height:${h}%"></div>`;
        });

    } catch (e) {
        grid.innerText = "Error loading analytics: " + e.message;
    }
}

// === Interactive 3D Rendering Logic ===
let rotationX = 0;
let rotationY = 0;
// Note: We'll hook this into the Acquisition loop

async function runAcquisition() {
    const status = document.getElementById('scan-status');
    const canvas = document.getElementById('scanCanvas');
    const ctx = canvas.getContext('2d');

    status.innerText = "Scanning High-Res Anatomy...";

    // Fetch Data
    try {
        const res = await fetch('/api/workflow/acquire');
        const resp = await res.json();
        const data = resp.data;

        status.innerHTML = `
            <strong>Modality:</strong> ${data.modality}<br/>
            <strong>Slices:</strong> ${data.slices.length} (256x256)<br/>
            <strong>Vertices:</strong> ${data.points.length}
        `;

        // Render Loop
        const render = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // --- Left: Axial Slice ---
            const sliceSize = 256;
            // Scale down to fit half screen
            const displaySize = 240;
            const scale = displaySize / sliceSize;
            const offX = 10;
            const offY = 20;

            ctx.fillStyle = "#000"; ctx.fillRect(offX, offY, displaySize, displaySize);
            ctx.strokeStyle = "#444"; ctx.strokeRect(offX, offY, displaySize, displaySize);

            const slice0 = data.slices[0];
            for (let r = 0; r < sliceSize; r += 2) { // Skip every other for speed in JS canvas
                for (let c = 0; c < sliceSize; c += 2) {
                    const val = slice0[r][c];
                    if (val > 10) {
                        ctx.fillStyle = `rgb(${val},${val},${val})`;
                        ctx.fillRect(offX + c * scale, offY + r * scale, scale * 2, scale * 2);
                    }
                }
            }
            ctx.fillStyle = "#fff"; ctx.fillText("Axial Slice", offX, offY - 5);

            // --- Right: 3D Volumetric (Interactive) ---
            const cx = 380;
            const cy = 150;

            // Auto-Rotate
            rotationY += 0.02;

            data.points.forEach(p => {
                // Rotate Y
                let x = p.x * Math.cos(rotationY) - p.z * Math.sin(rotationY);
                let z = p.x * Math.sin(rotationY) + p.z * Math.cos(rotationY);
                let y = p.y;

                // Project
                const fov = 300;
                const dist = 50 + z;
                const s = fov / (fov + dist);

                const px = cx + x * s * 2;
                const py = cy + y * s * 2;

                ctx.fillStyle = (p.part === 'femur') ? '#ff4d4d' : '#00cc66';
                ctx.fillRect(px, py, 1.5, 1.5);
            });
            ctx.fillStyle = "#fff"; ctx.fillText("3D Reconstruction", cx - 40, 20);

            requestAnimationFrame(render);
        };
        render();

    } catch (e) { status.innerText = "Error: " + e.message; }
}

async function fetchGeometry() {
    const output = document.getElementById('geometry-preview');
    // Using a simple 2D wireframe projection on a new canvas or text
    output.innerHTML = "Fetching geometry...";

    try {
        const type = Math.random() > 0.5 ? 'genai' : 'nvqlink';
        const res = await fetch(`/api/geometry/${type}`);
        const data = await res.json();

        // Simple Visualization
        const geo = data.data;
        const width = 200;
        const height = 150;

        let html = `<div style="margin-bottom:5px; font-weight:bold;">${geo.type}</div>`;
        html += `<canvas id="geoCanvas" width="${width}" height="${height}" style="background:#222; border-radius:4px;"></canvas>`;
        html += `<div style="font-size:0.8em; color:#aaa;">${geo.description}</div>`;
        output.innerHTML = html;

        const canvas = document.getElementById('geoCanvas');
        const ctx = canvas.getContext('2d');
        ctx.strokeStyle = "#00ffcc";
        ctx.lineWidth = 1;

        const centerX = width / 2;
        const centerY = height / 2;
        const scale = 40;

        // Draw Edges
        ctx.beginPath();
        geo.edges.forEach(edge => {
            const v1 = geo.vertices[edge[0]];
            const v2 = geo.vertices[edge[1]];

            // Isometric-ish projection
            // x_screen = x - z
            // y_screen = y + (x+z)/2

            const px1 = centerX + (v1[0] - v1[2] * 0.5) * scale;
            const py1 = centerY + (-v1[1] - v1[2] * 0.5) * scale;

            const px2 = centerX + (v2[0] - v2[2] * 0.5) * scale;
            const py2 = centerY + (-v2[1] - v2[2] * 0.5) * scale;

            ctx.moveTo(px1, py1);
            ctx.lineTo(px2, py2);
        });
        ctx.stroke();

    } catch (e) {
        output.innerText = "Error: " + e.message;
    }
}

async function runRegistration() {
    const status = document.getElementById('reg-status');
    status.innerText = "Aligning Manifolds (SVD)...";
    try {
        const res = await fetch('/api/workflow/register', { method: 'POST' });
        const resp = await res.json();
        const m = resp.data.transform_matrix;

        let matrixHtml = "<table style='font-size:0.7em; background:#111; margin-top:5px; border-collapse:collapse;'>";
        m.forEach(row => {
            matrixHtml += "<tr>";
            row.forEach(val => matrixHtml += `<td style='padding:2px 5px; border:1px solid #333;'>${val.toFixed(2)}</td>`);
            matrixHtml += "</tr>";
        });
        matrixHtml += "</table>";

        status.innerHTML = `
            <strong>RMS Error:</strong> ${resp.data.rms_error.toFixed(3)} mm<br/>
            <strong>Method:</strong> ${resp.data.method}<br/>
            ${matrixHtml}
        `;
    } catch (e) { status.innerText = "Error: " + e.message; }
}

async function startTracking() {
    const box = document.getElementById('track-data');
    box.innerText = "Tracking Active...";
    setInterval(async () => {
        try {
            const res = await fetch('/api/workflow/track');
            const resp = await res.json();
            const d = resp.data;
            box.innerText = `Femur: [${d.femur_marker.pos[0].toFixed(2)}, ${d.femur_marker.pos[1].toFixed(2)}]\nTibia: [${d.tibia_marker.pos[0].toFixed(2)}, ${d.tibia_marker.pos[1].toFixed(2)}]`;
        } catch (e) { }
    }, 500);
}

async function genAiRobotPath() {
    const canvas = document.getElementById('robotCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 300, 200);
    ctx.strokeStyle = "#4da6ff";
    ctx.beginPath();

    try {
        const res = await fetch('/api/robot/trajectory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cut_type: 'distal' })
        });
        const resp = await res.json();

        const path = resp.data.trajectory;
        path.forEach((p, i) => {
            // Visualize path trace
            const x = 20 + p.x * 6;
            const y = 20 + p.y * 8;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

    } catch (e) { console.error(e); }
}

// === Workflow / Planning Functions ===

async function simulateResection() {
    const canvas = document.getElementById('resectionCanvas');
    const ctx = canvas.getContext('2d');
    const info = document.getElementById('resection-info');

    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#222";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    info.innerText = "Calculating resection profiles...";

    try {
        const res = await fetch('/api/workflow/resection', { method: 'POST' });
        const resp = await res.json();
        const data = resp.data;

        // Draw Pre (Blue)
        ctx.strokeStyle = "#4da6ff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const scale = 3.0; // zoom

        data.before.x.forEach((x, i) => {
            const px = centerX + x * scale;
            const py = centerY - data.before.y[i] * scale;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        });
        ctx.stroke();

        // Animate Cut (Red)
        setTimeout(() => {
            ctx.strokeStyle = "#ff4d4d";
            ctx.beginPath();
            data.after.x.forEach((x, i) => {
                const px = centerX + x * scale;
                const py = centerY - (data.after.y[i] * scale) - 20; // Offset slightly
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            });
            ctx.stroke();
            info.innerHTML = `<strong>Resection Complete:</strong> ${data.info}`;
        }, 800);

    } catch (e) {
        info.innerText = "Error: " + e.message;
    }
}

async function simulateBalancing() {
    const canvas = document.getElementById('kinematicsCanvas');
    const ctx = canvas.getContext('2d');
    const angDisplay = document.getElementById('flexion-angle');

    try {
        const res = await fetch('/api/workflow/balancing', { method: 'POST' });
        const resp = await res.json();
        const data = resp.data; // flexion_angles, medial_gaps, lateral_gaps

        let idx = 0;
        const animate = () => {
            if (idx >= data.flexion_angles.length) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw Graph
            // Background axes
            ctx.strokeStyle = "#333";
            ctx.beginPath();
            ctx.moveTo(20, 230); ctx.lineTo(380, 230); // X
            ctx.moveTo(20, 230); ctx.lineTo(20, 20);   // Y
            ctx.stroke();

            // Draw Medial (Green)
            ctx.strokeStyle = "#00cc66";
            ctx.beginPath();
            for (let i = 0; i <= idx; i++) {
                const x = 20 + (data.flexion_angles[i] / 125) * 360;
                const y = 230 - (data.medial_gaps[i] * 5); // scale gap
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Draw Lateral (Orange)
            ctx.strokeStyle = "#ff9933";
            ctx.beginPath();
            for (let i = 0; i <= idx; i++) {
                const x = 20 + (data.flexion_angles[i] / 125) * 360;
                const y = 230 - (data.lateral_gaps[i] * 5);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Update Text
            angDisplay.innerText = `${data.flexion_angles[idx]}° | Gaps: M=${data.medial_gaps[idx]} L=${data.lateral_gaps[idx]}`;

            idx++;
            requestAnimationFrame(animate);
        };
        animate();

    } catch (e) {
        console.error(e);
    }
}

async function simulateImplant() {
    const statusDiv = document.getElementById('implant-status');
    statusDiv.innerHTML = "Inserting Implant...";

    try {
        const res = await fetch('/api/workflow/implant', { method: 'POST' });
        const resp = await res.json();

        statusDiv.innerHTML = `
            <div style="background:var(--bg-elevated); padding:10px; width:100%;">
                <h4 style="color:var(--primary); margin:0;">Implant Seated</h4>
                <div style="color:#aaa; font-size:0.9em;">
                    Cement: ${resp.data.cement_pressure}<br/>
                    Alignment: ${resp.data.alignment}
                </div>
                <div style="margin-top:5px; height:6px; background:#333; border-radius:3px;">
                     <div style="width:100%; height:100%; background:#4CAF50; border-radius:3px;"></div>
                </div>
            </div>
        `;

    } catch (e) {
        statusDiv.innerText = "Error: " + e.message;
    }
}


// === Robot Functions ===

function logRobot(msg) {
    const log = document.getElementById('robot-logs');
    if (log) {
        log.innerHTML += `<div>> ${msg}</div>`;
        log.scrollTop = log.scrollHeight;
    }
}

async function checkRobotStatus() {
    logRobot("Checking robot status...");
    try {
        const res = await fetch('/api/robot/status');
        const data = await res.json();

        document.getElementById('serial-status').innerText =
            `Status: ${data.status}\nJoints: ${data.serial_joints.map(j => j.toFixed(2)).join(', ')}`;

        document.getElementById('parallel-status').innerText =
            `Status: ${data.status}\nBase Radius: ${data.parallel_base}`;

        logRobot("Status received.");
    } catch (e) {
        logRobot("Error: " + e.message);
    }
}

async function executeRobotResection() {
    logRobot("Initiating resection sequence...");
    document.getElementById('parallel-status').innerText = "Status: MOVING...";

    try {
        const res = await fetch('/api/robot/resect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target: 'knee_distal', depth: 9.0 })
        });
        const data = await res.json();

        logRobot(data.data.message);
        checkRobotStatus();

    } catch (e) {
        logRobot("Resection Failed: " + e.message);
    }
}

// Load initial data
loadEconomics();
