
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
    const statsBox = document.getElementById('reg-stats');

    status.innerText = "Initializing Evolutionary Propagation...";
    try {
        const res = await fetch('/api/workflow/register', { method: 'POST' });
        const resp = await res.json();
        const m = resp.data.transform_matrix;
        const hist = resp.data.convergence_history;
        const stats = resp.data.statistics;

        // Animate Convergence
        let step = 0;
        const animateConv = () => {
            if (step >= hist.length) {
                // Final Matrix Display
                let matrixHtml = "<table style='font-size:0.7em; background:#111; margin-top:5px; border-collapse:collapse;'>";
                m.forEach(row => {
                    matrixHtml += "<tr>";
                    row.forEach(val => matrixHtml += `<td style='padding:2px 5px; border:1px solid #333;'>${val.toFixed(2)}</td>`);
                    matrixHtml += "</tr>";
                });
                matrixHtml += "</table>";

                status.innerHTML = `
                    <div style="color:#00ffcc">Converged via Quantum Geodesic Mapping</div>
                    <strong>Final RMS Error:</strong> ${resp.data.rms_error.toFixed(4)} mm<br/>
                    ${matrixHtml}
                `;

                // Show Statistics (Repeatability)
                if (statsBox) {
                    statsBox.innerHTML = `
                        <strong>Repeatability Score:</strong> <span style="color:#00ffcc; font-size:1.2em;">${stats.repeatability_score.toFixed(1)}/100</span><br/>
                        <strong>Confidence Interval:</strong> ${stats.confidence_interval}<br/>
                        <div style="margin-top:5px; height:30px; display:flex; align-items:flex-end; gap:2px;">
                            ${stats.trial_samples.map(s => `<div style="flex:1; background:#555; height:${(s / 10) * 100}%"></div>`).join('')}
                        </div>
                        <div style="font-size:0.8em; color:#aaa;">Stability across simulated trials</div>
                    `;
                }

                return;
            }

            status.innerText = `Iterating Epoch ${step + 1}/${hist.length}: Error ${hist[step].toFixed(4)}...`;
            step++;
            setTimeout(animateConv, 50); // Faster animation
        };
        animateConv();

    } catch (e) { status.innerText = "Error: " + e.message; }
}

// ... tracking unchanged ...

// === Workflow / Planning Functions ===

async function simulateResection() {
    const canvas = document.getElementById('resectionCanvas');
    const ctx = canvas.getContext('2d');
    const overlay = document.getElementById('plan-overlay');
    const profileSelect = document.getElementById('profile-select');
    const selectedProfile = profileSelect ? profileSelect.value : 'CR';

    // Reset Canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    try {
        const res = await fetch('/api/workflow/resection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ profile: selectedProfile })
        });
        const resp = await res.json();
        const data = resp.data;

        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const scale = 5.0; // zoom in

        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';

        // 1. Draw Native Bone (Semi-transparent Blue Fill)
        ctx.beginPath();
        data.x.forEach((x, i) => {
            const px = centerX + x * scale;
            const py = centerY - data.y_native[i] * scale + 50;
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        // complete the shape roughly for filling
        ctx.lineTo(centerX + 250, centerY + 200);
        ctx.lineTo(centerX - 250, centerY + 200);
        ctx.closePath();

        const gradBone = ctx.createLinearGradient(0, 0, 0, canvas.height);
        gradBone.addColorStop(0, "rgba(77, 166, 255, 0.2)");
        gradBone.addColorStop(1, "rgba(77, 166, 255, 0.05)");
        ctx.fillStyle = gradBone;
        ctx.fill();
        ctx.strokeStyle = "#4da6ff";
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        ctx.setLineDash([]);

        // 2. Animate Resection Cuts (White Solid)
        setTimeout(() => {
            ctx.beginPath();
            data.x.forEach((x, i) => {
                const px = centerX + x * scale;
                const py = centerY - data.y_cut[i] * scale + 50;
                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            });
            ctx.lineWidth = 3;
            ctx.strokeStyle = "#fff";
            ctx.stroke();

            // Trigger Implant Animation
            drawImplant();
        }, 500);

        // 3. Draw Implant (Metallic Gold Gradient)
        const drawImplant = () => {
            let offset = 50; // Start high
            const animateImp = () => {
                if (offset <= 0) {
                    overlay.style.display = 'block';
                    overlay.innerHTML = `
                        <div style="color:#4da6ff;">Native Bone</div>
                        <div style="color:#ffcc00;">${selectedProfile} Implant Profile</div>
                        <div style="color:#aaa;">Cut Depth: 9mm</div>
                     `;
                    simulateInsertionHaptics();
                    return;
                }

                // Redraw frame logic simplified: ideally we should redraw BG too but we'll overlay

                ctx.beginPath();

                // Outer Curve (Implant Top)
                for (let i = 0; i < data.x.length; i++) {
                    const px = centerX + data.x[i] * scale;
                    const py = centerY - (data.y_implant_outer[i] * scale) + 50 - offset;
                    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
                }
                // Inner Curve (Cut match) --> close shape
                for (let i = data.x.length - 1; i >= 0; i--) {
                    const px = centerX + data.x[i] * scale;
                    const py = centerY - (data.y_cut[i] * scale) + 50 - offset;
                    ctx.lineTo(px, py);
                }
                ctx.closePath();

                const gradImp = ctx.createLinearGradient(0, centerY - 100, 0, centerY + 100);
                gradImp.addColorStop(0, "#ffcc00");
                gradImp.addColorStop(0.5, "#ffee99");
                gradImp.addColorStop(1, "#b38f00");

                ctx.fillStyle = gradImp;
                ctx.shadowBlur = 10;
                ctx.shadowColor = "rgba(255, 204, 0, 0.5)";
                ctx.fill();
                ctx.shadowBlur = 0;

                // Draw Cam Post for PS
                if (selectedProfile === 'PS' && offset <= 25) {
                    // small detail
                }

                offset -= 2;
                requestAnimationFrame(animateImp);
            }
            animateImp();
        };

    } catch (e) {
        console.error(e);
        ctx.fillText("Error: " + e.message, 20, 30);
    }
}

// Update Analytic Loader for Tibial Stats
async function loadPostOp() {
    const grid = document.getElementById('analytics-grid');
    grid.style.display = 'grid';
    grid.innerHTML = '<div style="color:#aaa;">Simulating 5000 simulations...</div>';

    try {
        const res = await fetch('/api/workflow/postop');
        const resp = await res.json();
        const data = resp.data;

        grid.innerHTML = `
            <div class="card">
                <h3>Functional Outcomes</h3>
                <div class="result-box">
                    Flexion: ${data.avg_flexion.toFixed(1)}° (σ=${data.std_flexion.toFixed(1)})<br/>
                    Extension: ${data.avg_extension.toFixed(1)}°
                </div>
                <div id="flexion-hist" style="height:100px; background:#111; margin-top:5px; display:flex; align-items:flex-end;"></div>
            </div>
            
            <div class="card">
                <h3>Tibial Parameters</h3>
                <div class="result-box">
                    Slope: ${data.avg_tibial_slope.toFixed(1)}° (Target 3-7°)<br/>
                    Varus: ${data.avg_tibial_varus.toFixed(1)}° (Target 0°)
                </div>
            </div>
            
            <div class="card">
                <h3>Health Economics</h3>
                <div class="result-box">
                    Avg Cost: $${data.avg_cost.toLocaleString()}<br/>
                    Variance: ${(data.avg_cost * 0.1).toFixed(0)}
                </div>
                 <div id="cost-hist" style="height:100px; background:#111; margin-top:5px; display:flex; align-items:flex-end;"></div>
            </div>
        `;

        // Render Histograms
        const fHist = document.getElementById('flexion-hist');
        const maxF = Math.max(...data.flexion_bins);
        data.flexion_bins.forEach(val => {
            const h = (val / maxF) * 100;
            fHist.innerHTML += `<div style="flex:1; background:#4da6ff; margin:0 1px; height:${h}%"></div>`;
        });

        const cHist = document.getElementById('cost-hist');
        const maxC = Math.max(...data.cost_bins);
        data.cost_bins.forEach(val => {
            const h = (val / maxC) * 100;
            cHist.innerHTML += `<div style="flex:1; background:#ff9933; margin:0 1px; height:${h}%"></div>`;
        });

    } catch (e) { grid.innerText = "Error: " + e.message; }
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

// === Workflow / Planning Functions ===

async function simulateResection() {
    const canvas = document.getElementById('resectionCanvas');
    const ctx = canvas.getContext('2d');
    const overlay = document.getElementById('plan-overlay');

    // Reset Canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    try {
        const res = await fetch('/api/workflow/resection', { method: 'POST' });
        const resp = await res.json();
        const data = resp.data;
        // data has: x, y_native, y_cut, y_implant_outer

        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const scale = 5.0; // zoom in

        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';

        // 1. Draw Native Bone (Semi-transparent Blue Fill)
        ctx.beginPath();
        data.x.forEach((x, i) => {
            const px = centerX + x * scale;
            const py = centerY - data.y_native[i] * scale + 50; // shift down
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        // complete the shape roughly for filling
        ctx.lineTo(centerX + 250, centerY + 200);
        ctx.lineTo(centerX - 250, centerY + 200);
        ctx.closePath();

        const gradBone = ctx.createLinearGradient(0, 0, 0, canvas.height);
        gradBone.addColorStop(0, "rgba(77, 166, 255, 0.2)");
        gradBone.addColorStop(1, "rgba(77, 166, 255, 0.05)");
        ctx.fillStyle = gradBone;
        ctx.fill();
        ctx.strokeStyle = "#4da6ff";
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        ctx.setLineDash([]);

        // 2. Animate Resection Cuts (White Solid)
        setTimeout(() => {
            ctx.beginPath();
            data.x.forEach((x, i) => {
                const px = centerX + x * scale;
                const py = centerY - data.y_cut[i] * scale + 50;
                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            });
            ctx.lineWidth = 3;
            ctx.strokeStyle = "#fff";
            ctx.stroke();

            // Trigger Implant Animation
            drawImplant();
        }, 500);

        // 3. Draw Implant (Metallic Gold Gradient)
        const drawImplant = () => {
            let offset = 50; // Start high
            const animateImp = () => {
                if (offset <= 0) {
                    overlay.style.display = 'block';
                    simulateInsertionHaptics(); // Trigger haptics automatically
                    return;
                }

                // Redraw frame (simulated layering, ideally we'd use layers)
                // Just draw implant on top
                ctx.beginPath();

                // Outer Curve
                for (let i = 0; i < data.x.length; i++) {
                    const px = centerX + data.x[i] * scale;
                    const py = centerY - (data.y_implant_outer[i] * scale) + 50 - offset;
                    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
                }
                // Inner Curve (Cut match) --> close shape
                for (let i = data.x.length - 1; i >= 0; i--) {
                    const px = centerX + data.x[i] * scale;
                    const py = centerY - (data.y_cut[i] * scale) + 50 - offset;
                    ctx.lineTo(px, py);
                }
                ctx.closePath();

                const gradImp = ctx.createLinearGradient(0, centerY - 100, 0, centerY + 100);
                gradImp.addColorStop(0, "#ffcc00");
                gradImp.addColorStop(0.5, "#ffee99");
                gradImp.addColorStop(1, "#b38f00");

                ctx.fillStyle = gradImp;
                ctx.shadowBlur = 10;
                ctx.shadowColor = "rgba(255, 204, 0, 0.5)";
                ctx.fill();
                ctx.shadowBlur = 0;

                offset -= 2;
                requestAnimationFrame(animateImp);
            }
            animateImp();
        };

    } catch (e) {
        console.error(e);
        ctx.fillText("Error: " + e.message, 20, 30);
    }
}

async function simulateInsertionHaptics() {
    // Fills the "Implant Viz" box
    const div = document.getElementById('implant-viz');
    div.innerHTML = "<div style='color:#4da6ff'>Analysing Press-Fit...</div>";

    try {
        const res = await fetch('/api/workflow/implant', { method: 'POST' });
        const resp = await res.json();
        const d = resp.data;

        div.innerHTML = `
            <div style="display:flex; gap:15px; width:100%; padding:0 20px;">
                <div style="flex:1;">
                    <div style="font-size:0.8em; color:#aaa;">Alignment</div>
                    <div style="font-size:1.1em; color:#fff;">${d.alignment}</div>
                </div>
                <div style="flex:1;">
                    <div style="font-size:0.8em; color:#aaa;">Stability</div>
                    <div style="font-size:1.1em; color:#00ffcc;">${d.stability}</div>
                </div>
                <div style="flex:1; border-left:1px solid #333; padding-left:15px;">
                     <div style="font-size:0.8em; color:#aaa;">Cement Pressure</div>
                     <div style="height:6px; background:#444; width:100%; margin-top:5px; border-radius:3px;">
                         <div style="height:100%; background:#00ffcc; width:90%; border-radius:3px;"></div>
                     </div>
                </div>
            </div>
        `;

    } catch (e) {
        div.innerText = "Error.";
    }
}

async function simulateBalancing() {
    const canvas = document.getElementById('kinematicsCanvas');
    const ctx = canvas.getContext('2d');
    const angDisplay = document.getElementById('flexion-angle');

    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    try {
        const res = await fetch('/api/workflow/balancing', { method: 'POST' });
        const resp = await res.json();
        const data = resp.data;

        // Config
        const padding = 30;
        const w = canvas.width - padding * 2;
        const h = canvas.height - padding * 2;
        const maxY = 15; // mm gap

        // Helper: Map data to coords
        const toX = (angle) => padding + (angle / 145) * w;
        const toY = (mm) => padding + h - (mm / maxY) * h;

        // 1. Draw Grid & Background
        ctx.fillStyle = "#0f0f11";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Target Safe Zone (Green Band 9-11mm)
        const yHigh = toY(data.target_max);
        const yLow = toY(data.target_min);
        ctx.fillStyle = "rgba(0, 255, 128, 0.1)";
        ctx.fillRect(padding, yHigh, w, yLow - yHigh);

        // Grid Lines
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1;
        ctx.beginPath();
        // Horizontals (0, 5, 10, 15)
        for (let gap = 0; gap <= maxY; gap += 5) {
            const y = toY(gap);
            ctx.moveTo(padding, y); ctx.lineTo(padding + w, y);
            ctx.fillStyle = "#666"; ctx.fillText(gap, 5, y + 3);
        }
        // Verticals (0, 45, 90, 135)
        [0, 45, 90, 135].forEach(ang => {
            const x = toX(ang);
            ctx.moveTo(x, padding + h); ctx.lineTo(x, padding);
            ctx.fillText(ang + "°", x - 5, canvas.height - 5);
        });
        ctx.stroke();

        // 2. Draw Curves
        const drawCurve = (vals, color) => {
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            vals.forEach((mm, i) => {
                const x = toX(data.flexion_angles[i]);
                const y = toY(mm);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();

            // Draw Points
            ctx.fillStyle = color;
            vals.forEach((mm, i) => {
                if (i % 10 === 0) { // sparse points
                    const x = toX(data.flexion_angles[i]);
                    const y = toY(mm);
                    ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2); ctx.fill();
                }
            });
        };

        // Draw Medial
        drawCurve(data.medial_gaps, "#00cc66");
        // Draw Lateral
        drawCurve(data.lateral_gaps, "#ff9933");

        // 3. Labels
        ctx.fillStyle = "#00cc66"; ctx.fillText("Medial", padding + 10, padding + 10);
        ctx.fillStyle = "#ff9933"; ctx.fillText("Lateral", padding + 60, padding + 10);

        angDisplay.innerText = "Graph Generated";

    } catch (e) { console.error(e); }
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
