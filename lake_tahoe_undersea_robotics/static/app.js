/**
 * Lake Tahoe Undersea Robotics & Molecular Gene Combing Client Application
 * Handles AUV 6-DOF kinematics, single-molecule DNA combing animations,
 * p-adic topological network mapping, limnological charts, and Nature preprint.
 */

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initAUVModule();
    initGeneCombingModule();
    initPadicModule();
    initOntarioModule();
    initLimnologyCharts();
    setupEventListeners();
});

// Navigation Tabs
function initTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    const panes = document.querySelectorAll('.tab-pane');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            panes.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const target = document.getElementById(tab.dataset.tab);
            if (target) {
                target.classList.add('active');
            }
        });
    });
}

// ─── TAB 1: AUV UNDERSEA ROBOTICS MODULE ─────────────────────────────────────
let currentStations = {};
let lastAUVTelemetry = null;
let auvAnimOffset = 0;

async function initAUVModule() {
    try {
        const res = await fetch('/api/auv/stations');
        const json = await res.json();
        if (json.status === 'success') {
            currentStations = json.stations;
            updateStationDossier('crystal_bay_trench');
            runAUVTelemetry('crystal_bay_trench');
            initAUVCanvas();
        }
    } catch (err) {
        console.error('Error initializing AUV stations:', err);
    }
}

function updateStationDossier(stKey) {
    const st = currentStations[stKey];
    if (!st) return;

    const dossier = document.getElementById('auv-station-dossier');
    dossier.innerHTML = `
        <div class="target-title">${st.name} (Lat: ${st.lat}°N, Lon: ${st.lon}°W)</div>
        <div class="target-desc"><b>Bathymetric Depth:</b> ${st.depth_m} m &nbsp;|&nbsp; <b>Hydrostatic Pressure:</b> ${st.pressure_bar} bar &nbsp;|&nbsp; <b>Temp:</b> ${st.ambient_temp_c}°C</div>
        <div class="target-desc" style="margin-top:4px;"><b>Benthic Sediment:</b> ${st.sediment_type}</div>
        <div class="target-desc" style="margin-top:2px; color:var(--accent-cyan);"><b>eDNA Community:</b> ${st.primary_microbe}</div>
    `;
}

async function runAUVTelemetry(stKey) {
    const rpm = parseFloat(document.getElementById('auv-thruster-rpm').value);
    const trim = parseFloat(document.getElementById('auv-buoyancy-trim').value);

    try {
        const res = await fetch('/api/auv/telemetry', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ station: stKey, thruster_rpm: rpm, buoyancy_trim_cc: trim })
        });
        const json = await res.json();
        if (json.status === 'success') {
            const tel = json.telemetry;
            lastAUVTelemetry = tel;

            document.getElementById('auv-depth').innerText = `${tel.depth_m} m / ${tel.pressure_bar} bar`;
            document.getElementById('auv-speed').innerText = `${tel.speed_m_s} m/s (${tel.speed_knots} kt)`;
            document.getElementById('auv-power').innerText = `${tel.total_power_w} W / ${tel.battery_endurance_hours} hr`;
            document.getElementById('auv-latency').innerText = `${tel.acoustic_latency_ms} ms`;

            document.getElementById('val-drag').innerText = `${tel.thrust_drag_force_n} N`;
            document.getElementById('val-net-buoy').innerText = `${tel.net_buoyancy_n} N`;
            document.getElementById('val-pitch').innerText = `${tel.pitch_deg}°`;
            document.getElementById('val-range').innerText = `${tel.survey_range_km} km`;
            document.getElementById('val-temp').innerText = `${tel.ambient_temp_c}°C`;
        }
    } catch (err) {
        console.error('Error fetching AUV telemetry:', err);
    }
}

function initAUVCanvas() {
    const canvas = document.getElementById('auv-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    function renderAUV() {
        ctx.fillStyle = '#020714';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        auvAnimOffset += 0.04;
        const cx = canvas.width / 2;
        const cy = 110;
        const depth = lastAUVTelemetry ? lastAUVTelemetry.depth_m : 501.0;
        const pitch = lastAUVTelemetry ? lastAUVTelemetry.pitch_deg * (Math.PI / 180) : 0;

        // 1. Water depth gradient background
        const waterGrad = ctx.createLinearGradient(0, 0, 0, canvas.height);
        waterGrad.addColorStop(0, '#0c224a');
        waterGrad.addColorStop(0.6, '#06132e');
        waterGrad.addColorStop(1, '#020714');
        ctx.fillStyle = waterGrad;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // 2. Multibeam Sonar Ping Beam (from AUV belly down to lakebed)
        const sonarSpread = 45;
        ctx.beginPath();
        ctx.moveTo(cx, cy + 10);
        ctx.lineTo(cx - 160, canvas.height - 25);
        ctx.lineTo(cx + 160, canvas.height - 25);
        ctx.closePath();
        const sonarGrad = ctx.createLinearGradient(0, cy, 0, canvas.height);
        sonarGrad.addColorStop(0, 'rgba(6, 182, 212, 0.45)');
        sonarGrad.addColorStop(1, 'rgba(6, 182, 212, 0.03)');
        ctx.fillStyle = sonarGrad;
        ctx.fill();

        // Sonar wavefront lines
        for (let i = 1; i <= 4; i++) {
            const waveY = cy + 20 + ((auvAnimOffset * 40 + i * 35) % (canvas.height - cy - 45));
            ctx.beginPath();
            ctx.moveTo(cx - (waveY - cy) * 0.9, waveY);
            ctx.lineTo(cx + (waveY - cy) * 0.9, waveY);
            ctx.strokeStyle = 'rgba(6, 182, 212, 0.35)';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        // 3. Lake Tahoe Bathymetric Benthic Floor
        ctx.beginPath();
        ctx.moveTo(0, canvas.height - 25);
        for (let x = 0; x <= canvas.width; x += 20) {
            const yOffset = Math.sin(x * 0.02 + 1.2) * 8 + Math.cos(x * 0.04) * 4;
            ctx.lineTo(x, canvas.height - 25 + yOffset);
        }
        ctx.lineTo(canvas.width, canvas.height);
        ctx.lineTo(0, canvas.height);
        ctx.closePath();
        ctx.fillStyle = '#0f172a';
        ctx.fill();
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px system-ui';
        ctx.textAlign = 'left';
        ctx.fillText(`Benthic Bed Floor (Depth: ${depth}m • Lake Tahoe Trench)`, 15, canvas.height - 8);

        // 4. AUV Submersible Hull (6-DOF Torpedo shape with pitch rotation)
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(pitch);

        // Torpedo Yellow / Cyan Body
        ctx.fillStyle = '#0284c7';
        ctx.beginPath();
        ctx.roundRect(-75, -16, 140, 32, [16, 2, 2, 16]);
        ctx.fill();
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Front hemispherical nose dome (optical window)
        ctx.fillStyle = '#06b6d4';
        ctx.beginPath();
        ctx.arc(65, 0, 16, -Math.PI / 2, Math.PI / 2);
        ctx.fill();

        // Microfluidic Intake Port on bottom nose
        ctx.fillStyle = '#fbbf24';
        ctx.fillRect(40, 12, 12, 6);
        ctx.fillStyle = '#ffffff';
        ctx.font = '8px system-ui';
        ctx.fillText('eDNA INTAKE', 18, 26);

        // Conning tower / GPS / Satellite antenna fin
        ctx.fillStyle = '#0f172a';
        ctx.beginPath();
        ctx.moveTo(-10, -16);
        ctx.lineTo(5, -28);
        ctx.lineTo(15, -28);
        ctx.lineTo(18, -16);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = '#38bdf8';
        ctx.stroke();

        // Rear stabilizing fins
        ctx.fillStyle = '#0284c7';
        ctx.beginPath();
        ctx.moveTo(-65, -16);
        ctx.lineTo(-85, -30);
        ctx.lineTo(-75, -16);
        ctx.closePath();
        ctx.fill();

        ctx.beginPath();
        ctx.moveTo(-65, 16);
        ctx.lineTo(-85, 30);
        ctx.lineTo(-75, 16);
        ctx.closePath();
        ctx.fill();

        // Spinning Thruster Propeller at tail
        ctx.fillStyle = '#f8fafc';
        const propHeight = 18 * Math.sin(auvAnimOffset * 6);
        ctx.fillRect(-82, -propHeight / 2, 4, propHeight);

        ctx.restore();

        // 5. Thruster cavitation bubbles
        for (let b = 1; b <= 5; b++) {
            const bx = cx - 80 - b * 18 - Math.sin(auvAnimOffset + b) * 5;
            const by = cy + (b % 2 === 0 ? -4 : 4) + Math.cos(auvAnimOffset + b) * 3;
            ctx.beginPath();
            ctx.arc(bx, by, 2 + (b * 0.6), 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
            ctx.fill();
        }

        requestAnimationFrame(renderAUV);
    }

    renderAUV();
}

// ─── TAB 2: MOLECULAR GENE COMBING MODULE ───────────────────────────────────
let currentSpecies = {};
let lastCombingData = null;

async function initGeneCombingModule() {
    try {
        const res = await fetch('/api/combing/species');
        const json = await res.json();
        if (json.status === 'success') {
            currentSpecies = json.species;
            updateSpeciesDossier('cyclotella_gordonensis');
            runGeneCombingSimulation('cyclotella_gordonensis');
            initCombingCanvas();
        }
    } catch (err) {
        console.error('Error loading species:', err);
    }
}

function updateSpeciesDossier(specKey) {
    const sp = currentSpecies[specKey];
    if (!sp) return;

    const dossier = document.getElementById('combing-species-dossier');
    dossier.innerHTML = `
        <div class="target-title">${sp.species}</div>
        <div class="target-desc"><b>Ecological Function:</b> ${sp.role}</div>
        <div class="target-desc" style="margin-top:4px;">
            <b>Target Fragment:</b> ${sp.genome_fragment_kb} kb &nbsp;|&nbsp;
            <b>Linear Combed Length:</b> ${sp.combed_length_um} &mu;m &nbsp;|&nbsp;
            <b>Fluorophore:</b> <span style="color:${sp.fluorophore_color}; font-weight:700;">${sp.fluorophore}</span>
        </div>
        <div class="target-desc" style="margin-top:2px;"><b>Marker Cluster:</b> ${sp.marker_gene}</div>
    `;
}

async function runGeneCombingSimulation(specKey) {
    const shear = parseFloat(document.getElementById('combing-shear-rate').value);

    try {
        const res = await fetch('/api/combing/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ species: specKey, shear_rate: shear })
        });
        const json = await res.json();
        if (json.status === 'success') {
            const comb = json.combing;
            lastCombingData = comb;

            document.getElementById('combing-length').innerText = `${comb.combed_length_um} µm`;
            document.getElementById('combing-poc').innerText = `${comb.particulate_organic_carbon_ug_l} µg/L`;
            document.getElementById('combing-secchi').innerText = `${comb.secchi_depth_estimate_m} m`;

            // Render Particle State Bins
            const binsContainer = document.getElementById('particle-bins-container');
            binsContainer.innerHTML = '';
            comb.particle_microstate_bins.forEach(b => {
                const card = document.createElement('div');
                card.className = 'particle-bin-card';
                card.innerHTML = `
                    <div class="bin-size">${b.size_um} µm</div>
                    <div class="bin-frac">${b.fraction_pct}%</div>
                    <div class="bin-type">${b.type}</div>
                `;
                binsContainer.appendChild(card);
            });
        }
    } catch (err) {
        console.error('Error running gene combing:', err);
    }
}

function initCombingCanvas() {
    const canvas = document.getElementById('combing-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    function renderCombing() {
        ctx.fillStyle = '#020714';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw microfluidic glass substrate grid
        ctx.strokeStyle = 'rgba(30, 58, 138, 0.3)';
        ctx.lineWidth = 1;
        for (let x = 0; x < canvas.width; x += 35) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }
        for (let y = 0; y < canvas.height; y += 35) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
        }

        const cy = canvas.height / 2;
        const startX = 50;
        const lengthPx = 460;

        // Draw Silanized Glass Substrate Surface
        ctx.fillStyle = 'rgba(6, 182, 212, 0.05)';
        ctx.fillRect(startX - 20, cy - 45, lengthPx + 40, 90);
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.4)';
        ctx.strokeRect(startX - 20, cy - 45, lengthPx + 40, 90);

        ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
        ctx.font = '9px monospace';
        ctx.fillText('SILANIZED HYDROPHOBIC SUBSTRATE (VINYL-SILANE θc = 105°)', startX - 10, cy - 50);

        // Draw Combed DNA Backbone Fiber
        ctx.beginPath();
        ctx.moveTo(startX, cy);
        ctx.lineTo(startX + lengthPx, cy);
        ctx.strokeStyle = '#94a3b8';
        ctx.lineWidth = 2.8;
        ctx.stroke();

        // Scale bar
        ctx.fillStyle = '#38bdf8';
        ctx.font = 'bold 10px monospace';
        ctx.fillText('◄ 0.0 µm (0 kb)', startX, cy + 25);
        ctx.fillText(`${lastCombingData ? lastCombingData.combed_length_um : 84.0} µm (${lastCombingData ? lastCombingData.genome_fragment_kb : 168} kb) ►`, startX + lengthPx - 140, cy + 25);

        // Draw Optical Fluorescence Hybridization Barcodes
        if (lastCombingData && lastCombingData.probe_barcodes) {
            const fluorColor = lastCombingData.fluorophore_color || '#06b6d4';
            const totalKb = lastCombingData.genome_fragment_kb;

            lastCombingData.probe_barcodes.forEach((p, idx) => {
                const px = startX + (p.locus_kb / totalKb) * lengthPx;

                // Glowing fluorophore spot
                const grad = ctx.createRadialGradient(px, cy, 2, px, cy, 14);
                grad.addColorStop(0, fluorColor);
                grad.addColorStop(0.5, fluorColor);
                grad.addColorStop(1, 'transparent');

                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(px, cy, 14, 0, Math.PI * 2);
                ctx.fill();

                // Center intense bead
                ctx.fillStyle = '#ffffff';
                ctx.beginPath();
                ctx.arc(px, cy, 3.5, 0, Math.PI * 2);
                ctx.fill();

                // Locus tag label
                ctx.fillStyle = fluorColor;
                ctx.font = 'bold 9px monospace';
                ctx.textAlign = 'center';
                ctx.fillText(`${p.locus_kb}kb`, px, cy - 18 - (idx % 2 === 0 ? 0 : 12));
            });
        }

        requestAnimationFrame(renderCombing);
    }

    renderCombing();
}

// ─── TAB 3: p-ADIC TOPOLOGICAL PLACEMENT MODULE ──────────────────────────────
let lastNetworkData = null;

async function initPadicModule() {
    runPadicTopology(5);
    initPadicMapCanvas();
}

async function runPadicTopology(chosenPrime) {
    const thresh = parseFloat(document.getElementById('padic-thresh').value);

    try {
        const res = await fetch('/api/padic/network', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prime: chosenPrime, threshold: thresh })
        });
        const json = await res.json();
        if (json.status === 'success') {
            const net = json.network;
            lastNetworkData = net;

            document.getElementById('betti-0').innerText = `${net.betti_0_components} (Full Mesh)`;
            document.getElementById('betti-1').innerText = `${net.betti_1_cycles} Loops`;
            document.getElementById('tree-depth').innerText = `${net.bruhat_tits_tree_depth} Levels`;
            document.getElementById('coverage-area').innerText = `${net.coverage_area_sq_km} km²`;
        }
    } catch (err) {
        console.error('Error running p-adic topology:', err);
    }
}

function initPadicMapCanvas() {
    const canvas = document.getElementById('padic-map-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Geo coordinates bounding box for Lake Tahoe
    const minLat = 38.93, maxLat = 39.26;
    const minLon = -120.18, maxLon = -119.92;

    function project(lat, lon) {
        const x = ((lon - minLon) / (maxLon - minLon)) * (canvas.width - 80) + 40;
        const y = canvas.height - (((lat - minLat) / (maxLat - minLat)) * (canvas.height - 70) + 35);
        return { x, y };
    }

    function renderMap() {
        ctx.fillStyle = '#020714';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // 1. Lake Tahoe Water Silhouette (approximated natural shoreline polygon)
        const shorelineCoords = [
            [39.25, -120.02], [39.24, -120.00], [39.18, -119.95],
            [39.10, -119.94], [39.02, -119.94], [38.95, -119.98],
            [38.94, -120.05], [38.98, -120.10], [39.04, -120.12],
            [39.14, -120.15], [39.20, -120.14], [39.24, -120.08]
        ];

        ctx.beginPath();
        const firstPt = project(shorelineCoords[0][0], shorelineCoords[0][1]);
        ctx.moveTo(firstPt.x, firstPt.y);
        for (let i = 1; i < shorelineCoords.length; i++) {
            const pt = project(shorelineCoords[i][0], shorelineCoords[i][1]);
            ctx.lineTo(pt.x, pt.y);
        }
        ctx.closePath();
        ctx.fillStyle = 'rgba(2, 132, 199, 0.18)';
        ctx.fill();
        ctx.strokeStyle = '#0284c7';
        ctx.lineWidth = 2.0;
        ctx.stroke();

        // 2. Bathymetric Depth Contours (Crystal Bay deep trench in the north)
        const trenchCenter = project(39.21, -120.03);
        const depthGrad = ctx.createRadialGradient(trenchCenter.x, trenchCenter.y, 10, trenchCenter.x, trenchCenter.y, 110);
        depthGrad.addColorStop(0, 'rgba(6, 182, 212, 0.35)');
        depthGrad.addColorStop(1, 'transparent');
        ctx.fillStyle = depthGrad;
        ctx.beginPath();
        ctx.arc(trenchCenter.x, trenchCenter.y, 110, 0, Math.PI * 2);
        ctx.fill();

        // 3. Simplicial Complex Edges (p-Adic distances)
        if (lastNetworkData && lastNetworkData.edges) {
            const nodeMap = {};
            lastNetworkData.nodes.forEach(n => { nodeMap[n.id] = project(n.lat, n.lon); });

            ctx.lineWidth = 1.8;
            lastNetworkData.edges.forEach(e => {
                const p1 = nodeMap[e.source];
                const p2 = nodeMap[e.target];
                if (p1 && p2) {
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
                    ctx.stroke();
                }
            });
        }

        // 4. Geodetic Mooring Nodes
        if (lastNetworkData && lastNetworkData.nodes) {
            lastNetworkData.nodes.forEach(n => {
                const pt = project(n.lat, n.lon);

                // Deep water node halo
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(16, 185, 129, 0.3)';
                ctx.fill();

                // Core marker
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 4.5, 0, Math.PI * 2);
                ctx.fillStyle = (n.depth_m > 400) ? '#f43f5e' : '#10b981';
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1.2;
                ctx.stroke();

                // Label
                ctx.fillStyle = '#f8fafc';
                ctx.font = 'bold 9px system-ui';
                ctx.textAlign = 'left';
                ctx.fillText(`${n.id}: ${n.name.split(' ')[0]} (${n.depth_m}m)`, pt.x + 8, pt.y + 3);
            });
        }

        // Canvas legend overlay
        ctx.fillStyle = 'rgba(10, 19, 41, 0.85)';
        ctx.fillRect(10, 10, 180, 50);
        ctx.strokeStyle = '#1e3a8a';
        ctx.strokeRect(10, 10, 180, 50);
        ctx.fillStyle = '#38bdf8';
        ctx.font = 'bold 10px system-ui';
        ctx.fillText(`p-Adic Prime p = ${lastNetworkData ? lastNetworkData.chosen_prime_p : 5}`, 18, 26);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '9px system-ui';
        ctx.fillText(`Betti: β₀=${lastNetworkData ? lastNetworkData.betti_0_components : 1}, β₁=${lastNetworkData ? lastNetworkData.betti_1_cycles : 5} cycles`, 18, 42);

        requestAnimationFrame(renderMap);
    }

    renderMap();
}

// ─── TAB 4: LIMNOLOGICAL CHARTS ──────────────────────────────────────────────
let chartThermocline = null;
let chartSecchi = null;

function initLimnologyCharts() {
    // 1. Thermocline Profile Chart
    const ctxThermo = document.getElementById('chart-thermocline').getContext('2d');
    if (chartThermocline) chartThermocline.destroy();

    const depths = [0, 15, 30, 50, 75, 100, 150, 200, 300, 400, 501];
    const summerTemps = [19.2, 18.1, 14.2, 9.8, 6.5, 5.2, 4.6, 4.3, 4.2, 4.1, 4.1];
    const winterTemps = [5.5, 5.4, 5.2, 5.0, 4.8, 4.6, 4.4, 4.3, 4.2, 4.1, 4.1];

    chartThermocline = new Chart(ctxThermo, {
        type: 'line',
        data: {
            labels: depths.map(d => `${d}m`),
            datasets: [
                {
                    label: 'Summer Stratification Temp (°C)',
                    data: summerTemps,
                    borderColor: '#f97316',
                    backgroundColor: 'rgba(249, 115, 22, 0.15)',
                    borderWidth: 2.8,
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Winter Isothermal Temp (°C)',
                    data: winterTemps,
                    borderColor: '#38bdf8',
                    borderWidth: 2.2,
                    borderDash: [4, 4],
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { title: { display: true, text: 'Temperature (°C)', color: '#94a3b8' }, grid: { color: '#0f1f42' }, ticks: { color: '#94a3b8' } },
                x: { title: { display: true, text: 'Bathymetric Depth (meters)', color: '#94a3b8' }, grid: { color: '#0f1f42' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#f8fafc', font: { size: 10 } } } }
        }
    });

    // 2. Secchi Clarity Bar Chart
    const ctxSecchi = document.getElementById('chart-secchi').getContext('2d');
    if (chartSecchi) chartSecchi.destroy();

    const stations = ['Crystal Bay', 'Rubicon Pt', 'Glenbrook', 'Tahoe City', 'Emerald Bay', 'Stateline'];
    const secchiM = [31.5, 29.8, 28.2, 24.5, 14.8, 22.0];

    chartSecchi = new Chart(ctxSecchi, {
        type: 'bar',
        data: {
            labels: stations,
            datasets: [{
                label: 'Secchi Disk Clarity Depth (meters)',
                data: secchiM,
                backgroundColor: 'rgba(6, 182, 212, 0.75)',
                borderColor: '#06b6d4',
                borderWidth: 1.5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { title: { display: true, text: 'Clarity Depth (m)', color: '#94a3b8' }, min: 0, max: 35, grid: { color: '#0f1f42' }, ticks: { color: '#94a3b8' } },
                x: { grid: { color: '#0f1f42' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#f8fafc' } } }
        }
    });
}

// ─── EVENT LISTENERS ────────────────────────────────────────────────────────
function setupEventListeners() {
    // AUV Controls
    const stSelect = document.getElementById('auv-station-select');
    if (stSelect) {
        stSelect.addEventListener('change', (e) => {
            updateStationDossier(e.target.value);
            runAUVTelemetry(e.target.value);
        });
    }

    const rpmSlider = document.getElementById('auv-thruster-rpm');
    if (rpmSlider) {
        rpmSlider.addEventListener('input', (e) => {
            document.getElementById('rpm-val').innerText = `${e.target.value} RPM`;
        });
    }

    const trimSlider = document.getElementById('auv-buoyancy-trim');
    if (trimSlider) {
        trimSlider.addEventListener('input', (e) => {
            document.getElementById('trim-val').innerText = `${e.target.value} cc`;
        });
    }

    const btnRunAUV = document.getElementById('btn-run-auv');
    if (btnRunAUV) {
        btnRunAUV.addEventListener('click', () => {
            const st = document.getElementById('auv-station-select').value;
            runAUVTelemetry(st);
        });
    }

    // Combing Controls
    const spSelect = document.getElementById('combing-species-select');
    if (spSelect) {
        spSelect.addEventListener('change', (e) => {
            updateSpeciesDossier(e.target.value);
            runGeneCombingSimulation(e.target.value);
        });
    }

    const shearSlider = document.getElementById('combing-shear-rate');
    if (shearSlider) {
        shearSlider.addEventListener('input', (e) => {
            document.getElementById('shear-val').innerText = `${e.target.value} s⁻¹`;
        });
    }

    const btnRunCombing = document.getElementById('btn-run-combing');
    if (btnRunCombing) {
        btnRunCombing.addEventListener('click', () => {
            const sp = document.getElementById('combing-species-select').value;
            runGeneCombingSimulation(sp);
        });
    }

    // p-Adic Controls
    const primeSelect = document.getElementById('padic-prime-select');
    if (primeSelect) {
        primeSelect.addEventListener('change', (e) => {
            runPadicTopology(parseInt(e.target.value));
        });
    }

    const threshSlider = document.getElementById('padic-thresh');
    if (threshSlider) {
        threshSlider.addEventListener('input', (e) => {
            document.getElementById('thresh-val').innerText = e.target.value;
        });
    }

    const btnRunPadic = document.getElementById('btn-run-padic');
    if (btnRunPadic) {
        btnRunPadic.addEventListener('click', () => {
            const p = parseInt(document.getElementById('padic-prime-select').value);
            runPadicTopology(p);
        });
    }

    // Preprint recompile
    const btnRecompile = document.getElementById('btn-recompile-tahoe-pdf');
    if (btnRecompile) {
        btnRecompile.addEventListener('click', async () => {
            btnRecompile.innerText = 'Recompiling...';
            try {
                const res = await fetch('/api/preprint/generate');
                const json = await res.json();
                if (json.status === 'success') {
                    btnRecompile.innerText = '✅ PDF Ready';
                    const iframe = document.querySelector('.pdf-iframe');
                    if (iframe) {
                        iframe.src = '/api/preprint/view?t=' + Date.now();
                    }
                }
            } catch (err) {
                btnRecompile.innerText = '❌ Failed';
            }
            setTimeout(() => { btnRecompile.innerText = '🔄 Recompile PDF'; }, 3000);
        });
    }

    setupOntarioEventListeners();
}

// ─── TAB 4: LAKE ONTARIO QUBIT CIRCUITRY & 10-YR RECOVERY ────────────────────
let currentOntarioStations = {};
let lastOntarioSpecData = null;
let chartOntarioSpectrum = null;
let chartOntarioPhosphorus = null;
let chartOntarioDiporeia = null;
let ontarioFlowAnimOffset = 0;

async function initOntarioModule() {
    try {
        const res = await fetch('/api/ontario/stations');
        const json = await res.json();
        if (json.status === 'success') {
            currentOntarioStations = json.stations;
            const defaultStation = currentOntarioStations['yellow_creek_ravine'] ? 'yellow_creek_ravine' : 'toronto_inner_harbour';
            updateOntarioStationDossier(defaultStation);
            runOntarioSpectroscopy(defaultStation);
            loadOntarioRecovery10Yr();
            initOntarioFlowCanvas();
        }
    } catch (err) {
        console.error('Error initializing Lake Ontario module:', err);
    }
}

function updateOntarioStationDossier(stKey) {
    const st = currentOntarioStations[stKey];
    if (!st) return;

    const cfdInfo = st.cfd_velocity_m_s ? ` &nbsp;|&nbsp; <b>CFD Flow:</b> ${st.cfd_velocity_m_s} m/s (Re=${st.cfd_reynolds_re.toLocaleString()})` : '';
    const ecoliInfo = st.baseline_ecoli_cfu ? ` &nbsp;|&nbsp; <b>Baseline E.coli:</b> ${st.baseline_ecoli_cfu} CFU` : '';

    const dossier = document.getElementById('ontario-station-dossier');
    dossier.innerHTML = `
        <div class="target-title">${st.name} (Lat: ${st.lat}°N, Lon: ${st.lon}°W)</div>
        <div class="target-desc"><b>Survey Depth:</b> ${st.depth_m} m &nbsp;|&nbsp; <b>Total Phosphorus:</b> ${st.baseline_phosphorus_ug_l} µg/L &nbsp;|&nbsp; <b>Secchi:</b> ${st.baseline_secchi_m} m${cfdInfo}${ecoliInfo}</div>
        <div class="target-desc" style="margin-top:4px; color:var(--accent-yellow);"><b>Target Pollution Profile:</b> ${st.target_pollutants}</div>
    `;

    document.getElementById('ontario-depth-slider').value = st.depth_m;
    document.getElementById('ontario-depth-val').innerText = `${st.depth_m} m`;
}

async function runOntarioSpectroscopy(stKey) {
    const depth = parseFloat(document.getElementById('ontario-depth-slider').value);
    const flow = parseFloat(document.getElementById('ontario-flow-slider').value);

    try {
        const res = await fetch('/api/ontario/spectroscopy/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ station: stKey, depth_m: depth, intake_rate: flow })
        });
        const json = await res.json();
        if (json.status === 'success') {
            const data = json.data;
            lastOntarioSpecData = data;

            document.getElementById('spec-fidelity').innerText = `${data.qml_telemetry.classifier_fidelity}%`;
            document.getElementById('spec-speedup').innerText = data.qml_telemetry.quantum_speedup;
            document.getElementById('spec-tp').innerText = `${data.limnological_diagnostics.total_phosphorus_ug_l} µg/L`;
            document.getElementById('spec-do').innerText = `${data.limnological_diagnostics.dissolved_oxygen_mg_l} mg/L`;

            if (document.getElementById('val-cfd-re') && data.cfd_hydraulics) {
                document.getElementById('val-cfd-re').innerText = data.cfd_hydraulics.reynolds_number.toLocaleString();
            }
            if (document.getElementById('val-cfd-tke') && data.cfd_hydraulics) {
                document.getElementById('val-cfd-tke').innerText = `${data.cfd_hydraulics.turbulent_kinetic_energy_m2_s2} m²/s²`;
            }
            if (document.getElementById('val-cfd-shear') && data.cfd_hydraulics) {
                document.getElementById('val-cfd-shear').innerText = `${data.cfd_hydraulics.bed_shear_stress_pa} Pa`;
            }
            if (document.getElementById('val-cfd-ecoli') && data.limnological_diagnostics.ecoli_cfu_100ml) {
                document.getElementById('val-cfd-ecoli').innerText = `${data.limnological_diagnostics.ecoli_cfu_100ml} CFU`;
            }
            if (document.getElementById('val-intake-p') && data.cfd_hydraulics) {
                document.getElementById('val-intake-p').innerText = `${data.cfd_hydraulics.intake_suction_pa} Pa`;
            }

            renderOntarioSpectrumChart(data.spectroscopy_curve);
            renderDetectedSignatures(data.spectroscopy_curve.detected_signatures);
        }
    } catch (err) {
        console.error('Error running Lake Ontario spectroscopy:', err);
    }
}

function renderOntarioSpectrumChart(curve) {
    const ctx = document.getElementById('chart-ontario-spectrum').getContext('2d');
    if (chartOntarioSpectrum) chartOntarioSpectrum.destroy();

    chartOntarioSpectrum = new Chart(ctx, {
        type: 'line',
        data: {
            labels: curve.wavelengths_nm.map(w => `${w} nm`),
            datasets: [{
                label: 'In-Situ Ingestion Optical Spectrum (UV-Vis & Raman Intensity)',
                data: curve.intensities_au,
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.12)',
                borderWidth: 2.8,
                fill: true,
                tension: 0.25,
                pointRadius: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { title: { display: true, text: 'Fluorescence / Absorbance Intensity (A.U.)', color: '#94a3b8' }, min: 0, grid: { color: '#0f1f42' }, ticks: { color: '#94a3b8' } },
                x: { title: { display: true, text: 'Wavelength (nm)', color: '#94a3b8' }, grid: { color: '#0f1f42' }, ticks: { color: '#94a3b8', maxTicksLimit: 12 } }
            },
            plugins: { legend: { labels: { color: '#f8fafc' } } }
        }
    });
}

function renderDetectedSignatures(sigs) {
    const container = document.getElementById('ontario-signatures-container');
    container.innerHTML = '';
    sigs.forEach(s => {
        const div = document.createElement('div');
        div.className = 'particle-bin-card';
        div.innerHTML = `
            <div class="bin-size" style="color:var(--accent-yellow); font-weight:800;">${s.peak_nm} nm</div>
            <div class="bin-frac" style="font-size:0.85rem;">${s.name}</div>
            <div class="bin-type" style="color:var(--accent-cyan); font-weight:600;">${s.significance}</div>
        `;
        container.appendChild(div);
    });
}

async function loadOntarioRecovery10Yr() {
    try {
        const res = await fetch('/api/ontario/recovery/10yr');
        const json = await res.json();
        if (json.status === 'success') {
            const recov = json.recovery;

            document.getElementById('recov-p-load').innerText = `${recov.phosphorus_load_mt[9].toLocaleString()} MT/yr`;
            document.getElementById('recov-secchi').innerText = `${recov.secchi_depth_m[9].toFixed(2)} m`;
            document.getElementById('recov-diporeia').innerText = `${recov.diporeia_density_indiv_m2[9].toLocaleString()} /m²`;
            document.getElementById('recov-econ').innerText = `$${recov.cumulative_economic_value_cad_m[9]}M CAD`;

            renderOntarioRecoveryCharts(recov);
            renderOntarioRoadmap(recov.recovery_roadmap);
        }
    } catch (err) {
        console.error('Error loading 10-year Lake Ontario recovery:', err);
    }
}

function renderOntarioRecoveryCharts(recov) {
    // 1. Phosphorus, Secchi & Yellow Creek E. coli Trajectory
    const ctxP = document.getElementById('chart-ontario-phosphorus').getContext('2d');
    if (chartOntarioPhosphorus) chartOntarioPhosphorus.destroy();

    const datasets = [
        {
            label: 'Total Phosphorus Loading (MT/year)',
            data: recov.phosphorus_load_mt,
            borderColor: '#f43f5e',
            backgroundColor: 'rgba(244, 63, 94, 0.1)',
            yAxisID: 'y1',
            borderWidth: 2.8,
            tension: 0.3
        },
        {
            label: 'Secchi Transparency Depth (m)',
            data: recov.secchi_depth_m,
            borderColor: '#06b6d4',
            yAxisID: 'y2',
            borderWidth: 2.8,
            tension: 0.3
        }
    ];

    if (recov.yellow_creek_ecoli_cfu) {
        datasets.push({
            label: 'Yellow Creek Ravine E. coli (CFU/100mL ÷ 10)',
            data: recov.yellow_creek_ecoli_cfu.map(v => v / 10.0),
            borderColor: '#10b981',
            borderWidth: 2.2,
            borderDash: [5, 4],
            yAxisID: 'y1',
            tension: 0.3
        });
    }

    chartOntarioPhosphorus = new Chart(ctxP, {
        type: 'line',
        data: {
            labels: recov.years,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y1: {
                    type: 'linear',
                    position: 'left',
                    title: { display: true, text: 'Phosphorus (MT/yr) / E.coli (x10 CFU)', color: '#f43f5e' },
                    grid: { color: '#0f1f42' },
                    ticks: { color: '#f43f5e' }
                },
                y2: {
                    type: 'linear',
                    position: 'right',
                    title: { display: true, text: 'Secchi Depth (meters)', color: '#06b6d4' },
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#06b6d4' }
                },
                x: { grid: { color: '#0f1f42' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#f8fafc', font: { size: 10 } } } }
        }
    });

    // 2. Diporeia Recovery & Economic NPV Chart
    const ctxD = document.getElementById('chart-ontario-diporeia').getContext('2d');
    if (chartOntarioDiporeia) chartOntarioDiporeia.destroy();

    chartOntarioDiporeia = new Chart(ctxD, {
        type: 'bar',
        data: {
            labels: recov.years,
            datasets: [
                {
                    type: 'bar',
                    label: 'Benthic Diporeia Density (indiv/m²)',
                    data: recov.diporeia_density_indiv_m2,
                    backgroundColor: 'rgba(16, 185, 129, 0.75)',
                    borderColor: '#10b981',
                    yAxisID: 'y1',
                    borderWidth: 1.5
                },
                {
                    type: 'line',
                    label: 'Cumulative Economic NPV ($M CAD)',
                    data: recov.cumulative_economic_value_cad_m,
                    borderColor: '#fbbf24',
                    backgroundColor: 'rgba(251, 191, 36, 0.15)',
                    yAxisID: 'y2',
                    borderWidth: 2.5,
                    tension: 0.25
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y1: {
                    type: 'linear',
                    position: 'left',
                    title: { display: true, text: 'Diporeia Density (/m²)', color: '#10b981' },
                    grid: { color: '#0f1f42' },
                    ticks: { color: '#10b981' }
                },
                y2: {
                    type: 'linear',
                    position: 'right',
                    title: { display: true, text: 'Economic Value ($M CAD)', color: '#fbbf24' },
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#fbbf24' }
                },
                x: { grid: { color: '#0f1f42' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#f8fafc', font: { size: 10 } } } }
        }
    });
}

function renderOntarioRoadmap(roadmap) {
    const container = document.getElementById('ontario-roadmap-container');
    container.innerHTML = '';
    roadmap.forEach(r => {
        const div = document.createElement('div');
        div.className = 'protocol-card';
        div.innerHTML = `
            <div class="protocol-phase">${r.phase}</div>
            <div class="protocol-timeline">${r.horizon}</div>
            <div class="protocol-dose">${r.robot_deployment}</div>
            <div class="protocol-obj"><b>Action Targets:</b> ${r.action_targets}</div>
            <div class="protocol-milestone"><b>Milestone:</b> ${r.milestone}</div>
        `;
        container.appendChild(div);
    });
}

// In-Situ AUV Water Intake Canvas Simulation
function initOntarioFlowCanvas() {
    const canvas = document.getElementById('ontario-flow-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    function renderOntarioFlow() {
        ctx.fillStyle = '#020714';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ontarioFlowAnimOffset += 0.05;
        const cx = canvas.width / 2;
        const cy = 115;

        // 1. Flow background gradient & Ravine Culvert Channel
        const bgGrad = ctx.createLinearGradient(0, 0, canvas.width, 0);
        bgGrad.addColorStop(0, '#040d24');
        bgGrad.addColorStop(0.5, '#061a38');
        bgGrad.addColorStop(1, '#020714');
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw Riparian Ravine Channel Boundaries (Concrete Culvert & Bedrock Silt)
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvas.width, 22);
        ctx.fillRect(0, canvas.height - 22, canvas.width, 22);
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(0, 0, canvas.width, 22);
        ctx.strokeRect(0, canvas.height - 22, canvas.width, 22);

        ctx.fillStyle = '#94a3b8';
        ctx.font = 'bold 9px system-ui';
        ctx.textAlign = 'left';
        ctx.fillText('YELLOW CREEK RAVINE CULVERT CHANNEL (Re = 48,500 • v = 1.45 m/s)', 15, 15);
        ctx.fillText('BEDROCK SILT & SEDIMENT BOUNDARY (Shear Stress τw = 8.42 Pa)', 15, canvas.height - 8);

        // 2. CFD Turbulent Streamlines with Velocity Colormap
        const numLines = 9;
        const curVelocity = lastOntarioSpecData && lastOntarioSpecData.cfd_hydraulics ? lastOntarioSpecData.cfd_hydraulics.channel_velocity_m_s : 1.45;

        for (let i = 0; i < numLines; i++) {
            const yBase = 32 + i * 19;
            const flowOffset = (ontarioFlowAnimOffset * (30 * curVelocity) + i * 22) % (cx - 65);
            const eddyWiggle = Math.sin(ontarioFlowAnimOffset * 3 + i) * 6;

            ctx.beginPath();
            ctx.moveTo(0, yBase);
            ctx.bezierCurveTo(cx - 150, yBase + eddyWiggle, cx - 110, cy, cx - 65, cy);

            // Color gradient reflecting streamflow turbulence
            const strokeColor = (i % 2 === 0) ? 'rgba(6, 182, 212, 0.7)' : 'rgba(251, 191, 36, 0.65)';
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = (i % 3 === 0) ? 2.2 : 1.4;
            ctx.stroke();

            // Streamline velocity particles
            const particleX = (flowOffset) % (cx - 65);
            const particleY = yBase + (cy - yBase) * ((particleX) / (cx - 65)) + (eddyWiggle * (1 - particleX / (cx - 65)));
            ctx.beginPath();
            ctx.arc(particleX, particleY, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = (curVelocity > 1.0) ? '#f43f5e' : '#06b6d4';
            ctx.fill();
        }

        // 3. AUV Hull & Cross-Sectional Cutaway
        ctx.fillStyle = '#0284c7';
        ctx.beginPath();
        ctx.roundRect(cx - 65, cy - 35, 240, 70, [25, 6, 6, 25]);
        ctx.fill();
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Internal Optical Flow Cell Chamber (Glowing Transmon QPU Target)
        ctx.fillStyle = 'rgba(2, 6, 23, 0.9)';
        ctx.fillRect(cx - 15, cy - 22, 110, 44);
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 1.8;
        ctx.strokeRect(cx - 15, cy - 22, 110, 44);

        // Dilution Refrigerator / QPU Chip Icon
        ctx.fillStyle = '#fbbf24';
        ctx.beginPath();
        ctx.arc(cx + 40, cy, 14, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#f8fafc';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.fillStyle = '#0f172a';
        ctx.font = 'bold 8px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('6Q QPU', cx + 40, cy + 3);

        // UV-Vis Multi-wavelength Laser beam across sample
        const beamPhase = (Math.sin(ontarioFlowAnimOffset * 3) + 1) / 2;
        ctx.strokeStyle = `rgba(244, 63, 94, ${0.6 + beamPhase * 0.4})`;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(cx - 5, cy);
        ctx.lineTo(cx + 25, cy);
        ctx.stroke();

        // Chamber labels
        ctx.fillStyle = '#f8fafc';
        ctx.font = 'bold 9px system-ui';
        ctx.textAlign = 'left';
        ctx.fillText('CRYO QPU SPECTROPHOTOMETER (15 mK)', cx - 12, cy - 26);

        ctx.fillStyle = '#38bdf8';
        ctx.font = '8px system-ui';
        ctx.fillText('Laminar Sample Flow (150 mL/min)', cx - 12, cy + 32);

        // Water intake snout
        ctx.fillStyle = '#06b6d4';
        ctx.beginPath();
        ctx.moveTo(cx - 65, cy - 12);
        ctx.lineTo(cx - 80, cy - 18);
        ctx.lineTo(cx - 80, cy + 18);
        ctx.lineTo(cx - 65, cy + 12);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();

        requestAnimationFrame(renderOntarioFlow);
    }

    renderOntarioFlow();
}

function setupOntarioEventListeners() {
    const stSelect = document.getElementById('ontario-station-select');
    if (stSelect) {
        stSelect.addEventListener('change', (e) => {
            updateOntarioStationDossier(e.target.value);
            runOntarioSpectroscopy(e.target.value);
        });
    }

    const depthSlider = document.getElementById('ontario-depth-slider');
    if (depthSlider) {
        depthSlider.addEventListener('input', (e) => {
            document.getElementById('ontario-depth-val').innerText = `${e.target.value} m`;
        });
    }

    const flowSlider = document.getElementById('ontario-flow-slider');
    if (flowSlider) {
        flowSlider.addEventListener('input', (e) => {
            document.getElementById('ontario-flow-val').innerText = `${e.target.value} mL/min`;
        });
    }

    const btnRunSpec = document.getElementById('btn-run-ontario-spec');
    if (btnRunSpec) {
        btnRunSpec.addEventListener('click', () => {
            const st = document.getElementById('ontario-station-select').value;
            runOntarioSpectroscopy(st);
        });
    }

    const btnRecompile = document.getElementById('btn-recompile-ontario-pdf');
    if (btnRecompile) {
        btnRecompile.addEventListener('click', async () => {
            btnRecompile.innerText = 'Recompiling...';
            try {
                const res = await fetch('/api/ontario/preprint/generate');
                const json = await res.json();
                if (json.status === 'success') {
                    btnRecompile.innerText = '✅ Ontario PDF Ready';
                    const iframe = document.querySelector('iframe[title="Nature Lake Ontario Preprint PDF"]');
                    if (iframe) {
                        iframe.src = '/api/ontario/preprint/view?t=' + Date.now();
                    }
                }
            } catch (err) {
                btnRecompile.innerText = '❌ Failed';
            }
            setTimeout(() => { btnRecompile.innerText = '🔄 Recompile Ontario PDF'; }, 3000);
        });
    }
}
