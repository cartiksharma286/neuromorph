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
}
