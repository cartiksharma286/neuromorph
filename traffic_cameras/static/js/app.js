/**
 * Quantum-Statistical Traffic Signal Intelligence Client
 * Handles Leaflet map rendering, Plotly.js charts, tab navigation, and scenario simulation.
 */

// Global State
let allCameras = [];
let mapInstance = null;
let markersLayer = null;
let selectedCameraId = null;

// Dark theme common Plotly layout
const darkPlotLayout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { t: 26, b: 42, l: 48, r: 24 },
    font: { color: '#cbd5e1', family: 'Inter, sans-serif', size: 11 },
    xaxis: { gridcolor: 'rgba(255,255,255,0.06)', color: '#94a3b8' },
    yaxis: { gridcolor: 'rgba(255,255,255,0.06)', color: '#94a3b8' },
    legend: { orientation: 'h', x: 0, y: -0.25, font: { size: 10 } }
};

document.addEventListener('DOMContentLoaded', async () => {
    setupTabNavigation();
    initLeafletMap();
    await loadInitialData();
});

// ── 1. Tab Navigation ────────────────────────────────────────────────────────
function setupTabNavigation() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.getAttribute('data-tab');

            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.add('hidden'));

            btn.classList.add('active');
            const targetEl = document.getElementById(target);
            if (targetEl) targetEl.classList.remove('hidden');

            // Force map resize when switching to map tab
            if (target === 'map-tab' && mapInstance) {
                setTimeout(() => mapInstance.invalidateSize(), 200);
            }
        });
    });
}

// ── 2. Leaflet Map Initialization ────────────────────────────────────────────
function initLeafletMap() {
    // Centered at Toronto coordinates
    mapInstance = L.map('leaflet-map').setView([43.7150, -79.3850], 11);

    // Dark Matter CartoDB Basemap
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap contributors',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(mapInstance);

    markersLayer = L.layerGroup().addTo(mapInstance);
}

// ── 3. Data Fetching & Setup ─────────────────────────────────────────────────
async function loadInitialData() {
    try {
        // Fetch cameras
        const camsRes = await fetch('/api/cameras');
        const camsData = await camsRes.json();
        allCameras = camsData.cameras || [];

        updateSummaryKpis(allCameras);
        renderMapMarkers(allCameras);
        populateAllCamerasTable(allCameras);

        // Auto-select first camera for inspector
        if (allCameras.length > 0) {
            selectCameraNode(allCameras[0].id);
        }

        // Fetch Convergence data & render plots
        loadConvergenceData();
        loadExtremeValuesData();
        loadClustersData();
        loadMitigationImpactData();

    } catch (err) {
        console.error('Error loading initial data:', err);
    }
}

// ── 4. Summary KPIs Update ───────────────────────────────────────────────────
function updateSummaryKpis(cams) {
    const totalCams = cams.length;
    const totalViolations = cams.reduce((acc, c) => acc + (c.monthly_violations || 0), 0);
    const dilemmaCams = cams.filter(c => c.has_dilemma_zone).length;
    const dilemmaPct = ((dilemmaCams / max(1, totalCams)) * 100).toFixed(1);

    const meanV = (cams.reduce((acc, c) => acc + (c.quantum_observables?.expected_violation_operator || 0.5), 0) / max(1, totalCams)).toFixed(3);
    const meanEntropy = (cams.reduce((acc, c) => acc + (c.quantum_observables?.von_neumann_entropy || 0.9), 0) / max(1, totalCams)).toFixed(3);

    const el = id => document.getElementById(id);
    if (el('kpi-total-cameras')) el('kpi-total-cameras').textContent = totalCams;
    if (el('kpi-monthly-violations')) el('kpi-monthly-violations').textContent = totalViolations.toLocaleString();
    if (el('kpi-dilemma-count')) el('kpi-dilemma-count').textContent = `${dilemmaCams} (${dilemmaPct}%)`;
    if (el('kpi-quantum-operator')) el('kpi-quantum-operator').textContent = meanV;
    if (el('kpi-entropy')) el('kpi-entropy').textContent = meanEntropy;
}

function max(a, b) { return a > b ? a : b; }

// ── 5. Render Map Markers ────────────────────────────────────────────────────
function renderMapMarkers(cams) {
    markersLayer.clearLayers();

    cams.forEach(cam => {
        const color = cam.risk_color || '#eab308';
        const radius = cam.risk_tier === 'Severe Risk' ? 8 : (cam.risk_tier === 'High Risk' ? 7 : 5);

        const marker = L.circleMarker([cam.latitude, cam.longitude], {
            radius: radius,
            fillColor: color,
            color: '#ffffff',
            weight: 1.2,
            opacity: 0.9,
            fillOpacity: 0.75
        });

        // Hover tooltip
        marker.bindTooltip(`
            <div style="font-size:11px; font-family:sans-serif;">
                <b>${cam.name}</b><br/>
                <span style="color:${color}; font-weight:bold;">${cam.risk_tier}</span> · ${cam.monthly_violations} viol/mo<br/>
                Speed: ${cam.speed_85th_kmh} km/h · Dilemma: ${cam.dilemma_zone_gap_m} m
            </div>
        `, { direction: 'top', offset: [0, -5] });

        // Click to select & inspect
        marker.on('click', () => {
            selectCameraNode(cam.id);
        });

        markersLayer.addLayer(marker);
    });
}

function filterMapNodes() {
    const district = document.getElementById('filter-district').value;
    const risk = document.getElementById('filter-risk').value;

    let filtered = allCameras;
    if (district !== 'all') {
        filtered = filtered.filter(c => c.district === district);
    }
    if (risk !== 'all') {
        filtered = filtered.filter(c => c.risk_tier === risk);
    }
    renderMapMarkers(filtered);
}

// ── 6. Select Camera & Show Detail Telemetry ─────────────────────────────────
async function selectCameraNode(cid) {
    selectedCameraId = cid;
    const cam = allCameras.find(c => c.id === cid);
    if (!cam) return;

    // Pan map to marker smoothly
    if (mapInstance && cam.latitude && cam.longitude) {
        mapInstance.panTo([cam.latitude, cam.longitude], { animate: true, duration: 0.6 });
    }

    const el = id => document.getElementById(id);
    if (el('inspect-name')) el('inspect-name').textContent = cam.name;
    if (el('inspect-district')) el('inspect-district').textContent = `District: ${cam.district} · Ward: ${cam.ward} · Police Div: ${cam.police_division}`;
    if (el('inspect-tag')) {
        el('inspect-tag').textContent = cam.geospatial_tag || 'TAG-STANDARD';
        el('inspect-tag').className = `tag-badge ${cam.risk_tier === 'Severe Risk' ? 'tag-red' : (cam.risk_tier === 'High Risk' ? 'tag-orange' : 'tag-cyan')}`;
    }
    if (el('inspect-risk-badge')) {
        el('inspect-risk-badge').textContent = cam.risk_tier;
        el('inspect-risk-badge').style.color = cam.risk_color;
        el('inspect-risk-badge').style.borderColor = cam.risk_color;
    }
    if (el('inspect-speed')) el('inspect-speed').textContent = `${cam.speed_limit_kmh} / ${cam.speed_85th_kmh} km/h`;
    if (el('inspect-aadt')) el('inspect-aadt').textContent = `${cam.aadt_volume.toLocaleString()} veh/d`;
    if (el('inspect-dilemma')) el('inspect-dilemma').textContent = `${cam.dilemma_zone_gap_m} m (${cam.has_dilemma_zone ? 'Hazard' : 'Clear'})`;

    // Quantum readout
    const q = cam.quantum_observables || {};
    if (el('inspect-qfi')) el('inspect-qfi').textContent = `QFI F_Q: ${(q.quantum_fisher_information || 1.25).toFixed(2)}`;
    if (el('inspect-exp-v')) el('inspect-exp-v').textContent = (q.expected_violation_operator || 0.52).toFixed(3);
    if (el('inspect-exp-h')) el('inspect-exp-h').textContent = (q.expected_collision_hamiltonian || 0.88).toFixed(3);
    if (el('inspect-vn-entropy')) el('inspect-vn-entropy').textContent = (q.von_neumann_entropy || 0.91).toFixed(3);
    if (el('inspect-purity')) el('inspect-purity').textContent = (q.state_purity || 0.58).toFixed(3);
    if (el('inspect-uncertainty')) el('inspect-uncertainty').textContent = `${q.uncertainty_product || 3.35}`;

    // Fetch deep detail for plots
    try {
        const res = await fetch(`/api/camera/${cid}`);
        const data = await res.json();
        if (data.status === 'success') {
            renderCameraHourlyPlot(data.hourly_telemetry);
            renderKinematicCurvesPlot(data.kinematic_curves);
        }
    } catch (err) {
        console.error('Error fetching camera detail:', err);
    }
}

function renderCameraHourlyPlot(telemetry) {
    if (!telemetry || !document.getElementById('plot-hourly-violations')) return;

    Plotly.react('plot-hourly-violations', [{
        x: telemetry.hours,
        y: telemetry.violations,
        type: 'bar',
        marker: {
            color: '#38bdf8',
            opacity: 0.85
        }
    }], {
        ...darkPlotLayout,
        margin: { t: 10, b: 35, l: 35, r: 15 },
        xaxis: { ...darkPlotLayout.xaxis, tickangle: -45, tickfont: { size: 9 } },
        yaxis: { ...darkPlotLayout.yaxis, title: 'Violations / hr', tickfont: { size: 9 } }
    }, { responsive: true, displaylogo: false });
}

function renderKinematicCurvesPlot(kin) {
    if (!kin || !document.getElementById('plot-kinematic-curves')) return;

    Plotly.react('plot-kinematic-curves', [
        {
            x: kin.speeds_kmh,
            y: kin.stopping_dist_m,
            name: 'Stopping Dist X_0 (m)',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#ef4444', width: 2.2 }
        },
        {
            x: kin.speeds_kmh,
            y: kin.clearing_dist_m,
            name: 'Clearing Dist X_c (m)',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#10b981', width: 2.2, dash: 'dash' }
        }
    ], {
        ...darkPlotLayout,
        margin: { t: 10, b: 35, l: 40, r: 15 },
        xaxis: { ...darkPlotLayout.xaxis, title: 'Speed (km/h)', tickfont: { size: 9 } },
        yaxis: { ...darkPlotLayout.yaxis, title: 'Distance (m)', tickfont: { size: 9 } },
        legend: { orientation: 'h', x: 0, y: 1.15, font: { size: 9 } }
    }, { responsive: true, displaylogo: false });
}

// ── 7. Convergence Data & Plots (VQE, GLM, MCMC) ─────────────────────────────
async function loadConvergenceData() {
    try {
        const res = await fetch('/api/convergence');
        const data = await res.json();
        if (data.status !== 'success') return;

        // VQE Plot
        const vqe = data.quantum_vqe;
        if (vqe && document.getElementById('plot-vqe-convergence')) {
            Plotly.react('plot-vqe-convergence', [
                {
                    x: vqe.epochs, y: vqe.loss_history, name: 'VQE Energy Loss ⟨H(θ)⟩',
                    type: 'scatter', mode: 'lines+markers', line: { color: '#38bdf8', width: 2.5 }, marker: { size: 4 }
                },
                {
                    x: vqe.epochs, y: vqe.fidelity_history, name: 'Target State Fidelity',
                    type: 'scatter', mode: 'lines', line: { color: '#10b981', width: 2.0, dash: 'dot' }, yaxis: 'y2'
                }
            ], {
                ...darkPlotLayout,
                xaxis: { ...darkPlotLayout.xaxis, title: 'Training Epoch' },
                yaxis: { ...darkPlotLayout.yaxis, title: 'Hamiltonian Expectation Loss', color: '#38bdf8' },
                yaxis2: { title: 'Fidelity |⟨ψ|ψ*⟩|²', overlaying: 'y', side: 'right', color: '#10b981', range: [0.3, 1.05] }
            }, { responsive: true, displaylogo: false });
        }

        // Entropy & QFI Plot
        if (vqe && document.getElementById('plot-entropy-qfi')) {
            Plotly.react('plot-entropy-qfi', [
                {
                    x: vqe.epochs, y: vqe.entropy_history, name: 'Von Neumann Entropy S(ρ)',
                    type: 'scatter', mode: 'lines', line: { color: '#c084fc', width: 2.2 }
                },
                {
                    x: vqe.epochs, y: vqe.qfi_history, name: 'Quantum Fisher Info F_Q',
                    type: 'scatter', mode: 'lines', line: { color: '#f59e0b', width: 2.2, dash: 'dash' }, yaxis: 'y2'
                }
            ], {
                ...darkPlotLayout,
                xaxis: { ...darkPlotLayout.xaxis, title: 'Training Epoch' },
                yaxis: { ...darkPlotLayout.yaxis, title: 'Entropy (nats)', color: '#c084fc' },
                yaxis2: { title: 'Quantum Fisher Info F_Q', overlaying: 'y', side: 'right', color: '#f59e0b' }
            }, { responsive: true, displaylogo: false });
        }

        // Negative Binomial GLM Plot
        const glm = data.negative_binomial_glm;
        if (glm && document.getElementById('plot-glm-convergence')) {
            const iterAxis = Array.from({ length: glm.iterations }, (_, i) => i + 1);
            Plotly.react('plot-glm-convergence', [
                {
                    x: iterAxis, y: glm.log_likelihood, name: 'IRLS Log-Likelihood',
                    type: 'scatter', mode: 'lines+markers', line: { color: '#34d399', width: 2.5 }, marker: { size: 6 }
                },
                {
                    x: iterAxis, y: glm.beta_norm_diff, name: '||β^(t+1) - β^t|| Norm Diff',
                    type: 'scatter', mode: 'lines+markers', line: { color: '#f43f5e', width: 2.0, dash: 'dot' }, yaxis: 'y2'
                }
            ], {
                ...darkPlotLayout,
                xaxis: { ...darkPlotLayout.xaxis, title: 'Fisher Scoring Iteration (IRLS)', dtick: 1 },
                yaxis: { ...darkPlotLayout.yaxis, title: 'Log-Likelihood ln L(β)', color: '#34d399' },
                yaxis2: { title: 'Parameter Norm Diff', overlaying: 'y', side: 'right', color: '#f43f5e', type: 'log' }
            }, { responsive: true, displaylogo: false });

            // Populate GLM table
            populateGlmTable(glm.coefficients);
        }

        // MCMC Trace Plot
        const mcmc = data.mcmc_bayesian;
        if (mcmc && document.getElementById('plot-mcmc-trace')) {
            const stepAxis = Array.from({ length: mcmc.chain_1_trace.length }, (_, i) => i * 5);
            Plotly.react('plot-mcmc-trace', [
                { x: stepAxis, y: mcmc.chain_1_trace, name: 'Chain 1 (Init 0.15)', type: 'scatter', mode: 'lines', line: { color: '#38bdf8', width: 1.2 } },
                { x: stepAxis, y: mcmc.chain_2_trace, name: 'Chain 2 (Init 0.70)', type: 'scatter', mode: 'lines', line: { color: '#f59e0b', width: 1.2 } }
            ], {
                ...darkPlotLayout,
                xaxis: { ...darkPlotLayout.xaxis, title: 'MCMC Sampling Step' },
                yaxis: { ...darkPlotLayout.yaxis, title: 'Speed Elasticity β_speed', range: [0.1, 0.75] }
            }, { responsive: true, displaylogo: false });
        }

    } catch (err) {
        console.error('Error loading convergence data:', err);
    }
}

function populateGlmTable(coefs) {
    const tbody = document.getElementById('glm-tbody');
    if (!tbody || !coefs) return;

    tbody.innerHTML = coefs.map(c => {
        const isSig = c.p_value < 0.05;
        const sigBadge = isSig
            ? '<span style="color:#34d399; font-weight:700;">p &lt; 0.05 ***</span>'
            : '<span style="color:#94a3b8;">n.s.</span>';

        return `
            <tr>
                <td style="font-weight:600; color:#f8fafc;">${c.feature}</td>
                <td style="font-family:monospace; color:#38bdf8; font-weight:700;">${c.coefficient.toFixed(4)}</td>
                <td style="font-family:monospace; color:#cbd5e1;">${c.std_error.toFixed(4)}</td>
                <td style="font-family:monospace; color:#f59e0b;">${c.z_score.toFixed(3)}</td>
                <td style="font-family:monospace; color:#cbd5e1;">${c.p_value.toFixed(5)}</td>
                <td style="font-family:monospace; color:#a78bfa;">[${c.ci_lower_95.toFixed(4)}, ${c.ci_upper_95.toFixed(4)}]</td>
                <td>${sigBadge}</td>
            </tr>
        `;
    }).join('');
}

// ── 8. Extreme Value Theory (EVT) Data & Plots ───────────────────────────────
async function loadExtremeValuesData() {
    try {
        const res = await fetch('/api/extreme-values');
        const data = await res.json();
        if (data.status !== 'success') return;
        const evt = data.evt;

        const el = id => document.getElementById(id);
        if (el('evt-shape-val')) el('evt-shape-val').textContent = evt.gpd_shape_parameter_xi;
        if (el('evt-scale-val')) el('evt-scale-val').textContent = evt.gpd_scale_parameter_sigma;
        if (el('evt-return-1yr') && evt.return_levels_breach_seconds) {
            el('evt-return-1yr').textContent = `${evt.return_levels_breach_seconds[4]} s`;
        }

        // Survival curve plot
        if (document.getElementById('plot-evt-survival')) {
            Plotly.react('plot-evt-survival', [
                {
                    x: evt.t_grid_seconds, y: evt.empirical_survival, name: 'Empirical Tail Survival P(T > t)',
                    type: 'scatter', mode: 'markers', marker: { color: '#64748b', size: 5 }
                },
                {
                    x: evt.t_grid_seconds, y: evt.survival_gpd, name: 'GPD Parametric Fit G(z; ξ, σ)',
                    type: 'scatter', mode: 'lines', line: { color: '#f43f5e', width: 2.5 }
                }
            ], {
                ...darkPlotLayout,
                xaxis: { ...darkPlotLayout.xaxis, title: 'Red Clearance Encroachment Delay t (sec)' },
                yaxis: { ...darkPlotLayout.yaxis, title: 'Tail Survival Probability', type: 'log' }
            }, { responsive: true, displaylogo: false });
        }

        // Return levels plot
        if (document.getElementById('plot-evt-return')) {
            Plotly.react('plot-evt-return', [{
                x: evt.return_periods_days,
                y: evt.return_levels_breach_seconds,
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#fb923c', width: 2.5 },
                marker: { size: 6, color: '#f43f5e' }
            }], {
                ...darkPlotLayout,
                xaxis: { ...darkPlotLayout.xaxis, title: 'Return Period T (Days)', type: 'log' },
                yaxis: { ...darkPlotLayout.yaxis, title: 'Return Level Breach Time (seconds into red)' }
            }, { responsive: true, displaylogo: false });
        }

    } catch (err) {
        console.error('Error loading EVT data:', err);
    }
}

// ── 9. Spatial GMM Clusters ──────────────────────────────────────────────────
async function loadClustersData() {
    try {
        const res = await fetch('/api/clusters');
        const data = await res.json();
        if (data.status !== 'success') return;

        const container = document.getElementById('gmm-clusters-container');
        if (!container || !data.clusters) return;

        container.innerHTML = data.clusters.map(cl => `
            <div class="glass-panel" style="padding: 16px; border-left: 3px solid ${cl.color};">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span class="tag-badge" style="background:${cl.color}20; color:${cl.color}; border:1px solid ${cl.color}40;">${cl.geospatial_tag}</span>
                    <span style="font-size:11px; font-weight:700; color:#f8fafc;">${cl.camera_count} Cameras</span>
                </div>
                <h4 style="font-size:13.5px; font-weight:700; color:#f8fafc; margin-bottom:4px;">${cl.name}</h4>
                <p style="font-size:11px; color:#94a3b8; margin-bottom:10px;">${cl.description}</p>
                <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:6px; font-size:11px; background:rgba(0,0,0,0.3); padding:8px; border-radius:6px;">
                    <div>Speed: <b style="color:#38bdf8;">${cl.mean_speed} km/h</b></div>
                    <div>Gap: <b style="color:#f87171;">${cl.mean_dilemma_gap} m</b></div>
                    <div>Violations: <b style="color:#fbbf24;">${cl.mean_monthly_violations}/mo</b></div>
                </div>
            </div>
        `).join('');

    } catch (err) {
        console.error('Error loading clusters:', err);
    }
}

function populateAllCamerasTable(cams) {
    const tbody = document.getElementById('all-cameras-tbody');
    if (!tbody || !cams) return;

    tbody.innerHTML = cams.map(c => `
        <tr style="cursor:pointer;" onclick="selectCameraFromTable(${c.id})">
            <td style="font-family:monospace; color:#38bdf8; font-weight:700;">#${c.id}</td>
            <td style="color:#f8fafc; font-weight:600;">${c.name}</td>
            <td style="color:#94a3b8;">${c.district} (${c.ward})</td>
            <td style="color:#cbd5e1;">${c.speed_limit_kmh} / ${c.speed_85th_kmh} km/h</td>
            <td style="color:${c.has_dilemma_zone ? '#f87171' : '#34d399'}; font-weight:600;">${c.dilemma_zone_gap_m} m</td>
            <td style="font-family:monospace; color:#fbbf24; font-weight:700;">${c.monthly_violations}</td>
            <td>
                <span style="font-size:10px; padding:2px 7px; border-radius:10px; font-weight:700; background:${c.risk_color}22; color:${c.risk_color}; border:1px solid ${c.risk_color}44;">
                    ${c.risk_tier}
                </span>
            </td>
            <td>
                <span class="tag-badge tag-cyan">${c.geospatial_tag || 'TAG-STANDARD'}</span>
            </td>
        </tr>
    `).join('');
}

function selectCameraFromTable(cid) {
    // Switch to map tab and select camera
    const mapTabBtn = document.querySelector('[data-tab="map-tab"]');
    if (mapTabBtn) mapTabBtn.click();
    selectCameraNode(cid);
}

function searchCameraTable() {
    const q = document.getElementById('search-camera-table').value.toLowerCase();
    const filtered = allCameras.filter(c =>
        c.name.toLowerCase().includes(q) ||
        String(c.id).includes(q) ||
        c.district.toLowerCase().includes(q) ||
        (c.geospatial_tag && c.geospatial_tag.toLowerCase().includes(q))
    );
    populateAllCamerasTable(filtered);
}

// ── 10. Risk Mitigation & Scenario Simulator ─────────────────────────────────
async function triggerSimulation() {
    const yellow = parseFloat(document.getElementById('sim-yellow')?.value || 0.6);
    const allRed = parseFloat(document.getElementById('sim-allred')?.value || 0.4);
    const flow = parseFloat(document.getElementById('sim-flow')?.value || 1.0);
    const weather = document.getElementById('sim-weather')?.value || 'clear';
    const quantumGating = document.getElementById('sim-quantum-gating')?.checked ?? true;

    try {
        const res = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                yellow_extension_s: yellow,
                all_red_extension_s: allRed,
                flow_multiplier: flow,
                weather_condition: weather,
                quantum_gating: quantumGating
            })
        });

        const data = await res.json();
        if (data.status !== 'success') throw new Error(data.message || 'Simulation failed');
        const mit = data.mitigation;

        // Update KPIs
        const el = id => document.getElementById(id);
        if (el('kpi-risk-reduction')) el('kpi-risk-reduction').textContent = `-${mit.overall_reduction_pct}%`;
        if (el('pol-violations-monthly')) el('pol-violations-monthly').textContent = `${mit.total_violations_saved_monthly.toLocaleString()} / mo`;
        if (el('pol-crashes-annual')) el('pol-crashes-annual').textContent = `${Math.round(mit.annual_crashes_prevented)} Collisions`;
        if (el('pol-savings-cad')) el('pol-savings-cad').textContent = `$${(mit.annual_economic_savings_cad / 1e6).toFixed(1)}M CAD`;

        // Render Before vs After Plot
        renderMitigationImpactPlot(mit);

    } catch (err) {
        console.error('Simulation error:', err);
    }
}

async function loadMitigationImpactData() {
    await triggerSimulation();
}

function renderMitigationImpactPlot(mit) {
    if (!document.getElementById('plot-mitigation-impact') || !allCameras) return;

    // Group by district
    const districts = ['Toronto and East York', 'North York', 'Scarborough', 'Etobicoke York'];
    const beforeCounts = districts.map(d => {
        return allCameras.filter(c => c.district === d).reduce((acc, c) => acc + c.monthly_violations, 0);
    });
    const reductionRate = (mit.overall_reduction_pct || 45.0) / 100.0;
    const afterCounts = beforeCounts.map(b => Math.round(b * (1.0 - reductionRate)));

    Plotly.react('plot-mitigation-impact', [
        {
            x: districts, y: beforeCounts, name: 'Baseline Violations',
            type: 'bar', marker: { color: '#ef4444', opacity: 0.85 }
        },
        {
            x: districts, y: afterCounts, name: 'Mitigated Violations (Quantum Retimed)',
            type: 'bar', marker: { color: '#10b981', opacity: 0.85 }
        }
    ], {
        ...darkPlotLayout,
        barmode: 'group',
        margin: { t: 20, b: 45, l: 50, r: 20 },
        xaxis: { ...darkPlotLayout.xaxis, tickangle: -15 },
        yaxis: { ...darkPlotLayout.yaxis, title: 'Monthly Red Light Violations' }
    }, { responsive: true, displaylogo: false });
}
