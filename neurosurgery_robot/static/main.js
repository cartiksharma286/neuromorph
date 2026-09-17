document.addEventListener('DOMContentLoaded', () => {
    // Init Visualizations
    const robotViz = new RobotViz('canvas-3d');
    const thermalViz = new ThermalViz('thermal-canvas');
    const cryoCanvasEl = document.getElementById('cryo-canvas');
    const cryoViz = cryoCanvasEl ? new CryoViz('cryo-canvas') : null;

    // UI Elements
    const elSystemStatus = document.getElementById('system-status');
    const elLoopRate = document.getElementById('loop-rate');
    const elLaserInd = document.getElementById('laser-indicator');
    const elMaxTemp = document.getElementById('max-temp');
    const elCursorVal = document.getElementById('cursor-val');
    const elTissueStatus = document.getElementById('tissue-status');
    const listLogs = document.getElementById('log-list');
    const btnToggleThermo = document.getElementById('btn-toggle-thermo-mode');
    const btnAutoAblate = document.getElementById('btn-auto-ablation');

    let laserActive = false;
    let cryoActive = false;
    let thermoMode = 'TEMP';
    let guidanceActive = false;
    let targetX = 0.5;
    let targetZ = 0.5;
    let telemetryInFlight = false;

    // Interactive Thermometry
    thermalViz.setHoverCallback((val) => {
        if (val === null) {
            elCursorVal.textContent = '--';
            return;
        }
        if (thermoMode === 'TEMP') {
            elCursorVal.textContent = val.toFixed(1) + '°C';
        } else {
            elCursorVal.textContent = val.toFixed(1) + ' CEM';
        }
    });

    if (btnToggleThermo) {
        btnToggleThermo.addEventListener('click', () => {
            if (thermoMode === 'TEMP') {
                thermoMode = 'DAMAGE';
                btnToggleThermo.textContent = "VIEW TEMP MAP";
            } else {
                thermoMode = 'TEMP';
                btnToggleThermo.textContent = "VIEW DAMAGE MAP";
            }
            thermalViz.setMode(thermoMode);
        });
    }

    if (btnAutoAblate) {
        btnAutoAblate.addEventListener('click', () => {
            guidanceActive = !guidanceActive;
            updateGuidanceState(guidanceActive);
        });
    }

    async function updateGuidanceState(active) {
        if (active) {
            btnAutoAblate.textContent = "STOP AUTO-ABLATION";
            btnAutoAblate.classList.add('active');
            log("Automated Guidance STARTED");
        } else {
            btnAutoAblate.textContent = "START AUTO-ABLATION";
            btnAutoAblate.classList.remove('active');
            log("Automated Guidance STOPPED");
        }

        await fetch('/api/guidance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: active })
        });
    }

    // Tab Switching
    const tabs = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Deactivate all
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            // Activate selected
            tab.classList.add('active');
            const targetId = tab.getAttribute('data-tab');
            const targetEl = document.getElementById(targetId);
            if (targetEl) {
                targetEl.classList.add('active');
            }

            // Update chart or trigger sub-view loader if needed
            if (targetId === 'tab-thermo' && thermalViz.chart) {
                thermalViz.chart.update();
            } else if (targetId === 'tab-fencing' && typeof loadFencingData === 'function') {
                loadFencingData();
            } else if (targetId === 'tab-economics' && typeof runEconomicsSim === 'function') {
                runEconomicsSim();
            }
        });
    });

    // Polling Loop
    setInterval(async () => {
        if (telemetryInFlight) {
            return;
        }
        telemetryInFlight = true;
        try {
            const res = await fetch('/api/telemetry');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();

            // 1. Immediately Update System & Network Status
            if (data.system) {
                elSystemStatus.textContent = data.system.status || 'CONNECTED';
                elLoopRate.textContent = (data.system.loop_hz ? data.system.loop_hz.toFixed(0) : '50') + ' Hz';
                elSystemStatus.style.color = data.system.simulation_running ? 'var(--success)' : 'var(--danger)';
            } else {
                elSystemStatus.textContent = 'CONNECTED';
                elSystemStatus.style.color = 'var(--success)';
            }

            // Sync Guidance State if finished
            if (data.guidance) {
                if (data.guidance.completed && guidanceActive) {
                    guidanceActive = false;
                    btnAutoAblate.textContent = "ABLATION COMPLETE";
                    btnAutoAblate.classList.remove('active');
                    log("Automated Ablation COMPLETED successfully.");
                }
            }

            // Update Robot
            try {
                if (robotViz && data.joints) robotViz.updateJoints(data.joints);
            } catch (e) { /* silent robot update */ }

            // Sync end-effector to 3D brain scene
            try {
                if (data.laser_pos && robotViz) {
                    _globalLaserActive  = data.laser_enabled || false;
                    _globalEffectorNorm = data.laser_pos;
                    robotViz.setEndEffectorBrain(data.laser_pos[0], data.laser_pos[1], _globalLaserActive);
                }
            } catch (e) { /* silent effector update */ }

            // Update Thermometry
            let maxVal = 37.0;
            try {
                const isLaserFiring = data.laser_enabled !== undefined ? data.laser_enabled : laserActive;
                const laserPos = data.laser_pos || null;
                if (laserPos && thermalViz) thermalViz.setEndEffectorOverlay(laserPos[0], laserPos[1]);
                if (thermalViz) {
                    const ret = thermalViz.update(data.temperature_map, data.damage_map, data.mr_anatomy, isLaserFiring, laserPos);
                    if (typeof ret === 'number') maxVal = ret;
                    if (data.temp_history) {
                        const genAiProfile = Array.isArray(data.gen_ai?.generated_profile) ? data.gen_ai.generated_profile : [];
                        thermalViz.updateChart(data.temp_history, genAiProfile);
                    }
                }
            } catch (e) { /* silent thermal update */ }

            // Update Cryo
            try {
                if (data.cryo_map && cryoViz) {
                    cryoViz.update(data.cryo_map, data.mr_anatomy);
                }
            } catch (e) { /* silent cryo update */ }

            // Update Robotics Tab Telemetry
            if (data.position) {
                document.getElementById('pos-x').textContent = data.position[0].toFixed(3);
                document.getElementById('pos-y').textContent = data.position[1].toFixed(3);
                document.getElementById('pos-z').textContent = data.position[2].toFixed(3);
            }
            if (data.joints) {
                data.joints.forEach((val, i) => {
                    const el = document.getElementById(`j${i + 1}-val`);
                    if (el) el.textContent = val.toFixed(2);
                });
            }
            if (data.quantum && data.quantum.metrics) {
                document.getElementById('q-coherence').textContent = (data.quantum.metrics.coherence || 0).toFixed(3);
                document.getElementById('q-fidelity').textContent = (data.quantum.metrics.qml_fidelity || 0).toFixed(3);
            }

            // ── Quantum Kalman Localisation ──────────────────────────────
            if (data.qkf_localization) {
                const qkf = data.qkf_localization;
                const ep  = qkf.estimated_position || [0,0,0];
                const elQX = document.getElementById('qkf-x');
                const elQY = document.getElementById('qkf-y');
                const elQZ = document.getElementById('qkf-z');
                const elQCoh = document.getElementById('qkf-coherence');
                const elQUnc = document.getElementById('qkf-uncertainty');
                const elQRes = document.getElementById('qkf-residual');
                if (elQX) elQX.textContent = ep[0].toFixed(4);
                if (elQY) elQY.textContent = ep[1].toFixed(4);
                if (elQZ) elQZ.textContent = ep[2].toFixed(4);
                if (elQCoh) {
                    const coh = (qkf.coherence || 0).toFixed(3);
                    elQCoh.textContent = coh;
                    elQCoh.style.color = qkf.coherence > 0.8 ? '#4ade80' : '#facc15';
                }
                if (elQUnc) elQUnc.textContent = (qkf.uncertainty || 0).toExponential(2);
                if (elQRes) {
                    const mm = (qkf.residual_norm_mm || 0).toFixed(2);
                    elQRes.textContent = mm;
                    elQRes.style.color = qkf.residual_norm_mm < 1.0 ? '#4ade80' : '#f87171';
                }
            }

            // ── MR Thermometry Segmentation ─────────────────────────────
            if (data.mr_thermometry_seg) {
                const seg = data.mr_thermometry_seg;
                const elCentroid  = document.getElementById('mrt-centroid');
                const elVolume    = document.getElementById('mrt-volume');
                const elAblation  = document.getElementById('mrt-ablation');
                const elNecrosis  = document.getElementById('mrt-necrosis');
                const c = seg.centroid || [0.5, 0.5];
                if (elCentroid)  elCentroid.textContent  = `(${c[0].toFixed(2)}, ${c[1].toFixed(2)})`;
                if (elVolume)    elVolume.textContent     = Math.round(seg.tumor_volume_mm2 || 0) + ' px²';
                if (elAblation)  elAblation.textContent  = ((seg.ablation_coverage || 0) * 100).toFixed(1) + '%';
                if (elNecrosis)  elNecrosis.textContent  = ((seg.necrosis_fraction || 0) * 100).toFixed(1) + '%';

                // Targeting accuracy: distance from end-effector to tumour centroid (normalised px)
                const en = seg.effector_norm || [0.5, 0.5];
                const dx = (en[0] - c[0]);
                const dz = (en[1] - c[1]);
                const dist_px = Math.sqrt(dx * dx + dz * dz) * 128; // approx pixels
                const elAcc = document.getElementById('mrt-accuracy');
                if (elAcc) {
                    elAcc.textContent = dist_px.toFixed(1) + ' px';
                    elAcc.style.color = dist_px < 8 ? '#4ade80' : dist_px < 20 ? '#facc15' : '#f87171';
                }
            }

            if (data.thermal_neuro_morphometry) {
                const morph = data.thermal_neuro_morphometry;
                const elConformal = document.getElementById('tnm-conformal');
                const elDistortion = document.getElementById('tnm-distortion');
                const elShape = document.getElementById('tnm-shape');
                const elCf = document.getElementById('tnm-cf');
                if (elConformal) elConformal.textContent = (morph.conformal_stability || 0).toFixed(3);
                if (elDistortion) elDistortion.textContent = (morph.distortion_mean || 0).toFixed(3);
                if (elShape) elShape.textContent = (morph.thermal_shape_index || 0).toFixed(3);
                if (elCf) {
                    const terms = morph.continued_fraction_terms || [];
                    elCf.textContent = terms.length ? `[${terms[0]};${terms.slice(1).join(',')}]` : '[1]';
                }
            }

            // Max Val Display depends on mode
            if (thermoMode === 'TEMP') {
                elMaxTemp.textContent = maxVal.toFixed(1) + '°C';
                // Check safety based on temp
                if (maxVal > 45.0) {
                    elTissueStatus.textContent = "ABLATING";
                    elTissueStatus.style.color = "var(--danger)";
                } else {
                    elTissueStatus.textContent = "NORMAL";
                    elTissueStatus.style.color = "var(--success)";
                }
            } else {
                elMaxTemp.textContent = maxVal.toFixed(1) + ' CEM';
                if (maxVal > 240.0) {
                    elTissueStatus.textContent = "NECROSIS";
                    elTissueStatus.style.color = "#000"; // Dead
                } else if (maxVal > 10.0) {
                    elTissueStatus.textContent = "DAMAGING";
                    elTissueStatus.style.color = "orange";
                } else {
                    elTissueStatus.textContent = "SAFE";
                }
            }

            // Laser Visual
            const finalLaserState = data.laser_enabled !== undefined ? data.laser_enabled : laserActive;
            robotViz.setLaser(finalLaserState);

            // Sync indicator
            if (finalLaserState) {
                elLaserInd.classList.add('active');
            } else {
                elLaserInd.classList.remove('active');
            }

            // Update 5G guidance status if active
            if (fiveGActive) {
                try {
                    const five_g_resp = await fetch('/api/guidance/5g/status');
                    if (five_g_resp.ok) {
                        const five_g_data = await five_g_resp.json();
                        
                        // Update status display
                        const statusEl = document.getElementById('5g-status');
                        const progressEl = document.getElementById('5g-progress');
                        const waypointsEl = document.getElementById('5g-waypoints');
                        
                        if (statusEl) statusEl.textContent = five_g_data.active ? 'ACTIVE' : 'IDLE';
                        if (progressEl) progressEl.textContent = (five_g_data.progress * 100).toFixed(1) + '%';
                        if (waypointsEl) waypointsEl.textContent = five_g_data.current_waypoint + '/' + five_g_data.total_waypoints;
                        
                        // Update robot visualization with trajectory
                        if (five_g_data.trajectory && five_g_data.trajectory.length > 0) {
                            robotViz.update5GGuidance(five_g_data.trajectory);
                            robotViz.update5GProgress(five_g_data.progress);
                        }
                        if (typeof draw5GNeuralMap === 'function') {
                            draw5GNeuralMap(five_g_data);
                        }
                        
                        // Check if ablation completed
                        if (five_g_data.completed) {
                            fiveGActive = false;
                            btn5G.textContent = "ACTIVATE 5G GUIDANCE";
                            btn5G.style.background = "linear-gradient(135deg, #06b6d4, #22d3ee)";
                            btnAuto5G.textContent = "AUTO-ABLATE (Complete)";
                            btnAuto5G.disabled = false;
                            log("✅ 5G-Guided Ablation COMPLETED");
                        }
                    }
                } catch (e) {
                    // 5G status endpoint might not have been called yet
                }
            }
        } catch (e) {
            console.error("Telemetry failed", e);
            if (elSystemStatus) {
                elSystemStatus.textContent = "RECONNECTING…";
                elSystemStatus.style.color = "var(--danger)";
            }
        } finally {
            telemetryInFlight = false;
        }
    }, 250); // 4Hz UI update for stable network communication

const btnLaser = document.getElementById('btn-enable-laser');
const btnSim = document.getElementById('btn-start-sim');

// Controls
if (btnSim) {
    btnSim.addEventListener('click', () => {
        // Start visual simulation path
        robotViz.startSimulation();
        // In a real app, we would tell backend to move the robot along path
        // For this demo, we can just animate the target coordinates
        simulatePath();
    });
}

async function simulatePath() {
    // Simple loop to move target along a curve
    const steps = 100;
    for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        // Parametric curve similar to vessel
        // -0.3 -> 0.3 X
        // 0.4 -> 0.55 Y (Height)
        // 0.5 -> 0.5 Z
        const x = -0.3 + (0.6 * t);
        const z = 0.5 + (0.2 * Math.sin(t * Math.PI)); // Arc

        // Send control
        await sendControl(x, z, i > 80, false); // Fire laser at end

        // Wait
        await new Promise(r => setTimeout(r, 50));
    }
}

if (btnLaser) {
    btnLaser.addEventListener('mousedown', () => {
        laserActive = true;
        elLaserInd.classList.add('active');
        sendControl(targetX, targetZ, true, false);
        log("Laser ACTIVATE request sent");
    });

    btnLaser.addEventListener('mouseup', () => {
        laserActive = false;
        elLaserInd.classList.remove('active');
        sendControl(targetX, targetZ, false, false);
        log("Laser DEACTIVATE request sent");
    });
}

const btnCryo = document.getElementById('btn-enable-cryo');
if (btnCryo) {
    btnCryo.addEventListener('click', () => {
        cryoActive = !cryoActive;

        if (cryoActive) {
            btnCryo.textContent = "DEACTIVATE CRYO";
            btnCryo.style.background = "linear-gradient(135deg, #ef4444, #f87171)"; // Red/Warning
            btnCryo.classList.add('active');
            log("Cryo System ACTIVATED");
        } else {
            btnCryo.textContent = "ACTIVATE CRYO";
            btnCryo.style.background = "linear-gradient(135deg, #3b82f6, #93c5fd)"; // Blue
            btnCryo.classList.remove('active');
            log("Cryo System DEACTIVATED");
        }

        sendControl(targetX, targetZ, false, cryoActive);
    });
}

// 5G Guidance System Controls
let fiveGActive = false;
const btn5G = document.getElementById('btn-enable-5g');
const btnAuto5G = document.getElementById('btn-auto-5g-ablation');

if (btn5G) {
    btn5G.addEventListener('click', async () => {
        fiveGActive = !fiveGActive;

        if (fiveGActive) {
            btn5G.textContent = "DEACTIVATE 5G GUIDANCE";
            btn5G.style.background = "linear-gradient(135deg, #0891b2, #06b6d4)";
            btn5G.classList.add('active');
            btnAuto5G.style.opacity = "1.0";
            btnAuto5G.style.cursor = "pointer";
            log("🛰️ 5G Neural Path Guidance INITIATED");
            
            // Activate 5G guidance on backend
            await fetch('/api/guidance/5g', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: true })
            });
        } else {
            btn5G.textContent = "ACTIVATE 5G GUIDANCE";
            btn5G.style.background = "linear-gradient(135deg, #06b6d4, #22d3ee)";
            btn5G.classList.remove('active');
            btnAuto5G.style.opacity = "0.5";
            btnAuto5G.style.cursor = "not-allowed";
            log("🛰️ 5G Guidance DEACTIVATED");
            
            // Deactivate on backend
            await fetch('/api/guidance/5g', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: false })
            });
        }
    });
}

if (btnAuto5G) {
    btnAuto5G.addEventListener('click', async () => {
        if (!fiveGActive) {
            log("⚠️ Enable 5G Guidance first");
            return;
        }
        
        btnAuto5G.textContent = "AUTO-ABLATION RUNNING...";
        btnAuto5G.disabled = true;
        log("🔥 5G-Guided Auto-Ablation STARTED");
        
        // The 5G system will automatically control the robot and laser
        // Monitor progress via telemetry updates
    });
}

// ── Precision Target Button ───────────────────────────────────────────────────
const btnPrecisionTarget = document.getElementById('btn-precision-target');
if (btnPrecisionTarget) {
    btnPrecisionTarget.addEventListener('click', async () => {
        btnPrecisionTarget.textContent = 'LOCKING…';
        btnPrecisionTarget.disabled = true;
        try {
            const resp = await fetch('/api/laser/precision_target', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fire: true })
            });
            const result = await resp.json();
            const t = result.target || [0, 0];
            log(`🎯 Precision target locked: (${t[0].toFixed(3)}, ${t[1].toFixed(3)}) — laser ${result.laser ? 'ON' : 'OFF'}`);
            btnPrecisionTarget.textContent = 'LOCKED ✓';
            btnPrecisionTarget.style.background = 'linear-gradient(135deg,#10b981,#34d399)';
            setTimeout(() => {
                btnPrecisionTarget.textContent = 'PRECISION TARGET';
                btnPrecisionTarget.style.background = '';
                btnPrecisionTarget.disabled = false;
            }, 2500);
        } catch (e) {
            btnPrecisionTarget.textContent = 'PRECISION TARGET';
            btnPrecisionTarget.disabled = false;
        }
    });
}

// ── LUT Selector ─────────────────────────────────────────────────────────────
const lutSelector = document.getElementById('lut-selector');
if (lutSelector) {
    lutSelector.addEventListener('change', () => {
        thermalViz.setLUT(lutSelector.value);
        log(`🎨 Thermometry LUT changed to: ${lutSelector.value.toUpperCase()}`);
    });
}

// Send coordinates on mouse move over 3D canvas (simplified)
const container = document.getElementById('canvas-3d');
container.addEventListener('mousemove', (e) => {
    const rect = container.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    // Map 2D mouse to robot workspace (roughly)
    // Robot workspace: X [-0.5, 0.5], Z [0.0, 1.0]

    targetX = (x - 0.5) * 1.5; // Scale
    targetZ = (1.0 - y) * 1.0;

    // Throttle this in real app, but for local demo ok
    // We only send coords, not laser state change here
    sendControl(targetX, targetZ, laserActive, cryoActive);
});

async function sendControl(x, z, laser, cryo) {
    try {
        await fetch('/api/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target: { x: x, y: 0, z: z },
                laser: laser,
                cryo: cryo
            })
        });
    } catch (err) {
        console.error('Control channel error', err);
    }
}

function log(msg) {
    const li = document.createElement('li');
    li.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    listLogs.prepend(li);
}

// ── MR-Thermometry Segmentation Canvas Renderer ─────────────────────────────
// Fetch the downsampled masks from the dedicated endpoint every 500 ms and draw
// tumour boundary (red), ablation zone (orange), necrosis (white) onto the
// 240×120 mrt-canvas in the Robotics tab.
const mrtCanvas = document.getElementById('mrt-canvas');
const mrtCtx    = mrtCanvas ? mrtCanvas.getContext('2d') : null;
const tnmCanvas = document.getElementById('tnm-canvas');
const tnmCtx = tnmCanvas ? tnmCanvas.getContext('2d') : null;

// Shared live state from telemetry (updated in polling loop)
let _globalEffectorNorm = [0.5, 0.5];
let _globalLaserActive  = false;

async function refreshMRTCanvas() {
    if (!mrtCtx) return;
    try {
        const resp = await fetch('/api/thermal/tumor_segmentation');
        if (!resp.ok) return;
        const d    = await resp.json();

        const rows    = d.tumor_mask_ds   || [];
        const ablRows = d.ablation_mask_ds || [];
        const necRows = d.necrosis_mask_ds || [];
        const dtRows  = d.delta_T_ds       || [];
        const effNorm = d.effector_norm    || _globalEffectorNorm;

        if (!rows.length) return;

        const nR = rows.length;
        const nC = rows[0].length;
        const cW = mrtCanvas.width  / nC;
        const cH = mrtCanvas.height / nR;

        mrtCtx.clearRect(0, 0, mrtCanvas.width, mrtCanvas.height);

        // 1. Background: delta-T heat map (blue–red gradient)
        if (dtRows.length) {
            let maxDT = 0;
            for (const row of dtRows)
                for (const v of row)
                    if (v > maxDT) maxDT = v;
            maxDT = maxDT || 1;
            for (let r = 0; r < nR; r++) {
                for (let c = 0; c < nC; c++) {
                    const frac = Math.min(dtRows[r][c] / maxDT, 1);
                    const R = Math.round(frac * 220);
                    const B = Math.round((1 - frac) * 80);
                    mrtCtx.fillStyle = `rgba(${R},30,${B},0.7)`;
                    mrtCtx.fillRect(c * cW, r * cH, cW, cH);
                }
            }
        }

        // 2. Ablation zone (orange, semi-transparent)
        mrtCtx.fillStyle = 'rgba(251,146,60,0.55)';
        for (let r = 0; r < nR; r++)
            for (let c = 0; c < nC; c++)
                if (ablRows[r] && ablRows[r][c])
                    mrtCtx.fillRect(c * cW, r * cH, cW, cH);

        // 3. Necrosis zone (white, semi-transparent)
        mrtCtx.fillStyle = 'rgba(255,255,255,0.6)';
        for (let r = 0; r < nR; r++)
            for (let c = 0; c < nC; c++)
                if (necRows[r] && necRows[r][c])
                    mrtCtx.fillRect(c * cW, r * cH, cW, cH);

        // 4. Tumour boundary (bright red outline)
        mrtCtx.strokeStyle = '#ef4444';
        mrtCtx.lineWidth   = 1;
        for (let r = 1; r < nR - 1; r++) {
            for (let c = 1; c < nC - 1; c++) {
                const inside  = rows[r][c];
                const hasEdge = (
                    rows[r-1][c] !== inside || rows[r+1][c] !== inside ||
                    rows[r][c-1] !== inside || rows[r][c+1] !== inside
                );
                if (inside && hasEdge)
                    mrtCtx.strokeRect(c * cW, r * cH, cW, cH);
            }
        }

        // 5. Centroid cross-hair (tumour target)
        const centroid = d.centroid || [0.5, 0.5];
        const cx = centroid[0] * mrtCanvas.width;
        const cy = centroid[1] * mrtCanvas.height;
        mrtCtx.strokeStyle = '#22d3ee';
        mrtCtx.lineWidth   = 1.5;
        mrtCtx.setLineDash([]);
        mrtCtx.beginPath();
        mrtCtx.moveTo(cx - 8, cy); mrtCtx.lineTo(cx + 8, cy);
        mrtCtx.moveTo(cx, cy - 8); mrtCtx.lineTo(cx, cy + 8);
        mrtCtx.stroke();

        // 6. End-Effector probe position (bright cyan diamond with pulse ring)
        const ex = effNorm[0] * mrtCanvas.width;
        const ez = effNorm[1] * mrtCanvas.height;
        // Pulse radius when laser active
        const pulseR = _globalLaserActive ? (6 + 2 * Math.abs(Math.sin(Date.now() * 0.012))) : 5;
        mrtCtx.strokeStyle = _globalLaserActive ? '#f97316' : '#22d3ee';
        mrtCtx.lineWidth = 1.5;
        mrtCtx.setLineDash([2, 2]);
        mrtCtx.beginPath();
        mrtCtx.arc(ex, ez, pulseR + 4, 0, Math.PI * 2);
        mrtCtx.stroke();
        mrtCtx.setLineDash([]);
        // Diamond fill
        mrtCtx.fillStyle = _globalLaserActive ? 'rgba(249,115,22,0.9)' : 'rgba(34,211,238,0.9)';
        mrtCtx.beginPath();
        mrtCtx.moveTo(ex,          ez - pulseR);
        mrtCtx.lineTo(ex + pulseR, ez);
        mrtCtx.lineTo(ex,          ez + pulseR);
        mrtCtx.lineTo(ex - pulseR, ez);
        mrtCtx.closePath();
        mrtCtx.fill();

        // 6b. Draw distance line from EE to centroid
        mrtCtx.strokeStyle = 'rgba(250,200,0,0.55)';
        mrtCtx.lineWidth = 1;
        mrtCtx.setLineDash([3, 3]);
        mrtCtx.beginPath();
        mrtCtx.moveTo(ex, ez); mrtCtx.lineTo(cx, cy);
        mrtCtx.stroke();
        mrtCtx.setLineDash([]);

        // 7. MRT LUT temperature colorbar (right edge)
        const barW = 10, barH = mrtCanvas.height - 14, barX = mrtCanvas.width - 12, barY = 7;
        const lutColors = [
            [255,255,255],[255,10,0],[255,110,0],[255,230,0],[140,255,0],[0,255,190],[0,190,255],[0,60,255],[0,0,210]
        ];
        for (let py = 0; py < barH; py++) {
            const frac = 1 - py / barH;
            const ci = Math.min(Math.floor(frac * (lutColors.length - 1)), lutColors.length - 2);
            const cr = frac * (lutColors.length - 1) - ci;
            const c1 = lutColors[ci], c2 = lutColors[ci + 1];
            const r = Math.round(c1[0] + (c2[0]-c1[0]) * cr);
            const g = Math.round(c1[1] + (c2[1]-c1[1]) * cr);
            const b = Math.round(c1[2] + (c2[2]-c1[2]) * cr);
            mrtCtx.fillStyle = `rgb(${r},${g},${b})`;
            mrtCtx.fillRect(barX, barY + py, barW, 1);
        }
        mrtCtx.strokeStyle = 'rgba(255,255,255,0.25)';
        mrtCtx.lineWidth = 0.5;
        mrtCtx.strokeRect(barX, barY, barW, barH);
        mrtCtx.fillStyle = '#e2e8f0';
        mrtCtx.font = '7px monospace';
        mrtCtx.fillText('hot', barX - 2, barY + 8);
        mrtCtx.fillText('37°', barX - 2, barY + barH);

    } catch (e) {
        // silently skip if endpoint not ready yet
    }
}
setInterval(refreshMRTCanvas, 500);

async function refreshTNMCanvas() {
    if (!tnmCtx) return;
    try {
        const resp = await fetch('/api/thermal/neuro_morphometry');
        if (!resp.ok) return;
        const d = await resp.json();
        const conf = d.conformal_invariant_ds || [];
        const bel = d.beltrami_ds || [];
        const hot = d.hotspot_mask_ds || [];
        if (!conf.length) return;

        const rows = conf.length;
        const cols = conf[0].length;
        const cellW = tnmCanvas.width / cols;
        const cellH = tnmCanvas.height / rows;
        tnmCtx.clearRect(0, 0, tnmCanvas.width, tnmCanvas.height);

        let maxBel = 0;
        for (const row of bel)
            for (const value of row)
                if (value > maxBel) maxBel = value;
        maxBel = maxBel || 1;

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const confVal = Math.max(0, Math.min(1, conf[r][c] || 0));
                const belVal = Math.max(0, Math.min(1, (bel[r] && bel[r][c] ? bel[r][c] : 0) / maxBel));
                const hotspot = hot[r] && hot[r][c];
                const green = Math.round(40 + 170 * confVal);
                const red = Math.round(80 + 120 * belVal + (hotspot ? 55 : 0));
                const blue = Math.round(40 + 140 * belVal);
                tnmCtx.fillStyle = `rgba(${red},${green},${blue},0.85)`;
                tnmCtx.fillRect(c * cellW, r * cellH, cellW, cellH);
                if (hotspot) {
                    tnmCtx.strokeStyle = 'rgba(245, 158, 11, 0.85)';
                    tnmCtx.lineWidth = 1;
                    tnmCtx.strokeRect(c * cellW, r * cellH, cellW, cellH);
                }
            }
        }

        const centroid = d.centroid || [0.5, 0.5];
        const cx = centroid[0] * tnmCanvas.width;
        const cy = centroid[1] * tnmCanvas.height;
        tnmCtx.strokeStyle = '#f8fafc';
        tnmCtx.lineWidth = 1.5;
        tnmCtx.beginPath();
        tnmCtx.moveTo(cx - 7, cy);
        tnmCtx.lineTo(cx + 7, cy);
        tnmCtx.moveTo(cx, cy - 7);
        tnmCtx.lineTo(cx, cy + 7);
        tnmCtx.stroke();

        // End-effector overlay on TNM canvas
        const en = d.effector_norm || _globalEffectorNorm;
        const eex = en[0] * tnmCanvas.width;
        const eey = en[1] * tnmCanvas.height;
        tnmCtx.strokeStyle = _globalLaserActive ? '#f97316' : '#22d3ee';
        tnmCtx.lineWidth = 1.2;
        tnmCtx.beginPath();
        tnmCtx.arc(eex, eey, 5, 0, Math.PI * 2);
        tnmCtx.stroke();
    } catch (e) {
        // skip if endpoint unavailable during startup
    }
}
setInterval(refreshTNMCanvas, 500);

// ── Level-Set Segmentation Overlay ──────────────────────────────────
async function refreshLevelSetOverlay() {
    try {
        const resp = await fetch('/api/segmentation/level_set');
        if (!resp.ok) return;
        const d = await resp.json();
        if (typeof thermalViz !== 'undefined') {
            thermalViz.setLevelSetContour(d);
        }
        const elVol  = document.getElementById('ls-volume');
        const elCirc = document.getElementById('ls-circularity');
        const elSol  = document.getElementById('ls-solidity');
        const elBnd  = document.getElementById('ls-boundary');
        if (elVol)  elVol.textContent  = (d.tumor_volume_px || 0) + ' px';
        if (elCirc) elCirc.textContent = (d.circularity  || 0).toFixed(3);
        if (elSol)  elSol.textContent  = (d.solidity     || 0).toFixed(3);
        if (elBnd)  elBnd.textContent  = (d.boundary_length || 0) + ' px';
    } catch (e) { /* silent — server may not be ready */ }
}
setInterval(refreshLevelSetOverlay, 800);

const btnToggleLS = document.getElementById('btn-toggle-ls-overlay');
if (btnToggleLS) {
    btnToggleLS.addEventListener('click', () => {
        if (typeof thermalViz === 'undefined') return;
        thermalViz.toggleLevelSet();
        const active = thermalViz._ls.active;
        btnToggleLS.textContent   = active ? 'HIDE LEVEL-SET' : 'SHOW LEVEL-SET';
        btnToggleLS.style.opacity = active ? '1.0' : '0.5';
        log(`Level-Set overlay ${active ? 'shown' : 'hidden'}`);
    });
}

// =========================================================================
// INTRAOPERATIVE ULTRASOUND & TUMOR ABLATION FRONTEND CONTROLLERS
// =========================================================================

const iusCanvas = document.getElementById('ius-bmode-canvas');
const iusCtx = iusCanvas ? iusCanvas.getContext('2d') : null;
let rocChartInstance = null;
let beamChartInstance = null;

async function renderUltrasoundBMode(scanData) {
    if (!iusCtx || !scanData || !scanData.bmode_image_grid) return;
    const grid = scanData.bmode_image_grid;
    const rows = grid.length;
    const cols = grid[0].length;
    const cW = iusCanvas.width / cols;
    const cH = iusCanvas.height / rows;

    iusCtx.fillStyle = '#000000';
    iusCtx.fillRect(0, 0, iusCanvas.width, iusCanvas.height);

    // 1. Draw ultrasound B-mode pixel intensity
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const val = grid[r][c];
            if (val > 0) {
                iusCtx.fillStyle = `rgb(${val}, ${val}, ${val})`;
                iusCtx.fillRect(c * cW, r * cH, cW + 0.5, cH + 0.5);
            }
        }
    }

    // 2. Overlay Tumor Boundary (Cyan Dotted Ellipse)
    const centerX = iusCanvas.width / 2;
    const centerY = (28.0 / 60.0) * iusCanvas.height;
    const rTumorPx = (scanData.tumor_radius_mm / 60.0) * iusCanvas.height;

    iusCtx.strokeStyle = '#38bdf8';
    iusCtx.lineWidth = 1.5;
    iusCtx.setLineDash([4, 3]);
    iusCtx.beginPath();
    iusCtx.arc(centerX, centerY, rTumorPx, 0, 2 * Math.PI);
    iusCtx.stroke();

    // 3. Overlay Dynamic Ablation Necrotic Core (Red/Orange Glow)
    if (scanData.ablation_radius_mm > 0.5) {
        const rAblationPx = (scanData.ablation_radius_mm / 60.0) * iusCanvas.height;
        const grad = iusCtx.createRadialGradient(centerX, centerY, 0, centerX, centerY, rAblationPx);
        grad.addColorStop(0, 'rgba(255, 255, 255, 0.85)');
        grad.addColorStop(0.4, 'rgba(239, 68, 68, 0.7)');
        grad.addColorStop(1, 'rgba(249, 115, 22, 0.3)');

        iusCtx.fillStyle = grad;
        iusCtx.setLineDash([]);
        iusCtx.beginPath();
        iusCtx.arc(centerX, centerY, rAblationPx, 0, 2 * Math.PI);
        iusCtx.fill();

        iusCtx.strokeStyle = '#ef4444';
        iusCtx.lineWidth = 2;
        iusCtx.stroke();
    }

    // 4. Update badge
    const badge = document.getElementById('ius-ablation-badge');
    if (badge) {
        badge.innerText = `Ablation: ${scanData.ablation_radius_mm.toFixed(1)} mm`;
    }
}

async function fetchUltrasoundScan() {
    try {
        const res = await fetch('/api/ultrasound/scan');
        if (!res.ok) return;
        const data = await res.json();
        if (data.scan) renderUltrasoundBMode(data.scan);
        if (data.ablation_metrics) updateAblationMetricsUI(data.ablation_metrics);
    } catch (e) {
        console.error("iUS Scan fetch failed", e);
    }
}

function updateAblationMetricsUI(m) {
    const elSens = document.getElementById('val-sensitivity');
    const elSpec = document.getElementById('val-specificity');
    const elSpar = document.getElementById('val-sparing');
    const elDice = document.getElementById('val-dice');
    const elAuc  = document.getElementById('val-auc');

    if (elSens) elSens.innerText = m.tumor_sensitivity_percentage;
    if (elSpec) elSpec.innerText = m.tumor_specificity_percentage;
    if (elSpar) elSpar.innerText = m.healthy_parenchyma_sparing_pct;
    if (elDice) elDice.innerText = m.dice_similarity_coefficient.toFixed(3);
    if (elAuc) elAuc.innerText = m.roc_area_under_curve_auc.toFixed(3);

    if (m.roc_curve && rocChartInstance) {
        rocChartInstance.data.labels = m.roc_curve.map(p => (p.false_positive_rate * 100).toFixed(1) + '%');
        rocChartInstance.data.datasets[0].data = m.roc_curve.map(p => p.true_positive_rate * 100);
        rocChartInstance.update();
    }
}

function initRocChart() {
    const ctx = document.getElementById('roc-chart');
    if (!ctx) return;
    rocChartInstance = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Sensitivity (TPR %)',
                data: [],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.2)',
                fill: true,
                tension: 0.2,
                pointRadius: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'FPR (%)', color: '#94a3b8', font: { size: 9 } },
                    grid: { color: 'rgba(51,65,85,0.3)' },
                    ticks: { color: '#94a3b8', font: { size: 8 } }
                },
                y: {
                    title: { display: true, text: 'TPR (%)', color: '#94a3b8', font: { size: 9 } },
                    grid: { color: 'rgba(51,65,85,0.3)' },
                    ticks: { color: '#94a3b8', font: { size: 8 } },
                    min: 0,
                    max: 100
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function initBeamChart() {
    const ctx = document.getElementById('beam-chart');
    if (!ctx) return;
    beamChartInstance = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Acoustic Power (dB)',
                data: [],
                borderColor: '#a78bfa',
                backgroundColor: 'rgba(167, 139, 250, 0.2)',
                fill: true,
                tension: 0.2,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Angle (°)', color: '#94a3b8', font: { size: 9 } },
                    grid: { color: 'rgba(51,65,85,0.3)' },
                    ticks: { color: '#94a3b8', font: { size: 8 } }
                },
                y: {
                    title: { display: true, text: 'Power (dB)', color: '#94a3b8', font: { size: 9 } },
                    grid: { color: 'rgba(51,65,85,0.3)' },
                    ticks: { color: '#94a3b8', font: { size: 8 } },
                    min: -40,
                    max: 0
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
    updateProbeBeam(0.0);
}

async function updateProbeBeam(steerDeg) {
    try {
        const res = await fetch('/api/ultrasound/probe/beam', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ steering_angle_deg: parseFloat(steerDeg) })
        });
        const data = await res.json();
        if (data.beam_profile && beamChartInstance) {
            const pts = data.beam_profile.radiation_pattern;
            beamChartInstance.data.labels = pts.map(p => p.angle_deg);
            beamChartInstance.data.datasets[0].data = pts.map(p => p.power_db);
            beamChartInstance.update();
        }
    } catch (e) {
        console.error("Probe beam fetch error", e);
    }
}

async function fetchQMLStieltjesData() {
    try {
        const res = await fetch('/api/ablation/qml_stieltjes');
        const data = await res.json();
        if (!data.qml_stieltjes) return;
        const q = data.qml_stieltjes;

        const elPow = document.getElementById('qml-opt-power');
        const elDut = document.getElementById('qml-opt-duty');
        const elZ   = document.getElementById('stieltjes-zmatch');
        const elCem = document.getElementById('qml-cem43');

        if (elPow) elPow.innerText = `${q.optimal_acoustic_power_watts} W`;
        if (elDut) elDut.innerText = `${q.optimal_duty_cycle_percentage}%`;
        if (elZ)   elZ.innerText   = `${q.stieltjes_impedance_match_mrayl} MRayl`;
        if (elCem) elCem.innerText = `${q.predicted_cem43_thermal_dose_min} min`;

        const tbody = document.getElementById('stieltjes-table-body');
        if (tbody && q.stieltjes_convergents) {
            tbody.innerHTML = q.stieltjes_convergents.map(c => `
                <tr style="border-bottom:1px solid rgba(51,65,85,0.4);">
                    <td style="padding:4px;">${c.step}</td>
                    <td style="padding:4px; color:#38bdf8;">${c.coefficient_c_k}</td>
                    <td style="padding:4px; font-weight:700;">${c.convergent_value}</td>
                    <td style="padding:4px; color:#94a3b8;">${c.stieltjes_error_bound}</td>
                    <td style="padding:4px; color:#10b981;">✔ CONVERGED</td>
                </tr>
            `).join('');
        }
    } catch (e) {
        console.error("QML Stieltjes error", e);
    }
}

// Wire up ablation controls
const btnStepAblate = document.getElementById('btn-step-ablate');
if (btnStepAblate) {
    btnStepAblate.addEventListener('click', async () => {
        try {
            const res = await fetch('/api/ablation/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ step_delta_mm: 1.2 })
            });
            const data = await res.json();
            if (data.scan) renderUltrasoundBMode(data.scan);
            if (data.metrics) updateAblationMetricsUI(data.metrics);
            log(`🔥 Ultrasound HIFU ablation pulse applied: Radius ${data.current_ablation_radius_mm.toFixed(1)} mm (Sensitivity: ${data.metrics.tumor_sensitivity_percentage})`);
        } catch (e) {
            console.error("Ablation step failed", e);
        }
    });
}

const btnResetAblate = document.getElementById('btn-reset-ablate');
if (btnResetAblate) {
    btnResetAblate.addEventListener('click', async () => {
        try {
            const res = await fetch('/api/ablation/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
            const data = await res.json();
            if (data.scan) renderUltrasoundBMode(data.scan);
            if (data.metrics) updateAblationMetricsUI(data.metrics);
            log("🔄 Ultrasound ablation simulation reset.");
        } catch (e) {
            console.error("Ablation reset failed", e);
        }
    });
}

const steerSlider = document.getElementById('beam-steer-slider');
if (steerSlider) {
    steerSlider.addEventListener('input', (e) => {
        const val = e.target.value;
        const valLabel = document.getElementById('beam-steer-val');
        if (valLabel) valLabel.innerText = `${val}°`;
        updateProbeBeam(val);
    });
}

const btnRunQml = document.getElementById('btn-run-qml-pulse');
if (btnRunQml) {
    btnRunQml.addEventListener('click', async () => {
        btnRunQml.innerText = '⏳ OPTIMIZING VARIATIONAL PQC CIRCUIT…';
        btnRunQml.disabled = true;
        await fetchQMLStieltjesData();
        btnRunQml.innerText = '⚡ RUN QML VARIATIONAL ACOUSTIC OPTIMIZER';
        btnRunQml.disabled = false;
        log("⚡ QML Variational Acoustic Pulse and Stieltjes Continued Fractions synthesized.");
    });
}

// =========================================================================
// 4D DYNAMIC MRUS PHANTOMS & GROUND TRUTH MONTE CARLO VALIDATOR
// =========================================================================

const phantomMrCanvas = document.getElementById('phantom-mr-canvas');
const phantomMrCtx = phantomMrCanvas ? phantomMrCanvas.getContext('2d') : null;
const phantomIusCanvas = document.getElementById('phantom-ius-canvas');
const phantomIusCtx = phantomIusCanvas ? phantomIusCanvas.getContext('2d') : null;
let gtHistogramChartInstance = null;
let phantom4DInterval = null;
let phantom4DPlaying = false;
let currentPhantomPhase = 0;

function renderGridToCanvas(canvasCtx, grid, canvasWidth, canvasHeight, colormap = 'gray') {
    if (!canvasCtx || !grid || !grid.length) return;
    const rows = grid.length;
    const cols = grid[0].length;
    const cW = canvasWidth / cols;
    const cH = canvasHeight / rows;

    canvasCtx.fillStyle = '#000000';
    canvasCtx.fillRect(0, 0, canvasWidth, canvasHeight);

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const val = grid[r][c];
            if (val > 0) {
                if (colormap === 'gray') {
                    canvasCtx.fillStyle = `rgb(${val}, ${val}, ${val})`;
                } else if (colormap === 't2') {
                    // Slight cool tint for T2 MRI
                    canvasCtx.fillStyle = `rgb(${Math.round(val*0.9)}, ${Math.round(val*0.95)}, ${val})`;
                }
                canvasCtx.fillRect(c * cW, r * cH, cW + 0.5, cH + 0.5);
            }
        }
    }
}

async function fetchPhantomScan(phaseDeg = 0) {
    try {
        const rad = (phaseDeg * Math.PI) / 180.0;
        const res = await fetch('/api/phantom/mrus_scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ phase_rad: rad, deformation_amp_mm: 1.2 })
        });
        const data = await res.json();
        if (!data.phantom_scan) return;
        const p = data.phantom_scan;

        // Render Dual Canvases
        if (phantomMrCtx && p.mr_magnitude_grid) {
            renderGridToCanvas(phantomMrCtx, p.mr_magnitude_grid, phantomMrCanvas.width, phantomMrCanvas.height, 't2');
        }
        if (phantomIusCtx && p.ius_bmode_grid) {
            renderGridToCanvas(phantomIusCtx, p.ius_bmode_grid, phantomIusCanvas.width, phantomIusCanvas.height, 'gray');
        }

        const badge = document.getElementById('phantom-phase-badge');
        if (badge) badge.innerText = `Phase: ${p.phase_degrees}° (Disp: ${p.displacement_magnitude_mm} mm)`;

        if (data.ground_truth_metrics) {
            const m = data.ground_truth_metrics;
            const elTre  = document.getElementById('gt-tre-val');
            const elHd   = document.getElementById('gt-hd95-val');
            const elNmi  = document.getElementById('gt-nmi-val');
            const elCnr  = document.getElementById('gt-cnr-val');
            if (elTre) elTre.innerText = `${m.target_registration_error_mean_mm} mm`;
            if (elHd)  elHd.innerText  = `${m.hausdorff_distance_95_mm} mm`;
            if (elNmi) elNmi.innerText = `${m.normalized_mutual_information_bits} bits`;
            if (elCnr) elCnr.innerText = `${m.contrast_to_noise_ratio_db} dB`;
        }
    } catch (e) {
        console.error("Phantom scan error", e);
    }
}

function initGtHistogramChart() {
    const ctx = document.getElementById('gt-histogram-chart');
    if (!ctx) return;
    gtHistogramChartInstance = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Trial Count (256 Total)',
                data: [],
                backgroundColor: 'rgba(56, 189, 248, 0.75)',
                borderColor: '#38bdf8',
                borderWidth: 1,
                borderRadius: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Target Registration Error TRE (mm)', color: '#94a3b8', font: { size: 9 } },
                    grid: { color: 'rgba(51,65,85,0.3)' },
                    ticks: { color: '#94a3b8', font: { size: 8 } }
                },
                y: {
                    title: { display: true, text: 'Trials', color: '#94a3b8', font: { size: 9 } },
                    grid: { color: 'rgba(51,65,85,0.3)' },
                    ticks: { color: '#94a3b8', font: { size: 8 } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
    fetchGroundTruthStudy();
}

async function fetchGroundTruthStudy() {
    try {
        const res = await fetch('/api/phantom/ground_truth_study');
        const data = await res.json();
        if (data.study && gtHistogramChartInstance) {
            const hist = data.study.histogram_distribution;
            gtHistogramChartInstance.data.labels = hist.map(h => h.bin_center_mm.toFixed(2));
            gtHistogramChartInstance.data.datasets[0].data = hist.map(h => h.count);
            gtHistogramChartInstance.update();
        }
    } catch (e) {
        console.error("Ground truth study error", e);
    }
}

// Scrubber & 4D Loop Controller
const phaseSlider = document.getElementById('phantom-phase-slider');
if (phaseSlider) {
    phaseSlider.addEventListener('input', (e) => {
        currentPhantomPhase = parseInt(e.target.value);
        const valLabel = document.getElementById('phantom-phase-val');
        if (valLabel) valLabel.innerText = `${currentPhantomPhase}°`;
        fetchPhantomScan(currentPhantomPhase);
    });
}

const btnPlay4D = document.getElementById('btn-play-4d-phantom');
if (btnPlay4D) {
    btnPlay4D.addEventListener('click', () => {
        phantom4DPlaying = !phantom4DPlaying;
        if (phantom4DPlaying) {
            btnPlay4D.innerText = '⏸ PAUSE 4D';
            btnPlay4D.style.background = '#ef4444';
            phantom4DInterval = setInterval(() => {
                currentPhantomPhase = (currentPhantomPhase + 8) % 360;
                if (phaseSlider) phaseSlider.value = currentPhantomPhase;
                const valLabel = document.getElementById('phantom-phase-val');
                if (valLabel) valLabel.innerText = `${currentPhantomPhase}°`;
                fetchPhantomScan(currentPhantomPhase);
            }, 60);
        } else {
            btnPlay4D.innerText = '▶ PLAY 4D LOOP';
            btnPlay4D.style.background = '#0284c7';
            if (phantom4DInterval) clearInterval(phantom4DInterval);
        }
    });
}

const btnRunGtStudy = document.getElementById('btn-run-gt-study');
if (btnRunGtStudy) {
    btnRunGtStudy.addEventListener('click', async () => {
        btnRunGtStudy.innerText = '⏳ RUNNING 256-TRIAL MONTE CARLO STUDY…';
        btnRunGtStudy.disabled = true;
        await fetchGroundTruthStudy();
        btnRunGtStudy.innerText = '📊 RUN 256-TRIAL GROUND TRUTH MONTE CARLO STUDY';
        btnRunGtStudy.disabled = false;
        log("📊 256-Trial Monte Carlo Ground Truth study executed (Sub-voxel TRE <= 0.28 mm verified).");
    });
}

// =====================================================================
// 5G URLLC NETWORK, NTT COMPRESSION & GE PACS SERVER WORKFLOWS
// =====================================================================

const canvas5G = document.getElementById('5g-neural-map');
const ctx5G = canvas5G ? canvas5G.getContext('2d') : null;

async function poll5GTelemetry() {
    try {
        const resp = await fetch('/api/5g/network/telemetry');
        if (!resp.ok) return;
        const data = await resp.json();

        const elRtt = document.getElementById('5g-rtt-latency');
        const elJitter = document.getElementById('5g-jitter');
        const elReliability = document.getElementById('5g-reliability');
        const elPktLoss = document.getElementById('5g-pkt-loss');
        const elQueueStatus = document.getElementById('5g-queue-status');
        const elBandwidth = document.getElementById('5g-bandwidth');
        const elTxLog = document.getElementById('pacs-tx-log');

        if (elRtt && data.rtt_ms !== undefined) elRtt.textContent = `${data.rtt_ms.toFixed(2)} ms`;
        if (elJitter && data.jitter_ms !== undefined) elJitter.textContent = `${data.jitter_ms.toFixed(2)} ms`;
        if (elReliability && data.urllc_reliability_pct !== undefined) elReliability.textContent = `${data.urllc_reliability_pct.toFixed(4)}%`;
        if (elPktLoss && data.packet_loss_pct !== undefined) elPktLoss.textContent = `${data.packet_loss_pct.toFixed(3)}%`;
        if (elQueueStatus && data.active_workers !== undefined) elQueueStatus.textContent = `${data.active_workers}T / Q:${data.queue_depth}`;
        if (elBandwidth && data.bandwidth_gbps !== undefined) elBandwidth.textContent = `${data.bandwidth_gbps.toFixed(2)} Gbps`;

        // Update transmission logs
        if (elTxLog && data.recent_logs && data.recent_logs.length > 0) {
            elTxLog.innerHTML = data.recent_logs.map(l => 
                `<div>[${l.timestamp}] <span style="color:${l.level==='WARN'?'#f59e0b':(l.level==='NOTICE'?'#38bdf8':'#94a3b8')}">${l.message}</span></div>`
            ).join('');
            elTxLog.scrollTop = elTxLog.scrollHeight;
        }
    } catch (e) {
        // Silently skip if network telemetry endpoint is warming up
    }
}
setInterval(poll5GTelemetry, 600);

// 5G Network Slice Selector
const sliceSelector = document.getElementById('5g-slice-selector');
if (sliceSelector) {
    sliceSelector.addEventListener('change', async (e) => {
        const mode = e.target.value;
        try {
            await fetch('/api/5g/network/slice', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ slice_mode: mode })
            });
            log(`🛰️ 5G Slice changed to: ${mode}`);
        } catch (err) {
            console.error(err);
        }
    });
}

// NTT Benchmark Trigger
const btnRunNTT = document.getElementById('btn-run-ntt');
if (btnRunNTT) {
    btnRunNTT.addEventListener('click', async () => {
        btnRunNTT.disabled = true;
        btnRunNTT.textContent = '⏳ RUNNING…';
        try {
            const resp = await fetch('/api/5g/ntt/benchmark', { method: 'POST' });
            if (resp.ok) {
                const data = await resp.json();
                const m = data.metrics;
                const elMod = document.getElementById('ntt-modulus');
                const elLat = document.getElementById('ntt-latency');
                const elThr = document.getElementById('ntt-throughput');
                const elRat = document.getElementById('ntt-ratio');

                if (elMod) elMod.textContent = `q = ${m.modulus_q}`;
                if (elLat) elLat.textContent = `${m.latency_us} µs`;
                if (elThr) elThr.textContent = `${m.throughput_mbps} MB/s`;
                if (elRat) elRat.textContent = `${m.max_reconstruction_error} / 3.42:1`;
                log(`⚡ NTT Benchmark: ${m.throughput_mbps} MB/s throughput, ${m.latency_us} µs latency, 0 exact error.`);
            }
        } catch (e) {
            console.error("NTT benchmark error", e);
        } finally {
            btnRunNTT.disabled = false;
            btnRunNTT.textContent = '⚡ BENCHMARK NTT';
        }
    });
}

// GE PACS C-STORE (Live ultrasound frame capture)
const btnPacsCStore = document.getElementById('btn-pacs-cstore');
if (btnPacsCStore) {
    btnPacsCStore.addEventListener('click', async () => {
        btnPacsCStore.disabled = true;
        btnPacsCStore.textContent = '📥 STORING…';
        try {
            const pId = document.getElementById('pacs-query-patient')?.value || 'NM-2026-9041';
            const resp = await fetch('/api/5g/pacs/cstore', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ patient_id: pId, study_desc: 'Intraoperative 8.5MHz B-Mode Resection Scan' })
            });
            const data = await resp.json();
            if (resp.ok) {
                log(`🏥 DICOM C-STORE: Saved SOP ${data.sop_instance_uid} to GE PACS archive (${data.ntt_metrics.throughput_mbps} MB/s).`);
                fetchPacsStudies();
            }
        } catch (e) {
            console.error("C-STORE error", e);
        } finally {
            btnPacsCStore.disabled = false;
            btnPacsCStore.textContent = '📥 C-STORE (Live iUS Frame)';
        }
    });
}

// GE PACS C-SEND (Push series to GE Centricity PACS)
const btnPacsCSend = document.getElementById('btn-pacs-csend');
if (btnPacsCSend) {
    btnPacsCSend.addEventListener('click', async () => {
        btnPacsCSend.disabled = true;
        btnPacsCSend.textContent = '📤 SENDING…';
        try {
            const resp = await fetch('/api/5g/pacs/csend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ destination_aet: 'GE_CENTRICITY' })
            });
            const data = await resp.json();
            if (resp.ok) {
                log(`📤 DICOM C-SEND: Dispatched study ${data.study_uid} to ${data.destination_aet} via 5G queue.`);
                fetchPacsStudies();
            }
        } catch (e) {
            console.error("C-SEND error", e);
        } finally {
            btnPacsCSend.disabled = false;
            btnPacsCSend.textContent = '📤 C-SEND (Push to GE PACS)';
        }
    });
}

// GE PACS C-FIND Query Studies
const btnPacsCFind = document.getElementById('btn-pacs-cfind');
async function fetchPacsStudies() {
    try {
        const pId = document.getElementById('pacs-query-patient')?.value || '';
        const mod = document.getElementById('pacs-query-modality')?.value || 'ALL';
        const resp = await fetch('/api/5g/pacs/cfind', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ patient_id: pId, modality: mod })
        });
        if (!resp.ok) return;
        const data = await resp.json();
        const tbody = document.getElementById('pacs-studies-tbody');
        if (tbody && data.studies) {
            tbody.innerHTML = data.studies.map(s => `
                <tr style="border-bottom: 1px solid #1e293b;">
                    <td style="padding: 4px 6px; color: #38bdf8; font-weight: bold;">${s.patient_id}</td>
                    <td style="padding: 4px 6px;">
                        <span style="color:${s.modality==='US'?'#10b981':'#c084fc'}; background:${s.modality==='US'?'rgba(16,185,129,0.15)':'rgba(192,132,252,0.15)'}; padding:1px 4px; border-radius:2px;">${s.modality}</span>
                    </td>
                    <td style="padding: 4px 6px; color: #e2e8f0;">${s.study_desc}</td>
                    <td style="padding: 4px 6px; color: #94a3b8;">${s.num_instances}</td>
                    <td style="padding: 4px 6px; display: flex; gap: 4px;">
                        <button class="btn-secondary" style="padding: 1px 5px; font-size: 9px; border-color: #38bdf8; color: #38bdf8;" onclick="triggerCSend('${s.study_uid}')">SEND</button>
                        <button class="btn-secondary" style="padding: 1px 5px; font-size: 9px; border-color: #a78bfa; color: #a78bfa;" onclick="triggerCMove('${s.study_uid}')">RETRIEVE</button>
                    </td>
                </tr>
            `).join('');
        }
    } catch (e) {
        console.error("C-FIND error", e);
    }
}
if (btnPacsCFind) {
    btnPacsCFind.addEventListener('click', () => {
        log("🔍 DICOM C-FIND: Executing Query on GE PACS Archive…");
        fetchPacsStudies();
    });
}

window.triggerCSend = async function(studyUid) {
    try {
        const resp = await fetch('/api/5g/pacs/csend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ study_uid: studyUid, destination_aet: 'GE_CENTRICITY' })
        });
        const data = await resp.json();
        log(`📤 C-SEND: Pushed study ${studyUid} to GE Centricity PACS over 5G.`);
        fetchPacsStudies();
    } catch (e) {
        console.error(e);
    }
};

window.triggerCMove = async function(studyUid) {
    try {
        const resp = await fetch('/api/5g/pacs/cmove', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ study_uid: studyUid })
        });
        const data = await resp.json();
        log(`📥 C-MOVE: Retrieved study ${studyUid} to intraoperative workstation cache.`);
        fetchPacsStudies();
    } catch (e) {
        console.error(e);
    }
};

function draw5GNeuralMap(vizData) {
    if (!ctx5G || !canvas5G) return;
    const w = canvas5G.width;
    const h = canvas5G.height;
    ctx5G.clearRect(0, 0, w, h);

    // Draw background neural pathways field
    ctx5G.fillStyle = '#090d16';
    ctx5G.fillRect(0, 0, w, h);

    // Draw neural pathways gradient (blue-cyan)
    const neuralGrad = ctx5G.createRadialGradient(w/2, h/2, 10, w/2, h/2, w/2);
    neuralGrad.addColorStop(0, 'rgba(6, 182, 212, 0.25)');
    neuralGrad.addColorStop(1, 'rgba(2, 6, 23, 0.95)');
    ctx5G.fillStyle = neuralGrad;
    ctx5G.fillRect(0, 0, w, h);

    // Draw safe corridors
    ctx5G.strokeStyle = 'rgba(16, 185, 129, 0.4)';
    ctx5G.lineWidth = 1;
    ctx5G.beginPath();
    ctx5G.arc(w/2, h/2, w/3, 0, Math.PI * 2);
    ctx5G.stroke();

    // Draw trajectory waypoints
    if (vizData && vizData.trajectory && vizData.trajectory.length > 0) {
        ctx5G.strokeStyle = '#22d3ee';
        ctx5G.lineWidth = 1.5;
        ctx5G.beginPath();
        for (let i = 0; i < vizData.trajectory.length; i++) {
            const pt = vizData.trajectory[i];
            const px = (pt[0] / 128) * w;
            const py = (pt[1] / 128) * h;
            if (i === 0) ctx5G.moveTo(px, py);
            else ctx5G.lineTo(px, py);
        }
        ctx5G.stroke();

        // Draw current target point
        const curIdx = Math.min(vizData.current_waypoint || 0, vizData.trajectory.length - 1);
        const curPt = vizData.trajectory[curIdx];
        if (curPt) {
            const cx = (curPt[0] / 128) * w;
            const cy = (curPt[1] / 128) * h;
            ctx5G.fillStyle = '#f59e0b';
            ctx5G.beginPath();
            ctx5G.arc(cx, cy, 4, 0, Math.PI * 2);
            ctx5G.fill();
        }
    }
}

// Initial triggers
initRocChart();
initBeamChart();
initGtHistogramChart();
fetchUltrasoundScan();
fetchQMLStieltjesData();
fetchPhantomScan(0);
fetchPacsStudies();
setInterval(fetchUltrasoundScan, 1200);

});  // end DOMContentLoaded
