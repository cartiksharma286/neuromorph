document.addEventListener('DOMContentLoaded', () => {
    // Init Visualizations
    const robotViz = new RobotViz('canvas-3d');
    const thermalViz = new ThermalViz('thermal-canvas');
    const cryoViz = new CryoViz('cryo-canvas');

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
            document.getElementById(targetId).classList.add('active');

            // Update chart if needed
            if (targetId === 'tab-thermo' && thermalViz.chart) {
                thermalViz.chart.update();
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
            const data = await res.json();

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
            robotViz.updateJoints(data.joints);

            // Update Thermometry
            // Pass maps, anatomy, laser state and position
            const isLaserFiring = data.laser_enabled !== undefined ? data.laser_enabled : laserActive;
            const laserPos = data.laser_pos || null;
            const maxVal = thermalViz.update(data.temperature_map, data.damage_map, data.mr_anatomy, isLaserFiring, laserPos);

            if (data.temp_history) {
                const genAiProfile = (data.gen_ai && data.gen_ai.generated_profile) ? data.gen_ai.generated_profile : [];
                thermalViz.updateChart(data.temp_history, genAiProfile);
            }

            // Update Cryo
            if (data.cryo_map) {
                cryoViz.update(data.cryo_map, data.mr_anatomy);
            }

            // Update UI
            if (data.system) {
                elSystemStatus.textContent = data.system.status;
                elLoopRate.textContent = data.system.loop_hz.toFixed(0) + ' Hz';
                elSystemStatus.style.color = data.system.simulation_running ? 'var(--success)' : 'var(--danger)';
            }

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
});
