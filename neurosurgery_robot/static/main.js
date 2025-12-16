document.addEventListener('DOMContentLoaded', () => {
    // Init Visualizations
    const robotViz = new RobotViz('canvas-3d');
    const thermalViz = new ThermalViz('thermal-canvas');
    const cryoViz = new CryoViz('cryo-canvas');

    // UI Elements
    const elLinkStatus = document.getElementById('link-status');
    const elLinkLatency = document.getElementById('link-latency');
    const elLaserInd = document.getElementById('laser-indicator');
    const elMaxTemp = document.getElementById('max-temp');
    const elTissueStatus = document.getElementById('tissue-status');
    const listLogs = document.getElementById('log-list');


    let laserActive = false;
    let cryoActive = false;
    let targetX = 0.5;
    let targetZ = 0.5;

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
        try {
            const res = await fetch('/api/telemetry');
            const data = await res.json();

            // Update Robot
            robotViz.updateJoints(data.joints);

            // Update Thermometry
            const maxT = thermalViz.update(data.temperature_map);
            if (data.temp_history) {
                thermalViz.updateChart(data.temp_history);
            }

            // Update Cryo
            if (data.cryo_map) {
                cryoViz.update(data.cryo_map, data.mr_anatomy);
            }

            // Update UI
            if (data.nvqlink) {
                elLinkStatus.textContent = data.nvqlink.status;
                elLinkLatency.textContent = data.nvqlink.latency.toFixed(1) + ' ms';
                if (data.nvqlink.active) {
                    elLinkStatus.style.color = 'var(--success)';
                }
            }

            elMaxTemp.textContent = maxT.toFixed(1) + '°C';
            if (maxT > 45.0) {
                elTissueStatus.textContent = "ABLATING";
                elTissueStatus.style.color = "var(--danger)";
            } else {
                elTissueStatus.textContent = "NORMAL";
                elTissueStatus.style.color = "var(--success)";
            }

            // Laser Visual
            robotViz.setLaser(laserActive);

        } catch (e) {
            console.error("Telemetry failed", e);
        }
    }, 100); // 10Hz UI update

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
        await fetch('/api/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target: { x: x, y: 0, z: z },
                laser: laser,
                cryo: cryo
            })
        });
    }

    function log(msg) {
        const li = document.createElement('li');
        li.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        listLogs.prepend(li);
    }
});
