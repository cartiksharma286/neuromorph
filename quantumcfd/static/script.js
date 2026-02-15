document.getElementById('simForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const btn = document.getElementById('simBtn');
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const flowContainer = document.getElementById('flowContainer');
    const forcesContainer = document.getElementById('forcesContainer');

    // Reset UI
    btn.disabled = true;
    btn.querySelector('.btn-text').textContent = "Running Simulation...";
    statusDot.className = 'status-indicator active';
    statusText.textContent = "Solving Quantum Hyper-Fluid Equations...";
    flowContainer.innerHTML = '<div class="placeholder">Computing...</div>';
    forcesContainer.innerHTML = '<div class="placeholder">Computing...</div>';

    // Output dir is simulation_results in backend
    // but the backend returns base_path, which we can't access directly via browser if outside static
    // Wait, typical pattern: access via /results/{sim_id}/filename if we serve it
    // But the current server.py only has GET /results/{sim_id} which returns a list of files.
    // It doesn't serve the files themselves!
    // I need to update server.py to serve the files inside simulation_results directory.

    const config = {
        naca_code: document.getElementById('naca').value,
        angle_of_attack: parseFloat(document.getElementById('aoa').value),
        reynolds: parseFloat(document.getElementById('reynolds').value),
        grid_size: parseInt(document.getElementById('grid').value),
        steps: parseInt(document.getElementById('steps').value),
        forcing: document.getElementById('forcing').checked
    };

    try {
        // Start Sim
        const res = await fetch('/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        if (!res.ok) throw new Error("Simulation start failed");

        const data = await res.json();
        const simId = data.simulation_id;

        // Poll for completion
        // Poll for completion and real-time status
        const pollInterval = setInterval(async () => {
            try {
                const statusRes = await fetch(`/status/${simId}`);
                if (!statusRes.ok) return;

                const statusData = await statusRes.json();

                if (statusData.status === "running") {
                    // Update Text
                    statusText.textContent = `Computing Step ${statusData.current_step} / ${statusData.total_steps} (${statusData.progress}%)`;

                    // Show latest frame
                    const timestamp = new Date().getTime();
                    flowContainer.innerHTML = `<img src="/simulation_results/${simId}/latest.png?t=${timestamp}" alt="Real-time Flow">`;

                } else if (statusData.status === "complete") {
                    clearInterval(pollInterval);
                    finishSimulation(simId);
                }
            } catch (e) {
                console.error("Polling error", e);
            }
        }, 500); // 500ms polling for real-time feel

    } catch (err) {
        console.error(err);
        statusText.textContent = "Error: " + err.message;
        statusDot.className = 'status-indicator';
        statusDot.style.background = 'red';
        btn.disabled = false;
        btn.querySelector('.btn-text').textContent = "Initialize Simulation";
    }

    function finishSimulation(simId) {
        statusDot.className = 'status-indicator ready';
        statusText.textContent = "Simulation Complete";

        // Need an endpoint to serve the image files!
        // Assuming I've added a mount for /simulation_results in server.py

        const timestamp = new Date().getTime(); // Bust cache
        flowContainer.innerHTML = `<img src="/simulation_results/${simId}/flow.gif?t=${timestamp}" alt="Flow Animation">`;
        forcesContainer.innerHTML = `<img src="/simulation_results/${simId}/forces.png?t=${timestamp}" alt="Forces Graph">`;

        btn.disabled = false;
        btn.querySelector('.btn-text').textContent = "Initialize Simulation";
    }
});
