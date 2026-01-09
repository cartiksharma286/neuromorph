document.addEventListener('DOMContentLoaded', () => {
    const runFeaBtn = document.getElementById('runFeaBtn');
    const optimizeFeaBtn = document.getElementById('optimizeFeaBtn');
    const statusDiv = document.getElementById('feaStats');

    // Tab Handling hook if needed (assuming app.js does it generically)
    // If not, we might need to add it here, but let's assume app.js is well written.

    if (runFeaBtn) {
        runFeaBtn.addEventListener('click', async () => {
            runFeaBtn.textContent = "Simulating...";
            runFeaBtn.disabled = true;

            try {
                const vol = parseFloat(document.getElementById('voltageC1').value);
                const response = await fetch('/api/fea/simulate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        electrode_config: { voltages: { 'c1': vol } }
                    })
                });

                const data = await response.json();
                if (data.success) {
                    document.getElementById('imgPotential').src = "data:image/png;base64," + data.plots.potential;
                    document.getElementById('imgEField').src = "data:image/png;base64," + data.plots.e_field;
                    document.getElementById('imgVTA').src = "data:image/png;base64," + data.plots.vta;

                    statusDiv.innerHTML = `<span style="color: #4cd964">✓ Simulation Complete. Max Field: ${data.max_field.toFixed(2)} V/m</span>`;
                } else {
                    statusDiv.innerHTML = `<span style="color: #ff3b30">Error: ${data.error}</span>`;
                }
            } catch (e) {
                statusDiv.textContent = "Network Error";
            } finally {
                runFeaBtn.textContent = "Run Simulation";
                runFeaBtn.disabled = false;
            }
        });
    }

    if (optimizeFeaBtn) {
        optimizeFeaBtn.addEventListener('click', async () => {
            optimizeFeaBtn.textContent = "Optimizing...";
            optimizeFeaBtn.disabled = true;

            try {
                const tx = parseInt(document.getElementById('targetX').value);
                const ty = parseInt(document.getElementById('targetY').value);

                const response = await fetch('/api/fea/optimize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        target_coords: [tx, ty]
                    })
                });

                const data = await response.json();
                if (data.success) {
                    let log = "<b>Optimization Results:</b><br>";
                    data.optimization_log.forEach(item => {
                        log += `${item.contact}: Score ${item.score.toFixed(2)} (Cov: ${(item.coverage * 100).toFixed(1)}%)<br>`;
                    });
                    log += `<br><b style="color: #007aff">Recommended: ${Object.keys(data.recommended_config.voltages)[0]} at ${Object.values(data.recommended_config.voltages)[0]}V</b>`;
                    statusDiv.innerHTML = log;
                }
            } catch (e) {
                statusDiv.textContent = "Error optimizing";
            } finally {
                optimizeFeaBtn.textContent = "Auto-Optimize Platform";
                optimizeFeaBtn.disabled = false;
            }
        });
    }
});
