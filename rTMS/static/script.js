document.addEventListener('DOMContentLoaded', () => {
    
    // UI Elements
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabTitle = document.getElementById('tab-title');
    const tabSubtitle = document.getElementById('tab-subtitle');
    const runBtn = document.getElementById('run-simulation-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    const finalFreq = document.getElementById('final-freq');
    const finalIntensity = document.getElementById('final-intensity');
    const finalFitness = document.getElementById('final-fitness');
    
    // State
    let currentCondition = 'stroke';
    
    // Default Plotly layout styling for Dark Mode
    const plotlyLayoutDark = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#8b949e', family: 'Inter' },
        margin: { l: 40, r: 20, t: 20, b: 40 },
        xaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
        yaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
        scene: {
            xaxis: { gridcolor: 'rgba(255,255,255,0.1)', backgroundcolor: 'rgba(0,0,0,0)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.1)', backgroundcolor: 'rgba(0,0,0,0)' },
            zaxis: { gridcolor: 'rgba(255,255,255,0.1)', backgroundcolor: 'rgba(0,0,0,0)' },
            bgcolor: 'rgba(0,0,0,0)'
        }
    };

    // Tab Navigation Logic
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active state
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update semantic headers
            currentCondition = btn.getAttribute('data-tab');
            if(currentCondition === 'stroke') {
                tabTitle.textContent = 'Stroke Rehabilitation Optimization';
                tabSubtitle.textContent = 'Dynamic rTMS Parameter Optimizaton via FEA/BEM for Motor Cortex';
            } else {
                tabTitle.textContent = 'Dementia Cognitive Enhancement';
                tabSubtitle.textContent = 'Deep rTMS Parameter Optimizaton via FEA/BEM for Prefrontal Cortex';
            }
            
            // Clear or reset charts
            resetUI();
        });
    });

    // Run Simulation Logic
    runBtn.addEventListener('click', async () => {
        loadingOverlay.classList.remove('hidden');
        runBtn.disabled = true;
        runBtn.style.opacity = '0.5';
        
        try {
            const res = await fetch('/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ condition: currentCondition })
            });
            
            const payload = await res.json();
            
            // Artificial delay to mimic heavy computation and "wow" the user
            setTimeout(() => {
                loadingOverlay.classList.add('hidden');
                runBtn.disabled = false;
                runBtn.style.opacity = '1';
                
                if (payload.status === 'success') {
                    renderDashboard(payload.data);
                } else {
                    alert("Simulation failed on cloud.");
                }
            }, 1200);
            
        } catch (error) {
            console.error(error);
            loadingOverlay.classList.add('hidden');
            runBtn.disabled = false;
            runBtn.style.opacity = '1';
            alert("Connection error with Local Google Cloud Simulation Bridge.");
        }
    });

    function resetUI() {
        finalFreq.textContent = '--';
        finalIntensity.textContent = '--';
        finalFitness.textContent = '--';
        
        Plotly.purge('optimization-line-chart');
        Plotly.purge('fea-heatmap');
        Plotly.purge('bem-scatter');
    }

    function renderDashboard(data) {
        // 1. Update Parameters Pane
        const finalParams = data.optimization.final_parameters;
        finalFreq.textContent = finalParams.frequency_hz.toFixed(1);
        finalIntensity.textContent = finalParams.intensity_mso.toFixed(1);
        finalFitness.textContent = (finalParams.fitness * 100).toFixed(1) + "%";

        // 2. Plot Optimization Trajectory
        const traj = data.optimization.convergence_trajectory;
        const iterations = traj.map(t => t.iteration);
        const freqs = traj.map(t => t.frequency_hz);
        
        const traceOpt = {
            x: iterations,
            y: freqs,
            type: 'scatter',
            mode: 'lines+markers',
            line: {color: '#58a6ff', width: 3},
            marker: {size: 6, color: '#8a2be2'}
        };
        
        Plotly.newPlot('optimization-line-chart', [traceOpt], {
            ...plotlyLayoutDark,
            title: {text: 'Convergence of Protocol Synthesis', font: {color: '#e6edf3'}},
            yaxis: {title: 'Frequency (Hz)'}
        }, {responsive: true});

        // 3. Plot FEA Heatmap (2D)
        const gridData = data.fea_grid;
        
        const traceFEA = {
            z: gridData,
            type: 'heatmap',
            colorscale: 'Viridis'
        };

        Plotly.newPlot('fea-heatmap', [traceFEA], {
            ...plotlyLayoutDark,
            margin: {l: 40, r: 40, t: 20, b: 40}
        }, {responsive: true});

        // 4. Plot BEM Surface (3D Scatter/Mesh)
        const bemData = data.bem_mesh;
        const xList = bemData.map(v => v.x);
        const yList = bemData.map(v => v.y);
        const zList = bemData.map(v => v.z);
        const cList = bemData.map(v => v.c);

        const traceBEM = {
            x: xList,
            y: yList,
            z: zList,
            mode: 'markers',
            marker: {
                size: 5,
                color: cList,
                colorscale: 'Inferno',
                opacity: 0.8,
                showscale: true,
                colorbar: {
                    title: 'Magnetic Stress Strain',
                    titlefont: { color: '#e6edf3' },
                    tickfont: { color: '#e6edf3' },
                    thickness: 20
                }
            },
            type: 'scatter3d'
        };

        Plotly.newPlot('bem-scatter', [traceBEM], {
            ...plotlyLayoutDark,
            margin: {l: 0, r: 0, t: 0, b:0}
        }, {responsive: true});
    }

});
