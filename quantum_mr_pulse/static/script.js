// State
let currentPulseData = null;
let currentParams = {};

// UI Elements
const btnGenerate = document.getElementById('btn-generate');
const btnAnalyze = document.getElementById('btn-analyze');
const btnGeodesics = document.getElementById('btn-geodesics');
const btnReconstruct = document.getElementById('btn-reconstruct');
const btnVerify = document.getElementById('btn-verify');
const btnExport = document.getElementById('btn-export');
const consoleOutput = document.getElementById('console-output');

// Input Listeners
const inputs = ['te', 'tr', 'fa'];
inputs.forEach(id => {
    const el = document.getElementById(id);
    const display = document.getElementById(`val-${id}`);
    el.addEventListener('input', (e) => {
        let suffix = id === 'fa' ? '°' : ' ms';
        display.innerText = e.target.value + suffix;
    });
});

// Logger
function log(msg, type = 'info') {
    const div = document.createElement('div');
    div.classList.add('log-line');
    if (type) div.classList.add(type);
    div.innerText = `> ${msg}`;
    consoleOutput.appendChild(div);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

// Generate Sequence
btnGenerate.addEventListener('click', async () => {
    const params = {
        sequence_type: document.getElementById('seqType').value,
        te: parseFloat(document.getElementById('te').value),
        tr: parseFloat(document.getElementById('tr').value),
        flip_angle: parseFloat(document.getElementById('fa').value),
        fov: 200,
        matrix_size: 128,
        optimize: document.getElementById('optimize').checked
    };

    log(`Generating ${params.sequence_type} sequence...`);

    try {
        const res = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });

        currentPulseData = await res.json();
        currentParams = params;

        log('Sequence generated successfully.', 'success');

        renderPulsePlot(currentPulseData);

        btnAnalyze.disabled = false;
        btnGeodesics.disabled = false;
        btnReconstruct.disabled = false;
        btnVerify.disabled = false;
        btnExport.disabled = false;

        // Reset metrics
        document.getElementById('metrics-panel').classList.add('hidden');

    } catch (e) {
        log('Error generating sequence: ' + e, 'error');
    }
});

// Analyze
btnAnalyze.addEventListener('click', async () => {
    if (!currentPulseData) return;

    log('Running Quantum Surface Integral analysis...', 'system');

    try {
        const res = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentPulseData)
        });

        const results = await res.json();

        document.getElementById('metrics-panel').classList.remove('hidden');
        document.getElementById('metric-berry').innerText = results.berry_phase.toFixed(4) + ' rad';
        document.getElementById('metric-coherence').innerText = results.coherence_metric.toFixed(4);

        log(`Surface Integral: ${results.surface_integral.toFixed(2)}`, 'success');

        renderBlochSphere(results.trajectory);

    } catch (e) {
        log('Analysis failed: ' + e, 'error');
    }
});

// Verify
btnVerify.addEventListener('click', async () => {
    if (!currentPulseData) return;

    log('Initiating NVQLink Formal Verification...', 'verify');

    try {
        const res = await fetch('/api/verify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pulse_data: currentPulseData,
                params: currentParams
            })
        });

        const result = await res.json();

        if (result.verified) {
            log(`NVQLink: VERIFIED [Hash: ${result.metrics.formal_proof_hash}]`, 'success');
            document.getElementById('ver-status').innerText = 'PASS';
            document.getElementById('ver-status').className = 'ver-status PASS';
        } else {
            log('NVQLink: VERIFICATION FAILED', 'error');
            result.violations.forEach(v => log('!! ' + v, 'error'));
            document.getElementById('ver-status').innerText = 'FAIL';
            document.getElementById('ver-status').className = 'ver-status FAIL';
        }

    } catch (e) {
        log('Verification failed: ' + e, 'error');
    }
});

// Geodesics
btnGeodesics.addEventListener('click', async () => {
    if (!currentPulseData) return;

    log('Mapping Prime Geodesics on Bloch Manifold...', 'system');

    try {
        const res = await fetch('/api/analyze_geodesics', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentPulseData)
        });

        const results = await res.json();

        document.getElementById('metrics-panel').classList.remove('hidden');
        document.getElementById('row-prime').classList.remove('hidden');

        const projections = results.projection_states.length;
        document.getElementById('metric-prime').innerText = `${projections} States`;

        log(`Geodesics Mapped. Projection States found: ${projections}`, 'success');

        renderBlochSphere(results.geodesic_trajectory);

    } catch (e) {
        log('Geodesic analysis failed: ' + e, 'error');
    }
});

// Export
btnExport.addEventListener('click', async () => {
    if (!currentPulseData) return;

    log('Exporting to Pulseq codec...', 'system');

    try {
        const res = await fetch('/api/export_seq', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentPulseData)
        });

        const result = await res.json();

        const blob = new Blob([result.content], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = result.filename;
        a.click();
        window.URL.revokeObjectURL(url);

        log(`Sequence exported: ${result.filename}`, 'success');

    } catch (e) {
        log('Export failed: ' + e, 'error');
    }
});

// Reconstruct
btnReconstruct.addEventListener('click', async () => {
    if (!currentPulseData) return;

    log('Simulating K-Space Acquisition & FFT Reconstruction...', 'system');

    try {
        const res = await fetch('/api/reconstruct', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentPulseData)
        });

        const result = await res.json();

        log('Reconstruction Complete.', 'success');

        renderImage(result.image);

    } catch (e) {
        log('Reconstruction failed: ' + e, 'error');
    }
});

// Visualization Helpers
function renderPulsePlot(data) {
    const traces = [
        { x: data.time, y: data.rf, name: 'RF (B1)', type: 'scatter', line: { color: '#00f2ea' } },
        { x: data.time, y: data.gx, name: 'Gx (Read)', type: 'scatter', line: { color: '#7000ff' } },
        { x: data.time, y: data.gy, name: 'Gy (Phase)', type: 'scatter', line: { color: '#ff0055' } },
        { x: data.time, y: data.gz, name: 'Gz (Slice)', type: 'scatter', line: { color: '#ffee00' } }
    ];

    const layout = {
        title: 'Pulse Sequence Diagram',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e0e0e0' },
        xaxis: { title: 'Time (ms)', gridcolor: '#333' },
        yaxis: { title: 'Amplitude (a.u.)', gridcolor: '#333' },
        margin: { t: 40, b: 40, l: 40, r: 20 },
        showlegend: true,
        legend: { orientation: 'h', y: 1.1 }
    };

    Plotly.newPlot('pulse-plot', traces, layout);
}

function renderBlochSphere(traj) {
    const trace = {
        x: traj.x,
        y: traj.y,
        z: traj.z,
        type: 'scatter3d',
        mode: 'lines',
        line: { color: '#00f2ea', width: 4 }
    };

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        showlegend: false,
        margin: { t: 0, b: 0, l: 0, r: 0 },
        scene: {
            xaxis: { showgrid: false, zeroline: false, showticklabels: false },
            yaxis: { showgrid: false, zeroline: false, showticklabels: false },
            zaxis: { showgrid: false, zeroline: false, showticklabels: false }
        }
    };

    Plotly.newPlot('bloch-viz', [trace], layout);
}

function renderImage(imageData) {
    const trace = {
        z: imageData,
        type: 'heatmap',
        colorscale: 'Gray',
        showscale: false
    };

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 0, b: 0, l: 0, r: 0 },
        xaxis: { showticklabels: false, ticks: '' },
        yaxis: { showticklabels: false, ticks: '' }
    };

    Plotly.newPlot('image-viz', [trace], layout);
}
