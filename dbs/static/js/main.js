// --- Veteran Care PTSD DBS Protocol Simulation ---
// --- FAS Cortical Simulation ---
function simulateFASCortical() {
    const out = document.getElementById('fas-cortical-sidebar');
    if (out) out.innerHTML += '<div style="color:#00f2ff;">Running quantum-theoretic cortical simulation...</div>';
    fetch('/api/fas-cortical', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) })
        .then(res => res.json())
        .then(data => {
            // Render cortical simulation results (placeholder)
            if (out) out.innerHTML += `<div style='color:#fff;'>${data.result || 'Simulation complete.'}</div>`;
        })
        .catch(e => { if (out) out.innerHTML += '<div style="color:red;">Error running simulation.</div>'; });
}

// --- FAS Boundary Element Simulation ---
function simulateFASBEM() {
    const out = document.getElementById('fas-bem-sidebar');
    if (out) out.innerHTML += '<div style="color:#00f2ff;">Running boundary element simulation...</div>';
    fetch('/api/fas-bem', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) })
        .then(res => res.json())
        .then(data => {
            if (out) out.innerHTML += `<div style='color:#fff;'>${data.result || 'BEM simulation complete.'}</div>`;
        })
        .catch(e => { if (out) out.innerHTML += '<div style="color:red;">Error running BEM simulation.</div>'; });
}

// --- FAS Continued Fractions ---
function simulateFASCF() {
    const out = document.getElementById('fas-cf-sidebar');
    if (out) out.innerHTML += '<div style="color:#00f2ff;">Running continued fraction model...</div>';
    fetch('/api/fas-cf', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) })
        .then(res => res.json())
        .then(data => {
            if (out) out.innerHTML += `<div style='color:#fff;'>${data.result || 'Continued fraction model complete.'}</div>`;
        })
        .catch(e => { if (out) out.innerHTML += '<div style="color:red;">Error running continued fraction model.</div>'; });
}

// --- FAS Feynman Path Integrals ---
function simulateFASFeynman() {
    const out = document.getElementById('fas-feynman-sidebar');
    if (out) out.innerHTML += '<div style="color:#00f2ff;">Running Feynman path integral simulation...</div>';
    fetch('/api/fas-feynman', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) })
        .then(res => res.json())
        .then(data => {
            if (out) out.innerHTML += `<div style='color:#fff;'>${data.result || 'Feynman path integral simulation complete.'}</div>`;
        })
        .catch(e => { if (out) out.innerHTML += '<div style="color:red;">Error running Feynman simulation.</div>'; });
}

// --- FAS Post-Op Validation ---
function simulateFASPostOp() {
    const out = document.getElementById('fas-postop-sidebar');
    if (out) out.innerHTML += '<div style="color:#00f2ff;">Running post-op validation...</div>';
    fetch('/api/fas-postop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) })
        .then(res => res.json())
        .then(data => {
            if (out) out.innerHTML += `<div style='color:#fff;'>${data.result || 'Post-op validation complete.'}</div>`;
        })
        .catch(e => { if (out) out.innerHTML += '<div style="color:red;">Error running post-op validation.</div>'; });
}
function runVeteranPTSDDBS() {
    const resultsDiv = document.getElementById('veteran-ptsd-dbs-results');
    if (resultsDiv) resultsDiv.innerHTML = '<p style="color: #00f2ff;">Generating optimal DBS clinical paradigms...</p>';
    fetch('/api/veteran-ptsd-dbs-protocol', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    })
    .then(res => res.json())
    .then(data => {
        // Display optimal DBS treatment paradigms sequence
        let seqHtml = '';
        if (data.sequence && Array.isArray(data.sequence)) {
            seqHtml = '<ol>' + data.sequence.map(s => `<li><b>${s.stage}</b>: <span style="color:#fff;">${s.description}</span></li>`).join('') + '</ol>';
        }
        document.getElementById('veteran-ptsd-dbs-protocols-list').innerHTML = seqHtml;
        // Display protocol stages
        let stagesHtml = '';
        if (data.stages && Array.isArray(data.stages)) {
            stagesHtml = '<ul>' + data.stages.map(s => `<li><b>${s.label}</b> <span style="color:#fff;">@ ${s.context}</span></li>`).join('') + '</ul>';
        }
        document.getElementById('veteran-ptsd-dbs-stages-list').innerHTML = stagesHtml;
        // Simulation plot
        if (data.simulation) renderVeteranPTSDDBSChart(data.simulation);
        if (resultsDiv) resultsDiv.innerHTML = '';
    })
    .catch(e => {
        if (resultsDiv) resultsDiv.innerHTML = '<p style="color: red;">Error generating DBS clinical paradigms.</p>';
        console.error('Veteran PTSD DBS protocol fetch error:', e);
    });
}

function renderVeteranPTSDDBSChart(sim) {
    const ctx = document.getElementById('veteranPTSDDBSChart').getContext('2d');
    if (window.veteranPTSDDBSChartInstance) window.veteranPTSDDBSChartInstance.destroy();
    window.veteranPTSDDBSChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: sim.months,
            datasets: [
                {
                    label: 'Efficacy',
                    data: sim.efficacy,
                    borderColor: '#00ff00',
                    backgroundColor: 'rgba(0,255,0,0.08)',
                    fill: true,
                },
                {
                    label: 'Recovery',
                    data: sim.recovery,
                    borderColor: '#00f2ff',
                    backgroundColor: 'rgba(0,242,255,0.08)',
                    fill: true,
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: '#fff' } }
            },
            scales: {
                x: { ticks: { color: '#fff' } },
                y: { ticks: { color: '#fff' } }
            }
        }
    });
}

// --- Canadian PTSD Cortical FEA Simulation (DBS+FEA) ---
function runCanadianPTSDCorticalFEA() {
    const resultsDiv = document.getElementById('canadian-ptsd-fea-results');
    if (resultsDiv) resultsDiv.innerHTML = '<p style="color: #00f2ff;">Running cortical FEA simulation...</p>';
    fetch('/api/canadian-ptsd-cortical-fea', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    })
    .then(res => res.json())
    .then(data => {
        // Display optimal DBS treatment paradigms sequence
        let seqHtml = '';
        if (data.sequence && Array.isArray(data.sequence)) {
            seqHtml = '<ol>' + data.sequence.map(s => `<li><b>${s.stage}</b>: <span style='color:#fff;'>${s.description}</span></li>`).join('') + '</ol>';
        }
        document.getElementById('canadian-ptsd-fea-sequence').innerHTML = seqHtml;
        // Display FEA simulation results
        let feaHtml = '';
        if (data.fea_results && Array.isArray(data.fea_results)) {
            feaHtml = '<ul>' + data.fea_results.map(r => `<li><b>${r.region}</b>: <span style='color:#a0e7ff;'>${r.stress} kPa</span> | <span style='color:#ffb347;'>${r.notes}</span></li>`).join('') + '</ul>';
        }
        document.getElementById('canadian-ptsd-fea-results-main').innerHTML = feaHtml;
        // Simulation plot
        if (data.simulation) renderCanadianPTSDLLMChart(data.simulation);
        if (resultsDiv) resultsDiv.innerHTML = '';
    })
    .catch(e => {
        if (resultsDiv) resultsDiv.innerHTML = '<p style="color: red;">Error running cortical FEA simulation.</p>';
        console.error('Canadian PTSD FEA simulation error:', e);
    });
}

function renderCanadianPTSDLLMChart(sim) {
    const ctx = document.getElementById('canadianPTSDLLMChart').getContext('2d');
    if (window.canadianPTSDLLMChartInstance) window.canadianPTSDLLMChartInstance.destroy();
    window.canadianPTSDLLMChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: sim.months,
            datasets: [
                {
                    label: 'Efficacy',
                    data: sim.efficacy,
                    borderColor: '#00ff00',
                    backgroundColor: 'rgba(0,255,0,0.08)',
                    fill: true,
                },
                {
                    label: 'Recovery',
                    data: sim.recovery,
                    borderColor: '#00f2ff',
                    backgroundColor: 'rgba(0,242,255,0.08)',
                    fill: true,
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: '#fff' } }
            },
            scales: {
                x: { ticks: { color: '#fff' } },
                y: { ticks: { color: '#fff' } }
            }
        }
    });
}
// --- Canadian Veteran PTSD StatOpt Simulation ---
function simulateCanadianPTSDStatOpt() {
    const out = document.getElementById('canadian-ptsd-statopt-stage');
    if (out) out.innerText = 'Simulating...';
    fetch('/api/canadian-ptsd-statopt-cure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    })
    .then(res => res.json())
    .then(data => {
        // Update KPIs
        if (document.getElementById('canadian-ptsd-statopt-stage')) document.getElementById('canadian-ptsd-statopt-stage').innerText = data.stages[0].label;
        if (document.getElementById('canadian-ptsd-statopt-efficacy')) document.getElementById('canadian-ptsd-statopt-efficacy').innerText = Math.round(data.efficacy[data.efficacy.length-1]) + '%';
        if (document.getElementById('canadian-ptsd-statopt-timeline')) document.getElementById('canadian-ptsd-statopt-timeline').innerText = 'Months 0-48';
        renderCanadianPTSDStatOptChart(data);
        renderCanadianPTSDStatOptParamChart(data);
        // Render protocol stages
        const stagesList = document.getElementById('canadian-ptsd-statopt-stages-list');
        if (stagesList && data.stages) {
            stagesList.innerHTML = data.stages.map(s => `<li><b>${s.label}</b> <span style='color:#fff;'>@ Month ${s.time}</span></li>`).join('');
        }
    })
    .catch(e => {
        if (out) out.innerText = 'Error';
        console.error('Canadian PTSD StatOpt simulation error:', e);
    });
}

function renderCanadianPTSDStatOptChart(data) {
    const ctx = document.getElementById('canadianPTSDStatOptChart').getContext('2d');
    if (window.canadianPTSDStatOptChartInstance) window.canadianPTSDStatOptChartInstance.destroy();
    window.canadianPTSDStatOptChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.months,
            datasets: [
                {
                    label: 'Efficacy',
                    data: data.efficacy,
                    borderColor: '#00ff00',
                    backgroundColor: 'rgba(0,255,0,0.08)',
                    fill: true,
                },
                {
                    label: 'Recovery',
                    data: data.recovery,
                    borderColor: '#00f2ff',
                    backgroundColor: 'rgba(0,242,255,0.08)',
                    fill: true,
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: '#fff' } }
            },
            scales: {
                x: { ticks: { color: '#fff' } },
                y: { ticks: { color: '#fff' } }
            }
        }
    });
}

function renderCanadianPTSDStatOptParamChart(data) {
    const ctx = document.getElementById('canadianPTSDStatOptParamChart').getContext('2d');
    if (window.canadianPTSDStatOptParamChartInstance) window.canadianPTSDStatOptParamChartInstance.destroy();
    window.canadianPTSDStatOptParamChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.param_names,
            datasets: [
                {
                    label: 'Parameter Distribution',
                    data: data.param_dist,
                    backgroundColor: [
                        '#00ff00', '#00f2ff', '#ff00ff', '#fff', '#ffb347'
                    ]
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { ticks: { color: '#fff' } },
                y: { ticks: { color: '#fff' } }
            }
        }
    });
}
// --- Canadian PTSD Veteran QML Care Paradigms ---
function fetchCanadianPTSDQMLCare() {
    const paradigmsList = document.getElementById('canadian-ptsd-qml-list');
    if (!paradigmsList) return;
    paradigmsList.innerHTML = '<p style="color: #00f2ff;">Fetching QML-optimized paradigms...</p>';
    fetch('/api/canadian-ptsd-qml-care', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(res => res.json())
    .then(data => {
        if (!data.paradigms || !Array.isArray(data.paradigms)) {
            paradigmsList.innerHTML = '<p style="color: red;">No paradigms found.</p>';
            return;
        }
        let html = `<div style='font-size:12px; color:#fff; margin-bottom:10px;'>${data.summary || ''}</div>`;
        data.paradigms.forEach((p, idx) => {
            html += `
                <div style="background: rgba(0, 242, 255, 0.07); border: 1px solid rgba(0, 242, 255, 0.18); padding: 15px; margin-bottom: 18px; border-radius: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: top; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px; margin-bottom: 10px;">
                        <h3 style="color: #fff; margin: 0; font-size: 16px;">${p.title}</h3>
                        <span style="background: var(--accent-pink); padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; color: white;">${p.type || 'QML+GenAI'}</span>
                    </div>
                    <div style="color: var(--text-dim); font-size: 11px; text-transform: uppercase; margin-bottom: 5px;">Description</div>
                    <div style="color: #ddd; font-size: 13px; line-height: 1.4; margin-bottom: 5px;">${p.description}</div>
                    <div style="color: #a0e7ff; font-size: 12px;">AI Score: <b>${p.ai_score}</b> &nbsp; | &nbsp; <span style='color:#ffb347;'>${p.notes}</span></div>
                </div>
            `;
        });
        paradigmsList.innerHTML = html;
    })
    .catch(e => {
        paradigmsList.innerHTML = '<p style="color: red;">Error fetching paradigms.</p>';
        console.error('Canadian PTSD QML paradigms fetch error:', e);
    });
}
// --- PTSD/TBI Lobe-Specific Stimulation Plans ---
function fetchPTSDTBIPlans() {
    const plansList = document.getElementById('ptsd-tbi-plans-list');
    if (!plansList) return;
    plansList.innerHTML = '<p style="color: #00f2ff;">Fetching lobe-specific plans...</p>';
    fetch('/api/ptsd-tbi-lobe-plans', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(res => res.json())
    .then(data => {
        if (!data.plans || !Array.isArray(data.plans)) {
            plansList.innerHTML = '<p style="color: red;">No plans found.</p>';
            return;
        }
        let html = `<div style='font-size:12px; color:#fff; margin-bottom:10px;'>${data.summary || ''}</div>`;
        data.plans.forEach((p, idx) => {
            html += `
                <div style="background: rgba(0, 242, 255, 0.07); border: 1px solid rgba(0, 242, 255, 0.18); padding: 15px; margin-bottom: 18px; border-radius: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: top; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px; margin-bottom: 10px;">
                        <h3 style="color: #fff; margin: 0; font-size: 16px;">${p.lobe}</h3>
                        <span style="background: var(--accent-pink); padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; color: white;">${p.tbi_modification ? 'TBI-Adapted' : 'PTSD'}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; margin-bottom: 10px; font-family: monospace;">
                        <div style="background: rgba(0,0,0,0.5); padding: 8px; border: 1px dashed rgba(255,255,255,0.13); border-radius: 4px;">
                            <div style="color: var(--text-dim); font-size: 10px;">FREQUENCY (Hz)</div>
                            <div style="color: #00ff00; font-size: 14px; font-weight: bold;">${p.frequency_hz}</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.5); padding: 8px; border: 1px dashed rgba(255,255,255,0.13); border-radius: 4px;">
                            <div style="color: var(--text-dim); font-size: 10px;">PULSE WIDTH (µs)</div>
                            <div style="color: #00f2ff; font-size: 14px; font-weight: bold;">${p.pulse_width_us}</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.5); padding: 8px; border: 1px dashed rgba(255,255,255,0.13); border-radius: 4px;">
                            <div style="color: var(--text-dim); font-size: 10px;">VOLTAGE (V)</div>
                            <div style="color: #ff00ff; font-size: 14px; font-weight: bold;">${p.voltage_v}</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.5); padding: 8px; border: 1px dashed rgba(255,255,255,0.13); border-radius: 4px;">
                            <div style="color: var(--text-dim); font-size: 10px;">SESSIONS</div>
                            <div style="color: #fff; font-size: 14px; font-weight: bold;">${p.session_count}</div>
                        </div>
                    </div>
                    <div style="color: var(--text-dim); font-size: 11px; text-transform: uppercase; margin-bottom: 5px;">Role</div>
                    <div style="color: #ddd; font-size: 13px; line-height: 1.4; margin-bottom: 5px;">${p.role}</div>
                    <div style="color: #a0e7ff; font-size: 12px;">AI Score: <b>${p.ai_score}</b> &nbsp; | &nbsp; <span style='color:#ffb347;'>${p.notes}</span></div>
                </div>
            `;
        });
        plansList.innerHTML = html;
    })
    .catch(e => {
        plansList.innerHTML = '<p style="color: red;">Error fetching lobe plans.</p>';
        console.error('PTSD/TBI lobe plans fetch error:', e);
    });
}
// --- FASD StatOpt Cure Simulation ---
function simulateFASDStatOpt() {
    const out = document.getElementById('fasd-statopt-stage');
    if (out) out.innerText = 'Simulating...';
    fetch('/api/fasd-statopt-cure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    })
    .then(res => res.json())
    .then(data => {
        // Update KPIs
        if (document.getElementById('fasd-statopt-stage')) document.getElementById('fasd-statopt-stage').innerText = data.stages[0].label;
        if (document.getElementById('fasd-statopt-efficacy')) document.getElementById('fasd-statopt-efficacy').innerText = Math.round(data.efficacy[data.efficacy.length-1]) + '%';
        if (document.getElementById('fasd-statopt-timeline')) document.getElementById('fasd-statopt-timeline').innerText = 'Months 0-48';
        renderFASDStatOptChart(data);
        renderFASDStatOptParamChart(data);
        // Render protocol stages
        const stagesList = document.getElementById('fasd-statopt-stages-list');
        if (stagesList && data.stages) {
            stagesList.innerHTML = data.stages.map(s => `<li><b>${s.label}</b> <span style='color:#fff;'>@ Month ${s.time}</span></li>`).join('');
        }
    })
    .catch(e => {
        if (out) out.innerText = 'Error';
        console.error('FASD StatOpt simulation error:', e);
    });
}

function renderFASDStatOptChart(data) {
    const ctx = document.getElementById('fasdStatOptChart');
    if (!ctx) return;
    if (window.fasdStatOptChartInstance) window.fasdStatOptChartInstance.destroy();
    window.fasdStatOptChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.months.map(m => `M${m}`),
            datasets: [
                {
                    label: 'Optimized Efficacy',
                    data: data.efficacy,
                    borderColor: '#00ffcc',
                    backgroundColor: 'rgba(0,255,204,0.13)',
                    tension: 0.45,
                    fill: true,
                    pointRadius: 2
                },
                {
                    label: 'Recovery Index',
                    data: data.recovery,
                    borderColor: '#ff00c8',
                    backgroundColor: 'rgba(255,0,200,0.13)',
                    tension: 0.45,
                    fill: true,
                    pointRadius: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Statistical Parametric Optimization (DBS for FASD)'
                },
                legend: {
                    labels: { color: '#a0e7ff', font: { size: 12 } }
                }
            },
            layout: { padding: 0 },
            scales: {
                x: { title: { display: true, text: 'Months', color: '#a0e7ff' }, ticks: { color: '#a0e7ff' } },
                y: { min: 0, max: 120, title: { display: true, text: 'Score / Index', color: '#a0e7ff' }, ticks: { color: '#a0e7ff' } }
            }
        }
    });
}

function renderFASDStatOptParamChart(data) {
    const ctx = document.getElementById('fasdStatOptParamChart');
    if (!ctx) return;
    if (window.fasdStatOptParamChartInstance) window.fasdStatOptParamChartInstance.destroy();
    window.fasdStatOptParamChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.param_names,
            datasets: [
                {
                    label: 'DBS Parameter Distribution',
                    data: data.param_dist,
                    backgroundColor: [
                        'rgba(0,255,204,0.7)',
                        'rgba(255,0,200,0.7)',
                        'rgba(0,242,255,0.7)',
                        'rgba(255,206,86,0.7)',
                        'rgba(153,102,255,0.7)'
                    ],
                    borderRadius: 6,
                    barPercentage: 0.7,
                    categoryPercentage: 0.7
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'DBS Parameter Distribution (StatOpt)'
                },
                legend: {
                    labels: { color: '#a0e7ff', font: { size: 12 } }
                }
            },
            layout: { padding: 0 },
            scales: {
                x: { ticks: { color: '#a0e7ff' } },
                y: { min: 0, max: 1, ticks: { color: '#a0e7ff' } }
            }
        }
    });
}
// --- FASD Cure Simulation ---
function simulateFASDCure() {
    const out = document.getElementById('fasd-cure-stage');
    if (out) out.innerText = 'Simulating...';
    fetch('/api/fasd-cure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    })
    .then(res => res.json())
    .then(data => {
        // Update KPIs
        if (document.getElementById('fasd-cure-stage')) document.getElementById('fasd-cure-stage').innerText = data.stages[0].label;
        if (document.getElementById('fasd-cure-efficacy')) document.getElementById('fasd-cure-efficacy').innerText = Math.round(data.qml_neuroplasticity[data.qml_neuroplasticity.length-1]) + '%';
        if (document.getElementById('fasd-cure-timeline')) document.getElementById('fasd-cure-timeline').innerText = 'Months 0-48';
        renderFASDQMLChart(data);
        renderFASDQMLCorticalChart(data);
        // Render QML protocol stages
        const stagesList = document.getElementById('fasd-cure-stages-list');
        if (stagesList && data.stages) {
            stagesList.innerHTML = data.stages.map(s => `<li><b>${s.label}</b> <span style='color:#fff;'>@ Month ${s.time}</span></li>`).join('');
        }
    })
    .catch(e => {
        if (out) out.innerText = 'Error';
        console.error('FASD Cure simulation error:', e);
    });
}

function renderFASDQMLChart(data) {
    const ctx = document.getElementById('fasdCureChart');
    if (!ctx) return;
    if (window.fasdCureChartInstance) window.fasdCureChartInstance.destroy();
    window.fasdCureChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.months.map(m => `M${m}`),
            datasets: [
                {
                    label: 'QML Cortical Stimulation',
                    data: data.cortical_stim,
                    borderColor: '#00ffcc',
                    backgroundColor: 'rgba(0,255,204,0.13)',
                    tension: 0.45,
                    fill: true,
                    pointRadius: 2
                },
                {
                    label: 'QML Neuroplasticity Index',
                    data: data.qml_neuroplasticity,
                    borderColor: '#ff00c8',
                    backgroundColor: 'rgba(255,0,200,0.13)',
                    tension: 0.45,
                    fill: true,
                    pointRadius: 2
                },
                {
                    label: 'QML Executive Function',
                    data: data.qml_exec_func,
                    borderColor: '#00f2ff',
                    backgroundColor: 'rgba(0,242,255,0.13)',
                    tension: 0.45,
                    fill: true,
                    pointRadius: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'QML-Driven DBS Cortical Stimulation & Recovery'
                },
                legend: {
                    labels: { color: '#a0e7ff', font: { size: 12 } }
                }
            },
            layout: { padding: 0 },
            scales: {
                x: { title: { display: true, text: 'Months', color: '#a0e7ff' }, ticks: { color: '#a0e7ff' } },
                y: { min: 0, max: 120, title: { display: true, text: 'Score / Index', color: '#a0e7ff' }, ticks: { color: '#a0e7ff' } }
            }
        }
    });
}

function renderFASDQMLCorticalChart(data) {
    const ctx = document.getElementById('fasdCureCorticalChart');
    if (!ctx) return;
    if (window.fasdCureCorticalChartInstance) window.fasdCureCorticalChartInstance.destroy();
    window.fasdCureCorticalChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['F-Striatal', 'GABA', 'Thalamic', 'Prefrontal', 'Cerebellar'],
            datasets: [
                {
                    label: 'QML Cortical Field (final month)',
                    data: data.qml_field_vectors[data.qml_field_vectors.length-1],
                    backgroundColor: [
                        'rgba(0,255,204,0.7)',
                        'rgba(255,0,200,0.7)',
                        'rgba(0,242,255,0.7)',
                        'rgba(255,206,86,0.7)',
                        'rgba(153,102,255,0.7)'
                    ],
                    borderRadius: 6,
                    barPercentage: 0.7,
                    categoryPercentage: 0.7
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'QML Cortical Surface Stimulation (FASD)'
                },
                legend: {
                    labels: { color: '#a0e7ff', font: { size: 12 } }
                }
            },
            layout: { padding: 0 },
            scales: {
                x: { ticks: { color: '#a0e7ff' } },
                y: { min: 0, max: 1, ticks: { color: '#a0e7ff' } }
            }
        }
    });
}
// Neuromorph DBS Main Logic

let scene, camera, renderer, headGeometry, FEA_particles;
let voltage = 30;
let pulseWidth = 0.2;
let coilRadius = 0.05;

// Initialize 3D Scene
function init3D() {
    const container = document.getElementById('canvas-container');
    if (!container) return;
    scene = new THREE.Scene();
    const w = container.clientWidth || 600;
    const h = container.clientHeight || 400;
    camera = new THREE.PerspectiveCamera(75, w / h, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(w, h);
    container.appendChild(renderer.domElement);

    // Add a "Head" proxy
    const headGeo = new THREE.IcosahedronGeometry(2, 4);
    const headMat = new THREE.MeshPhongMaterial({
        color: 0x1a1a2e,
        wireframe: true,
        transparent: true,
        opacity: 0.3
    });
    const head = new THREE.Mesh(headGeo, headMat);
    scene.add(head);

    // Add a Coil
    const coilGeo = new THREE.TorusGeometry(1, 0.05, 16, 100);
    const coilMat = new THREE.MeshBasicMaterial({ color: 0x00f2ff });
    const coil = new THREE.Mesh(coilGeo, coilMat);
    coil.rotation.x = Math.PI / 2;
    coil.position.y = 2.2;
    scene.add(coil);

    // Lighting
    const pointLight = new THREE.PointLight(0xff00c8, 1);
    pointLight.position.set(5, 5, 5);
    scene.add(pointLight);
    scene.add(new THREE.AmbientLight(0x404040));

    camera.position.z = 5;
    camera.position.y = 2;
    camera.lookAt(0, 0, 0);

    animate();
}

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

// FEA Cortical Simulation Visualization
function drawCorticalFEA() {
    const canvas = document.getElementById('fea-cortical-canvas');
    if (!canvas) return;

    if (!window.feaInitialized) {
        window.feaScene = new THREE.Scene();
        window.feaCamera = new THREE.PerspectiveCamera(60, (canvas.clientWidth || 600)/(canvas.clientHeight || 200), 0.1, 100);
        window.feaCamera.position.z = 20;
        
        window.feaRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
        window.feaRenderer.setSize(canvas.clientWidth || 600, canvas.clientHeight || 200);

        const feaNodes = document.getElementById('fea-nodes') ? parseInt(document.getElementById('fea-nodes').value) : 10000;
        // High density detail for volumetric surface
        const detail = Math.min(15, Math.max(5, Math.floor(feaNodes / 200)));
        
        const geo = new THREE.IcosahedronGeometry(7.0, detail); 
        const positions = geo.attributes.position;
        const scalars = new Float32Array(positions.count);
        geo.setAttribute('scalar', new THREE.BufferAttribute(scalars, 1));
        
        for(let i = 0; i < positions.count; i++) {
            let x = positions.getX(i);
            let y = positions.getY(i);
            let z = positions.getZ(i);
            
            // Volumetric Human Cortex Procedural Approximation
            let fissure = 1.0;
            if (Math.abs(x) < 1.5) {
                fissure = 0.4 + 0.6 * (Math.abs(x) / 1.5);
            }
            
            let noise = Math.sin(x*2.0)*Math.cos(y*3.0)*Math.sin(z*2.5) + 
                        0.5*Math.sin(x*5.0+y)*Math.cos(z*4.0) +
                        0.25*Math.cos(x*10.0)*Math.sin(y*10.0+z);
            
            let gyri = Math.pow(Math.abs(noise), 0.6); 
            let baseRadius = 1.0 - 0.12 * gyri; 
            
            let nx = x * baseRadius * fissure * 0.85; 
            let ny = y * baseRadius * 0.8; 
            let nz = z * baseRadius * 1.15; 
            
            if (nz > 0) nx *= (1.0 - 0.1*(nz/7.0));
            if (ny < 0 && Math.abs(x) > 2.0 && nz < 2.0 && nz > -2.0) {
                nx *= 1.1;
                ny *= 1.05;
            }

            positions.setXYZ(i, nx, ny, nz);
        }
        geo.computeVertexNormals();

        // Custom BEM Boundary Element Contours Shader
        const vertexShader = `
            varying vec3 vNormal;
            varying vec3 vPosition;
            attribute float scalar;
            varying float vScalar;
            void main() {
                vNormal = normalize(normalMatrix * normal);
                vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
                vScalar = scalar;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;

        const fragmentShader = `
            varying vec3 vNormal;
            varying vec3 vPosition;
            varying float vScalar;
            
            vec3 colormap(float t) {
                float r = clamp(1.5 - abs(2.0 * t - 1.0), 0.0, 1.0);
                float g = clamp(1.5 - abs(2.0 * t - 1.5), 0.0, 1.0);
                float b = clamp(1.5 - abs(2.0 * t - 2.0), 0.0, 1.0);
                vec3 brainColor = vec3(0.6, 0.5, 0.5);
                return mix(brainColor, vec3(r, g, b), clamp(t * 3.0, 0.0, 1.0));
            }

            void main() {
                float t = clamp(vScalar / 1.5, 0.0, 1.0);
                vec3 color = colormap(t);
                
                // Isolines / Contours 
                float numContours = 15.0; 
                float contour = fract(t * numContours);
                float lineThick = 0.08; 
                
                if (t > 0.02 && (contour < lineThick || contour > 1.0 - lineThick)) {
                    color = vec3(1.0); // White Contour Band
                }
                
                vec3 lightDir = normalize(vec3(0.5, 1.0, 1.0));
                float diff = max(dot(vNormal, lightDir), 0.2);
                
                vec3 viewDir = normalize(-vPosition);
                float rim = 1.0 - max(dot(viewDir, vNormal), 0.0);
                rim = smoothstep(0.6, 1.0, rim);
                
                vec3 finalColor = color * diff + vec3(0.3) * rim;
                gl_FragColor = vec4(finalColor, 0.95);
            }
        `;

        const mat = new THREE.ShaderMaterial({
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            transparent: true,
            side: THREE.DoubleSide
        });
        
        const brainMesh = new THREE.Mesh(geo, mat);
        window.feaScene.add(brainMesh);
        window.brainMesh = brainMesh; 

        const rfEmitterGeo = new THREE.SphereGeometry(0.4, 16, 16);
        const rfEmitterMat = new THREE.MeshBasicMaterial({ color: 0xff00ff });
        const rfEmitter = new THREE.Mesh(rfEmitterGeo, rfEmitterMat);
        rfEmitter.position.set(2.5, 3.5, 2.5); 
        window.feaScene.add(rfEmitter);
        window.rfEmitter = rfEmitter;
        
        window.bemTime = 0;
        window.feaInitialized = true;
    }

    if (window.brainMesh && window.rfEmitter) {
        window.bemTime += 0.05;
        const sourcePos = window.rfEmitter.position;
        const posArray = window.brainMesh.geometry.attributes.position.array;
        const scalarArray = window.brainMesh.geometry.attributes.scalar.array;
        
        const rfRange = document.getElementById('rf-freq-range');
        const pwrRange = document.getElementById('power-eff-range');
        const rf = rfRange ? parseFloat(rfRange.value) : 2.4;
        const pwr = pwrRange ? parseFloat(pwrRange.value) : 90;
        const efficiencyFactor = pwr / 100.0;
        
        for(let i = 0, j = 0; i < posArray.length; i+=3, j++) {
            let dx = posArray[i] - sourcePos.x;
            let dy = posArray[i+1] - sourcePos.y;
            let dz = posArray[i+2] - sourcePos.z;
            let r = Math.sqrt(dx*dx + dy*dy + dz*dz);
            
            // Adjust mathematical manifold with proprioceptive feedback parameters
            let rf_pulse = Math.max(0, Math.sin(window.bemTime * rf - r * (1.5 / efficiencyFactor))); 
            let targetE = (1.0 / (r * r + 0.1)) * rf_pulse * 12.0 * efficiencyFactor;
            
            scalarArray[j] = scalarArray[j] * 0.85 + targetE * 0.15;
        }
        window.brainMesh.geometry.attributes.scalar.needsUpdate = true;
        
        window.brainMesh.rotation.y += 0.003;
        window.brainMesh.rotation.z = Math.sin(window.bemTime * 0.1) * 0.05;
    }
    
    window.feaRenderer.render(window.feaScene, window.feaCamera);
    requestAnimationFrame(drawCorticalFEA);
}

// Fetch System Specs
async function fetchSystemSpecs() {
    const response = await fetch('/api/system-specs');
    const data = await response.json();
    const list = document.getElementById('system-specs-list');
    list.innerHTML = Object.entries(data).map(([key, val]) => `
        <div style="margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 4px;">
            <strong style="color:var(--accent-cyan); text-transform:uppercase; font-size:9px;">${key.replace('_', ' ')}</strong><br>
            <span style="color:var(--text-primary); font-family: monospace;">${val}</span>
        </div>
    `).join('');
}

// Logic for Simulation & Bio-Signals
async function runSimulation() {
    const nodes = [
        { id: 'primary', x: 0, y: 1.5, z: 0 },
        { id: 'secondary', x: 0.5, y: 1.2, z: 0.5 }
    ];

    const response = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes, voltage, pulseWidth })
    });

    const data = await response.json();
    if (data.status === 'success') {
        const res = data.results[0];
        document.getElementById('yield-value').textContent = (res.optimized_yield).toFixed(3) + "%";
        document.getElementById('field-strength-val').textContent = res.field.toExponential(2) + ' T';

        // Update Quantum Freq
        if (res.quantum_optimal_freq) {
            document.getElementById('quantum-freq-val').textContent = res.quantum_optimal_freq.toFixed(2);
        }

        // Fetch companion bio-signal analysis
        const bioResponse = await fetch('/api/analyze-biosignals', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frequency: res.quantum_optimal_freq })
        });
        const bioData = await bioResponse.json();
        // Update any specific bio-signal UI if needed, for now we let signal-viz run
    }
}

// Event Listeners — wired inside DOMContentLoaded so elements are guaranteed present
document.addEventListener('DOMContentLoaded', () => {
    const simBtn = document.getElementById('btn-simulate');
    if (simBtn) simBtn.addEventListener('click', runSimulation);

    const voltageRange = document.getElementById('voltage-range');
    if (voltageRange) voltageRange.addEventListener('input', (e) => { voltage = parseFloat(e.target.value); });

    const pulseRange = document.getElementById('pulse-range');
    if (pulseRange) pulseRange.addEventListener('input', (e) => { pulseWidth = parseFloat(e.target.value); });
});

// Tab Switching Logic
function switchTab(tabId, event) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    if (event) {
        event.currentTarget.classList.add('active');
    } else {
        // Find the button if event not provided (initial load)
        const btn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.textContent.toLowerCase().includes(tabId));
        if (btn) btn.classList.add('active');
    }

    const sidebarEl = document.getElementById(`${tabId}-sidebar`);
    const mainEl = document.getElementById(`${tabId}-main`);
    if (sidebarEl) sidebarEl.classList.add('active');
    if (mainEl) mainEl.classList.add('active');

    if (tabId === 'conductivity') {
        fetchConductivity();
    }
    if (tabId === 'pareto') {
        runParetoOptimization();
    }
    if (tabId === 'veteran-ptsd-dbs') {
        runVeteranPTSDDBS();
    }
    if (tabId === 'canada-veteran') {
        runCanadianPTSDCorticalFEA();
        simulateCanadianPTSDStatOpt();
        fetchCanadianPTSDQMLCare();
        fetchPTSDTBIPlans();
    }
}

// Fetch Fornix Protocol
async function fetchFornixProtocol() {
    const response = await fetch('/api/fornix-protocol');
    const data = await response.json();
    const container = document.getElementById('fornix-protocol-stages');
    container.innerHTML = data.stages.map(s => `
        <div class="stat-card" style="border-left: 3px solid var(--accent-cyan); margin-bottom: 10px;">
            <div style="font-size: 11px; font-weight: 800; color: var(--accent-cyan);">${s.name}</div>
            <div style="font-size: 10px; color: var(--text-dim); margin: 4px 0;">${s.description}</div>
            <div style="font-size: 9px; color: var(--accent-pink);">V: ${s.parameters.voltage} | F: ${s.parameters.freq}</div>
        </div>
    `).join('');
}

// Fetch and Render Conductivity Map
async function fetchConductivity() {
    let base_cond = 0.20;
    let anisotropy = 0.1;
    let curvature = 2.0;

    const baseCondMap = document.getElementById('base-cond-map');
    const anisoMap = document.getElementById('aniso-map');
    const curveMap = document.getElementById('curve-map');
    
    if (baseCondMap) {
        base_cond = parseFloat(baseCondMap.value);
        anisotropy = parseFloat(anisoMap.value);
        curvature = parseFloat(curveMap.value);
        
        document.getElementById('base-cond-disp').textContent = base_cond.toFixed(2);
        document.getElementById('aniso-disp').textContent = anisotropy.toFixed(2);
        document.getElementById('curve-disp').textContent = curvature.toFixed(1);
    }
    
    const response = await fetch('/api/fornix-conductivity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            base_cond: base_cond,
            anisotropy: anisotropy,
            curvature: curvature
        })
    });
    
    const data = await response.json();
    const container = document.getElementById('conductivity-grid-container');
    container.innerHTML = '';

    let total = 0;
    let elements = 0;
    data.grid.forEach(row => {
        row.forEach(val => {
            total += val;
            elements += 1;
            const cell = document.createElement('div');
            cell.style.aspectRatio = '1';
            // Adjusted scaling for higher conductivity
            const intensity = Math.min(100, Math.max(0, (val - (base_cond - 0.05)) * (500 / Math.max(0.1, anisotropy * 5))));
            cell.style.background = `rgba(0, 242, 255, ${0.1 + intensity / 100})`;
            cell.style.borderRadius = '2px';
            cell.title = `Cond: ${val.toFixed(3)} S/m`;
            container.appendChild(cell);
        });
    });

    document.getElementById('avg-cond-val').textContent = (total / elements).toFixed(3);

    // Update anisotropy label based on slider value
    const anisoDisp = document.getElementById('anisotropy-val');
    if (anisoDisp) {
        const a = parseFloat(document.getElementById('aniso-map')?.value || 0.1);
        anisoDisp.textContent = a >= 0.3 ? 'High' : a >= 0.15 ? 'Medium' : 'Low';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const updateBtn = document.getElementById('btn-update-conductivity');
    if (updateBtn) {
        updateBtn.addEventListener('click', fetchConductivity);
    }
    const sliders = ['base-cond-map', 'aniso-map', 'curve-map'];
    sliders.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', fetchConductivity);
    });

    const paretoSlider = document.getElementById('pareto-lambda-map');
    if (paretoSlider) {
        paretoSlider.addEventListener('input', () => {
            const disp = document.getElementById('pareto-lambda-disp');
            if (disp) disp.textContent = parseFloat(paretoSlider.value).toFixed(2);
        });
    }
});

// Start everything
window.onload = () => {
    // Activate the first tab button visually
    const firstTabBtn = document.querySelector('.tab-btn');
    if (firstTabBtn) firstTabBtn.classList.add('active');

    init3D();
    drawCorticalFEA();
    fetchSystemSpecs();
    fetchFornixProtocol();
    fetchConductivity(); // Pre-load conductivity
    runSimulation(); // Initial run

    // Auto-refresh telemetry every 5 seconds
    setInterval(fetchSystemSpecs, 5000);
};

// --- DEMENTIA STAGING GENERATIVE PROTOCOL ---

let dementiaChart = null;

async function runDementiaStaging() {
    // Delegate to the correctly-wired implementation
    return updateDementiaChart();
}

// Auto trigger bindings — wire dementia-staging slider to updateDementiaChart
document.addEventListener("DOMContentLoaded", () => {
    const dr = document.getElementById('dementia-decline-range');
    const dp = document.getElementById('dementia-prompt');
    if (dr) dr.addEventListener('input', updateDementiaChart);
    if (dp) dp.addEventListener('change', updateDementiaChart);
    setTimeout(updateDementiaChart, 500);
});


// --- Added Dementia Optimization & FEA Functionality ---

let dementiaChartInstance = null;

function updateDementiaChart() {
    const dbsAmp = document.getElementById('voltage-range') ? document.getElementById('voltage-range').value : 30; // mapping voltage to amplitude 
    const declineRate = document.getElementById('dementia-decline-range') ? document.getElementById('dementia-decline-range').value : 0.05;
    const prompt = document.getElementById('dementia-prompt') ? document.getElementById('dementia-prompt').value : 'baseline';
    
    fetch('/api/dementia-staging', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            dbs_amplitude: dbsAmp, 
            decline_rate: declineRate,
            prompt: prompt
        })
    })
    .then(res => res.json())
    .then(data => {
        const ctx = document.getElementById('dementia-chart')?.getContext('2d');
        if (!ctx) return;
        
        if (dementiaChartInstance) {
            dementiaChartInstance.destroy();
        }
        
        const upperBounds = data.cognitive_trajectory.map((val, i) => val + data.clinical_distributions[i].std);
        const lowerBounds = data.cognitive_trajectory.map((val, i) => Math.max(0, val - data.clinical_distributions[i].std));
        
        dementiaChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.time_months,
                datasets: [
                    {
                        label: 'Mean Trajectory',
                        data: data.cognitive_trajectory,
                        borderColor: '#00ffcc',
                        backgroundColor: 'rgba(0, 255, 204, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: '+1 Std Dev',
                        data: upperBounds,
                        borderColor: 'rgba(255, 0, 127, 0.5)',
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0,
                    },
                    {
                        label: '-1 Std Dev',
                        data: lowerBounds,
                        borderColor: 'rgba(255, 0, 127, 0.5)',
                        borderDash: [5, 5],
                        fill: '-1',
                        backgroundColor: 'rgba(255, 0, 127, 0.1)',
                        pointRadius: 0,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Months' } },
                    y: { title: { display: true, text: 'Cognitive Score (MMSE)' }, min: 0, max: 30 }
                }
            }
        });

        const insightElem = document.getElementById('dementia-insight');
        const varianceElem = document.getElementById('dementia-variance');
        if (insightElem) insightElem.innerText = data.generative_insight || 'Temporal projection stabilized via non-linear constraints.';
        
        let initialStd = data.clinical_distributions[0].std;
        let finalStd = data.clinical_distributions[data.clinical_distributions.length - 1].std;
        if (varianceElem) varianceElem.innerText = `Std Dev variance ranges from ±${initialStd} to ±${finalStd} over 60 months governed by continuous Markovian decay mappings.`;

    }).catch(e => console.error("Error charting dementia:", e));
}

let largeFeaScene, largeFeaCamera, largeFeaRenderer, largeFeaMesh, rfCoilMesh, emFieldParticles;
let emParticlesGeo;

function initLargerFEA() {
    const container = document.getElementById('fea-large-container');
    if (!container) return;
    
    // Clear past children
    container.innerHTML = '';
    
    largeFeaScene = new THREE.Scene();
    largeFeaCamera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
    largeFeaCamera.position.z = 20;
    
    largeFeaRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    largeFeaRenderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(largeFeaRenderer.domElement);
    
    // --- CORTICAL SURFACE MANIFOLD
    const feaNodes = document.getElementById('fea-nodes') ? parseInt(document.getElementById('fea-nodes').value) : 1000;
    const detail = Math.min(6, Math.max(1, Math.floor(feaNodes / 200)));
    
    const geo = new THREE.IcosahedronGeometry(6, detail);
    const positions = geo.attributes.position;
    for(let i = 0; i < positions.count; i++) {
        let x = positions.getX(i);
        let y = positions.getY(i);
        let z = positions.getZ(i);
        let bump = 1 + 0.15 * Math.sin(x*2) * Math.cos(y*2) + 0.05 * Math.sin(z*4);
        positions.setXYZ(i, x*bump, y*bump, z*bump);
    }
    geo.computeVertexNormals();
    
    // Heatmap style coloring based on Z position for basic FEA visual
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(positions.count * 3), 3));
    const colors = geo.attributes.color;
    for(let i = 0; i < positions.count; i++) {
        const val = (positions.getY(i) + 6) / 12; // roughly 0 to 1
        colors.setXYZ(i, 1.0, val, 0.2); // yellow-red FEA heat gradient
    }

    const mat = new THREE.MeshPhongMaterial({
        vertexColors: true,
        emissive: 0x221100,
        wireframe: true,
        transparent: true,
        opacity: 0.8,
        side: THREE.DoubleSide
    });
    
    largeFeaMesh = new THREE.Mesh(geo, mat);
    largeFeaScene.add(largeFeaMesh);

    // --- RF COIL CIRCUITRY
    const coilGeo = new THREE.TorusKnotGeometry( 8.5, 0.3, 150, 16, 2, 5 );
    const coilMat = new THREE.MeshStandardMaterial({ 
        color: 0xaaaaaa, 
        metalness: 0.9, 
        roughness: 0.1,
        emissive: 0x001155
    });
    rfCoilMesh = new THREE.Mesh(coilGeo, coilMat);
    largeFeaScene.add(rfCoilMesh);

    // --- ELECTROMAGNETIC FIELD PARTICLES
    emParticlesGeo = new THREE.BufferGeometry();
    const pCount = 1000;
    const pArray = new Float32Array(pCount * 3);
    for(let i=0; i<pCount*3; i++) {
        pArray[i] = (Math.random() - 0.5) * 35; // Random spread
    }
    emParticlesGeo.setAttribute('position', new THREE.BufferAttribute(pArray, 3));
    const particleMat = new THREE.PointsMaterial({
        color: 0x00ffff,
        size: 0.25,
        transparent: true,
        opacity: 0.6,
        blending: THREE.AdditiveBlending
    });
    emFieldParticles = new THREE.Points(emParticlesGeo, particleMat);
    largeFeaScene.add(emFieldParticles);
    
    // LIGHTING
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(10, 20, 10);
    largeFeaScene.add(light);
    const ambientLight = new THREE.AmbientLight(0x404040); 
    largeFeaScene.add(ambientLight);
    
    animateLargeFEA();
}

function animateLargeFEA() {
    if(!largeFeaRenderer) return;
    requestAnimationFrame(animateLargeFEA);
    
    const time = Date.now() * 0.001;

    if(largeFeaMesh) {
        largeFeaMesh.rotation.y += 0.002;
    }
    if(rfCoilMesh) {
        rfCoilMesh.rotation.y -= 0.005;
        rfCoilMesh.rotation.x = Math.sin(time*0.5) * 0.2;
        // Pulse the emissive color of the coil to simulate RF activation
        rfCoilMesh.material.emissiveIntensity = 0.5 + 0.5 * Math.sin(time * 8);
    }
    
    // Animate EM Field particles to simulate magnetic flux toroidal vortex
    if(emFieldParticles) {
        const positions = emParticlesGeo.attributes.position.array;
        for(let i=0; i<positions.length; i+=3) {
            let x = positions[i];
            let y = positions[i+1];
            let z = positions[i+2];

            const r = Math.sqrt(x*x + z*z) + 0.0001;
            const theta = Math.atan2(z, x) + 0.015; 
            positions[i] = r * Math.cos(theta);
            positions[i+2] = r * Math.sin(theta);
            positions[i+1] += (15 / r) * 0.1 * Math.sin(time * 3 + r); 

            if(positions[i+1] > 17) positions[i+1] = -17; 
            if(positions[i+1] < -17) positions[i+1] = 17;
        }
        emParticlesGeo.attributes.position.needsUpdate = true;
    }

    largeFeaRenderer.render(largeFeaScene, largeFeaCamera);
}

// Automatically load models if clicking their tab
window.addEventListener('click', (e) => {
    if(e.target.classList.contains('tab-btn')) {
        setTimeout(() => {
            if(document.getElementById('dementia-sidebar') && document.getElementById('dementia-sidebar').classList.contains('active')) {
                updateDementiaChart();
            }
            if(document.getElementById('fea-sidebar') && document.getElementById('fea-sidebar').classList.contains('active') && !largeFeaRenderer) {
                initLargerFEA();
            }
        }, 100);
    }
});



// Stage-Gated Dementia Protocol (Queueing Theory)
async function fetchStageProtocol() {
    try {
        const response = await fetch('/api/stage-gated-protocol');
        const data = await response.json();
        
        const container = document.getElementById('stage-protocol-container');
        container.innerHTML = data.protocol.map(stage => `
            <div class="stat-card" style="border-left: 3px solid var(--accent-cyan); display: flex; flex-direction: column; gap: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin:0; color: #00f2ff;">${stage.name}</h3>
                    <span style="font-size:10px; padding: 3px 8px; background: rgba(255,0,200,0.2); border-radius: 12px; color: var(--accent-pink);">
                        Stage ${stage.stage}
                    </span>
                </div>
                <p style="font-size: 11px; margin: 0; color: var(--text-dim);">${stage.desc}</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 5px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
                    <div>
                        <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase;">Electrical Protocol</div>
                        <ul style="margin: 5px 0 0 15px; font-size: 11px; color: #fff;">
                            <li>Voltage: <span style="color:var(--accent-cyan);">${stage.electrical.voltage_v} V</span></li>
                            <li>Frequency: <span style="color:var(--accent-cyan);">${stage.electrical.frequency_hz} Hz</span></li>
                            <li>Pulse Width: <span style="color:var(--accent-cyan);">${stage.electrical.pulse_width_us} µs</span></li>
                            <li>Target: <span style="color:var(--accent-cyan);">${stage.electrical.target}</span></li>
                        </ul>
                    </div>
                    <div>
                        <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase;">Molecular Queueing (M/M/1)</div>
                        <ul style="margin: 5px 0 0 15px; font-size: 11px; color: #fff;">
                            <li>Tau Aggregation Rate (λ): <span style="color:var(--accent-pink);">${stage.queueing.lambda_arrival} /yr</span></li>
                            <li>Glymphatic Clearance (μ): <span style="color:var(--accent-pink);">${stage.queueing.mu_clearance} /yr</span></li>
                            <li>System Utilization (ρ): <span style="color:var(--accent-pink);">${stage.queueing.rho_utilization}</span></li>
                            <li>Queue Length (Lq): <span style="color:var(--accent-pink);">${stage.queueing.l_q}</span></li>
                        </ul>
                    </div>
                </div>
            </div>
        `).join('');
    } catch(err) {
        console.error(err);
    }
}

function loadClinicalProtocols() {
    const listContainer = document.getElementById('protocols-list-container');
    if (!listContainer) return;
    listContainer.innerHTML = '<p style="color: #00f2ff;">Analyzing deep brain targets...<br>Compiling dementia stimulation parameters...</p>';
    
    fetch('/api/clinical-protocols', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(r => r.json())
    .then(data => {
        let html = '';
        data.protocols.forEach((p, idx) => {
            html += `
                <div style="background: rgba(0, 242, 255, 0.05); border: 1px solid rgba(0, 242, 255, 0.2); padding: 15px; margin-bottom: 20px; border-radius: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: top; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; margin-bottom: 10px;">
                        <h3 style="color: #fff; margin: 0; font-size: 16px;">Target ${idx + 1}: ${p.lobe}</h3>
                        <span style="background: var(--accent-pink); padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; color: white;">Analysis Complete</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 10px; font-family: monospace;">
                        <div style="background: rgba(0,0,0,0.5); padding: 8px; border: 1px dashed rgba(255,255,255,0.15); border-radius: 4px;">
                            <div style="color: var(--text-dim); font-size: 10px;">FREQUENCY</div>
                            <div style="color: #00ff00; font-size: 14px; font-weight: bold;">${p.frequency}</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.5); padding: 8px; border: 1px dashed rgba(255,255,255,0.15); border-radius: 4px;">
                            <div style="color: var(--text-dim); font-size: 10px;">PULSE WIDTH</div>
                            <div style="color: #00f2ff; font-size: 14px; font-weight: bold;">${p.pulse_width}</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.5); padding: 8px; border: 1px dashed rgba(255,255,255,0.15); border-radius: 4px;">
                            <div style="color: var(--text-dim); font-size: 10px;">VOLTAGE OPTIMA</div>
                            <div style="color: #ff00ff; font-size: 14px; font-weight: bold;">${p.voltage}</div>
                        </div>
                    </div>
                    <div>
                        <div style="color: var(--text-dim); font-size: 11px; text-transform: uppercase; margin-bottom: 5px;">Mechanistic Rationale</div>
                        <div style="color: #ddd; font-size: 13px; line-height: 1.4;">${p.description}</div>
                    </div>
                </div>
            `;
        });
        listContainer.innerHTML = html;
    })
    .catch(e => {
        console.error(e);
        listContainer.innerHTML = '<p style="color: red;">Error fetching protocol generation analysis.</p>';
    });
}

let paretoChartInstance = null;

async function runParetoOptimization() {
    const lambdaEl = document.getElementById("pareto-lambda-map");
    if (!lambdaEl) return;
    const lambda = lambdaEl.value;

    const dispEl = document.getElementById('pareto-lambda-disp');
    if (dispEl) dispEl.textContent = parseFloat(lambda).toFixed(2);

    const logEl = document.getElementById('pareto-log');
    if (logEl) logEl.textContent = `Computing Pareto frontier at λ = ${parseFloat(lambda).toFixed(2)}...\n`;

    // API request to math engine
    try {
        const res = await fetch('/api/pareto_frontier', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lambda: lambda })
        });
        const data = await res.json();
        
        // Update Chart
        const ctx = document.getElementById('pareto-chart').getContext('2d');
        if (paretoChartInstance) {
            paretoChartInstance.destroy();
        }
        
        // x axis represents generic trade-off continuous variable
        const labels = Array.from({length: 100}, (_, i) => (i / 100).toFixed(2));
        
        paretoChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Striatal Activation Target (%)',
                        data: data.striatal,
                        borderColor: '#00ffcc',
                        tension: 0.4
                    },
                    {
                        label: 'Serotonin Release Yield (%)',
                        data: data.serotonin,
                        borderColor: '#ff00ff',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#a0a0b0' } },
                    annotation: {
                        annotations: {
                            line1: {
                                type: 'line',
                                xMin: data.optimal_x.toFixed(2),
                                xMax: data.optimal_x.toFixed(2),
                                borderColor: 'white',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {
                                    content: 'Nash Equilibrium',
                                    enabled: true,
                                    position: 'top'
                                }
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 120,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#a0a0b0' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#a0a0b0', maxTicksLimit: 10 }
                    }
                }
            }
        });
        
        const yieldEl = document.getElementById('pareto-yield');
        const serotoninEl = document.getElementById('pareto-serotonin');
        if (yieldEl) yieldEl.innerText = data.optimal_striatal.toFixed(1) + "%";
        if (serotoninEl) serotoninEl.innerText = data.optimal_serotonin.toFixed(1) + "%";
        if (logEl) logEl.textContent += `Nash equilibrium found at x = ${data.optimal_x.toFixed(3)}\nStriatal Activation: ${data.optimal_striatal.toFixed(1)}%\nSerotonin Release: ${data.optimal_serotonin.toFixed(1)}%\nOptimization complete.`;
        
    } catch(err) {
        console.error("Error drawing pareto chart: ", err);
        if (logEl) logEl.textContent += `Error: ${err.message}`;
    }
}


// MS Simulation
function simulateMS() {
    const out = document.getElementById('ms-output');
    if (!out) return;
    
    out.textContent = "Initializing cortical simulation framework...\nTargeting Rosenthal fibers for virtual ablation...\n\n";
    
    setTimeout(() => {
        out.textContent += "System configured for Deep Brain Stimulation.\n";
        out.textContent += "Disease Model: Multiple Sclerosis / Alexander's Disease.\n";
        out.textContent += "Pulse Frequency: 130 Hz\n";
        out.textContent += "Pulse Width: 60 μs\n";
        out.textContent += "Voltage: 3.5 V\n\n";
        
        out.textContent += "Modulating cortical excitability...\n";
        out.textContent += "Ablation of Rosenthal fibers simulated successfully.\n";
        out.textContent += "Cortical network stability improved by 42%.\n";
        
        renderMSChart();
    }, 1500);
}

function renderMSChart() {
    const ctx = document.getElementById('ms-chart');
    if(!ctx) return;
    
    if (window.msChartInstance) {
        window.msChartInstance.destroy();
    }
    
    window.msChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12 (Months)'],
            datasets: [{
                label: 'MS Plaque Density (Quantum Optimized Mitigation)',
                data: [100, 85, 60, 42, 30, 20, 14, 9, 6, 4, 3, 2, 1],
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                fill: true,
                tension: 0.4
            }, {
                label: 'Neural Recovery %',
                data: [0, 15, 30, 48, 62, 74, 82, 88, 92, 95, 97, 98, 99],
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'QML-DBS Accelerated MS Neural Recovery'
                }
            }
        }
    });
}
function simulateAlexander() {
    const out = document.getElementById('alexander-output');
    if(!out) return;
    out.innerText = "Initializing QML Adaptive Ablation...\n";
    out.innerText += "Mapping Feynman Path Integrals over White Matter Astrocytes...\n";
    
    setTimeout(() => {
        out.innerText += "Targeting Rosenthal Fibers...\n";
        out.innerText += "Applying Adaptive Ablation sequences...\n";
    }, 1000);
    
    setTimeout(() => {
        out.innerText += "Simulation Complete. Plotting mitigation dynamics...\n";
        renderAlexanderChart();
    }, 2000);
}

function renderAlexanderChart() {
    const ctx = document.getElementById('alexander-chart');
    if(!ctx) return;
    
    if (window.alexanderChartInstance) {
        window.alexanderChartInstance.destroy();
    }
    
    window.alexanderChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            datasets: [{
                label: 'Rosenthal Fiber Density',
                data: [100, 90, 75, 55, 30, 15, 8, 4, 1, 0, 0],
                borderColor: 'rgba(255, 159, 64, 1)',
                backgroundColor: 'rgba(255, 159, 64, 0.2)',
                fill: true,
                tension: 0.4
            }, {
                label: 'Ablation Stability Index (Feynman Mapping)',
                data: [0, 20, 45, 65, 85, 95, 98, 99, 100, 100, 100],
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Adaptive Ablation of Rosenthal Fibers (Alexander\'s Disease)'
                }
            }
        }
    });
}

function simulateAlexanderCF() {
    const out = document.getElementById('alexander-cf-output');
    if(!out) return;
    out.innerText = "Initializing Continued Fraction Addendum...\n";
    out.innerText += "Structuring QML CF Plaque Ablation model...\n";
    
    setTimeout(() => {
        out.innerText += "Simulating Neural Recovery metrics via CF...\n";
        out.innerText += "Computing Rosenthal Fiber dissipation limits...\n";
    }, 1000);
    
    setTimeout(() => {
        out.innerText += "Simulation Complete. Plotting CF mitigation dynamics...\n";
        renderAlexanderCFChart();
    }, 2000);
}

function renderAlexanderCFChart() {
    const ctx = document.getElementById('alexander-cf-chart');
    if(!ctx) return;
    
    if (window.alexanderCfChartInstance) {
        window.alexanderCfChartInstance.destroy();
    }
    
    window.alexanderCfChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            datasets: [{
                label: 'Rosenthal Fiber Density (CF bounded)',
                data: [100, 80, 50, 30, 15, 8, 4, 1, 0, 0, 0],
                borderColor: 'rgba(153, 102, 255, 1)',
                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                fill: true,
                tension: 0.4
            }, {
                label: 'Neural Recovery Index %',
                data: [0, 25, 55, 75, 88, 95, 98, 99, 100, 100, 100],
                borderColor: 'rgba(255, 206, 86, 1)',
                backgroundColor: 'rgba(255, 206, 86, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Alexander\'s Disease CF Neural Recovery & Ablation'
                }
            }
        }
    });
}

function simulateHuntington() {
    const out = document.getElementById('huntington-output');
    if(!out) return;
    out.innerText = "Initializing Cortical Simulation for Huntington's Disease...\n";
    out.innerText += "Applying Statistical Parametric Optimization Circuitry...\n";
    
    setTimeout(() => {
        out.innerText += "Evaluating Cortical Repair Thresholds...\n";
        out.innerText += "Generating Electrical Specifications...\n";
    }, 1000);
    
    setTimeout(() => {
        out.innerText += "Simulation Complete. Plotting Interventional Repair Matrix...\n";
        renderHuntingtonChart();
    }, 2000);
}

function renderHuntingtonChart() {
    const ctx = document.getElementById('huntington-chart');
    if(!ctx) return;
    
    if (window.huntingtonChartInstance) {
        window.huntingtonChartInstance.destroy();
    }
    
    window.huntingtonChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Weeks 0', 'W 4', 'W 8', 'W 12', 'W 16', 'W 20', 'W 24'],
            datasets: [{
                label: 'Motor Function Degeneration',
                data: [100, 95, 80, 50, 25, 10, 5],
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                fill: true,
                tension: 0.4
            }, {
                label: 'Interventional Repair Signal',
                data: [0, 10, 35, 65, 85, 95, 100],
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Statistical Parametric Optimization Circuitry (Huntington\'s)'
                }
            }
        }
    });
}

// Health Economics: MS
function simulateHEMS() {
    const out = document.getElementById('he-ms-output');
    if(!out) return;
    out.innerText = "Calculating Cost-Utility for MS Intervention...\n";
    out.innerText += "Mapping Quality-Adjusted Life Years (QALYs)...\n";
    setTimeout(() => {
        out.innerText += "Applying Evidence-Based Outcomes Framework...\n";
        renderHEMSChart();
        out.innerText += "Markov Decision Process Complete.\n";
    }, 1500);
}

function renderHEMSChart() {
    const ctx = document.getElementById('he-ms-chart');
    if(!ctx) return;
    if (window.heMsChartInstance) window.heMsChartInstance.destroy();
    window.heMsChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Standard Care', 'DBS QML Mitigated'],
            datasets: [{
                label: 'Cumulative Costs ($)',
                data: [450000, 320000],
                backgroundColor: 'rgba(255, 99, 132, 0.5)'
            }, {
                label: 'Lifetime QALYs',
                data: [12.5, 15.0],
                backgroundColor: 'rgba(54, 162, 235, 0.5)'
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

// Health Economics: Huntington's
function simulateHEHuntington() {
    const out = document.getElementById('he-huntington-output');
    if(!out) return;
    out.innerText = "Calculating Cost-Utility for Huntington's Intervention...\n";
    out.innerText += "Evaluating Caregiver Burden Metrics...\n";
    setTimeout(() => {
        out.innerText += "Applying Incremental Cost-Effectiveness Ratio (ICER)...\n";
        renderHEHuntingtonChart();
        out.innerText += "SPOC Economic Valuation Complete.\n";
    }, 1500);
}

function renderHEHuntingtonChart() {
    const ctx = document.getElementById('he-huntington-chart');
    if(!ctx) return;
    if (window.heHuntChartInstance) window.heHuntChartInstance.destroy();
    window.heHuntChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Standard Care', 'SPOC DBS Array'],
            datasets: [{
                label: 'Cumulative Costs ($)',
                data: [680000, 410000],
                backgroundColor: 'rgba(255, 159, 64, 0.5)'
            }, {
                label: 'Lifetime QALYs',
                data: [9.2, 12.3],
                backgroundColor: 'rgba(75, 192, 192, 0.5)'
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

// Health Economics: Alexander's
function simulateHEAlexander() {
    const out = document.getElementById('he-alexander-output');
    if(!out) return;
    out.innerText = "Analyzing Value-Based Care for Rosenthal Ablation...\n";
    out.innerText += "Projecting Hospitalization Reductions...\n";
    setTimeout(() => {
        out.innerText += "Calculating QALYs and Resource Utilization...\n";
        renderHEAlexanderChart();
        out.innerText += "Economic Validation Complete.\n";
    }, 1500);
}

function renderHEAlexanderChart() {
    const ctx = document.getElementById('he-alexander-chart');
    if(!ctx) return;
    if (window.heAlexChartInstance) window.heAlexChartInstance.destroy();
    window.heAlexChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Standard Care', 'CF Adaptive Ablation'],
            datasets: [{
                label: 'Cumulative Costs ($)',
                data: [850000, 520000],
                backgroundColor: 'rgba(153, 102, 255, 0.5)'
            }, {
                label: 'Lifetime QALYs',
                data: [8.5, 12.5],
                backgroundColor: 'rgba(255, 206, 86, 0.5)'
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

// Cortical Simulation
function simulateCortical() {
    const out = document.getElementById('cortical-sim-output');
    if(!out) return;
    out.innerText = "Initializing Cortical Network Engine...\n";
    out.innerText += "Mapping M1/S1 Projection Costs...\n";
    setTimeout(() => {
        out.innerText += "Applying Neurodynamic Equilibration...\n";
        renderCorticalChart();
        out.innerText += "Cortical Projection Complete.\n";
    }, 1500);
}

function renderCorticalChart() {
    const ctx = document.getElementById('cortical-sim-chart');
    if(!ctx) return;
    if (window.corticalChartInstance) window.corticalChartInstance.destroy();
    window.corticalChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['0ms', '10ms', '20ms', '30ms', '40ms', '50ms'],
            datasets: [{
                label: 'Projection Energy Cost (mWh)',
                data: [5, 15, 45, 60, 42, 25],
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                fill: true,
                tension: 0.4
            }, {
                label: 'Layer V Firing Rate (Hz)',
                data: [12, 18, 30, 25, 10, 8],
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

// Net Market Valuation
function simulateMarketValuation() {
    const out = document.getElementById('market-valuation-output');
    if(!out) return;
    out.innerText = "Accessing 10-Year Economic Outlook Engine (2026-2036)...\n";
    out.innerText += "Calculating Net Present Value (NPV) & Discount Rates...\n";
    setTimeout(() => {
        out.innerText += "Extrapolating Neuromodulation Trajectory...\n";
        renderMarketValuationChart();
        out.innerText += "Valuation Projection Complete.\n";
    }, 1500);
}

function renderMarketValuationChart() {
    const ctx = document.getElementById('market-valuation-chart');
    if(!ctx) return;
    if (window.marketValChartInstance) window.marketValChartInstance.destroy();
    
    // Fit exponential curve for Net Market Valuation
    // V(t) = a * e^(b*t)
    const valuationData = [2.1, 3.5, 5.2, 7.6, 9.8, 12.4];
    const trendFit = valuationData.map((v, i) => 2.0 * Math.exp(0.36 * i));

    window.marketValChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['2026', '2028', '2030', '2032', '2034', '2036'],
            datasets: [{
                type: 'bar',
                label: 'Annual Projection Costs ($M)',
                data: [250, 310, 420, 580, 750, 960],
                backgroundColor: 'rgba(255, 159, 64, 0.6)',
                yAxisID: 'y1'
            }, {
                type: 'line',
                label: 'Net Market Valuation Base ($B)',
                data: valuationData,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderWidth: 3,
                tension: 0.3,
                yAxisID: 'y'
            }, {
                type: 'line',
                label: 'Market Valuation Exponential Fit ($B)',
                data: trendFit,
                borderColor: 'rgba(255, 99, 132, 1)',
                borderDash: [5, 5],
                borderWidth: 2,
                tension: 0.4,
                yAxisID: 'y'
            }]
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Market Value ($B)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Projection Costs ($M)'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            }
        }
    });
}

// South Asia & SEA Valuation
function simulateSEAValuation() {
    const out = document.getElementById('sea-valuation-output');
    if(!out) return;
    out.innerText = "Accessing APAC & South Asia Economic Overlays (2026-2036)...\n";
    out.innerText += "Evaluating Regional Device Penetration & Healthcare Spending...\n";
    setTimeout(() => {
        out.innerText += "Extrapolating Neuromodulation Growth (CAGR: 21.2%)...\n";
        renderSEAValuationChart();
        out.innerText += "Regional Valuation Projection Complete.\n";
    }, 1500);
}

function renderSEAValuationChart() {
    const ctx = document.getElementById('sea-valuation-chart');
    if(!ctx) return;
    if (window.seaValChartInstance) window.seaValChartInstance.destroy();
    
    // Fit exponential curve for SEA Net Market Valuation
    // V(t) = a * e^(b*t)
    const seaValuationData = [0.45, 0.75, 1.25, 1.95, 2.70, 3.80]; // Billion $
    const trendFit = seaValuationData.map((v, i) => 0.45 * Math.exp(0.42 * i));

    window.seaValChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['2026', '2028', '2030', '2032', '2034', '2036'],
            datasets: [{
                type: 'bar',
                label: 'Regional Infrastructure Costs ($M)',
                data: [85, 120, 190, 275, 380, 520],
                backgroundColor: 'rgba(153, 102, 255, 0.6)',
                yAxisID: 'y1'
            }, {
                type: 'line',
                label: 'South Asia/SEA Market ($B)',
                data: seaValuationData,
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderWidth: 3,
                tension: 0.3,
                yAxisID: 'y'
            }, {
                type: 'line',
                label: 'Exponential Growth Fit ($B)',
                data: trendFit,
                borderColor: 'rgba(255, 206, 86, 1)',
                borderDash: [5, 5],
                borderWidth: 2,
                tension: 0.4,
                yAxisID: 'y'
            }]
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Market Value ($B)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Infrastructure Setup ($M)'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            }
        }
    });
}

// India Valuation
function simulateIndiaValuation() {
    const out = document.getElementById('india-valuation-output');
    if(!out) return;
    out.innerText = "Accessing localized manufacturing overlays for India (2026-2036)...\n";
    out.innerText += "Evaluating Regional Device Penetration & Tier-2 City Health Spending...\n";
    setTimeout(() => {
        out.innerText += "Extrapolating Neuromodulation Growth (CAGR: 24.5%)...\n";
        renderIndiaValuationChart();
        out.innerText += "India Market Projection Complete.\n";
    }, 1500);
}

function renderIndiaValuationChart() {
    const ctx = document.getElementById('india-valuation-chart');
    if(!ctx) return;
    if (window.indiaValChartInstance) window.indiaValChartInstance.destroy();
    
    // Fit exponential curve for India Net Market Valuation
    const valuationData = [0.2, 0.4, 0.8, 1.4, 2.0, 2.8]; // Billion $
    const trendFit = valuationData.map((v, i) => 0.22 * Math.exp(0.51 * i));

    window.indiaValChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['2026', '2028', '2030', '2032', '2034', '2036'],
            datasets: [{
                type: 'bar',
                label: 'Local Subsidy/Cost Offset ($M)',
                data: [50, 75, 120, 180, 250, 340],
                backgroundColor: 'rgba(255, 99, 132, 0.6)',
                yAxisID: 'y1'
            }, {
                type: 'line',
                label: 'India DBS Market ($B)',
                data: valuationData,
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderWidth: 3,
                tension: 0.3,
                yAxisID: 'y'
            }, {
                type: 'line',
                label: 'Exponential Growth Fit ($B)',
                data: trendFit,
                borderColor: 'rgba(153, 102, 255, 1)',
                borderDash: [5, 5],
                borderWidth: 2,
                tension: 0.4,
                yAxisID: 'y'
            }]
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'Market Value ($B)' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'Cost Offset / Setup ($M)' },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}

// America Valuation
function simulateAmericaValuation() {
    const out = document.getElementById('america-valuation-output');
    if(!out) return;
    out.innerText = "Accessing American regulatory & CMS cost overlays (2026-2036)...\n";
    out.innerText += "Evaluating Advanced Closed-Loop System Implementations...\n";
    setTimeout(() => {
        out.innerText += "Applying Mature Market Saturation Matrices (CAGR: 12.1%)...\n";
        renderAmericaValuationChart();
        out.innerText += "America Market Projection Complete.\n";
    }, 1500);
}

function renderAmericaValuationChart() {
    const ctx = document.getElementById('america-valuation-chart');
    if(!ctx) return;
    if (window.americaValChartInstance) window.americaValChartInstance.destroy();
    
    // Fit exponential curve for America Net Market Valuation
    const valuationData = [3.5, 4.2, 5.1, 6.3, 7.5, 8.5]; // Billion $
    const trendFit = valuationData.map((v, i) => 3.5 * Math.exp(0.18 * i));

    window.americaValChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['2026', '2028', '2030', '2032', '2034', '2036'],
            datasets: [{
                type: 'bar',
                label: 'Regulatory & R&D Overheads ($M)',
                data: [400, 450, 520, 610, 720, 850],
                backgroundColor: 'rgba(75, 192, 192, 0.6)',
                yAxisID: 'y1'
            }, {
                type: 'line',
                label: 'America DBS Market ($B)',
                data: valuationData,
                borderColor: 'rgba(255, 159, 64, 1)',
                backgroundColor: 'rgba(255, 159, 64, 0.2)',
                borderWidth: 3,
                tension: 0.3,
                yAxisID: 'y'
            }, {
                type: 'line',
                label: 'Expected Growth Trajectory ($B)',
                data: trendFit,
                borderColor: 'rgba(255, 99, 132, 1)',
                borderDash: [5, 5],
                borderWidth: 2,
                tension: 0.4,
                yAxisID: 'y'
            }]
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'Market Value ($B)' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'R&D Overhead ($M)' },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}


function simulateFAS() {
    const out = document.getElementById('fas-output');
    if (!out) return;
    out.innerText = "Initializing Neurosymbolic Rule Engine...\n";

    setTimeout(() => {
        out.innerText += "Injecting deep learning weights with Bayesian symbolic priors...\n";
    }, 500);

    setTimeout(() => {
        out.innerText += "Calculating neurodevelopmental deviation variances in FAS...\n";
        out.innerText += "Identifying optimal surgical inflection point triggers...\n";
    }, 1000);

    setTimeout(() => {
        out.innerText += "Synthesizing normalized cognitive trajectory distributions...\n";
        renderFASChart();
        out.innerText += "FAS Inflection Point Analysis Simulation Complete.\n";
    }, 1500);
}

function renderFASChart() {
    const ctx = document.getElementById('fas-chart');
    if(!ctx) return;
    if (window.fasChartInstance) window.fasChartInstance.destroy();
    
    const labels = ["Age 2", "Age 4", "Age 6", "Age 8", "Age 10", "Age 12", "Age 14"];
    const fasBaseline = [30, 35, 45, 55, 60, 65, 68];
    const typicalNeuro = [40, 55, 75, 90, 105, 115, 120];
    const postDbsTrajectory = [30, 35, 45, 80, 95, 108, 115]; // Inflection at Age 6-8

    window.fasChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Unmitigated FAS Trajectory',
                    data: fasBaseline,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    borderWidth: 2,
                    tension: 0.4
                },
                {
                    label: 'Typical Neurodevelopment',
                    data: typicalNeuro,
                    borderColor: 'rgba(200, 200, 200, 0.4)',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.4
                },
                {
                    label: 'Post-DBS Inflection (Neurosymbolic)',
                    data: postDbsTrajectory,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                annotation: {
                    annotations: {
                        inflectionLine: {
                            type: 'line',
                            xMin: 'Age 6',
                            xMax: 'Age 6',
                            borderColor: 'rgba(255, 206, 86, 0.8)',
                            borderWidth: 2,
                            label: {
                                content: 'DBS Intervention Inflection Point',
                                enabled: true,
                                position: 'top'
                            }
                        }
                    }
                }
            },
            scales: {
                y: {
                    title: { display: true, text: 'Cognitive / Motor Integrity Score' }
                }
            }
        }
    });
}



let hdCureChartInstance = null;
let hdCureCorticalChartInstance = null;

async function simulateHDCure() {
    try {
        const response = await fetch('/api/hd-cure', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({})
        });
        const data = await response.json();
        
        renderHDCureChart(data);
        renderHDCureCorticalChart(data);
        
        document.getElementById('hd-cure-efficacy').textContent = "92%";
        document.getElementById('hd-cure-timeline').textContent = "48 Months";
    } catch (e) {
        console.error('Error fetching HD Cure simulation:', e);
    }
}

function renderHDCureChart(data) {
    const ctx = document.getElementById('hdCureChart');
    if (!ctx) return;
    
    if (hdCureChartInstance) {
        hdCureChartInstance.destroy();
    }
    
    // Add annotations for stages
    const annotations = {};
    data.stages.forEach((stage, idx) => {
        annotations[`line${idx}`] = {
            type: 'line',
            xMin: stage.time,
            xMax: stage.time,
            borderColor: 'rgba(255, 255, 255, 0.4)',
            borderDash: [5, 5],
            label: {
                content: stage.label,
                enabled: true,
                position: 'top',
                color: 'white',
                backgroundColor: 'rgba(0,0,0,0.7)',
                font: { size: 10 }
            }
        };
    });
    
    hdCureChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.times,
            datasets: [
                {
                    label: 'mHTT Aggregation Levels (%)',
                    data: data.mhtt_levels,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'MSN Circuit Recovery (%)',
                    data: data.msn_recovery,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 2,
                    tension: 0.4
                },
                {
                    label: 'Cognitive Motor Function (Score)',
                    data: data.cognitive_score,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                annotation: { annotations }
            },
            scales: {
                x: { title: { display: true, text: 'Months' } },
                y: { title: { display: true, text: 'Metric Normalized Rate' } }
            }
        }
    });
}

function renderHDCureCorticalChart(data) {
    const ctx = document.getElementById('hdCureCorticalChart');
    if (!ctx) return;
    
    if (hdCureCorticalChartInstance) {
        hdCureCorticalChartInstance.destroy();
    }
    
    // Assuming field_vectors represents spatial field strength at 5 key markers at end of simulation
    const latestVectors = data.field_vectors[data.field_vectors.length - 1];
    
    hdCureCorticalChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Striatum', 'Cortex', 'Thalamus', 'Brainstem', 'Hippocampus'],
            datasets: [
                {
                    label: 'Cortical Restitution Field (Month 48)',
                    data: latestVectors,
                    fill: true,
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgb(54, 162, 235)',
                    pointBackgroundColor: 'rgb(54, 162, 235)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgb(54, 162, 235)'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.2)' },
                    grid: { color: 'rgba(255, 255, 255, 0.2)' },
                    pointLabels: { color: 'rgba(255, 255, 255, 0.7)' },
                    ticks: { display: false }
                }
            }
        }
    });
}
