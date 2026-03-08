document.addEventListener('DOMContentLoaded', () => {

    // ── DOM refs ─────────────────────────────────────────────────
    const navButtons      = document.querySelectorAll('.nav-btn');
    const tabTitle        = document.getElementById('tab-title');
    const tabSubtitle     = document.getElementById('tab-subtitle');
    const runBtn          = document.getElementById('run-simulation-btn');
    const loadingOverlay  = document.getElementById('loading-overlay');

    // Views
    const simulationView  = document.getElementById('simulation-view');
    const tremorView      = document.getElementById('tremor-view');
    const paradigmView    = document.getElementById('paradigm-view');
    const equipmentView   = document.getElementById('equipment-view');

    // Sim result spans
    const finalFreq       = document.getElementById('final-freq');
    const finalIntensity  = document.getElementById('final-intensity');
    const finalFitness    = document.getElementById('final-fitness');

    // Paradigm sub-selector + loading indicator
    const paradigmCondBtns = document.querySelectorAll('.paradigm-cond-btn');
    const paradigmLoading  = document.getElementById('paradigm-loading');
    const equipmentList    = document.getElementById('equipment-list');

    // ── State ────────────────────────────────────────────────────
    let currentCondition = 'stroke';
    let equipmentLoaded  = false;
    let tremorLoaded     = false;
    let paradigmCache    = {};

    // ── Plotly dark theme ────────────────────────────────────────
    const PL = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor:  'rgba(0,0,0,0)',
        font:   { color: '#8b949e', family: 'Inter' },
        margin: { l: 50, r: 30, t: 30, b: 50 },
        xaxis:  { gridcolor: 'rgba(255,255,255,0.1)' },
        yaxis:  { gridcolor: 'rgba(255,255,255,0.1)' },
        scene: {
            xaxis: { gridcolor: 'rgba(255,255,255,0.1)', backgroundcolor: 'rgba(0,0,0,0)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.1)', backgroundcolor: 'rgba(0,0,0,0)' },
            zaxis: { gridcolor: 'rgba(255,255,255,0.1)', backgroundcolor: 'rgba(0,0,0,0)' },
            bgcolor: 'rgba(0,0,0,0)'
        }
    };

    // ── All views array for easy hide-all ────────────────────────
    const allViews = [simulationView, tremorView, paradigmView, equipmentView];

    function hideAllViews() {
        allViews.forEach(v => v && v.classList.add('hidden'));
        runBtn.classList.add('hidden');
    }

    // ── Tab metadata ─────────────────────────────────────────────
    const tabConfig = {
        stroke: {
            title:    'Stroke Rehabilitation Optimization',
            subtitle: 'Dynamic rTMS Parameter Optimization via FEA/BEM for Motor Cortex',
            view:     simulationView, showRunBtn: true
        },
        dementia: {
            title:    'Dementia Cognitive Enhancement',
            subtitle: 'Deep rTMS Parameter Optimization via FEA/BEM for Prefrontal Cortex',
            view:     simulationView, showRunBtn: true
        },
        tremor: {
            title:    'Essential Tremor Clinical Care',
            subtitle: 'Inhibitory rTMS targeting the cerebello-thalamo-cortical circuit',
            view:     tremorView, showRunBtn: false
        },
        paradigm: {
            title:    'Optimal Treatment Paradigm',
            subtitle: 'Stage-gating · Hebbian-DBS Amplification · Continued Fraction Optimization',
            view:     paradigmView, showRunBtn: false
        },
        equipment: {
            title:    'rTMS Equipment & Machinery',
            subtitle: 'Clinical operating characteristics and system specifications',
            view:     equipmentView, showRunBtn: false
        }
    };

    // ── Main tab switcher ─────────────────────────────────────────
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.getAttribute('data-tab');
            if (!tabConfig[tab]) return;

            // Update nav active state
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentCondition = tab;

            // Update header
            tabTitle.textContent    = tabConfig[tab].title;
            tabSubtitle.textContent = tabConfig[tab].subtitle;

            // Hide all, then show the right view
            hideAllViews();
            tabConfig[tab].view.classList.remove('hidden');
            if (tabConfig[tab].showRunBtn) runBtn.classList.remove('hidden');

            // Reset sim panel when switching between stroke/dementia
            if (tab === 'stroke' || tab === 'dementia') resetSimUI();

            // Lazy-load data
            if (tab === 'tremor'    && !tremorLoaded)          loadTremorData();
            if (tab === 'equipment' && !equipmentLoaded)       loadEquipment();
            if (tab === 'paradigm') {
                const cond = document.querySelector('.paradigm-cond-btn.active')
                    ?.getAttribute('data-cond') || 'stroke';
                loadParadigm(cond);
            }
        });
    });

    // ── Run simulation button ─────────────────────────────────────
    runBtn.addEventListener('click', async () => {
        loadingOverlay.classList.remove('hidden');
        runBtn.disabled      = true;
        runBtn.style.opacity = '0.5';

        try {
            const res     = await fetch('/api/simulate', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ condition: currentCondition })
            });
            const payload = await res.json();

            setTimeout(() => {
                loadingOverlay.classList.add('hidden');
                runBtn.disabled      = false;
                runBtn.style.opacity = '1';
                if (payload.status === 'success') renderSimDashboard(payload.data);
                else alert('Simulation failed on cloud.');
            }, 1200);
        } catch (err) {
            console.error(err);
            loadingOverlay.classList.add('hidden');
            runBtn.disabled      = false;
            runBtn.style.opacity = '1';
        }
    });

    // ── Reset simulation UI ───────────────────────────────────────
    function resetSimUI() {
        if (finalFreq)      finalFreq.textContent      = '--';
        if (finalIntensity) finalIntensity.textContent = '--';
        if (finalFitness)   finalFitness.textContent   = '--';
        Plotly.purge('optimization-line-chart');
        Plotly.purge('fea-heatmap');
        Plotly.purge('bem-scatter');
    }

    // ── Simulation dashboard ──────────────────────────────────────
    function renderSimDashboard(data) {
        const p = data.optimization.final_parameters;
        finalFreq.textContent      = p.frequency_hz.toFixed(1);
        finalIntensity.textContent = p.intensity_mso.toFixed(1);
        finalFitness.textContent   = (p.fitness * 100).toFixed(1) + '%';

        const traj = data.optimization.convergence_trajectory;
        Plotly.newPlot('optimization-line-chart', [{
            x: traj.map(t => t.iteration), y: traj.map(t => t.frequency_hz),
            type: 'scatter', mode: 'lines+markers',
            line: { color: '#58a6ff', width: 3 }, marker: { size: 6, color: '#8a2be2' }
        }], {
            ...PL,
            title: { text: 'Convergence of Protocol Synthesis', font: { color: '#e6edf3' } },
            yaxis: { ...PL.yaxis, title: 'Frequency (Hz)' }
        }, { responsive: true });

        Plotly.newPlot('fea-heatmap', [{
            z: data.fea_grid, type: 'heatmap', colorscale: 'Viridis'
        }], { ...PL }, { responsive: true });

        const bem = data.bem_mesh;
        Plotly.newPlot('bem-scatter', [{
            x: bem.map(v => v.x), y: bem.map(v => v.y), z: bem.map(v => v.z),
            mode: 'markers', type: 'scatter3d',
            marker: {
                size: 5, color: bem.map(v => v.c), colorscale: 'Inferno', opacity: 0.8,
                showscale: true,
                colorbar: {
                    title: 'Magnetic Stress Strain',
                    titlefont: { color: '#e6edf3' }, tickfont: { color: '#e6edf3' }, thickness: 20
                }
            }
        }], { ...PL, margin: { l: 0, r: 0, t: 0, b: 0 } }, { responsive: true });
    }

    // ═══════════════════════════════════════════════════════════════
    //  EQUIPMENT TAB
    // ═══════════════════════════════════════════════════════════════

    function categoryBadge(cat) {
        const map = {
            'Stimulator Unit':        'badge-stimulator',
            'Stimulation Coil':       'badge-coil',
            'Neuronavigation System': 'badge-navigation',
            'EEG Monitoring':         'badge-eeg',
            'Positioning System':     'badge-positioning',
            'Cloud Infrastructure':   'badge-cloud',
        };
        return map[cat] || 'badge-default';
    }

    function barRow(label, value, max) {
        const pct = Math.min(100, Math.round((value / max) * 100));
        return `<div class="eq-op-bar-row">
            <span class="eq-op-bar-label">${label}</span>
            <div class="eq-op-bar-track"><div class="eq-op-bar-fill" style="width:${pct}%"></div></div>
            <span class="eq-op-value">${value}</span>
        </div>`;
    }

    async function loadEquipment() {
        try {
            const res     = await fetch('/api/equipment');
            const payload = await res.json();
            if (payload.status !== 'success') return;
            renderEquipmentCharts(payload.data);
            renderEquipmentCards(payload.data);
            equipmentLoaded = true;
        } catch (err) { console.error('Equipment error:', err); }
    }

    function renderEquipmentCharts(items) {
        const names = items.map(e => e.name.length > 22 ? e.name.slice(0, 20) + '…' : e.name);
        const oc    = items.map(e => e.operating_characteristics);

        Plotly.newPlot('eq-bar-chart', [
            { name: 'Efficiency (%)',    type: 'bar', x: names, y: oc.map(o => o.efficiency_pct),    marker: { color: '#58a6ff' } },
            { name: 'EMI Shielding (dB)',type: 'bar', x: names, y: oc.map(o => o.emi_shielding_db), marker: { color: '#8a2be2' } }
        ], { ...PL, barmode: 'group', legend: { font: { color: '#e6edf3' } }, xaxis: { ...PL.xaxis, tickangle: -25, tickfont: { size: 11 } } }, { responsive: true });

        Plotly.newPlot('eq-scatter-chart', [{
            x: oc.map(o => o.emi_shielding_db), y: oc.map(o => o.efficiency_pct), text: names,
            mode: 'markers+text', textposition: 'top center', textfont: { color: '#8b949e', size: 10 },
            marker: { size: oc.map(o => Math.sqrt(o.heat_dissipation_w) * 2), color: oc.map(o => o.efficiency_pct),
                colorscale: 'Viridis', showscale: true,
                colorbar: { title: 'Efficiency %', titlefont: { color: '#e6edf3' }, tickfont: { color: '#e6edf3' }, thickness: 15 } },
            type: 'scatter'
        }], { ...PL, xaxis: { ...PL.xaxis, title: 'EMI Shielding (dB)' }, yaxis: { ...PL.yaxis, title: 'Efficiency (%)' } }, { responsive: true });

        Plotly.newPlot('eq-heat-chart', [{
            type: 'bar', orientation: 'h',
            y: [...names].reverse(), x: [...oc].reverse().map(o => o.heat_dissipation_w),
            marker: { color: [...oc].reverse().map(o => o.heat_dissipation_w), colorscale: 'Inferno', showscale: false }
        }], { ...PL, xaxis: { ...PL.xaxis, title: 'Dissipation (W)' }, margin: { l: 180, r: 20, t: 20, b: 50 } }, { responsive: true });
    }

    function renderEquipmentCards(items) {
        equipmentList.innerHTML = '';
        items.forEach(eq => {
            const specRows = Object.entries(eq.specs).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
            const oc = eq.operating_characteristics;
            equipmentList.insertAdjacentHTML('beforeend', `
            <div class="eq-card">
                <div class="eq-card-header">
                    <div>
                        <div class="eq-card-id">${eq.id}</div>
                        <div class="eq-card-title">${eq.name}</div>
                    </div>
                    <span class="eq-badge ${categoryBadge(eq.category)}">${eq.category}</span>
                </div>
                <p class="eq-description">${eq.description}</p>
                <table class="eq-spec-table">${specRows}</table>
                <div>
                    ${barRow('Efficiency',     oc.efficiency_pct,    100)}
                    ${barRow('EMI Shield (dB)', oc.emi_shielding_db, 80)}
                    ${barRow('Op Temp (°C)',   oc.op_temp_c,          60)}
                    ${barRow('Max Temp (°C)',  oc.max_temp_c,         60)}
                </div>
            </div>`);
        });
    }

    // ═══════════════════════════════════════════════════════════════
    //  ESSENTIAL TREMOR TAB
    // ═══════════════════════════════════════════════════════════════

    async function loadTremorData() {
        try {
            const res     = await fetch('/api/tremor-clinical');
            const payload = await res.json();
            if (payload.status !== 'success') return;
            renderTremorTab(payload.data);
            tremorLoaded = true;
        } catch (err) { console.error('Tremor error:', err); }
    }

    function renderTremorTab(d) {
        renderTremorProtocolCard(d.recommended_protocol);
        renderTremorEvidenceChart(d.clinical_evidence);
        renderTremorSpectrumChart(d.tremor_spectrum);
        renderTremorReductionChart(d.session_outcomes);
        renderTremorTetrasChart(d.session_outcomes);
        renderVimChart(d.vim_target);
    }

    function renderTremorProtocolCard(p) {
        document.getElementById('tremor-protocol-card').innerHTML = `
        <table class="eq-spec-table" style="margin-top:8px;">
            <tr><td>Target Region</td><td>${p.target}</td></tr>
            <tr><td>Frequency</td><td>${p.frequency_hz} Hz (Inhibitory)</td></tr>
            <tr><td>Intensity</td><td>${p.intensity_mso}% MSO</td></tr>
            <tr><td>Pulses / Session</td><td>${p.pulses_session}</td></tr>
            <tr><td>Total Sessions</td><td>${p.sessions_total}</td></tr>
            <tr><td>Inter-Train Interval</td><td>${p.inter_train_s} s</td></tr>
            <tr><td>Coil Type</td><td>${p.coil_type}</td></tr>
            <tr><td>Pulse Type</td><td>${p.pulse_type}</td></tr>
        </table>`;
    }

    function renderTremorEvidenceChart(evidence) {
        const lc = { 'Level A': '#56d364', 'Level B': '#58a6ff', 'Level C': '#f1c40f' };
        Plotly.newPlot('tremor-evidence-chart', [{
            type: 'bar', orientation: 'h',
            y: evidence.map(e => e.region), x: evidence.map(e => e.pct),
            text: evidence.map(e => `${e.level} — ${e.pct}%`), textposition: 'outside',
            textfont: { color: '#e6edf3' }, marker: { color: evidence.map(e => lc[e.level] || '#8b949e') }
        }], { ...PL, xaxis: { ...PL.xaxis, title: 'Evidence Strength (%)', range: [0, 115] }, margin: { l: 220, r: 80, t: 20, b: 50 } }, { responsive: true });
    }

    function renderTremorSpectrumChart(spec) {
        Plotly.newPlot('tremor-spectrum-chart', [
            { x: spec.frequencies, y: spec.power, type: 'scatter', mode: 'lines',
              line: { color: '#e74c3c', width: 2.5 }, fill: 'tozeroy', fillcolor: 'rgba(231,76,60,0.15)', name: 'ET Power' },
            { x: [3, 3, 12, 12], y: [0, 3, 3, 0], type: 'scatter', mode: 'none',
              fill: 'toself', fillcolor: 'rgba(241,196,15,0.08)', name: 'Pathological Band (3–12 Hz)' }
        ], { ...PL, xaxis: { ...PL.xaxis, title: 'Frequency (Hz)' }, yaxis: { ...PL.yaxis, title: 'Power (a.u.)' }, legend: { font: { color: '#e6edf3' } } }, { responsive: true });
    }

    function renderTremorReductionChart(so) {
        Plotly.newPlot('tremor-reduction-chart', [{
            x: so.sessions, y: so.tremor_reduction, type: 'scatter', mode: 'lines+markers',
            line: { color: '#56d364', width: 3, shape: 'spline' }, marker: { size: 8, color: '#238636' },
            fill: 'tozeroy', fillcolor: 'rgba(86,211,100,0.1)'
        }], { ...PL, xaxis: { ...PL.xaxis, title: 'Session #', dtick: 1 }, yaxis: { ...PL.yaxis, title: 'Tremor Reduction (%)', range: [0, 80] } }, { responsive: true });
    }

    function renderTremorTetrasChart(so) {
        Plotly.newPlot('tremor-tetras-chart', [{
            x: so.sessions, y: so.tetras_scores, type: 'scatter', mode: 'lines+markers',
            line: { color: '#b06ef5', width: 3, shape: 'spline' }, marker: { size: 8, color: '#8a2be2' },
            fill: 'tozeroy', fillcolor: 'rgba(176,110,245,0.1)'
        }], { ...PL, xaxis: { ...PL.xaxis, title: 'Session #', dtick: 1 }, yaxis: { ...PL.yaxis, title: 'TETRAS Score' } }, { responsive: true });
    }

    function renderVimChart(vim) {
        Plotly.newPlot('tremor-vim-chart', [{
            x: vim.x, y: vim.y, z: vim.z, mode: 'markers', type: 'scatter3d',
            marker: { size: 6, color: vim.intensity, colorscale: 'Plasma', opacity: 0.85, showscale: true,
                colorbar: { title: 'Field Intensity', titlefont: { color: '#e6edf3' }, tickfont: { color: '#e6edf3' }, thickness: 18 } }
        }], { ...PL, margin: { l: 0, r: 0, t: 0, b: 0 },
            scene: { ...PL.scene, xaxis: { ...PL.scene.xaxis, title: 'x (MNI)' }, yaxis: { ...PL.scene.yaxis, title: 'y (MNI)' }, zaxis: { ...PL.scene.zaxis, title: 'z (MNI)' } }
        }, { responsive: true });
    }

    // ═══════════════════════════════════════════════════════════════
    //  TREATMENT PARADIGM TAB
    // ═══════════════════════════════════════════════════════════════

    // Paradigm condition sub-selector
    paradigmCondBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            paradigmCondBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadParadigm(btn.getAttribute('data-cond'));
        });
    });

    async function loadParadigm(condition) {
        if (paradigmCache[condition]) { renderParadigm(paradigmCache[condition]); return; }
        paradigmLoading && paradigmLoading.classList.remove('hidden');
        try {
            const res     = await fetch(`/api/treatment-paradigm?condition=${condition}`);
            const payload = await res.json();
            if (payload.status !== 'success') return;
            paradigmCache[condition] = payload.data;
            renderParadigm(payload.data);
        } catch (e) { console.error('Paradigm error:', e); }
        finally { paradigmLoading && paradigmLoading.classList.add('hidden'); }
    }

    function renderParadigm(d) {
        renderStageGate(d.stage_gates);
        renderHebbian(d.hebbian_dbs);
        renderDbsBurst(d.hebbian_dbs);
        renderCF(d.cf_convergents);
        renderGateEvents(d.stage_gates);
        renderDbsHardwareCard(d.dbs_hardware, d.stage_gates);
    }

    function renderStageGate(sg) {
        document.getElementById('paradigm-gate-subtitle').textContent =
            `Metric: ${sg.metric_name}  ·  DBS Target: ${sg.dbs_target}  ·  ${sg.dbs_freq_hz} Hz / ${sg.dbs_pw_us} µs`;

        const phaseColors = {
            'I — rTMS Induction':          '#58a6ff',
            'II — rTMS + DBS Integration': '#56d364',
            'III — DBS Maintenance':        '#f1c40f'
        };

        const traces = [{
            x: sg.sessions, y: sg.metric, type: 'scatter', mode: 'lines+markers',
            line: { color: '#58a6ff', width: 3, shape: 'spline' },
            marker: { size: 7, color: sg.phases.map(p => phaseColors[p] || '#58a6ff') },
            name: sg.metric_name, fill: 'tozeroy', fillcolor: 'rgba(88,166,255,0.06)'
        }];

        const tColors = ['#58a6ff', '#56d364', '#f1c40f'];
        sg.gate_thresholds.forEach((thr, i) => traces.push({
            x: [sg.sessions[0], sg.sessions[sg.sessions.length - 1]], y: [thr, thr],
            type: 'scatter', mode: 'lines',
            line: { color: tColors[i], width: 1.5, dash: 'dash' }, name: `Gate ${i + 1} (θ=${thr})`
        }));

        const phaseShades = ['rgba(88,166,255,0.06)', 'rgba(86,211,100,0.06)', 'rgba(241,196,15,0.06)'];
        const shapes = [[1, sg.N1], [sg.N1 + 1, sg.N2], [sg.N2 + 1, sg.N3]].map(([a, b], i) => ({
            type: 'rect', xref: 'x', yref: 'paper',
            x0: a - 0.5, x1: b + 0.5, y0: 0, y1: 1,
            fillcolor: phaseShades[i], line: { width: 0 }
        }));

        Plotly.newPlot('paradigm-gate-chart', traces, {
            ...PL, shapes,
            xaxis: { ...PL.xaxis, title: 'Session #', dtick: 1 },
            yaxis: { ...PL.yaxis, title: 'Outcome Metric (%)', range: [0, 105] },
            legend: { font: { color: '#e6edf3' } }
        }, { responsive: true });
    }

    function renderHebbian(h) {
        Plotly.newPlot('paradigm-hebbian-chart', [
            { x: h.sessions, y: h.weights, type: 'scatter', mode: 'lines+markers',
              line: { color: '#b06ef5', width: 3, shape: 'spline' }, marker: { size: 7, color: '#8a2be2' },
              fill: 'tozeroy', fillcolor: 'rgba(176,110,245,0.1)', name: 'Synaptic Weight w(t)' },
            { x: h.sessions, y: h.pre_rates, type: 'scatter', mode: 'lines',
              line: { color: '#58a6ff', width: 2, dash: 'dot' }, name: 'Pre-synaptic Rate' },
            { x: h.sessions, y: h.post_rates, type: 'scatter', mode: 'lines',
              line: { color: '#56d364', width: 2, dash: 'dot' }, name: 'Post-synaptic Rate' }
        ], { ...PL, xaxis: { ...PL.xaxis, title: 'Session #', dtick: 1 },
            yaxis: { ...PL.yaxis, title: 'Amplitude (normalized)' }, legend: { font: { color: '#e6edf3' } }
        }, { responsive: true });
    }

    function renderDbsBurst(h) {
        Plotly.newPlot('paradigm-dbs-chart', [{
            x: h.sessions, y: h.dbs_bursts, type: 'bar',
            marker: { color: h.dbs_bursts, colorscale: 'Viridis', showscale: false }, name: 'Mean DBS Burst'
        }], { ...PL, xaxis: { ...PL.xaxis, title: 'Session #', dtick: 1 }, yaxis: { ...PL.yaxis, title: 'Burst Amplitude (a.u.)' } }, { responsive: true });
    }

    function renderCF(cf) {
        Plotly.newPlot('paradigm-cf-chart', [
            { x: cf.map(c => c.iteration), y: cf.map(c => c.approx_freq),
              text: cf.map(c => `${c.numerator}/${c.denominator}`),
              type: 'scatter', mode: 'lines+markers+text', textposition: 'top center',
              textfont: { color: '#8b949e', size: 9 },
              line: { color: '#f1c40f', width: 2.5 }, marker: { size: 8, color: '#e67e22' },
              name: 'Convergent pₖ/qₖ (Hz)', yaxis: 'y' },
            { x: cf.map(c => c.iteration), y: cf.map(c => c.error_pct),
              type: 'bar', marker: { color: 'rgba(231,76,60,0.4)' }, name: 'Error (%)', yaxis: 'y2' }
        ], {
            ...PL, xaxis: { ...PL.xaxis, title: 'CF Depth k', dtick: 1 },
            yaxis:  { ...PL.yaxis, title: 'Approx Freq (Hz)' },
            yaxis2: { ...PL.yaxis, title: 'Error (%)', overlaying: 'y', side: 'right', showgrid: false },
            legend: { font: { color: '#e6edf3' } }
        }, { responsive: true });
    }

    function renderGateEvents(sg) {
        const names = ['I — rTMS Induction', 'II — rTMS + DBS', 'III — DBS Maintenance'];
        document.getElementById('paradigm-gate-events').innerHTML = sg.gate_events.map((g, i) => `
        <div class="gate-event-card">
            <span class="gate-event-pill gate-pill-${i + 1}">Phase ${g.phase}</span>
            <div class="gate-event-text">
                Gate <strong>θ = ${g.threshold}%</strong> crossed at <strong>Session ${g.session_crossed ?? 'N/A'}</strong>
                <br><span style="font-size:11px;">${names[i]}</span>
            </div>
        </div>`).join('');
    }

    function renderDbsHardwareCard(hw, sg) {
        const el = document.getElementById('paradigm-dbs-card');
        if (!hw || !hw.device) { el.innerHTML = '<p style="color:#8b949e;font-size:13px;">No DBS hardware data.</p>'; return; }
        el.innerHTML = `
        <table class="eq-spec-table" style="margin-top:8px;">
            <tr><td>Device</td><td>${hw.device}</td></tr>
            <tr><td>Lead</td><td>${hw.lead}</td></tr>
            <tr><td>Target</td><td>${hw.target}</td></tr>
            <tr><td>Voltage</td><td>${hw.voltage_v} V</td></tr>
            <tr><td>Impedance</td><td>${hw.impedance_ohm} Ω</td></tr>
            <tr><td>Frequency</td><td>${sg.dbs_freq_hz} Hz</td></tr>
            <tr><td>Pulse Width</td><td>${sg.dbs_pw_us} µs</td></tr>
            <tr><td>DBS Target</td><td>${sg.dbs_target}</td></tr>
        </table>`;
    }

});
