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
    const ocdView         = document.getElementById('ocd-view');
    const jcView          = document.getElementById('jc-view');
    const paradigmView    = document.getElementById('paradigm-view');
    const equipmentView   = document.getElementById('equipment-view');
    const dementiaLtView  = document.getElementById('dementia-lt-view');
    const dementiaDbsView = document.getElementById('dementia-dbs-view');
    const sleepapneaView  = document.getElementById('sleepapnea-view');
    const depressionView  = document.getElementById('depression-view');
    const anxietyView     = document.getElementById('anxiety-view');
    const moduliBemView   = document.getElementById('moduli-bem-view');
    const touretteView    = document.getElementById('tourette-view');
    const nashGeodesicView = document.getElementById('nash-geodesic-view');
    const tbiPtsdView      = document.getElementById('tbi-ptsd-view');

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
    let dementiaLtLoaded = false;
    let dementiaDbsLoaded = false;
    let sleepapneaLoaded = false;
    let jcLoaded         = false;
    let nashGeodesicLoaded = false;
    let paradigmCache    = {};

    // ── All views array for easy hide-all ────────────────────────
    const allViews = [simulationView, tremorView, ocdView, jcView, paradigmView, equipmentView, dementiaLtView, dementiaDbsView, sleepapneaView, depressionView, anxietyView, moduliBemView, touretteView, nashGeodesicView, tbiPtsdView];

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
        ocd: {
            title:    'OCD Treatment Clinical Care',
            subtitle: 'Deep Continued Fractions rTMS & Cortical Surface Deep FEA',
            view:     ocdView, showRunBtn: false
        },
        jaynes: {
            title:    'Jaynes-Cummings rTMS Predictions',
            subtitle: 'Quantum-neural excitation forecasts and treatment paradigm characteristics',
            view:     jcView, showRunBtn: false
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
        },
        'dementia-lt': {
            title:    'Long-Term Dementia Care — Smart Aging',
            subtitle: 'Cortical Surface Geodesics · Boundary Element Simulation · Protocol Optimization',
            view:     dementiaLtView, showRunBtn: false
        },
        'dementia-dbs': {
            title:    'Dementia DBS Treatment Protocol',
            subtitle: 'Statistical Manifold Distributions · Optimal Stage Gating',
            view:     dementiaDbsView, showRunBtn: false
        },
        'sleepapnea': {
            title:    'Sleep Apnea Neuromodulation Care',
            subtitle: 'Adaptive Closed-loop rTMS & Statistical Continued Fraction Phase Synchronization',
            view:     sleepapneaView, showRunBtn: false
        },
        'depression': {
            title:    'Depression rTMS + CBT Research Studio',
            subtitle: 'Statistical distributions · finite optimal control · cognitive state theory · number signatures',
            view:     depressionView, showRunBtn: false
        },
        'anxiety': {
            title:    'Millennial Anxiety rTMS & Pharmacological Optimization',
            subtitle: 'BEM/FEA Cortical Surfaces · EEG Waveforms (FAA) · Multi-Arm Trials · Long-Term Markov Horizon',
            view:     anxietyView, showRunBtn: false
        },
        'moduli-bem': {
            title:    'Moduli-Theoretic Treatment Paradigm & BEM Simulation',
            subtitle: 'SL(2,Z) Fundamental Domain · Elliptic Resonance Points · Single-Layer BEM Cortical Heat Maps',
            view:     moduliBemView, showRunBtn: false
        },
        'tourette': {
            title:    'Tourette Syndrome Combinatorial rTMS Treatment Paradigm',
            subtitle: 'Discrete CSTC Knapsack Pulse Allocation · 1Hz LTD pre-SMA · PUTS Urge Quenching · Permutation Entropy',
            view:     touretteView, showRunBtn: false
        },
        'nash-geodesic': {
            title:    'Nash / Geodesic Registration',
            subtitle: 'Laser-MRI-CT Registration · Eigen Spectra · Cauchy-Schwarz Convergence Bounds',
            view:     nashGeodesicView, showRunBtn: false
        },
        'tbi-ptsd': {
            title:    'TBI & PTSD rTMS Neuromodulation & Enterprise Healthcare Economics',
            subtitle: 'BEM Electric Field Analysis · Longitudinal PCL-5 / RPQ Trajectories · 5-Year Revenue Projections',
            view:     tbiPtsdView, showRunBtn: false
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
            if (tab === 'dementia-lt' && !dementiaLtLoaded)     loadDementiaLt();
            if (tab === 'dementia-dbs' && !dementiaDbsLoaded)   loadDementiaDbs();
            if (tab === 'ocd')                                  loadOcdData();
            if (tab === 'jaynes' && !jcLoaded)                  loadJaynesCummingsData();
            if (tab === 'sleepapnea')                           runSleepApneaRtms();
            if (tab === 'depression')                           runDepressionRtms();
            if (tab === 'anxiety')                              runAnxietyRtms();
            if (tab === 'moduli-bem')                            runModuliBemParadigm();
            if (tab === 'tourette')                             runTouretteRtms();
            if (tab === 'nash-geodesic' && !nashGeodesicLoaded)   loadNashGeodesicRegistration();
            if (tab === 'tbi-ptsd')                              runTbiPtsdRtms();
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

    async function loadJaynesCummingsData() {
        try {
            const [jcRes, paradigmRes] = await Promise.all([
                fetch('/api/jaynes-cummings?omega_c=20&omega_a=20&g=0.75&n_photons=4'),
                fetch('/api/treatment-paradigm?condition=stroke')
            ]);
            const jcPayload = await jcRes.json();
            const paradigmPayload = await paradigmRes.json();
            if (jcPayload.status === 'success' && paradigmPayload.status === 'success') {
                renderJaynesCummingsTab(jcPayload.data, paradigmPayload.data);
                jcLoaded = true;
            }
        } catch (err) {
            console.error('Jaynes-Cummings tab error:', err);
        }
    }

    function renderJaynesCummingsTab(data, paradigmData) {
        const stage = paradigmData.stage_gates || {};
        Plotly.newPlot('jc-excitation-chart', [
            { x: data.time, y: data.p_excited, type: 'scatter', mode: 'lines', line: { color: '#58a6ff', width: 3 }, name: 'Excited State P_e(t)' },
            { x: data.time, y: data.p_ground, type: 'scatter', mode: 'lines', line: { color: '#8b949e', width: 2, dash: 'dot' }, name: 'Ground State P_g(t)' },
            { x: data.time, y: data.sigma_z, type: 'scatter', mode: 'lines', line: { color: '#b06ef5', width: 2 }, name: 'Inversion <σ_z>' }
        ], { ...PL, xaxis: { ...PL.xaxis, title: 'Time (a.u.)' }, yaxis: { ...PL.yaxis, title: 'Probability / inversion' }, legend: { font: { color: '#e6edf3' }, orientation: 'h', x: 0, y: 1.15 } }, { responsive: true });

        const weights = data.combinatorial_weights || [];
        const weightLabels = weights.map(w => `State ${w.state}`);
        Plotly.newPlot('jc-weights-chart', [{
            x: weightLabels,
            y: weights.map(w => w.weight),
            type: 'bar',
            text: weights.map(w => w.weight.toFixed(4)),
            textposition: 'outside',
            hovertemplate: 'State %{customdata.state}<br>Weight: %{y:.4f}<br>C(n,k): %{customdata.coeff}<extra></extra>',
            customdata: weights.map(w => ({ state: w.state, coeff: w.binomial_coefficient })),
            marker: { color: weights.map((_, i) => ['#58a6ff', '#b06ef5', '#56d364', '#f1c40f', '#f85149', '#79c0ff'][i % 6]) },
            name: 'Binomial weight'
        }], {
            ...PL,
            xaxis: { ...PL.xaxis, title: 'Photon state', type: 'category', tickmode: 'array', tickvals: weightLabels, ticktext: weightLabels },
            yaxis: { ...PL.yaxis, title: 'Weight' }
        }, { responsive: true });

        const spectrum = data.vacuum_rabi_spectrum || {};
        Plotly.newPlot('jc-spectrum-chart', [{
            x: spectrum.freq_GHz || [], y: spectrum.intensity || [], type: 'scatter', mode: 'lines+markers',
            line: { color: '#56d364', width: 2.5 }, marker: { size: 6, color: '#238636' }, name: 'Vacuum Rabi Spectrum'
        }], { ...PL, xaxis: { ...PL.xaxis, title: 'Frequency (GHz)' }, yaxis: { ...PL.yaxis, title: 'Relative intensity' } }, { responsive: true });

        document.getElementById('jc-characteristics-card').innerHTML = `
            <table class="eq-spec-table" style="margin-top:8px;">
                <tr><td>Model</td><td>${data.model}</td></tr>
                <tr><td>Omega C</td><td>${data.omega_c.toFixed(2)} GHz</td></tr>
                <tr><td>Omega A</td><td>${data.omega_a.toFixed(2)} GHz</td></tr>
                <tr><td>Coupling</td><td>${data.coupling_g.toFixed(3)}</td></tr>
                <tr><td>Photon States</td><td>${data.n_photons}</td></tr>
                <tr><td>Rabi Frequency</td><td>${data.rabi_freq_MHz} MHz</td></tr>
                <tr><td>Resonance Shift</td><td>${data.neural_analogy.resonance_shift}</td></tr>
                <tr><td>Coherence Index</td><td>${data.neural_analogy.coherence_index}</td></tr>
            </table>`;

        document.getElementById('jc-paradigm-card').innerHTML = `
            <table class="eq-spec-table" style="margin-top:8px;">
                <tr><td>Condition</td><td>${paradigmData.condition}</td></tr>
                <tr><td>Metric</td><td>${stage.metric_name || 'Outcome metric'}</td></tr>
                <tr><td>DBS Target</td><td>${stage.dbs_target || 'N/A'}</td></tr>
                <tr><td>DBS Frequency</td><td>${stage.dbs_freq_hz || 'N/A'} Hz</td></tr>
                <tr><td>Gate Thresholds</td><td>${(stage.gate_thresholds || []).join(', ')}</td></tr>
                <tr><td>Phase Crossings</td><td>${(stage.gate_events || []).map(g => `θ${g.threshold}@${g.session_crossed}`).join(' · ')}</td></tr>
            </table>`;
    }

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
            const data = payload.data;
            renderEquipmentCharts(data.equipment || data);
            renderEquipmentCards(data.equipment || data);
            if (data.global_optima_convergence) {
                renderGlobalOptimaConvergenceChart('eq-convergence-chart', data.global_optima_convergence, 'equipment');
            }
            if (data.recommended_protocol) {
                renderRecommendedCard('eq-recommended-card', data.recommended_protocol);
            }
            equipmentLoaded = true;
        } catch (err) { console.error('Equipment error:', err); }
    }


    function renderEquipmentCharts(items) {
        const names = items.map(e => e.name.length > 22 ? e.name.slice(0, 20) + '…' : e.name);
        const oc    = items.map(e => e.operating_characteristics);

        Plotly.newPlot('eq-bar-chart', [
            { name: 'Efficiency',       type: 'bar', x: names, y: oc.map(o => o.efficiency_pct),                  customdata: oc.map(o => o.efficiency_pct + '%'),      hovertemplate: '%{x}<br>Efficiency: %{customdata}<extra></extra>', marker: { color: '#58a6ff' } },
            { name: 'EMI Shielding',    type: 'bar', x: names, y: oc.map(o => (o.emi_shielding_db / 80) * 100),   customdata: oc.map(o => o.emi_shielding_db + ' dB'), hovertemplate: '%{x}<br>EMI Shielding: %{customdata}<extra></extra>', marker: { color: '#8a2be2' } },
            { name: 'Op Temp',          type: 'bar', x: names, y: oc.map(o => (o.op_temp_c / 60) * 100),          customdata: oc.map(o => o.op_temp_c + ' °C'),         hovertemplate: '%{x}<br>Op Temp: %{customdata}<extra></extra>', marker: { color: '#2ea043' } },
            { name: 'Max Temp',         type: 'bar', x: names, y: oc.map(o => (o.max_temp_c / 60) * 100),         customdata: oc.map(o => o.max_temp_c + ' °C'),        hovertemplate: '%{x}<br>Max Temp: %{customdata}<extra></extra>', marker: { color: '#f85149' } },
            { name: 'Heat Dissipation', type: 'bar', x: names, y: oc.map(o => (o.heat_dissipation_w / 900) * 100),customdata: oc.map(o => o.heat_dissipation_w + ' W'), hovertemplate: '%{x}<br>Heat Dissip: %{customdata}<extra></extra>', marker: { color: '#db6d28' } }
        ], { ...PL, barmode: 'group', legend: { font: { color: '#e6edf3' } },
             xaxis: { ...PL.xaxis, tickangle: -25, tickfont: { size: 11 } },
             yaxis: { ...PL.yaxis, title: 'Relative Scale (%)', range: [0, 100] }
           }, { responsive: true });

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
                    ${barRow('Heat Dissip. (W)', oc.heat_dissipation_w, 900)}
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

    // ═════════════════════════════════════════════════════════════════
    //  LONG-TERM DEMENTIA CARE TAB
    // ═════════════════════════════════════════════════════════════════

    async function loadDementiaLt() {
        try {
            const res = await fetch('/api/dementia-longterm');
            const payload = await res.json();
            if (payload.status !== 'success') return;
            renderDementiaLtTab(payload.data);
            dementiaLtLoaded = true;
        } catch (err) { console.error('Dementia LT error:', err); }
    }

    function renderDementiaLtTab(d) {
        renderCognitiveTracking(d.cognitive_tracking);
        renderBiomarkers(d.biomarkers);
        renderCorticalGeodesics(d.geodesics);
        renderGeodesicTable(d.geodesics.geodesics);
        renderDltBem(d.bem_simulation);
        renderAttenuation(d.bem_simulation.attenuation);
        renderGlobalOptimaConvergenceChart('dlt-optimization-chart', d.global_optima_convergence, 'dementia');
        renderProtocolCards(d.protocols);
        if (d.jaynes_cummings) {
            renderJaynesCummingsPlot('dlt-jc-chart', d.jaynes_cummings);
        }
        if (d.recommended_protocol) {
            renderRecommendedCard('dlt-recommended-card', d.recommended_protocol);
        }
    }


    function renderCognitiveTracking(ct) {
        Plotly.newPlot('dlt-cognitive-chart', [
            { x: ct.months, y: ct.adas_cog, type: 'scatter', mode: 'lines+markers',
              line: { color: '#e74c3c', width: 3, shape: 'spline' }, marker: { size: 6 },
              name: 'ADAS-Cog (lower = better)', fill: 'tozeroy', fillcolor: 'rgba(231,76,60,0.06)' },
            { x: ct.months, y: ct.mmse, type: 'scatter', mode: 'lines+markers',
              line: { color: '#58a6ff', width: 3, shape: 'spline' }, marker: { size: 6 },
              name: 'MMSE (higher = better)' },
            { x: ct.months, y: ct.moca, type: 'scatter', mode: 'lines+markers',
              line: { color: '#56d364', width: 3, shape: 'spline' }, marker: { size: 6 },
              name: 'MoCA (higher = better)' }
        ], {
            ...PL,
            xaxis: { ...PL.xaxis, title: 'Month' },
            yaxis: { ...PL.yaxis, title: 'Score' },
            legend: { font: { color: '#e6edf3' } }
        }, { responsive: true });
    }

    function renderBiomarkers(bm) {
        Plotly.newPlot('dlt-biomarker-chart', [
            { x: bm.months, y: bm.amyloid_pet_suvr, type: 'scatter', mode: 'lines+markers',
              line: { color: '#f1c40f', width: 2.5 }, name: 'Amyloid PET (SUVR)', yaxis: 'y' },
            { x: bm.months, y: bm.tau_pet_suvr, type: 'scatter', mode: 'lines+markers',
              line: { color: '#e74c3c', width: 2.5, dash: 'dot' }, name: 'Tau PET (SUVR)', yaxis: 'y' },
            { x: bm.months, y: bm.hippocampal_volume_ml, type: 'scatter', mode: 'lines+markers',
              line: { color: '#56d364', width: 2.5 }, name: 'Hippocampal Vol (mL)', yaxis: 'y2' },
            { x: bm.months, y: bm.cortical_thickness_mm, type: 'scatter', mode: 'lines+markers',
              line: { color: '#b06ef5', width: 2.5, dash: 'dash' }, name: 'Cortical Thickness (mm)', yaxis: 'y2' }
        ], {
            ...PL,
            xaxis: { ...PL.xaxis, title: 'Month' },
            yaxis:  { ...PL.yaxis, title: 'PET SUVR', side: 'left' },
            yaxis2: { ...PL.yaxis, title: 'Volume / Thickness', overlaying: 'y', side: 'right', showgrid: false },
            legend: { font: { color: '#e6edf3' }, x: 0, y: -0.3, orientation: 'h' }
        }, { responsive: true });
    }

    function renderCorticalGeodesics(geo) {
        const traces = [];
        // Surface mesh
        traces.push({
            x: geo.surface.x.flat(), y: geo.surface.y.flat(), z: geo.surface.z.flat(),
            type: 'mesh3d', opacity: 0.15, color: '#58a6ff', name: 'Cortical Surface'
        });
        // Geodesic paths
        const colors = ['#e74c3c', '#58a6ff', '#56d364', '#f1c40f', '#b06ef5', '#e67e22',
                        '#1abc9c', '#e91e63', '#00bcd4', '#ff9800', '#9c27b0', '#8bc34a'];
        geo.geodesics.forEach((g, i) => {
            traces.push({
                x: g.x, y: g.y, z: g.z, type: 'scatter3d', mode: 'lines',
                line: { color: colors[i % colors.length], width: 6 },
                name: `${g.from_roi} → ${g.to_roi} (d=${g.arc_length.toFixed(2)})`
            });
        });
        Plotly.newPlot('dlt-geodesic-chart', traces, {
            ...PL, margin: { l: 0, r: 0, t: 0, b: 0 },
            scene: {
                ...PL.scene,
                xaxis: { ...PL.scene.xaxis, title: 'x' },
                yaxis: { ...PL.scene.yaxis, title: 'y' },
                zaxis: { ...PL.scene.zaxis, title: 'z' },
                camera: { eye: { x: 1.5, y: 1.5, z: 0.8 } }
            },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 1, y: 1, bgcolor: 'rgba(0,0,0,0.3)' },
            showlegend: true
        }, { responsive: true });
    }

    function renderGeodesicTable(geodesics) {
        const rows = geodesics.map(g => `
            <tr>
                <td style="color:#58a6ff;">${g.from_roi}</td>
                <td style="color:#56d364;">${g.to_roi}</td>
                <td style="color:#f1c40f;font-weight:600;">${g.arc_length.toFixed(3)}</td>
            </tr>
        `).join('');
        document.getElementById('dlt-geodesic-table').innerHTML = `
            <table class="eq-spec-table" style="margin-top:8px;">
                <thead><tr><th style="color:#8b949e;">From ROI</th><th style="color:#8b949e;">To ROI</th><th style="color:#8b949e;">Geodesic Distance</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>`;
    }

    function renderDltBem(bem) {
        const traces = [];
        const layerColors = ['#e67e22', '#8b949e', '#58a6ff', '#e74c3c'];
        bem.layers.forEach((layer, i) => {
            traces.push({
                x: layer.x, y: layer.y, z: layer.z, surfacecolor: layer.potential,
                type: 'surface', opacity: 0.15 + i * 0.15,
                colorscale: 'Jet', showscale: i === bem.layers.length - 1,
                colorbar: { title: 'Atten. Potential', thickness: 15, titlefont: { color: '#e6edf3' }, tickfont: { color: '#e6edf3' } },
                name: layer.name
            });
        });
        Plotly.newPlot('dlt-bem-chart', traces, {
            ...PL, margin: { l: 0, r: 0, t: 0, b: 0 },
            scene: {
                ...PL.scene,
                xaxis: { ...PL.scene.xaxis, title: 'x (m)' },
                yaxis: { ...PL.scene.yaxis, title: 'y (m)' },
                zaxis: { ...PL.scene.zaxis, title: 'z (m)' },
                camera: { eye: { x: 1.8, y: 0.8, z: 0.6 } }
            },
            legend: { font: { color: '#e6edf3' } },
            showlegend: true
        }, { responsive: true });
    }

    function renderAttenuation(att) {
        Plotly.newPlot('dlt-attenuation-chart', [{
            x: att.depths, y: att.field_pct, type: 'scatter', mode: 'lines',
            line: { color: '#e74c3c', width: 3, shape: 'spline' },
            fill: 'tozeroy', fillcolor: 'rgba(231,76,60,0.1)', name: 'E-Field (%)'
        }], {
            ...PL,
            xaxis: { ...PL.xaxis, title: 'Normalised Depth (scalp → cortex)' },
            yaxis: { ...PL.yaxis, title: 'Residual E-Field (%)', range: [0, 110] },
            shapes: [
                { type: 'line', x0: 0.08, x1: 0.08, y0: 0, y1: 110, line: { color: '#8b949e', dash: 'dot', width: 1 } },
                { type: 'line', x0: 0.13, x1: 0.13, y0: 0, y1: 110, line: { color: '#8b949e', dash: 'dot', width: 1 } },
                { type: 'line', x0: 0.20, x1: 0.20, y0: 0, y1: 110, line: { color: '#8b949e', dash: 'dot', width: 1 } }
            ],
            annotations: [
                { x: 0.04, y: 105, text: 'Scalp', showarrow: false, font: { color: '#e67e22', size: 10 } },
                { x: 0.105, y: 105, text: 'Skull', showarrow: false, font: { color: '#8b949e', size: 10 } },
                { x: 0.165, y: 105, text: 'CSF', showarrow: false, font: { color: '#58a6ff', size: 10 } },
                { x: 0.30, y: 105, text: 'Grey Matter', showarrow: false, font: { color: '#e74c3c', size: 10 } }
            ]
        }, { responsive: true });
    }

    function renderProtocolCards(protocols) {
        const stageColors = ['#58a6ff', '#56d364', '#f1c40f', '#b06ef5'];
        document.getElementById('dlt-protocol-cards').innerHTML = protocols.map((p, i) => `
            <div style="border-left:4px solid ${stageColors[i]};padding:16px 20px;margin-bottom:16px;
                        background:rgba(255,255,255,0.03);border-radius:0 8px 8px 0;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                    <h4 style="color:${stageColors[i]};margin:0;font-size:15px;">${p.stage}</h4>
                    <span style="color:#8b949e;font-size:12px;">${p.duration}</span>
                </div>
                <table class="eq-spec-table" style="margin:0;">
                    <tr><td>Target</td><td>${p.target}</td></tr>
                    <tr><td>Frequency</td><td>${p.frequency_hz} Hz</td></tr>
                    <tr><td>Intensity</td><td>${p.intensity_mso}% MSO</td></tr>
                    <tr><td>Sessions/Week</td><td>${p.sessions_per_week}</td></tr>
                    <tr><td>Pulses/Session</td><td>${p.pulses_session}</td></tr>
                    <tr><td>Coil</td><td>${p.coil}</td></tr>
                    <tr><td>Adjunct Therapy</td><td>${p.adjunct}</td></tr>
                    <tr><td style="color:#f1c40f;">Biomarker Gate</td><td style="color:#f1c40f;">${p.biomarker_gate}</td></tr>
                </table>
            </div>
        `).join('');
    }

    // ── Dementia DBS Protocol Loader ──────────────────────────────
    async function loadDementiaDbs() {
        try {
            const res = await fetch('/api/dbs-imaging');
            if(!res.ok) throw new Error('API Error');
            const result = await res.json();
            const data = result.data;

            // 1. Manifold Heatmap
            Plotly.newPlot('dbs-manifold-chart', [{
                z: data.manifold.z,
                x: data.manifold.x,
                y: data.manifold.y,
                type: 'contour',
                colorscale: 'Magma',
                contours: { coloring: 'heatmap' }
            }], {
                ...PL,
                xaxis: { title: 'Manifold Coord x', color: '#8b949e', gridcolor: 'rgba(255,255,255,0.1)' },
                yaxis: { title: 'Manifold Coord y', color: '#8b949e', gridcolor: 'rgba(255,255,255,0.1)' }
            });

            // 2. Stage Gating Chart
            const gateTraces = [
                {
                    x: data.gating.sessions,
                    y: data.gating.patient_state,
                    mode: 'lines',
                    name: 'Patient State',
                    line: { color: '#00f2fe', width: 3 }
                }
            ];
            const colors = ['#f39c12', '#e74c3c', '#e67e22'];
            Object.keys(data.gating.stages).forEach((stage, idx) => {
                gateTraces.push({
                    x: data.gating.sessions,
                    y: data.gating.stages[stage],
                    mode: 'lines',
                    name: stage,
                    line: { color: colors[idx % colors.length], dash: 'dash', width: 2 }
                });
            });

            Plotly.newPlot('dbs-gating-chart', gateTraces, {
                ...PL,
                xaxis: { title: 'DBS Sessions', color: '#8b949e', gridcolor: 'rgba(255,255,255,0.1)' },
                yaxis: { title: 'Cognitive / Biomarker State', color: '#8b949e', gridcolor: 'rgba(255,255,255,0.1)' },
                legend: { orientation: 'h', y: 1.1 }
            });

            // 3. Timeline Chart
            Plotly.newPlot('dbs-timeline-chart', [
                {
                    x: data.dbs_timeline.sessions,
                    y: data.dbs_timeline.frequency,
                    mode: 'lines+markers',
                    name: 'Frequency (Hz)',
                    line: { color: '#ff6b6b' },
                    marker: { size: 4 }
                },
                {
                    x: data.dbs_timeline.sessions,
                    y: data.dbs_timeline.intensity,
                    mode: 'lines',
                    name: 'Intensity (% MSO)',
                    line: { color: '#4ecdc4', width: 3 },
                    yaxis: 'y2'
                }
            ], {
                ...PL,
                xaxis: { title: 'DBS Sessions', color: '#8b949e', gridcolor: 'rgba(255,255,255,0.1)' },
                yaxis: { title: 'Frequency (Hz)', titlefont: { color: '#ff6b6b' }, tickfont: { color: '#ff6b6b' }, gridcolor: 'rgba(255,255,255,0.1)' },
                yaxis2: {
                    title: 'Intensity (% MSO)',
                    titlefont: { color: '#4ecdc4' },
                    tickfont: { color: '#4ecdc4' },
                    overlaying: 'y',
                    side: 'right',
                    gridcolor: 'rgba(255,255,255,0.05)'
                },
                legend: { orientation: 'h', y: 1.1 }
            });

            dementiaDbsLoaded = true;
        } catch (e) {
            console.error('Failed to load Dementia DBS data:', e);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  NASH / GEODESIC REGISTRATION TAB
    // ═══════════════════════════════════════════════════════════════
    async function loadNashGeodesicRegistration() {
        try {
            const res     = await fetch('/api/nash-geodesic-registration');
            const payload = await res.json();
            if (payload.status !== 'success') return;
            renderNashGeodesicTab(payload.data);
            nashGeodesicLoaded = true;
        } catch (err) {
            console.error('Nash/Geodesic registration error:', err);
        }
    }

    function renderNashGeodesicTab(data) {
        const nash = data.nash;
        const eig  = data.eigen_spectra;
        const cs   = data.cauchy_schwarz;
        const geo  = data.geodesic;

        // Nash equilibrium utility convergence
        Plotly.newPlot('ng-nash-utility-chart', [
            { x: nash.iterations, y: nash.utility_laser, type: 'scatter', mode: 'lines', name: 'Laser Scan U', line: { color: '#58a6ff', width: 2.5 } },
            { x: nash.iterations, y: nash.utility_mri,   type: 'scatter', mode: 'lines', name: 'MRI U',        line: { color: '#b06ef5', width: 2.5 } },
            { x: nash.iterations, y: nash.utility_ct,    type: 'scatter', mode: 'lines', name: 'CT U',         line: { color: '#56d364', width: 2.5 } }
        ], { ...PL, xaxis: { ...PL.xaxis, title: 'Iteration' }, yaxis: { ...PL.yaxis, title: 'Utility (Alignment Fidelity)' },
             legend: { font: { color: '#e6edf3' }, orientation: 'h', x: 0, y: 1.15 } }, { responsive: true });

        // Equilibrium mixed-strategy density
        Plotly.newPlot('ng-nash-density-chart', [{
            x: nash.equilibrium_bins, y: nash.equilibrium_density, type: 'scatter', mode: 'lines',
            fill: 'tozeroy', line: { color: '#f1c40f', width: 2 }, name: 'Equilibrium Density'
        }], { ...PL, xaxis: { ...PL.xaxis, title: 'Transform Perturbation' }, yaxis: { ...PL.yaxis, title: 'Probability Density' } }, { responsive: true });

        // Laplace-Beltrami eigen spectra
        Plotly.newPlot('ng-eigen-chart', [
            { x: eig.modes, y: eig.eigenvalues_laser, type: 'scatter', mode: 'lines+markers', name: 'Laser Scan λ', line: { color: '#58a6ff' }, marker: { size: 5 } },
            { x: eig.modes, y: eig.eigenvalues_mri,   type: 'scatter', mode: 'lines+markers', name: 'MRI λ',        line: { color: '#b06ef5' }, marker: { size: 5 } },
            { x: eig.modes, y: eig.eigenvalues_ct,    type: 'scatter', mode: 'lines+markers', name: 'CT λ',         line: { color: '#56d364' }, marker: { size: 5 } }
        ], { ...PL, xaxis: { ...PL.xaxis, title: 'Mode index' }, yaxis: { ...PL.yaxis, title: 'Eigenvalue λ (rad/mm)' },
             legend: { font: { color: '#e6edf3' } } }, { responsive: true });

        // Nash gap decay
        Plotly.newPlot('ng-nash-gap-chart', [{
            x: nash.iterations, y: nash.nash_gap, type: 'scatter', mode: 'lines', fill: 'tozeroy',
            line: { color: '#f85149', width: 2.5 }, name: 'Nash Gap'
        }], { ...PL, xaxis: { ...PL.xaxis, title: 'Iteration' }, yaxis: { ...PL.yaxis, title: 'Max Unilateral Gain' } }, { responsive: true });

        // Cauchy-Schwarz bounds
        Plotly.newPlot('ng-cauchy-schwarz-chart', [
            { x: cs.iterations, y: cs.ratio_laser_mri, type: 'scatter', mode: 'lines', name: 'Laser↔MRI ratio', line: { color: '#58a6ff', width: 2.5 } },
            { x: cs.iterations, y: cs.ratio_mri_ct,    type: 'scatter', mode: 'lines', name: 'MRI↔CT ratio',    line: { color: '#b06ef5', width: 2.5 } },
            { x: cs.iterations, y: cs.ratio_laser_ct,  type: 'scatter', mode: 'lines', name: 'Laser↔CT ratio',  line: { color: '#56d364', width: 2.5 } },
            { x: cs.iterations, y: cs.upper_bound,     type: 'scatter', mode: 'lines', name: 'Cauchy-Schwarz upper bound (1.0)', line: { color: '#f85149', width: 1.5, dash: 'dash' } }
        ], { ...PL, xaxis: { ...PL.xaxis, title: 'Iteration' }, yaxis: { ...PL.yaxis, title: '|⟨f,g⟩|² / (⟨f,f⟩⟨g,g⟩)', range: [0, 1.05] },
             legend: { font: { color: '#e6edf3' }, orientation: 'h', x: 0, y: 1.15 } }, { responsive: true });

        // Geodesic mapping path
        Plotly.newPlot('ng-geodesic-chart', [
            { x: geo.x, y: geo.y, z: geo.z, type: 'scatter3d', mode: 'lines', line: { color: '#58a6ff', width: 5 }, name: 'Geodesic Path' },
            { x: geo.landmark_x, y: geo.landmark_y, z: geo.landmark_z, type: 'scatter3d', mode: 'markers', marker: { size: 5, color: '#f1c40f' }, name: 'Registered Landmarks' }
        ], { ...PL, margin: { l: 0, r: 0, t: 0, b: 0 }, legend: { font: { color: '#e6edf3' } } }, { responsive: true });

        document.getElementById('ng-fidelity-card').innerHTML = `
            <table class="eq-spec-table" style="margin-top:8px;">
                <tr><td>Max Geodesic Error</td><td>${geo.max_geodesic_error_mm.toFixed(2)} mm</td></tr>
                <tr><td>Mean Geodesic Error</td><td>${geo.mean_geodesic_error_mm.toFixed(2)} mm</td></tr>
                <tr><td>Final Nash Gap</td><td>${nash.nash_gap[nash.nash_gap.length - 1].toFixed(4)}</td></tr>
                <tr><td>Final Laser↔MRI C-S Ratio</td><td>${cs.ratio_laser_mri[cs.ratio_laser_mri.length - 1].toFixed(4)}</td></tr>
                <tr><td>Final MRI↔CT C-S Ratio</td><td>${cs.ratio_mri_ct[cs.ratio_mri_ct.length - 1].toFixed(4)}</td></tr>
                <tr><td>Final Laser↔CT C-S Ratio</td><td>${cs.ratio_laser_ct[cs.ratio_laser_ct.length - 1].toFixed(4)}</td></tr>
            </table>`;

        // Longitudinal registration tracking across clinical sessions
        const lt = data.longitudinal;
        if (lt) {
            Plotly.newPlot('ng-longitudinal-error-chart', [
                { x: lt.sessions, y: lt.mean_error_mm, type: 'scatter', mode: 'lines+markers', name: 'Mean Error (mm)', line: { color: '#58a6ff', width: 2.5 }, marker: { size: 6 } },
                { x: lt.sessions, y: lt.max_error_mm,  type: 'scatter', mode: 'lines+markers', name: 'Max Error (mm)',  line: { color: '#f1c40f', width: 2 }, marker: { size: 5 } },
                { x: lt.sessions, y: lt.clinical_tolerance_mm, type: 'scatter', mode: 'lines', name: 'Clinical Tolerance (1.0 mm)', line: { color: '#f85149', width: 1.5, dash: 'dash' } }
            ], { ...PL, xaxis: { ...PL.xaxis, title: 'Clinical Session' }, yaxis: { ...PL.yaxis, title: 'Registration Error (mm)' },
                 legend: { font: { color: '#e6edf3' }, orientation: 'h', x: 0, y: 1.15 } }, { responsive: true });

            Plotly.newPlot('ng-longitudinal-quality-chart', [
                { x: lt.sessions, y: lt.nash_gap_final, type: 'bar', name: 'Final Nash Gap', marker: { color: '#f85149' }, yaxis: 'y' },
                { x: lt.sessions, y: lt.cauchy_schwarz_final, type: 'scatter', mode: 'lines+markers', name: 'Final C-S Ratio', line: { color: '#56d364', width: 2.5 }, marker: { size: 6 }, yaxis: 'y2' }
            ], { ...PL, xaxis: { ...PL.xaxis, title: 'Clinical Session' },
                 yaxis: { ...PL.yaxis, title: 'Nash Gap' },
                 yaxis2: { title: 'Cauchy-Schwarz Ratio', overlaying: 'y', side: 'right', range: [0, 1.05], gridcolor: 'rgba(255,255,255,0.05)' },
                 legend: { font: { color: '#e6edf3' }, orientation: 'h', x: 0, y: 1.15 } }, { responsive: true });
        }
    }

});

    // ── OCD Data Loader ──────────────────────────────────────────
    async function loadOcdData() {
        try {
            const r = await fetch('/api/ocd-treatment');
            const res = await r.json();
            const data = res.data;

            // 1. Continued Fractions Series
            const depths = Array.from({length: data.continued_fractions.length}, (_, i) => i+1);
            const freq_y = data.continued_fractions.map(c => c.approx_freq);
            const err_y  = data.continued_fractions.map(c => c.error_pct);
            
            Plotly.newPlot('ocd-cf-chart', [
                {
                    x: depths,
                    y: freq_y,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { color: '#00ffcc', width: 3 },
                    marker: { size: 8, color: '#ff00ff' },
                    name: 'Convergent Freq (Hz)'
                },
                {
                    x: depths,
                    y: err_y,
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#ff4444', width: 2, dash: 'dot' },
                    name: 'Error (%)',
                    yaxis: 'y2'
                }
            ], {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: { t: 10, b: 40, l: 40, r: 40 },
                xaxis: { title: 'Continued Fraction Depth (k)', gridcolor: 'rgba(255,255,255,0.05)', color: '#8b949e' },
                yaxis: { title: 'Tuned Frequency (Hz)', gridcolor: 'rgba(255,255,255,0.05)', color: '#8b949e', titlefont: {color: '#00ffcc'}, tickfont: {color: '#00ffcc'} },
                yaxis2: {
                    title: 'Error %',
                    titlefont: {color: '#ff4444'},
                    tickfont: {color: '#ff4444'},
                    overlaying: 'y',
                    side: 'right',
                    gridcolor: 'transparent'
                },
                font: { color: '#c9d1d9' },
                legend: {x: 0, y: 1.1, orientation: 'h'}
            });

            // 2. Cortical Surface FEA Heatmap
            Plotly.newPlot('ocd-fea-surface', [{
                x: data.fea_surface.x,
                y: data.fea_surface.y,
                z: data.fea_surface.z,
                type: 'surface',
                colorscale: 'Jet'
            }], {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: { t: 0, b: 0, l: 0, r: 0 },
                scene: {
                    xaxis: { title: 'X (cm)', color: '#8b949e', gridcolor: 'rgba(255,255,255,0.1)' },
                    yaxis: { title: 'Y (cm)', color: '#8b949e', gridcolor: 'rgba(255,255,255,0.1)' },
                    zaxis: { title: 'E-Field (V/m)', color: '#8b949e', gridcolor: 'rgba(255,255,255,0.1)' },
                    camera: { eye: {x: 1.5, y: -1.5, z: 1.2} }
                }
            });

            // 3. Jaynes-Cummings Quantum Rabi Excitations
            if (data.jaynes_cummings) {
                renderJaynesCummingsPlot('ocd-jc-chart', data.jaynes_cummings);
            }

            // 4. Quantum VQC Global Optima Convergence
            if (data.global_optima_convergence) {
                renderGlobalOptimaConvergenceChart('ocd-convergence-chart', data.global_optima_convergence, 'ocd');
            }

            // 5. Y-BOCS Trajectory Chart
            if (data.ybocs_trajectory) {
                Plotly.newPlot('ocd-ybocs-chart', [{
                    x: data.ybocs_trajectory.sessions,
                    y: data.ybocs_trajectory.ybocs_scores,
                    type: 'scatter', mode: 'lines+markers',
                    line: { color: '#ff7b72', width: 3, shape: 'spline' },
                    marker: { size: 7, color: '#da3633' },
                    fill: 'tozeroy', fillcolor: 'rgba(255,123,114,0.1)',
                    name: 'Y-BOCS Score'
                }], {
                    ...PL,
                    xaxis: { ...PL.xaxis, title: 'Treatment Session #' },
                    yaxis: { ...PL.yaxis, title: 'Y-BOCS Severity Score', range: [0, 40] }
                }, { responsive: true });
            }

            // 5b. 6-Month longitudinal recovery plot
            if (data.six_month_recovery) {
                Plotly.newPlot('ocd-six-month-recovery-chart', [
                    {
                        x: data.six_month_recovery.days,
                        y: data.six_month_recovery.scores,
                        type: 'scatter', mode: 'lines',
                        line: { color: '#56d364', width: 3.5, shape: 'spline' },
                        fill: 'tozeroy', fillcolor: 'rgba(86,211,100,0.06)',
                        name: 'Longitudinal Recovery'
                    },
                    {
                        x: [1, 29, 90, 180],
                        y: [34, 12, 8, 6],
                        type: 'scatter', mode: 'markers',
                        marker: { size: 10, color: '#f85149', symbol: 'diamond' },
                        name: 'Clinical Assessment Gates'
                    }
                ], {
                    ...PL,
                    xaxis: { ...PL.xaxis, title: 'Longitudinal Treatment (Days)' },
                    yaxis: { ...PL.yaxis, title: 'Y-BOCS Score / Severity Threshold', range: [0, 40] },
                    legend: { font: { color: '#cbd5e1' }, orientation: 'h', x: 0, y: 1.12 }
                }, { responsive: true });
            }

            // 6. Recommended Protocol Card
            if (data.recommended_protocol) {
                renderRecommendedCard('ocd-recommended-card', data.recommended_protocol);
            }

        } catch(e) {
            console.error('Error loading OCD data:', e);
        }
    }

    function renderJaynesCummingsPlot(containerId, jc) {
        if (!jc || !document.getElementById(containerId)) return;
        Plotly.newPlot(containerId, [
            {
                x: jc.time, y: jc.p_excited,
                type: 'scatter', mode: 'lines',
                line: { color: '#00f2fe', width: 3 },
                name: 'Neural Excitation P_e(t)'
            },
            {
                x: jc.time, y: jc.p_ground,
                type: 'scatter', mode: 'lines',
                line: { color: '#8b949e', width: 2, dash: 'dot' },
                name: 'Resting State P_g(t)'
            },
            {
                x: jc.time, y: jc.sigma_z,
                type: 'scatter', mode: 'lines',
                line: { color: '#b06ef5', width: 2 },
                name: 'Inversion <σ_z(t)>'
            }
        ], {
            ...PL,
            xaxis: { ...PL.xaxis, title: 'Pulse Duration / Time (t)' },
            yaxis: { ...PL.yaxis, title: 'Quantum Probability / Inversion', range: [-1.1, 1.1] },
            legend: { font: { color: '#e6edf3' }, orientation: 'h', x: 0, y: 1.15 }
        }, { responsive: true });
    }

    function renderGlobalOptimaConvergenceChart(containerId, conv, type) {
        if (!conv || !document.getElementById(containerId)) return;
        const iterations = conv.map(c => c.iteration || c.step);
        const qLoss = conv.map(c => c.quantum_vqe_loss || c.qpu_gate_fidelity_pct);
        const cLoss = conv.map(c => c.classical_sgd_loss || c.system_stability_pct);
        const hasFeedback = conv[0] && conv[0].proprioceptive_feedback !== undefined;

        const traces = [
            {
                x: iterations, y: qLoss,
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#56d364', width: 3 }, marker: { size: 6, color: '#238636' },
                name: type === 'equipment' ? 'QPU Fidelity (%)' : 'Quantum VQE Loss'
            },
            {
                x: iterations, y: cLoss,
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#ff7b72', width: 2, dash: 'dash' }, marker: { size: 5, color: '#da3633' },
                name: type === 'equipment' ? 'System Stability (%)' : 'Classical SGD Loss'
            }
        ];
        if (hasFeedback) {
            traces.push({
                x: iterations, y: conv.map(c => c.proprioceptive_feedback),
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#f1c40f', width: 2, dash: 'dot' }, marker: { size: 5, color: '#d4a017' },
                name: 'Proprioceptive Feedback (paradigm-gated)', yaxis: 'y2'
            });
        }

        Plotly.newPlot(containerId, traces, {
            ...PL,
            xaxis: { ...PL.xaxis, title: 'Optimization Step / Iteration' },
            yaxis: { ...PL.yaxis, title: 'Convergence Loss / Metric' },
            yaxis2: hasFeedback ? { ...PL.yaxis, title: 'Feedback Amplitude', overlaying: 'y', side: 'right', showgrid: false } : undefined,
            legend: { font: { color: '#e6edf3' }, orientation: 'h', x: 0, y: 1.15 }
        }, { responsive: true });
    }

    function renderRecommendedCard(containerId, rec) {
        const container = document.getElementById(containerId);
        if (!container || !rec) return;
        const rows = Object.entries(rec).map(([k, v]) => {
            if (k === 'title') return `<div style="font-weight:700;color:#58a6ff;font-size:15px;margin-bottom:8px;">${v}</div>`;
            const keyLabel = k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            return `<tr><td style="color:#8b949e;padding:4px 8px;font-size:12px;">${keyLabel}</td><td style="color:#e6edf3;font-weight:600;padding:4px 8px;font-size:12px;">${v}</td></tr>`;
        }).join('');
        container.innerHTML = `
            <div style="background:rgba(35,134,54,0.06);border:1px solid rgba(35,134,54,0.3);border-radius:8px;padding:14px;margin-top:4px;">
                <table class="eq-spec-table" style="margin:0;width:100%;">
                    ${rows}
                </table>
            </div>
        `;
    }


    // ═════════════════════════════════════════════════════════════════
    //  SLEEP APNEA RTMS NEUROMODULATION SUITE logic
    // ═════════════════════════════════════════════════════════════════

    let sleepApneaDebounceTimer = null;
    window.runSleepApneaRtmsDebounced = function() {
        clearTimeout(sleepApneaDebounceTimer);
        sleepApneaDebounceTimer = setTimeout(runSleepApneaRtms, 25);
    };

    window.runSleepApneaRtms = async function() {
        const baselineAhiEl = document.getElementById('sa-baseline-ahi');
        if (!baselineAhiEl) return; // not initialized or view not loaded

        const baselineAhi = baselineAhiEl.value;
        const rtmsFreq = document.getElementById('sa-rtms-freq').value;
        const adaptiveGain = document.getElementById('sa-adaptive-gain').value;
        const durationDays = document.getElementById('sa-duration-days').value;
        const targetSyncRatio = document.getElementById('sa-target-sync-ratio').value;

        const url = `/api/sleep-apnea-rtms?baseline_ahi=${baselineAhi}&rtms_freq_hz=${rtmsFreq}&adaptive_gain=${adaptiveGain}&duration_days=${durationDays}&target_sync_ratio=${targetSyncRatio}`;

        try {
            const r = await fetch(url);
            const data = await r.json();
            if (data.error) {
                console.error("Sleep Apnea rTMS API Error:", data.error);
                return;
            }

            // 1. Update Metrics
            document.getElementById('sa-metric-baseline').textContent = parseFloat(baselineAhi).toFixed(1) + " events/hr";
            const finalCpap = data.ahi_cpap[data.ahi_cpap.length - 1];
            const finalStd = data.ahi_rtms_std[data.ahi_rtms_std.length - 1];
            const finalOpt = data.ahi_rtms_opt[data.ahi_rtms_opt.length - 1];

            document.getElementById('sa-metric-cpap').textContent = finalCpap.toFixed(1) + " events/hr";
            document.getElementById('sa-metric-rtms-std').textContent = finalStd.toFixed(1) + " events/hr";
            document.getElementById('sa-metric-rtms-opt').textContent = finalOpt.toFixed(1) + " events/hr";
            document.getElementById('sa-metric-convergents').textContent = data.convergents.join(', ');
            document.getElementById('sa-metric-expansion').textContent = "[" + data.cf_expansion.join(', ') + "]";

            // 2. Render ASCII Schematic
            document.getElementById('sa-ascii-schematic').textContent = data.ascii_schematic;

            // 3. Render Prescription Markdown -> HTML
            let txt = data.genai_prescription || "";
            txt = txt.replace(/\*\*(.*?)\*\//g, '<strong>$1</strong>');
            txt = txt.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            txt = txt.replace(/\$(.*?)\$/g, '<em>$1</em>');
            txt = txt.replace(/\n\n/g, '<br><br>');
            txt = txt.replace(/### /g, '');
            document.getElementById('sa-genai-text').innerHTML = txt;

            // 4. Plot 1: AHI Trajectories
            const traceBaseline = {
                x: data.days,
                y: data.ahi_baseline,
                name: 'Untreated Baseline AHI',
                type: 'scatter',
                mode: 'lines',
                line: {color: '#ff7b72', width: 2, dash: 'dot'}
            };
            const traceCpap = {
                x: data.days,
                y: data.ahi_cpap,
                name: 'Standard CPAP Therapy',
                type: 'scatter',
                mode: 'lines',
                line: {color: '#ffa657', width: 2}
            };
            const traceStdRtms = {
                x: data.days,
                y: data.ahi_rtms_std,
                name: 'Open-loop rTMS (Standard)',
                type: 'scatter',
                mode: 'lines',
                line: {color: '#58a6ff', width: 2.5}
            };
            const traceOptRtms = {
                x: data.days,
                y: data.ahi_rtms_opt,
                name: 'Adaptive Closed-loop rTMS (Optimal)',
                type: 'scatter',
                mode: 'lines+markers',
                line: {color: '#56d364', width: 3.5},
                marker: {size: 5}
            };

            const darkThemeLayout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: { t: 50, b: 40, l: 40, r: 40 },
                xaxis: { title: 'Duration (Days)', gridcolor: 'rgba(255,255,255,0.05)', color: '#8b949e' },
                yaxis: { title: 'AHI (events / hour)', gridcolor: 'rgba(255,255,255,0.05)', color: '#8b949e' },
                font: { color: '#c9d1d9', family: 'Inter' },
                legend: { font: { color: '#c9d1d9', size: 10 }, bgcolor: 'rgba(0,0,0,0.4)', orientation: 'h', x: 0, y: -0.25 }
            };

            Plotly.react('sa-plot-trajectory', [traceBaseline, traceCpap, traceStdRtms, traceOptRtms], darkThemeLayout, {responsive:true, displaylogo:false});

        } catch (err) {
            console.error("Error fetching Sleep Apnea rTMS data:", err);
        }
    };

    // Depression rTMS + CBT computational research suite
    let depressionDebounceTimer = null;
    window.runDepressionRtmsDebounced = function() {
        clearTimeout(depressionDebounceTimer);
        depressionDebounceTimer = setTimeout(runDepressionRtms, 40);
    };

    window.runDepressionRtms = async function() {
        const baselineEl = document.getElementById('dep-phq9');
        if (!baselineEl) return;

        const params = new URLSearchParams({
            baseline_phq9: baselineEl.value,
            sessions: document.getElementById('dep-sessions').value,
            rtms_frequency_hz: document.getElementById('dep-frequency').value,
            cbt_weight: document.getElementById('dep-cbt').value,
            control_gain: document.getElementById('dep-control').value,
            signature_ratio: document.getElementById('dep-ratio').value
        });
        document.getElementById('dep-preprint-link').href = `/api/depression-rtms-preprint?${params.toString()}`;

        try {
            const response = await fetch(`/api/depression-rtms?${params.toString()}`);
            const data = await response.json();
            if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);

            document.getElementById('dep-final-phq9').textContent = data.metrics.final_phq9.toFixed(2);
            document.getElementById('dep-response').textContent = `${data.metrics.modeled_response_pct.toFixed(1)}%`;
            document.getElementById('dep-cognitive-state').textContent = data.metrics.final_distortion_state.toFixed(3);
            document.getElementById('dep-control-effort').textContent = data.metrics.mean_control_effort.toFixed(3);
            document.getElementById('dep-prime-sessions').textContent = data.prime_sessions.join(', ');

            const commonLayout = {
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                margin: {t: 28, b: 46, l: 48, r: 20},
                font: {color: '#c9d1d9', family: 'Inter'},
                xaxis: {title: 'Session', gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e'},
                yaxis: {gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e'},
                legend: {orientation: 'h', x: 0, y: -0.24, font: {size: 10}}
            };
            Plotly.react('dep-trajectory-chart', [
                {x:data.sessions, y:data.phq9_usual_care, name:'Usual-care model', type:'scatter', mode:'lines', line:{color:'#8b949e', dash:'dot'}},
                {x:data.sessions, y:data.phq9_cbt_only, name:'CBT-only model', type:'scatter', mode:'lines', line:{color:'#58a6ff'}},
                {x:data.sessions, y:data.phq9_rtms_only, name:'rTMS-only model', type:'scatter', mode:'lines', line:{color:'#ffa657'}},
                {x:data.sessions, y:data.phq9_adaptive_combined, name:'Adaptive combined', type:'scatter', mode:'lines+markers', line:{color:'#56d364', width:3}, marker:{size:4}}
            ], {...commonLayout, yaxis:{...commonLayout.yaxis, title:'PHQ-9', range:[0,27]}}, {responsive:true, displaylogo:false});

            Plotly.react('dep-distribution-chart', [
                {x:data.distribution.bin_centers, y:data.distribution.baseline_counts, name:'Baseline', type:'bar', marker:{color:'#58a6ff'}},
                {x:data.distribution.bin_centers, y:data.distribution.post_counts, name:'Synthetic post-model', type:'bar', marker:{color:'#56d364'}}
            ], {...commonLayout, barmode:'group', xaxis:{...commonLayout.xaxis, title:'PHQ-9 bin'}, yaxis:{...commonLayout.yaxis, title:'Count'}}, {responsive:true, displaylogo:false});

            Plotly.react('dep-control-chart', [
                {x:data.sessions, y:data.control_effort, name:'Control effort', type:'scatter', mode:'lines', line:{color:'#ff7b72', width:2}},
                {x:data.sessions, y:data.cognitive_distortion_state, name:'Cognitive state', type:'scatter', mode:'lines', line:{color:'#d2a8ff', width:2}}
            ], {...commonLayout, yaxis:{...commonLayout.yaxis, title:'Normalized state', range:[0,1.05]}}, {responsive:true, displaylogo:false});

            Plotly.react('dep-signature-chart', [
                {x:data.sessions, y:data.objective, name:'Finite objective', type:'scatter', mode:'lines', line:{color:'#56d364'}, yaxis:'y'},
                {x:data.sessions, y:data.number_signature, name:'Modulo-17 signature', type:'scatter', mode:'lines+markers', line:{color:'#ffa657'}, marker:{size:4}, yaxis:'y2'}
            ], {...commonLayout, yaxis:{...commonLayout.yaxis, title:'Objective'}, yaxis2:{title:'Signature', overlaying:'y', side:'right', color:'#ffa657'}}, {responsive:true, displaylogo:false});

            const paradigm = data.paradigm;
            document.getElementById('dep-paradigm-card').innerHTML = `
                <div style="color:#ff7b72; font-weight:700; margin-bottom:10px;">${paradigm.status}</div>
                <table class="eq-spec-table" style="margin:0; width:100%;">
                    <tr><td>Target abstraction</td><td>${paradigm.target}</td></tr>
                    <tr><td>Finite horizon</td><td>${paradigm.sessions} sessions</td></tr>
                    <tr><td>CBT component</td><td>${paradigm.cbt_component}</td></tr>
                    <tr><td>Controller</td><td>${paradigm.control_rule}</td></tr>
                    <tr><td>Safety</td><td>${paradigm.safety}</td></tr>
                    <tr><td>CF expansion</td><td>[${data.continued_fraction.coefficients.join(', ')}]</td></tr>
                </table>`;
        } catch (error) {
            console.error('Depression rTMS API error:', error);
            document.getElementById('dep-paradigm-card').textContent = `Model unavailable: ${error.message}`;
        }
    };

    // ─────────────────────────────────────────────────────────────
    // Anxiety in Millennials rTMS + Pharmacotherapy Research Suite
    // ─────────────────────────────────────────────────────────────
    let anxietyDebounceTimer = null;
    window.runAnxietyRtmsDebounced = function() {
        clearTimeout(anxietyDebounceTimer);
        anxietyDebounceTimer = setTimeout(runAnxietyRtms, 40);
    };

    window.runAnxietyRtms = async function() {
        const baselineEl = document.getElementById('anx-baseline-gad7');
        if (!baselineEl) return;

        const params = new URLSearchParams({
            baseline_gad7: baselineEl.value,
            treatment_weeks: document.getElementById('anx-weeks').value,
            rtms_freq_hz: document.getElementById('anx-freq').value,
            stimulation_intensity_pct: document.getElementById('anx-intensity').value,
            cbt_synergy_gain: document.getElementById('anx-cbt-gain').value,
            cf_signature_ratio: document.getElementById('anx-ratio').value
        });

        const preprintBtn = document.getElementById('anx-preprint-btn');
        if (preprintBtn) {
            preprintBtn.href = `/api/anxiety-rtms-preprint?${params.toString()}`;
        }

        try {
            const response = await fetch(`/api/anxiety-rtms?${params.toString()}`);
            const result = await response.json();
            if (!response.ok || result.error) throw new Error(result.error || `HTTP ${response.status}`);
            const data = result.data;

            // Update Metrics
            document.getElementById('anx-metric-final-gad7').textContent = data.metrics.final_gad7.toFixed(2);
            document.getElementById('anx-metric-reduction').textContent = `-${data.metrics.absolute_reduction.toFixed(1)} pts (${data.metrics.percent_reduction.toFixed(1)}%)`;
            document.getElementById('anx-metric-cohend').textContent = `${data.metrics.cohen_d.toFixed(2)} (Large)`;
            document.getElementById('anx-metric-delta-faa').textContent = `+${data.metrics.delta_faa.toFixed(3)} (Valence Restored)`;
            document.getElementById('anx-metric-peak-e').textContent = `${data.metrics.peak_e_vm.toFixed(1)} V/m`;
            document.getElementById('anx-metric-convergents').textContent = `[${data.cf_convergents.slice(0, 4).map(c => c.fraction).join(', ')}]`;

            const commonLayout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: {t: 28, b: 46, l: 48, r: 24},
                font: {color: '#c9d1d9', family: 'Inter'},
                xaxis: {title: 'Treatment Horizon (Weeks)', gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e'},
                yaxis: {gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e'},
                legend: {orientation: 'h', x: 0, y: -0.24, font: {size: 10}}
            };

            // 1. Trajectory plot
            Plotly.react('anx-plot-trajectory', [
                {x: data.weeks, y: data.gad7_sham, name: 'Placebo / Sham rTMS', type: 'scatter', mode: 'lines', line: {color: '#8b949e', dash: 'dot'}},
                {x: data.weeks, y: data.gad7_pharm, name: 'SSRI / SNRI Monotherapy', type: 'scatter', mode: 'lines', line: {color: '#58a6ff'}},
                {x: data.weeks, y: data.gad7_rtms, name: '1Hz dlPFC Monotherapy', type: 'scatter', mode: 'lines', line: {color: '#ffa657'}},
                {x: data.weeks, y: data.gad7_synergistic, name: 'Synergistic (rTMS + Pharm + CBT)', type: 'scatter', mode: 'lines+markers', line: {color: '#56d364', width: 3}, marker: {size: 4}}
            ], {
                ...commonLayout,
                yaxis: {...commonLayout.yaxis, title: 'GAD-7 Score (0-21)', range: [0, 22]}
            }, {responsive: true, displaylogo: false});

            // 2. FEA Cortical Penetration
            Plotly.react('anx-plot-fea', [
                {x: data.fea.depths_mm, y: data.fea.e_field_vm, name: 'E-Field (V/m)', type: 'scatter', mode: 'lines', line: {color: '#ff7b72', width: 2.5}},
                {x: data.fea.depths_mm, y: data.fea.current_density_am2, name: 'Current Density (A/m²)', type: 'scatter', mode: 'lines', line: {color: '#d2a8ff', width: 2}, yaxis: 'y2'}
            ], {
                ...commonLayout,
                xaxis: {...commonLayout.xaxis, title: 'Cranial Depth z (mm)'},
                yaxis: {...commonLayout.yaxis, title: 'E-Field (V/m)'},
                yaxis2: {title: 'J (A/m²)', overlaying: 'y', side: 'right', color: '#d2a8ff', gridcolor: 'rgba(210,168,255,0.08)'}
            }, {responsive: true, displaylogo: false});

            // 3. EEG Power Spectral Density
            Plotly.react('anx-plot-psd', [
                {x: data.eeg.frequencies, y: data.eeg.psd_pre, name: 'Pre-Op Anxious (Negative FAA)', type: 'scatter', mode: 'lines', line: {color: '#ff7b72', width: 2}},
                {x: data.eeg.frequencies, y: data.eeg.psd_post, name: 'Post-Op Regulated (+FAA)', type: 'scatter', mode: 'lines', line: {color: '#38bdf8', width: 2.5}}
            ], {
                ...commonLayout,
                xaxis: {...commonLayout.xaxis, title: 'Frequency (Hz)'},
                yaxis: {...commonLayout.yaxis, title: 'PSD (μV²/Hz)'}
            }, {responsive: true, displaylogo: false});

            // 4. Raw EEG Time-Domain traces
            Plotly.react('anx-plot-raw-eeg', [
                {x: data.eeg.time_pts, y: data.eeg.raw_eeg_pre, name: 'Pre-Op Trace (F4/F3)', type: 'scatter', mode: 'lines', line: {color: '#ff7b72', width: 1.2}},
                {x: data.eeg.time_pts, y: data.eeg.raw_eeg_post, name: 'Post-Op Trace (Normalized)', type: 'scatter', mode: 'lines', line: {color: '#56d364', width: 1.2}}
            ], {
                ...commonLayout,
                xaxis: {...commonLayout.xaxis, title: 'Time (Seconds)'},
                yaxis: {...commonLayout.yaxis, title: 'Amplitude (μV)'}
            }, {responsive: true, displaylogo: false});

            // 5. Clinical Trials Bar Chart
            const armLabels = data.trials.trial_arms.map(a => a.arm.split('(')[0].trim());
            const remRates = data.trials.trial_arms.map(a => a.remission_pct);
            const cohenList = data.trials.trial_arms.map(a => a.cohen_d);
            Plotly.react('anx-plot-trials', [
                {x: armLabels, y: remRates, name: 'Remission Rate (%)', type: 'bar', marker: {color: '#38bdf8'}},
                {x: armLabels, y: cohenList.map(c => c * 40.0), name: "Cohen's d (Scaled x40)", type: 'bar', marker: {color: '#56d364'}}
            ], {
                ...commonLayout,
                barmode: 'group',
                xaxis: {...commonLayout.xaxis, title: 'Trial Protocol Arm', tickangle: -20},
                yaxis: {...commonLayout.yaxis, title: 'Percentage / Scaled Score'}
            }, {responsive: true, displaylogo: false});

            // 6. Markov Remission & Relapse Hazard
            Plotly.react('anx-plot-markov', [
                {x: data.weeks, y: data.remission_probability, name: 'Remission Probability (%)', type: 'scatter', mode: 'lines', line: {color: '#56d364', width: 2.5}},
                {x: data.weeks, y: data.relapse_hazard_pct, name: 'Relapse Hazard Rate (%)', type: 'scatter', mode: 'lines', line: {color: '#ff7b72', width: 2, dash: 'dash'}}
            ], {
                ...commonLayout,
                yaxis: {...commonLayout.yaxis, title: 'Probability / Hazard (%)', range: [0, 105]}
            }, {responsive: true, displaylogo: false});

            // 7. Staging candidate costs
            Plotly.react('anx-plot-staging', [
                {x: data.staging.candidate_rank, y: data.staging.candidate_costs, name: 'Candidate Gate Cost J', type: 'scatter', mode: 'lines+markers', line: {color: '#d2a8ff'}, marker: {size: 4}}
            ], {
                ...commonLayout,
                xaxis: {...commonLayout.xaxis, title: 'Ranked Gate Pair Candidate'},
                yaxis: {...commonLayout.yaxis, title: 'Multi-Objective Cost'}
            }, {responsive: true, displaylogo: false});

            // ASCII schematic
            const asciiEl = document.getElementById('anx-ascii-schematic');
            if (asciiEl) asciiEl.textContent = data.ascii_schematic;

            // Clinical summary text
            const summaryEl = document.getElementById('anx-genai-text');
            if (summaryEl) {
                summaryEl.innerHTML = data.clinical_summary
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\n\n/g, '<br/><br/>');
            }

        } catch (error) {
            console.error('Anxiety rTMS API error:', error);
            const summaryEl = document.getElementById('anx-genai-text');
            if (summaryEl) summaryEl.textContent = `Simulation unavailable: ${error.message}`;
        }
    };

    // ─────────────────────────────────────────────────────────────
    // Moduli-Theoretic Treatment Paradigm & BEM Simulation Suite
    // ─────────────────────────────────────────────────────────────
    let moduliBemDebounceTimer = null;
    window.runModuliBemParadigmDebounced = function() {
        clearTimeout(moduliBemDebounceTimer);
        moduliBemDebounceTimer = setTimeout(runModuliBemParadigm, 50);
    };

    window.runModuliBemParadigm = async function() {
        const condEl = document.getElementById('mb-condition');
        if (!condEl) return;

        const condition = condEl.value;
        const freqMax = document.getElementById('mb-freq-max').value;
        const intensityMax = document.getElementById('mb-intensity-max').value;

        const preprintBtn = document.getElementById('mb-preprint-btn');
        if (preprintBtn) {
            preprintBtn.href = `/api/moduli-bem-preprint?condition=${condition}&freq_max=${freqMax}&intensity_max=${intensityMax}`;
        }

        try {
            const response = await fetch(`/api/moduli-bem-paradigm?condition=${condition}&freq_max=${freqMax}&intensity_max=${intensityMax}`);
            const result = await response.json();
            if (!response.ok || result.error) throw new Error(result.error || `HTTP ${response.status}`);
            const data = result.data;
            const opt = data.optimal_protocol;

            // Update Metrics
            document.getElementById('mb-metric-freq').textContent = `${opt.frequency_hz.toFixed(2)} Hz`;
            document.getElementById('mb-metric-intensity').textContent = `${opt.intensity_pct.toFixed(1)}% MSO`;
            document.getElementById('mb-metric-zreduced').textContent = `${opt.z_reduced.re.toFixed(3)} + ${opt.z_reduced.im.toFixed(3)}i`;
            document.getElementById('mb-metric-elliptic').textContent = opt.nearest_elliptic_point;
            document.getElementById('mb-metric-stability').textContent = opt.stability_score.toFixed(4);
            document.getElementById('mb-metric-peak').textContent = `${data.bem_heatmap.peak_potential.toFixed(2)} (Scaled)`;

            const commonLayout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: {t: 30, b: 46, l: 48, r: 24},
                font: {color: '#c9d1d9', family: 'Inter'},
                xaxis: {gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e'},
                yaxis: {gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e'},
            };

            // 1. Moduli Stability Heatmap
            Plotly.react('mb-plot-moduli-heatmap', [{
                z: data.moduli_grid.stability,
                x: data.moduli_grid.freq_axis,
                y: data.moduli_grid.intensity_axis,
                type: 'heatmap',
                colorscale: 'Viridis',
                colorbar: {title: 'Phi(z)', tickfont: {color: '#cbd5e1', size: 8}}
            }], {
                ...commonLayout,
                xaxis: {...commonLayout.xaxis, title: 'Frequency f (Hz)'},
                yaxis: {...commonLayout.yaxis, title: 'Intensity I (% MSO)'}
            }, {responsive: true, displaylogo: false});

            // 2. BEM Cortical Potential Heatmap
            Plotly.react('mb-plot-bem-heatmap', [{
                z: data.bem_heatmap.potential,
                x: data.bem_heatmap.theta,
                y: data.bem_heatmap.phi,
                type: 'heatmap',
                colorscale: 'Plasma',
                colorbar: {title: 'Potential', tickfont: {color: '#cbd5e1', size: 8}}
            }], {
                ...commonLayout,
                xaxis: {...commonLayout.xaxis, title: 'Azimuth theta (rad)'},
                yaxis: {...commonLayout.yaxis, title: 'Polar phi (rad)'}
            }, {responsive: true, displaylogo: false});

            // 3. Cortical Depth Field Attenuation
            Plotly.react('mb-plot-attenuation', [{
                x: data.bem_attenuation.depths_mm,
                y: data.bem_attenuation.field_pct,
                name: 'Induced E-Field',
                type: 'scatter',
                mode: 'lines+markers',
                line: {color: '#a78bfa', width: 2.5},
                marker: {size: 4}
            }], {
                ...commonLayout,
                xaxis: {...commonLayout.xaxis, title: 'Cortical Depth (mm)'},
                yaxis: {...commonLayout.yaxis, title: 'Field Strength (% MSO)'}
            }, {responsive: true, displaylogo: false});

            // 4. Continued Fraction Convergents Bar
            const cfLabels = data.cf_convergents.slice(0, 6).map(c => `${c.numerator}/${c.denominator}`);
            const cfErrors = data.cf_convergents.slice(0, 6).map(c => Math.max(0.0001, c.error_pct));
            Plotly.react('mb-plot-cf', [{
                x: cfLabels,
                y: cfErrors,
                name: 'Approximation Error (%)',
                type: 'bar',
                marker: {color: '#38bdf8'}
            }], {
                ...commonLayout,
                xaxis: {...commonLayout.xaxis, title: 'Rational Convergent p_k / q_k'},
                yaxis: {...commonLayout.yaxis, title: 'Error vs f* (%)', type: 'log'}
            }, {responsive: true, displaylogo: false});

            // ASCII schematic
            const asciiEl = document.getElementById('mb-ascii-schematic');
            if (asciiEl) asciiEl.textContent = data.ascii_schematic;

            // Clinical summary text
            const summaryEl = document.getElementById('mb-genai-text');
            if (summaryEl) {
                summaryEl.innerHTML = data.clinical_summary
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\n\n/g, '<br/><br/>');
            }

        } catch (error) {
            console.error('Moduli-BEM API error:', error);
            const summaryEl = document.getElementById('mb-genai-text');
            if (summaryEl) summaryEl.textContent = `Simulation unavailable: ${error.message}`;
        }
    };

    // ─────────────────────────────────────────────────────────────
    // Tourette Syndrome (CSTC Combinatorics) Research Suite
    // ─────────────────────────────────────────────────────────────
    let touretteDebounceTimer = null;
    window.runTouretteRtmsDebounced = function() {
        clearTimeout(touretteDebounceTimer);
        touretteDebounceTimer = setTimeout(runTouretteRtms, 50);
    };

    window.runTouretteRtms = async function() {
        const baselineEl = document.getElementById('ts-baseline-ygtss');
        if (!baselineEl) return;

        const params = new URLSearchParams({
            baseline_ygtss: baselineEl.value,
            treatment_weeks: document.getElementById('ts-weeks').value,
            daily_pulses: document.getElementById('ts-pulses').value,
            stimulation_intensity_pct: document.getElementById('ts-intensity').value,
            hrt_synergy_gain: document.getElementById('ts-hrt-gain').value,
            cf_signature_ratio: document.getElementById('ts-ratio').value
        });

        const preprintBtn = document.getElementById('ts-preprint-btn');
        if (preprintBtn) {
            preprintBtn.href = `/api/tourette-rtms-preprint?${params.toString()}`;
        }

        try {
            const response = await fetch(`/api/tourette-rtms?${params.toString()}`);
            const result = await response.json();
            if (!response.ok || result.error) throw new Error(result.error || `HTTP ${response.status}`);
            const data = result.data;
            const metrics = data.metrics;

            // Update Metrics
            document.getElementById('ts-metric-final-ygtss').textContent = metrics.final_ygtss.toFixed(2);
            document.getElementById('ts-metric-reduction').textContent = `-${metrics.absolute_reduction.toFixed(1)} pts (${metrics.percent_reduction.toFixed(1)}%)`;
            document.getElementById('ts-metric-puts').textContent = `${metrics.final_puts.toFixed(1)} (-${metrics.puts_reduction_pct.toFixed(1)}%)`;
            document.getElementById('ts-metric-entropy').textContent = `${data.allocation.combinatorial_entropy.toFixed(3)} nats`;
            document.getElementById('ts-metric-peak-e').textContent = `${metrics.peak_e_vm.toFixed(1)} V/m`;
            document.getElementById('ts-metric-convergents').textContent = `[${data.cf_convergents.slice(0, 4).map(c => c.fraction).join(', ')}]`;

            const commonLayout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: {t: 28, b: 46, l: 48, r: 24},
                font: {color: '#c9d1d9', family: 'Inter'},
                xaxis: {title: 'Treatment Horizon (Weeks)', gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e'},
                yaxis: {gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e'},
                legend: {orientation: 'h', x: 0, y: -0.24, font: {size: 10}}
            };

            // 1. Longitudinal YGTSS Trajectory plot
            Plotly.react('ts-plot-trajectory', [
                {x: data.weeks, y: data.ygtss_sham, name: 'Sham rTMS Control', type: 'scatter', mode: 'lines', line: {color: '#8b949e', dash: 'dot'}},
                {x: data.weeks, y: data.ygtss_hrt, name: 'HRT Behavioral Monotherapy', type: 'scatter', mode: 'lines', line: {color: '#58a6ff'}},
                {x: data.weeks, y: data.ygtss_rtms, name: '1Hz pre-SMA Monotherapy', type: 'scatter', mode: 'lines', line: {color: '#ffa657'}},
                {x: data.weeks, y: data.ygtss_synergistic, name: 'Combinatorial rTMS + HRT', type: 'scatter', mode: 'lines+markers', line: {color: '#56d364', width: 3}, marker: {size: 4}}
            ], {
                ...commonLayout,
                yaxis: {...commonLayout.yaxis, title: 'Total YGTSS Score (0-50)', range: [0, 52]}
            }, {responsive: true, displaylogo: false});

            // 2. Subscores Plot (Motor, Vocal, PUTS)
            Plotly.react('ts-plot-subscores', [
                {x: data.weeks, y: data.motor_tic_score, name: 'Motor Tics Subscore', type: 'scatter', mode: 'lines', line: {color: '#d2a8ff', width: 2}},
                {x: data.weeks, y: data.vocal_tic_score, name: 'Vocal Tics Subscore', type: 'scatter', mode: 'lines', line: {color: '#38bdf8', width: 2}},
                {x: data.weeks, y: data.puts_urge_score, name: 'Premonitory Urge (PUTS 9-36)', type: 'scatter', mode: 'lines', line: {color: '#ff7b72', width: 2.5}}
            ], {
                ...commonLayout,
                yaxis: {...commonLayout.yaxis, title: 'Component Score'}
            }, {responsive: true, displaylogo: false});

            // 3. Combinatorial Pulse Allocation Bar Chart
            const nodeLabels = data.allocation.allocated_nodes.map(n => n.target_id);
            const nodePulses = data.allocation.allocated_nodes.map(n => n.allocated_pulses);
            const nodeShares = data.allocation.allocated_nodes.map(n => n.pulse_fraction_pct * 30.0);
            Plotly.react('ts-plot-allocation', [
                {x: nodeLabels, y: nodePulses, name: 'Allocated Pulses / Day', type: 'bar', marker: {color: '#a78bfa'}},
                {x: nodeLabels, y: nodeShares, name: 'Pulse Share (% x30)', type: 'bar', marker: {color: '#34d399'}}
            ], {
                ...commonLayout,
                barmode: 'group',
                xaxis: {...commonLayout.xaxis, title: 'CSTC Cortical Target Node'},
                yaxis: {...commonLayout.yaxis, title: 'Pulses / Scaled Share'}
            }, {responsive: true, displaylogo: false});

            // 4. BEM Depth Field Attenuation & Current Density
            Plotly.react('ts-plot-bem', [
                {x: data.bem_field.depths_mm, y: data.bem_field.e_field_vm, name: 'E-Field (V/m)', type: 'scatter', mode: 'lines', line: {color: '#ff7b72', width: 2.5}},
                {x: data.bem_field.depths_mm, y: data.bem_field.current_density_am2, name: 'Current Density J (A/m²)', type: 'scatter', mode: 'lines', line: {color: '#38bdf8', width: 2}, yaxis: 'y2'}
            ], {
                ...commonLayout,
                xaxis: {...commonLayout.xaxis, title: 'Tissue Depth z (mm)'},
                yaxis: {...commonLayout.yaxis, title: 'E-Field (V/m)'},
                yaxis2: {title: 'J (A/m²)', overlaying: 'y', side: 'right', color: '#38bdf8', gridcolor: 'rgba(56,189,248,0.08)'}
            }, {responsive: true, displaylogo: false});

            // 5. Permutation Entropy & Control Effort
            Plotly.react('ts-plot-entropy', [
                {x: data.weeks, y: data.tic_cluster_entropy, name: 'Permutation Entropy H_perm', type: 'scatter', mode: 'lines+markers', line: {color: '#34d399', width: 2.5}, marker: {size: 4}},
                {x: data.weeks, y: data.control_effort, name: 'Control Effort u_k', type: 'scatter', mode: 'lines', line: {color: '#fbbf24', width: 2, dash: 'dot'}}
            ], {
                ...commonLayout,
                yaxis: {...commonLayout.yaxis, title: 'Entropy / Control Effort [0-1]', range: [0, 1.05]}
            }, {responsive: true, displaylogo: false});

            // 6. Staging candidate costs
            Plotly.react('ts-plot-staging', [
                {x: data.staging.candidate_rank, y: data.staging.candidate_costs, name: 'Candidate Gate Cost J', type: 'scatter', mode: 'lines+markers', line: {color: '#d2a8ff'}, marker: {size: 4}}
            ], {
                ...commonLayout,
                xaxis: {...commonLayout.xaxis, title: 'Ranked Gate Pair Candidate'},
                yaxis: {...commonLayout.yaxis, title: 'Multi-Objective Cost J_stage'}
            }, {responsive: true, displaylogo: false});

            // ASCII schematic
            const asciiEl = document.getElementById('ts-ascii-schematic');
            if (asciiEl) asciiEl.textContent = data.ascii_schematic;

            // Clinical summary text
            const summaryEl = document.getElementById('ts-genai-text');
            if (summaryEl) {
                summaryEl.innerHTML = data.clinical_summary
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\n\n/g, '<br/><br/>');
            }

        } catch (error) {
            console.error('Tourette rTMS API error:', error);
            const summaryEl = document.getElementById('ts-genai-text');
            if (summaryEl) summaryEl.textContent = `Simulation unavailable: ${error.message}`;
        }
    };

    // ── TBI & PTSD rTMS Neuromodulation ──────────────────────────
    let _tbiDebounceTimer = null;
    function runTbiPtsdRtmsDebounced() {
        clearTimeout(_tbiDebounceTimer);
        _tbiDebounceTimer = setTimeout(runTbiPtsdRtms, 350);
    }
    window.runTbiPtsdRtmsDebounced = runTbiPtsdRtmsDebounced;

    async function runTbiPtsdRtms() {
        const pcl5  = parseFloat(document.getElementById('tbi-pcl5')?.value  || 58);
        const rpq   = parseFloat(document.getElementById('tbi-rpq')?.value   || 42);
        const weeks = parseInt(document.getElementById('tbi-weeks')?.value   || 24);
        const freq  = parseFloat(document.getElementById('tbi-freq')?.value  || 10);
        const coils = parseInt(document.getElementById('tbi-coils')?.value   || 3);
        const price = parseInt(document.getElementById('tbi-price')?.value   || 300);

        const qs = `baseline_pcl5=${pcl5}&baseline_rpq=${rpq}&treatment_weeks=${weeks}&rtms_freq_hz=${freq}&clinic_coils=${coils}&session_price=${price}`;

        try {
            const res  = await fetch(`/api/tbi-ptsd-rtms?${qs}`);
            const json = await res.json();
            if (json.status !== 'success') throw new Error(json.message || 'API error');
            const d = json.data;

            const darkBg   = 'rgba(0,0,0,0)';
            const gridClr  = 'rgba(255,255,255,0.06)';
            const fontClr  = '#8b949e';
            const commonLayout = {
                paper_bgcolor: darkBg, plot_bgcolor: darkBg,
                font: { color: fontClr, size: 11, family: 'Inter, sans-serif' },
                margin: { l: 55, r: 25, t: 30, b: 45 },
                legend: { orientation: 'h', y: -0.22, font: { size: 10 } },
                xaxis: { gridcolor: gridClr, zeroline: false },
                yaxis: { gridcolor: gridClr, zeroline: false }
            };

            // -- Metric spans --
            const el = id => document.getElementById(id);
            el('tbi-metric-pcl5').textContent    = d.pcl5_synergistic[d.pcl5_synergistic.length-1].toFixed(1);
            el('tbi-metric-rpq').textContent     = d.rpq_synergistic[d.rpq_synergistic.length-1].toFixed(1);
            el('tbi-metric-savings').textContent  = '$' + d.economics.total_savings_per_patient.toLocaleString();
            el('tbi-metric-revenue').textContent  = '$' + d.economics.total_revenue_5yr.toLocaleString();
            el('tbi-metric-npv').textContent      = '$' + d.economics.npv.toLocaleString();
            el('tbi-metric-payback').textContent   = d.economics.payback_months.toFixed(1) + ' months';

            // -- 1. Longitudinal Trajectories --
            Plotly.newPlot('tbi-plot-trajectories', [
                { x: d.weeks, y: d.pcl5_standard,     name: 'PCL-5 Standard Care',   line: { color: '#ff7b72', dash: 'dot' } },
                { x: d.weeks, y: d.pcl5_rtms_only,     name: 'PCL-5 rTMS Only',       line: { color: '#ffa657' } },
                { x: d.weeks, y: d.pcl5_synergistic,   name: 'PCL-5 rTMS+CBT Synergy',line: { color: '#56d364', width: 3 } },
                { x: d.weeks, y: d.rpq_standard,       name: 'RPQ Standard Care',     line: { color: '#79c0ff', dash: 'dot' } },
                { x: d.weeks, y: d.rpq_rtms_only,      name: 'RPQ rTMS Only',         line: { color: '#d2a8ff' } },
                { x: d.weeks, y: d.rpq_synergistic,    name: 'RPQ rTMS+CBT Synergy',  line: { color: '#38bdf8', width: 3 } },
                { x: d.weeks, y: d.bdnf_index,         name: 'BDNF Index (×fold)',    line: { color: '#f0883e', dash: 'dashdot' }, yaxis: 'y2' }
            ], {
                ...commonLayout,
                xaxis: { ...commonLayout.xaxis, title: 'Treatment Week' },
                yaxis: { ...commonLayout.yaxis, title: 'Symptom Score (PCL-5 / RPQ)' },
                yaxis2: { title: 'BDNF Fold', overlaying: 'y', side: 'right', showgrid: false, titlefont: { color: '#f0883e' }, tickfont: { color: '#f0883e' } }
            }, { responsive: true, displaylogo: false });

            // -- 2. BEM Surface E-Field Heatmap --
            Plotly.newPlot('tbi-plot-bem-surface', [{
                z: d.bem_field.efield_norm,
                type: 'heatmap',
                colorscale: 'Hot',
                colorbar: { title: 'V/m', titleside: 'right', tickfont: { size: 10 } }
            }], {
                ...commonLayout,
                xaxis: { ...commonLayout.xaxis, title: 'θ (Azimuthal BEM Node)' },
                yaxis: { ...commonLayout.yaxis, title: 'φ (Polar BEM Node)' }
            }, { responsive: true, displaylogo: false });

            // -- 3. Depth Attenuation --
            Plotly.newPlot('tbi-plot-depth', [{
                x: d.bem_depth.depths_mm,
                y: d.bem_depth.field_v_m,
                mode: 'lines+markers',
                name: 'E-field (V/m)',
                line: { color: '#38bdf8', width: 2 },
                marker: { size: 4 },
                fill: 'tozeroy',
                fillcolor: 'rgba(56,189,248,0.08)'
            }], {
                ...commonLayout,
                xaxis: { ...commonLayout.xaxis, title: 'Tissue Depth (mm)' },
                yaxis: { ...commonLayout.yaxis, title: 'Induced E-field (V/m)' },
                annotations: [{
                    x: 40, y: d.bem_depth.target_depth_amygdala_v_m,
                    text: 'Amygdala ' + d.bem_depth.target_depth_amygdala_v_m.toFixed(1) + ' V/m',
                    showarrow: true, arrowhead: 2, ax: -40, ay: -30,
                    font: { color: '#ff7b72', size: 11 }
                }]
            }, { responsive: true, displaylogo: false });

            // -- 4. 5-Year Revenue --
            const years = ['Year 1','Year 2','Year 3','Year 4','Year 5'];
            Plotly.newPlot('tbi-plot-revenue', [
                { x: years, y: d.economics.revenue_5yr, type: 'bar', name: 'Revenue ($)',
                  marker: { color: ['#38bdf8','#56d364','#d2a8ff','#ffa657','#ff7b72'] } },
                { x: years, y: d.economics.revenue_5yr.map(r => r - d.economics.annual_opex),
                  type: 'scatter', mode: 'lines+markers', name: 'Net Profit ($)',
                  line: { color: '#56d364', width: 2, dash: 'dash' }, marker: { size: 7 } }
            ], {
                ...commonLayout,
                xaxis: { ...commonLayout.xaxis, title: 'Fiscal Year' },
                yaxis: { ...commonLayout.yaxis, title: 'Revenue / Profit ($)' },
                barmode: 'group'
            }, { responsive: true, displaylogo: false });

            // -- 5. Savings Breakdown --
            Plotly.newPlot('tbi-plot-savings', [{
                values: [d.economics.disability_savings_per_patient, d.economics.er_cost_avoidance],
                labels: ['Disability Claims Avoidance', 'ER / Inpatient Cost Reduction'],
                type: 'pie',
                hole: 0.55,
                marker: { colors: ['#56d364','#38bdf8'] },
                textinfo: 'label+percent',
                textposition: 'outside',
                textfont: { size: 11 }
            }], {
                ...commonLayout,
                showlegend: false,
                annotations: [{ text: '$' + d.economics.total_savings_per_patient.toLocaleString() + '/yr',
                    x: 0.5, y: 0.5, font: { size: 16, color: '#56d364' }, showarrow: false }]
            }, { responsive: true, displaylogo: false });

            // -- ASCII Schematic --
            const asciiEl = document.getElementById('tbi-ascii-schematic');
            if (asciiEl) asciiEl.textContent = d.ascii_schematic;

            // -- Clinical Prescription --
            const rxEl = document.getElementById('tbi-genai-text');
            if (rxEl) {
                rxEl.innerHTML = d.clinical_prescription
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\n\n/g, '<br/><br/>');
            }
        } catch (err) {
            console.error('TBI/PTSD API error:', err);
            const rxEl = document.getElementById('tbi-genai-text');
            if (rxEl) rxEl.textContent = 'Simulation unavailable: ' + err.message;
        }
    }
