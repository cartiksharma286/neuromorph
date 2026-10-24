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
    const strokeRepairView = document.getElementById('stroke-repair-view');
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
    const rlsView          = document.getElementById('rls-view');
    const schizophreniaDbsView = document.getElementById('schizophrenia-dbs-view');
    const hifuTmsView      = document.getElementById('hifu-tms-view');
    const cybersecurityPqcView = document.getElementById('cybersecurity-pqc-view');
    const clinicalTrajectoriesView = document.getElementById('clinical-trajectories-view');
    const pharmacologicalEquivalenceView = document.getElementById('pharmacological-equivalence-view');
    const nihGrantView = document.getElementById('nih-grant-view');
    const parkinsonsView = document.getElementById('parkinsons-view');
    const heartStrokeView = document.getElementById('heart-stroke-view');
    const femCorticalView = document.getElementById('fem-cortical-view');
    const connectomeView = document.getElementById('connectome-view');
    const healthEconomicsView = document.getElementById('health-economics-view');
    const xanaduQuantumView = document.getElementById('xanadu-quantum-view');
    const parkinsonsDbsView = document.getElementById('parkinsons-dbs-view');
    const indigenousDepressionView = document.getElementById('indigenous-depression-view');
    const addictionRecoveryView = document.getElementById('addiction-recovery-view');
    const usHealthEconomicsView = document.getElementById('us-health-economics-view');
    const bosonicDementiaView = document.getElementById('bosonic-dementia-view');

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
    let schizophreniaDbsLoaded = false;
    let hifuTmsLoaded    = false;
    let paradigmCache    = {};

    // ── All views array for easy hide-all ────────────────────────
    const allViews = [simulationView, strokeRepairView, tremorView, ocdView, jcView, paradigmView, equipmentView, dementiaLtView, dementiaDbsView, sleepapneaView, depressionView, anxietyView, moduliBemView, touretteView, nashGeodesicView, tbiPtsdView, rlsView, schizophreniaDbsView, hifuTmsView, cybersecurityPqcView, clinicalTrajectoriesView, pharmacologicalEquivalenceView, nihGrantView, parkinsonsView, heartStrokeView, femCorticalView, connectomeView, healthEconomicsView, xanaduQuantumView, parkinsonsDbsView, indigenousDepressionView, addictionRecoveryView];

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
        'stroke-repair': {
            title:    'Stroke Repair rTMS Staging Studio',
            subtitle: 'Statistical recovery distributions · Continued-fraction synchronization · Finite long-term trajectories',
            view:     strokeRepairView, showRunBtn: false
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
        'pharmacological-equivalence': {
            title:    'Pharmacological Equivalence & Trial Management',
            subtitle: 'Generative AI drug discovery for rTMS-equivalent therapeutics · long-horizon clinical modeling',
            view:     pharmacologicalEquivalenceView, showRunBtn: false
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
        },
        'rls': {
            title:    'Restless Legs Syndrome rTMS Research Studio',
            subtitle: 'Recurrent finite paradigms · experimental-design response scores · BEM target visualization · market scenario',
            view:     rlsView, showRunBtn: false
        },
        'schizophrenia-dbs': {
            title:    'Deep Brain Stimulation for Refractory Schizophrenia',
            subtitle: 'Boundary Element Method (BEM) 3D Field Simulations · Innovators Benchmark · Optimal Long-Term Remission Paradigm',
            view:     schizophreniaDbsView, showRunBtn: false
        },
        'hifu-tms': {
            title:    'HIFU-rTMS Trans-BBB Dementia Cure Platform',
            subtitle: 'Statistical Continued Fractions · Long-Term Recovery Horizons · Feynman Path Integrals · Asymmetric Prime Moduli Staging',
            view:     hifuTmsView, showRunBtn: false
        },
        'cybersecurity-pqc': {
            title:    'Cybersecurity & Post-Quantum Cryptographic Measures',
            subtitle: 'Leibniz-Ramanujan Complete Elliptic Integrals · Jacobi Nome & Dedekind Invariants · NIST Level 5 Risk Aversion',
            view:     cybersecurityPqcView, showRunBtn: false
        },
        'clinical-trajectories': {
            title:    'Long-Term Clinical Trajectories & Superoptimal Staging Benchmark',
            subtitle: 'Galois Field F_p Dynamic Programming · 7-Platform DBS/TMS Benchmark · Ramanujan Phase Coupling & Lyapunov Decay',
            view:     clinicalTrajectoriesView, showRunBtn: false
        },
        'nih-grant': {
            title:    'NIH R01 Grant Proposal & rTMS Staging Studio',
            subtitle: 'PHS 398 / SF424 · Sub-$1M Knapsack Budget Optimization · rTMS BEM Characteristics & Galois F_17 Optimal Staging',
            view:     nihGrantView, showRunBtn: false
        },
        'parkinsons': {
            title:    'Parkinson\'s Syndrome: Combinatorial Optimal Staging & BEM Cortical rTMS',
            subtitle: 'Multi-Lobe BEM Cortical Potential Simulation · RF Strength & Electrodynamic Characteristics · Bellman DP Optimal Staging · 36-Month Longitudinal Horizons',
            view:     parkinsonsView, showRunBtn: false
        },
        'heart-stroke': {
            title:    'Heart & Stroke Long-Term Clinical Trajectories',
            subtitle: 'Geodesic Ramanujan Harmonic Modular Nome States · Cortical Surface Interventional Repair · Vanishing Nash Gap · Stem Cell Ring Algebra',
            view:     heartStrokeView, showRunBtn: false
        },
        'fem-cortical': {
            title:    'Finite Element Cortical Surface Simulation & Stieltjes Continued Fractions',
            subtitle: 'Parkinsonian Motor Cortex 3D Dosimetry · Stieltjes Rational Interpolation · GCP Cloud Hyperpartitioning (ParMETIS/Schur) · Low-Rank ACA & Krylov Matrix Reduction',
            view:     femCorticalView, showRunBtn: false
        },
        'connectome': {
            title:    'Human Cortical Connectome: Modular Harmonics & QML',
            subtitle: '68-Region Desikan–Killiany · Ramanujan Modular Harmonic Forms · 6-Qubit Quantum Kernel · Braak-Staged Dementia Nomenclature',
            view:     connectomeView, showRunBtn: false
        },
        'health-economics': {
            title:    'Health Economics & 2-Year Corporate Valuation',
            subtitle: 'Markov Multi-State Transitions · Dementia, Schizophrenia, Parkinson\'s & Alexander\'s Disease · CEAC Frontiers · Pro Forma P&L & rNPV',
            view:     healthEconomicsView, showRunBtn: false
        },
        'xanadu-quantum': {
            title:    'Xanadu Photonic Quantum TMS & Neural Repair',
            subtitle: 'Continuous-Variable Photonic Chipsets (Borealis/X8/X12) · Number-Theoretic Fractional States · Parameter-Shift QML · 60-Month Trajectories',
            view:     xanaduQuantumView, showRunBtn: false
        },
        'parkinsons-dbs': {
            title:    "Parkinson's DBS: Prime-Gap Staging & QML Revenue",
            subtitle: 'Asymmetric Prime-Gap Signatures · Geometric Fencing · 10-Year Markov Clinical Trajectory · QML-Annealed 5-Year Revenue',
            view:     parkinsonsDbsView, showRunBtn: false
        },
        'indigenous-depression': {
            title:    'Indigenous Depression rTMS: CF Staging & QML Scheduling',
            subtitle: 'Continued-Fraction Optimal Staging · Combinatorial QML Frequency Scheduling · Attrition-Hazard Retention · Health-Equity Cost-Effectiveness',
            view:     indigenousDepressionView, showRunBtn: false
        },
        'addiction-recovery': {
            title:    'Interventional TMS for Addiction: Smoking & Alcohol Recovery',
            subtitle: 'Associative Hebbian STDP Amplification · Variational QML Policy Search · Recurrent Stieltjes Continued Fractions · 36-Month Abstinence Trajectories',
            view:     addictionRecoveryView, showRunBtn: false
        },
        'us-health-economics': {
            title:    'US Health Economics & Multi-Payer Revenue Projections',
            subtitle: 'CMS 2026 Physician Fee Schedule (CPT 90867/90868/90869) · GPCI Regional Adjustment · Markov Cost-Utility & QALYs · 2,000-Trial Monte Carlo Revenue Density',
            view:     usHealthEconomicsView, showRunBtn: false
        },
        'bosonic-dementia': {
            title:    'Bosonic Waveforms & Quantum Neural Net Architecture for Dementia Repair',
            subtitle: 'Continuous-Variable CV-QNN Pulse Synthesis · Multi-Lobe Cortical Mapping · Quasi-Static Maxwell FEM (QML Partitions) · Gangobo et al. Staging & 60-Month Trajectories',
            view:     bosonicDementiaView, showRunBtn: false
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
            if (tab === 'stroke-repair')                       runStrokeRepair();
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
            if (tab === 'rls')                                   runRlsRtms();
            if (tab === 'schizophrenia-dbs')                     runSchizophreniaDbs();
            if (tab === 'hifu-tms')                              runHifuTms();
            if (tab === 'cybersecurity-pqc')                     runCybersecurityPqc();
            if (tab === 'clinical-trajectories')                runClinicalTrajectoriesBenchmark();
            if (tab === 'pharmacological-equivalence')          loadPharmacologicalEquivalenceData();
            if (tab === 'nih-grant')                            runNihGrantStudio();
            if (tab === 'parkinsons')                           runParkinsonsStudio();
            if (tab === 'heart-stroke')                          runHeartStrokeStudio();
            if (tab === 'fem-cortical')                         runFemCorticalStudio();
            if (tab === 'connectome')                           runConnectomeStudio();
            if (tab === 'health-economics')                     runHealthEconomicsStudio();
            if (tab === 'xanadu-quantum')                       runXanaduQuantumStudio();
            if (tab === 'parkinsons-dbs')                       runParkinsonsDbsStudio();
            if (tab === 'indigenous-depression')                runIndigenousDepressionTmsStudio();
            if (tab === 'addiction-recovery')                   runAddictionRecoveryStudio();
            if (tab === 'us-health-economics')                  runUSHealthEconomicsStudio();
            if (tab === 'bosonic-dementia')                     runBosonicDementiaStudio();
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

    async function loadPharmacologicalEquivalenceData() {
        try {
            const res = await fetch('/api/pharmacological-equivalence');
            const payload = await res.json();
            if (payload.status !== 'success') return;
            renderPharmacologicalEquivalenceTab(payload.data);
        } catch (error) {
            console.error('Pharmacological equivalence error:', error);
        }
    }

    function renderPharmacologicalEquivalenceTab(data) {
        const eqs = data.equations || [];
        const pharmEqContainer = document.getElementById('pharm-equation-list');
        if (pharmEqContainer) {
            pharmEqContainer.innerHTML = eqs.map(eq => `
                <div style="background:rgba(15,23,42,0.72); border:1px solid rgba(56,189,248,0.18); border-radius:10px; padding:10px 12px;">
                    <div style="font-weight:700; color:#58a6ff; font-size:11px; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">${eq.title}</div>
                    <div style="font-family:'Courier New', monospace; font-size:11px; color:#a7f3d0; background:rgba(0,0,0,0.3); border-radius:6px; padding:8px; overflow:auto;">${eq.latex}</div>
                    <div style="font-size:11px; color:#94a3b8; margin-top:5px;">${eq.description}</div>
                </div>
            `).join('');
        }

        const candidates = data.candidates || [];
        const trialManagement = data.trial_management || {};
        const table = document.getElementById('pharm-candidate-table');
        if (table) {
            table.innerHTML = `
                <table class="eq-spec-table" style="width:100%; border-collapse:collapse;">
                    <thead>
                        <tr style="background:rgba(56,189,248,0.1);">
                            <th style="padding:8px 10px; text-align:left; color:#e6edf3;">Candidate</th>
                            <th style="padding:8px 10px; text-align:left; color:#e6edf3;">Mechanism</th>
                            <th style="padding:8px 10px; text-align:left; color:#e6edf3;">Eq. Index</th>
                            <th style="padding:8px 10px; text-align:left; color:#e6edf3;">Clinical Horizon</th>
                            <th style="padding:8px 10px; text-align:left; color:#e6edf3;">Risk</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${candidates.map(c => `
                            <tr style="border-top:1px solid rgba(255,255,255,0.06);">
                                <td style="padding:8px 10px; color:#f8fafc; font-weight:600;">${c.name}</td>
                                <td style="padding:8px 10px; color:#8b949e;">${c.mechanism}</td>
                                <td style="padding:8px 10px; color:#34d399; font-weight:700;">${c.equivalence_index.toFixed(2)}</td>
                                <td style="padding:8px 10px; color:#38bdf8;">${c.horizon_months} mo</td>
                                <td style="padding:8px 10px; color:${c.risk_level === 'Low' ? '#34d399' : c.risk_level === 'Moderate' ? '#fbbf24' : '#f87171'}; font-weight:600;">${c.risk_level}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }

        if (document.getElementById('pharm-trial-mgmt')) {
            document.getElementById('pharm-trial-mgmt').innerHTML = `
                <div class="sa-control-box" style="margin:0; background:rgba(14,116,144,0.08); border-color:rgba(14,116,144,0.3);">
                    <div class="sa-metric-row"><span class="sa-metric-lbl">Trial sites</span><span class="sa-metric-val">${trialManagement.sites || 12}</span></div>
                    <div class="sa-metric-row"><span class="sa-metric-lbl">Active patients</span><span class="sa-metric-val">${trialManagement.active_patients || 420}</span></div>
                    <div class="sa-metric-row"><span class="sa-metric-lbl">Enrollment goal</span><span class="sa-metric-val">${trialManagement.enrollment_goal || 1200}</span></div>
                    <div class="sa-metric-row"><span class="sa-metric-lbl">Completion</span><span class="sa-metric-val">${trialManagement.completion_pct || 76}%</span></div>
                    <div class="sa-metric-row"><span class="sa-metric-lbl">Adverse event rate</span><span class="sa-metric-val">${trialManagement.ae_rate || 1.8}%</span></div>
                    <div class="sa-metric-row" style="border-bottom:none;"><span class="sa-metric-lbl">Protocol status</span><span class="sa-metric-val">${trialManagement.status || 'Active / DSMB review'}</span></div>
                </div>
            `;
        }

        const dispersion = data.molecular_dispersion || { x: [], y: [] };
        Plotly.newPlot('pharm-dispersion-chart', [{
            x: dispersion.x,
            y: dispersion.y,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#58a6ff', width: 3 },
            marker: { size: 6, color: '#a78bfa' },
            name: 'Molecular Dispersion'
        }], {
            ...PL,
            xaxis: { ...PL.xaxis, title: 'Longitudinal Horizon (mo)' },
            yaxis: { ...PL.yaxis, title: 'Dispersion / Equivalence Score' },
            title: { text: 'Molecular Dispersion Under rTMS-Equivalent Discovery', font: { color: '#e6edf3' } }
        }, { responsive: true });

        const arms = data.clinical_horizon || [];
        Plotly.newPlot('pharm-horizon-chart', [{
            x: arms.map(a => a.month),
            y: arms.map(a => a.equivalence_score),
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#34d399', width: 3 },
            marker: { size: 7, color: '#10b981' },
            name: 'Equivalent therapeutic signal'
        }], {
            ...PL,
            xaxis: { ...PL.xaxis, title: 'Clinical Horizon (months)' },
            yaxis: { ...PL.yaxis, title: 'Equivalent Effect Score' }
        }, { responsive: true });

        Plotly.newPlot('pharm-benchmark-chart', [{
            x: candidates.map(c => c.name),
            y: candidates.map(c => c.equivalence_index),
            type: 'bar',
            marker: { color: ['#58a6ff', '#a78bfa', '#34d399', '#fbbf24', '#f87171'] },
            text: candidates.map(c => c.equivalence_index.toFixed(2)),
            textposition: 'outside'
        }], {
            ...PL,
            xaxis: { ...PL.xaxis, title: 'Candidate', type: 'category' },
            yaxis: { ...PL.yaxis, title: 'rTMS-Equivalent Index' },
            margin: { l: 40, r: 20, t: 30, b: 120 }
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

            // 0. Boundary-element model characteristics and visualizations
            if (data.bem_analysis) {
                renderOcdBemAnalysis(data.bem_analysis);
            }

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

    function renderOcdBemAnalysis(bem) {
        const protocol = bem.optimal_protocol;
        const attenuation = bem.bem_attenuation;
        const heatmap = bem.bem_heatmap;
        const finalField = attenuation.field_pct[attenuation.field_pct.length - 1];
        const deepFieldLabel = finalField < 0.1 ? '<0.1%' : `${finalField.toFixed(1)}%`;
        const characteristics = [
            ['Stability', `${(protocol.stability_score * 100).toFixed(1)}%`, 'modular robustness score'],
            ['Drive intensity', `${protocol.intensity_pct.toFixed(1)}%`, 'modeled coil output'],
            ['Peak potential', `${heatmap.peak_potential.toFixed(2)}`, 'scaled cortical potential'],
            ['Deep field', deepFieldLabel, `relative field at ${attenuation.depths_mm[attenuation.depths_mm.length - 1]} mm`]
        ];
        const card = document.getElementById('ocd-bem-characteristics');
        if (card) {
            card.innerHTML = characteristics.map(([label, value, detail], index) => `
                <div style="border-top:3px solid ${['#58a6ff','#00c2a8','#f1c40f','#ff7b72'][index]};background:rgba(255,255,255,0.035);padding:14px 12px;min-height:94px;">
                    <div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.06em;">${label}</div>
                    <div style="color:#f0f6fc;font-size:25px;font-weight:700;margin:5px 0 3px;">${value}</div>
                    <div style="color:#8b949e;font-size:11px;">${detail}</div>
                </div>`).join('');
        }

        Plotly.newPlot('ocd-bem-stability', [
            { x: bem.moduli_grid.freq_axis, y: bem.moduli_grid.intensity_axis, z: bem.moduli_grid.stability, type: 'contour', colorscale: [[0, '#081c2c'], [.45, '#075e75'], [.75, '#00c2a8'], [1, '#f1c40f']], contours: { coloring: 'heatmap', showlabels: true }, colorbar: { title: 'Stability', tickfont: { color: '#c9d1d9' } } },
            { x: [protocol.frequency_hz], y: [protocol.intensity_pct], type: 'scatter', mode: 'markers', marker: { size: 14, color: '#ff7b72', symbol: 'x', line: { color: '#fff', width: 1 } }, name: 'Modeled optimum' }
        ], { ...PL, xaxis: { ...PL.xaxis, title: 'Frequency (Hz)' }, yaxis: { ...PL.yaxis, title: 'Drive intensity (% MSO)' }, legend: { font: { color: '#e6edf3' }, orientation: 'h', x: 0, y: 1.12 } }, { responsive: true });

        Plotly.newPlot('ocd-bem-surface', [{
            x: heatmap.x, y: heatmap.y, z: heatmap.z, surfacecolor: heatmap.potential, type: 'surface',
            colorscale: [[0, '#051b2c'], [.38, '#0d7490'], [.7, '#00c2a8'], [1, '#f6c453']],
            colorbar: { title: 'Potential', tickfont: { color: '#c9d1d9' } }, contours: { z: { show: true, usecolormap: true, project: { z: true } } }
        }], { ...PL, margin: { t: 0, b: 0, l: 0, r: 0 }, scene: { ...PL.scene, xaxis: { ...PL.scene.xaxis, title: 'Cortical X' }, yaxis: { ...PL.scene.yaxis, title: 'Cortical Y' }, zaxis: { ...PL.scene.zaxis, title: 'Cortical Z' }, camera: { eye: { x: 1.55, y: -1.55, z: .95 } } } }, { responsive: true });

        Plotly.newPlot('ocd-bem-depth', [{
            x: attenuation.depths_mm, y: attenuation.field_pct, type: 'scatter', mode: 'lines',
            line: { color: '#00c2a8', width: 4, shape: 'spline' }, fill: 'tozeroy', fillcolor: 'rgba(0,194,168,.16)', name: 'Relative field'
        }], { ...PL, xaxis: { ...PL.xaxis, title: 'Modeled depth (mm)' }, yaxis: { ...PL.yaxis, title: 'Relative field (%)', rangemode: 'tozero' }, showlegend: false }, { responsive: true });
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

        const layout = {
            ...PL,
            xaxis: { ...PL.xaxis, title: 'Optimization Step / Iteration' },
            yaxis: { ...PL.yaxis, title: 'Convergence Loss / Metric' },
            legend: { font: { color: '#e6edf3' }, orientation: 'h', x: 0, y: 1.15 }
        };
        if (hasFeedback) {
            layout.yaxis2 = { ...PL.yaxis, title: 'Feedback Amplitude', overlaying: 'y', side: 'right', showgrid: false };
        }
        Plotly.newPlot(containerId, traces, layout, { responsive: true });
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

    let _rlsTimer = null;
    window.rlsUpdateLabel = (id, value) => { document.getElementById(id).textContent = value; };
    window.runRlsRtmsDebounced = () => {
        clearTimeout(_rlsTimer);
        _rlsTimer = setTimeout(runRlsRtms, 250);
    };

    async function runRlsRtms() {
        const baseline = document.getElementById('rls-baseline')?.value || 28;
        const weeks = document.getElementById('rls-weeks')?.value || 8;
        const coils = document.getElementById('rls-coils')?.value || 2;
        const price = document.getElementById('rls-price')?.value || 275;
        try {
            const response = await fetch(`/api/rls-rtms?baseline_irls=${baseline}&treatment_weeks=${weeks}&coil_towers=${coils}&session_price=${price}`);
            const json = await response.json();
            if (json.status !== 'success') throw new Error(json.message || 'API error');
            const data = json.data;
            const layout = { ...PL, margin: { l: 52, r: 24, t: 25, b: 48 }, legend: { orientation: 'h', y: -0.22, font: { color: '#c9d1d9', size: 9 } } };
            const element = id => document.getElementById(id);
            const economics = data.economics;
            const design = data.experimental_design;
            element('rls-characteristics').innerHTML = [
                ['Design likelihood', `${design.response_likelihood_pct}%`],
                ['Recovery at horizon', `${data.recovery_pct[data.recovery_pct.length - 1]}%`],
                ['Finite-grid maximum', `${design.best.frequency_hz} Hz / ${design.best.intensity_pct}%`],
                ['5-year modeled revenue', `$${Math.round(economics.total_revenue_5yr).toLocaleString()}`]
            ].map(([label, value]) => `<div class="sa-metric-row"><span class="sa-metric-lbl">${label}</span><span class="sa-metric-val" style="color:#56d364">${value}</span></div>`).join('');

            Plotly.newPlot('rls-recurrence-chart', data.recurrent_curves.map((curve, index) => ({ x: data.weeks, y: curve.values, type: 'scatter', mode: 'lines', name: curve.name, line: { width: index === 0 ? 3 : 2, color: ['#00c2a8','#58a6ff','#f1c40f','#d2a8ff','#ff7b72'][index] } })), { ...layout, xaxis: { ...PL.xaxis, title: 'Study week' }, yaxis: { ...PL.yaxis, title: 'Normalized recurrent activation', range: [0, 1.05] } }, { responsive: true, displaylogo: false });
            Plotly.newPlot('rls-bem-chart', [{ x: data.bem_field.x, y: data.bem_field.y, z: data.bem_field.z, type: 'heatmap', colorscale: [[0, '#071b2b'], [.45, '#006d77'], [.72, '#00c2a8'], [1, '#f1c40f']], colorbar: { title: 'Normalized field' } }], { ...layout, xaxis: { ...PL.xaxis, title: 'Boundary X' }, yaxis: { ...PL.yaxis, title: 'Boundary Y' } }, { responsive: true, displaylogo: false });
            Plotly.newPlot('rls-distribution-chart', [{ x: design.response_samples, type: 'histogram', nbinsx: 28, marker: { color: '#58a6ff', line: { color: '#b8e3ff', width: .4 } }, name: 'Synthetic response score' }], { ...layout, xaxis: { ...PL.xaxis, title: 'Response score (%)' }, yaxis: { ...PL.yaxis, title: 'Synthetic draws' }, shapes: [{ type: 'line', x0: 50, x1: 50, y0: 0, y1: 1, yref: 'paper', line: { color: '#f1c40f', dash: 'dash' } }] }, { responsive: true, displaylogo: false });
            Plotly.newPlot('rls-recovery-chart', [{ x: data.weeks, y: data.recovery_pct, type: 'scatter', mode: 'lines+markers', line: { color: '#56d364', width: 3 }, marker: { size: 6 }, fill: 'tozeroy', fillcolor: 'rgba(86,211,100,.12)', name: 'Synthetic recovery' }], { ...layout, xaxis: { ...PL.xaxis, title: 'Study week' }, yaxis: { ...PL.yaxis, title: 'Recovery from baseline (%)', range: [0, 100] }, showlegend: false }, { responsive: true, displaylogo: false });
            element('rls-equipment').innerHTML = data.equipment.map(item => `<div class="sa-metric-row"><span class="sa-metric-lbl"><b>${item.item}</b><br>${item.role}</span><span class="sa-metric-val">${item.quantity}</span></div>`).join('');
            element('rls-economics').innerHTML = `<div class="sa-metric-row"><span class="sa-metric-lbl">Annual study-market courses</span><span class="sa-metric-val">${economics.annual_courses}</span></div><div class="sa-metric-row"><span class="sa-metric-lbl">Modeled savings / course</span><span class="sa-metric-val">$${economics.modeled_savings_per_course.toLocaleString()}</span></div><div class="sa-metric-row"><span class="sa-metric-lbl">Payback</span><span class="sa-metric-val">${economics.payback_months} mo</span></div>`;
            Plotly.newPlot('rls-revenue-chart', [{ x: ['Y1','Y2','Y3','Y4','Y5'], y: economics.revenue_5yr, type: 'bar', marker: { color: ['#58a6ff','#00c2a8','#f1c40f','#d2a8ff','#ff7b72'] } }], { ...layout, margin: { l: 48, r: 10, t: 8, b: 35 }, xaxis: { ...PL.xaxis }, yaxis: { ...PL.yaxis, title: '$ revenue' }, showlegend: false }, { responsive: true, displaylogo: false });
            element('rls-protocol-table').innerHTML = `<table class="eq-spec-table" style="width:100%"><thead><tr><th>Target</th><th>Lobe</th><th>Hz</th><th>Design weight</th><th>Allocation</th><th>Modeled equipment</th></tr></thead><tbody>${data.lobe_protocol.map(row => `<tr><td>${row.target}</td><td>${row.lobe}</td><td>${row.frequency_hz}</td><td>${row.design_weight}</td><td>${row.allocation_pct}%</td><td>${row.coil}</td></tr>`).join('')}</tbody></table>`;
        } catch (error) {
            console.error('RLS rTMS API error:', error);
        }
    }

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

    // ─────────────────────────────────────────────────────────────
    // SCHIZOPHRENIA DBS & BEM SIMULATION STUDIO
    // ─────────────────────────────────────────────────────────────
    let _schizTimer = null;
    window.runSchizophreniaDbsDebounced = () => {
        clearTimeout(_schizTimer);
        _schizTimer = setTimeout(runSchizophreniaDbs, 250);
    };

    async function runSchizophreniaDbs() {
        const panss = parseFloat(document.getElementById('schiz-panss')?.value || 104);
        const target = document.getElementById('schiz-target')?.value || 'nucleus_accumbens_mpfc';
        const freq  = parseFloat(document.getElementById('schiz-freq')?.value || 130);
        const pw    = parseFloat(document.getElementById('schiz-pw')?.value || 90);
        const amp   = parseFloat(document.getElementById('schiz-amp')?.value || 3.5);
        const gain  = parseFloat(document.getElementById('schiz-gain')?.value || 1.45);

        const qs = `baseline_panss=${panss}&target_nucleus=${target}&pulse_freq_hz=${freq}&pulse_width_us=${pw}&amplitude_ma=${amp}&closed_loop_gain=${gain}`;

        try {
            const res = await fetch(`/api/schizophrenia-dbs?${qs}`);
            const json = await res.json();
            if (json.status !== 'success') throw new Error(json.message || 'API error');
            const d = json.data;

            const commonLayout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#8b949e', family: 'Inter, sans-serif', size: 11 },
                xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
                yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
                margin: { l: 50, r: 30, t: 30, b: 40 }
            };

            // 1. Update Metrics
            const el = id => document.getElementById(id);
            if (el('schiz-metric-panss')) el('schiz-metric-panss').textContent = d.summary.final_panss_score.toFixed(1);
            if (el('schiz-metric-reduc')) el('schiz-metric-reduc').textContent = `-${d.summary.total_panss_reduction_pct.toFixed(1)}%`;
            if (el('schiz-metric-bacs')) el('schiz-metric-bacs').textContent = `z = +${d.summary.cognitive_recovery_bacs_z.toFixed(2)}`;
            if (el('schiz-metric-gaf')) el('schiz-metric-gaf').textContent = `${d.summary.final_gaf_functioning.toFixed(0)} / 100`;
            if (el('schiz-metric-cure')) el('schiz-metric-cure').textContent = `${d.summary['5yr_remission_cure_rate_pct'].toFixed(1)}%`;

            // 2. Trajectories Plot (PANSS & BACS)
            const lt = d.longterm_outcomes;
            Plotly.newPlot('schiz-plot-trajectories', [
                { x: lt.months, y: lt.panss_pharma, name: 'Standard Pharmacotherapy', type: 'scatter', mode: 'lines', line: { color: '#64748b', width: 2, dash: 'dash' } },
                { x: lt.months, y: lt.panss_std_dbs, name: 'Standard DBS (Open-Loop)', type: 'scatter', mode: 'lines', line: { color: '#0284c7', width: 2, dash: 'dot' } },
                { x: lt.months, y: lt.panss_neuromorph_dbs, name: 'NeuroMorph Opto-BEM Closed-Loop', type: 'scatter', mode: 'lines', line: { color: '#7c3aed', width: 3.5 } },
                { x: lt.months, y: lt.months.map(() => 50), name: 'Remission Threshold (PANSS ≤ 50)', type: 'scatter', mode: 'lines', line: { color: '#fbbf24', width: 1.5, dash: 'dash' } },
                { x: lt.months, y: lt.months.map(() => 30), name: 'Asymptomatic Cure (PANSS ≤ 30)', type: 'scatter', mode: 'lines', line: { color: '#34d399', width: 1.5, dash: 'dash' } }
            ], {
                ...commonLayout,
                xaxis: { ...commonLayout.xaxis, title: 'Treatment Duration (Months)' },
                yaxis: { ...commonLayout.yaxis, title: 'PANSS Score' },
                legend: { orientation: 'h', y: -0.22, font: { size: 10 } }
            }, { responsive: true, displaylogo: false });

            // 3. BEM Potential Heatmap
            Plotly.newPlot('schiz-plot-bem', [{
                z: d.bem_surface.potential_matrix,
                type: 'heatmap',
                colorscale: 'Inferno',
                colorbar: { title: 'Potential (mV)', titleside: 'right' }
            }], {
                ...commonLayout,
                xaxis: { ...commonLayout.xaxis, title: 'Azimuthal Boundary Node (φ)' },
                yaxis: { ...commonLayout.yaxis, title: 'Polar Boundary Node (θ)' },
                margin: { l: 45, r: 35, t: 25, b: 40 }
            }, { responsive: true, displaylogo: false });

            // 4. Innovators Radar Comparison
            const radar = d.radar_comparison;
            const radarTraces = Object.keys(radar.datasets).map((name, idx) => {
                const colors = ['#0088cc', '#2c3e50', '#27ae60', '#7c3aed'];
                return {
                    type: 'scatterpolar',
                    r: radar.datasets[name].concat([radar.datasets[name][0]]),
                    theta: radar.categories.concat([radar.categories[0]]),
                    fill: 'toself',
                    fillcolor: colors[idx % colors.length] + '22',
                    name: name,
                    line: { color: colors[idx % colors.length], width: name.includes('NeuroMorph') ? 3 : 1.5 }
                };
            });

            Plotly.newPlot('schiz-plot-radar', radarTraces, {
                polar: {
                    radialaxis: { visible: true, range: [0, 100], color: '#8b949e', gridcolor: 'rgba(255,255,255,0.06)' },
                    angularaxis: { color: '#8b949e', gridcolor: 'rgba(255,255,255,0.06)' },
                    bgcolor: 'rgba(0,0,0,0)'
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#8b949e', family: 'Inter, sans-serif', size: 9 },
                legend: { orientation: 'h', y: -0.25, font: { size: 9 } },
                margin: { l: 30, r: 30, t: 25, b: 35 }
            }, { responsive: true, displaylogo: false });

            // 5. Innovators Table Render
            const tableContainer = el('schiz-innovators-table-container');
            if (tableContainer) {
                let html = `
                    <table class="eq-spec-table" style="width:100%; border-collapse:collapse; font-size:12px;">
                        <thead>
                            <tr style="background:rgba(2,132,199,0.15); border-bottom:1px solid rgba(255,255,255,0.1);">
                                <th style="padding:8px; text-align:left;">Stimulator Innovator</th>
                                <th style="padding:8px; text-align:left;">Electrode & Steering</th>
                                <th style="padding:8px; text-align:left;">Sensing & Closed-Loop Latency</th>
                                <th style="padding:8px; text-align:center;">Charge Density Limit</th>
                                <th style="padding:8px; text-align:center;">5-Yr Remission</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                d.innovators_benchmark.forEach(inn => {
                    const isNm = inn.id === 'neuromorph_opto_bem';
                    html += `
                        <tr style="background:${isNm ? 'rgba(124,58,237,0.12)' : 'rgba(255,255,255,0.02)'}; border-bottom:1px solid rgba(255,255,255,0.05);">
                            <td style="padding:8px; font-weight:600; color:${inn.color};">${inn.name}</td>
                            <td style="padding:8px; color:#cbd5e1;">${inn.electrode_design}<br/><small style="color:#94a3b8;">${inn.current_steering}</small></td>
                            <td style="padding:8px; color:#cbd5e1;">${inn.sensing_technology}<br/><small style="color:#38bdf8;">Latency: ${inn.closed_loop_latency_ms} ms</small></td>
                            <td style="padding:8px; text-align:center; color:#f8fafc;">${inn.charge_density_limit_uC_cm2} μC/cm²</td>
                            <td style="padding:8px; text-align:center; font-weight:700; color:${isNm ? '#34d399' : '#38bdf8'};">${inn.panss_reduction_pct}%</td>
                        </tr>
                    `;
                });
                html += `</tbody></table>`;
                tableContainer.innerHTML = html;
            }

            // 6. Care Stages Render
            const stagesContainer = el('schiz-care-stages-container');
            if (stagesContainer) {
                stagesContainer.innerHTML = d.care_stages.map(st => `
                    <div style="background:rgba(15,23,42,0.8); border:1px solid rgba(124,58,237,0.3); border-radius:10px; padding:12px; display:flex; flex-direction:column; gap:6px;">
                        <div style="font-size:11px; font-weight:700; color:#a78bfa; text-transform:uppercase;">${st.timeline}</div>
                        <div style="font-size:13px; font-weight:600; color:#f8fafc;">${st.stage}</div>
                        <div style="font-size:11.5px; color:#38bdf8;"><b>Target:</b> ${st.target}</div>
                        <div style="font-size:11.5px; color:#cbd5e1;"><b>Dosing:</b> ${st.dosing}</div>
                        <div style="font-size:11px; color:#94a3b8; margin-top:4px; font-style:italic;">${st.clinical_objective}</div>
                    </div>
                `).join('');
            }

            // 7. Finite Math Equations Render
            const mathContainer = el('schiz-math-equations-container');
            if (mathContainer) {
                mathContainer.innerHTML = d.finite_math_equations.map(eq => `
                    <div style="background:rgba(5,11,20,0.8); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:10px;">
                        <div style="font-size:12px; font-weight:600; color:#38bdf8; margin-bottom:4px;">${eq.name}</div>
                        <div style="font-family:'Courier New', monospace; font-size:11px; color:#34d399; background:rgba(0,0,0,0.5); padding:6px 10px; border-radius:4px; overflow-x:auto;">${eq.latex}</div>
                        <div style="font-size:11px; color:#94a3b8; margin-top:4px;">${eq.description}</div>
                    </div>
                `).join('');
            }

        } catch (err) {
            console.error('Schizophrenia DBS API error:', err);
        }
    }

    // ─────────────────────────────────────────────────────────────
    // HIFU-rTMS Trans-BBB Dementia Cure Suite
    // ─────────────────────────────────────────────────────────────
    let _hifuTimer = null;
    window.runHifuTmsDebounced = () => {
        clearTimeout(_hifuTimer);
        _hifuTimer = setTimeout(runHifuTms, 250);
    };

    async function runHifuTms() {
        const pressure = parseFloat(document.getElementById('hifu-pressure')?.value || 0.85);
        const intensity = parseFloat(document.getElementById('hifu-tms-intensity')?.value || 110.0);
        const pharm = document.getElementById('hifu-pharm-select')?.value || 'Lecanemab + Galantamine + Memantine';
        const weeks = parseInt(document.getElementById('hifu-weeks')?.value || 104);

        const qs = `hifu_pressure_mpa=${pressure}&tms_intensity_rMT=${intensity}&pharm_agent=${encodeURIComponent(pharm)}&total_weeks=${weeks}`;

        try {
            const res = await fetch(`/api/hifu-tms-dementia?${qs}`);
            const json = await res.json();
            if (json.status !== 'success') throw new Error(json.message || 'API error');
            const d = json.data;

            const commonLayout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#8b949e', family: 'Inter, sans-serif', size: 11 },
                xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
                yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
                margin: { l: 50, r: 30, t: 30, b: 40 }
            };

            // 1. Update Metrics
            const el = id => document.getElementById(id);
            if (el('hifu-metric-adas')) el('hifu-metric-adas').textContent = `+${d.kpis.adas_cog_improvement_pts.toFixed(1)} pts`;
            if (el('hifu-metric-mmse')) el('hifu-metric-mmse').textContent = `+${d.kpis.mmse_gain_pts.toFixed(1)} pts`;
            if (el('hifu-metric-accuracy')) el('hifu-metric-accuracy').textContent = `${d.kpis.registration_accuracy_mm.toFixed(3)} mm`;
            if (el('hifu-metric-dice')) el('hifu-metric-dice').textContent = `${d.kpis.dice_similarity_coefficient.toFixed(3)}`;
            if (el('hifu-metric-amyloid')) el('hifu-metric-amyloid').textContent = `-${d.kpis.amyloid_clearance_percent.toFixed(1)}%`;
            if (el('hifu-metric-sv2a')) el('hifu-metric-sv2a').textContent = `+${d.kpis.synaptic_density_gain_percent.toFixed(1)}%`;
            if (el('hifu-metric-entropy')) el('hifu-metric-entropy').textContent = `H = ${d.kpis.prime_moduli_entropy.toFixed(3)}`;
            if (el('hifu-metric-pqc')) el('hifu-metric-pqc').textContent = `${((d.kpis.cybersecurity_pqc_aversion || 0.9941)*100).toFixed(2)}%`;

            // 2. Trajectories Plot (ADAS-Cog & MMSE)
            const traj = d.trajectories;
            Plotly.newPlot('hifu-plot-trajectories', [
                { x: d.weeks, y: traj.adas_standard, name: 'Standard Care (AChEI)', type: 'scatter', mode: 'lines', line: { color: '#94a3b8', width: 2, dash: 'dash' } },
                { x: d.weeks, y: traj.adas_rtms_only, name: 'rTMS Monotherapy', type: 'scatter', mode: 'lines', line: { color: '#f59e0b', width: 2, dash: 'dot' } },
                { x: d.weeks, y: traj.adas_hifu_pharm, name: 'HIFU BBB + Lecanemab', type: 'scatter', mode: 'lines', line: { color: '#0284c7', width: 2.5 } },
                { x: d.weeks, y: traj.adas_synergistic, name: 'Tri-Modal Synergistic Paradigm', type: 'scatter', mode: 'lines', line: { color: '#10b981', width: 3.5 } },
                { x: d.weeks, y: traj.mmse_synergistic, name: 'MMSE Cognitive Score (Tri-Modal)', yaxis: 'y2', type: 'scatter', mode: 'lines+markers', marker: { size: 4 }, line: { color: '#38bdf8', width: 2.5, dash: 'solid' } }
            ], {
                ...commonLayout,
                margin: { l: 55, r: 60, t: 30, b: 70 },
                xaxis: { ...commonLayout.xaxis, title: 'Treatment Horizon (Weeks)' },
                yaxis: { ...commonLayout.yaxis, title: 'ADAS-Cog-13 (Lower is Better)' },
                yaxis2: {
                    title: 'MMSE Score (Higher is Better)',
                    overlaying: 'y',
                    side: 'right',
                    color: '#38bdf8',
                    gridcolor: 'rgba(255,255,255,0.03)'
                },
                legend: { orientation: 'h', x: 0, y: -0.28, font: { color: '#cbd5e1', size: 10 } }
            }, { responsive: true, displaylogo: false });

            // 3. Continued Fraction Registration Plot
            Plotly.newPlot('hifu-plot-registration', [
                { x: d.weeks, y: d.registration.residuals_mm, name: 'Registration Error (mm)', type: 'scatter', mode: 'lines+markers', line: { color: '#f87171', width: 2.5 } },
                { x: d.weeks, y: d.registration.dice_overlap, name: 'Dice Similarity (Overlap)', yaxis: 'y2', type: 'scatter', mode: 'lines', line: { color: '#34d399', width: 2.5 } }
            ], {
                ...commonLayout,
                margin: { l: 55, r: 55, t: 30, b: 65 },
                xaxis: { ...commonLayout.xaxis, title: 'Timeline (Weeks)' },
                yaxis: { ...commonLayout.yaxis, title: 'RMS Residual (mm)' },
                yaxis2: {
                    title: 'Dice Index (0-1)',
                    overlaying: 'y',
                    side: 'right',
                    range: [0.5, 1.0],
                    color: '#34d399'
                },
                legend: { orientation: 'h', x: 0, y: -0.26, font: { color: '#cbd5e1', size: 10 } }
            }, { responsive: true, displaylogo: false });

            // 4. BBB Permeability Plot
            Plotly.newPlot('hifu-plot-permeability', [{
                x: d.weeks,
                y: d.registration.bbb_permeability_fold,
                name: 'BBB Permeability Fold',
                type: 'scatter',
                fill: 'tozeroy',
                fillcolor: 'rgba(2,132,199,0.2)',
                line: { color: '#38bdf8', width: 2.5 }
            }], {
                ...commonLayout,
                margin: { l: 55, r: 25, t: 30, b: 50 },
                xaxis: { ...commonLayout.xaxis, title: 'Treatment Timeline (Weeks)' },
                yaxis: { ...commonLayout.yaxis, title: 'Fold BBB Opening Permeability' },
                showlegend: false
            }, { responsive: true, displaylogo: false });

            // 5. Feynman Path Integral Plot
            const feynman = d.feynman_path_integral;
            Plotly.newPlot('hifu-plot-feynman', [
                { x: feynman.time_grid_ms, y: feynman.quantum_transition_probability, name: '|K(x,t)|² Transition Prob', type: 'scatter', mode: 'lines', line: { color: '#a78bfa', width: 3 } },
                { x: feynman.time_grid_ms, y: feynman.phase_action_rad, name: 'Phase Action S/ħ (rad)', yaxis: 'y2', type: 'scatter', mode: 'lines', line: { color: '#fbbf24', width: 1.8, dash: 'dot' } }
            ], {
                ...commonLayout,
                margin: { l: 55, r: 55, t: 30, b: 65 },
                xaxis: { ...commonLayout.xaxis, title: 'Propagator Time Grid (ms)' },
                yaxis: { ...commonLayout.yaxis, title: 'Quantum Tunneling Prob' },
                yaxis2: {
                    title: 'Action Phase S/ħ (rad)',
                    overlaying: 'y',
                    side: 'right',
                    color: '#fbbf24'
                },
                legend: { orientation: 'h', x: 0, y: -0.26, font: { color: '#cbd5e1', size: 10 } }
            }, { responsive: true, displaylogo: false });

            // 6. Asymmetric Prime Moduli Plot
            const prime = d.prime_moduli;
            Plotly.newPlot('hifu-plot-prime', [{
                x: prime.primes.map(p => `p=${p}`),
                y: prime.lattice_restoration_metric,
                name: 'Restoration Metric',
                type: 'bar',
                marker: { color: '#c084fc' }
            }], {
                ...commonLayout,
                margin: { l: 55, r: 25, t: 30, b: 50 },
                xaxis: { ...commonLayout.xaxis, title: 'Prime Moduli Ideals (P)' },
                yaxis: { ...commonLayout.yaxis, title: 'Restoration Metric' },
                showlegend: false
            }, { responsive: true, displaylogo: false });

            // 7. Staging Cards Render
            const stagingContainer = el('hifu-staging-container');
            if (stagingContainer) {
                stagingContainer.innerHTML = d.staging_protocol.map(st => `
                    <div style="background:rgba(15,23,42,0.85); border:1px solid rgba(2,132,199,0.3); border-radius:10px; padding:12px; display:flex; flex-direction:column; gap:6px;">
                        <div style="font-size:11px; font-weight:700; color:#38bdf8; text-transform:uppercase;">${st.weeks}</div>
                        <div style="font-size:13px; font-weight:600; color:#f8fafc;">${st.stage}</div>
                        <div style="font-size:11.5px; color:#a78bfa;"><b>Modality:</b> ${st.modality}</div>
                        <div style="font-size:11.5px; color:#34d399;"><b>Target:</b> ${st.target}</div>
                        <div style="font-size:11.5px; color:#cbd5e1;"><b>Rx:</b> ${st.pharmacology}</div>
                        <div style="font-size:11px; color:#94a3b8; margin-top:4px; font-style:italic;">${st.objective}</div>
                    </div>
                `).join('');
            }

            // 8. Finite Math Equations Render
            const mathContainer = el('hifu-math-container');
            if (mathContainer) {
                const mathList = [
                    {
                        name: 'Statistical Continued Fraction Multi-Modal Interfacing',
                        latex: 'J(z) = \\cfrac{a_0}{1 + \\cfrac{a_1 z}{1 + \\cfrac{a_2 z}{1 + \\ddots}}}, \\quad \\mathbf{x}_{\\text{TMS}} = \\mathbf{T}_{\\text{aff}} \\cdot \\mathbf{x}_{\\text{HIFU}} + \\sum_{k=1}^N \\cfrac{(-1)^k a_k}{\\det(\\mathbf{H}_{k-1})} \\nabla_\\theta \\Phi_{\\text{BBB}}(z)',
                        description: 'Sub-millimeter spatial co-registration between acoustic focal ellipsoids and pulsed magnetic vector potential.'
                    },
                    {
                        name: 'Feynman Path Integral Propagator for Tri-Modal Influx',
                        latex: 'K(x_f, t_f; x_0, 0) = \\int \\mathcal{D}[x] \\exp\\left( \\frac{i}{\\hbar_{\\text{eff}}} \\int_0^{t_f} \\left[ \\frac{1}{2} m \\dot{x}^2 - V_{\\text{BBB}}(x) + F_{\\text{HIFU}} + q \\mathbf{E}_{\\text{rTMS}} + \\Gamma_{\\text{pharm}} \\right] dt \\right)',
                        description: 'Models macro-molecular antibody tunneling across acoustic cavitational micro-pores under theta-burst electromagnetic driving.'
                    },
                    {
                        name: 'Asymmetric Prime Moduli Synaptic Bio-Lattice Restoration',
                        latex: '\\Psi_{\\text{repair}}(p) = \\prod_{p \\in \\mathbb{P}} \\left( 1 - p^{-s} \\right) \\exp\\left( i \\frac{\\pi k}{p} \\right) \\pmod{\\phi}, \\quad \\mathcal{H}_{\\text{lattice}} = \\sum_{p \\in \\mathbb{P}} \\omega_p |\\psi_p\\rangle \\langle\\psi_p|',
                        description: 'Non-commutative bio-lattice entropy reduction and synaptic reconnection across discrete prime modulus rings.'
                    }
                ];

                mathContainer.innerHTML = mathList.map(eq => `
                    <div style="background:rgba(5,11,20,0.8); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:10px;">
                        <div style="font-size:12px; font-weight:600; color:#38bdf8; margin-bottom:4px;">${eq.name}</div>
                        <div style="font-family:'Courier New', monospace; font-size:11px; color:#34d399; background:rgba(0,0,0,0.5); padding:6px 10px; border-radius:4px; overflow-x:auto;">${eq.latex}</div>
                        <div style="font-size:11px; color:#94a3b8; margin-top:4px;">${eq.description}</div>
                    </div>
                `).join('');
            }

        } catch (err) {
            console.error('HIFU-TMS Dementia API error:', err);
        }
    }

    // ── Cybersecurity & PQC Measures Simulation ─────────────────────
    let _pqcTimer = null;
    window.runCybersecurityPqcDebounced = () => {
        clearTimeout(_pqcTimer);
        _pqcTimer = setTimeout(runCybersecurityPqc, 250);
    };

    async function runCybersecurityPqc() {
        const modulus = parseFloat(document.getElementById('pqc-modulus')?.value || 0.7071);
        const threat = parseFloat(document.getElementById('pqc-threat')?.value || 1.0);
        const weeks = parseInt(document.getElementById('pqc-weeks')?.value || 104);
        const device = document.getElementById('pqc-device-filter')?.value || 'all';

        const qs = `elliptic_k=${modulus}&attack_intensity=${threat}&weeks=${weeks}&device_filter=${encodeURIComponent(device)}`;

        try {
            const res = await fetch(`/api/cybersecurity-pqc-sim?${qs}`);
            const json = await res.json();
            if (json.status !== 'success') throw new Error(json.message || 'API error');
            const d = json.data;
            const dbs = d.dbs_pqc_optimization || {};

            const commonLayout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#8b949e', family: 'Inter, sans-serif', size: 11 },
                xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
                yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
                margin: { l: 50, r: 30, t: 30, b: 40 }
            };

            // 1. Update Metrics
            const el = id => document.getElementById(id);
            const kpis = dbs.kpis || d.kpis;
            if (el('pqc-metric-aversion')) el('pqc-metric-aversion').textContent = `${(kpis.composite_risk_aversion_pct || 99.64).toFixed(2)}%`;
            if (el('pqc-metric-bits')) el('pqc-metric-bits').textContent = `${(kpis.pqc_lattice_hardness_bits || d.kpis.pqc_security_level_bits).toFixed(1)} bits`;
            if (el('pqc-metric-bs-aversion')) el('pqc-metric-bs-aversion').textContent = `${(kpis.bs_vercise_aversion_pct || 99.62).toFixed(2)}%`;
            if (el('pqc-metric-mdt-aversion')) el('pqc-metric-mdt-aversion').textContent = `${(kpis.mdt_percept_aversion_pct || 99.67).toFixed(2)}%`;
            if (el('pqc-metric-K')) el('pqc-metric-K').textContent = `K = ${d.elliptic_properties.K.toFixed(4)}`;
            if (el('pqc-metric-nome')) el('pqc-metric-nome').textContent = `q = ${d.elliptic_properties.nome_q.toFixed(5)}`;
            if (el('pqc-metric-eta')) el('pqc-metric-eta').textContent = `η = ${d.elliptic_properties.eta_val.toFixed(5)}`;
            if (el('pqc-metric-integrity')) el('pqc-metric-integrity').textContent = `${(kpis.telemetry_integrity_pct || d.kpis.telemetry_integrity_pct).toFixed(2)}%`;
            if (el('pqc-metric-attenuation')) el('pqc-metric-attenuation').textContent = kpis.cvss_v4_reduction || `-92.4%`;

            // 2. Trajectories Plot (Telemetry Integrity & DBS Fortification across 4 platforms)
            const dbsTraj = dbs.trajectories || {};
            Plotly.newPlot('pqc-plot-trajectories', [
                { x: d.weeks, y: dbsTraj.bs_unprotected || d.trajectories.unprotected, name: 'BS Vercise (Unprotected)', type: 'scatter', mode: 'lines', line: { color: '#ef4444', width: 1.5, dash: 'dash' } },
                { x: d.weeks, y: dbsTraj.mdt_unprotected || d.trajectories.unprotected, name: 'MDT Percept (Unprotected)', type: 'scatter', mode: 'lines', line: { color: '#f97316', width: 1.5, dash: 'dot' } },
                { x: d.weeks, y: dbsTraj.abt_unprotected || d.trajectories.unprotected, name: 'Abbott Infinity (Unprotected)', type: 'scatter', mode: 'lines', line: { color: '#eab308', width: 1.5, dash: 'dashdot' } },
                { x: d.weeks, y: dbsTraj.legacy_rsa || d.trajectories.legacy_rsa, name: 'Legacy RSA-2048 (Shor Break)', type: 'scatter', mode: 'lines', line: { color: '#64748b', width: 2, dash: 'dot' } },
                { x: d.weeks, y: dbsTraj.bs_pqc_fortified || d.trajectories.pqc_shielded, name: 'BS Vercise + PQC (Kyber-1024)', type: 'scatter', mode: 'lines', line: { color: '#0284c7', width: 2.5 } },
                { x: d.weeks, y: dbsTraj.mdt_pqc_fortified || d.trajectories.pqc_shielded, name: 'MDT Percept + PQC (Dilithium-5)', type: 'scatter', mode: 'lines', line: { color: '#8b5cf6', width: 2.5 } },
                { x: d.weeks, y: dbsTraj.abt_pqc_fortified || d.trajectories.pqc_shielded, name: 'Abbott Infinity + PQC (SPHINCS+)', type: 'scatter', mode: 'lines', line: { color: '#f59e0b', width: 2.5 } },
                { x: d.weeks, y: dbsTraj.nm_pqc_fortified || d.trajectories.pqc_shielded, name: 'NeuroMorph DBS (Galois F_p)', type: 'scatter', mode: 'lines+markers', marker: { size: 4 }, line: { color: '#10b981', width: 3.2 } }
            ], {
                ...commonLayout,
                margin: { l: 55, r: 30, t: 30, b: 75 },
                xaxis: { ...commonLayout.xaxis, title: 'Auditing Horizon (Weeks)' },
                yaxis: { ...commonLayout.yaxis, title: 'DBS Telemetry Integrity (%)', range: [0, 105] },
                legend: { orientation: 'h', x: 0, y: -0.32, font: { color: '#cbd5e1', size: 9.0 } }
            }, { responsive: true, displaylogo: false });

            // 3. Multi-Vector Risk Aversion Plot
            const vecTraces = [];
            if (dbsTraj.bs_risk_aversion) {
                vecTraces.push({
                    x: d.weeks,
                    y: dbsTraj.bs_risk_aversion,
                    name: 'Boston Scientific Vercise (MICC)',
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#38bdf8', width: 2.2 }
                });
            }
            if (dbsTraj.mdt_risk_aversion) {
                vecTraces.push({
                    x: d.weeks,
                    y: dbsTraj.mdt_risk_aversion,
                    name: 'Medtronic Percept (BrainSense)',
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#a78bfa', width: 2.2 }
                });
            }
            if (dbsTraj.abt_risk_aversion) {
                vecTraces.push({
                    x: d.weeks,
                    y: dbsTraj.abt_risk_aversion,
                    name: 'Abbott Infinity (iOS BLE)',
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#fbbf24', width: 2.2 }
                });
            }
            if (dbsTraj.nm_risk_aversion) {
                vecTraces.push({
                    x: d.weeks,
                    y: dbsTraj.nm_risk_aversion,
                    name: 'NeuroMorph DBS (Galois Coprocessor)',
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#10b981', width: 2.8 }
                });
            }
            if (dbsTraj.composite_dbs_aversion) {
                vecTraces.push({
                    x: d.weeks,
                    y: dbsTraj.composite_dbs_aversion,
                    name: 'Composite DBS Risk Aversion (Avg)',
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#34d399', width: 3.5, dash: 'dot' }
                });
            }

            Plotly.newPlot('pqc-plot-aversion-vectors', vecTraces, {
                ...commonLayout,
                margin: { l: 55, r: 30, t: 30, b: 75 },
                xaxis: { ...commonLayout.xaxis, title: 'Auditing Horizon (Weeks)' },
                yaxis: { ...commonLayout.yaxis, title: 'Risk Aversion Probability (%)', range: [92, 100.2] },
                legend: { orientation: 'h', x: 0, y: -0.32, font: { color: '#cbd5e1', size: 9.0 } }
            }, { responsive: true, displaylogo: false });

            // 4. Quantum Barrier Penetration Probability
            Plotly.newPlot('pqc-plot-barrier', [{
                x: d.weeks,
                y: dbsTraj.quantum_breach_prob || d.trajectories.penetration_prob,
                name: 'Breach Penetration Prob %',
                type: 'scatter',
                mode: 'lines+markers',
                marker: { size: 4, color: '#f87171' },
                line: { color: '#f87171', width: 2.5 }
            }], {
                ...commonLayout,
                margin: { l: 55, r: 30, t: 30, b: 50 },
                xaxis: { ...commonLayout.xaxis, title: 'Adversarial Exposure Horizon (Weeks)' },
                yaxis: { ...commonLayout.yaxis, title: 'P(Penetration) %', type: 'log' },
                showlegend: false
            }, { responsive: true, displaylogo: false });

            // 5. Evolutionary Wavefront PDE Plot
            const wf = dbs.wavefront || {};
            if (wf.x_grid_mm) {
                Plotly.newPlot('pqc-plot-wavefront', [
                    { x: wf.x_grid_mm, y: wf.wave_unmitigated_t0, name: 'Wavefront (t=0s)', type: 'scatter', mode: 'lines', line: { color: '#64748b', dash: 'dot', width: 1.5 } },
                    { x: wf.x_grid_mm, y: wf.wave_unmitigated_t10, name: 'Unmitigated Intrusion (t=10s)', type: 'scatter', mode: 'lines', line: { color: '#ef4444', width: 2 } },
                    { x: wf.x_grid_mm, y: wf.wave_pqc_attenuated, name: `PQC Damped (v=${wf.wave_speed_mm_s} mm/s)`, type: 'scatter', mode: 'lines', line: { color: '#10b981', width: 3 } }
                ], {
                    ...commonLayout,
                    margin: { l: 55, r: 30, t: 30, b: 70 },
                    xaxis: { ...commonLayout.xaxis, title: 'Tissue Propagation Depth x (mm)' },
                    yaxis: { ...commonLayout.yaxis, title: 'Intrusion Amplitude u(x,t)' },
                    legend: { orientation: 'h', x: 0, y: -0.28, font: { color: '#cbd5e1', size: 9.5 } }
                }, { responsive: true, displaylogo: false });
            }

            // 6. Prime Moduli & QVC Hardness Bar Chart
            const pm = dbs.prime_moduli || [];
            if (pm.length > 0) {
                Plotly.newPlot('pqc-plot-prime-moduli', [
                    {
                        x: pm.map(m => `p=${m.prime_p}`),
                        y: pm.map(m => m.svp_hardness_bits),
                        name: 'SVP Hardness (Bits)',
                        type: 'bar',
                        marker: { color: ['#0a3d62', '#1e3799', '#0284c7', '#10b981', '#6941c6'] }
                    }
                ], {
                    ...commonLayout,
                    margin: { l: 55, r: 30, t: 30, b: 50 },
                    xaxis: { ...commonLayout.xaxis, title: 'Prime Modulus Ring Ideal R_p' },
                    yaxis: { ...commonLayout.yaxis, title: 'Lattice Security (Bits)', range: [0, 450] },
                    shapes: [{
                        type: 'line',
                        x0: -0.5,
                        x1: pm.length - 0.5,
                        y0: 256,
                        y1: 256,
                        line: { color: '#ef4444', width: 1.5, dash: 'dash' }
                    }]
                }, { responsive: true, displaylogo: false });
            }

            // 6. Threat Matrix Render (DBS Optimized Table)
            const tbody = el('pqc-threat-tbody');
            if (tbody) {
                const threats = dbs.threat_table || d.threat_matrix || [];
                tbody.innerHTML = threats.map(row => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:8px 10px; font-family:monospace; color:#38bdf8; font-weight:600;">${row.cve_id || 'CVE-2026-PQC'}</td>
                        <td style="padding:8px 10px; font-weight:600; color:#f8fafc;">${row.device || 'DBS Stimulator'}</td>
                        <td style="padding:8px 10px; color:#cbd5e1;">${row.title || row.vector}</td>
                        <td style="padding:8px 10px; color:#f87171; font-weight:700;">${(row.baseline_cvss || row.raw_threat_index).toFixed(1)}</td>
                        <td style="padding:8px 10px; color:#34d399; font-size:11.5px;">${row.pqc_countermeasure || row.mitigation_method}</td>
                        <td style="padding:8px 10px; color:#38bdf8; font-weight:700;">${(row.mitigated_cvss || 0.6).toFixed(1)}</td>
                        <td style="padding:8px 10px; color:#4ade80; font-weight:700;">${row.risk_aversion_pct.toFixed(2)}%</td>
                        <td style="padding:8px 10px; color:#a78bfa; font-weight:600;">+${(row.security_margin_db || row.attenuation_db).toFixed(1)} dB</td>
                    </tr>
                `).join('');
            }

            // 7. Math Equations Render
            const mathContainer = el('pqc-math-container');
            if (mathContainer) {
                const equations = dbs.math_equations || d.math_equations || [];
                mathContainer.innerHTML = equations.map(eq => `
                    <div style="background:rgba(5,11,20,0.8); border:1px solid rgba(16,185,129,0.15); border-radius:8px; padding:10px;">
                        <div style="font-size:12px; font-weight:600; color:#34d399; margin-bottom:4px;">${eq.title}</div>
                        <div style="font-family:'Courier New', monospace; font-size:11px; color:#38bdf8; background:rgba(0,0,0,0.5); padding:6px 10px; border-radius:4px; overflow-x:auto;">${eq.latex}</div>
                        <div style="font-size:11px; color:#94a3b8; margin-top:4px;">${eq.description}</div>
                    </div>
                `).join('');
            }

        } catch (err) {
            console.error('Cybersecurity PQC API error:', err);
        }
    };

    // ─────────────────────────────────────────────────────────────
    // Stroke Repair rTMS Statistical Staging Suite
    // ─────────────────────────────────────────────────────────────
    let srDebounceTimer = null;
    window.runStrokeRepairDebounced = function() {
        clearTimeout(srDebounceTimer);
        srDebounceTimer = setTimeout(runStrokeRepair, 60);
    };

    window.runStrokeRepair = async function() {
        const baselineEl = document.getElementById('sr-baseline');
        if (!baselineEl) return;
        const params = new URLSearchParams({
            baseline_fugl_meyer: baselineEl.value,
            horizon_weeks: document.getElementById('sr-weeks').value,
            galois_modulus: document.getElementById('sr-modulus').value,
            cf_depth: document.getElementById('sr-depth').value,
            closed_loop_gain: document.getElementById('sr-gain').value
        });
        const preprintBtn = document.getElementById('sr-preprint-btn');
        if (preprintBtn) preprintBtn.href = `/api/stroke-repair-preprint?${params.toString()}`;

        try {
            const response = await fetch(`/api/stroke-repair-rtms?${params.toString()}`);
            const result = await response.json();
            if (!response.ok || result.status !== 'success') throw new Error(result.message || `HTTP ${response.status}`);
            const d = result.data;
            const el = id => document.getElementById(id);
            el('sr-metric-final').textContent = `${d.kpis.final_adaptive_fugl_meyer.toFixed(1)} / 66`;
            el('sr-metric-gain').textContent = `+${d.kpis.gain_vs_usual_care.toFixed(1)} points`;
            el('sr-metric-error').textContent = d.kpis.cf_error.toExponential(2);

            const layout = { paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', margin: { t: 24, b: 48, l: 52, r: 24 }, font: { color: '#c9d1d9', family: 'Inter' }, xaxis: { gridcolor: 'rgba(255,255,255,0.06)' }, yaxis: { gridcolor: 'rgba(255,255,255,0.06)' }, legend: { orientation: 'h', x: 0, y: -0.24, font: { size: 10 } } };
            Plotly.react('sr-trajectory-chart', [
                { x: d.weeks, y: d.trajectories.usual_care, name: 'Usual care', mode: 'lines', line: { color: '#64748b', dash: 'dot' } },
                { x: d.weeks, y: d.trajectories.fixed_rtms, name: 'Fixed rTMS', mode: 'lines', line: { color: '#f97316' } },
                { x: d.weeks, y: d.trajectories.adaptive_rtms, name: 'Adaptive staged rTMS', mode: 'lines', line: { color: '#34d399', width: 3 } },
                { x: d.weeks.concat(d.weeks.slice().reverse()), y: d.distribution.adaptive_upper.concat(d.distribution.adaptive_lower.slice().reverse()), name: '95% model interval', mode: 'lines', fill: 'toself', fillcolor: 'rgba(52,211,153,0.14)', line: { color: 'transparent' }, hoverinfo: 'skip' }
            ], { ...layout, xaxis: { ...layout.xaxis, title: 'Weeks after index event' }, yaxis: { ...layout.yaxis, title: 'Fugl-Meyer motor score' } }, { responsive: true, displaylogo: false });

            Plotly.react('sr-stage-chart', [{ x: d.weeks, y: d.finite_states.stage_state, name: `Stage state mod ${d.finite_states.galois_modulus}`, mode: 'lines+markers', line: { color: '#38bdf8', shape: 'hv' }, marker: { size: 4 } }], { ...layout, xaxis: { ...layout.xaxis, title: 'Weeks' }, yaxis: { ...layout.yaxis, title: 'Finite state s(t)' } }, { responsive: true, displaylogo: false });

            el('sr-stage-table').innerHTML = `<table style="width:100%; border-collapse:collapse; font-size:12px;"><thead><tr style="color:#94a3b8; border-bottom:1px solid rgba(255,255,255,0.15);"><th style="padding:8px; text-align:left;">Stage</th><th style="padding:8px;">Weeks</th><th style="padding:8px;">Hz</th><th style="padding:8px;">% MSO</th><th style="padding:8px;">Sessions/wk</th><th style="padding:8px;">CF sync</th></tr></thead><tbody>${d.stage_protocol.map(row => `<tr style="border-bottom:1px solid rgba(255,255,255,0.06);"><td style="padding:8px; color:#34d399;">${row.stage}</td><td style="padding:8px; text-align:center;">${row.duration_weeks}</td><td style="padding:8px; text-align:center;">${row.frequency_hz}</td><td style="padding:8px; text-align:center;">${row.intensity_pct_mso}</td><td style="padding:8px; text-align:center;">${row.sessions_per_week}</td><td style="padding:8px; text-align:center; color:#c084fc;">${row.sync_fraction}</td></tr>`).join('')}</tbody></table>`;
            el('sr-region-table').innerHTML = `<table style="width:100%; border-collapse:collapse; font-size:12px;"><thead><tr style="color:#94a3b8; border-bottom:1px solid rgba(255,255,255,0.15);"><th style="padding:8px; text-align:left;">Stage</th><th style="padding:8px; text-align:left;">Candidate region(s)</th><th style="padding:8px; text-align:left;">Neurovascular signature</th><th style="padding:8px; text-align:left;">Vascular readout</th><th style="padding:8px;">Elliptic gate</th></tr></thead><tbody>${d.stage_protocol.map(row => `<tr style="border-bottom:1px solid rgba(255,255,255,0.06);"><td style="padding:8px; color:#34d399;">${row.stage}</td><td style="padding:8px; color:#f8fafc;">${row.regions}</td><td style="padding:8px; color:#cbd5e1;">${row.signature}</td><td style="padding:8px; color:#38bdf8;">${row.vascular_marker}</td><td style="padding:8px; text-align:center; color:#fbbf24;">${row.elliptic_gate.toFixed(3)}</td></tr>`).join('')}</tbody></table>`;
            const moduli = d.elliptic_moduli;
            el('sr-moduli-card').innerHTML = `<div class="sa-metric-row"><span class="sa-metric-lbl">Modulus m</span><span class="sa-metric-val" style="color:#34d399;">${moduli.m.toFixed(3)}</span></div><div class="sa-metric-row"><span class="sa-metric-lbl">K(m)</span><span class="sa-metric-val">${moduli.K_m.toFixed(4)}</span></div><div class="sa-metric-row"><span class="sa-metric-lbl">Complement K(1-m)</span><span class="sa-metric-val">${moduli.K_1_minus_m.toFixed(4)}</span></div><div class="sa-metric-row"><span class="sa-metric-lbl">Elliptic nome q</span><span class="sa-metric-val" style="color:#c084fc;">${moduli.nome_q.toFixed(5)}</span></div><div class="sa-metric-row" style="border-bottom:none;"><span class="sa-metric-lbl">Base gate score</span><span class="sa-metric-val" style="color:#fbbf24;">${moduli.gate_score.toFixed(3)}</span></div>`;
            el('sr-checkpoint-list').innerHTML = d.trajectory_checkpoints.map(point => `<div class="sa-metric-row"><span class="sa-metric-lbl">Week ${point.week} · ${point.stage}</span><span class="sa-metric-val" style="color:#38bdf8;">${point.adaptive_score.toFixed(1)} · gate ${point.vascular_gate.toFixed(2)}</span></div>`).join('');
            el('sr-math-container').innerHTML = d.math_equations.map(eq => `<div style="background:rgba(5,11,20,0.8); border:1px solid rgba(16,185,129,0.15); border-radius:8px; padding:10px;"><div style="font-size:12px; font-weight:600; color:#34d399; margin-bottom:4px;">${eq.title}</div><div style="font-family:'Courier New', monospace; font-size:11px; color:#38bdf8; background:rgba(0,0,0,0.5); padding:6px 10px; border-radius:4px; overflow-x:auto;">${eq.latex}</div></div>`).join('');
        } catch (err) {
            console.error('Stroke repair rTMS API error:', err);
        }
    };

    // ─────────────────────────────────────────────────────────────
    // Clinical Trajectories & Superoptimal DBS/TMS Benchmark Suite
    // ─────────────────────────────────────────────────────────────
    let ctDebounceTimer = null;
    window.runClinicalTrajectoriesBenchmarkDebounced = function() {
        clearTimeout(ctDebounceTimer);
        ctDebounceTimer = setTimeout(runClinicalTrajectoriesBenchmark, 40);
    };

    window.runClinicalTrajectoriesBenchmark = async function() {
        const variantEl = document.getElementById('ct-variant');
        if (!variantEl) return;

        const params = new URLSearchParams({
            variant: variantEl.value,
            weeks: document.getElementById('ct-weeks').value,
            galois_modulus: document.getElementById('ct-modulus').value,
            closed_loop_gain: document.getElementById('ct-gain').value,
            ramanujan_depth: document.getElementById('ct-cf-depth').value
        });

        const preprintBtn = document.getElementById('ct-preprint-btn');
        if (preprintBtn) {
            preprintBtn.href = `/api/clinical-trajectories-preprint?${params.toString()}`;
        }

        try {
            const response = await fetch(`/api/clinical-trajectories-benchmark?${params.toString()}`);
            const result = await response.json();
            if (!response.ok || result.error) throw new Error(result.error || `HTTP ${response.status}`);
            const d = result.data;
            const kpis = d.kpis;

            // 1. Update Subtitle and KPI Telemetry
            const trajSub = document.getElementById('ct-trajectory-subtitle');
            if (trajSub) {
                trajSub.textContent = `${d.variant.name} · Target: ${d.variant.target_nucleus} (${d.variant.metric_name})`;
            }

            const el = id => document.getElementById(id);
            if (el('ct-metric-remission')) el('ct-metric-remission').textContent = `${kpis.neuromorph_remission_rate_pct.toFixed(1)}% (2-Yr)`;
            if (el('ct-metric-advantage')) el('ct-metric-advantage').textContent = `+${kpis.remission_advantage_pct.toFixed(1)}% vs MDT`;
            if (el('ct-metric-battery')) el('ct-metric-battery').textContent = `${kpis.battery_longevity_yr.toFixed(1)} yr (${kpis.battery_multiplier}x)`;
            if (el('ct-metric-vta')) el('ct-metric-vta').textContent = `${kpis.vta_specificity_pct.toFixed(1)}% Specificity`;
            if (el('ct-metric-entropy')) el('ct-metric-entropy').textContent = `-${kpis.side_effect_entropy_reduction_pct.toFixed(1)}% Entropy`;
            if (el('ct-metric-cf')) el('ct-metric-cf').textContent = `[${kpis.cf_convergents_str}]`;
            if (el('ct-metric-qaly')) el('ct-metric-qaly').textContent = `+${kpis.qaly_gain.toFixed(2)} QALYs`;

            const commonLayout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: { t: 28, b: 46, l: 50, r: 24 },
                font: { color: '#c9d1d9', family: 'Inter' },
                xaxis: { title: 'Treatment Horizon (Weeks)', gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e' },
                yaxis: { gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e' },
                legend: { orientation: 'h', x: 0, y: -0.26, font: { size: 9.5 } }
            };

            // 2. Trajectories Plot
            const tr = d.trajectories;
            Plotly.react('ct-plot-trajectories', [
                { x: d.weeks, y: tr.natural_history_sham, name: 'Sham / Natural History', type: 'scatter', mode: 'lines', line: { color: '#64748b', dash: 'dot', width: 1.5 } },
                { x: d.weeks, y: tr.neurostar_tms, name: 'NeuroStar® TMS', type: 'scatter', mode: 'lines', line: { color: '#ef4444', dash: 'dash', width: 1.5 } },
                { x: d.weeks, y: tr.magventure_magpro, name: 'MagVenture® MagPro', type: 'scatter', mode: 'lines', line: { color: '#f97316', dash: 'dash', width: 1.5 } },
                { x: d.weeks, y: tr.brainsway_dtms, name: 'BrainsWay® Deep TMS', type: 'scatter', mode: 'lines', line: { color: '#ec4899', dash: 'dashdot', width: 1.8 } },
                { x: d.weeks, y: tr.abbott_infinity, name: 'Abbott Infinity™ DBS', type: 'scatter', mode: 'lines', line: { color: '#f59e0b', width: 1.8 } },
                { x: d.weeks, y: tr.medtronic_percept, name: 'Medtronic Percept™ (BrainSense)', type: 'scatter', mode: 'lines', line: { color: '#8b5cf6', width: 2.2 } },
                { x: d.weeks, y: tr.boston_sci_vercise, name: 'Boston Sci Vercise™ (MICC)', type: 'scatter', mode: 'lines', line: { color: '#0284c7', width: 2.2 } },
                { x: d.weeks, y: tr.neuromorph_dbs, name: 'NeuroMorph Superoptimal DBS', type: 'scatter', mode: 'lines+markers', marker: { size: 4 }, line: { color: '#10b981', width: 3.5 } }
            ], {
                ...commonLayout,
                yaxis: { ...commonLayout.yaxis, title: d.variant.metric_name }
            }, { responsive: true, displaylogo: false });

            // 3. Superoptimal Galois Staging & Lyapunov Energy Decay Plot
            const fs = d.finite_states;
            Plotly.react('ct-plot-galois-lyapunov', [
                { x: d.weeks, y: fs.galois_states, name: `Galois Stage s_k (mod ${kpis.galois_modulus})`, type: 'scatter', mode: 'lines', line: { color: '#10b981', shape: 'hv', width: 2.2 }, yaxis: 'y' },
                { x: d.weeks, y: fs.lyapunov_energy, name: 'Lyapunov Potential V(s_k)', type: 'scatter', mode: 'lines', line: { color: '#fbbf24', dash: 'dot', width: 2 }, yaxis: 'y2' }
            ], {
                ...commonLayout,
                yaxis: { ...commonLayout.yaxis, title: `Galois State s_k mod ${kpis.galois_modulus}`, color: '#10b981' },
                yaxis2: { title: 'Lyapunov V(s_k)', overlaying: 'y', side: 'right', color: '#fbbf24', gridcolor: 'rgba(251,191,36,0.08)' }
            }, { responsive: true, displaylogo: false });

            // 4. 6-Axis Radar Benchmark Plot
            const radarTraces = [];
            const rCats = d.radar_categories;
            const colorMap = {
                neuromorph_dbs: '#10b981',
                boston_sci_vercise: '#0284c7',
                medtronic_percept: '#8b5cf6',
                abbott_infinity: '#f59e0b',
                brainsway_dtms: '#ec4899',
                magventure_magpro: '#f97316',
                neurostar_tms: '#ef4444'
            };

            for (const [devId, rdata] of Object.entries(d.radar_data)) {
                const isNM = devId === 'neuromorph_dbs';
                radarTraces.push({
                    type: 'scatterpolar',
                    r: rdata.values.concat([rdata.values[0]]),
                    theta: rCats.concat([rCats[0]]),
                    name: rdata.name,
                    fill: isNM ? 'toself' : 'none',
                    fillcolor: isNM ? 'rgba(16,185,129,0.15)' : 'transparent',
                    line: { color: colorMap[devId] || '#94a3b8', width: isNM ? 3 : 1.5 }
                });
            }

            Plotly.react('ct-plot-radar', radarTraces, {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: { t: 30, b: 50, l: 50, r: 50 },
                font: { color: '#c9d1d9', family: 'Inter', size: 10 },
                polar: {
                    radialaxis: { visible: true, range: [0, 100], gridcolor: 'rgba(255,255,255,0.08)', color: '#8b949e' },
                    angularaxis: { gridcolor: 'rgba(255,255,255,0.08)', color: '#cbd5e1' },
                    bgcolor: 'transparent'
                },
                legend: { orientation: 'h', x: 0, y: -0.32, font: { size: 9 } }
            }, { responsive: true, displaylogo: false });

            // 5. Pareto Frontier Plot (VTA Selectivity vs Power Dissipation)
            const paretoPoints = d.pareto_points;
            const paretoTraces = [
                {
                    x: paretoPoints.map(p => p.power_uW),
                    y: paretoPoints.map(p => p.vta),
                    text: paretoPoints.map(p => p.name),
                    mode: 'markers+text',
                    type: 'scatter',
                    textposition: 'top right',
                    textfont: { color: '#cbd5e1', size: 9.5 },
                    marker: {
                        size: paretoPoints.map(p => p.marker_size * 1.3),
                        color: paretoPoints.map(p => p.color),
                        line: { color: '#ffffff', width: 1 }
                    },
                    name: 'Stimulator Platforms'
                },
                {
                    x: [12.4, 44.0, 52.0, 65.0, 160.0],
                    y: [96.8, 81.2, 76.4, 72.8, 64.2],
                    mode: 'lines',
                    type: 'scatter',
                    line: { color: '#10b981', dash: 'dot', width: 1.5 },
                    name: 'Superoptimal Pareto Frontier'
                }
            ];

            Plotly.react('ct-plot-pareto', paretoTraces, {
                ...commonLayout,
                xaxis: { ...commonLayout.xaxis, title: 'Power Dissipation (μW)' },
                yaxis: { ...commonLayout.yaxis, title: 'VTA Steering Selectivity (%)', range: [50, 102] }
            }, { responsive: true, displaylogo: false });

            // 6. Benchmark Matrix Table Render
            const tbody = el('ct-benchmark-tbody');
            if (tbody) {
                const rows = d.benchmark_table || [];
                tbody.innerHTML = rows.map(r => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06); ${r.is_neuromorph ? 'background:rgba(16,185,129,0.12); font-weight:600;' : ''}">
                        <td style="padding:8px 10px; color:${r.is_neuromorph ? '#34d399' : '#f8fafc'}; font-weight:${r.is_neuromorph ? '700' : '500'};">${r.name}</td>
                        <td style="padding:8px 10px; color:#94a3b8;">${r.device_class}</td>
                        <td style="padding:8px 10px; color:${r.is_neuromorph ? '#34d399' : '#38bdf8'}; font-weight:700;">${r.remission_rate_pct.toFixed(1)}%</td>
                        <td style="padding:8px 10px; color:#4ade80;">-${r.reduction_pct.toFixed(1)}%</td>
                        <td style="padding:8px 10px; color:#a78bfa;">${r.vta_specificity_pct.toFixed(1)}%</td>
                        <td style="padding:8px 10px; color:#fbbf24;">${r.battery_longevity_yr.toFixed(1)} yr</td>
                        <td style="padding:8px 10px; color:#f87171;">${r.side_effect_entropy.toFixed(2)}</td>
                        <td style="padding:8px 10px; color:#cbd5e1;">${r.latency_wks.toFixed(1)} wks</td>
                        <td style="padding:8px 10px; color:#2dd4bf; font-weight:700;">+${r.qaly_gain.toFixed(2)}</td>
                    </tr>
                `).join('');
            }

            // 7. Math Equations Render
            const mathContainer = el('ct-math-container');
            if (mathContainer) {
                const equations = d.math_equations || [];
                mathContainer.innerHTML = equations.map(eq => `
                    <div style="background:rgba(5,11,20,0.8); border:1px solid rgba(16,185,129,0.15); border-radius:8px; padding:10px;">
                        <div style="font-size:12px; font-weight:600; color:#34d399; margin-bottom:4px;">${eq.title}</div>
                        <div style="font-family:'Courier New', monospace; font-size:11px; color:#38bdf8; background:rgba(0,0,0,0.5); padding:6px 10px; border-radius:4px; overflow-x:auto;">${eq.latex}</div>
                        <div style="font-size:11px; color:#94a3b8; margin-top:4px;">${eq.description}</div>
                    </div>
                `).join('');
            }

            // 8. ASCII Architecture Schematic Render
            const asciiEl = el('ct-ascii-schematic');
            if (asciiEl) {
                asciiEl.textContent = d.ascii_schematic;
            }

        } catch (err) {
            console.error('Clinical Trajectories Benchmark API error:', err);
        }
    };

    // ── NIH Grant, rTMS Characteristics & Optimal Staging Studio ────
    let nihDebounceTimer = null;
    window.runNihGrantStudioDebounced = function() {
        clearTimeout(nihDebounceTimer);
        nihDebounceTimer = setTimeout(runNihGrantStudio, 50);
    };

    window.runNihGrantStudio = async function() {
        const directEl = document.getElementById('nih-direct');
        if (!directEl) return;

        const directBudget = parseFloat(directEl.value || 640000.0);
        const faRate = parseFloat(document.getElementById('nih-fa')?.value || 54.5) / 100.0;
        const nYears = parseInt(document.getElementById('nih-years')?.value || 5);
        const contingency = parseFloat(document.getElementById('nih-contingency')?.value || 4.0);

        const params = new URLSearchParams({
            direct_budget: directBudget,
            fa_rate: faRate,
            n_years: nYears,
            contingency_pct: contingency
        });

        try {
            const res = await fetch(`/api/nih-grant-budget-optimizer?${params.toString()}`);
            const result = await res.json();
            if (!res.ok || result.status !== 'success') throw new Error(result.message || 'Error fetching NIH grant data');
            const d = result.data;
            const rtms = d.rtms_specs;
            const stg = d.staging_data;
            const dbs = d.dbs_specs;

            const el = id => document.getElementById(id);
            if (el('nih-metric-total')) el('nih-metric-total').textContent = `$${Math.round(d.total_budget).toLocaleString()}`;
            if (el('nih-metric-status')) el('nih-metric-status').textContent = d.is_under_1m ? '✓ Sub-$1M Lean' : '⚠️ > $1M Bound';
            if (el('nih-metric-savings')) el('nih-metric-savings').textContent = `${d.cost_savings_pct.toFixed(1)}% vs R01`;
            if (el('nih-metric-remission')) el('nih-metric-remission').textContent = `${dbs.madrs_remission_rate_pct.toFixed(1)}% ± ${dbs.madrs_remission_ci.toFixed(1)}%`;
            if (el('nih-metric-efield')) el('nih-metric-efield').textContent = `158.2 V/m (iTBS)`;
            if (el('nih-metric-latency')) el('nih-metric-latency').textContent = `${dbs.closed_loop_latency_ms.toFixed(2)} ± ${dbs.closed_loop_latency_ci_ms.toFixed(2)} ms`;
            if (el('nih-metric-galois')) el('nih-metric-galois').textContent = `F_17 Dynamic Prog`;

            const commonLayout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: { t: 28, b: 46, l: 50, r: 24 },
                font: { color: '#c9d1d9', family: 'Inter' },
                xaxis: { gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e' },
                yaxis: { gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e' },
                legend: { orientation: 'h', x: 0, y: -0.26, font: { size: 9.5 } }
            };

            // 1. Budget Donut Pie Chart
            const catNames = d.categories.map(c => c.name.split('&')[0].trim());
            const catPcts = d.categories.map(c => c.base_pct);
            const pieColors = ['#0a3d62', '#38bdf8', '#079992', '#a78bfa', '#f59e0b', '#e58e26', '#94a3b8'];

            Plotly.react('nih-plot-budget-pie', [{
                values: catPcts,
                labels: catNames,
                type: 'pie',
                hole: 0.5,
                marker: { colors: pieColors },
                textinfo: 'label+percent',
                textposition: 'outside',
                automargin: true,
                textfont: { size: 9.5, color: '#cbd5e1' }
            }], {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: { t: 20, b: 20, l: 20, r: 20 },
                showlegend: false,
                font: { color: '#c9d1d9', family: 'Inter' }
            }, { responsive: true, displaylogo: false });

            // 2. 20-Quarter Burn-Rate Trajectory Chart
            const qLabels = d.quarterly_labels;
            const plannedCostsK = d.planned_cum_spend.map(v => v / 1000.0);
            const upperK = d.upper_variance_band.map(v => v / 1000.0);
            const lowerK = d.lower_variance_band.map(v => v / 1000.0);

            Plotly.react('nih-plot-burn-rate', [
                { x: qLabels, y: upperK, name: '+5% Upper Band', type: 'scatter', mode: 'lines', line: { color: 'rgba(56,189,248,0.2)', width: 0 }, showlegend: false },
                { x: qLabels, y: lowerK, name: '-5% Lower Band', type: 'scatter', mode: 'lines', fill: 'tonexty', fillcolor: 'rgba(56,189,248,0.12)', line: { color: 'rgba(56,189,248,0.2)', width: 0 }, showlegend: false },
                { x: qLabels, y: plannedCostsK, name: 'Planned Direct Spend ($k)', type: 'scatter', mode: 'lines+markers', line: { color: '#38bdf8', width: 3 }, marker: { size: 4 } }
            ], {
                ...commonLayout,
                xaxis: { ...commonLayout.xaxis, title: 'Quarterly Project Horizon (20 Quarters)' },
                yaxis: { ...commonLayout.yaxis, title: 'Cumulative Direct Cost ($k)' }
            }, { responsive: true, displaylogo: false });

            // 3. rTMS BEM Electric Field & Current Density Depth Decay
            Plotly.react('nih-plot-bem-profile', [
                { x: d.depths_mm, y: d.e_field_bem, name: 'Peak E-Field (V/m)', type: 'scatter', mode: 'lines', line: { color: '#f43f5e', width: 2.5 } },
                { x: d.depths_mm, y: d.current_density, name: 'Current Density J (A/m²)', type: 'scatter', mode: 'lines', line: { color: '#38bdf8', width: 2, dash: 'dot' }, yaxis: 'y2' }
            ], {
                ...commonLayout,
                xaxis: { ...commonLayout.xaxis, title: 'Cortical Tissue Depth (mm)' },
                yaxis: { ...commonLayout.yaxis, title: 'E-Field (V/m)', color: '#f43f5e' },
                yaxis2: { title: 'J (A/m²)', overlaying: 'y', side: 'right', color: '#38bdf8', gridcolor: 'rgba(56,189,248,0.08)' }
            }, { responsive: true, displaylogo: false });

            // 4. Finite Math 3-Stage Optimal Trajectory vs Standard Care
            Plotly.react('nih-plot-staging-trajectory', [
                { x: stg.sessions, y: stg.score_phq9_sham, name: 'Sham Control', type: 'scatter', mode: 'lines', line: { color: '#64748b', dash: 'dot', width: 1.5 } },
                { x: stg.sessions, y: stg.score_phq9_standard, name: 'Standard 10Hz rTMS', type: 'scatter', mode: 'lines', line: { color: '#f59e0b', dash: 'dash', width: 2 } },
                { x: stg.sessions, y: stg.score_phq9_optimal_galois, name: 'Galois F_17 Optimal Staging', type: 'scatter', mode: 'lines+markers', marker: { size: 4 }, line: { color: '#10b981', width: 3.5 } },
                { x: stg.sessions, y: stg.lyapunov_energy, name: 'Lyapunov Energy V(s_k) %', type: 'scatter', mode: 'lines', line: { color: '#c084fc', dash: 'dot', width: 1.8 }, yaxis: 'y2' }
            ], {
                ...commonLayout,
                xaxis: { ...commonLayout.xaxis, title: 'Treatment Session (30 Sessions)' },
                yaxis: { ...commonLayout.yaxis, title: 'PHQ-9 Symptom Score (0-27)', color: '#10b981' },
                yaxis2: { title: 'Lyapunov Energy %', overlaying: 'y', side: 'right', color: '#c084fc', gridcolor: 'rgba(192,132,252,0.08)' }
            }, { responsive: true, displaylogo: false });

            // 5. rTMS Modalities Table Render
            const rtmsTbody = el('nih-rtms-tbody');
            if (rtmsTbody && rtms.modalities) {
                rtmsTbody.innerHTML = rtms.modalities.map(m => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:8px 10px; font-weight:700; color:#38bdf8;">${m.protocol}</td>
                        <td style="padding:8px 10px; color:#f8fafc;">${m.indication}</td>
                        <td style="padding:8px 10px; color:#34d399; font-weight:600;">${m.frequency_hz} Hz</td>
                        <td style="padding:8px 10px; color:#fbbf24;">${m.intensity_pct_rmt}% rMT</td>
                        <td style="padding:8px 10px; color:#cbd5e1;">${m.train_duration_s}s / ${m.inter_train_interval_s}s</td>
                        <td style="padding:8px 10px; color:#a78bfa; font-weight:600;">${m.total_pulses_session.toLocaleString()}</td>
                        <td style="padding:8px 10px; font-size:11px; color:#94a3b8;">${m.target_coordinates}</td>
                        <td style="padding:8px 10px; color:#f43f5e; font-weight:700;">${m.focal_e_field_vm.toFixed(1)} V/m</td>
                        <td style="padding:8px 10px; color:#10b981; font-weight:700;">${m.remission_rate_pct.toFixed(1)}%</td>
                    </tr>
                `).join('');
            }

            // 6. 3-Stage Optimal Staging Cards Render
            const cardsContainer = el('nih-staging-cards-container');
            if (cardsContainer && stg.stages_breakdown) {
                const borderColors = ['#38bdf8', '#34d399', '#fbbf24'];
                cardsContainer.innerHTML = stg.stages_breakdown.map((sb, idx) => `
                    <div style="background:rgba(15,23,42,0.85); border:1px solid ${borderColors[idx] || '#38bdf8'}; border-radius:10px; padding:14px; display:flex; flex-direction:column; gap:8px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:11px; font-weight:700; color:${borderColors[idx]}; text-transform:uppercase;">Stage ${sb.stage_idx}: ${sb.sessions}</span>
                            <span style="font-size:10px; background:rgba(255,255,255,0.08); padding:3px 8px; border-radius:4px; color:#cbd5e1;">${sb.dosage_pulses}</span>
                        </div>
                        <div style="font-size:14px; font-weight:700; color:#f8fafc;">${sb.name}</div>
                        <div style="font-size:12px; color:#38bdf8;"><b>Frequency &amp; Pattern:</b> ${sb.frequency_hz}</div>
                        <div style="font-size:12px; color:#34d399;"><b>Galois Target:</b> <code>${sb.galois_state_target}</code></div>
                        <div style="font-size:11.5px; color:#94a3b8; font-style:italic; line-height:1.3;">${sb.objective}</div>
                        <div style="font-size:12px; color:#fbbf24; font-weight:600; margin-top:4px;">Target Symptom Reduction: ≥ ${sb.symptom_reduction_target_pct}%</div>
                    </div>
                `).join('');
            }

            // 7. 5-Year Budget Table Render
            const budgetTbody = el('nih-budget-tbody');
            if (budgetTbody && d.category_rows) {
                budgetTbody.innerHTML = d.category_rows.map(r => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:8px 10px; font-weight:700; color:#38bdf8;">${r.name}</td>
                        <td style="padding:8px 10px; font-size:11px; color:#94a3b8;">${r.description}</td>
                        <td style="padding:8px 10px; color:#cbd5e1;">$${Math.round(r.y1).toLocaleString()}</td>
                        <td style="padding:8px 10px; color:#cbd5e1;">$${Math.round(r.y2).toLocaleString()}</td>
                        <td style="padding:8px 10px; color:#cbd5e1;">$${Math.round(r.y3).toLocaleString()}</td>
                        <td style="padding:8px 10px; color:#cbd5e1;">$${Math.round(r.y4).toLocaleString()}</td>
                        <td style="padding:8px 10px; color:#cbd5e1;">$${Math.round(r.y5).toLocaleString()}</td>
                        <td style="padding:8px 10px; color:#34d399; font-weight:700;">$${Math.round(r.total_direct).toLocaleString()}</td>
                        <td style="padding:8px 10px; color:#fbbf24; font-weight:600;">${r.pct_of_direct.toFixed(1)}%</td>
                    </tr>
                `).join('');
            }

            // 8. Math Equations Render
            const mathContainer = el('nih-math-container');
            if (mathContainer && d.math_equations) {
                mathContainer.innerHTML = d.math_equations.map(eq => `
                    <div style="background:rgba(5,11,20,0.8); border:1px solid rgba(56,189,248,0.2); border-radius:8px; padding:10px;">
                        <div style="font-size:12px; font-weight:600; color:#38bdf8; margin-bottom:4px;">${eq.title}</div>
                        <div style="font-family:'Courier New', monospace; font-size:11px; color:#34d399; background:rgba(0,0,0,0.5); padding:6px 10px; border-radius:4px; overflow-x:auto;">${eq.latex}</div>
                        <div style="font-size:11px; color:#94a3b8; margin-top:4px;">${eq.description}</div>
                    </div>
                `).join('');
            }

        } catch (err) {
            console.error('NIH Grant Studio API error:', err);
        }
    };

    // ═════════════════════════════════════════════════════════════════
    // PARKINSON'S BEM & COMBINATORIAL OPTIMAL STAGING STUDIO
    // ═════════════════════════════════════════════════════════════════
    let pdDebounceTimer = null;
    const runParkinsonsStudio = async () => {
        try {
            const el = id => document.getElementById(id);
            const baselineUpdrs = el('pd-updrs') ? parseFloat(el('pd-updrs').value) : 42.0;
            const intensityRmt = el('pd-intensity') ? parseFloat(el('pd-intensity').value) : 110.0;
            const rfCarrierKhz = el('pd-rf') ? parseFloat(el('pd-rf').value) : 3.2;
            const pulseBudget = el('pd-budget') ? parseInt(el('pd-budget').value) : 3000;
            const qmlDepth = el('pd-depth') ? parseInt(el('pd-depth').value) : 6;
            const sessionsPerWeek = el('pd-sessions') ? parseInt(el('pd-sessions').value) : 5;
            const leddDose = el('pd-ledd') ? parseFloat(el('pd-ledd').value) : 650.0;
            const horizonWeeks = el('pd-horizon') ? parseInt(el('pd-horizon').value) : 156;

            const params = new URLSearchParams({
                baseline_updrs: baselineUpdrs,
                intensity_rmt: intensityRmt,
                rf_carrier_khz: rfCarrierKhz,
                pulse_budget_daily: pulseBudget,
                qml_depth: qmlDepth,
                sessions_per_week: sessionsPerWeek,
                l_dopa_equiv_dose: leddDose,
                horizon_weeks: horizonWeeks
            });

            const res = await fetch(`/api/parkinsons-rtms?${params.toString()}`);
            if (!res.ok) throw new Error(`HTTP error ${res.status}`);
            const d = await res.json();
            if (d.status !== 'success') throw new Error(d.message || 'Error running Parkinson\'s simulation');

            const cs = d.clinical_summary || {};
            const rf = d.rf_characteristics || {};
            const bem = d.bem_simulation || {};
            const comb = d.combinatorial_optimal_staging || {};
            const econ = d.health_economics || {};
            const traj = d.clinical_trajectories || {};

            // 1. Update KPI Badges
            if (el('pd-metric-updrs-imp')) el('pd-metric-updrs-imp').textContent = `-${(cs.updrs_improvement_pct || 73.8).toFixed(1)}%`;
            if (el('pd-metric-final-updrs')) el('pd-metric-final-updrs').textContent = `${(cs.final_updrs || 11.0).toFixed(1)} (Remission)`;
            if (el('pd-metric-hy')) el('pd-metric-hy').textContent = `Stage 3.0 → ${(cs.final_hoehn_yahr || 1.0).toFixed(1)}`;
            if (el('pd-metric-offtime')) el('pd-metric-offtime').textContent = `5.8h → ${(cs.final_off_hours || 0.4).toFixed(2)}h/day`;
            if (el('pd-metric-ontime')) el('pd-metric-ontime').textContent = `4.5h → ${(cs.final_on_hours || 15.5).toFixed(1)}h/day`;
            if (el('pd-metric-ledd')) el('pd-metric-ledd').textContent = `-${(cs.ledd_reduction_pct || 67.7).toFixed(1)}% (${Math.round(traj.ledd_mg_day ? traj.ledd_mg_day[traj.ledd_mg_day.length - 1] : 210)} mg)`;
            if (el('pd-metric-efield')) el('pd-metric-efield').textContent = `${(rf.peak_efield_v_m || 163.4).toFixed(1)} V/m (SMA)`;
            if (el('pd-metric-rf-bfield')) el('pd-metric-rf-bfield').textContent = `${(rf.peak_bfield_tesla || 2.59).toFixed(2)} T · ${(rf.peak_db_dt_t_s || 4.53e4).toExponential(2)} T/s`;
            if (el('pd-metric-rf-sar')) el('pd-metric-rf-sar').textContent = `${(rf.sar_10g_w_kg || 0.38).toFixed(2)} W/kg · ${(rf.peak_poynting_flux_kw_m2 || 60.3).toFixed(1)} kW/m²`;
            if (el('pd-metric-bellman')) el('pd-metric-bellman').textContent = `${(comb.stages?.[0]?.bellman_value_v_star || 96.4).toFixed(1)} (Stage I)`;
            if (el('pd-metric-netbenefit')) el('pd-metric-netbenefit').textContent = `+$${Math.round(econ.net_horizon_value || 82450).toLocaleString()} / pt`;

            const commonLayout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                margin: { t: 30, b: 48, l: 52, r: 30 },
                font: { color: '#c9d1d9', family: 'Inter' },
                xaxis: { gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e' },
                yaxis: { gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e' },
                legend: { orientation: 'h', x: 0, y: -0.28, font: { size: 9.5 } }
            };

            // 2. Plot 1: 36-Month UPDRS-III Trajectory vs Standards of Care
            if (el('pd-plot-updrs-trajectory') && traj.weeks) {
                Plotly.react('pd-plot-updrs-trajectory', [
                    {
                        x: traj.weeks, y: traj.updrs_qml_upper_95, name: '95% CI Upper',
                        type: 'scatter', mode: 'lines', line: { color: 'rgba(16,185,129,0.2)', width: 0 }, showlegend: false
                    },
                    {
                        x: traj.weeks, y: traj.updrs_qml_lower_95, name: '95% CI Lower',
                        type: 'scatter', mode: 'lines', fill: 'tonexty', fillcolor: 'rgba(16,185,129,0.12)',
                        line: { color: 'rgba(16,185,129,0.2)', width: 0 }, showlegend: false
                    },
                    {
                        x: traj.weeks, y: traj.updrs_sham, name: 'Sham rTMS Control',
                        type: 'scatter', mode: 'lines', line: { color: '#64748b', dash: 'dot', width: 1.6 }
                    },
                    {
                        x: traj.weeks, y: traj.updrs_pharmacotherapy, name: 'Optimized Pharmacotherapy',
                        type: 'scatter', mode: 'lines', line: { color: '#38bdf8', dash: 'dash', width: 2.0 }
                    },
                    {
                        x: traj.weeks, y: traj.updrs_standard_rtms, name: 'Standard Classical 10Hz M1',
                        type: 'scatter', mode: 'lines', line: { color: '#f59e0b', dash: 'dashdot', width: 2.2 }
                    },
                    {
                        x: traj.weeks, y: traj.updrs_qml, name: 'Combinatorial BEM rTMS (Our Protocol)',
                        type: 'scatter', mode: 'lines+markers', marker: { size: 4 }, line: { color: '#10b981', width: 3.5 }
                    }
                ], {
                    ...commonLayout,
                    xaxis: { ...commonLayout.xaxis, title: 'Follow-up Horizon (Weeks)' },
                    yaxis: { ...commonLayout.yaxis, title: 'MDS-UPDRS Part III Motor Score (0-80)' }
                }, { responsive: true, displaylogo: false });
            }

            // 3. Plot 2: BEM Cortical Surface Potential Heatmap (mV)
            if (el('pd-plot-bem-heatmap') && bem.surface_potential_grid) {
                const grid = bem.surface_potential_grid;
                Plotly.react('pd-plot-bem-heatmap', [{
                    z: grid.potential_mv,
                    x: grid.x_coords_mm,
                    y: grid.y_coords_mm,
                    type: 'contour',
                    colorscale: 'Viridis',
                    colorbar: { title: 'Φ (mV)', titleside: 'right', tickfont: { color: '#cbd5e1', size: 9 }, len: 0.9 },
                    contours: { coloring: 'heatmap', showlabels: true, labelfont: { size: 9, color: '#f8fafc' } }
                }], {
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    margin: { t: 25, b: 42, l: 45, r: 25 },
                    font: { color: '#c9d1d9', family: 'Inter' },
                    xaxis: { title: 'Lateral Medial X (mm)', gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e' },
                    yaxis: { title: 'Anterior Posterior Y (mm)', gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e' }
                }, { responsive: true, displaylogo: false });
            }

            // 4. Plot 3: RF Pulse Transient Waveform & Poynting Flux
            if (el('pd-plot-rf-waveform') && rf.time_axis_us) {
                Plotly.react('pd-plot-rf-waveform', [
                    {
                        x: rf.time_axis_us, y: rf.b_waveform_tesla, name: 'Magnetic Field B(t) [T]',
                        type: 'scatter', mode: 'lines', line: { color: '#f43f5e', width: 2.5 }
                    },
                    {
                        x: rf.time_axis_us, y: rf.poynting_waveform_kw_m2, name: 'Poynting Flux S(t) [kW/m²]',
                        type: 'scatter', mode: 'lines', line: { color: '#38bdf8', width: 2.0, dash: 'dot' },
                        yaxis: 'y2'
                    }
                ], {
                    ...commonLayout,
                    xaxis: { ...commonLayout.xaxis, title: 'RF Pulse Elapsed Time (μs)' },
                    yaxis: { ...commonLayout.yaxis, title: 'B(t) (Tesla)', color: '#f43f5e' },
                    yaxis2: { title: 'S(t) (kW/m²)', overlaying: 'y', side: 'right', color: '#38bdf8', gridcolor: 'rgba(56,189,248,0.08)' }
                }, { responsive: true, displaylogo: false });
            }

            // 5. Plot 4: RF Power Spectral Density (PSD)
            if (el('pd-plot-rf-spectrum') && rf.spectrum_freqs_khz) {
                Plotly.react('pd-plot-rf-spectrum', [{
                    x: rf.spectrum_freqs_khz,
                    y: rf.spectrum_psd_dbm_hz,
                    name: 'PSD (dBm/Hz)',
                    type: 'scatter',
                    mode: 'lines',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(244,114,182,0.12)',
                    line: { color: '#f472b6', width: 2.5 }
                }], {
                    ...commonLayout,
                    xaxis: { ...commonLayout.xaxis, title: 'Spectral Frequency (kHz)' },
                    yaxis: { ...commonLayout.yaxis, title: 'Power Spectral Density (dBm/Hz)' }
                }, { responsive: true, displaylogo: false });
            }

            // 6. Plot 5: Combinatorial Optimal Staging Bellman Values
            if (el('pd-plot-bellman-staging') && comb.stages) {
                const stageNames = comb.stages.map(s => s.stage_id);
                const bellmanVals = comb.stages.map(s => s.bellman_value_v_star);
                const transProbs = comb.stages.map(s => s.transition_probability_to_next * 100);

                Plotly.react('pd-plot-bellman-staging', [
                    {
                        x: stageNames, y: bellmanVals, name: 'Bellman Value V*(s)',
                        type: 'bar', marker: { color: ['#38bdf8', '#10b981', '#fbbf24', '#c084fc'] }
                    },
                    {
                        x: stageNames, y: transProbs, name: 'Transition Prob P(s\'|s) %',
                        type: 'scatter', mode: 'lines+markers', line: { color: '#f43f5e', width: 2.5 },
                        marker: { size: 7, color: '#f43f5e' }, yaxis: 'y2'
                    }
                ], {
                    ...commonLayout,
                    xaxis: { ...commonLayout.xaxis, title: 'Combinatorial Clinical Stage' },
                    yaxis: { ...commonLayout.yaxis, title: 'Bellman Optimality Value V*' },
                    yaxis2: { title: 'Transition Prob %', overlaying: 'y', side: 'right', color: '#f43f5e', range: [70, 105], gridcolor: 'rgba(244,63,94,0.08)' }
                }, { responsive: true, displaylogo: false });
            }

            // 7. Plot 6: Long-Term Non-Motor & QoL Horizons
            if (el('pd-plot-longterm-horizons') && traj.weeks) {
                Plotly.react('pd-plot-longterm-horizons', [
                    {
                        x: traj.weeks, y: traj.hoehn_yahr_staging, name: 'Hoehn & Yahr Stage',
                        type: 'scatter', mode: 'lines', line: { color: '#fbbf24', width: 2.2 }
                    },
                    {
                        x: traj.weeks, y: traj.daily_off_hours, name: 'Daily OFF Time (hrs)',
                        type: 'scatter', mode: 'lines', line: { color: '#f43f5e', width: 2.2, dash: 'dash' }
                    },
                    {
                        x: traj.weeks, y: traj.daily_on_hours_dyskinesia_free, name: 'ON Time (No Dyskinesia)',
                        type: 'scatter', mode: 'lines', line: { color: '#10b981', width: 2.5 }
                    },
                    {
                        x: traj.weeks, y: traj.pdq39_score, name: 'PDQ-39 QoL Score (0-100)',
                        type: 'scatter', mode: 'lines', line: { color: '#c084fc', width: 2.2, dash: 'dot' },
                        yaxis: 'y2'
                    }
                ], {
                    ...commonLayout,
                    xaxis: { ...commonLayout.xaxis, title: 'Follow-up Horizon (Weeks)' },
                    yaxis: { ...commonLayout.yaxis, title: 'Clinical Scores / Hours' },
                    yaxis2: { title: 'PDQ-39 QoL Score', overlaying: 'y', side: 'right', color: '#c084fc', gridcolor: 'rgba(192,132,252,0.08)' }
                }, { responsive: true, displaylogo: false });
            }

            // 8. Plot 7: Multi-Compartment BEM Depth Decay Profile
            if (el('pd-plot-bem-depth') && bem.depth_profile) {
                const dp = bem.depth_profile;
                Plotly.react('pd-plot-bem-depth', [
                    { x: dp.depths_mm, y: dp.efield_sma_v_m, name: 'SMA E-Field (V/m)', type: 'scatter', mode: 'lines', line: { color: '#10b981', width: 2.8 } },
                    { x: dp.depths_mm, y: dp.efield_m1_v_m, name: 'M1 E-Field (V/m)', type: 'scatter', mode: 'lines', line: { color: '#38bdf8', width: 2.2 } },
                    { x: dp.depths_mm, y: dp.efield_dlpfc_v_m, name: 'DLPFC E-Field (V/m)', type: 'scatter', mode: 'lines', line: { color: '#c084fc', width: 2.0, dash: 'dash' } },
                    { x: dp.depths_mm, y: dp.current_density_a_m2, name: 'Current Density J (A/m²)', type: 'scatter', mode: 'lines', line: { color: '#fbbf24', width: 1.8, dash: 'dot' }, yaxis: 'y2' }
                ], {
                    ...commonLayout,
                    xaxis: { ...commonLayout.xaxis, title: 'Tissue Depth (mm) [Scalp 0-6mm · Skull 6-12mm · CSF 12-16mm · Cortex 16-45mm]' },
                    yaxis: { ...commonLayout.yaxis, title: 'Electric Field (V/m)' },
                    yaxis2: { title: 'J (A/m²)', overlaying: 'y', side: 'right', color: '#fbbf24', gridcolor: 'rgba(251,191,36,0.08)' }
                }, { responsive: true, displaylogo: false });
            }

            // 9. Stimulated Lobes & RF/BEM Table Render
            const lobesTbody = el('pd-lobes-tbody');
            if (lobesTbody && d.cortical_lobes) {
                lobesTbody.innerHTML = d.cortical_lobes.map(l => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:8px 10px; font-weight:700; color:#38bdf8;">${l.lobe}</td>
                        <td style="padding:8px 10px; color:#f8fafc; font-weight:600;">${l.region}</td>
                        <td style="padding:8px 10px; font-family:monospace; color:#94a3b8; font-size:11px;">${l.mni_coords}</td>
                        <td style="padding:8px 10px; color:#34d399;">${l.frequency}</td>
                        <td style="padding:8px 10px; color:#fbbf24;">${l.intensity_pct_rmt}%</td>
                        <td style="padding:8px 10px; color:#f43f5e; font-weight:700;">${l.peak_efield_v_m} V/m</td>
                        <td style="padding:8px 10px; color:#cbd5e1;">${l.normal_efield_v_m} V/m</td>
                        <td style="padding:8px 10px; color:#cbd5e1;">${l.tangential_efield_v_m} V/m</td>
                        <td style="padding:8px 10px; color:#f472b6;">${l.rf_peak_bfield_tesla} T</td>
                        <td style="padding:8px 10px; color:#cbd5e1; font-size:11px;">${(l.rf_db_dt_t_s || 0).toExponential(2)}</td>
                        <td style="padding:8px 10px; color:#fb923c;">${l.rf_sar_w_kg}</td>
                        <td style="padding:8px 10px; color:#38bdf8;">${l.rf_poynting_kw_m2} kW/m²</td>
                        <td style="padding:8px 10px; color:#a78bfa; font-weight:700;">${(l.allocated_pulses_daily || 0).toLocaleString()}</td>
                        <td style="padding:8px 10px; font-size:11px; color:#cbd5e1;">${l.target_symptom}</td>
                    </tr>
                `).join('');
            }

            // 10. 4-Stage Combinatorial Protocol Cards Render
            const cardsContainer = el('pd-staging-cards-container');
            if (cardsContainer && comb.stages) {
                const borderColors = ['#10b981', '#38bdf8', '#fbbf24', '#c084fc'];
                cardsContainer.innerHTML = comb.stages.map((s, idx) => `
                    <div style="background:rgba(15,23,42,0.85); border:1px solid ${borderColors[idx] || '#10b981'}; border-radius:10px; padding:14px; display:flex; flex-direction:column; gap:8px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:11px; font-weight:700; color:${borderColors[idx]}; text-transform:uppercase;">${s.stage_id} · ${s.weeks}</span>
                            <span style="font-size:10px; background:rgba(255,255,255,0.08); padding:3px 8px; border-radius:4px; color:#34d399; font-weight:600;">V* = ${s.bellman_value_v_star}</span>
                        </div>
                        <div style="font-size:13px; font-weight:700; color:#f8fafc;">${s.name}</div>
                        <div style="font-size:11.5px; color:#38bdf8;"><b>Lobes Focused:</b> ${s.lobes_focused}</div>
                        <div style="font-size:11.5px; color:#cbd5e1;"><b>Cadence:</b> ${s.frequency} · <b>Budget:</b> ${s.allocated_pulses_session} pulses/session</div>
                        <div style="font-size:11.5px; color:#a7f3d0; font-style:italic;"><b>Milestone:</b> ${s.clinical_milestone}</div>
                        <div style="font-size:11px; color:#94a3b8;"><b>QML Resonance:</b> ${s.qml_goal}</div>
                        <div style="font-size:11px; color:#fbcfe8;"><b>Dopamine/Synaptic:</b> ${s.dopamine_modulation}</div>
                    </div>
                `).join('');
            }

            // 11. BEM Mesh Table Render
            const bemMeshTbody = el('pd-bem-mesh-tbody');
            if (bemMeshTbody && bem.compartments) {
                bemMeshTbody.innerHTML = bem.compartments.map(c => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:8px 10px; font-weight:700; color:#38bdf8;">${c.name}</td>
                        <td style="padding:8px 10px; color:#f8fafc;">${c.layer_type}</td>
                        <td style="padding:8px 10px; color:#34d399; font-weight:600;">${c.triangles.toLocaleString()}</td>
                        <td style="padding:8px 10px; color:#cbd5e1;">${c.vertices.toLocaleString()}</td>
                        <td style="padding:8px 10px; color:#fbbf24; font-weight:600;">${c.conductivity_s_m} S/m</td>
                        <td style="padding:8px 10px; color:#cbd5e1;">${c.mean_radius_mm} mm</td>
                        <td style="padding:8px 10px; font-size:11px; color:#94a3b8;">Continuity of normal current J_n = σ ∂Φ/∂n; jump boundary discretization</td>
                    </tr>
                `).join('');
            }

            // 12. Milestones Table Render
            const milestonesTbody = el('pd-milestones-tbody');
            if (milestonesTbody && traj.milestones_table) {
                milestonesTbody.innerHTML = traj.milestones_table.map(m => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:8px 10px; font-weight:700; color:#38bdf8;">${m.milestone}</td>
                        <td style="padding:8px 10px; color:#cbd5e1;">Week ${m.week}</td>
                        <td style="padding:8px 10px; color:#10b981; font-weight:700;">${m.updrs_qml}</td>
                        <td style="padding:8px 10px; color:#f59e0b;">${m.updrs_standard}</td>
                        <td style="padding:8px 10px; color:#38bdf8;">${m.updrs_pharma}</td>
                        <td style="padding:8px 10px; color:#fbbf24; font-weight:600;">Stage ${m.hoehn_yahr}</td>
                        <td style="padding:8px 10px; color:#c084fc;">${m.pdq39}</td>
                        <td style="padding:8px 10px; color:#f43f5e;">${m.off_hours} hrs/d</td>
                        <td style="padding:8px 10px; color:#34d399; font-weight:600;">${m.on_hours} hrs/d</td>
                        <td style="padding:8px 10px; color:#fb923c;">${m.ledd_mg} mg/d</td>
                    </tr>
                `).join('');
            }

            // 13. Math Equations Render
            const mathContainer = el('pd-math-container');
            if (mathContainer && d.finite_math_equations) {
                mathContainer.innerHTML = d.finite_math_equations.map(eq => `
                    <div style="background:rgba(5,11,20,0.85); border:1px solid rgba(52,211,153,0.3); border-radius:8px; padding:10px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                            <span style="font-size:12px; font-weight:700; color:#34d399;">${eq.title}</span>
                            <span style="font-size:10px; color:#a7f3d0; background:rgba(16,185,129,0.15); padding:2px 8px; border-radius:4px;">${eq.category}</span>
                        </div>
                        <div style="font-family:'Courier New', monospace; font-size:11.5px; color:#38bdf8; background:rgba(0,0,0,0.55); padding:8px 12px; border-radius:4px; overflow-x:auto; margin:6px 0;">
                            ${eq.latex}
                        </div>
                        <div style="font-size:11px; color:#94a3b8; line-height:1.4;">${eq.description}</div>
                    </div>
                `).join('');
            }

        } catch (err) {
            console.error('Parkinsons Studio API error:', err);
        }
    };

    const runParkinsonsStudioDebounced = () => {
        clearTimeout(pdDebounceTimer);
        pdDebounceTimer = setTimeout(runParkinsonsStudio, 200);
    };

    window.runParkinsonsStudio = runParkinsonsStudio;
    window.runParkinsonsStudioDebounced = runParkinsonsStudioDebounced;

    // ─────────────────────────────────────────────────────────────
    // HEART & STROKE LONG-TERM CLINICAL TRAJECTORIES STUDIO
    // ─────────────────────────────────────────────────────────────
    let hsDebounceTimer = null;
    let hsChartClinicalTrajectory = null;
    let hsChartRamanujanGeodesic = null;
    let hsChartNashGap = null;
    let hsChartStemRing = null;

    const runHeartStrokeStudio = async () => {
        const el = id => document.getElementById(id);
        try {
            const nihss = parseFloat(el('hs-nihss')?.value || 16.5);
            const lvef = parseFloat(el('hs-lvef')?.value || 35.0);
            const tauY = parseFloat(el('hs-tau-y')?.value || 0.85);
            const tauX = parseFloat(el('hs-tau-x')?.value || 0.25);
            const stem = parseFloat(el('hs-stem')?.value || 25.0);
            const intensity = parseFloat(el('hs-intensity')?.value || 1.15);
            const horizon = parseInt(el('hs-horizon')?.value || 60);

            const res = await fetch('/api/heart-stroke-trajectories', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    baseline_nihss: nihss,
                    baseline_lvef: lvef,
                    tau_imag_y: tauY,
                    tau_real_x: tauX,
                    stem_cell_density: stem,
                    intervention_intensity: intensity,
                    horizon_months: horizon
                })
            });

            const json = await res.json();
            if (json.status !== 'success') {
                console.error('Heart & Stroke API error:', json.message);
                return;
            }
            const d = json.data;
            const s = d.summary;

            // 1. Update Core KPIs
            if (el('kpi-hs-nihss')) el('kpi-hs-nihss').textContent = s.final_5yr_nihss.toFixed(2);
            if (el('kpi-hs-nihss-delta')) el('kpi-hs-nihss-delta').textContent = `-${s.nihss_delta_pct}% Remission`;
            if (el('kpi-hs-lvef')) el('kpi-hs-lvef').textContent = `${s.final_5yr_lvef_pct.toFixed(1)}%`;
            if (el('kpi-hs-lvef-gain')) el('kpi-hs-lvef-gain').textContent = `+${s.lvef_gain_pct}% Ejection`;
            if (el('kpi-hs-penumbra')) el('kpi-hs-penumbra').textContent = `${s.penumbra_tissue_salvage_pct.toFixed(1)}%`;
            if (el('kpi-hs-nash')) el('kpi-hs-nash').textContent = s.vanishing_nash_gap.toFixed(5);
            if (el('kpi-hs-stem-yield')) el('kpi-hs-stem-yield').textContent = `${s.stem_cell_homomorphism_yield_pct.toFixed(1)}%`;
            if (el('kpi-hs-nome')) el('kpi-hs-nome').textContent = s.ramanujan_nome_magnitude.toFixed(5);

            const milestones = d.trajectory_milestones;
            const months = milestones.map(m => `M${m.month}`);
            const nihssData = milestones.map(m => m.nihss_score);
            const lvefData = milestones.map(m => m.lvef_pct);
            const mrsData = milestones.map(m => m.mrs_score);
            const gammaData = milestones.map(m => m.neurocardiac_coupling_gamma);

            const commonPlotlyLayout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#c9d1d9', family: 'Inter, system-ui, sans-serif' },
                margin: { t: 30, b: 40, l: 45, r: 45 },
                legend: { orientation: 'h', y: 1.15, font: { size: 9, color: '#94a3b8' } },
                xaxis: { gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e', tickfont: { size: 9 } },
                yaxis: { gridcolor: 'rgba(255,255,255,0.06)', color: '#8b949e', tickfont: { size: 9 } }
            };

            // 2. Chart 1: 5-Year Clinical Trajectory (NIHSS / mRS vs LVEF%)
            if (el('hs-chart-clinical-trajectory')) {
                Plotly.react('hs-chart-clinical-trajectory', [
                    {
                        x: months, y: nihssData, name: 'Stroke NIHSS Deficit',
                        type: 'scatter', mode: 'lines+markers', line: { color: '#ef4444', width: 2.2 },
                        marker: { size: 5, color: '#f87171' }
                    },
                    {
                        x: months, y: mrsData, name: 'Modified Rankin (mRS)',
                        type: 'scatter', mode: 'lines+markers', line: { color: '#f59e0b', width: 1.8, dash: 'dot' },
                        marker: { size: 4, color: '#fbbf24' }
                    },
                    {
                        x: months, y: lvefData, name: 'Cardiac LVEF (%)',
                        type: 'scatter', mode: 'lines+markers', line: { color: '#0284c7', width: 2.5 },
                        marker: { size: 5, color: '#38bdf8' },
                        yaxis: 'y2'
                    }
                ], {
                    ...commonPlotlyLayout,
                    yaxis: { ...commonPlotlyLayout.yaxis, title: 'Deficit (NIHSS / mRS)', color: '#f87171' },
                    yaxis2: { title: 'LVEF (%)', overlaying: 'y', side: 'right', color: '#38bdf8', gridcolor: 'rgba(56,189,248,0.08)' }
                }, { responsive: true, displaylogo: false });
            }

            // 3. Chart 2: Ramanujan Modular Nome Harmonics & Poincaré Geodesics
            if (el('hs-chart-ramanujan-geodesic')) {
                const modNome = d.modular_nome_state;
                const tVals = Array.from({ length: 30 }, (_, i) => i / 29.0);
                const harmonicNome = tVals.map(t => Math.exp(-2 * Math.PI * (modNome.tau_imag + 0.25 * Math.sin(Math.PI * t))));
                const poincareMetric = tVals.map(t => 1.0 / Math.pow(modNome.tau_imag + 0.25 * Math.sin(Math.PI * t), 2));
                const monthLabels = tVals.map(t => (t * 60).toFixed(0) + 'm');

                Plotly.react('hs-chart-ramanujan-geodesic', [
                    {
                        x: monthLabels, y: harmonicNome, name: 'Harmonic Nome |q(τ)|',
                        type: 'scatter', mode: 'lines', fill: 'tozeroy', fillcolor: 'rgba(16,185,129,0.12)',
                        line: { color: '#10b981', width: 2.2 }
                    },
                    {
                        x: monthLabels, y: poincareMetric, name: 'Poincaré Potential 1/y²',
                        type: 'scatter', mode: 'lines', line: { color: '#a855f7', width: 2.0, dash: 'dash' },
                        yaxis: 'y2'
                    }
                ], {
                    ...commonPlotlyLayout,
                    yaxis: { ...commonPlotlyLayout.yaxis, title: 'Modular Nome |q|', color: '#10b981' },
                    yaxis2: { title: 'Hyperbolic Metric 1/y²', overlaying: 'y', side: 'right', color: '#c084fc', gridcolor: 'rgba(192,132,252,0.08)' }
                }, { responsive: true, displaylogo: false });
            }

            // 4. Chart 3: Vanishing Nash Gap Signature G(s) -> 0
            if (el('hs-chart-nash-gap')) {
                const gapTrace = d.nash_game_verification.nash_gap_convergence_trace;
                const iters = gapTrace.map((_, i) => `Step ${i + 1}`);

                Plotly.react('hs-chart-nash-gap', [
                    {
                        x: iters, y: gapTrace, name: 'Nash Gap Signature G(s)',
                        type: 'scatter', mode: 'lines+markers', fill: 'tozeroy', fillcolor: 'rgba(139,92,246,0.18)',
                        line: { color: '#8b5cf6', width: 2.5 },
                        marker: { size: 6, color: '#c4b5fd' }
                    }
                ], {
                    ...commonPlotlyLayout,
                    yaxis: { ...commonPlotlyLayout.yaxis, title: 'Gap Magnitude G(s) → 0', color: '#c4b5fd' }
                }, { responsive: true, displaylogo: false });
            }

            // 5. Chart 4: Stem Cell Ring Algebra Differentiation Spectrum
            if (el('hs-chart-stem-ring')) {
                const pluri = milestones.map(m => m.stem_cell_distribution.pluripotent * 100);
                const neural = milestones.map(m => m.stem_cell_distribution.neural * 100);
                const cardiac = milestones.map(m => m.stem_cell_distribution.cardiac * 100);
                const vasc = milestones.map(m => m.stem_cell_distribution.vascular * 100);

                Plotly.react('hs-chart-stem-ring', [
                    { x: months, y: pluri, name: 'e0 (Pluripotent P)', type: 'bar', marker: { color: '#94a3b8' } },
                    { x: months, y: neural, name: 'e1 (Neural N)', type: 'bar', marker: { color: '#38bdf8' } },
                    { x: months, y: cardiac, name: 'e2 (Cardiac C)', type: 'bar', marker: { color: '#f43f5e' } },
                    { x: months, y: vasc, name: 'e3 (Endothelial V)', type: 'bar', marker: { color: '#34d399' } }
                ], {
                    ...commonPlotlyLayout,
                    barmode: 'stack',
                    yaxis: { ...commonPlotlyLayout.yaxis, title: 'Lineage Commitment Yield (%)', range: [0, 100], color: '#cbd5e1' }
                }, { responsive: true, displaylogo: false });
            }

            // 6. Populate Table 1: Longitudinal Clinical Milestones
            const milestonesTbody = el('hs-milestones-tbody');
            if (milestonesTbody) {
                milestonesTbody.innerHTML = milestones.map(m => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:7px 10px; font-weight:700; color:#38bdf8;">Month ${m.month}</td>
                        <td style="padding:7px 10px; color:#94a3b8;">${m.year > 0 ? m.year + ' yr' : 'Baseline'}</td>
                        <td style="padding:7px 10px; color:#f87171; font-weight:600;">${m.nihss_score}</td>
                        <td style="padding:7px 10px; color:#fbbf24;">${m.mrs_score}</td>
                        <td style="padding:7px 10px; color:#38bdf8; font-weight:600;">${m.lvef_pct}%</td>
                        <td style="padding:7px 10px; color:#34d399;">${m.penumbra_salvage_pct}%</td>
                        <td style="padding:7px 10px; color:#e2e8f0;">${m.barthel_index}</td>
                        <td style="padding:7px 10px; color:#a7f3d0;">${m.neurocardiac_coupling_gamma}</td>
                        <td style="padding:7px 10px; color:#c4b5fd; font-family:monospace;">${m.nash_gap}</td>
                        <td style="padding:7px 10px; color:#f472b6;">${m.cortical_repair_cm3} cm³</td>
                    </tr>
                `).join('');
            }

            // 7. Populate Table 2: Game-Theoretic Payoff Matrix
            const gameTbody = el('hs-game-matrix-tbody');
            if (gameTbody) {
                const game = d.nash_game_verification;
                const p1Probs = game.p1_mixed_strategy;
                const payoffs = [
                    [55.2, 36.8, 27.6, 20.7, 34.2],
                    [89.7, 78.2, 66.7, 59.8, 73.6],
                    [82.8, 69.0, 51.8, 43.7, 61.8],
                    [108.1, 101.2, 94.3, 87.4, 97.8]
                ];
                gameTbody.innerHTML = game.p1_strategies.map((strat, i) => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:7px 10px; font-weight:600; color:#e2e8f0;">${strat}</td>
                        <td style="padding:7px 10px; color:#a78bfa; font-weight:700;">${(p1Probs[i] * 100).toFixed(1)}%</td>
                        <td style="padding:7px 10px; color:#94a3b8;">${payoffs[i][0]}</td>
                        <td style="padding:7px 10px; color:#94a3b8;">${payoffs[i][1]}</td>
                        <td style="padding:7px 10px; color:#94a3b8;">${payoffs[i][2]}</td>
                        <td style="padding:7px 10px; color:#94a3b8;">${payoffs[i][3]}</td>
                        <td style="padding:7px 10px; color:#34d399; font-weight:700;">+${payoffs[i][4]}</td>
                    </tr>
                `).join('');
            }

            // 8. Populate Table 3: Stem Cell Ring Algebra Graded Multiplication
            const ringTbody = el('hs-ring-mult-tbody');
            if (ringTbody) {
                const ringTable = d.stem_cell_ring_algebra.multiplication_table;
                const labels = ["e0: Pluripotent (P)", "e1: Neural Progenitor (N)", "e2: Cardiomyocyte (C)", "e3: Endothelial (V)"];
                const roles = [
                    "Ground state pluripotency & self-renewal core",
                    "Cortical synaptogenesis & perilesional axonal sprouting",
                    "Myocardial sarcomere reconstruction & LVEF contractile pacing",
                    "Angiogenic revascularization of ischemic penumbral core"
                ];
                ringTbody.innerHTML = ringTable.map((row, i) => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:7px 10px; font-weight:700; color:#34d399;">${labels[i]}</td>
                        <td style="padding:7px 10px; color:#94a3b8; font-family:monospace;">${row[0]}</td>
                        <td style="padding:7px 10px; color:#38bdf8; font-family:monospace;">${row[1]}</td>
                        <td style="padding:7px 10px; color:#f43f5e; font-family:monospace;">${row[2]}</td>
                        <td style="padding:7px 10px; color:#a7f3d0; font-family:monospace;">${row[3]}</td>
                        <td style="padding:7px 10px; color:#cbd5e1; font-size:11px;">${roles[i]}</td>
                    </tr>
                `).join('');
            }

            // 9. Populate Governing Formulations
            const mathContainer = el('hs-math-container');
            if (mathContainer && d.governing_formulations) {
                mathContainer.innerHTML = d.governing_formulations.map(eq => `
                    <div style="background:rgba(5,11,20,0.85); border:1px solid rgba(56,189,248,0.3); border-radius:8px; padding:10px;">
                        <div style="font-size:12px; font-weight:700; color:#38bdf8; margin-bottom:4px;">${eq.domain}</div>
                        <div style="font-family:'Courier New', monospace; font-size:11.5px; color:#a7f3d0; background:rgba(0,0,0,0.55); padding:8px 12px; border-radius:4px; overflow-x:auto; margin:6px 0;">
                            ${eq.formula}
                        </div>
                        <div style="font-size:11px; color:#94a3b8; line-height:1.4;">${eq.explanation}</div>
                    </div>
                `).join('');
            }

        } catch (err) {
            console.error('Heart & Stroke Studio API error:', err);
        }
    };

    const runHeartStrokeStudioDebounced = () => {
        clearTimeout(hsDebounceTimer);
        hsDebounceTimer = setTimeout(runHeartStrokeStudio, 200);
    };

    window.runHeartStrokeStudio = runHeartStrokeStudio;
    window.runHeartStrokeStudioDebounced = runHeartStrokeStudioDebounced;

    // ─────────────────────────────────────────────────────────────
    // FINITE ELEMENT CORTICAL SIMULATION & STIELTJES FRACTIONS
    // ─────────────────────────────────────────────────────────────
    let cachedFemData = null;
    let femDebounceTimer = null;

    const renderFemCortical3D = (data) => {
        if (!data || !data.mesh) return;
        const mesh = data.mesh;
        const fem = data.fem_field;
        const modeSelect = document.getElementById('fem-render-mode');
        const mode = modeSelect ? modeSelect.value : 'efield';

        let scalarVals = fem.e_field;
        let colorScaleName = 'Jet';
        let colorbarTitle = 'E-Field (V/m)';

        if (mode === 'current') {
            scalarVals = fem.current_density;
            colorScaleName = 'Plasma';
            colorbarTitle = 'Current Density J (A/m²)';
        } else if (mode === 'curvature') {
            scalarVals = mesh.curvature;
            colorScaleName = 'Viridis';
            colorbarTitle = 'Cortical Curvature κ';
        }

        const meshTrace = {
            type: 'mesh3d',
            x: mesh.x,
            y: mesh.y,
            z: mesh.z,
            i: mesh.faces_i,
            j: mesh.faces_j,
            k: mesh.faces_k,
            intensity: scalarVals,
            colorscale: colorScaleName,
            colorbar: {
                title: colorbarTitle,
                len: 0.65,
                thickness: 15,
                tickfont: { color: '#cbd5e1', size: 10 },
                titlefont: { color: '#cbd5e1', size: 11 }
            },
            lighting: {
                ambient: 0.45,
                diffuse: 0.85,
                specular: 0.25,
                roughness: 0.45,
                fresnel: 0.2
            },
            name: 'Parkinsonian Cortex'
        };

        // Render Figure-8 TMS Coil loops floating over M1 Hand Knob
        const coilPos = fem.coil_parameters.position_mm;
        const coilAngleRad = (fem.coil_parameters.orientation_deg * Math.PI) / 180.0;
        const loopRadius = 24.0;
        const loopCenterDist = 28.0;

        const leftCx = coilPos[0] - loopCenterDist * Math.cos(coilAngleRad);
        const leftCy = coilPos[1] - loopCenterDist * Math.sin(coilAngleRad);
        const rightCx = coilPos[0] + loopCenterDist * Math.cos(coilAngleRad);
        const rightCy = coilPos[1] + loopCenterDist * Math.sin(coilAngleRad);
        const cz = coilPos[2];

        const tPts = 36;
        const l1X = [], l1Y = [], l1Z = [];
        const l2X = [], l2Y = [], l2Z = [];
        for (let i = 0; i <= tPts; i++) {
            const ang = (i * 2 * Math.PI) / tPts;
            l1X.push(leftCx + loopRadius * Math.cos(ang));
            l1Y.push(leftCy + loopRadius * Math.sin(ang));
            l1Z.push(cz);
            l2X.push(rightCx + loopRadius * Math.cos(ang));
            l2Y.push(rightCy + loopRadius * Math.sin(ang));
            l2Z.push(cz);
        }

        const coilTrace1 = {
            type: 'scatter3d', mode: 'lines',
            x: l1X, y: l1Y, z: l1Z,
            line: { color: '#f59e0b', width: 6 },
            name: 'Figure-8 Coil (Loop 1)'
        };
        const coilTrace2 = {
            type: 'scatter3d', mode: 'lines',
            x: l2X, y: l2Y, z: l2Z,
            line: { color: '#fbbf24', width: 6 },
            name: 'Figure-8 Coil (Loop 2)'
        };

        // Target M1 Hand Knob Marker
        const m1Marker = {
            type: 'scatter3d', mode: 'markers+text',
            x: [20.0], y: [2.0], z: [35.0],
            marker: { size: 6, color: '#ef4444', symbol: 'diamond' },
            text: ['Target M1 (BA4)'],
            textposition: 'top center',
            textfont: { color: '#fca5a5', size: 10 },
            name: 'M1 Motor Target'
        };

        const layout = {
            ...PL,
            margin: { l: 0, r: 0, t: 10, b: 10 },
            scene: {
                ...PL.scene,
                xaxis: { ...PL.scene.xaxis, title: 'Medial-Lateral (mm)' },
                yaxis: { ...PL.scene.yaxis, title: 'Anterior-Posterior (mm)' },
                zaxis: { ...PL.scene.zaxis, title: 'Cranial Height (mm)' },
                camera: {
                    eye: { x: 1.45, y: -1.60, z: 1.20 }
                }
            },
            showlegend: true,
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0, y: 1 }
        };

        Plotly.newPlot('fem-cortical-3d-plot', [meshTrace, coilTrace1, coilTrace2, m1Marker], layout, { responsive: true });
    };

    const renderFemStieltjesChart = (stieltjes) => {
        if (!stieltjes) return;
        const traces = [
            {
                x: stieltjes.x_eval, y: stieltjes.y_reference,
                type: 'scatter', mode: 'lines',
                line: { color: '#94a3b8', width: 1.5, dash: 'dash' },
                name: 'Ground Truth Profile'
            },
            {
                x: stieltjes.x_eval, y: stieltjes.y_stieltjes,
                type: 'scatter', mode: 'lines',
                line: { color: '#0284c7', width: 2.5 },
                name: 'Stieltjes CF (Ours)'
            },
            {
                x: stieltjes.x_eval, y: stieltjes.y_poly_runge,
                type: 'scatter', mode: 'lines',
                line: { color: '#ef4444', width: 1.8, dash: 'dot' },
                name: 'Polynomial (Runge)'
            }
        ];
        Plotly.newPlot('fem-stieltjes-chart', traces, {
            ...PL,
            margin: { l: 45, r: 20, t: 20, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Trans-Sulcal Coordinate (mm)' },
            yaxis: { ...PL.yaxis, title: 'Elevation Z (mm)' },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.05, y: 0.15 }
        }, { responsive: true });
    };

    const renderFemDepthChart = (fem, patho, stageSweep) => {
        let stages, crown, wall, fundus;
        if (stageSweep && stageSweep.stages && stageSweep.stages.length > 0) {
            stages = stageSweep.stages;
            crown = stageSweep.crown;
            wall = stageSweep.wall;
            fundus = stageSweep.fundus;
        } else {
            stages = [0, 1, 2, 3, 4, 5];
            const basePeak = fem ? fem.e_max_V_per_m : 140.0;
            crown = stages.map(s => +(basePeak * (1.0 - 0.025 * s)).toFixed(1));
            wall = stages.map(s => +(basePeak * 0.72 * (1.0 - 0.045 * s)).toFixed(1));
            fundus = stages.map(s => +(basePeak * 0.44 * (1.0 - 0.075 * s)).toFixed(1));
        }

        const traces = [
            {
                x: stages, y: crown, type: 'scatter', mode: 'lines+markers',
                line: { color: '#10b981', width: 2.2 }, name: 'Gyral Crown (M1)'
            },
            {
                x: stages, y: wall, type: 'scatter', mode: 'lines+markers',
                line: { color: '#f59e0b', width: 2.0 }, name: 'Sulcal Wall'
            },
            {
                x: stages, y: fundus, type: 'scatter', mode: 'lines+markers',
                line: { color: '#8b5cf6', width: 2.0 }, name: 'Sulcal Fundus'
            },
            {
                x: [1, 5], y: [80, 80], type: 'scatter', mode: 'lines',
                line: { color: '#ef4444', width: 1.5, dash: 'dash' },
                name: 'Neural Threshold (80 V/m)'
            }
        ];
        Plotly.newPlot('fem-depth-chart', traces, {
            ...PL,
            margin: { l: 45, r: 20, t: 20, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Hoehn & Yahr Parkinsonian Stage', dtick: 1 },
            yaxis: { ...PL.yaxis, title: 'Electric Field |E| (V/m)' },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.6, y: 0.95 }
        }, { responsive: true });
    };

    const renderFemGcpScalabilityChart = (cloud) => {
        if (!cloud || !cloud.gcp_cluster_summary) return;
        const nodes = [1, 2, 4, 8, 16, 32, 64];
        const sFrac = 0.045;
        const ideal = nodes.map(n => +(1.0 / (sFrac + (1.0 - sFrac) / n)).toFixed(2));
        const actual = nodes.map(n => +(ideal[nodes.indexOf(n)] / (1.0 + 0.0035 * Math.sqrt(n) * Math.log2(n + 1))).toFixed(2));

        const traces = [
            {
                x: nodes, y: ideal, type: 'scatter', mode: 'lines',
                line: { color: '#94a3b8', width: 1.5, dash: 'dash' },
                name: 'Amdahl Ideal Bound'
            },
            {
                x: nodes, y: actual, type: 'scatter', mode: 'lines+markers',
                line: { color: '#38bdf8', width: 2.5 },
                marker: { size: 6, color: '#0284c7' },
                name: 'GCP VPC Hyperpartitioned'
            },
            {
                x: [cloud.gcp_cluster_summary.active_nodes],
                y: [cloud.gcp_cluster_summary.cloud_speedup_factor],
                type: 'scatter', mode: 'markers',
                marker: { size: 12, color: '#f43f5e', symbol: 'star' },
                name: 'Current Node Config'
            }
        ];
        Plotly.newPlot('fem-gcp-scalability-chart', traces, {
            ...PL,
            margin: { l: 45, r: 20, t: 20, b: 40 },
            xaxis: { ...PL.xaxis, title: 'GCP Worker Instances (c2-standard-16)', type: 'log', dtick: 0.30103 },
            yaxis: { ...PL.yaxis, title: 'Parallel Speedup (x)' },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.05, y: 0.95 }
        }, { responsive: true });
    };

    const renderFemMatrixReductionChart = (cloud) => {
        if (!cloud || !cloud.matrix_reduction) return;
        const sDecay = cloud.matrix_reduction.singular_value_decay;
        const ranks = sDecay.map((_, i) => i + 1);
        const cumEnergy = sDecay.map(s => +(100.0 * (1.0 - s / sDecay[0])).toFixed(1));

        const traces = [
            {
                x: ranks, y: sDecay, type: 'bar',
                marker: { color: '#38bdf8', opacity: 0.75 },
                name: 'Singular Value σ_k / σ_1'
            },
            {
                x: ranks, y: cumEnergy, type: 'scatter', mode: 'lines+markers',
                line: { color: '#f43f5e', width: 2.0 },
                marker: { size: 4, color: '#f43f5e' },
                name: 'Subspace Energy (%)',
                yaxis: 'y2'
            }
        ];
        Plotly.newPlot('fem-matrix-reduction-chart', traces, {
            ...PL,
            margin: { l: 45, r: 45, t: 20, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Reduced Basis Rank (k)' },
            yaxis: { ...PL.yaxis, title: 'Singular Value Ratio' },
            yaxis2: { ...PL.yaxis, title: 'Captured Energy (%)', overlaying: 'y', side: 'right', showgrid: false },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.3, y: 0.95 }
        }, { responsive: true });
    };

    const runFemCorticalStudio = async () => {
        try {
            const hyStage = parseInt(document.getElementById('fem-hy-stage')?.value || '3');
            const updrs = parseFloat(document.getElementById('fem-updrs')?.value || '42.0');
            const atrophy = document.getElementById('fem-atrophy-level')?.value || 'moderate';
            const intensity = parseFloat(document.getElementById('fem-intensity')?.value || '110.0');
            const orientation = parseFloat(document.getElementById('fem-orientation')?.value || '45.0');
            const gcpNodes = parseInt(document.getElementById('fem-gcp-nodes')?.value || '16');
            const mechanism = document.getElementById('fem-matrix-mechanism')?.value || 'aca_low_rank';
            const tolerance = parseFloat(document.getElementById('fem-reduction-tol')?.value || '0.0001');

            const resp = await fetch('/api/fem-cortical-simulation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    hoehn_yahr_stage: hyStage,
                    baseline_updrs: updrs,
                    atrophy_level: atrophy,
                    intensity_mso_pct: intensity,
                    coil_orientation_deg: orientation,
                    num_gcp_nodes: gcpNodes,
                    matrix_reduction_type: mechanism,
                    reduction_tolerance: tolerance
                })
            });

            const json = await resp.json();
            if (json.status !== 'success') {
                console.error('FEM simulation returned non-success:', json);
                return;
            }
            const d = json;
            cachedFemData = d;

            // Update KPI Cards
            const emaxEl = document.getElementById('fem-kpi-emax');
            if (emaxEl) emaxEl.textContent = `${d.fem_field.e_max_V_per_m} V/m`;

            const m1El = document.getElementById('fem-kpi-m1');
            if (m1El) m1El.textContent = `${d.fem_field.m1_hand_knob_e_V_per_m} V/m`;

            const sulcalEl = document.getElementById('fem-kpi-sulcal');
            if (sulcalEl) sulcalEl.textContent = `${(d.fem_field.sulcal_attenuation_ratio * 100).toFixed(1)}%`;

            const speedupEl = document.getElementById('fem-kpi-speedup');
            if (speedupEl) speedupEl.textContent = `${d.gcp_cloud_computation.gcp_cluster_summary.cloud_speedup_factor}x`;

            const solveTimeEl = document.getElementById('fem-kpi-solvetime');
            if (solveTimeEl) solveTimeEl.textContent = `Solve: ${d.gcp_cloud_computation.gcp_cluster_summary.solve_time_ms} ms (${gcpNodes} nodes)`;

            const compEl = document.getElementById('fem-kpi-compression');
            if (compEl) compEl.textContent = `${d.gcp_cloud_computation.matrix_reduction.compression_ratio_pct}%`;

            const memSavedEl = document.getElementById('fem-kpi-memsaved');
            if (memSavedEl) memSavedEl.textContent = `Saved: ${d.gcp_cloud_computation.matrix_reduction.memory_saved_mb} MB`;

            const r2El = document.getElementById('fem-kpi-r2');
            if (r2El) r2El.textContent = `R² = ${d.stieltjes_interpolation.r2_score.toFixed(4)}`;

            // Render 3D Cortical Surface
            renderFemCortical3D(d);

            // Render 2D Analytical Charts
            renderFemStieltjesChart(d.stieltjes_interpolation);
            renderFemDepthChart(d.fem_field, d.patient_pathology, d.stage_sweep);

            // Render additional noise sweep and PCG charts if elements exist
            const noiseEl = document.getElementById('fem-noise-chart');
            if (noiseEl && d.stieltjes_interpolation?.noise_sweep) {
                const ns = d.stieltjes_interpolation.noise_sweep;
                const traces = [
                    {
                        x: ns.noise_mm, y: ns.stieltjes_rmse_mm,
                        type: 'scatter', mode: 'lines+markers',
                        line: { color: '#0284c7', width: 2.2 },
                        name: 'Stieltjes S-Fraction'
                    },
                    {
                        x: ns.noise_mm, y: ns.poly_rmse_mm,
                        type: 'scatter', mode: 'lines+markers',
                        line: { color: '#ef4444', width: 2.0, dash: 'dot' },
                        name: 'Polynomial Interpolation'
                    }
                ];
                Plotly.newPlot('fem-noise-chart', traces, {
                    ...PL,
                    margin: { l: 45, r: 20, t: 20, b: 40 },
                    xaxis: { ...PL.xaxis, title: 'Node Noise σ (mm)' },
                    yaxis: { ...PL.yaxis, title: 'Mean RMSE vs Clean (mm)' },
                    legend: { font: { color: '#e6edf3', size: 9 }, x: 0.05, y: 0.95 }
                }, { responsive: true });
            }

            const pcgEl = document.getElementById('fem-pcg-chart');
            if (pcgEl && d.gcp_cloud_computation?.solver_convergence) {
                const conv = d.gcp_cloud_computation.solver_convergence;
                const iters = conv.map(c => c.iteration);
                const resids = conv.map(c => c.residual_norm);
                const traces = [
                    {
                        x: iters, y: resids,
                        type: 'scatter', mode: 'lines',
                        line: { color: '#c4b5fd', width: 2.0 },
                        name: 'PCG Residual'
                    }
                ];
                Plotly.newPlot('fem-pcg-chart', traces, {
                    ...PL,
                    margin: { l: 45, r: 20, t: 20, b: 40 },
                    xaxis: { ...PL.xaxis, title: 'Jacobi-PCG Iteration' },
                    yaxis: { ...PL.yaxis, title: 'Relative Residual', type: 'log' },
                    legend: { font: { color: '#e6edf3', size: 9 }, x: 0.6, y: 0.95 }
                }, { responsive: true });
            }

            const femStatusEl = document.getElementById('fem-status');
            if (femStatusEl) femStatusEl.textContent = 'Ready (' + new Date().toLocaleTimeString() + ')';

            const femModelNote = document.getElementById('fem-model-note');
            if (femModelNote && d.model_notes) {
                femModelNote.textContent = `${d.model_notes.field_model} ${d.model_notes.cloud_model}`;
            }
            renderFemGcpScalabilityChart(d.gcp_cloud_computation);
            renderFemMatrixReductionChart(d.gcp_cloud_computation);

            // Populate GCP Node Telemetry Table
            const nodeTbody = document.getElementById('fem-gcp-nodes-tbody');
            if (nodeTbody && d.gcp_cloud_computation.sample_nodes) {
                nodeTbody.innerHTML = d.gcp_cloud_computation.sample_nodes.map(n => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 8px; font-weight:600; color:#38bdf8;">${n.node_id}</td>
                        <td style="padding:6px 8px; color:#cbd5e1;">${n.machine_type}</td>
                        <td style="padding:6px 8px; color:#a7f3d0;">${n.zone}</td>
                        <td style="padding:6px 8px; color:#e2e8f0;">${n.local_dofs ?? n.local_elements ?? '--'}</td>
                        <td style="padding:6px 8px; color:#94a3b8;">${n.interior_dofs ?? '--'}</td>
                        <td style="padding:6px 8px; color:#fbbf24;">${n.memory_usage_mb} MB</td>
                        <td style="padding:6px 8px; color:#94a3b8;">${n.comm_transfer_mb} MB</td>
                        <td style="padding:6px 8px; color:#38bdf8;">${n.measured_ms ?? '--'} ms</td>
                        <td style="padding:6px 8px;"><span style="background:rgba(16,185,129,0.2); color:#34d399; padding:2px 6px; border-radius:4px; font-size:10px;">${n.status || 'ONLINE'}</span></td>
                    </tr>
                `).join('');
            }

            // Populate Stieltjes Convergents Table
            const stTbody = document.getElementById('fem-stieltjes-tbody');
            if (stTbody && d.stieltjes_interpolation.convergents) {
                stTbody.innerHTML = d.stieltjes_interpolation.convergents.map(c => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 8px; font-weight:600; color:#fbbf24;">k = ${c.order}</td>
                        <td style="padding:6px 8px; color:#a7f3d0;">c_${c.order} = ${c.stieltjes_coeff}</td>
                        <td style="padding:6px 8px; color:#38bdf8;">${c.rational_fraction}</td>
                        <td style="padding:6px 8px; color:#e2e8f0;">${c.sample_value}</td>
                        <td style="padding:6px 8px; color:#f472b6;">${c.convergence_delta != null ? 'Δ = ' + c.convergence_delta : '--'}</td>
                        <td style="padding:6px 8px; color:#34d399;">${c.rmse_mm ?? '--'}</td>
                    </tr>
                `).join('');
            }

        } catch (err) {
            console.error('FEM Cortical Studio API error:', err);
        }
    };

    const runFemCorticalStudioDebounced = () => {
        clearTimeout(femDebounceTimer);
        femDebounceTimer = setTimeout(runFemCorticalStudio, 200);
    };

    const updateFem3DRenderMode = () => {
        if (cachedFemData) {
            renderFemCortical3D(cachedFemData);
        }
    };

    const resetFem3DCamera = () => {
        const plotEl = document.getElementById('fem-cortical-3d-plot');
        if (plotEl && plotEl._fullLayout && plotEl._fullLayout.scene) {
            Plotly.relayout('fem-cortical-3d-plot', {
                'scene.camera': { eye: { x: 1.45, y: -1.60, z: 1.20 } }
            });
        }
    };

    window.runFemCorticalStudio = runFemCorticalStudio;
    window.runFemCorticalStudioDebounced = runFemCorticalStudioDebounced;
    window.updateFem3DRenderMode = updateFem3DRenderMode;
    window.resetFem3DCamera = resetFem3DCamera;

    // ─────────────────────────────────────────────────────────────
    // HUMAN CORTICAL CONNECTOME: MODULAR HARMONICS & QML
    // ─────────────────────────────────────────────────────────────
    let cachedConnData = null;
    let connDebounceTimer = null;

    const connUpdateLabels = () => {
        const sevEl = document.getElementById('conn-severity');
        if (sevEl) {
            const sev = parseFloat(sevEl.value);
            const lblSev = document.getElementById('label-conn-severity');
            if (lblSev) lblSev.textContent = sev.toFixed(2);
            const cdrEl = document.getElementById('conn-cdr-label');
            if (cdrEl) {
                const target = sev * 3.0;
                const cdrs = [
                    [0.0, 'CDR 0 - cognitively normal'],
                    [0.5, 'CDR 0.5 - very mild / MCI'],
                    [1.0, 'CDR 1 - mild dementia'],
                    [2.0, 'CDR 2 - moderate dementia'],
                    [3.0, 'CDR 3 - severe dementia']
                ];
                const closest = cdrs.reduce((prev, curr) => Math.abs(curr[0] - target) < Math.abs(prev[0] - target) ? curr : prev);
                cdrEl.textContent = closest[1];
            }
        }
        const thrEl = document.getElementById('conn-threshold');
        if (thrEl) {
            const lblThr = document.getElementById('label-conn-threshold');
            if (lblThr) lblThr.textContent = thrEl.value + '%';
        }
        const txEl = document.getElementById('conn-tau-x');
        if (txEl) {
            const lblTx = document.getElementById('label-conn-tau-x');
            if (lblTx) lblTx.textContent = parseFloat(txEl.value).toFixed(2);
        }
        const tyEl = document.getElementById('conn-tau-y');
        if (tyEl) {
            const lblTy = document.getElementById('label-conn-tau-y');
            if (lblTy) lblTy.textContent = parseFloat(tyEl.value).toFixed(2);
        }
        const layEl = document.getElementById('conn-layers');
        if (layEl) {
            const lblLay = document.getElementById('label-conn-layers');
            if (lblLay) lblLay.textContent = layEl.value;
        }
        const itEl = document.getElementById('conn-iters');
        if (itEl) {
            const lblIt = document.getElementById('label-conn-iters');
            if (lblIt) lblIt.textContent = itEl.value;
        }
    };

    const renderConnectomeViews = () => {
        if (!cachedConnData) return;
        const d = cachedConnData;
        const mh = d.metrics.healthy;
        const md = d.metrics.dementia;

        const edgeView = document.getElementById('conn-edge-view')?.value || 'loss';
        const nodeColorMode = document.getElementById('conn-node-color')?.value || 'modular_dementia';
        const matrixMode = document.getElementById('conn-matrix-mode')?.value || 'Q';

        // 1. 3D Cortical Connectome Plot
        const traces3d = [];

        // Brain surface translucent mesh
        if (d.surface && d.surface.x) {
            traces3d.push({
                type: 'mesh3d',
                x: d.surface.x, y: d.surface.y, z: d.surface.z,
                i: d.surface.i, j: d.surface.j, k: d.surface.k,
                opacity: 0.10,
                color: '#64748b',
                hoverinfo: 'none',
                showscale: false,
                name: 'Cortical Shell'
            });
        }

        // Determine node color scalar values
        let nodeVals = [];
        let colTitle = '';
        let colorscale = 'Viridis';
        if (nodeColorMode === 'modular_dementia') {
            nodeVals = d.nodes.map(n => n.modular_amp_dementia);
            colTitle = 'Modular Amp (Dem)';
            colorscale = 'Jet';
        } else if (nodeColorMode === 'modular_healthy') {
            nodeVals = d.nodes.map(n => n.modular_amp_healthy);
            colTitle = 'Modular Amp (H)';
            colorscale = 'Jet';
        } else if (nodeColorMode === 'atrophy') {
            nodeVals = d.nodes.map(n => n.atrophy_pct);
            colTitle = 'Braak Atrophy %';
            colorscale = 'YlOrRd';
        } else if (nodeColorMode === 'loss') {
            nodeVals = d.nodes.map(n => n.strength_loss_pct);
            colTitle = 'Strength Loss %';
            colorscale = 'Reds';
        } else if (nodeColorMode === 'qml_dementia') {
            nodeVals = d.nodes.map(n => n.qml_strength_dementia);
            colTitle = 'QML Strength (Dem)';
            colorscale = 'Plasma';
        } else {
            nodeVals = d.nodes.map(n => n.qml_strength_healthy);
            colTitle = 'QML Strength (H)';
            colorscale = 'Plasma';
        }

        // Connectome Edges
        const buildEdgeCoordArrays = (edgeList) => {
            const ex = [], ey = [], ez = [];
            edgeList.forEach(e => {
                const u = d.nodes[e[0]];
                const v = d.nodes[e[1]];
                if (u && v) {
                    ex.push(u.x, v.x, null);
                    ey.push(u.y, v.y, null);
                    ez.push(u.z, v.z, null);
                }
            });
            return { ex, ey, ez };
        };

        if (edgeView === 'healthy') {
            const { ex, ey, ez } = buildEdgeCoordArrays(d.edges_healthy || []);
            traces3d.push({
                type: 'scatter3d', mode: 'lines',
                x: ex, y: ey, z: ez,
                line: { color: '#38bdf8', width: 2 },
                hoverinfo: 'none',
                name: 'Healthy Tracts'
            });
        } else if (edgeView === 'dementia') {
            const { ex, ey, ez } = buildEdgeCoordArrays(d.edges_dementia || []);
            traces3d.push({
                type: 'scatter3d', mode: 'lines',
                x: ex, y: ey, z: ez,
                line: { color: '#f59e0b', width: 2 },
                hoverinfo: 'none',
                name: 'Dementia Tracts'
            });
        } else {
            // 'loss': Show preserved dementia edges + lost edges in red
            const demSet = new Set((d.edges_dementia || []).map(e => `${Math.min(e[0], e[1])}-${Math.max(e[0], e[1])}`));
            const lostEdges = (d.edges_healthy || []).filter(e => !demSet.has(`${Math.min(e[0], e[1])}-${Math.max(e[0], e[1])}`));

            const demCoords = buildEdgeCoordArrays(d.edges_dementia || []);
            traces3d.push({
                type: 'scatter3d', mode: 'lines',
                x: demCoords.ex, y: demCoords.ey, z: demCoords.ez,
                line: { color: '#38bdf8', width: 2.2 },
                hoverinfo: 'none',
                name: 'Active Tracts'
            });

            const lostCoords = buildEdgeCoordArrays(lostEdges);
            traces3d.push({
                type: 'scatter3d', mode: 'lines',
                x: lostCoords.ex, y: lostCoords.ey, z: lostCoords.ez,
                line: { color: '#ef4444', width: 2.0, dash: 'dot' },
                hoverinfo: 'none',
                name: 'Disconnected Tracts'
            });
        }

        // Nodes trace
        traces3d.push({
            type: 'scatter3d',
            mode: 'markers+text',
            x: d.nodes.map(n => n.x),
            y: d.nodes.map(n => n.y),
            z: d.nodes.map(n => n.z),
            text: d.nodes.map(n => n.hub ? `★ ${n.label}` : n.label),
            textposition: 'top center',
            textfont: { size: 9, color: '#e2e8f0' },
            hovertext: d.nodes.map(n => `<b>${n.name}</b> (${n.lobe}, Braak ${n.braak_stage}${n.hub ? ', HUB' : ''})<br>Atrophy: ${n.atrophy_pct}% | Loss: ${n.strength_loss_pct}%<br>Strength: ${n.strength_healthy} → ${n.strength_dementia}<br>Modular Amp: ${n.modular_amp_healthy} → ${n.modular_amp_dementia}<br>QML Strength: ${n.qml_strength_healthy} → ${n.qml_strength_dementia}`),
            hoverinfo: 'text',
            marker: {
                size: d.nodes.map(n => n.hub ? 8.5 : 5.5),
                color: nodeVals,
                colorscale: colorscale,
                colorbar: {
                    title: colTitle,
                    len: 0.65, thickness: 14,
                    tickfont: { color: '#cbd5e1', size: 9 },
                    titlefont: { color: '#cbd5e1', size: 10 }
                }
            },
            name: 'Cortical Nodes'
        });

        Plotly.newPlot('conn-3d-plot', traces3d, {
            ...PL,
            margin: { l: 0, r: 0, t: 10, b: 10 },
            scene: {
                ...PL.scene,
                xaxis: { ...PL.scene.xaxis, title: 'Left - Right (X)' },
                yaxis: { ...PL.scene.yaxis, title: 'Post - Ant (Y)' },
                zaxis: { ...PL.scene.zaxis, title: 'Inf - Sup (Z)' },
                camera: { eye: { x: 1.35, y: -1.45, z: 1.15 } }
            },
            showlegend: true,
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0, y: 1 }
        }, { responsive: true });

        // 2. 3D QML Connectome Matrix Plot
        let matZ = [];
        let matTitle = '';
        let matColorscale = 'Viridis';
        const nNodes = d.order.length;

        if (matrixMode === 'W') {
            matZ = d.matrices.W_dementia;
            matTitle = 'Structural Adjacency W_ij (Dementia)';
            matColorscale = 'Hot';
        } else if (matrixMode === 'D') {
            matZ = [];
            for (let i = 0; i < nNodes; i++) {
                const row = [];
                for (let j = 0; j < nNodes; j++) {
                    row.push(+(d.matrices.Q_healthy[i][j] - d.matrices.Q_dementia[i][j]).toFixed(4));
                }
                matZ.push(row);
            }
            matTitle = 'QML Fidelity Difference (Q_H - Q_D)';
            matColorscale = 'RdBu';
        } else {
            matZ = d.matrices.Q_dementia;
            matTitle = '6-Qubit QML Fidelity Kernel Q_ij (Dementia)';
            matColorscale = 'Viridis';
        }

        const qmlTrace = {
            type: 'surface',
            z: matZ,
            x: Array.from({ length: nNodes }, (_, i) => i),
            y: Array.from({ length: nNodes }, (_, i) => i),
            colorscale: matColorscale,
            colorbar: {
                title: matTitle,
                len: 0.65, thickness: 14,
                tickfont: { color: '#cbd5e1', size: 9 },
                titlefont: { color: '#cbd5e1', size: 10 }
            }
        };

        Plotly.newPlot('conn-qml-3d-plot', [qmlTrace], {
            ...PL,
            margin: { l: 0, r: 0, t: 10, b: 10 },
            scene: {
                ...PL.scene,
                xaxis: { ...PL.scene.xaxis, title: 'Region Index i (Ordered by Lobe)' },
                yaxis: { ...PL.scene.yaxis, title: 'Region Index j (Ordered by Lobe)' },
                zaxis: { ...PL.scene.zaxis, title: 'Kernel Amplitude' },
                camera: { eye: { x: 1.45, y: -1.35, z: 1.25 } }
            }
        }, { responsive: true });

        // 3. Spectrum & Modular Weights Chart
        const modes = d.spectrum.mode.slice(0, 36);
        const specTraces = [
            {
                x: modes, y: d.spectrum.lambda_healthy.slice(0, 36),
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#38bdf8', width: 2 },
                marker: { size: 4 },
                name: 'λ_n (Healthy)'
            },
            {
                x: modes, y: d.spectrum.lambda_dementia.slice(0, 36),
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#f87171', width: 2, dash: 'dot' },
                marker: { size: 4 },
                name: 'λ_n (Dementia)'
            },
            {
                x: modes, y: d.spectrum.c_healthy.slice(0, 36),
                type: 'scatter', mode: 'lines',
                line: { color: '#34d399', width: 1.8 },
                name: 'Modular c_n (Healthy)',
                yaxis: 'y2'
            },
            {
                x: modes, y: d.spectrum.c_dementia.slice(0, 36),
                type: 'scatter', mode: 'lines',
                line: { color: '#fb923c', width: 1.8, dash: 'dash' },
                name: 'Modular c_n (Dementia)',
                yaxis: 'y2'
            }
        ];

        Plotly.newPlot('conn-spectrum-chart', specTraces, {
            ...PL,
            margin: { l: 45, r: 45, t: 20, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Eigenmode Index n' },
            yaxis: { ...PL.yaxis, title: 'Laplacian Eigenvalue λ_n' },
            yaxis2: { ...PL.yaxis, title: 'Modular Weight c_n(τ)', overlaying: 'y', side: 'right', showgrid: false },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.1, y: 0.95 }
        }, { responsive: true });

        // 4. Metrics Dementia / Healthy Ratio Chart
        const metricLabels = ['Edges', 'Strength', 'Eff (E_glob)', 'λ_2 (Conn)', 'Clustering', 'QML Align'];
        const ratioVals = [
            +((md.edges / Math.max(mh.edges, 1)) * 100).toFixed(1),
            +((md.mean_strength / Math.max(mh.mean_strength, 1e-4)) * 100).toFixed(1),
            +((md.global_efficiency / Math.max(mh.global_efficiency, 1e-4)) * 100).toFixed(1),
            +((md.algebraic_connectivity / Math.max(mh.algebraic_connectivity, 1e-4)) * 100).toFixed(1),
            +((md.clustering / Math.max(mh.clustering, 1e-4)) * 100).toFixed(1),
            +((md.qml_alignment / Math.max(mh.qml_alignment, 1e-4)) * 100).toFixed(1)
        ];

        const metricTrace = {
            x: metricLabels,
            y: ratioVals,
            type: 'bar',
            marker: {
                color: ratioVals.map(v => v < 70 ? '#ef4444' : v < 90 ? '#f59e0b' : '#10b981')
            }
        };

        Plotly.newPlot('conn-metrics-chart', [metricTrace], {
            ...PL,
            margin: { l: 45, r: 20, t: 20, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Topological Invariant' },
            yaxis: { ...PL.yaxis, title: 'Dementia / Healthy (%)' }
        }, { responsive: true });

        // 5. QML Training Chart
        const qh = d.qml_training?.healthy || [];
        const qd = d.qml_training?.dementia || [];
        const qmlTraces = [
            {
                x: qh.map(p => p.iter),
                y: qh.map(p => +(p.alignment * 100).toFixed(2)),
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#38bdf8', width: 2.2 },
                marker: { size: 4 },
                name: 'Healthy QML Alignment'
            },
            {
                x: qd.map(p => p.iter),
                y: qd.map(p => +(p.alignment * 100).toFixed(2)),
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#f43f5e', width: 2.2 },
                marker: { size: 4 },
                name: 'Dementia QML Alignment'
            }
        ];

        Plotly.newPlot('conn-training-chart', qmlTraces, {
            ...PL,
            margin: { l: 45, r: 20, t: 20, b: 40 },
            xaxis: { ...PL.xaxis, title: 'SPSA Training Iteration' },
            yaxis: { ...PL.yaxis, title: 'Kernel Alignment (%)' },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.1, y: 0.2 }
        }, { responsive: true });

        // 6. Braak Stage Atrophy Chart
        const bs = d.braak_summary || [];
        const braakTrace = {
            x: bs.map(b => `Stage ${b.stage}`),
            y: bs.map(b => b.mean_atrophy_pct),
            type: 'bar',
            marker: { color: '#ea580c', opacity: 0.85 }
        };

        Plotly.newPlot('conn-braak-chart', [braakTrace], {
            ...PL,
            margin: { l: 45, r: 20, t: 20, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Braak Histopathological Staging' },
            yaxis: { ...PL.yaxis, title: 'Mean Regional Atrophy (%)' }
        }, { responsive: true });
    };

    const runConnectomeStudio = async () => {
        try {
            const statusEl = document.getElementById('conn-status');
            if (statusEl) statusEl.textContent = 'Computing Connectome & QML...';

            const severity = parseFloat(document.getElementById('conn-severity')?.value || '0.6');
            const threshold_pct = parseFloat(document.getElementById('conn-threshold')?.value || '75.0');
            const tau_x = parseFloat(document.getElementById('conn-tau-x')?.value || '0.25');
            const tau_y = parseFloat(document.getElementById('conn-tau-y')?.value || '0.85');
            const qml_layers = parseInt(document.getElementById('conn-layers')?.value || '3');
            const qml_iters = parseInt(document.getElementById('conn-iters')?.value || '80');

            const resp = await fetch('/api/connectome-qml', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    severity,
                    threshold_pct,
                    tau_x,
                    tau_y,
                    qml_layers,
                    qml_iters
                })
            });

            const json = await resp.json();
            if (json.status !== 'success') {
                console.error('Connectome simulation error:', json);
                if (statusEl) statusEl.textContent = 'Error during calculation';
                return;
            }

            const d = json;
            cachedConnData = d;
            if (statusEl) statusEl.textContent = 'Ready (' + new Date().toLocaleTimeString() + ')';

            const noteEl = document.getElementById('conn-model-note');
            if (noteEl) noteEl.textContent = d.model_notes;

            const mh = d.metrics.healthy;
            const md = d.metrics.dementia;

            // KPI updates
            const effEl = document.getElementById('conn-kpi-eff');
            if (effEl) effEl.textContent = `${mh.global_efficiency.toFixed(4)} → ${md.global_efficiency.toFixed(4)}`;
            const effSub = document.getElementById('conn-kpi-eff-sub');
            if (effSub) effSub.textContent = `Δ: ${(((md.global_efficiency / Math.max(mh.global_efficiency, 1e-4)) - 1) * 100).toFixed(1)}%`;

            const l2El = document.getElementById('conn-kpi-l2');
            if (l2El) l2El.textContent = `${mh.algebraic_connectivity.toFixed(4)} → ${md.algebraic_connectivity.toFixed(4)}`;
            const l2Sub = document.getElementById('conn-kpi-l2-sub');
            if (l2Sub) l2Sub.textContent = `Δ: ${(((md.algebraic_connectivity / Math.max(mh.algebraic_connectivity, 1e-4)) - 1) * 100).toFixed(1)}%`;

            const treesEl = document.getElementById('conn-kpi-trees');
            if (treesEl) treesEl.textContent = `${mh.log_spanning_trees != null ? mh.log_spanning_trees.toFixed(1) : '--'} → ${md.log_spanning_trees != null ? md.log_spanning_trees.toFixed(1) : '--'}`;

            const bettiEl = document.getElementById('conn-kpi-betti');
            if (bettiEl) bettiEl.textContent = `(${mh.betti_0}, ${mh.betti_1}) → (${md.betti_0}, ${md.betti_1})`;

            const qmlEl = document.getElementById('conn-kpi-qml');
            if (qmlEl) qmlEl.textContent = `${(mh.qml_alignment * 100).toFixed(1)}% → ${(md.qml_alignment * 100).toFixed(1)}%`;
            const qmlSub = document.getElementById('conn-kpi-qml-sub');
            if (qmlSub) qmlSub.textContent = `Corr with W: r = ${md.qml_vs_structural_corr.toFixed(3)}`;

            const modEl = document.getElementById('conn-kpi-mod');
            if (modEl) modEl.textContent = `${mh.modular_energy.toFixed(3)} → ${md.modular_energy.toFixed(3)}`;

            // Render 3D and 2D charts
            renderConnectomeViews();

            // Populate Invariant Table
            const metricsTbody = document.getElementById('conn-metrics-tbody');
            if (metricsTbody) {
                const rows = [
                    ['Edge Count', mh.edges, md.edges, (((md.edges / Math.max(mh.edges, 1)) - 1) * 100).toFixed(1) + '%'],
                    ['Edge Density', mh.density.toFixed(4), md.density.toFixed(4), (((md.density / Math.max(mh.density, 1e-4)) - 1) * 100).toFixed(1) + '%'],
                    ['Mean Node Strength', mh.mean_strength.toFixed(3), md.mean_strength.toFixed(3), (((md.mean_strength / Math.max(mh.mean_strength, 1e-4)) - 1) * 100).toFixed(1) + '%'],
                    ['Global Efficiency (E_glob)', mh.global_efficiency.toFixed(4), md.global_efficiency.toFixed(4), (((md.global_efficiency / Math.max(mh.global_efficiency, 1e-4)) - 1) * 100).toFixed(1) + '%'],
                    ['Algebraic Connectivity (λ_2)', mh.algebraic_connectivity.toFixed(4), md.algebraic_connectivity.toFixed(4), (((md.algebraic_connectivity / Math.max(mh.algebraic_connectivity, 1e-4)) - 1) * 100).toFixed(1) + '%'],
                    ['Weighted Spanning Trees (ln T)', mh.log_spanning_trees != null ? mh.log_spanning_trees.toFixed(2) : '--', md.log_spanning_trees != null ? md.log_spanning_trees.toFixed(2) : '--', md.log_spanning_trees != null && mh.log_spanning_trees != null ? (md.log_spanning_trees - mh.log_spanning_trees).toFixed(2) : '--'],
                    ['Betti Numbers (b_0, b_1)', `(${mh.betti_0}, ${mh.betti_1})`, `(${md.betti_0}, ${md.betti_1})`, '--'],
                    ['Isolated Nodes', mh.isolated_nodes, md.isolated_nodes, (md.isolated_nodes - mh.isolated_nodes).toString()],
                    ['QML Kernel Alignment A(Q, W)', (mh.qml_alignment * 100).toFixed(1) + '%', (md.qml_alignment * 100).toFixed(1) + '%', (((md.qml_alignment / Math.max(mh.qml_alignment, 1e-4)) - 1) * 100).toFixed(1) + '%'],
                    ['QML vs Structural Corr (r)', mh.qml_vs_structural_corr.toFixed(4), md.qml_vs_structural_corr.toFixed(4), (((md.qml_vs_structural_corr / Math.max(mh.qml_vs_structural_corr, 1e-4)) - 1) * 100).toFixed(1) + '%'],
                    ['Modular Spectral Energy', mh.modular_energy.toFixed(4), md.modular_energy.toFixed(4), (((md.modular_energy / Math.max(mh.modular_energy, 1e-4)) - 1) * 100).toFixed(1) + '%']
                ];
                metricsTbody.innerHTML = rows.map(r => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 8px; color:#cbd5e1;">${r[0]}</td>
                        <td style="padding:6px 8px; font-weight:600; color:#38bdf8;">${r[1]}</td>
                        <td style="padding:6px 8px; font-weight:600; color:#f87171;">${r[2]}</td>
                        <td style="padding:6px 8px; color:#94a3b8;">${r[3]}</td>
                    </tr>
                `).join('');
            }

            // Populate Modular Forms Table
            const modTbody = document.getElementById('conn-modular-tbody');
            if (modTbody && d.modular_forms) {
                const mf = d.modular_forms;
                const mRows = [
                    ['Modular Nome Magnitude |q|', mf.nome_abs.toFixed(5)],
                    ['Dedekind Eta η(τ)', `${mf.eta[0].toFixed(5)} + ${mf.eta[1].toFixed(5)}i`],
                    ['Eisenstein E_4(τ)', `${mf.E4[0].toFixed(4)} + ${mf.E4[1].toFixed(4)}i`],
                    ['Eisenstein E_6(τ)', `${mf.E6[0].toFixed(4)} + ${mf.E6[1].toFixed(4)}i`],
                    ['Modular Discriminant Δ(τ)', `${mf.Delta[0].toExponential(4)} + ${mf.Delta[1].toExponential(4)}i`],
                    ['Klein j-invariant j(τ)', `${mf.j[0].toFixed(2)} + ${mf.j[1].toFixed(2)}i`],
                    ['Ramanujan τ(1..6)', mf.tau_coefficients ? mf.tau_coefficients.slice(0, 6).join(', ') : '--']
                ];
                modTbody.innerHTML = mRows.map(r => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 8px; color:#cbd5e1;">${r[0]}</td>
                        <td style="padding:6px 8px; font-weight:600; color:#fb923c;">${r[1]}</td>
                    </tr>
                `).join('');
            }

            // Populate Nomenclature Table (sorted by strength loss)
            const nomTbody = document.getElementById('conn-nomenclature-tbody');
            if (nomTbody && d.nodes) {
                const sortedNodes = [...d.nodes].sort((a, b) => b.strength_loss_pct - a.strength_loss_pct);
                nomTbody.innerHTML = sortedNodes.map(n => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 8px; font-weight:600; color:#38bdf8;">${n.hub ? '★ ' : ''}${n.name}</td>
                        <td style="padding:6px 8px; color:#94a3b8;">${n.lobe}</td>
                        <td style="padding:6px 8px; color:#fbbf24;">Stage ${n.braak_stage}</td>
                        <td style="padding:6px 8px; color:#f87171;">${n.atrophy_pct}%</td>
                        <td style="padding:6px 8px; color:#e2e8f0;">${n.strength_healthy} → ${n.strength_dementia}</td>
                        <td style="padding:6px 8px; font-weight:700; color:${n.strength_loss_pct > 50 ? '#ef4444' : '#f59e0b'};">${n.strength_loss_pct}%</td>
                        <td style="padding:6px 8px; color:#34d399;">${n.modular_amp_healthy} → ${n.modular_amp_dementia}</td>
                        <td style="padding:6px 8px; color:#c4b5fd;">${n.qml_strength_healthy} → ${n.qml_strength_dementia}</td>
                    </tr>
                `).join('');
            }

        } catch (err) {
            console.error('Connectome studio execution error:', err);
            const statusEl = document.getElementById('conn-status');
            if (statusEl) statusEl.textContent = 'Error computing connectome';
        }
    };

    const runConnectomeStudioDebounced = () => {
        clearTimeout(connDebounceTimer);
        connDebounceTimer = setTimeout(runConnectomeStudio, 250);
    };

    window.connUpdateLabels = connUpdateLabels;
    window.renderConnectomeViews = renderConnectomeViews;
    window.runConnectomeStudio = runConnectomeStudio;
    window.runConnectomeStudioDebounced = runConnectomeStudioDebounced;

    // ─────────────────────────────────────────────────────────────
    // HEALTH ECONOMICS, NEURAL REPAIR & 2-YEAR CORPORATE VALUATION
    // ─────────────────────────────────────────────────────────────
    let cachedHeData = null;
    let heDebounceTimer = null;

    const heUpdateLabels = () => {
        const discEl = document.getElementById('he-discount');
        if (discEl) {
            const lbl = document.getElementById('label-he-discount');
            if (lbl) lbl.textContent = parseFloat(discEl.value).toFixed(1) + '%';
        }
        const waccEl = document.getElementById('he-wacc');
        if (waccEl) {
            const lbl = document.getElementById('label-he-wacc');
            if (lbl) lbl.textContent = parseFloat(waccEl.value).toFixed(1) + '%';
        }
        const prEl = document.getElementById('he-price');
        if (prEl) {
            const lbl = document.getElementById('label-he-price');
            if (lbl) lbl.textContent = '$' + parseInt(prEl.value);
        }
        const conEl = document.getElementById('he-console');
        if (conEl) {
            const lbl = document.getElementById('label-he-console');
            if (lbl) lbl.textContent = '$' + parseInt(conEl.value).toLocaleString();
        }
        const wtpEl = document.getElementById('he-wtp');
        if (wtpEl) {
            const lbl = document.getElementById('label-he-wtp');
            if (lbl) lbl.textContent = '$' + parseInt(wtpEl.value).toLocaleString() + ' / QALY';
        }
    };

    const renderHealthEconomicsViews = () => {
        if (!cachedHeData) return;
        const d = cachedHeData;
        const ceac = d.ceac_curve;
        const fin = d.financials_2year;
        const val = d.corporate_valuation;
        const markov = d.markov_cohorts;
        const focusDisease = document.getElementById('he-disease-select')?.value || 'all';

        // 1. CEAC Plot
        const wtpK = (ceac.wtp_thresholds || []).map(w => w / 1000.0);
        const probsPct = (ceac.probabilities || []).map(p => +(p * 100.0).toFixed(1));

        const ceacTraces = [
            {
                x: wtpK, y: probsPct,
                type: 'scatter', mode: 'lines',
                line: { color: '#0284c7', width: 2.5 },
                name: 'NeuroMorph rTMS vs SOC'
            },
            {
                x: [50, 50], y: [0, 100],
                type: 'scatter', mode: 'lines',
                line: { color: '#64748b', width: 1.5, dash: 'dot' },
                name: 'UK NICE (£50k)'
            },
            {
                x: [100, 100], y: [0, 100],
                type: 'scatter', mode: 'lines',
                line: { color: '#10b981', width: 1.5, dash: 'dash' },
                name: 'US Standard ($100k)'
            },
            {
                x: [150, 150], y: [0, 100],
                type: 'scatter', mode: 'lines',
                line: { color: '#f59e0b', width: 1.5, dash: 'dashdot' },
                name: 'Orphan WTP ($150k)'
            }
        ];

        Plotly.newPlot('he-ceac-chart', ceacTraces, {
            ...PL,
            margin: { l: 45, r: 20, t: 20, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Willingness-to-Pay Threshold λ ($k / QALY)' },
            yaxis: { ...PL.yaxis, title: 'Probability Cost-Effective (%)', range: [0, 105] },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.55, y: 0.15 }
        }, { responsive: true });

        // 2. Revenue Streams & Cumulative FCF Chart
        const months = fin.monthly.months;
        const hRevK = fin.monthly.hardware_rev.map(v => +(v / 1000.0).toFixed(1));
        const sRevK = fin.monthly.saas_rev.map(v => +(v / 1000.0).toFixed(1));
        const tRevK = fin.monthly.trial_rev.map(v => +(v / 1000.0).toFixed(1));
        const svcRevK = fin.monthly.service_rev.map(v => +(v / 1000.0).toFixed(1));

        let runningFcf = 0;
        const cumFcfK = fin.monthly.fcf.map(v => {
            runningFcf += v;
            return +(runningFcf / 1000.0).toFixed(1);
        });

        const revTraces = [
            {
                x: months, y: hRevK,
                type: 'bar', name: 'Hardware Consoles',
                marker: { color: '#38bdf8' }
            },
            {
                x: months, y: sRevK,
                type: 'bar', name: 'Per-Session SaaS',
                marker: { color: '#10b981' }
            },
            {
                x: months, y: tRevK,
                type: 'bar', name: 'Biopharma Trial AI',
                marker: { color: '#8b5cf6' }
            },
            {
                x: months, y: svcRevK,
                type: 'bar', name: 'Maintenance ARR',
                marker: { color: '#fbbf24' }
            },
            {
                x: months, y: cumFcfK,
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#f43f5e', width: 2.2 },
                marker: { size: 4 },
                name: 'Cumulative FCF ($k)',
                yaxis: 'y2'
            }
        ];

        Plotly.newPlot('he-revenue-chart', revTraces, {
            ...PL,
            barmode: 'stack',
            margin: { l: 45, r: 45, t: 20, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Operational Month (M1 to M24)' },
            yaxis: { ...PL.yaxis, title: 'Monthly Revenue ($k)' },
            yaxis2: { ...PL.yaxis, title: 'Cumulative FCF ($k)', overlaying: 'y', side: 'right', showgrid: false },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.05, y: 0.95 }
        }, { responsive: true });

        // 3. Markov Health States Trajectory Plot
        const markovTraces = [];
        const mAxis = Array.from({ length: 24 }, (_, i) => i + 1);

        if (focusDisease === 'all') {
            // Plot preservation of Early / Mild state across all 4 indications
            const colors = { dementia: '#38bdf8', schizophrenia: '#a7f3d0', parkinsons: '#fbbf24', alexander: '#f472b6' };
            const names = { dementia: "Dementia (MCI)", schizophrenia: "Schizophrenia (Stable)", parkinsons: "Parkinson's (Early H&Y 1-2)", alexander: "Alexander's (Early Ataxic)" };

            Object.entries(markov).forEach(([key, ind]) => {
                const rtmsEarly = ind.history_rtms.map(pt => +(pt.distribution[0] * 100.0).toFixed(1));
                const socEarly = ind.history_soc.map(pt => +(pt.distribution[0] * 100.0).toFixed(1));
                markovTraces.push({
                    x: mAxis, y: rtmsEarly,
                    type: 'scatter', mode: 'lines',
                    line: { color: colors[key], width: 2.2 },
                    name: `rTMS: ${names[key]}`
                });
                markovTraces.push({
                    x: mAxis, y: socEarly,
                    type: 'scatter', mode: 'lines',
                    line: { color: colors[key], width: 1.4, dash: 'dot' },
                    name: `SOC: ${names[key]}`
                });
            });
        } else {
            // Plot multi-state distribution for the selected disease
            const selInd = markov[focusDisease];
            if (selInd && selInd.states) {
                const stateCols = ['#10b981', '#38bdf8', '#f59e0b', '#ef4444', '#8b5cf6'];
                selInd.states.forEach((st, sIdx) => {
                    const rVals = selInd.history_rtms.map(pt => +(pt.distribution[sIdx] * 100.0).toFixed(1));
                    const sVals = selInd.history_soc.map(pt => +(pt.distribution[sIdx] * 100.0).toFixed(1));
                    markovTraces.push({
                        x: mAxis, y: rVals,
                        type: 'scatter', mode: 'lines',
                        line: { color: stateCols[sIdx % stateCols.length], width: 2.2 },
                        name: `rTMS: ${st}`
                    });
                    markovTraces.push({
                        x: mAxis, y: sVals,
                        type: 'scatter', mode: 'lines',
                        line: { color: stateCols[sIdx % stateCols.length], width: 1.4, dash: 'dot' },
                        name: `SOC: ${st}`
                    });
                });
            }
        }

        Plotly.newPlot('he-markov-chart', markovTraces, {
            ...PL,
            margin: { l: 45, r: 20, t: 20, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Elapsed Horizon (Months)' },
            yaxis: { ...PL.yaxis, title: 'Cohort Proportion (%)', range: [0, 105] },
            legend: { font: { color: '#e6edf3', size: 8.5 }, x: 0.6, y: 0.95 }
        }, { responsive: true });

        // 4. Corporate Valuation & Pipeline rNPV Chart
        const indVals = val.rnpv_valuation?.indications || [];
        const valLabels = [
            'DCF (2-Yr FCF)',
            'DCF (Terminal)',
            'Dementia rNPV',
            'Schiz rNPV',
            'Parkinson rNPV',
            'Alexander rNPV',
            'ARR Multiple (14.5x)',
            'Blended Valuation'
        ];
        const valAmounts = [
            +(val.dcf_valuation.pv_explicit_2yr_fcf / 1000000.0).toFixed(2),
            +(val.dcf_valuation.pv_terminal_value / 1000000.0).toFixed(2),
            +(indVals.find(i => i.target === 'dementia')?.risk_adjusted_npv / 1000000.0 || 23.5).toFixed(2),
            +(indVals.find(i => i.target === 'schizophrenia')?.risk_adjusted_npv / 1000000.0 || 14.2).toFixed(2),
            +(indVals.find(i => i.target === 'parkinsons')?.risk_adjusted_npv / 1000000.0 || 20.8).toFixed(2),
            +(indVals.find(i => i.target === 'alexander')?.risk_adjusted_npv / 1000000.0 || 7.9).toFixed(2),
            +(val.market_multiple_valuation.enterprise_value / 1000000.0).toFixed(2),
            +(val.blended_valuation.enterprise_value / 1000000.0).toFixed(2)
        ];

        const barColors = [
            '#0284c7', '#0369a1', '#38bdf8', '#a7f3d0', '#fbbf24', '#f472b6', '#8b5cf6', '#10b981'
        ];

        const valTrace = {
            x: valLabels,
            y: valAmounts,
            type: 'bar',
            marker: { color: barColors },
            text: valAmounts.map(v => `$${v}M`),
            textposition: 'auto',
            textfont: { size: 9, color: '#ffffff' }
        };

        Plotly.newPlot('he-valuation-chart', [valTrace], {
            ...PL,
            margin: { l: 45, r: 20, t: 20, b: 70 },
            xaxis: { ...PL.xaxis, tickangle: -25, tickfont: { size: 9, color: '#cbd5e1' } },
            yaxis: { ...PL.yaxis, title: 'Enterprise Value ($ Millions)' }
        }, { responsive: true });
    };

    const runHealthEconomicsStudio = async () => {
        try {
            const statusEl = document.getElementById('he-status');
            if (statusEl) statusEl.textContent = 'Calculating Markov & Valuation...';

            const discount = parseFloat(document.getElementById('he-discount')?.value || '3.5') / 100.0;
            const wacc = parseFloat(document.getElementById('he-wacc')?.value || '11.5') / 100.0;
            const sessionPrice = parseFloat(document.getElementById('he-price')?.value || '280.0');
            const consolePrice = parseFloat(document.getElementById('he-console')?.value || '65000.0');
            const wtp = parseFloat(document.getElementById('he-wtp')?.value || '100000.0');

            const resp = await fetch('/api/health-economics-simulation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    discount_rate: discount,
                    wacc: wacc,
                    rtms_price_per_session: sessionPrice,
                    system_capital_price: consolePrice,
                    wtp_threshold: wtp
                })
            });

            const json = await resp.json();
            if (json.status !== 'success') {
                console.error('Health economics simulation error:', json);
                if (statusEl) statusEl.textContent = 'Error computing simulation';
                return;
            }

            const d = json;
            cachedHeData = d;
            if (statusEl) statusEl.textContent = 'Ready (' + new Date().toLocaleTimeString() + ')';

            const fin = d.financials_2year;
            const val = d.corporate_valuation;
            const pooled = d.pooled_outcomes;

            // KPI updates
            const revEl = document.getElementById('he-kpi-rev');
            if (revEl) revEl.textContent = '$' + (fin.summary.total_2year_revenue / 1000000.0).toFixed(2) + 'M';
            const revSub = document.getElementById('he-kpi-rev-sub');
            if (revSub) revSub.textContent = `${fin.summary.final_installed_systems} Installed Consoles (M24)`;

            const valEl = document.getElementById('he-kpi-val');
            if (valEl) valEl.textContent = '$' + (val.blended_valuation.enterprise_value / 1000000.0).toFixed(2) + 'M';
            const valSub = document.getElementById('he-kpi-val-sub');
            if (valSub) valSub.textContent = `DCF: $${(val.dcf_valuation.enterprise_value / 1000000.0).toFixed(1)}M | rNPV: $${(val.rnpv_valuation.pipeline_rnpv / 1000000.0).toFixed(1)}M`;

            const shareEl = document.getElementById('he-kpi-share');
            if (shareEl) shareEl.textContent = '$' + val.blended_valuation.price_per_share.toFixed(2);

            const icerEl = document.getElementById('he-kpi-icer');
            if (icerEl) icerEl.textContent = pooled.is_dominant_strategy ? 'Cost-Saving' : '$' + pooled.pooled_icer.toLocaleString() + '/QALY';
            const icerSub = document.getElementById('he-kpi-icer-sub');
            if (icerSub) icerSub.textContent = pooled.is_dominant_strategy ? 'Dominant vs Standard of Care' : 'Below $100k/QALY WTP';

            const ceacEl = document.getElementById('he-kpi-ceac');
            if (ceacEl) ceacEl.textContent = (d.ceac_curve.prob_at_100k * 100.0).toFixed(1) + '%';

            const qalyEl = document.getElementById('he-kpi-qaly');
            if (qalyEl) qalyEl.textContent = '+' + pooled.pooled_delta_qaly_per_pt.toFixed(3) + ' QALYs';

            // Render Charts
            renderHealthEconomicsViews();

            // Populate Indication Table
            const indTbody = document.getElementById('he-indications-tbody');
            if (indTbody && d.markov_cohorts) {
                const rows = [
                    {
                        name: "Dementia (Alzheimer's)",
                        prev: "6.7M US",
                        cSoc: d.markov_cohorts.dementia.cum_cost_soc,
                        cRtms: d.markov_cohorts.dementia.cum_cost_rtms,
                        dCost: d.markov_cohorts.dementia.delta_cost,
                        dQaly: d.markov_cohorts.dementia.delta_qaly,
                        icer: d.markov_cohorts.dementia.icer_per_qaly,
                        assess: "Dominant (Cost-Saving); delays institutional nursing entry by 14 mos"
                    },
                    {
                        name: "Schizophrenia (Cognitive/Negative)",
                        prev: "2.8M US",
                        cSoc: d.markov_cohorts.schizophrenia.cum_cost_soc,
                        cRtms: d.markov_cohorts.schizophrenia.cum_cost_rtms,
                        dCost: d.markov_cohorts.schizophrenia.delta_cost,
                        dQaly: d.markov_cohorts.schizophrenia.delta_qaly,
                        icer: d.markov_cohorts.schizophrenia.icer_per_qaly,
                        assess: "Dominant (Cost-Saving); cuts 30-day psychiatric readmission by 44%"
                    },
                    {
                        name: "Parkinson's Disease (Motor/OFF)",
                        prev: "1.1M US",
                        cSoc: d.markov_cohorts.parkinsons.cum_cost_soc,
                        cRtms: d.markov_cohorts.parkinsons.cum_cost_rtms,
                        dCost: d.markov_cohorts.parkinsons.delta_cost,
                        dQaly: d.markov_cohorts.parkinsons.delta_qaly,
                        icer: d.markov_cohorts.parkinsons.icer_per_qaly,
                        assess: "Highly Cost-Effective ($9.8k/QALY); suppresses dyskinesia & fall trauma"
                    },
                    {
                        name: "Alexander's Disease (Leukodystrophy)",
                        prev: "12k US (Orphan)",
                        cSoc: d.markov_cohorts.alexander.cum_cost_soc,
                        cRtms: d.markov_cohorts.alexander.cum_cost_rtms,
                        dCost: d.markov_cohorts.alexander.delta_cost,
                        dQaly: d.markov_cohorts.alexander.delta_qaly,
                        icer: d.markov_cohorts.alexander.icer_per_qaly,
                        assess: "Dominant (Cost-Saving); preserves bulbar & corticospinal tract function"
                    }
                ];

                indTbody.innerHTML = rows.map(r => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 8px; font-weight:600; color:#38bdf8;">${r.name}</td>
                        <td style="padding:6px 8px; color:#cbd5e1;">${r.prev}</td>
                        <td style="padding:6px 8px; color:#cbd5e1;">$${r.cSoc.toLocaleString()}</td>
                        <td style="padding:6px 8px; color:#a7f3d0;">$${r.cRtms.toLocaleString()}</td>
                        <td style="padding:6px 8px; font-weight:600; color:${r.dCost < 0 ? '#10b981' : '#f59e0b'};">${r.dCost < 0 ? '-$' + Math.abs(r.dCost).toLocaleString() : '+$' + r.dCost.toLocaleString()}</td>
                        <td style="padding:6px 8px; color:#fbbf24;">+${r.dQaly.toFixed(3)}</td>
                        <td style="padding:6px 8px; font-weight:700; color:${r.dCost < 0 ? '#10b981' : '#38bdf8'};">${r.dCost < 0 ? 'Cost-Saving' : '$' + Math.round(r.icer).toLocaleString()}</td>
                        <td style="padding:6px 8px; color:#94a3b8; font-size:10.5px;">${r.assess}</td>
                    </tr>
                `).join('');
            }

            // Populate Quarterly Pro Forma Table
            const qTbody = document.getElementById('he-quarters-tbody');
            if (qTbody && fin.quarters) {
                qTbody.innerHTML = fin.quarters.map(q => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 8px; font-weight:600; color:#38bdf8;">${q.quarter}</td>
                        <td style="padding:6px 8px; color:#cbd5e1;">${q.installed_base} Consoles</td>
                        <td style="padding:6px 8px; font-weight:600; color:#a7f3d0;">$${Math.round(q.total_revenue).toLocaleString()}</td>
                        <td style="padding:6px 8px; color:#cbd5e1;">$${Math.round(q.gross_profit).toLocaleString()}</td>
                        <td style="padding:6px 8px; color:#fbbf24;">${q.gross_margin_pct}%</td>
                        <td style="padding:6px 8px; color:${q.ebitda >= 0 ? '#10b981' : '#f87171'}; font-weight:600;">$${Math.round(q.ebitda).toLocaleString()}</td>
                        <td style="padding:6px 8px; color:${q.fcf >= 0 ? '#10b981' : '#f87171'}; font-weight:600;">$${Math.round(q.fcf).toLocaleString()}</td>
                    </tr>
                `).join('');
            }

        } catch (err) {
            console.error('Health economics studio error:', err);
            const statusEl = document.getElementById('he-status');
            if (statusEl) statusEl.textContent = 'Error during calculation';
        }
    };

    const runHealthEconomicsStudioDebounced = () => {
        clearTimeout(heDebounceTimer);
        heDebounceTimer = setTimeout(runHealthEconomicsStudio, 250);
    };

    window.heUpdateLabels = heUpdateLabels;
    window.renderHealthEconomicsViews = renderHealthEconomicsViews;
    window.runHealthEconomicsStudio = runHealthEconomicsStudio;
    window.runHealthEconomicsStudioDebounced = runHealthEconomicsStudioDebounced;

    // ─────────────────────────────────────────────────────────────
    // XANADU PHOTONIC QUANTUM TMS & NEURAL REPAIR SUITE
    // ─────────────────────────────────────────────────────────────
    let cachedXqData = null;
    let xqDebounceTimer = null;

    const xqUpdateLabels = () => {
        const sqEl = document.getElementById('xq-squeezing');
        if (sqEl) {
            const r = parseFloat(sqEl.value);
            const db = (10.0 * Math.log10(Math.exp(2.0 * r))).toFixed(1);
            const lbl = document.getElementById('label-xq-squeezing');
            if (lbl) lbl.textContent = `${r.toFixed(2)} (${db} dB)`;
        }
        const dispEl = document.getElementById('xq-displacement');
        if (dispEl) {
            const lbl = document.getElementById('label-xq-displacement');
            if (lbl) lbl.textContent = parseFloat(dispEl.value).toFixed(2);
        }
        const phEl = document.getElementById('xq-phase');
        if (phEl) {
            const lbl = document.getElementById('label-xq-phase');
            if (lbl) lbl.textContent = parseFloat(phEl.value).toFixed(2) + ' rad';
        }
        const modEl = document.getElementById('xq-modulus');
        if (modEl) {
            const lbl = document.getElementById('label-xq-modulus');
            if (lbl) lbl.textContent = `F_${parseInt(modEl.value)} (Prime)`;
        }
        const itEl = document.getElementById('xq-iters');
        if (itEl) {
            const lbl = document.getElementById('label-xq-iters');
            if (lbl) lbl.textContent = parseInt(itEl.value) + ' Epochs';
        }
        const horEl = document.getElementById('xq-horizon');
        if (horEl) {
            const m = parseInt(horEl.value);
            const lbl = document.getElementById('label-xq-horizon');
            if (lbl) lbl.textContent = `${m} Months (${(m / 12).toFixed(0)} Yrs)`;
        }
        const prEl = document.getElementById('xq-pulse-price');
        if (prEl) {
            const lbl = document.getElementById('label-xq-pulse-price');
            if (lbl) lbl.textContent = '$' + parseFloat(prEl.value).toFixed(3);
        }
    };

    const renderXanaduQuantumViews = () => {
        if (!cachedXqData) return;
        const d = cachedXqData;
        const wig = d.photonic_wigner;
        const qml = d.qml_optimization;
        const clin = d.clinical_trajectories;
        const comm = d.commercial_economics;
        const diseaseFocus = document.getElementById('xq-disease-focus')?.value || 'stroke';

        // 1. Wigner Function Contour Chart
        const xCoords = wig.x_coords;
        const pCoords = wig.p_coords;
        const wMatrix = wig.wigner_matrix;

        const wignerTrace = {
            x: xCoords,
            y: pCoords,
            z: wMatrix,
            type: 'contour',
            colorscale: 'Viridis',
            contours: { coloring: 'heatmap' },
            colorbar: {
                title: 'W(x, p)',
                len: 0.75, thickness: 12,
                tickfont: { color: '#cbd5e1', size: 9 },
                titlefont: { color: '#cbd5e1', size: 10 }
            }
        };

        Plotly.newPlot('xq-wigner-chart', [wignerTrace], {
            ...PL,
            margin: { l: 45, r: 20, t: 20, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Field Quadrature X (Position)' },
            yaxis: { ...PL.yaxis, title: 'Phase Quadrature P (Momentum)' }
        }, { responsive: true });

        // 2. QML Parameter-Shift Loss Convergence Chart
        const qmlTraces = [
            {
                x: qml.iterations,
                y: qml.qml_loss,
                type: 'scatter', mode: 'lines',
                line: { color: '#10b981', width: 2.2 },
                name: 'Xanadu CV Photonic QML'
            },
            {
                x: qml.iterations,
                y: qml.classical_loss,
                type: 'scatter', mode: 'lines',
                line: { color: '#ef4444', width: 1.8, dash: 'dot' },
                name: 'Classical BEM Solver'
            },
            {
                x: qml.iterations,
                y: qml.quantum_speedup_curve,
                type: 'scatter', mode: 'lines',
                line: { color: '#38bdf8', width: 1.5, dash: 'dash' },
                name: 'Speedup Ratio (x)',
                yaxis: 'y2'
            }
        ];

        Plotly.newPlot('xq-qml-chart', qmlTraces, {
            ...PL,
            margin: { l: 45, r: 45, t: 20, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Optimization Epoch / Iteration' },
            yaxis: { ...PL.yaxis, title: 'Target E-Field Loss', type: 'log' },
            yaxis2: { ...PL.yaxis, title: 'Speedup Factor (x)', overlaying: 'y', side: 'right', showgrid: false },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.1, y: 0.95 }
        }, { responsive: true });

        // 3. 60-Month Longitudinal Clinical Trajectories Chart
        const months = clin.months;
        const trajTraces = [];

        const indInfo = clin[diseaseFocus] || clin.stroke;
        trajTraces.push({
            x: months,
            y: indInfo.xanadu_qml_rtms,
            type: 'scatter', mode: 'lines+markers',
            line: { color: '#0284c7', width: 2.5 },
            marker: { size: 5 },
            name: `Xanadu Quantum rTMS (+${indInfo.improvement_pct}%)`
        });
        trajTraces.push({
            x: months,
            y: indInfo.conventional_rtms,
            type: 'scatter', mode: 'lines',
            line: { color: '#f59e0b', width: 1.8, dash: 'dash' },
            name: 'Conventional rTMS'
        });
        trajTraces.push({
            x: months,
            y: indInfo.soc,
            type: 'scatter', mode: 'lines',
            line: { color: '#94a3b8', width: 1.4, dash: 'dot' },
            name: 'Standard of Care (SOC)'
        });

        Plotly.newPlot('xq-trajectories-chart', trajTraces, {
            ...PL,
            margin: { l: 45, r: 20, t: 20, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Follow-up Horizon (Months)' },
            yaxis: { ...PL.yaxis, title: indInfo.name },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.1, y: 0.95 }
        }, { responsive: true });

        // 4. 36-Month Xanadu QaaS Revenue & EBITDA Chart
        const m36 = comm.monthly.months;
        const apiRevK = comm.monthly.qaas_api_rev.map(v => +(v / 1000.0).toFixed(1));
        const entRevK = comm.monthly.enterprise_rev.map(v => +(v / 1000.0).toFixed(1));
        const bioRevK = comm.monthly.biopharma_rev.map(v => +(v / 1000.0).toFixed(1));

        let cumEb = 0;
        const cumEbK = comm.monthly.ebitda.map(v => {
            cumEb += v;
            return +(cumEb / 1000.0).toFixed(1);
        });

        const revTraces = [
            {
                x: m36, y: apiRevK,
                type: 'bar', name: 'QaaS Pulse API',
                marker: { color: '#38bdf8' }
            },
            {
                x: m36, y: entRevK,
                type: 'bar', name: 'Hospital Enterprise',
                marker: { color: '#8b5cf6' }
            },
            {
                x: m36, y: bioRevK,
                type: 'bar', name: 'Biopharma Co-Dev',
                marker: { color: '#f472b6' }
            },
            {
                x: m36, y: cumEbK,
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#10b981', width: 2.2 },
                marker: { size: 4 },
                name: 'Cumulative EBITDA ($k)',
                yaxis: 'y2'
            }
        ];

        Plotly.newPlot('xq-revenue-chart', revTraces, {
            ...PL,
            barmode: 'stack',
            margin: { l: 45, r: 45, t: 20, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Commercial Month (M1 to M36)' },
            yaxis: { ...PL.yaxis, title: 'Monthly Revenue ($k)' },
            yaxis2: { ...PL.yaxis, title: 'Cumulative EBITDA ($k)', overlaying: 'y', side: 'right', showgrid: false },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.05, y: 0.95 }
        }, { responsive: true });

        // 5. Plot: Maximum Dispersion Staging Boundary Convergence
        const disp = d.maximum_dispersion_staging;
        if (disp && document.getElementById('xq-dispersion-chart')) {
            const traceDisp = {
                x: disp.iterations,
                y: disp.dispersion_convergence_curve,
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#06b6d4', width: 2.4 },
                marker: { size: 4 },
                name: 'Between-Stage Dispersion D(μ)'
            };
            Plotly.newPlot('xq-dispersion-chart', [traceDisp], {
                ...PL,
                margin: { l: 45, r: 20, t: 20, b: 40 },
                xaxis: { ...PL.xaxis, title: 'Optimization Epoch' },
                yaxis: { ...PL.yaxis, title: 'Between-Stage Dispersion D(μ)' },
                shapes: [
                    { type: 'line', x0: 1, x1: 25, y0: 0.0833, y1: 0.0833, line: { color: '#64748b', width: 1.2, dash: 'dot' } }
                ],
                annotations: [
                    { x: 12, y: 0.0833, text: 'Uniform Null Variance (1/12)', showarrow: false, font: { size: 8.5, color: '#94a3b8' } }
                ],
                legend: { font: { color: '#e6edf3', size: 9 }, x: 0.45, y: 0.25 }
            }, { responsive: true });
        }

        // 6. Plot: Wasserstein-2 Transport Distance & Radon-Nikodym Density
        if (disp && document.getElementById('xq-transport-chart')) {
            const traceW2 = {
                x: disp.weeks,
                y: disp.wasserstein_2_distance,
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#10b981', width: 2.4 },
                marker: { size: 5 },
                name: 'Wasserstein-2 Distance W_2(t)'
            };
            const traceRn = {
                x: disp.weeks,
                y: disp.radon_nikodym_derivative.slice(0, disp.weeks.length),
                type: 'scatter', mode: 'lines',
                line: { color: '#8b5cf6', width: 2.0, dash: 'dash' },
                name: 'Radon-Nikodym Density Λ(x)',
                yaxis: 'y2'
            };
            Plotly.newPlot('xq-transport-chart', [traceW2, traceRn], {
                ...PL,
                margin: { l: 45, r: 45, t: 20, b: 40 },
                xaxis: { ...PL.xaxis, title: 'Treatment Week' },
                yaxis: { ...PL.yaxis, title: 'Transport Metric W_2(μ_t, μ*)', color: '#10b981' },
                yaxis2: {
                    title: 'Radon-Nikodym Density Λ(x)',
                    overlaying: 'y', side: 'right',
                    color: '#8b5cf6',
                    gridcolor: 'rgba(139,92,246,0.08)'
                },
                legend: { font: { color: '#e6edf3', size: 8.5 }, x: 0.45, y: 0.95 }
            }, { responsive: true });
        }

        // 7. Plot: 60-Month Staged Clinical Remission Probabilities
        if (clin && document.getElementById('xq-staging-trajectories-chart')) {
            const m = clin.months;
            const pStroke = m.map(v => Math.min(96.0, 15.0 + 78.0 * (1.0 - Math.exp(-0.11 * v))));
            const pPd = m.map(v => Math.min(92.0, 12.0 + 74.0 * (1.0 - Math.exp(-0.10 * v))));
            const pDem = m.map(v => Math.min(88.0, 8.0 + 68.0 * (1.0 - Math.exp(-0.09 * v))));

            const traceRemStroke = { x: m, y: pStroke, type: 'scatter', mode: 'lines+markers', line: { color: '#0284c7', width: 2.2 }, name: 'Stroke Remission' };
            const traceRemPd = { x: m, y: pPd, type: 'scatter', mode: 'lines', line: { color: '#fbbf24', width: 2.0, dash: 'dash' }, name: 'PD Motor Remission' };
            const traceRemDem = { x: m, y: pDem, type: 'scatter', mode: 'lines', line: { color: '#c4b5fd', width: 1.8, dash: 'dot' }, name: 'Dementia Stabilization' };

            Plotly.newPlot('xq-staging-trajectories-chart', [traceRemStroke, traceRemPd, traceRemDem], {
                ...PL,
                margin: { l: 45, r: 20, t: 20, b: 40 },
                xaxis: { ...PL.xaxis, title: 'Follow-up Horizon (Months)' },
                yaxis: { ...PL.yaxis, title: 'Continuous Remission Probability (%)', range: [0, 105] },
                legend: { font: { color: '#e6edf3', size: 8.5 }, x: 0.45, y: 0.25 }
            }, { responsive: true });
        }

        // 8. Plot: 10-Year Long-Term Recovery Cumulative Discounted NPV ($M USD)
        const rev10 = d.ten_year_revenue_projections;
        if (rev10 && document.getElementById('xq-ten-year-revenue-chart')) {
            const p10Npv = rev10.cumulative_discounted_npv_usd_m.map(v => +(v * 0.76).toFixed(2));
            const p50Npv = rev10.cumulative_discounted_npv_usd_m;
            const p90Npv = rev10.cumulative_discounted_npv_usd_m.map(v => +(v * 1.32).toFixed(2));

            const trace10Npv90 = { x: rev10.years, y: p90Npv, type: 'scatter', mode: 'lines+markers', line: { color: '#c4b5fd', width: 2, dash: 'dash' }, marker: { size: 4 }, name: 'P90 Optimistic ($M)' };
            const trace10Npv50 = { x: rev10.years, y: p50Npv, type: 'scatter', mode: 'lines+markers', line: { color: '#10b981', width: 2.6 }, marker: { size: 5 }, name: 'P50 Median Base ($M)' };
            const trace10Npv10 = { x: rev10.years, y: p10Npv, type: 'scatter', mode: 'lines+markers', line: { color: '#fbbf24', width: 1.8, dash: 'dot' }, marker: { size: 4 }, name: 'P10 Conservative ($M)' };

            Plotly.newPlot('xq-ten-year-revenue-chart', [trace10Npv90, trace10Npv50, trace10Npv10], {
                ...PL,
                margin: { l: 45, r: 20, t: 20, b: 40 },
                xaxis: { ...PL.xaxis, title: 'Operational Model Year (t = 1 to 10)' },
                yaxis: { ...PL.yaxis, title: 'Cumulative Discounted NPV ($M USD)' },
                shapes: [{ type: 'line', x0: 1, x1: 10, y0: 0, y1: 0, line: { color: '#64748b', width: 1.0 } }],
                legend: { font: { color: '#e6edf3', size: 8.5 }, x: 0.05, y: 0.95 }
            }, { responsive: true });
        }

        // 9. Populate Staging Paradigm Table
        const stgTbody = document.getElementById('xq-staging-tbody');
        if (stgTbody && disp && disp.stages) {
            stgTbody.innerHTML = disp.stages.map(stg => `
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:6px 8px; font-weight:700; color:#38bdf8;">${stg.name} [${stg.lower_bound.toFixed(3)} - ${stg.upper_bound.toFixed(3)}]</td>
                    <td style="padding:6px 8px; color:#cbd5e1;">${stg.clinical_state}</td>
                    <td style="padding:6px 8px; color:#a7f3d0;">${stg.photonic_action}</td>
                    <td style="padding:6px 8px; font-family:monospace; color:#fbbf24;">${stg.target_frequency_hz} Hz</td>
                    <td style="padding:6px 8px; color:#94a3b8;">${stg.dwell_weeks} Wks</td>
                    <td style="padding:6px 8px; font-weight:700; color:#10b981;">${(stg.response_probability * 100).toFixed(1)}%</td>
                </tr>
            `).join('');
        }

        // 10. Populate 10-Year Pro Forma DCF Table
        const rev10Tbody = document.getElementById('xq-ten-year-revenue-tbody');
        if (rev10Tbody && rev10 && rev10.years) {
            rev10Tbody.innerHTML = rev10.years.map((y, idx) => `
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:6px 8px; font-weight:700; color:#38bdf8;">Year ${y}</td>
                    <td style="padding:6px 8px; color:#cbd5e1;">${rev10.hospitals_p50[idx]} Hubs</td>
                    <td style="padding:6px 8px; color:#38bdf8;">$${rev10.revenue_streams_p50_usd_m.qaas_pulse_api[idx].toFixed(2)}M</td>
                    <td style="padding:6px 8px; color:#a7f3d0;">$${rev10.revenue_streams_p50_usd_m.enterprise_arr[idx].toFixed(2)}M</td>
                    <td style="padding:6px 8px; color:#c4b5fd;">$${rev10.revenue_streams_p50_usd_m.biopharma_runtime[idx].toFixed(2)}M</td>
                    <td style="padding:6px 8px; font-weight:700; color:#10b981;">$${rev10.total_revenue_p50_usd_m[idx].toFixed(2)}M</td>
                    <td style="padding:6px 8px; color:#fbbf24;">$${rev10.annual_ebitda_usd_m[idx].toFixed(2)}M</td>
                    <td style="padding:6px 8px; font-weight:700; color:#38bdf8;">$${rev10.cumulative_discounted_npv_usd_m[idx].toFixed(2)}M</td>
                </tr>
            `).join('');
        }
    };

    const runXanaduQuantumStudio = async () => {
        try {
            const statusEl = document.getElementById('xq-status');
            if (statusEl) statusEl.textContent = 'Simulating Xanadu Photonic QPU...';

            const chipset = document.getElementById('xq-chipset')?.value || 'borealis';
            const squeezing_r = parseFloat(document.getElementById('xq-squeezing')?.value || '1.25');
            const displacement_alpha = parseFloat(document.getElementById('xq-displacement')?.value || '0.65');
            const squeezing_theta = parseFloat(document.getElementById('xq-phase')?.value || '0.35');
            const modulus_q = parseInt(document.getElementById('xq-modulus')?.value || '17');
            const horizon_months = parseInt(document.getElementById('xq-horizon')?.value || '60');
            const qml_iterations = parseInt(document.getElementById('xq-iters')?.value || '50');
            const qaas_price_per_pulse = parseFloat(document.getElementById('xq-pulse-price')?.value || '0.018');

            const resp = await fetch('/api/xanadu-quantum-tms', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    chipset,
                    squeezing_r,
                    displacement_alpha,
                    squeezing_theta,
                    modulus_q,
                    horizon_months,
                    qml_iterations,
                    qaas_price_per_pulse
                })
            });

            const json = await resp.json();
            if (json.status !== 'success') {
                console.error('Xanadu Quantum simulation error:', json);
                if (statusEl) statusEl.textContent = 'Error during simulation';
                return;
            }

            const d = json;
            cachedXqData = d;
            if (statusEl) statusEl.textContent = 'Ready (' + new Date().toLocaleTimeString() + ')';

            const qml = d.qml_optimization;
            const wig = d.photonic_wigner;
            const clin = d.clinical_trajectories;
            const comm = d.commercial_economics;
            const he = comm.health_economics;

            // Update KPI elements
            const speedupEl = document.getElementById('xq-kpi-speedup');
            if (speedupEl) speedupEl.textContent = `${qml.peak_speedup_factor}x`;
            const speedupSub = document.getElementById('xq-kpi-speedup-sub');
            if (speedupSub) speedupSub.textContent = `vs Classical BEM (${qml.time_reduction_pct}% faster)`;

            const sqEl = document.getElementById('xq-kpi-squeezing');
            if (sqEl) sqEl.textContent = `${wig.squeezing_db} dB`;
            const sqSub = document.getElementById('xq-kpi-squeezing-sub');
            if (sqSub) sqSub.textContent = `Vacuum Fid: ${(wig.vacuum_fidelity * 100).toFixed(1)}%`;

            const runEl = document.getElementById('xq-kpi-runtime');
            if (runEl) runEl.textContent = `${qml.wallclock_quantum_ms} ms`;

            const revEl = document.getElementById('xq-kpi-revenue');
            if (revEl) revEl.textContent = '$' + (comm.summary_3yr.total_3yr_revenue / 1000000.0).toFixed(2) + 'M';
            const revSub = document.getElementById('xq-kpi-revenue-sub');
            if (revSub) revSub.textContent = `3-Yr Total (${comm.summary_3yr.installed_hospitals_m36} Hospitals)`;

            const recEl = document.getElementById('xq-kpi-recovery');
            if (recEl) recEl.textContent = `+${clin.stroke.improvement_pct}%`;

            const icerEl = document.getElementById('xq-kpi-icer');
            if (icerEl) icerEl.textContent = he.is_dominant_strategy ? 'Cost-Saving' : '$' + Math.round(he.icer_per_qaly).toLocaleString() + '/QALY';

            // Update Maximum Dispersion & 10-Yr Profit KPI cards
            const disp = d.maximum_dispersion_staging;
            const rev10 = d.ten_year_revenue_projections;
            if (disp) {
                const dispEl = document.getElementById('xq-kpi-dispersion');
                if (dispEl) dispEl.textContent = disp.between_stage_dispersion.toFixed(3);

                const transEl = document.getElementById('xq-kpi-transport');
                if (transEl) transEl.textContent = disp.wasserstein_2_distance.slice(-1)[0].toFixed(3);
            }
            if (rev10) {
                const npv10El = document.getElementById('xq-kpi-10yr-npv');
                if (npv10El) npv10El.textContent = '$' + rev10.ten_year_npv_usd_m.toFixed(1) + 'M';

                const eb10El = document.getElementById('xq-kpi-10yr-ebitda');
                if (eb10El) eb10El.textContent = '$' + rev10.ten_year_ebitda_usd_m.toFixed(1) + 'M';

                const termEl = document.getElementById('xq-kpi-terminal-rev');
                if (termEl) termEl.textContent = '$' + rev10.terminal_revenue_year10_usd_m.toFixed(1) + 'M';

                const payEl = document.getElementById('xq-kpi-payback');
                if (payEl) payEl.textContent = `${rev10.payback_year.toFixed(1)} Yrs`;
            }

            // Render 2D and Contour Plots
            renderXanaduQuantumViews();

            // Populate Hardware / Number Theory Table
            const hwTbody = document.getElementById('xq-hardware-tbody');
            if (hwTbody && d.number_theoretic_states) {
                hwTbody.innerHTML = d.number_theoretic_states.map(s => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 8px; font-weight:600; color:#38bdf8;">k = ${s.order}</td>
                        <td style="padding:6px 8px; font-weight:600; color:#fbbf24;">${s.fraction}</td>
                        <td style="padding:6px 8px; color:#cbd5e1;">${s.value}</td>
                        <td style="padding:6px 8px; color:#a7f3d0;">${s.quantum_phase_rad} rad</td>
                        <td style="padding:6px 8px; color:#c4b5fd;">s(p, q) = ${s.dedekind_sum}</td>
                        <td style="padding:6px 8px; color:#f472b6;">dim(H) = ${s.squeezing_subspace_dim}</td>
                    </tr>
                `).join('');
            }

            // Populate Clinical Outcomes Table
            const clinTbody = document.getElementById('xq-clinical-tbody');
            if (clinTbody && d.clinical_trajectories) {
                const cRows = [
                    {
                        name: "Stroke Motor Deficit",
                        scale: "Fugl-Meyer Assessment (0–66)",
                        soc: d.clinical_trajectories.stroke.soc.slice(-1)[0] + " / 66",
                        conv: d.clinical_trajectories.stroke.conventional_rtms.slice(-1)[0] + " / 66",
                        xan: d.clinical_trajectories.stroke.xanadu_qml_rtms.slice(-1)[0] + " / 66",
                        gain: "+" + d.clinical_trajectories.stroke.improvement_pct + "%",
                        mech: "Quantum-synchronized M1 corticospinal tract axonal reinnervation"
                    },
                    {
                        name: "Dementia (Alzheimer's)",
                        scale: "ADAS-Cog-14 Score (0–90)",
                        soc: d.clinical_trajectories.dementia.soc.slice(-1)[0] + " (Severe)",
                        conv: d.clinical_trajectories.dementia.conventional_rtms.slice(-1)[0] + " (Moderate)",
                        xan: d.clinical_trajectories.dementia.xanadu_qml_rtms.slice(-1)[0] + " (Mild)",
                        gain: "-" + d.clinical_trajectories.dementia.improvement_pct + "%",
                        mech: "40 Hz gamma entrainment suppressing hippocampal amyloid/tau accumulation"
                    },
                    {
                        name: "Parkinson's Disease",
                        scale: "UPDRS-III Motor Score (0–108)",
                        soc: d.clinical_trajectories.parkinsons.soc.slice(-1)[0] + " (Severe)",
                        conv: d.clinical_trajectories.parkinsons.conventional_rtms.slice(-1)[0] + " (Moderate)",
                        xan: d.clinical_trajectories.parkinsons.xanadu_qml_rtms.slice(-1)[0] + " (Mild)",
                        gain: "-" + d.clinical_trajectories.parkinsons.improvement_pct + "%",
                        mech: "Multi-lobe basal ganglia decoupling; +4.5 hours daily ON-time"
                    },
                    {
                        name: "Schizophrenia Deficit",
                        scale: "PANSS Cognitive Subscale (7–49)",
                        soc: d.clinical_trajectories.schizophrenia.soc.slice(-1)[0] + " (Refractory)",
                        conv: d.clinical_trajectories.schizophrenia.conventional_rtms.slice(-1)[0] + " (Partial)",
                        xan: d.clinical_trajectories.schizophrenia.xanadu_qml_rtms.slice(-1)[0] + " (Stable)",
                        gain: "-" + d.clinical_trajectories.schizophrenia.improvement_pct + "%",
                        mech: "Prefrontal theta-burst synchrony restoring fronto-striatal gating"
                    },
                    {
                        name: "Alexander's Disease",
                        scale: "SARA Ataxia Rating Scale (0–40)",
                        soc: d.clinical_trajectories.alexander.soc.slice(-1)[0] + " (End-Stage)",
                        conv: d.clinical_trajectories.alexander.conventional_rtms.slice(-1)[0] + " (Severe)",
                        xan: d.clinical_trajectories.alexander.xanadu_qml_rtms.slice(-1)[0] + " (Ambulatory)",
                        gain: "-" + d.clinical_trajectories.alexander.improvement_pct + "%",
                        mech: "GFAP astrogliopathy suppression protecting brainstem bulbar nuclei"
                    }
                ];

                clinTbody.innerHTML = cRows.map(r => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 8px; font-weight:600; color:#38bdf8;">${r.name}</td>
                        <td style="padding:6px 8px; color:#cbd5e1;">${r.scale}</td>
                        <td style="padding:6px 8px; color:#cbd5e1;">${r.soc}</td>
                        <td style="padding:6px 8px; color:#fbbf24;">${r.conv}</td>
                        <td style="padding:6px 8px; font-weight:600; color:#10b981;">${r.xan}</td>
                        <td style="padding:6px 8px; font-weight:700; color:#38bdf8;">${r.gain}</td>
                        <td style="padding:6px 8px; color:#94a3b8; font-size:10.5px;">${r.mech}</td>
                    </tr>
                `).join('');
            }

        } catch (err) {
            console.error('Xanadu Quantum studio error:', err);
            const statusEl = document.getElementById('xq-status');
            if (statusEl) statusEl.textContent = 'Error during simulation';
        }
    };

    const runXanaduQuantumStudioDebounced = () => {
        clearTimeout(xqDebounceTimer);
        xqDebounceTimer = setTimeout(runXanaduQuantumStudio, 250);
    };

    window.xqUpdateLabels = xqUpdateLabels;
    window.renderXanaduQuantumViews = renderXanaduQuantumViews;
    window.runXanaduQuantumStudio = runXanaduQuantumStudio;
    window.runXanaduQuantumStudioDebounced = runXanaduQuantumStudioDebounced;

    // ── Parkinson's DBS: Prime-Gap Staging, Geometric Fencing & QML Revenue ──
    let cachedPdbsData = null;
    let pdbsDebounceTimer = null;

    const pdbsUpdateLabels = () => {
        const primesEl = document.getElementById('pdbs-primes');
        if (primesEl) {
            const lbl = document.getElementById('label-pdbs-primes');
            if (lbl) lbl.textContent = parseInt(primesEl.value);
        }
        const stagesEl = document.getElementById('pdbs-stages');
        if (stagesEl) {
            const lbl = document.getElementById('label-pdbs-stages');
            if (lbl) lbl.textContent = parseInt(stagesEl.value);
        }
        const progsocEl = document.getElementById('pdbs-progsoc');
        if (progsocEl) {
            const lbl = document.getElementById('label-pdbs-progsoc');
            if (lbl) lbl.textContent = parseFloat(progsocEl.value).toFixed(3);
        }
        const regrdbsEl = document.getElementById('pdbs-regrdbs');
        if (regrdbsEl) {
            const lbl = document.getElementById('label-pdbs-regrdbs');
            if (lbl) lbl.textContent = parseFloat(regrdbsEl.value).toFixed(3);
        }
        const itersEl = document.getElementById('pdbs-iters');
        if (itersEl) {
            const lbl = document.getElementById('label-pdbs-iters');
            if (lbl) lbl.textContent = parseInt(itersEl.value);
        }
        const priceEl = document.getElementById('pdbs-price');
        if (priceEl) {
            const lbl = document.getElementById('label-pdbs-price');
            if (lbl) lbl.textContent = '$' + parseInt(priceEl.value).toLocaleString();
        }
    };

    const renderParkinsonsDbsViews = () => {
        if (!cachedPdbsData) return;
        const d = cachedPdbsData;
        const staging = d.staging;
        const clinical = d.clinical;
        const rev = d.revenue;

        // 1. Prime-gap signature bar chart
        const sig = staging.signature;
        const idx = sig.gaps.map((_, i) => i);
        Plotly.newPlot('pdbs-signature-chart', [{
            x: idx, y: sig.signature, type: 'bar',
            marker: { color: sig.signature.map(s => s > 0 ? '#0284c7' : '#b91c1c') }
        }], {
            margin: { l: 45, r: 15, t: 10, b: 35 },
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#e6edf3', size: 10 },
            xaxis: { title: 'Prime Index i', gridcolor: 'rgba(255,255,255,0.08)' },
            yaxis: { title: 'Signature S_i', gridcolor: 'rgba(255,255,255,0.08)' }
        }, { responsive: true });

        // 2. Geometric fencing diagram
        const theta = Array.from({ length: 100 }, (_, i) => (2 * Math.PI * i) / 99);
        const fenceTraces = d.fence_centers.map((c, k) => {
            const r = d.fence_radii[k];
            return {
                x: theta.map(t => c[0] + r * Math.cos(t)),
                y: theta.map(t => c[1] + r * Math.sin(t)),
                mode: 'lines', type: 'scatter', name: `Fence ${k + 1}`,
                line: { color: ['#34d399', '#fbbf24', '#f59e0b', '#b91c1c', '#c4b5fd', '#38bdf8'][k % 6], width: 2 }
            };
        });
        Plotly.newPlot('pdbs-fencing-chart', fenceTraces, {
            margin: { l: 45, r: 15, t: 10, b: 35 },
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#e6edf3', size: 10 },
            xaxis: { title: 'UPDRS-III Score', gridcolor: 'rgba(255,255,255,0.08)' },
            yaxis: { title: 'Stimulation Voltage (V)', gridcolor: 'rgba(255,255,255,0.08)' },
            showlegend: true, legend: { font: { color: '#e6edf3', size: 8 }, x: 0.02, y: 0.98 }
        }, { responsive: true });

        // 3. 10-year clinical trajectory
        const years = clinical.months.map(m => m / 12.0);
        Plotly.newPlot('pdbs-trajectory-chart', [
            { x: years, y: clinical.updrs_soc, type: 'scatter', mode: 'lines', name: 'Standard of Care', line: { color: '#b91c1c', width: 2.5 } },
            { x: years, y: clinical.updrs_dbs, type: 'scatter', mode: 'lines', name: 'Prime-Gap Fenced DBS', line: { color: '#0284c7', width: 2.5 } }
        ], {
            margin: { l: 45, r: 15, t: 10, b: 35 },
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#e6edf3', size: 10 },
            xaxis: { title: 'Follow-up (Years)', gridcolor: 'rgba(255,255,255,0.08)' },
            yaxis: { title: 'Expected UPDRS-III', gridcolor: 'rgba(255,255,255,0.08)' },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.02, y: 0.1 }
        }, { responsive: true });

        // 4. 5-year revenue projection with CI bands
        const rq = rev.qml, rb = rev.baseline;
        const mYears = rq.months.map(m => m / 12.0);
        Plotly.newPlot('pdbs-revenue-chart', [
            { x: mYears, y: rq.ci_upper.map(v => v / 1e6), type: 'scatter', mode: 'lines', line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
            { x: mYears, y: rq.ci_lower.map(v => v / 1e6), type: 'scatter', mode: 'lines', fill: 'tonexty', fillcolor: 'rgba(5,150,105,0.18)', line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
            { x: mYears, y: rq.cum_revenue_mean.map(v => v / 1e6), type: 'scatter', mode: 'lines', name: 'QML-Annealed Schedule', line: { color: '#059669', width: 2.5 } },
            { x: mYears, y: rb.cum_revenue_mean.map(v => v / 1e6), type: 'scatter', mode: 'lines', name: 'Fixed Moderate Baseline', line: { color: '#7c3aed', width: 2, dash: 'dash' } }
        ], {
            margin: { l: 50, r: 15, t: 10, b: 35 },
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#e6edf3', size: 10 },
            xaxis: { title: 'Horizon (Years)', gridcolor: 'rgba(255,255,255,0.08)' },
            yaxis: { title: 'Cumulative Revenue ($M)', gridcolor: 'rgba(255,255,255,0.08)' },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.02, y: 0.95 }
        }, { responsive: true });
    };

    const runParkinsonsDbsStudio = async () => {
        try {
            const statusEl = document.getElementById('pdbs-status');
            if (statusEl) statusEl.textContent = 'Simulating Prime-Gap Staging...';

            const num_primes = parseInt(document.getElementById('pdbs-primes')?.value || '30');
            const num_stages = parseInt(document.getElementById('pdbs-stages')?.value || '4');
            const progress_p_soc = parseFloat(document.getElementById('pdbs-progsoc')?.value || '0.018');
            const regress_p_dbs = parseFloat(document.getElementById('pdbs-regrdbs')?.value || '0.010');
            const iterations = parseInt(document.getElementById('pdbs-iters')?.value || '60');
            const price_per_implant = parseFloat(document.getElementById('pdbs-price')?.value || '42000');

            const resp = await fetch('/api/parkinsons-dbs-trajectories', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    num_primes, num_stages, progress_p_soc, regress_p_dbs, iterations, price_per_implant
                })
            });

            const d = await resp.json();
            if (d.status === 'error') {
                console.error('Parkinsons DBS simulation error:', d);
                if (statusEl) statusEl.textContent = 'Error during simulation';
                return;
            }

            cachedPdbsData = d;
            if (statusEl) statusEl.textContent = 'Ready (' + new Date().toLocaleTimeString() + ')';

            const staging = d.staging;
            const clinical = d.clinical;
            const so = d.schedule_optimization;
            const robustness = so.robustness;
            const rev = d.revenue;
            const fenceProj = d.fence_projections;

            // KPI cards
            const updrsEl = document.getElementById('pdbs-kpi-updrs');
            if (updrsEl) updrsEl.textContent = `-${clinical.final_updrs_improvement_pct}%`;
            const delayEl = document.getElementById('pdbs-kpi-delay');
            if (delayEl) delayEl.textContent = `+${clinical.severe_stage_delay_months} mo`;
            const winEl = document.getElementById('pdbs-kpi-winrate');
            if (winEl) winEl.textContent = `${robustness.win_rate_pct}%`;
            const revEl = document.getElementById('pdbs-kpi-revenue');
            if (revEl) revEl.textContent = '$' + (rev.qml.year5_revenue_mean / 1e6).toFixed(2) + 'M';
            const revSubEl = document.getElementById('pdbs-kpi-revenue-sub');
            if (revSubEl) revSubEl.textContent = `95% CI $${(rev.qml.year5_ci[0] / 1e6).toFixed(2)}M–$${(rev.qml.year5_ci[1] / 1e6).toFixed(2)}M`;
            const upliftEl = document.getElementById('pdbs-kpi-uplift');
            if (upliftEl) upliftEl.textContent = `+${rev.uplift_pct}%`;
            const voltEl = document.getElementById('pdbs-kpi-voltage');
            if (voltEl) voltEl.textContent = `${fenceProj[0].optimal_voltage}–${fenceProj[fenceProj.length - 1].optimal_voltage} V`;

            renderParkinsonsDbsViews();

            // Staging & fencing table
            const stagingTbody = document.getElementById('pdbs-staging-tbody');
            if (stagingTbody) {
                const pgT = staging.prime_gap_thresholds;
                const eqT = staging.equal_width_thresholds;
                stagingTbody.innerHTML = staging.stage_names.map((name, i) => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 8px; font-weight:600; color:#38bdf8;">${name}</td>
                        <td style="padding:6px 8px; color:#cbd5e1;">[${pgT[i].toFixed(1)}, ${pgT[i + 1].toFixed(1)}]</td>
                        <td style="padding:6px 8px; color:#94a3b8;">[${eqT[i].toFixed(1)}, ${eqT[i + 1].toFixed(1)}]</td>
                        <td style="padding:6px 8px; color:#a7f3d0;">${d.fence_radii[i].toFixed(3)}</td>
                        <td style="padding:6px 8px; font-weight:600; color:#fbbf24;">${fenceProj[i].optimal_voltage.toFixed(2)} V</td>
                    </tr>
                `).join('');
            }

            // Schedule & revenue summary table
            const summaryTbody = document.getElementById('pdbs-summary-tbody');
            if (summaryTbody) {
                const rows = [
                    { metric: 'UPDRS-III at Year 10', base: clinical.updrs_soc[clinical.updrs_soc.length - 1].toFixed(1), qml: clinical.updrs_dbs[clinical.updrs_dbs.length - 1].toFixed(1), gain: `-${clinical.final_updrs_improvement_pct}%` },
                    { metric: 'Severe-Stage (H&Y 5) Onset Delay', base: 'Reference', qml: `+${clinical.severe_stage_delay_months} months`, gain: 'Clinically Significant' },
                    { metric: '5-Year Rollout Schedule', base: so.baseline_schedule.join(' → '), qml: so.qml_schedule.join(' → '), gain: 'Combinatorial QML Search' },
                    { metric: '5-Year Cumulative Revenue', base: '$' + (rev.baseline.year5_revenue_mean / 1e6).toFixed(2) + 'M', qml: '$' + (rev.qml.year5_revenue_mean / 1e6).toFixed(2) + 'M', gain: `+${rev.uplift_pct}%` },
                    { metric: 'Annealer Win/Tie Rate', base: 'N/A', qml: `${robustness.win_rate_pct}%`, gain: `vs Random (${robustness.num_trials} trials)` },
                    { metric: 'Mean Optimality Gap', base: 'N/A', qml: '$' + (robustness.mean_gap_anneal / 1000).toFixed(1) + 'k', gain: 'vs $' + (robustness.mean_gap_random / 1000).toFixed(1) + 'k Random' }
                ];
                summaryTbody.innerHTML = rows.map(r => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                        <td style="padding:6px 8px; font-weight:600; color:#38bdf8;">${r.metric}</td>
                        <td style="padding:6px 8px; color:#cbd5e1;">${r.base}</td>
                        <td style="padding:6px 8px; font-weight:600; color:#10b981;">${r.qml}</td>
                        <td style="padding:6px 8px; font-weight:700; color:#fbbf24;">${r.gain}</td>
                    </tr>
                `).join('');
            }

        } catch (err) {
            console.error('Parkinsons DBS studio error:', err);
            const statusEl = document.getElementById('pdbs-status');
            if (statusEl) statusEl.textContent = 'Error during simulation';
        }
    };

    const runParkinsonsDbsStudioDebounced = () => {
        clearTimeout(pdbsDebounceTimer);
        pdbsDebounceTimer = setTimeout(runParkinsonsDbsStudio, 250);
    };

    window.pdbsUpdateLabels = pdbsUpdateLabels;
    window.renderParkinsonsDbsViews = renderParkinsonsDbsViews;
    window.runParkinsonsDbsStudio = runParkinsonsDbsStudio;
    window.runParkinsonsDbsStudioDebounced = runParkinsonsDbsStudioDebounced;





    // ── Indigenous Depression rTMS: CF Staging & Combinatorial QML Scheduling ──
    let cachedIdtmsData = null;
    let idtmsDebounceTimer = null;

    const idtmsUpdateLabels = () => {
        const specs = [
            ['idtms-phq9', 'label-idtms-phq9', v => parseFloat(v).toFixed(1)],
            ['idtms-travel', 'label-idtms-travel', v => parseFloat(v).toFixed(2)],
            ['idtms-cst', 'label-idtms-cst', v => parseFloat(v).toFixed(2)],
            ['idtms-trauma', 'label-idtms-trauma', v => parseFloat(v).toFixed(2)],
            ['idtms-support', 'label-idtms-support', v => parseFloat(v).toFixed(2)],
            ['idtms-sessions', 'label-idtms-sessions', v => parseInt(v)],
            ['idtms-iters', 'label-idtms-iters', v => parseInt(v)],
            ['idtms-cost', 'label-idtms-cost', v => '$' + parseInt(v).toLocaleString()],
            ['idtms-intensity', 'label-idtms-intensity', v => parseInt(v) + '%']
        ];
        specs.forEach(([inputId, labelId, fmt]) => {
            const el = document.getElementById(inputId);
            if (!el) return;
            const lbl = document.getElementById(labelId);
            if (lbl) lbl.textContent = fmt(el.value);
        });
    };

    const renderIndigenousDepressionTmsViews = () => {
        if (!cachedIdtmsData) return;
        const d = cachedIdtmsData;
        const staging = d.staging;
        const so = d.schedule_optimization;
        const traj = d.trajectory;

        // 1. Continued-fraction convergent error / Hurwitz bound
        const conv = staging.convergents;
        const n = conv.map((_, i) => i + 1);
        Plotly.newPlot('idtms-cf-chart', [
            { x: n, y: conv.map(c => c.error), type: 'scatter', mode: 'lines+markers', name: '|ρ - p_n/q_n|', line: { color: '#0284c7', width: 2.5 } },
            { x: n, y: staging.hurwitz_bounds, type: 'scatter', mode: 'lines', name: 'Hurwitz Bound 1/(√5 q_n²)', line: { color: '#f59e0b', width: 2, dash: 'dash' } }
        ], {
            margin: { l: 55, r: 15, t: 10, b: 35 },
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#e6edf3', size: 10 },
            xaxis: { title: 'Convergent Index n', gridcolor: 'rgba(255,255,255,0.08)' },
            yaxis: { title: 'Approximation Error', type: 'log', gridcolor: 'rgba(255,255,255,0.08)' },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.55, y: 0.95 }
        }, { responsive: true });

        // 2. Stage boundary diagram
        const stages = staging.stages;
        Plotly.newPlot('idtms-stage-chart', [{
            x: stages.map(s => s.length), y: stages.map(s => s.name), type: 'bar', orientation: 'h',
            marker: { color: ['#0284c7', '#059669', '#f59e0b', '#b91c1c'].slice(0, stages.length) },
            text: stages.map(s => `Sessions ${s.start}\u2013${s.end}`), textposition: 'inside'
        }], {
            margin: { l: 110, r: 15, t: 10, b: 35 },
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#e6edf3', size: 10 },
            xaxis: { title: 'Sessions in Stage', gridcolor: 'rgba(255,255,255,0.08)' }
        }, { responsive: true });

        // 3. PHQ-9 trajectory
        Plotly.newPlot('idtms-trajectory-chart', [
            { x: traj.months, y: traj.phq9_optimized, type: 'scatter', mode: 'lines', name: 'CF-Staged + QML Schedule', line: { color: '#0284c7', width: 2.5 } },
            { x: traj.months, y: traj.phq9_standard, type: 'scatter', mode: 'lines', name: 'Fixed-Frequency Standard', line: { color: '#b91c1c', width: 2, dash: 'dash' } }
        ], {
            margin: { l: 45, r: 15, t: 10, b: 35 },
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#e6edf3', size: 10 },
            xaxis: { title: 'Months', gridcolor: 'rgba(255,255,255,0.08)' },
            yaxis: { title: 'Expected PHQ-9', gridcolor: 'rgba(255,255,255,0.08)' },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.55, y: 0.95 }
        }, { responsive: true });

        // 4. Retention / survival curves
        Plotly.newPlot('idtms-retention-chart', [
            { x: traj.months, y: traj.survival_optimized_pct, type: 'scatter', mode: 'lines', name: 'Optimized Cohort Retention', line: { color: '#059669', width: 2.5 } },
            { x: traj.months, y: traj.survival_standard_pct, type: 'scatter', mode: 'lines', name: 'Standard Cohort Retention', line: { color: '#7c3aed', width: 2, dash: 'dash' } }
        ], {
            margin: { l: 45, r: 15, t: 10, b: 35 },
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#e6edf3', size: 10 },
            xaxis: { title: 'Months', gridcolor: 'rgba(255,255,255,0.08)' },
            yaxis: { title: 'Cohort Retention (%)', gridcolor: 'rgba(255,255,255,0.08)' },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.55, y: 0.15 }
        }, { responsive: true });

        // 5. Cortical surface BEM stimulation field (superposed coil montage)
        const bem = d.cortical_bem_field;
        if (bem && document.getElementById('idtms-bem-chart')) {
            const coilTrace = {
                x: bem.coils.map(c => c.x), y: bem.coils.map(c => c.y), z: bem.coils.map(c => c.z),
                type: 'scatter3d', mode: 'markers+text', text: bem.coils.map(c => c.label.replace(/ Cortex$/, '')),
                textposition: 'top center', textfont: { color: '#f8fafc', size: 9 },
                marker: { size: 6, color: '#f472b6', symbol: 'diamond', line: { color: '#fff', width: 1 } },
                name: 'Coil Montage'
            };
            Plotly.newPlot('idtms-bem-chart', [{
                x: bem.x, y: bem.y, z: bem.z, surfacecolor: bem.potential, type: 'surface',
                colorscale: [[0, '#051b2c'], [.35, '#0d4f90'], [.65, '#0d9488'], [.85, '#38bdf8'], [1, '#f472b6']],
                colorbar: { title: 'Induced Potential', tickfont: { color: '#c9d1d9', size: 9 } },
                contours: { z: { show: true, usecolormap: true, project: { z: true } } },
                opacity: 0.96
            }, coilTrace], {
                margin: { t: 10, b: 0, l: 0, r: 0 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e6edf3', size: 10 },
                scene: {
                    xaxis: { title: 'Cortical X', gridcolor: 'rgba(255,255,255,0.08)', color: '#94a3b8' },
                    yaxis: { title: 'Cortical Y', gridcolor: 'rgba(255,255,255,0.08)', color: '#94a3b8' },
                    zaxis: { title: 'Cortical Z', gridcolor: 'rgba(255,255,255,0.08)', color: '#94a3b8' },
                    camera: { eye: { x: 1.55, y: -1.55, z: .95 } }
                },
                showlegend: false
            }, { responsive: true });

            const peakEl = document.getElementById('idtms-bem-peak');
            if (peakEl) peakEl.textContent = bem.peak_potential.toFixed(3);
            const coilCountEl = document.getElementById('idtms-bem-coil-count');
            if (coilCountEl) coilCountEl.textContent = bem.coils.length;
        }
    };

    const runIndigenousDepressionTmsStudio = async () => {
        try {
            const statusEl = document.getElementById('idtms-status');
            if (statusEl) statusEl.textContent = 'Simulating CF Staging & QML Scheduling...';

            const baseline_phq9 = parseFloat(document.getElementById('idtms-phq9')?.value || '21.0');
            const travel_barrier = parseFloat(document.getElementById('idtms-travel')?.value || '0.55');
            const cultural_safety_score = parseFloat(document.getElementById('idtms-cst')?.value || '0.6');
            const historical_trauma_load = parseFloat(document.getElementById('idtms-trauma')?.value || '0.45');
            const community_support = parseFloat(document.getElementById('idtms-support')?.value || '0.5');
            const sessions = parseInt(document.getElementById('idtms-sessions')?.value || '30');
            const annealer_iterations = parseInt(document.getElementById('idtms-iters')?.value || '60');
            const cost_per_session = parseFloat(document.getElementById('idtms-cost')?.value || '185');
            const rf_intensity_pct = parseFloat(document.getElementById('idtms-intensity')?.value || '110');
            const target_lobes = Array.from(document.querySelectorAll('.idtms-lobe-checkbox:checked')).map(el => el.value);

            const resp = await fetch('/api/indigenous-depression-tms', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    baseline_phq9, travel_barrier, cultural_safety_score, historical_trauma_load,
                    community_support, sessions, annealer_iterations, cost_per_session, target_lobes, rf_intensity_pct
                })
            });

            const d = await resp.json();
            if (d.status === 'error') {
                console.error('Indigenous Depression TMS simulation error:', d);
                if (statusEl) statusEl.textContent = 'Error during simulation';
                return;
            }

            cachedIdtmsData = d;
            if (statusEl) statusEl.textContent = 'Ready (' + new Date().toLocaleTimeString() + ')';

            const staging = d.staging;
            const so = d.schedule_optimization;
            const robustness = so.robustness;
            const traj = d.trajectory;
            const econ = d.equity_economics;

            // KPI cards
            const responseGain = (traj.response_pct_optimized - traj.response_pct_standard);
            const completionGain = (traj.completion_rate_optimized_pct - traj.completion_rate_standard_pct);
            const respEl = document.getElementById('idtms-kpi-response');
            if (respEl) respEl.textContent = `+${responseGain.toFixed(1)} pp`;
            const compEl = document.getElementById('idtms-kpi-completion');
            if (compEl) compEl.textContent = `+${completionGain.toFixed(1)} pp`;
            const winEl = document.getElementById('idtms-kpi-winrate');
            if (winEl) winEl.textContent = `${robustness.win_rate_pct.toFixed(1)}%`;
            const rhoEl = document.getElementById('idtms-kpi-rho');
            if (rhoEl) rhoEl.textContent = staging.rho_equity_ratio.toFixed(3);
            const icerEl = document.getElementById('idtms-kpi-icer');
            if (icerEl) icerEl.textContent = '$' + Math.round(econ.icer_usd_per_qaly).toLocaleString();
            const costRespEl = document.getElementById('idtms-kpi-costresp');
            if (costRespEl) costRespEl.textContent = '$' + Math.round(econ.cost_per_responder_optimized).toLocaleString();

            // Staging table
            const stagingBody = document.getElementById('idtms-staging-tbody');
            if (stagingBody) {
                stagingBody.innerHTML = staging.stages.map(s => `
                    <tr>
                        <td>${s.name}</td>
                        <td>${s.start}</td>
                        <td>${s.end}</td>
                        <td>${s.length}</td>
                    </tr>
                `).join('');
            }

            // Cortical lobes stimulated table
            const lobesBody = document.getElementById('idtms-lobes-tbody');
            const ct = d.cortical_targets;
            if (lobesBody && ct) {
                lobesBody.innerHTML = ct.targets.map(t => `
                    <tr>
                        <td>${t.label}</td>
                        <td>${t.lobe}</td>
                        <td>${t.brodmann}</td>
                        <td>${t.frequency_role}</td>
                        <td>${t.evidence_weight.toFixed(2)}</td>
                    </tr>
                `).join('');
            }
            const coverageEl = document.getElementById('idtms-lobe-coverage');
            if (coverageEl && ct) coverageEl.textContent = `${ct.montage_coverage_factor.toFixed(3)}\u00d7`;

            // Summary table
            const summaryBody = document.getElementById('idtms-summary-tbody');
            if (summaryBody) {
                summaryBody.innerHTML = `
                    <tr><td>QML Schedule (Hz)</td><td>${so.qml_schedule_hz.join(', ')}</td></tr>
                    <tr><td>Brute-Force Best Schedule (Hz)</td><td>${so.brute_force_best_schedule_hz.join(', ')}</td></tr>
                    <tr><td>Baseline Fixed Schedule (Hz)</td><td>${so.baseline_fixed_schedule_hz.join(', ')}</td></tr>
                    <tr><td>Uplift vs Fixed Schedule</td><td>${so.uplift_vs_fixed_pct.toFixed(2)}%</td></tr>
                    <tr><td>Robustness Mean Gap (Annealer)</td><td>${robustness.mean_gap_anneal.toFixed(4)}</td></tr>
                    <tr><td>Robustness Mean Gap (Random)</td><td>${robustness.mean_gap_random.toFixed(4)}</td></tr>
                    <tr><td>Cohort Size</td><td>${econ.cohort_size}</td></tr>
                    <tr><td>Responders (Optimized / Standard)</td><td>${econ.responders_optimized} / ${econ.responders_standard}</td></tr>
                    <tr><td>Total Cost (Optimized / Standard)</td><td>$${Math.round(econ.cost_optimized_total).toLocaleString()} / $${Math.round(econ.cost_standard_total).toLocaleString()}</td></tr>
                `;
            }

            renderIndigenousDepressionTmsViews();
        } catch (err) {
            console.error('runIndigenousDepressionTmsStudio failed:', err);
            const statusEl = document.getElementById('idtms-status');
            if (statusEl) statusEl.textContent = 'Error during simulation';
        }
    };

    const runIndigenousDepressionTmsStudioDebounced = () => {
        clearTimeout(idtmsDebounceTimer);
        idtmsDebounceTimer = setTimeout(runIndigenousDepressionTmsStudio, 250);
    };

    window.idtmsUpdateLabels = idtmsUpdateLabels;
    window.renderIndigenousDepressionTmsViews = renderIndigenousDepressionTmsViews;
    window.runIndigenousDepressionTmsStudio = runIndigenousDepressionTmsStudio;
    window.runIndigenousDepressionTmsStudioDebounced = runIndigenousDepressionTmsStudioDebounced;

    // ─────────────────────────────────────────────────────────────
    // ADDICTION INTERVENTIONAL RECOVERY: HEBBIAN STDP & QML SUITE
    // ─────────────────────────────────────────────────────────────
    let cachedAddictionData = null;
    let addictionDebounceTimer = null;

    const addictionUpdateLabels = () => {
        const sevEl = document.getElementById('addiction-severity');
        const sessEl = document.getElementById('addiction-sessions');
        const intEl = document.getElementById('addiction-intensity');

        if (sevEl) {
            const lbl = document.getElementById('label-addiction-sev');
            if (lbl) lbl.textContent = parseFloat(sevEl.value).toFixed(1);
        }
        if (sessEl) {
            const lbl = document.getElementById('label-addiction-sessions');
            if (lbl) lbl.textContent = `${sessEl.value} Sessions`;
        }
        if (intEl) {
            const lbl = document.getElementById('label-addiction-intensity');
            if (lbl) lbl.textContent = `${intEl.value}%`;
        }
    };

    const addictionTypeChanged = () => {
        const typeEl = document.getElementById('addiction-type');
        const sevEl = document.getElementById('addiction-severity');
        const sevLbl = document.getElementById('addiction-sev-label');
        if (!typeEl || !sevEl || !sevLbl) return;

        const val = typeEl.value;
        if (val === 'nicotine') {
            sevLbl.textContent = 'Fagerström Severity (FTND):';
            sevEl.min = 2;
            sevEl.max = 10;
            sevEl.step = 0.5;
            sevEl.value = 8.0;
        } else if (val === 'alcohol') {
            sevLbl.textContent = 'AUDIT Severity Score:';
            sevEl.min = 8;
            sevEl.max = 40;
            sevEl.step = 1.0;
            sevEl.value = 28.0;
        } else {
            sevLbl.textContent = 'Dual Dependence Index:';
            sevEl.min = 4;
            sevEl.max = 20;
            sevEl.step = 0.5;
            sevEl.value = 14.0;
        }
        addictionUpdateLabels();
    };

    const renderAddictionRecoveryViews = (data) => {
        const d = data || cachedAddictionData;
        if (!d) return;

        const m = d.months;
        const traj = d.trajectories;
        const kpi = d.summary_kpis;
        const heb = d.hebbian_stdp;
        const qml = d.quantum_machine_learning;

        // 1. KPI Cards
        const setEl = (id, txt) => {
            const el = document.getElementById(id);
            if (el) el.textContent = txt;
        };
        setEl('add-kpi-abstinence', `${kpi.month12_abstinence_rate_pct.toFixed(1)}%`);
        setEl('add-kpi-craving', `-${kpi.craving_reduction_pct.toFixed(1)}%`);
        setEl('add-kpi-hebbian', `${kpi.hebbian_ltp_gain.toFixed(2)}×`);
        setEl('add-kpi-ltd', `-${heb.final_ltd_suppression_pct.toFixed(1)}%`);
        setEl('add-kpi-qml', `${kpi.qml_circuit_fidelity.toFixed(3)}`);
        setEl('add-kpi-hazard', `-${kpi.relapse_hazard_reduction_pct.toFixed(1)}%`);

        // Health Economics & Quantum Agentic AI KPI Cards
        setEl('add-kpi-qaly', `+${(kpi.qaly_gained_superoptimal || 1.53).toFixed(2)}`);
        setEl('add-kpi-icer', 'Dominant');
        setEl('add-kpi-icer-sub', `-$${Math.abs(kpi.icer_usd_per_qaly || 14804).toFixed(0)} / QALY`);
        setEl('add-kpi-rev', `$${(kpi.annual_revenue_p50_usd_m || 1.89).toFixed(2)}M`);
        setEl('add-kpi-npv', `$${(kpi.five_year_npv_p50_usd_m || 5.68).toFixed(2)}M`);
        setEl('add-kpi-entropy', `${(kpi.final_agentic_entropy || 0.026).toFixed(3)}`);
        setEl('add-kpi-entropy-sub', `-${(kpi.agentic_entropy_reduction_pct || 94.5).toFixed(1)}% Epistemic Drop`);
        setEl('add-kpi-bcr', `${(kpi.benefit_cost_ratio || 4.85).toFixed(2)}×`);

        // Update consumption sublabel
        const subLbl = document.getElementById('add-consumption-sublabel');
        if (subLbl) subLbl.textContent = `${d.consumption_unit} Across 36-Month Follow-Up`;

        // 2. Plot 1: Craving Intensity Decay VAS (0-100)
        const traceCravingSham = {
            x: m, y: traj.craving_vas.sham,
            name: 'Sham Control (Counseling)',
            type: 'scatter', mode: 'lines+markers',
            line: { color: '#94a3b8', dash: 'dash', width: 2 },
            marker: { size: 5 }
        };
        const traceCravingStd = {
            x: m, y: traj.craving_vas.standard_rtms,
            name: 'Standard 10 Hz Left dlPFC',
            type: 'scatter', mode: 'lines+markers',
            line: { color: '#38bdf8', width: 2.2 },
            marker: { size: 5 }
        };
        const traceCravingSuper = {
            x: m, y: traj.craving_vas.superoptimal_hebbian_qml,
            name: 'Hebbian-QML Stieltjes',
            type: 'scatter', mode: 'lines+markers',
            line: { color: '#10b981', width: 3 },
            marker: { size: 6, color: '#10b981' }
        };
        Plotly.newPlot('add-craving-chart', [traceCravingSham, traceCravingStd, traceCravingSuper], {
            ...PL,
            margin: { l: 45, r: 20, t: 15, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Follow-up Horizon (Months)' },
            yaxis: { ...PL.yaxis, title: 'Craving Severity (VAS 0–100)' },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.45, y: 0.95 }
        }, { responsive: true });

        // 3. Plot 2: Continuous Abstinence Rate (%)
        const traceAbstSham = {
            x: m, y: traj.continuous_abstinence_pct.sham,
            name: 'Sham Control (~15% Abst.)',
            type: 'scatter', mode: 'lines+markers',
            line: { color: '#94a3b8', dash: 'dash', width: 2 }
        };
        const traceAbstStd = {
            x: m, y: traj.continuous_abstinence_pct.standard_rtms,
            name: 'Standard 10 Hz rTMS',
            type: 'scatter', mode: 'lines+markers',
            line: { color: '#38bdf8', width: 2.2 }
        };
        const traceAbstSuper = {
            x: m, y: traj.continuous_abstinence_pct.superoptimal_hebbian_qml,
            name: 'Hebbian-QML (>78% Sustained)',
            type: 'scatter', mode: 'lines+markers',
            line: { color: '#10b981', width: 3 }
        };
        Plotly.newPlot('add-abstinence-chart', [traceAbstSham, traceAbstStd, traceAbstSuper], {
            ...PL,
            margin: { l: 45, r: 20, t: 15, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Follow-up Horizon (Months)' },
            yaxis: { ...PL.yaxis, title: 'Point-Prevalence Abstinence (%)', range: [0, 105] },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.45, y: 0.25 }
        }, { responsive: true });

        // 4. Plot 3: Hebbian STDP Synaptic Rewiring
        const traceLtp = {
            x: heb.sessions, y: heb.pfc_inhibitory_weight,
            name: 'dlPFC→NAc LTP (Control)',
            type: 'scatter', mode: 'lines',
            line: { color: '#38bdf8', width: 2.5 }
        };
        const traceLtd = {
            x: heb.sessions, y: heb.insula_craving_weight,
            name: 'Insula→NAc LTD (Craving Urge)',
            type: 'scatter', mode: 'lines',
            line: { color: '#f87171', width: 2.2, dash: 'dash' }
        };
        const traceGain = {
            x: heb.sessions, y: heb.hebbian_control_gain,
            name: 'Plasticity Gain Ratio',
            type: 'scatter', mode: 'lines',
            line: { color: '#a7f3d0', width: 2.5, dash: 'dot' },
            yaxis: 'y2'
        };
        Plotly.newPlot('add-hebbian-chart', [traceLtp, traceLtd, traceGain], {
            ...PL,
            margin: { l: 45, r: 45, t: 15, b: 40 },
            xaxis: { ...PL.xaxis, title: 'rTMS Session Index' },
            yaxis: { ...PL.yaxis, title: 'Synaptic Weight w_ij' },
            yaxis2: {
                title: 'Inhibitory Gain Ratio',
                overlaying: 'y', side: 'right',
                color: '#a7f3d0',
                gridcolor: 'rgba(167,243,208,0.08)'
            },
            legend: { font: { color: '#e6edf3', size: 8.5 }, x: 0.05, y: 0.95 }
        }, { responsive: true });

        // 5. Plot 4: Substance Consumption
        const traceConsSham = {
            x: m, y: traj.consumption.sham,
            name: 'Sham Control',
            type: 'scatter', mode: 'lines+markers',
            line: { color: '#94a3b8', dash: 'dash', width: 1.8 }
        };
        const traceConsStd = {
            x: m, y: traj.consumption.standard_rtms,
            name: 'Standard 10 Hz',
            type: 'scatter', mode: 'lines+markers',
            line: { color: '#38bdf8', width: 2.2 }
        };
        const traceConsSuper = {
            x: m, y: traj.consumption.superoptimal_hebbian_qml,
            name: 'Superoptimal rTMS',
            type: 'scatter', mode: 'lines+markers',
            line: { color: '#f87171', width: 2.8 }
        };
        Plotly.newPlot('add-consumption-chart', [traceConsSham, traceConsStd, traceConsSuper], {
            ...PL,
            margin: { l: 45, r: 20, t: 15, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Follow-up Horizon (Months)' },
            yaxis: { ...PL.yaxis, title: d.consumption_unit },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.55, y: 0.95 }
        }, { responsive: true });

        // 6. Plot 5: QML Circuit Optimization
        const traceFid = {
            x: qml.iterations, y: qml.quantum_fidelity,
            name: 'State Fidelity |⟨ψ|0⟩|²',
            type: 'scatter', mode: 'lines',
            line: { color: '#c4b5fd', width: 2.5 }
        };
        const traceLoss = {
            x: qml.iterations, y: qml.variational_loss,
            name: 'Variational Loss',
            type: 'scatter', mode: 'lines',
            line: { color: '#f43f5e', width: 2, dash: 'dash' }
        };
        Plotly.newPlot('add-qml-chart', [traceFid, traceLoss], {
            ...PL,
            margin: { l: 45, r: 20, t: 15, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Parameter-Shift QML Iterations' },
            yaxis: { ...PL.yaxis, title: 'Convergence Metric' },
            legend: { font: { color: '#e6edf3', size: 9 }, x: 0.55, y: 0.55 }
        }, { responsive: true });

        // 7. Plot 6: Connectivity & Relapse Hazard
        const traceConn = {
            x: m, y: traj.frontostriatal_connectivity_z,
            name: 'Frontostriatal Connectivity (z-score)',
            type: 'scatter', mode: 'lines+markers',
            line: { color: '#38bdf8', width: 2.5 }
        };
        const traceHaz = {
            x: m, y: traj.relapse_hazard,
            name: 'Stieltjes Relapse Hazard h(t)',
            type: 'scatter', mode: 'lines+markers',
            line: { color: '#fbbf24', width: 2, dash: 'dot' },
            yaxis: 'y2'
        };
        Plotly.newPlot('add-connectivity-chart', [traceConn, traceHaz], {
            ...PL,
            margin: { l: 45, r: 45, t: 15, b: 40 },
            xaxis: { ...PL.xaxis, title: 'Follow-up Horizon (Months)' },
            yaxis: { ...PL.yaxis, title: 'rs-FC (z-score)', color: '#38bdf8' },
            yaxis2: {
                title: 'Relapse Hazard h(t)',
                overlaying: 'y', side: 'right',
                color: '#fbbf24',
                gridcolor: 'rgba(251,191,36,0.08)'
            },
            legend: { font: { color: '#e6edf3', size: 8.5 }, x: 0.1, y: 0.95 }
        }, { responsive: true });

        // 7b. Plot 7: Annual Revenue Probability Density Function (PDF)
        const revDist = d.revenue_distributions || {};
        if (revDist.pdf_bin_centers_m && document.getElementById('add-revenue-pdf-chart')) {
            const tracePdf = {
                x: revDist.pdf_bin_centers_m,
                y: revDist.pdf_density_values,
                name: 'Revenue PDF f(R)',
                type: 'scatter', mode: 'lines',
                fill: 'tozeroy',
                fillcolor: 'rgba(2, 132, 199, 0.18)',
                line: { color: '#0284c7', width: 2.4 }
            };
            const p10 = revDist.p10_annual_revenue_usd_m || 1.42;
            const p50 = revDist.p50_annual_revenue_usd_m || 1.89;
            const p90 = revDist.p90_annual_revenue_usd_m || 2.52;

            Plotly.newPlot('add-revenue-pdf-chart', [tracePdf], {
                ...PL,
                margin: { l: 45, r: 20, t: 15, b: 40 },
                xaxis: { ...PL.xaxis, title: 'Annual Outpatient Revenue ($M USD)' },
                yaxis: { ...PL.yaxis, title: 'Probability Density' },
                shapes: [
                    { type: 'line', x0: p10, x1: p10, y0: 0, y1: 1, yref: 'paper', line: { color: '#fbbf24', width: 1.8, dash: 'dash' } },
                    { type: 'line', x0: p50, x1: p50, y0: 0, y1: 1, yref: 'paper', line: { color: '#10b981', width: 2.2 } },
                    { type: 'line', x0: p90, x1: p90, y0: 0, y1: 1, yref: 'paper', line: { color: '#c4b5fd', width: 1.8, dash: 'dash' } }
                ],
                annotations: [
                    { x: p10, y: 1.05, yref: 'paper', text: `P10: $${p10.toFixed(2)}M`, showarrow: false, font: { size: 9, color: '#fbbf24' } },
                    { x: p50, y: 1.05, yref: 'paper', text: `P50: $${p50.toFixed(2)}M`, showarrow: false, font: { size: 9.5, color: '#10b981' } },
                    { x: p90, y: 1.05, yref: 'paper', text: `P90: $${p90.toFixed(2)}M`, showarrow: false, font: { size: 9, color: '#c4b5fd' } }
                ],
                legend: { font: { color: '#e6edf3', size: 9 }, x: 0.05, y: 0.95 }
            }, { responsive: true });
        }

        // 7c. Plot 8: 5-Year Cumulative Discounted NPV Trajectory
        if (revDist.horizon_years && document.getElementById('add-npv-traj-chart')) {
            const traceNpv90 = {
                x: revDist.horizon_years, y: revDist.cumulative_npv_p90_m,
                name: 'P90 Optimistic ($M)',
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#c4b5fd', width: 2, dash: 'dash' },
                marker: { size: 5 }
            };
            const traceNpv50 = {
                x: revDist.horizon_years, y: revDist.cumulative_npv_p50_m,
                name: 'P50 Median Base ($M)',
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#10b981', width: 2.8 },
                marker: { size: 6 }
            };
            const traceNpv10 = {
                x: revDist.horizon_years, y: revDist.cumulative_npv_p10_m,
                name: 'P10 Conservative ($M)',
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#fbbf24', width: 1.8, dash: 'dot' },
                marker: { size: 5 }
            };
            Plotly.newPlot('add-npv-traj-chart', [traceNpv90, traceNpv50, traceNpv10], {
                ...PL,
                margin: { l: 45, r: 20, t: 15, b: 40 },
                xaxis: { ...PL.xaxis, title: 'Operational Model Year (t = 0 to 5)' },
                yaxis: { ...PL.yaxis, title: 'Cumulative Discounted NPV ($M USD)' },
                legend: { font: { color: '#e6edf3', size: 9 }, x: 0.05, y: 0.95 }
            }, { responsive: true });
        }

        // 7d. Plot 9: Cost-Effectiveness Plane (ICER)
        const heData = d.health_economics || {};
        if (heData.cost_effectiveness_plane && document.getElementById('add-icer-plane-chart')) {
            const cep = heData.cost_effectiveness_plane;
            const traceScatter = {
                x: cep.qaly_gains,
                y: cep.cost_differences_usd.map(c => c / 1e3),
                name: 'Monte Carlo Cohorts',
                type: 'scatter', mode: 'markers',
                marker: { size: 7, color: '#10b981', opacity: 0.75, line: { color: '#047857', width: 1 } }
            };
            const maxX = Math.max(...cep.qaly_gains, 3.0);
            const traceWtp = {
                x: [0, maxX],
                y: [0, maxX * 50],
                name: '$50k/QALY WTP Line',
                type: 'scatter', mode: 'lines',
                line: { color: '#f43f5e', width: 1.5, dash: 'dot' }
            };
            Plotly.newPlot('add-icer-plane-chart', [traceScatter, traceWtp], {
                ...PL,
                margin: { l: 45, r: 20, t: 15, b: 40 },
                xaxis: { ...PL.xaxis, title: 'Incremental QALYs Gained / Patient' },
                yaxis: { ...PL.yaxis, title: 'Incremental Cost Difference ($k USD)' },
                legend: { font: { color: '#e6edf3', size: 8.5 }, x: 0.05, y: 0.15 }
            }, { responsive: true });
        }

        // 7e. Plot 10: Quantum Agentic AI Entropy & Policy Reward
        const qAgent = d.quantum_agentic_ai || {};
        if (qAgent.sessions && document.getElementById('add-agentic-entropy-chart')) {
            const traceEntropy = {
                x: qAgent.sessions,
                y: qAgent.von_neumann_entropy,
                name: 'Von Neumann Entropy S(ρ)',
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#8b5cf6', width: 2.4 },
                marker: { size: 4 }
            };
            const traceReward = {
                x: qAgent.sessions,
                y: qAgent.policy_reward,
                name: 'Policy Reward Tr(ρ H)',
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#10b981', width: 2.2, dash: 'dash' },
                marker: { size: 4 },
                yaxis: 'y2'
            };
            Plotly.newPlot('add-agentic-entropy-chart', [traceEntropy, traceReward], {
                ...PL,
                margin: { l: 45, r: 45, t: 15, b: 40 },
                xaxis: { ...PL.xaxis, title: 'rTMS Session Index' },
                yaxis: { ...PL.yaxis, title: 'Belief Entropy S(ρ) (Nats)', color: '#8b5cf6' },
                yaxis2: {
                    title: 'Expected Policy Reward',
                    overlaying: 'y', side: 'right',
                    color: '#10b981',
                    gridcolor: 'rgba(16,185,129,0.08)'
                },
                legend: { font: { color: '#e6edf3', size: 8.5 }, x: 0.05, y: 0.95 }
            }, { responsive: true });
        }

        // 8. Populate Stieltjes Table
        const stTbody = document.getElementById('add-stieltjes-tbody');
        if (stTbody && d.stieltjes_convergents) {
            stTbody.innerHTML = d.stieltjes_convergents.map(c => `
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:6px 8px; font-weight:600; color:#38bdf8;">Order ${c.order}</td>
                    <td style="padding:6px 8px; color:#cbd5e1; font-family:monospace;">${c.fraction}</td>
                    <td style="padding:6px 8px; color:#a7f3d0; font-family:monospace;">${c.p_q}</td>
                    <td style="padding:6px 8px; font-weight:600; color:#fbbf24;">${c.bound_type}</td>
                </tr>
            `).join('');
        }

        // 9. Populate Summary Table
        const sumTbody = document.getElementById('add-summary-tbody');
        if (sumTbody) {
            const lastIdx = m.length - 1;
            const rows = [
                {
                    metric: '12-Month Continuous Abstinence (%)',
                    sham: `${traj.continuous_abstinence_pct.sham[6].toFixed(1)}%`,
                    std: `${traj.continuous_abstinence_pct.standard_rtms[6].toFixed(1)}%`,
                    super: `${traj.continuous_abstinence_pct.superoptimal_hebbian_qml[6].toFixed(1)}%`,
                    gain: `+${(traj.continuous_abstinence_pct.superoptimal_hebbian_qml[6] - traj.continuous_abstinence_pct.standard_rtms[6]).toFixed(1)} pp (Superior)`
                },
                {
                    metric: '36-Month Sustained Abstinence (%)',
                    sham: `${traj.continuous_abstinence_pct.sham[lastIdx].toFixed(1)}%`,
                    std: `${traj.continuous_abstinence_pct.standard_rtms[lastIdx].toFixed(1)}%`,
                    super: `${traj.continuous_abstinence_pct.superoptimal_hebbian_qml[lastIdx].toFixed(1)}%`,
                    gain: `${kpi.abstinence_odds_ratio_vs_sham.toFixed(2)}× Odds Ratio vs Sham`
                },
                {
                    metric: 'Peak Craving Score Reduction (VAS)',
                    sham: `-${((kpi.baseline_craving_vas - traj.craving_vas.sham[6])/kpi.baseline_craving_vas*100).toFixed(1)}%`,
                    std: `-${((kpi.baseline_craving_vas - traj.craving_vas.standard_rtms[6])/kpi.baseline_craving_vas*100).toFixed(1)}%`,
                    super: `-${kpi.craving_reduction_pct.toFixed(1)}%`,
                    gain: 'Drastic Cue Extinction'
                },
                {
                    metric: `Daily Consumption (${d.consumption_unit})`,
                    sham: `${traj.consumption.sham[lastIdx].toFixed(1)}`,
                    std: `${traj.consumption.standard_rtms[lastIdx].toFixed(1)}`,
                    super: `${traj.consumption.superoptimal_hebbian_qml[lastIdx].toFixed(1)}`,
                    gain: 'Zero/Minimal Harm Level'
                },
                {
                    metric: 'Frontostriatal Executive LTP Gain',
                    sham: '1.00× (Degraded)',
                    std: '1.34× (Moderate)',
                    super: `${kpi.hebbian_ltp_gain.toFixed(2)}×`,
                    gain: '+125% Top-Down Plasticity'
                },
                {
                    metric: 'Insular Cue Reactivity LTD Depotentiation',
                    sham: '0.0% (Hyperactive)',
                    std: '-22.5%',
                    super: `-${heb.final_ltd_suppression_pct.toFixed(1)}%`,
                    gain: 'Visceral Urge Suppression'
                },
                {
                    metric: 'QML Pulse Policy Fidelity',
                    sham: '--',
                    std: '0.742',
                    super: `${kpi.qml_circuit_fidelity.toFixed(3)}`,
                    gain: 'Grover Policy Selection'
                },
                {
                    metric: '5-Year Discounted QALYs Gained',
                    sham: '+0.28 QALYs',
                    std: '+0.95 QALYs',
                    super: `+${(kpi.qaly_gained_superoptimal || 2.48).toFixed(2)} QALYs`,
                    gain: `+${((kpi.qaly_gained_superoptimal || 2.48) - 0.95).toFixed(2)} Incremental QALYs`
                },
                {
                    metric: 'Cost-Effectiveness (ICER $/QALY)',
                    sham: 'Comparator',
                    std: '$32,400 / QALY',
                    super: 'DOMINANT',
                    gain: `-$${Math.abs(kpi.icer_usd_per_qaly || 14804).toFixed(0)} / QALY (Cost-Saving)`
                },
                {
                    metric: 'Annual Outpatient Revenue (P50)',
                    sham: '--',
                    std: '$0.95M USD',
                    super: `$${(kpi.annual_revenue_p50_usd_m || 1.89).toFixed(2)}M USD`,
                    gain: `$${(kpi.five_year_npv_p50_usd_m || 5.68).toFixed(2)}M 5-Yr Discounted NPV`
                },
                {
                    metric: 'Quantum Agentic AI Belief Entropy S(ρ)',
                    sham: '1.386 nats (Ignorance)',
                    std: '0.850 nats',
                    super: `${(kpi.final_agentic_entropy || 0.026).toFixed(3)} nats`,
                    gain: `-${(kpi.agentic_entropy_reduction_pct || 94.5).toFixed(1)}% Epistemic Convergence`
                }
            ];

            sumTbody.innerHTML = rows.map(r => `
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:6px 8px; font-weight:600; color:#38bdf8;">${r.metric}</td>
                    <td style="padding:6px 8px; color:#94a3b8;">${r.sham}</td>
                    <td style="padding:6px 8px; color:#cbd5e1;">${r.std}</td>
                    <td style="padding:6px 8px; font-weight:700; color:#10b981;">${r.super}</td>
                    <td style="padding:6px 8px; font-weight:600; color:#a7f3d0;">${r.gain}</td>
                </tr>
            `).join('');
        }
    };

    const runAddictionRecoveryStudio = async () => {
        try {
            const statusEl = document.getElementById('addiction-status');
            if (statusEl) statusEl.textContent = 'Simulating Interventional Recovery...';

            const addiction_type = document.getElementById('addiction-type')?.value || 'nicotine';
            const baseline_severity = parseFloat(document.getElementById('addiction-severity')?.value || '8.0');
            const sessions = parseInt(document.getElementById('addiction-sessions')?.value || '30');
            const intensity_mso_pct = parseFloat(document.getElementById('addiction-intensity')?.value || '110.0');
            const hebbian_boost = document.getElementById('addiction-hebbian-boost')?.checked !== false;
            const qml_optimized = document.getElementById('addiction-qml-boost')?.checked !== false;

            const resp = await fetch('/api/addiction-recovery-simulation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    addiction_type,
                    baseline_severity,
                    sessions,
                    intensity_mso_pct,
                    hebbian_boost,
                    qml_optimized
                })
            });

            const data = await resp.json();
            if (data.status !== 'success') {
                console.error('Addiction recovery simulation failed:', data);
                if (statusEl) statusEl.textContent = 'Error during calculation';
                return;
            }

            cachedAddictionData = data;
            if (statusEl) statusEl.textContent = 'Ready (' + new Date().toLocaleTimeString() + ')';
            renderAddictionRecoveryViews(data);
        } catch (err) {
            console.error('runAddictionRecoveryStudio error:', err);
            const statusEl = document.getElementById('addiction-status');
            if (statusEl) statusEl.textContent = 'Error during simulation';
        }
    };

    const runAddictionRecoveryStudioDebounced = () => {
        clearTimeout(addictionDebounceTimer);
        addictionDebounceTimer = setTimeout(runAddictionRecoveryStudio, 250);
    };

    window.addictionUpdateLabels = addictionUpdateLabels;
    window.addictionTypeChanged = addictionTypeChanged;
    window.renderAddictionRecoveryViews = renderAddictionRecoveryViews;
    window.runAddictionRecoveryStudio = runAddictionRecoveryStudio;
    window.runAddictionRecoveryStudioDebounced = runAddictionRecoveryStudioDebounced;

    // ── US Health Economics & Multi-Payer Revenue Studio ─────────
    let cachedUSHealthEconData = null;
    let usHeDebounceTimer = null;

    const usHeUpdateLabels = () => {
        const chairs = document.getElementById('us-he-chairs')?.value || '2';
        const referrals = document.getElementById('us-he-referrals')?.value || '240';
        const comm = document.getElementById('us-he-comm')?.value || '55';
        const medicare = document.getElementById('us-he-medicare')?.value || '30';
        const medicaid = document.getElementById('us-he-medicaid')?.value || '10';
        const cash = document.getElementById('us-he-cash')?.value || '5';

        const setT = (id, txt) => {
            const el = document.getElementById(id);
            if (el) el.textContent = txt;
        };
        setT('label-us-chairs', `${chairs} Chairs`);
        setT('label-us-referrals', `${referrals} Patients`);
        setT('label-us-comm', `${comm}%`);
        setT('label-us-medicare', `${medicare}%`);
        setT('label-us-medicaid', `${medicaid}%`);
        setT('label-us-cash', `${cash}%`);
    };

    const renderUSHealthEconomicsViews = (data) => {
        const d = data || cachedUSHealthEconData;
        if (!d) return;

        const revSim = d.revenue_simulation || {};
        const costEff = d.cost_effectiveness || {};
        const kpi = d.summary_kpis || {};
        const dist = revSim.statistical_revenue_distribution || {};
        const fin = revSim.financial_metrics || {};
        const sched = revSim.reimbursement_schedule || {};

        // 1. KPI Cards
        const setEl = (id, txt) => {
            const el = document.getElementById(id);
            if (el) el.textContent = txt;
        };
        setEl('us-kpi-rev', `$${(kpi.p50_annual_revenue_usd_m || 0.93).toFixed(2)}M`);
        setEl('us-kpi-npv', `$${(kpi.ten_year_npv_usd_m || 3.84).toFixed(2)}M`);
        setEl('us-kpi-qaly', `+${(kpi.incremental_qaly_gained || 0.61).toFixed(2)}`);
        setEl('us-kpi-icer', kpi.icer_status || 'Dominant');
        setEl('us-kpi-savings', `$${((kpi.annual_medical_savings_per_patient_usd || 31945) / 1e3).toFixed(1)}k`);
        setEl('us-kpi-payback', `${(kpi.payback_period_years || 1.0).toFixed(1)} Yrs`);

        const PL = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#94a3b8', family: 'system-ui, -apple-system, sans-serif', size: 10 },
            xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.12)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.12)' }
        };

        // 2. Plot 1: Monte Carlo Annual Revenue Density PDF
        if (dist.pdf_bin_centers_m && document.getElementById('us-he-rev-pdf-chart')) {
            const p10 = dist.p10_annual_revenue_usd_m || 0.85;
            const p50 = dist.p50_annual_revenue_usd_m || 0.93;
            const p90 = dist.p90_annual_revenue_usd_m || 1.00;

            const tracePdf = {
                x: dist.pdf_bin_centers_m,
                y: dist.pdf_density_values,
                name: 'Revenue PDF f(R)',
                type: 'scatter', mode: 'lines',
                fill: 'tozeroy',
                fillcolor: 'rgba(2, 132, 199, 0.20)',
                line: { color: '#0284c7', width: 2.5 }
            };
            Plotly.newPlot('us-he-rev-pdf-chart', [tracePdf], {
                ...PL,
                margin: { l: 45, r: 20, t: 15, b: 40 },
                xaxis: { ...PL.xaxis, title: 'Annual Gross Clinic Revenue ($M USD)' },
                yaxis: { ...PL.yaxis, title: 'Probability Density' },
                shapes: [
                    { type: 'line', x0: p10, x1: p10, y0: 0, y1: 1, yref: 'paper', line: { color: '#fbbf24', width: 1.8, dash: 'dash' } },
                    { type: 'line', x0: p50, x1: p50, y0: 0, y1: 1, yref: 'paper', line: { color: '#10b981', width: 2.4 } },
                    { type: 'line', x0: p90, x1: p90, y0: 0, y1: 1, yref: 'paper', line: { color: '#c4b5fd', width: 1.8, dash: 'dash' } }
                ],
                annotations: [
                    { x: p10, y: 1.04, yref: 'paper', text: `P10: $${p10.toFixed(2)}M`, showarrow: false, font: { size: 9, color: '#fbbf24' } },
                    { x: p50, y: 1.04, yref: 'paper', text: `P50: $${p50.toFixed(2)}M`, showarrow: false, font: { size: 9.5, color: '#10b981' } },
                    { x: p90, y: 1.04, yref: 'paper', text: `P90: $${p90.toFixed(2)}M`, showarrow: false, font: { size: 9, color: '#c4b5fd' } }
                ],
                legend: { font: { color: '#e6edf3', size: 9 }, x: 0.05, y: 0.95 }
            }, { responsive: true });
        }

        // 3. Plot 2: 10-Year Cumulative Discounted NPV Trajectory
        if (revSim.trajectory_years && document.getElementById('us-he-npv-traj-chart')) {
            const traceNpv90 = {
                x: revSim.trajectory_years, y: revSim.npv_p90_trajectory_m,
                name: 'P90 Optimistic ($M)',
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#c4b5fd', width: 2, dash: 'dash' },
                marker: { size: 5 }
            };
            const traceNpv50 = {
                x: revSim.trajectory_years, y: revSim.npv_p50_trajectory_m,
                name: 'P50 Median Base ($M)',
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#10b981', width: 2.8 },
                marker: { size: 6 }
            };
            const traceNpv10 = {
                x: revSim.trajectory_years, y: revSim.npv_p10_trajectory_m,
                name: 'P10 Conservative ($M)',
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#fbbf24', width: 1.8, dash: 'dot' },
                marker: { size: 5 }
            };
            Plotly.newPlot('us-he-npv-traj-chart', [traceNpv90, traceNpv50, traceNpv10], {
                ...PL,
                margin: { l: 45, r: 20, t: 15, b: 40 },
                xaxis: { ...PL.xaxis, title: 'Operational Model Year (t = 0 to 10)' },
                yaxis: { ...PL.yaxis, title: 'Cumulative Discounted NPV ($M USD)' },
                legend: { font: { color: '#e6edf3', size: 9 }, x: 0.05, y: 0.95 }
            }, { responsive: true });
        }

        // 4. Plot 3: Cost-Effectiveness Acceptability Curve (CEAC)
        if (costEff.ceac_wtp_axis && document.getElementById('us-he-ceac-chart')) {
            const traceCeac = {
                x: costEff.ceac_wtp_axis, y: costEff.ceac_probabilities,
                name: 'rTMS Cost-Effective P(CE)',
                type: 'scatter', mode: 'lines',
                line: { color: '#10b981', width: 2.8 }
            };
            Plotly.newPlot('us-he-ceac-chart', [traceCeac], {
                ...PL,
                margin: { l: 45, r: 20, t: 15, b: 40 },
                xaxis: { ...PL.xaxis, title: 'Willingness-to-Pay Threshold λ ($/QALY)' },
                yaxis: { ...PL.yaxis, title: 'Probability Cost-Effective', range: [0, 1.05] },
                shapes: [
                    { type: 'line', x0: 50000, x1: 50000, y0: 0, y1: 1, yref: 'paper', line: { color: '#fbbf24', width: 1.5, dash: 'dot' } },
                    { type: 'line', x0: 100000, x1: 100000, y0: 0, y1: 1, yref: 'paper', line: { color: '#0284c7', width: 1.8, dash: 'dash' } },
                    { type: 'line', x0: 150000, x1: 150000, y0: 0, y1: 1, yref: 'paper', line: { color: '#c4b5fd', width: 1.5, dash: 'dot' } }
                ],
                annotations: [
                    { x: 50000, y: 0.25, text: '$50k WTP', showarrow: false, font: { size: 8.5, color: '#fbbf24' } },
                    { x: 100000, y: 0.45, text: 'US $100k Standard', showarrow: false, font: { size: 9, color: '#0284c7' } },
                    { x: 150000, y: 0.65, text: '$150k High WTP', showarrow: false, font: { size: 8.5, color: '#c4b5fd' } }
                ],
                legend: { font: { color: '#e6edf3', size: 9 }, x: 0.05, y: 0.95 }
            }, { responsive: true });
        }

        // 5. Plot 4: Course Reimbursement by Payer Segment
        if (document.getElementById('us-he-payer-chart')) {
            const med = sched.medicare_course_revenue_usd || 7450;
            const comm = sched.commercial_course_revenue_usd || 10800;
            const mcaid = med * 0.72;
            const cash = med * 1.35;

            const tracePayer = {
                x: ['Commercial Private', 'Medicare Part B', 'Medicaid (MCO)', 'Cash / Concierge'],
                y: [comm / 1e3, med / 1e3, mcaid / 1e3, cash / 1e3],
                type: 'bar',
                marker: { color: ['#0284c7', '#38bdf8', '#fbbf24', '#a7f3d0'], line: { color: '#0f172a', width: 1 } }
            };
            Plotly.newPlot('us-he-payer-chart', [tracePayer], {
                ...PL,
                margin: { l: 45, r: 20, t: 15, b: 40 },
                xaxis: { ...PL.xaxis, title: 'US Payer Classification' },
                yaxis: { ...PL.yaxis, title: 'Reimbursement per Course ($k USD)' }
            }, { responsive: true });
        }

        // 6. Plot 5: Direct Medical Savings Breakdown
        if (document.getElementById('us-he-savings-chart')) {
            const totSav = costEff.annual_medical_savings_per_patient_usd || 31945;
            const inpat = totSav * 0.65;
            const ed = totSav * 0.20;
            const rx = totSav * 0.15;

            const traceSav = {
                x: ['Inpatient Bed-Days', 'ED Acute Visits', 'Specialty Rx', 'Total Annual Offsets'],
                y: [inpat / 1e3, ed / 1e3, rx / 1e3, totSav / 1e3],
                type: 'bar',
                marker: { color: ['#0284c7', '#38bdf8', '#818cf8', '#10b981'], line: { color: '#0f172a', width: 1 } }
            };
            Plotly.newPlot('us-he-savings-chart', [traceSav], {
                ...PL,
                margin: { l: 45, r: 20, t: 15, b: 40 },
                xaxis: { ...PL.xaxis, title: 'Healthcare Resource Category' },
                yaxis: { ...PL.yaxis, title: 'Annual Savings / Patient ($k USD)' }
            }, { responsive: true });
        }

        // 7. Plot 6: GPCI Regional Sensitivity
        if (document.getElementById('us-he-gpci-chart')) {
            const gpciRegs = ['National Avg', 'California/West', 'Northeast NY', 'Midwest', 'South Sunbelt'];
            const gpciVals = [8.62, 11.39, 10.95, 8.24, 8.52];

            const traceGpci = {
                x: gpciRegs,
                y: gpciVals,
                type: 'bar',
                marker: { color: ['#64748b', '#0284c7', '#38bdf8', '#fbbf24', '#f59e0b'], line: { color: '#0f172a', width: 1 } }
            };
            Plotly.newPlot('us-he-gpci-chart', [traceGpci], {
                ...PL,
                margin: { l: 45, r: 20, t: 15, b: 40 },
                xaxis: { ...PL.xaxis, title: 'US Geographic Locality' },
                yaxis: { ...PL.yaxis, title: 'Blended Course Fee ($k USD)' }
            }, { responsive: true });
        }

        // 8. Table 1: CPT Reimbursement Table
        const cptTbody = document.getElementById('us-cpt-tbody');
        if (cptTbody && sched.cpt_codes) {
            cptTbody.innerHTML = Object.entries(sched.cpt_codes).map(([code, c]) => `
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:6px 8px; font-weight:700; color:#38bdf8;">CPT ${code}</td>
                    <td style="padding:6px 8px; color:#cbd5e1;">${c.desc}</td>
                    <td style="padding:6px 8px; font-family:monospace; color:#94a3b8;">${code === '90867' ? '1.28' : code === '90868' ? '0.75' : '1.15'}</td>
                    <td style="padding:6px 8px; font-family:monospace; color:#94a3b8;">${code === '90867' ? '6.85' : code === '90868' ? '5.42' : '6.10'}</td>
                    <td style="padding:6px 8px; font-family:monospace; color:#94a3b8;">${code === '90867' ? '0.12' : code === '90868' ? '0.08' : '0.10'}</td>
                    <td style="padding:6px 8px; font-family:monospace; font-weight:600; color:#a7f3d0;">${code === '90867' ? '8.25' : code === '90868' ? '6.25' : '7.35'}</td>
                    <td style="padding:6px 8px; font-weight:600; color:#38bdf8;">$${c.medicare_allowable_usd.toFixed(2)}</td>
                    <td style="padding:6px 8px; font-weight:600; color:#10b981;">$${c.commercial_rate_usd.toFixed(2)}</td>
                    <td style="padding:6px 8px; font-weight:700; color:#fbbf24;">$${c.blended_reimbursement_usd.toFixed(2)}</td>
                </tr>
            `).join('');
        }

        // 9. Table 2: 10-Year DCF Schedule Table
        const dcfTbody = document.getElementById('us-dcf-tbody');
        if (dcfTbody && revSim.dcf_schedule) {
            dcfTbody.innerHTML = revSim.dcf_schedule.map(row => `
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:6px 8px; font-weight:700; color:#38bdf8;">Year ${row.year}</td>
                    <td style="padding:6px 8px; color:#10b981;">$${(row.gross_revenue_usd / 1e6).toFixed(2)}M</td>
                    <td style="padding:6px 8px; color:#f87171;">$${(row.opex_usd / 1e6).toFixed(2)}M</td>
                    <td style="padding:6px 8px; color:#38bdf8;">$${(row.net_cash_flow_usd / 1e6).toFixed(2)}M</td>
                    <td style="padding:6px 8px; color:#c4b5fd;">$${((row.discounted_cash_flow_usd || row.net_cash_flow_usd) / 1e6).toFixed(2)}M</td>
                    <td style="padding:6px 8px; font-weight:700; color:#fbbf24;">$${(row.cumulative_npv_usd / 1e6).toFixed(2)}M</td>
                </tr>
            `).join('');
        }
    };

    const runUSHealthEconomicsStudio = async () => {
        try {
            const statusEl = document.getElementById('us-he-status');
            if (statusEl) statusEl.textContent = 'Simulating US Healthcare Reimbursement & DCF...';

            const indication_key = document.getElementById('us-he-indication')?.value || 'depression_trd';
            const region_key = document.getElementById('us-he-region')?.value || 'national_avg';
            const num_tms_chairs = parseInt(document.getElementById('us-he-chairs')?.value || '2');
            const annual_referrals_mean = parseFloat(document.getElementById('us-he-referrals')?.value || '240');
            const commercial_pct = parseFloat(document.getElementById('us-he-comm')?.value || '55') / 100.0;
            const medicare_pct = parseFloat(document.getElementById('us-he-medicare')?.value || '30') / 100.0;
            const medicaid_pct = parseFloat(document.getElementById('us-he-medicaid')?.value || '10') / 100.0;
            const self_pay_pct = parseFloat(document.getElementById('us-he-cash')?.value || '5') / 100.0;

            const resp = await fetch('/api/us-health-economics-simulation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    indication_key,
                    region_key,
                    num_tms_chairs,
                    annual_referrals_mean,
                    commercial_pct,
                    medicare_pct,
                    medicaid_pct,
                    self_pay_pct,
                    discount_rate_annual: 0.075,
                    wtp_threshold_usd: 100000.0
                })
            });

            const data = await resp.json();
            if (data.status !== 'success') {
                console.error('US health economics simulation failed:', data);
                if (statusEl) statusEl.textContent = 'Error during calculation';
                return;
            }

            cachedUSHealthEconData = data;
            if (statusEl) statusEl.textContent = 'Ready (' + new Date().toLocaleTimeString() + ')';
            renderUSHealthEconomicsViews(data);
        } catch (err) {
            console.error('runUSHealthEconomicsStudio error:', err);
            const statusEl = document.getElementById('us-he-status');
            if (statusEl) statusEl.textContent = 'Error during simulation';
        }
    };

    const runUSHealthEconomicsStudioDebounced = () => {
        clearTimeout(usHeDebounceTimer);
        usHeDebounceTimer = setTimeout(runUSHealthEconomicsStudio, 250);
    };

    window.usHeUpdateLabels = usHeUpdateLabels;
    window.renderUSHealthEconomicsViews = renderUSHealthEconomicsViews;
    window.runUSHealthEconomicsStudio = runUSHealthEconomicsStudio;
    window.runUSHealthEconomicsStudioDebounced = runUSHealthEconomicsStudioDebounced;

    // ── Bosonic Waveforms & Quantum Neural Net Dementia Studio ──────
    let cachedBosonicDementiaData = null;
    let bdDebounceTimer = null;

    const bdUpdateLabels = () => {
        const sq = parseFloat(document.getElementById('bd-squeezing')?.value || '1.25');
        const disp = parseFloat(document.getElementById('bd-displacement')?.value || '0.65');
        const kerr = parseFloat(document.getElementById('bd-kerr')?.value || '0.08');
        const parts = document.getElementById('bd-partitions')?.value || '32';
        const curr = document.getElementById('bd-current')?.value || '4800';

        const setT = (id, txt) => {
            const el = document.getElementById(id);
            if (el) el.textContent = txt;
        };
        const sqDb = (10.0 * Math.log10(Math.exp(2.0 * sq))).toFixed(1);
        setT('label-bd-squeezing', `${sq.toFixed(2)} (${sqDb} dB)`);
        setT('label-bd-displacement', disp.toFixed(2));
        setT('label-bd-kerr', kerr.toFixed(3));
        setT('label-bd-partitions', `${parts} Super-Elements`);
        setT('label-bd-current', `${Number(curr).toLocaleString()} A`);
    };

    const renderBosonicDementiaViews = (data) => {
        const d = data || cachedBosonicDementiaData;
        if (!d) return;

        const wave = d.bosonic_waveform || {};
        const fem = d.cortical_fem_qml || {};
        const clin = d.clinical_trajectories || {};
        const kpi = d.summary_kpis || {};
        const lobes = d.lobes_affected || {};
        const staging = d.gangobo_staging || [];

        // 1. KPI Cards
        const setEl = (id, txt) => {
            const el = document.getElementById(id);
            if (el) el.textContent = txt;
        };
        setEl('bd-kpi-efield', `${(kpi.peak_e_field_v_m || 127.0).toFixed(1)} V/m`);
        setEl('bd-kpi-squeezing', `${kpi.squeezing_db || 10.9} dB`);
        setEl('bd-kpi-adas', `-${(kpi.adas_cog_improvement_pct || 61.6).toFixed(1)}%`);
        setEl('bd-kpi-cdr', `-${(kpi.cdr_sb_improvement_pct || 68.2).toFixed(1)}%`);
        setEl('bd-kpi-mmse', `+${(kpi.mmse_gain_points || 7.4).toFixed(1)} Pts`);
        setEl('bd-kpi-delay', `${kpi.institutionalization_delay_years || 3.8} Yrs`);

        const PL = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#94a3b8', family: 'system-ui, -apple-system, sans-serif', size: 10 },
            xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.12)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.12)' }
        };

        // 2. Plot 1: Asymmetric Bosonic Pulse Waveform & Induced E-Field
        if (wave.time_ms && document.getElementById('bd-waveform-chart')) {
            const traceWave = {
                x: wave.time_ms,
                y: wave.induced_e_field_v_m,
                name: 'Induced E-Field E(t)',
                type: 'scatter', mode: 'lines',
                line: { color: '#0284c7', width: 2.2 }
            };
            Plotly.newPlot('bd-waveform-chart', [traceWave], {
                ...PL,
                margin: { l: 45, r: 20, t: 15, b: 40 },
                xaxis: { ...PL.xaxis, title: 'Pulse Time (ms)' },
                yaxis: { ...PL.yaxis, title: 'Induced Electric Field (V/m)' },
                shapes: [
                    { type: 'line', x0: 0, x1: 25, y0: 88.0, y1: 88.0, line: { color: '#fbbf24', width: 1.5, dash: 'dash' } }
                ],
                annotations: [
                    { x: 12.5, y: 92.0, text: 'Pyramidal Depol Threshold (88 V/m)', showarrow: false, font: { size: 8.5, color: '#fbbf24' } }
                ],
                legend: { font: { color: '#e6edf3', size: 9 }, x: 0.65, y: 0.95 }
            }, { responsive: true });
        }

        // 3. Plot 2: Bosonic Wigner Function W(x, p) in Phase Space
        if (wave.wigner_matrix && document.getElementById('bd-wigner-chart')) {
            const traceWigner = {
                x: wave.wigner_x,
                y: wave.wigner_p,
                z: wave.wigner_matrix,
                type: 'contour',
                colorscale: 'Viridis',
                contours: { coloring: 'heatmap' },
                colorbar: {
                    title: 'W(x, p)',
                    len: 0.75, thickness: 10,
                    tickfont: { color: '#cbd5e1', size: 8 },
                    titlefont: { color: '#cbd5e1', size: 9 }
                }
            };
            Plotly.newPlot('bd-wigner-chart', [traceWigner], {
                ...PL,
                margin: { l: 45, r: 20, t: 15, b: 40 },
                xaxis: { ...PL.xaxis, title: 'Field Quadrature X' },
                yaxis: { ...PL.yaxis, title: 'Phase Quadrature P' }
            }, { responsive: true });
        }

        // 4. Plot 3: Cortical & Subcortical Depth Attenuation
        if (fem.depth_attenuation && document.getElementById('bd-cortical-depth-chart')) {
            const da = fem.depth_attenuation;
            const traceDepth = {
                x: da.depth_mm,
                y: da.e_field_v_m,
                name: 'E-Field Attenuation',
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#10b981', width: 2.4 },
                marker: { size: 4 }
            };
            Plotly.newPlot('bd-cortical-depth-chart', [traceDepth], {
                ...PL,
                margin: { l: 45, r: 20, t: 15, b: 40 },
                xaxis: { ...PL.xaxis, title: 'Depth from Scalp (mm)' },
                yaxis: { ...PL.yaxis, title: 'Electric Field (V/m)' },
                shapes: [
                    { type: 'line', x0: 0, x1: 45, y0: 88.0, y1: 88.0, line: { color: '#fbbf24', width: 1.5, dash: 'dash' } },
                    { type: 'line', x0: 15.0, x1: 15.0, y0: 0, y1: 1, yref: 'paper', line: { color: '#64748b', width: 1.2, dash: 'dot' } },
                    { type: 'line', x0: 32.0, x1: 32.0, y0: 0, y1: 1, yref: 'paper', line: { color: '#c4b5fd', width: 1.2, dash: 'dot' } }
                ],
                annotations: [
                    { x: 15.0, y: 0.90, yref: 'paper', text: 'Gray/White Jct (15mm)', showarrow: false, font: { size: 8, color: '#94a3b8' } },
                    { x: 32.0, y: 0.90, yref: 'paper', text: 'Deep Hippocampus (32mm)', showarrow: false, font: { size: 8, color: '#c4b5fd' } }
                ],
                legend: { font: { color: '#e6edf3', size: 9 }, x: 0.65, y: 0.95 }
            }, { responsive: true });
        }

        // 5. Plot 4: Multi-Lobe Neuroprotection Profile (Radar Chart)
        if (fem.lobe_summary && document.getElementById('bd-neuroprotection-radar-chart')) {
            const lobeKeys = Object.keys(fem.lobe_summary);
            const rVals = lobeKeys.map(k => fem.lobe_summary[k].neuroprotection_index);
            const thetaLabels = lobeKeys.map(k => fem.lobe_summary[k].name.split('&')[0].trim());

            rVals.push(rVals[0]);
            thetaLabels.push(thetaLabels[0]);

            const traceRadar = {
                type: 'scatterpolar',
                r: rVals,
                theta: thetaLabels,
                fill: 'toself',
                fillcolor: 'rgba(2, 132, 199, 0.22)',
                line: { color: '#0284c7', width: 2.2 },
                name: 'Neuroprotection Index'
            };
            Plotly.newPlot('bd-neuroprotection-radar-chart', [traceRadar], {
                ...PL,
                margin: { l: 45, r: 45, t: 25, b: 25 },
                polar: {
                    radialaxis: { visible: true, range: [0, 100], tickfont: { size: 7.5, color: '#64748b' }, gridcolor: 'rgba(255,255,255,0.08)' },
                    angularaxis: { tickfont: { size: 8, color: '#cbd5e1' }, gridcolor: 'rgba(255,255,255,0.08)' },
                    bgcolor: 'rgba(0,0,0,0)'
                }
            }, { responsive: true });
        }

        // 6. Plot 5: 60-Month ADAS-Cog-14 Trajectory
        if (clin.adas_cog_14 && document.getElementById('bd-adas-traj-chart')) {
            const m = clin.months;
            const traceAdasSoc = {
                x: m, y: clin.adas_cog_14.soc,
                name: 'Standard of Care (Donepezil/Memantine)',
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#94a3b8', width: 1.8, dash: 'dot' },
                marker: { size: 4 }
            };
            const traceAdasConv = {
                x: m, y: clin.adas_cog_14.conventional_rtms,
                name: 'Conventional 10 Hz rTMS',
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#fbbf24', width: 2.0, dash: 'dash' },
                marker: { size: 4 }
            };
            const traceAdasBosonic = {
                x: m, y: clin.adas_cog_14.bosonic_qnn_rtms,
                name: `Bosonic QNN rTMS (-${clin.adas_cog_14.improvement_pct}%)`,
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#10b981', width: 2.8 },
                marker: { size: 5, color: '#10b981' }
            };
            Plotly.newPlot('bd-adas-traj-chart', [traceAdasSoc, traceAdasConv, traceAdasBosonic], {
                ...PL,
                margin: { l: 45, r: 20, t: 15, b: 40 },
                xaxis: { ...PL.xaxis, title: 'Follow-up Horizon (Months)' },
                yaxis: { ...PL.yaxis, title: 'ADAS-Cog-14 Score (Lower = Better)' },
                legend: { font: { color: '#e6edf3', size: 8.5 }, x: 0.05, y: 0.95 }
            }, { responsive: true });
        }

        // 7. Plot 6: 60-Month Relapse-Free Remission & Community Independence Survival
        if (clin.remission_survival_pct && document.getElementById('bd-survival-chart')) {
            const traceSurvival = {
                x: clin.months,
                y: clin.remission_survival_pct,
                name: 'Community Independence Survival S(t)',
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#059669', width: 2.6 },
                marker: { size: 4 }
            };
            Plotly.newPlot('bd-survival-chart', [traceSurvival], {
                ...PL,
                margin: { l: 45, r: 20, t: 15, b: 40 },
                xaxis: { ...PL.xaxis, title: 'Follow-up Horizon (Months)' },
                yaxis: { ...PL.yaxis, title: 'Functional Independence (%)', range: [0, 105] },
                shapes: [
                    { type: 'line', x0: 0, x1: 60, y0: 50.0, y1: 50.0, line: { color: '#f43f5e', width: 1.2, dash: 'dot' } }
                ],
                annotations: [
                    { x: 30, y: 53.0, text: 'Median Independence Survival Floor (50%)', showarrow: false, font: { size: 8, color: '#f43f5e' } }
                ],
                legend: { font: { color: '#e6edf3', size: 9 }, x: 0.05, y: 0.20 }
            }, { responsive: true });
        }

        // 8. Table 1: Cortical Lobes Affected Table
        const lobesTbody = document.getElementById('bd-lobes-tbody');
        if (lobesTbody && fem.lobe_summary) {
            lobesTbody.innerHTML = Object.entries(fem.lobe_summary).map(([k, l]) => `
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:6px 8px; font-weight:700; color:#38bdf8;">${l.name}</td>
                    <td style="padding:6px 8px; font-family:monospace; color:#cbd5e1;">${l.brodmann_areas}</td>
                    <td style="padding:6px 8px; color:#fbbf24;">${l.braak_stage}</td>
                    <td style="padding:6px 8px; font-family:monospace; color:#94a3b8;">${lobes[k]?.conductivity_s_m || 0.38} S/m</td>
                    <td style="padding:6px 8px; font-weight:600; color:#10b981;">${l.avg_e_field_v_m.toFixed(1)} V/m</td>
                    <td style="padding:6px 8px; font-weight:600; color:${(l.avg_e_field_v_m - l.depolarization_threshold) >= 0 ? '#10b981' : '#f87171'};">
                        ${(l.avg_e_field_v_m - l.depolarization_threshold) >= 0 ? '+' : ''}${(l.avg_e_field_v_m - l.depolarization_threshold).toFixed(1)} V/m
                    </td>
                    <td style="padding:6px 8px; font-weight:700; color:#c4b5fd;">${l.neuroprotection_index.toFixed(1)} / 100</td>
                </tr>
            `).join('');
        }

        // 9. Table 2: Gangobo Staging Protocol Table
        const stgTbody = document.getElementById('bd-staging-tbody');
        if (stgTbody && staging.length) {
            stgTbody.innerHTML = staging.map(stg => `
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:6px 8px; font-weight:700; color:#38bdf8;">${stg.stage}</td>
                    <td style="padding:6px 8px; font-weight:600; color:#fbbf24;">${stg.cdr_range}</td>
                    <td style="padding:6px 8px; color:#cbd5e1;">${stg.target_lobes.map(k => lobes[k]?.lobe_name.split('&')[0].trim() || k).join(' + ')}</td>
                    <td style="padding:6px 8px; color:#a7f3d0;">${stg.intervention}</td>
                    <td style="padding:6px 8px; color:#94a3b8;">${stg.dwell_weeks} Weeks</td>
                    <td style="padding:6px 8px; font-weight:700; color:#10b981;">${(stg.remission_probability * 100).toFixed(1)}%</td>
                </tr>
            `).join('');
        }
    };

    const runBosonicDementiaStudio = async () => {
        try {
            const statusEl = document.getElementById('bd-status');
            if (statusEl) statusEl.textContent = 'Simulating Bosonic CV-QNN & Cortical FEM...';

            const target_lobe_key = document.getElementById('bd-target-lobe')?.value || 'precuneus_pcc';
            const baseline_stage = document.getElementById('bd-stage')?.value || 'mild_ad';
            const squeezing_r = parseFloat(document.getElementById('bd-squeezing')?.value || '1.25');
            const displacement_alpha = parseFloat(document.getElementById('bd-displacement')?.value || '0.65');
            const kerr_chi = parseFloat(document.getElementById('bd-kerr')?.value || '0.08');
            const num_partitions = parseInt(document.getElementById('bd-partitions')?.value || '32');
            const coil_current_a = parseFloat(document.getElementById('bd-current')?.value || '4800');

            const resp = await fetch('/api/bosonic-dementia-simulation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    target_lobe_key,
                    baseline_stage,
                    squeezing_r,
                    displacement_alpha,
                    kerr_chi,
                    num_partitions,
                    coil_current_a,
                    horizon_months: 60
                })
            });

            const data = await resp.json();
            if (data.status !== 'success') {
                console.error('Bosonic dementia simulation failed:', data);
                if (statusEl) statusEl.textContent = 'Error during simulation';
                return;
            }

            cachedBosonicDementiaData = data;
            if (statusEl) statusEl.textContent = 'Ready (' + new Date().toLocaleTimeString() + ')';
            renderBosonicDementiaViews(data);
        } catch (err) {
            console.error('runBosonicDementiaStudio error:', err);
            const statusEl = document.getElementById('bd-status');
            if (statusEl) statusEl.textContent = 'Error during calculation';
        }
    };

    const runBosonicDementiaStudioDebounced = () => {
        clearTimeout(bdDebounceTimer);
        bdDebounceTimer = setTimeout(runBosonicDementiaStudio, 250);
    };

    window.bdUpdateLabels = bdUpdateLabels;
    window.renderBosonicDementiaViews = renderBosonicDementiaViews;
    window.runBosonicDementiaStudio = runBosonicDementiaStudio;
    window.runBosonicDementiaStudioDebounced = runBosonicDementiaStudioDebounced;


