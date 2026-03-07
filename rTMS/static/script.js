document.addEventListener('DOMContentLoaded', () => {

    // ── UI refs ──────────────────────────────────────────────────
    const navButtons      = document.querySelectorAll('.nav-btn');
    const tabTitle        = document.getElementById('tab-title');
    const tabSubtitle     = document.getElementById('tab-subtitle');
    const runBtn          = document.getElementById('run-simulation-btn');
    const loadingOverlay  = document.getElementById('loading-overlay');
    const simulationView  = document.getElementById('simulation-view');
    const equipmentView   = document.getElementById('equipment-view');
    const equipmentList   = document.getElementById('equipment-list');

    const finalFreq       = document.getElementById('final-freq');
    const finalIntensity  = document.getElementById('final-intensity');
    const finalFitness    = document.getElementById('final-fitness');

    // ── State ────────────────────────────────────────────────────
    let currentCondition  = 'stroke';
    let equipmentLoaded   = false;

    // ── Plotly dark theme base ───────────────────────────────────
    const plotlyLayoutDark = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor:  'rgba(0,0,0,0)',
        font: { color: '#8b949e', family: 'Inter' },
        margin: { l: 50, r: 30, t: 30, b: 50 },
        xaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
        yaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
        scene: {
            xaxis: { gridcolor: 'rgba(255,255,255,0.1)', backgroundcolor: 'rgba(0,0,0,0)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.1)', backgroundcolor: 'rgba(0,0,0,0)' },
            zaxis: { gridcolor: 'rgba(255,255,255,0.1)', backgroundcolor: 'rgba(0,0,0,0)' },
            bgcolor: 'rgba(0,0,0,0)'
        }
    };

    // ── Helpers ──────────────────────────────────────────────────
    function categoryBadge(cat) {
        const map = {
            'Stimulator Unit':       'badge-stimulator',
            'Stimulation Coil':      'badge-coil',
            'Neuronavigation System':'badge-navigation',
            'EEG Monitoring':        'badge-eeg',
            'Positioning System':    'badge-positioning',
            'Cloud Infrastructure':  'badge-cloud',
        };
        return map[cat] || 'badge-default';
    }

    function barRow(label, value, max) {
        const pct = Math.min(100, Math.round((value / max) * 100));
        return `
        <div class="eq-op-bar-row">
            <span class="eq-op-bar-label">${label}</span>
            <div class="eq-op-bar-track">
                <div class="eq-op-bar-fill" style="width:${pct}%"></div>
            </div>
            <span class="eq-op-value">${value}</span>
        </div>`;
    }

    // ── Tab switching ────────────────────────────────────────────
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const tab = btn.getAttribute('data-tab');
            currentCondition = tab;

            if (tab === 'equipment') {
                // Show equipment view, hide simulation view and run-btn
                simulationView.classList.add('hidden');
                equipmentView.classList.remove('hidden');
                runBtn.classList.add('hidden');
                tabTitle.textContent    = 'rTMS Equipment & Machinery';
                tabSubtitle.textContent = 'Clinical operating characteristics and system specifications';
                if (!equipmentLoaded) loadEquipment();

            } else {
                // Show simulation view, hide equipment view
                simulationView.classList.remove('hidden');
                equipmentView.classList.add('hidden');
                runBtn.classList.remove('hidden');

                if (tab === 'stroke') {
                    tabTitle.textContent    = 'Stroke Rehabilitation Optimization';
                    tabSubtitle.textContent = 'Dynamic rTMS Parameter Optimization via FEA/BEM for Motor Cortex';
                } else {
                    tabTitle.textContent    = 'Dementia Cognitive Enhancement';
                    tabSubtitle.textContent = 'Deep rTMS Parameter Optimization via FEA/BEM for Prefrontal Cortex';
                }
                resetSimUI();
            }
        });
    });

    // ── Run simulation ───────────────────────────────────────────
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
            alert('Connection error with Local Google Cloud Simulation Bridge.');
        }
    });

    // ── Reset simulation UI ───────────────────────────────────────
    function resetSimUI() {
        finalFreq.textContent       = '--';
        finalIntensity.textContent  = '--';
        finalFitness.textContent    = '--';
        Plotly.purge('optimization-line-chart');
        Plotly.purge('fea-heatmap');
        Plotly.purge('bem-scatter');
    }

    // ── Render simulation dashboard ──────────────────────────────
    function renderSimDashboard(data) {
        const finalParams = data.optimization.final_parameters;
        finalFreq.textContent      = finalParams.frequency_hz.toFixed(1);
        finalIntensity.textContent = finalParams.intensity_mso.toFixed(1);
        finalFitness.textContent   = (finalParams.fitness * 100).toFixed(1) + '%';

        // Optimization convergence
        const traj = data.optimization.convergence_trajectory;
        Plotly.newPlot('optimization-line-chart', [{
            x:    traj.map(t => t.iteration),
            y:    traj.map(t => t.frequency_hz),
            type: 'scatter', mode: 'lines+markers',
            line:   { color: '#58a6ff', width: 3 },
            marker: { size: 6, color: '#8a2be2' }
        }], {
            ...plotlyLayoutDark,
            title: { text: 'Convergence of Protocol Synthesis', font: { color: '#e6edf3' } },
            yaxis: { ...plotlyLayoutDark.yaxis, title: 'Frequency (Hz)' }
        }, { responsive: true });

        // FEA heatmap
        Plotly.newPlot('fea-heatmap', [{
            z: data.fea_grid, type: 'heatmap', colorscale: 'Viridis'
        }], { ...plotlyLayoutDark }, { responsive: true });

        // BEM 3D scatter
        const bem  = data.bem_mesh;
        Plotly.newPlot('bem-scatter', [{
            x: bem.map(v => v.x), y: bem.map(v => v.y), z: bem.map(v => v.z),
            mode: 'markers',
            marker: {
                size: 5, color: bem.map(v => v.c),
                colorscale: 'Inferno', opacity: 0.8,
                showscale: true,
                colorbar: {
                    title:     'Magnetic Stress Strain',
                    titlefont: { color: '#e6edf3' },
                    tickfont:  { color: '#e6edf3' },
                    thickness: 20
                }
            },
            type: 'scatter3d'
        }], { ...plotlyLayoutDark, margin: { l: 0, r: 0, t: 0, b: 0 } }, { responsive: true });
    }

    // ── Load & render equipment tab ──────────────────────────────
    async function loadEquipment() {
        try {
            const res     = await fetch('/api/equipment');
            const payload = await res.json();
            if (payload.status !== 'success') return;

            const items = payload.data;
            renderEquipmentCharts(items);
            renderEquipmentCards(items);
            equipmentLoaded = true;
        } catch (err) {
            console.error('Equipment fetch error:', err);
        }
    }

    function renderEquipmentCharts(items) {
        const names = items.map(e => e.name.length > 22 ? e.name.slice(0,20)+'…' : e.name);
        const oc    = items.map(e => e.operating_characteristics);

        // ── Bar chart: efficiency + EMI shielding grouped ────────
        Plotly.newPlot('eq-bar-chart', [
            {
                name: 'Efficiency (%)',
                type: 'bar', x: names, y: oc.map(o => o.efficiency_pct),
                marker: { color: '#58a6ff' }
            },
            {
                name: 'EMI Shielding (dB)',
                type: 'bar', x: names, y: oc.map(o => o.emi_shielding_db),
                marker: { color: '#8a2be2' }
            }
        ], {
            ...plotlyLayoutDark,
            barmode: 'group',
            legend: { font: { color: '#e6edf3' } },
            xaxis: { ...plotlyLayoutDark.xaxis, tickangle: -25, tickfont: { size: 11 } }
        }, { responsive: true });

        // ── Scatter: thermal efficiency vs EMI shielding ─────────
        Plotly.newPlot('eq-scatter-chart', [{
            x:    oc.map(o => o.emi_shielding_db),
            y:    oc.map(o => o.efficiency_pct),
            text: names,
            mode: 'markers+text',
            textposition: 'top center',
            textfont: { color: '#8b949e', size: 10 },
            marker: {
                size:       oc.map(o => Math.sqrt(o.heat_dissipation_w) * 2),
                color:      oc.map(o => o.efficiency_pct),
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {
                    title: 'Efficiency %',
                    titlefont: { color: '#e6edf3' },
                    tickfont:  { color: '#e6edf3' },
                    thickness: 15
                }
            },
            type: 'scatter'
        }], {
            ...plotlyLayoutDark,
            xaxis: { ...plotlyLayoutDark.xaxis, title: 'EMI Shielding (dB)' },
            yaxis: { ...plotlyLayoutDark.yaxis, title: 'Efficiency (%)' }
        }, { responsive: true });

        // ── Heat dissipation horizontal bar ───────────────────────
        Plotly.newPlot('eq-heat-chart', [{
            type: 'bar',
            orientation: 'h',
            y: [...names].reverse(),
            x: [...oc].reverse().map(o => o.heat_dissipation_w),
            marker: {
                color:      [...oc].reverse().map(o => o.heat_dissipation_w),
                colorscale: 'Inferno',
                showscale: false
            }
        }], {
            ...plotlyLayoutDark,
            xaxis: { ...plotlyLayoutDark.xaxis, title: 'Dissipation (W)' },
            margin: { l: 180, r: 20, t: 20, b: 50 }
        }, { responsive: true });
    }

    function renderEquipmentCards(items) {
        equipmentList.innerHTML = '';
        items.forEach(eq => {
            const specRows = Object.entries(eq.specs).map(([k, v]) =>
                `<tr><td>${k}</td><td>${v}</td></tr>`).join('');

            const oc = eq.operating_characteristics;
            const html = `
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
                    ${barRow('Efficiency',    oc.efficiency_pct,    100)}
                    ${barRow('EMI Shield(dB)',oc.emi_shielding_db,  80)}
                    ${barRow('Op Temp (°C)', oc.op_temp_c,          60)}
                    ${barRow('Max Temp (°C)',oc.max_temp_c,         60)}
                </div>
            </div>`;
            equipmentList.insertAdjacentHTML('beforeend', html);
        });
    }

});
