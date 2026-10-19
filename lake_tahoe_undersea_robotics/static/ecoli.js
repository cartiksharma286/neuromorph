/* Dedicated synthetic NMR spectroscopy & chemometrics dashboard; SVG plots need no external library. */
document.addEventListener('DOMContentLoaded', async () => {
    const byId = id => document.getElementById(id);
    const form = byId('ecoli-form');
    const status = byId('ecoli-status');
    const run = byId('ecoli-run');
    const pdf = byId('ecoli-pdf');
    let lastInputs = null;
    const fields = {
        ecoli_cfu_100ml: 'Assumed E. coli (CFU/100 mL)',
        nitrate_mg_l: 'Nitrate (mg/L)',
        phosphate_mg_l: 'Phosphate (mg/L)',
        turbidity_ntu: 'Turbidity (NTU)',
        organic_mg_l: 'Dissolved organic carbon (mg/L)',
        decay_per_hour: 'Sample decay rate (1/hour)',
        intervention_per_year: 'Source-control rate (1/year)',
        sensor_noise: 'Spectral noise SD (a.u.)',
        seed: 'Reproducible random seed'
    };
    const palette = ['#06b6d4', '#fbbf24', '#c084fc', '#10b981'];

    async function request(path, inputs) {
        const response = await fetch(`/api/ontario/ecoli/${path}`, {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(inputs)
        });
        if (!response.ok) {
            const type = response.headers.get('content-type') || '';
            const message = type.includes('application/json') ? (await response.json()).message : `HTTP ${response.status}`;
            throw new Error(message);
        }
        return response;
    }

    function rows(id, entries) {
        const target = byId(id);
        if (!target) return;
        target.replaceChildren();
        entries.forEach(([label, value]) => {
            const p = document.createElement('p');
            const strong = document.createElement('strong');
            strong.textContent = `${label}: `;
            p.append(strong, document.createTextNode(String(value)));
            target.append(p);
        });
    }

    function plot(id, series, xlabel, ylabel, equalScale = false) {
        const target = byId(id);
        if (!target) return;
        const namespace = 'http://www.w3.org/2000/svg';
        target.replaceChildren();
        const svg = document.createElementNS(namespace, 'svg');
        svg.setAttribute('viewBox', '0 0 600 310');
        svg.setAttribute('role', 'img');
        svg.setAttribute('aria-label', `${ylabel} against ${xlabel}. ${series.map(s => s.name).join(', ')}`);
        function add(tag, attributes, content) {
            const node = document.createElementNS(namespace, tag);
            Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
            if (content !== undefined) node.textContent = content;
            svg.append(node);
            return node;
        }
        const points = series.flatMap(s => s.points);
        const xs = points.map(p => p[0]);
        const ys = points.map(p => p[1]);
        let xmin = Math.min(...xs), xmax = Math.max(...xs);
        let ymin = Math.min(0, ...ys), ymax = Math.max(...ys);
        if (xmax === xmin) xmax += 1;
        if (ymax === ymin) ymax += 1;
        if (equalScale) {
            const span = Math.max(xmax - xmin, (ymax - ymin) * 510 / 200);
            const midX = (xmin + xmax) / 2, midY = (ymin + ymax) / 2;
            xmin = midX - span / 2; xmax = midX + span / 2;
            ymin = midY - span * 200 / 1020; ymax = midY + span * 200 / 1020;
        }
        const x = v => 65 + (v - xmin) / (xmax - xmin) * 510;
        const y = v => 225 - (v - ymin) / (ymax - ymin) * 200;
        add('title', {}, `${ylabel} against ${xlabel}`);
        for (let i = 0; i <= 4; i++) {
            const yy = ymin + (ymax - ymin) * i / 4;
            const xx = xmin + (xmax - xmin) * i / 4;
            add('line', {x1: 65, x2: 575, y1: y(yy), y2: y(yy), stroke: '#334155'});
            add('text', {x: 58, y: y(yy) + 4, 'text-anchor': 'end'}, yy.toFixed(2));
            add('text', {x: x(xx), y: 245, 'text-anchor': 'middle'}, xx.toFixed(1));
        }
        add('text', {x: 65, y: 15}, ylabel);
        add('text', {x: 310, y: 266, 'text-anchor': 'middle'}, xlabel);
        series.forEach((s, i) => {
            const color = palette[i % palette.length];
            if (s.markers) {
                s.points.forEach(p => add('circle', {cx: x(p[0]), cy: y(p[1]), r: 4, fill: color}));
            } else {
                add('polyline', {points: s.points.map(p => `${x(p[0])},${y(p[1])}`).join(' '),
                    fill: 'none', stroke: color, 'stroke-width': 2});
            }
            add('text', {x: 65 + (i % 2) * 260, y: 285 + Math.floor(i / 2) * 17, fill: color}, s.name);
        });
        target.append(svg);
    }

    function render(r) {
        const ch = r.chemometrics || r.qml;
        const s = r.spectra;
        const percent = value => `${(100 * value).toFixed(1)}%`;
        rows('ecoli-metrics', [
            ['Sample Metabolomic Class', ch.synthetic_class],
            ['Discriminant Score (not probability)', ch.decision_score.toFixed(3)],
            ['Chemometric Inversion', `${ch.technique} (in lieu of QML)`],
            ['Held-out Regularized / Linear Accuracy', `${percent(ch.accuracy)} / ${percent(ch.linear_baseline_accuracy)}`],
            ['Held-out Sensitivity / Specificity', `${percent(ch.sensitivity)} / ${percent(ch.specificity)}`],
            ['Held-out Confusion Counts', JSON.stringify(ch.confusion)]
        ]);
        byId('ecoli-qml-scope').textContent = ch.scope;
        rows('ecoli-bands', s.characteristics.map(b => [b.band, b.assignment]));

        // Plot 1H-NMR spectrum
        const nmrTarget = byId('ecoli-nmr') ? 'ecoli-nmr' : 'ecoli-raman';
        const ppmData = s.chemical_shift_ppm || s.raman_shift_cm_inverse;
        const intensityData = s.nmr_intensity_au || s.raman_au;
        plot(nmrTarget, [{name: '1H-NMR Chemical Shift (0-10 ppm)', points: ppmData.map((v, i) => [v, intensityData[i]])}], 'Chemical shift delta (ppm)', 'Intensity (a.u.)');

        // Plot CPMG / DOSY relaxation
        const relaxTarget = byId('ecoli-relaxation') ? 'ecoli-relaxation' : 'ecoli-uv';
        if (s.cpmg_time_ms && s.cpmg_intensity_au) {
            plot(relaxTarget, [
                {name: 'CPMG T2 Relaxation (ms)', points: s.cpmg_time_ms.map((v, i) => [v, s.cpmg_intensity_au[i]])},
                {name: 'DOSY Diffusion Attenuation', points: s.cpmg_time_ms.map((v, i) => [v, s.dosy_attenuation_au[i]])}
            ], 'Echo time tau (ms) / Normalized gradient', 'Normalized Attenuation (a.u.)');
        } else {
            plot(relaxTarget, [{name: 'UV-visible', points: s.absorbance_au.map((v, i) => [s.uv_wavelength_nm[i], v])}], 'Wavelength (nm)', 'Absorbance (a.u.)');
        }

        // Plot 72-hour bacterial burden trajectory
        plot('ecoli-decay', ['ecoli_cfu_100ml', 'low', 'high'].map((key, i) => ({
            name: ['Central', 'Lower scenario', 'Upper scenario'][i],
            points: r.sample_trajectory.map(v => [v.hour, v[key]])
        })), 'Sample time (hours)', 'Assumed burden (CFU/100 mL)');

        // Plot water sample chemical characteristics
        plot('ecoli-chemistry', ['nitrate_mg_l', 'phosphate_mg_l', 'organic_mg_l', 'turbidity_ntu'].map((key, i) => ({
            name: ['Nitrate (mg/L)', 'Phosphate (mg/L)', 'DOC (mg/L)', 'Turbidity (NTU)'][i],
            points: r.sample_trajectory.map(v => [v.hour, v[key]])
        })), 'Sample time (hours)', 'Assumed chemistry (mg/L) / turbidity (NTU)');

        // Plot 10-year remediation trajectory
        plot('ecoli-recovery', ['baseline_cfu_100ml', 'intervention_cfu_100ml'].map((key, i) => ({
            name: ['No change', 'Source control'][i], points: r.recovery.map(v => [v.year, v[key]])
        })), 'Elapsed year', 'Assumed burden (CFU/100 mL)');

        // Plot robotic SLAM
        plot('ecoli-slam', [
            {name: 'Synthetic truth', points: r.slam.truth_xy_m},
            {name: 'Joint SLAM estimate', points: r.slam.estimated_xy_m},
            {name: 'Odometry only', points: r.slam.dead_reckoning_xy_m},
            {name: 'Estimated landmarks', points: r.slam.estimated_landmarks_xy_m, markers: true}
        ], 'Local x (m)', 'Local y (m)', true);

        // Plot SLAM uncertainty
        plot('ecoli-uncertainty', [{name: 'RMS marginal position SD', points: r.slam.position_sigma_m.map((v, i) => [i, v])}],
            'Survey step', 'Position SD (m)');
        rows('ecoli-slam-metrics', [
            ['SLAM positional RMSE vs synthetic truth', `${r.slam.rmse_m.toFixed(3)} m`],
            ['Odometry-only positional RMSE', `${r.slam.dead_reckoning_rmse_m.toFixed(3)} m`]
        ]);
        byId('ecoli-slam-scope').textContent = r.slam.scope;
        byId('ecoli-trajectory-scope').textContent = r.trajectory_scope;
    }

    async function analyze() {
        run.disabled = true;
        pdf.disabled = true;
        form.querySelectorAll('input').forEach(input => { input.disabled = true; });
        status.textContent = 'Computing NMR spectra, CPMG T2 relaxation, chemometrics and SLAM...';
        try {
            const inputs = Object.fromEntries(Object.keys(fields).map(key => [key, Number(byId(`ecoli-${key}`).value)]));
            const response = await request('analyze', inputs);
            const result = await response.json();
            render(result.report);
            lastInputs = inputs;
            pdf.disabled = false;
            status.textContent = 'Analysis complete. Synthetic NMR research results only; not a water-safety or clinical assessment.';
        } catch (error) {
            status.textContent = `Analysis failed: ${error.message}. Previous results, if present, are not updated.`;
            console.error('E. coli analysis failed:', error);
        } finally {
            run.disabled = false;
            form.querySelectorAll('input').forEach(input => { input.disabled = false; });
        }
    }

    const openBtn = byId('open-ecoli-analysis');
    if (openBtn) {
        openBtn.addEventListener('click', () => {
            document.querySelector('[data-tab="tab-ecoli"]').click();
            byId('tab-ecoli').scrollIntoView({behavior: 'smooth', block: 'start'});
        });
    }
    form.addEventListener('submit', event => { event.preventDefault(); analyze(); });
    form.addEventListener('input', () => {
        pdf.disabled = true;
        status.textContent = 'Inputs changed. Run analysis to refresh results and PDF.';
    });
    pdf.addEventListener('click', async () => {
        pdf.disabled = true;
        run.disabled = true;
        form.querySelectorAll('input').forEach(input => { input.disabled = true; });
        status.textContent = 'Generating Nature-style NMR preprint PDF for the analyzed sample...';
        try {
            const response = await request('preprint', lastInputs);
            const url = URL.createObjectURL(await response.blob());
            const link = document.createElement('a');
            link.href = url;
            link.download = 'Nature_Style_Ontario_Ecoli_NMR_SLAM_Preprint.pdf';
            link.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
            status.textContent = 'Research NMR preprint PDF downloaded for the last analyzed inputs.';
        } catch (error) {
            status.textContent = `PDF generation failed: ${error.message}`;
            console.error('Spectroscopy PDF failed:', error);
        } finally {
            pdf.disabled = false;
            run.disabled = false;
            form.querySelectorAll('input').forEach(input => { input.disabled = false; });
        }
    });
    try {
        const response = await fetch('/api/ontario/ecoli/defaults');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        Object.entries(fields).forEach(([key, title]) => {
            const label = document.createElement('label');
            label.textContent = title;
            const input = document.createElement('input');
            input.id = `ecoli-${key}`;
            input.className = 'form-control';
            input.type = 'number';
            input.required = true;
            [input.min, input.max] = data.limits[key];
            input.step = key === 'seed' ? '1' : 'any';
            input.value = data.inputs[key];
            label.append(input);
            byId('ecoli-inputs').append(label);
        });
        await analyze();
    } catch (error) {
        status.textContent = `Unable to initialize spectroscopy: ${error.message}. Reload to retry.`;
        console.error('Spectroscopy initialization failed:', error);
    }
});
