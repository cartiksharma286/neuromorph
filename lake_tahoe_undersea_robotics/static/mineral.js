/**
 * Benthic Mineral Ore Excavation & QML Fermionic Parity Client Application
 * Handles simulation of subsea mineral ore perception (Fe, U, REE, Ag, Co),
 * hyperspectral reflectance plotting, Jordan-Wigner fermionic parity curves,
 * 10-year discounted cash flow trajectories, and Nature preprint download.
 */

document.addEventListener('DOMContentLoaded', async () => {
    const byId = id => document.getElementById(id);
    const form = byId('mineral-form');
    const status = byId('mineral-status');
    const runBtn = byId('btn-run-mineral');
    const pdfBtn = byId('btn-mineral-pdf');
    const depositSelect = byId('mineral-deposit-select');
    let lastInputs = null;

    const fields = {
        excavation_rate_tpd: 'Excavation Rate (Tonnes/Day)',
        operating_days_year: 'Operating Days/Year',
        capital_capex_usd_m: 'Fleet CapEx ($M USD)',
        operating_opex_per_tonne: 'OpEx ($/Tonne Ore)',
        environmental_levy_pct: 'Environmental Levy Fraction',
        discount_rate: 'Real Discount Rate',
        sensor_turbidity_ntu: 'Water Turbidity (NTU)',
        pqc_depth: 'PQC Variational Depth',
        seed: 'Simulation Random Seed'
    };

    const palette = ['#06b6d4', '#fbbf24', '#c084fc', '#10b981', '#f43f5e', '#38bdf8'];

    async function request(path, inputs) {
        const response = await fetch(`/api/mineral/${path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputs)
        });
        if (!response.ok) {
            const type = response.headers.get('content-type') || '';
            const message = type.includes('application/json') ? (await response.json()).message : `HTTP ${response.status}`;
            throw new Error(message);
        }
        return response;
    }

    function plot(id, series, xlabel, ylabel) {
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
            Object.entries(attributes).forEach(([k, v]) => node.setAttribute(k, String(v)));
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

        const x = v => 65 + (v - xmin) / (xmax - xmin) * 510;
        const y = v => 225 - (v - ymin) / (ymax - ymin) * 200;

        add('title', {}, `${ylabel} against ${xlabel}`);
        for (let i = 0; i <= 4; i++) {
            const yy = ymin + (ymax - ymin) * i / 4;
            const xx = xmin + (xmax - xmin) * i / 4;
            add('line', { x1: 65, x2: 575, y1: y(yy), y2: y(yy), stroke: '#334155' });
            add('text', { x: 58, y: y(yy) + 4, 'text-anchor': 'end' }, yy.toFixed(1));
            add('text', { x: x(xx), y: 245, 'text-anchor': 'middle' }, xx.toFixed(1));
        }

        add('text', { x: 65, y: 15 }, ylabel);
        add('text', { x: 310, y: 266, 'text-anchor': 'middle'}, xlabel);

        series.forEach((s, i) => {
            const color = palette[i % palette.length];
            add('polyline', {
                points: s.points.map(p => `${x(p[0])},${y(p[1])}`).join(' '),
                fill: 'none', stroke: color, 'stroke-width': 2
            });
            add('text', { x: 65 + (i % 3) * 175, y: 285 + Math.floor(i / 3) * 16, fill: color, 'font-size': '11px' }, s.name);
        });

        target.append(svg);
    }

    function renderGrades(dep) {
        const target = byId('mineral-grades-readout');
        if (!target) return;
        target.replaceChildren();

        const grades = [
            { label: 'Iron (Fe)', value: `${dep.iron_grade_pct.toFixed(1)} wt%`, color: '#06b6d4' },
            { label: 'Uranium (U)', value: `${dep.uranium_grade_ppm.toFixed(0)} ppm`, color: '#fbbf24' },
            { label: 'Rare Earths', value: `${dep.ree_grade_ppm.toFixed(0)} ppm`, color: '#c084fc' },
            { label: 'Silver (Ag)', value: `${dep.silver_grade_g_tonne.toFixed(0)} g/t`, color: '#10b981' },
            { label: 'Cobalt (Co)', value: `${dep.cobalt_grade_pct.toFixed(2)} wt%`, color: '#f43f5e' },
        ];

        grades.forEach(g => {
            const div = document.createElement('div');
            div.style.padding = '4px 8px';
            div.style.background = 'rgba(15, 23, 42, 0.7)';
            div.style.borderRadius = '4px';
            div.style.borderLeft = `3px solid ${g.color}`;
            div.innerHTML = `<span style="color:var(--text-secondary);">${g.label}:</span> <b>${g.value}</b>`;
            target.append(div);
        });
    }

    function render(report) {
        const dep = report.deposit_preset;
        const econ = report.economics;
        const qml = report.qml_parity;
        const spec = report.spectroscopy;
        const rev = econ.annual_revenue_by_commodity_usd;

        // Telemetry readouts
        byId('min-gross-rev').textContent = `$${(econ.annual_gross_revenue_usd / 1e6).toFixed(1)}M / yr`;
        byId('min-npv').textContent = `$${(econ.ten_year_discounted_npv_usd / 1e6).toFixed(1)}M`;
        byId('min-parity').textContent = `${qml.parity_expectation.toFixed(3)} (${qml.parity_expectation >= 0 ? '+1' : '-1'})`;
        byId('min-confidence').textContent = `${(qml.qml_confidence * 100).toFixed(1)}%`;

        byId('min-ore-rate').textContent = `${(report.telemetry.excavation_rate_tph).toFixed(1)} t/h`;
        byId('min-annual-ore').textContent = `${(econ.annual_ore_excavated_tonnes).toLocaleString()} t/yr`;
        byId('min-payback').textContent = econ.payback_period_years ? `${econ.payback_period_years} yrs` : '>10 yrs';
        byId('min-irr').textContent = econ.internal_rate_of_return_irr ? `${(econ.internal_rate_of_return_irr * 100).toFixed(1)}%` : '--';
        byId('min-env-levy').textContent = `$${(econ.annual_environmental_remediation_usd / 1e6).toFixed(1)}M/yr`;

        // Render ore grades badges
        renderGrades(dep);

        // SVG Plots
        // 1. Hyperspectral Optical Reflectance (400 - 1000 nm)
        plot('min-plot-spectral', [
            { name: 'Fe (890nm)', points: spec.wavelengths_nm.map((w, i) => [w, spec.iron_spectrum[i]]) },
            { name: 'U (530nm)', points: spec.wavelengths_nm.map((w, i) => [w, spec.uranium_spectrum[i]]) },
            { name: 'REE (Nd/Dy)', points: spec.wavelengths_nm.map((w, i) => [w, spec.ree_spectrum[i]]) },
            { name: 'Silver (Ag)', points: spec.wavelengths_nm.map((w, i) => [w, spec.silver_spectrum[i]]) },
            { name: 'Cobalt (Co)', points: spec.wavelengths_nm.map((w, i) => [w, spec.cobalt_spectrum[i]]) },
            { name: 'Composite Ore', points: spec.wavelengths_nm.map((w, i) => [w, spec.composite_ore_spectrum[i]]) },
        ], 'Wavelength lambda (nm)', 'Spectral Reflectance rho(lambda)');

        // 2. Fermionic Parity Operator Expectation <P> vs Variational Circuit Depth
        plot('min-plot-parity', [
            {
                name: 'Parity Expectation <P>',
                points: qml.depth_axis.map((d, i) => [d, qml.depth_parity_sweep[i]])
            }
        ], 'Variational Circuit Depth (Layers)', 'Fermionic Parity <P>');

        // 3. 10-Year Cumulative Discounted NPV Trajectory ($M USD)
        plot('min-plot-npv', [
            {
                name: 'Cumulative Discounted NPV ($M)',
                points: econ.trajectory.map(row => [row.year, row.cumulative_discounted_npv_usd / 1e6])
            }
        ], 'Operational Model Year (t = 0 to 10)', 'Discounted NPV ($M USD)');

        // 4. Annual Gross Revenue Breakdown by Commodity ($M USD)
        const years = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        plot('min-plot-revenue', [
            { name: 'Iron ($M)', points: years.map(y => [y, rev.iron / 1e6 * (1 + 0.015 * (y - 1))]) },
            { name: 'Cobalt ($M)', points: years.map(y => [y, rev.cobalt / 1e6 * (1 + 0.015 * (y - 1))]) },
            { name: 'Rare Earths ($M)', points: years.map(y => [y, rev.ree / 1e6 * (1 + 0.015 * (y - 1))]) },
            { name: 'Silver ($M)', points: years.map(y => [y, rev.silver / 1e6 * (1 + 0.015 * (y - 1))]) },
            { name: 'Uranium ($M)', points: years.map(y => [y, rev.uranium / 1e6 * (1 + 0.015 * (y - 1))]) },
        ], 'Operational Model Year', 'Commodity Revenue ($M USD)');
    }

    async function simulate() {
        runBtn.disabled = true;
        pdfBtn.disabled = true;
        status.textContent = 'Simulating mineral spectroscopy, Jordan-Wigner fermionic parity, and 10-year revenue...';

        try {
            const inputs = Object.fromEntries(
                Object.keys(fields).map(key => [key, Number(byId(`min-${key}`).value)])
            );
            inputs.chosen_deposit = depositSelect.value;
            lastInputs = inputs;

            const response = await request('simulate', inputs);
            const result = await response.json();
            render(result.report);
            pdfBtn.disabled = false;
            status.textContent = 'Benthic excavation simulation complete. Multi-commodity reserves and cash flows updated.';
        } catch (err) {
            status.textContent = `Simulation failed: ${err.message}`;
            console.error('Mineral simulation error:', err);
        } finally {
            runBtn.disabled = false;
        }
    }

    // Form submission
    form.addEventListener('submit', event => {
        event.preventDefault();
        simulate();
    });

    depositSelect.addEventListener('change', () => {
        simulate();
    });

    // PDF Download
    pdfBtn.addEventListener('click', async () => {
        pdfBtn.disabled = true;
        status.textContent = 'Generating Nature-style mineral excavation preprint PDF...';
        try {
            const response = await request('preprint', lastInputs || {});
            const url = URL.createObjectURL(await response.blob());
            const link = document.createElement('a');
            link.href = url;
            link.download = 'Nature_Benthic_Mineral_Excavation_QML_Parity_Preprint.pdf';
            link.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
            status.textContent = 'Nature mineral preprint PDF downloaded successfully.';
        } catch (err) {
            status.textContent = `PDF generation failed: ${err.message}`;
            console.error('Preprint error:', err);
        } finally {
            pdfBtn.disabled = false;
        }
    });

    // Initialize defaults from backend
    try {
        const response = await fetch('/api/mineral/defaults');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        const grid = byId('mineral-inputs-grid');
        grid.replaceChildren();

        Object.entries(fields).forEach(([key, title]) => {
            const label = document.createElement('label');
            label.textContent = title;
            const input = document.createElement('input');
            input.id = `min-${key}`;
            input.className = 'form-control';
            input.type = 'number';
            input.required = true;
            [input.min, input.max] = data.limits[key];
            input.step = key in ['operating_days_year', 'pqc_depth', 'seed'] ? '1' : 'any';
            input.value = data.inputs[key];
            label.append(input);
            grid.append(label);
        });

        if (data.inputs.chosen_deposit) {
            depositSelect.value = data.inputs.chosen_deposit;
        }

        await simulate();
    } catch (err) {
        status.textContent = `Initialization failed: ${err.message}. Please reload.`;
        console.error('Mineral init error:', err);
    }
});
