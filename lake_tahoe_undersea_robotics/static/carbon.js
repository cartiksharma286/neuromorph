const carbonLabels = {
    fleet: 'Robots (count)',
    robot_capex: 'Capital per robot (CAD)',
    annual_opex: 'Annual non-energy cost per robot (CAD)',
    hours: 'Operating hours per year',
    robot_kw: 'Robot power (kW)',
    qml_kw: 'Shared QML / cryogenic facility power (kW)',
    qml_capex: 'QML facility capital (CAD)',
    qml_embodied: 'QML embodied footprint (kg CO2e)',
    qml_reduction: 'Assumed robot energy savings (fraction, 0-0.5)',
    electricity_price: 'Electricity price (CAD/kWh)',
    grid_kg_kwh: 'Grid emissions (kg CO2e/kWh)',
    embodied_kg: 'Embodied footprint per robot (kg CO2e)',
    baseline_cost: 'Conventional monitoring annual cost (CAD)',
    baseline_kg: 'Conventional monitoring annual footprint (kg CO2e)',
    annual_benefit: 'Additional non-carbon annual benefit (CAD)',
    carbon_price: 'Carbon shadow price (CAD/tonne CO2e)',
    discount: 'Annual discount rate (fraction)',
    prime: 'Scheduling prime (2, 3, 5, 7, 11)'
};
let signedCarbonEnvelope = null;
const carbonCharts = {};

function carbonInputs() {
    const form = document.getElementById('carbon-form');
    if (!form.reportValidity()) throw new Error('Correct the highlighted assumptions.');
    return Object.fromEntries(Object.keys(carbonLabels).map(name =>
        [name, Number(document.getElementById(`carbon-${name}`).value)]));
}

async function carbonPost(path, data) {
    const response = await fetch(path, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!response.ok) {
        const contentType = response.headers.get('content-type') || '';
        const message = contentType.includes('application/json')
            ? (await response.json()).message : `Request failed (${response.status}).`;
        throw new Error(message || `Request failed (${response.status}).`);
    }
    return response;
}

function carbonChart(id, labels, datasets) {
    if (typeof Chart === 'undefined') throw new Error('Chart.js is unavailable; charts cannot be rendered.');
    if (carbonCharts[id]) carbonCharts[id].destroy();
    carbonCharts[id] = new Chart(document.getElementById(id), {
        type: 'line',
        data: { labels, datasets: datasets.map(([label, data], i) => ({
            label, data, borderColor: i ? '#fbbf24' : '#06b6d4',
            backgroundColor: 'transparent', tension: 0
        })) },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#f8fafc' } } },
            scales: {
                x: { ticks: { color: '#94a3b8' }, title: { display: true, text: 'Elapsed model year / multiplier', color: '#94a3b8' } },
                y: { ticks: { color: '#94a3b8' } }
            }
        }
    });
}

function renderCarbonReport(report) {
    const results = document.getElementById('carbon-results');
    results.replaceChildren();
    const metrics = [
        ['Financial NPV (CAD)', report.financial_npv_cad],
        ['Carbon shadow NPV (CAD; not credit revenue)', report.carbon_valued_npv_cad],
        ['Annual electricity (kWh)', report.annual_kwh],
        ['Net avoided footprint, 10 years (kg CO2e)', report.net_avoided_kg_10yr],
        ['Carbon break-even (years)', report.carbon_break_even_years],
        ['Joint viability under assumptions', report.viable_under_assumptions ? 'Yes' : 'No']
    ];
    for (const [label, value] of metrics) {
        const item = document.createElement('p');
        item.textContent = `${label}: ${value === null ? 'No finite break-even' :
            typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value}`;
        results.append(item);
    }
    document.getElementById('carbon-qml').textContent =
        `Synthetic kernel score: ${report.qml.synthetic_score.toFixed(4)}; training RMSE: ${report.qml.training_rmse.toFixed(4)}. No demonstrated advantage.`;
    document.getElementById('carbon-number-theory').textContent =
        `Scheduling prime p=${report.number_theory.prime}; station residues: ` +
        report.number_theory.station_groups.map(v => `${v.station_id}→${v.residue}`).join(', ');
    const years = report.trajectory.map(v => v.year);
    carbonChart('carbon-npv-chart', years, [
        ['Financial NPV', report.trajectory.map(v => v.npv_cad)],
        ['Carbon shadow NPV', report.trajectory.map(v => v.carbon_valued_npv_cad)]
    ]);
    carbonChart('carbon-emissions-chart', years, [
        ['Robotics incl. embodied', report.trajectory.map(v => v.project_kg)],
        ['Conventional baseline', report.trajectory.map(v => v.baseline_kg)]
    ]);
    carbonChart('carbon-sensitivity-chart', report.capex_sensitivity.map(v => `${v.capex_multiplier}x`), [
        ['Financial NPV', report.capex_sensitivity.map(v => v.npv_cad)]
    ]);
}

function carbonDownload(blob, name) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function calculateCarbon() {
    const status = document.getElementById('carbon-status');
    status.textContent = 'Calculating and signing...';
    signedCarbonEnvelope = null;
    try {
        const response = await carbonPost('/api/carbon/evaluate', carbonInputs());
        const data = await response.json();
        signedCarbonEnvelope = data.envelope;
        renderCarbonReport(data.report);
        status.textContent = 'Scenario calculated and signed. Inputs are assumptions, not verified measurements.';
        document.getElementById('pqc-status').textContent = 'A signed scenario snapshot is ready for verification.';
    } catch (error) {
        status.textContent = `Error: ${error.message}`;
        console.error('Carbon scenario failed:', error);
    }
}

async function pqcAction(action) {
    const status = document.getElementById('pqc-status');
    status.textContent = 'Verifying...';
    try {
        if (action === 'self-test') {
            const response = await carbonPost('/api/carbon/pqc/self-test', carbonInputs());
            const data = await response.json();
            document.getElementById('pqc-results').textContent = JSON.stringify(data.verification, null, 2);
            status.textContent = 'ML-DSA and ML-KEM functional tests passed; HTTP is not protected.';
            return;
        }
        if (!signedCarbonEnvelope) throw new Error('Calculate a scenario first.');
        if (action === 'download') {
            carbonDownload(new Blob([JSON.stringify(signedCarbonEnvelope, null, 2)],
                { type: 'application/json' }), 'Ontario_Signed_Carbon_Report.json');
            status.textContent = 'Signed snapshot downloaded; key expires on server restart.';
            return;
        }
        const envelope = structuredClone(signedCarbonEnvelope);
        if (action === 'tamper') envelope.report.capital_cad += 1;
        const response = await carbonPost('/api/carbon/pqc/verify', envelope);
        const data = await response.json();
        if (data.signature_valid !== (action !== 'tamper')) throw new Error('Unexpected verification result.');
        status.textContent = action === 'tamper'
            ? 'PASS: changing the report invalidated the signature.'
            : 'PASS: latest signed snapshot matches the current server key.';
        document.getElementById('pqc-results').textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        status.textContent = `Error: ${error.message}`;
        console.error('PQC verification failed:', error);
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    const form = document.getElementById('carbon-form');
    form.addEventListener('submit', event => {
        event.preventDefault();
        calculateCarbon();
    });
    form.addEventListener('input', () => {
        document.getElementById('carbon-status').textContent = 'Inputs changed: recalculate to update results and signed snapshot.';
        document.getElementById('pqc-status').textContent = 'Verification uses the last calculated snapshot, not unsaved inputs.';
    });
    for (const action of ['self-test', 'verify', 'tamper', 'download']) {
        document.getElementById(`pqc-${action}`).addEventListener('click', () => pqcAction(action));
    }
    document.getElementById('carbon-pdf').addEventListener('click', async () => {
        const status = document.getElementById('carbon-status');
        try {
            status.textContent = 'Generating PDF for current inputs...';
            const response = await carbonPost('/api/carbon/preprint', carbonInputs());
            carbonDownload(await response.blob(), 'Ontario_Carbon_Economics_PQC_Research_Preprint.pdf');
            status.textContent = 'PDF generated for current inputs; existing charts remain the last calculated snapshot.';
        } catch (error) {
            status.textContent = `PDF error: ${error.message}`;
            console.error('Scenario PDF failed:', error);
        }
    });
    try {
        const response = await fetch('/api/carbon/defaults');
        if (!response.ok) throw new Error(`Defaults request failed (${response.status}).`);
        const data = await response.json();
        const inputs = document.getElementById('carbon-inputs');
        for (const [name, label] of Object.entries(carbonLabels)) {
            const wrapper = document.createElement('div');
            wrapper.className = 'form-group';
            const title = document.createElement('label');
            title.htmlFor = `carbon-${name}`;
            title.textContent = label;
            const input = document.createElement('input');
            input.id = `carbon-${name}`;
            input.className = 'form-control';
            input.type = 'number';
            input.required = true;
            [input.min, input.max] = data.limits[name];
            input.step = ['fleet', 'prime'].includes(name) ? '1' : 'any';
            input.value = data.assumptions[name];
            wrapper.append(title, input);
            inputs.append(wrapper);
        }
        await calculateCarbon();
    } catch (error) {
        document.getElementById('carbon-status').textContent = `Initialization error: ${error.message}`;
        console.error('Carbon initialization failed:', error);
    }
});
