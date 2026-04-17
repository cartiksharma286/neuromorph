with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'r') as f:
    js = f.read()

dementia_js = """
// --- DEMENTIA STAGING GENERATIVE PROTOCOL ---

let dementiaChart = null;

async function runDementiaStaging() {
    const prompt = document.getElementById('gen-ai-prompt').value;
    const declineRate = parseFloat(document.getElementById('decline-range').value);
    const dbsAmp = parseFloat(document.getElementById('dementia-dbs-amp').value);

    // Call backend endpoint
    const response = await fetch('/api/dementia-staging', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt, decline_rate: declineRate, dbs_amplitude: dbsAmp })
    });
    
    if (!response.ok) return;
    const res = await response.json();
    
    // Update text
    document.getElementById('dementia-insight').innerText = res.generative_insight;
    
    // Render Chart
    const ctx = document.getElementById('dementia-chart');
    if (!ctx) return;
    if (dementiaChart) dementiaChart.destroy();
    
    const times = res.time_months;
    const traj = res.cognitive_trajectory;
    // Map variance
    const upper_bound = res.clinical_distributions.map(d => d.mean + d.std/1.5);
    const lower_bound = res.clinical_distributions.map(d => Math.max(0, d.mean - d.std/1.5));

    dementiaChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: times.map(t => Math.round(t)),
            datasets: [
                {
                    label: 'Clinical Variance (Upper)',
                    data: upper_bound,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(0, 242, 255, 0.2)',
                    fill: '+1',
                    pointRadius: 0,
                    tension: 0.4
                },
                {
                    label: 'Clinical Variance (Lower)',
                    data: lower_bound,
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    fill: false,
                    pointRadius: 0,
                    tension: 0.4
                },
                {
                    label: 'Temporal Cognitive Trajectory',
                    data: traj,
                    borderColor: '#ff00c8',
                    borderWidth: 3,
                    fill: false,
                    pointRadius: 1,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, labels: { color: '#a0aab5' } }
            },
            scales: {
                x: { title: { display: true, text: 'Treatment Timeline (Months)', color: '#00f2ff'}, ticks: { color: '#a0aab5' } },
                y: { title: { display: true, text: 'Structural Neurological Retention (Score)', color: '#00f2ff'}, ticks: { color: '#a0aab5' }, min: 0, max: 35 }
            }
        }
    });
}
"""
js += dementia_js

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js', 'w') as f:
    f.write(js)
