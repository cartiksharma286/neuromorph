const simForm = document.getElementById('sim-form');
const ctx = document.getElementById('kineticsChart').getContext('2d');
let kineticsChart = null;

// Dynamic sliders
const bindSlider = (id, target) => {
    document.getElementById(id).addEventListener('input', (e) => {
        document.getElementById(target).innerText = parseFloat(e.target.value).toFixed(2);
    });
};
bindSlider('days', 'val-days');
bindSlider('start_plaque', 'val-plaque');
bindSlider('dose_mg', 'val-dose');
bindSlider('affinity', 'val-affinity');
bindSlider('clearance', 'val-clear');

simForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const body = {
        days: parseInt(document.getElementById('days').value),
        start_plaque: parseFloat(document.getElementById('start_plaque').value),
        dose_mg: parseFloat(document.getElementById('dose_mg').value),
        affinity: parseFloat(document.getElementById('affinity').value),
        clearance: parseFloat(document.getElementById('clearance').value)
    };

    try {
        const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        const json = await response.json();
        
        if(json.status === "success") {
            const data = json.data;
            plotData(data.time, data.amyloid_beta, data.aducanumab, data.immune_complex);
            
            const finalBurden = data.amyloid_beta[data.amyloid_beta.length - 1];
            document.getElementById('final-burden').innerText = finalBurden.toFixed(2) + " mg";
        }
    } catch(err) {
        console.error(err);
    }
});

function plotData(time, plaque, drug, complex) {
    if(kineticsChart) kineticsChart.destroy();

    kineticsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: time.map(t => Math.round(t)),
            datasets: [
                {
                    label: 'Aβ Plaque Burden (mg)',
                    data: plaque,
                    borderColor: '#ff4d4d',
                    borderWidth: 3,
                    tension: 0.1,
                    pointRadius: 0
                },
                {
                    label: 'Free Aducanumab [D] (mg)',
                    data: drug,
                    borderColor: '#58a6ff',
                    borderWidth: 2,
                    tension: 0.1,
                    pointRadius: 0
                },
                {
                    label: 'Aβ-Drug Complex (Cleared)',
                    data: complex,
                    borderColor: '#76b900',
                    borderWidth: 2,
                    tension: 0.1,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'Time (Days)', color: '#8b949e' } },
                y: { title: { display: true, text: 'Mass (mg) / Relative Conc.', color: '#8b949e' }, min: 0 }
            },
            plugins: {
                legend: { labels: { color: '#c9d1d9' } }
            }
        }
    });
}
