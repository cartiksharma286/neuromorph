// Main logic for Schizoaffective Disorder Modeling & QML Intervention
let chart;

function format(num) {
    return parseFloat(num).toFixed(2);
}

function initChart() {
    const ctx = document.getElementById('distribution-chart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Symptom Severity PDF',
                data: [],
                borderColor: '#00f2ff',
                backgroundColor: 'rgba(0, 242, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                pointRadius: 0,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { title: { display: true, text: 'Clinical Symptom Axis (− Stabilized, + Psychotic)', color: '#a0aab5' }, ticks: { display: false } },
                y: { title: { display: true, text: 'Probability Density', color: '#a0aab5' }, ticks: { display: false }, min: 0 }
            },
            animation: {
                duration: 400
            }
        }
    });
}

function drawBrainNodes(nodes) {
    const container = document.getElementById('brain-container');
    container.innerHTML = '';
    
    // Abstract coordinates mapped to percentage container
    const mapping = {
        'Prefrontal Cortex (PFC)': { x: 20, y: 30 },
        'Amygdala': { x: 70, y: 70 },
        'Hippocampus': { x: 60, y: 65 },
        'Striatum': { x: 50, y: 50 }
    };
    
    nodes.forEach(n => {
        let el = document.createElement('div');
        el.className = 'node';
        
        let pos = mapping[n.id];
        el.style.left = `${pos.x}%`;
        el.style.top = `${pos.y}%`;
        
        // Size and brightness based on neural activity 'val'
        let s = n.val * 30;
        let b = Math.min(255, Math.floor(n.val * 150));
        el.style.width = `${s}px`;
        el.style.height = `${s}px`;
        if (n.id === 'Prefrontal Cortex (PFC)') {
            // Inhibitory (Blue/Cyan proxy)
            el.style.background = `radial-gradient(circle, rgba(0,242,255,${n.val*0.8}) 0%, transparent 70%)`;
        } else {
            // Excitatory (Pink/Red proxy)
            el.style.background = `radial-gradient(circle, rgba(255,0,127,${n.val/2}) 0%, transparent 70%)`;
        }
        
        container.appendChild(el);
    });
}

async function runSimulation() {
    const beer = document.getElementById('beer-slider').value;
    const medication = document.getElementById('medication-slider').value;
    const nicotine = document.getElementById('nicotine-slider').value;
    const polymath_load = document.getElementById('polymath-slider').value;
    const legal_trouble = document.getElementById('legal-slider').value;
    
    const response = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ beer, medication, nicotine, polymath_load, legal_trouble })
    });
    
    const data = await response.json();
    
    // Updates
    document.getElementById('dopamine-var').innerText = format(data.dopamine_variance);
    document.getElementById('symptom-score').innerText = format(data.mean_symptom_score);
    document.getElementById('tms-freq').innerText = `${format(data.opt_tms_freq)} Hz`;
    document.getElementById('treatment-yield').innerText = `${format(data.treatment_yield)}%`;

    document.getElementById('hebbian-amp').innerText = `${format(data.hebbian_amplification)}%`;
    document.getElementById('cortical-control').innerText = format(data.cortical_control_index);
    
    // Highlight relapse prob dynamically
    const rprob = document.getElementById('relapse-prob');
    rprob.innerText = `${format(data.relapse_probability)}%`;
    if(data.relapse_probability > 40) rprob.parentElement.style.borderColor = '#ff0055';
    else if(data.relapse_probability > 20) rprob.parentElement.style.borderColor = '#ffaa00';
    else rprob.parentElement.style.borderColor = '#00ff88';

    // CBT Traits and Feedback processing
    document.getElementById('cbt-feedback').innerHTML = `<span style="color: #a0aab5;">Feedback Correlate:</span> ${data.feedback_correlate}`;
    document.getElementById('cbt-traits-list').innerHTML = data.cbt_traits.map(t => 
        `<div><span style="color: #00ff88;">✓</span> ${t}</div>`
    ).join('');

    // Chart update
    chart.data.labels = data.stats.x;
    chart.data.datasets[0].data = data.stats.y;
    chart.update();
    
    // Brain Nodes
    drawBrainNodes(data.neural_nodes);
    
    // Regional Activity list
    const rDiv = document.getElementById('regions');
    rDiv.innerHTML = data.neural_nodes.map(n => `
        <div style="display:flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 5px;">
            <span>${n.id}</span>
            <span style="color: ${n.val > 1.2 ? '#ff007f' : '#00f2ff'}">${format(n.val)} Units</span>
        </div>
    `).join('');
}

// Bindings
document.getElementById('beer-slider').addEventListener('input', (e) => {
    document.getElementById('beer-val').innerText = format(e.target.value);
    runSimulation();
});

document.getElementById('medication-slider').addEventListener('input', (e) => {
    document.getElementById('medication-val').innerText = format(e.target.value);
    runSimulation();
});

document.getElementById('nicotine-slider').addEventListener('input', (e) => {
    document.getElementById('nicotine-val').innerText = format(e.target.value);
    runSimulation();
});

document.getElementById('polymath-slider').addEventListener('input', (e) => {
    document.getElementById('polymath-val').innerText = format(e.target.value);
    runSimulation();
});

document.getElementById('legal-slider').addEventListener('input', (e) => {
    document.getElementById('legal-val').innerText = format(e.target.value);
    runSimulation();
});

window.onload = () => {
    initChart();
    runSimulation();
};