function showSection(id) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.getElementById(id).classList.add('active');

    document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
    document.querySelector(`button[onclick="showSection('${id}')"]`).classList.add('active');
}

async function runOptimization() {
    const output = document.getElementById('optimization-result');
    output.innerText = "Optimizing...";
    try {
        const res = await fetch('/api/optimize');
        const data = await res.json();
        output.innerText = JSON.stringify(data.result, null, 2);
    } catch (e) {
        output.innerText = "Error: " + e.message;
    }
}

async function fetchGeometry() {
    const output = document.getElementById('geometry-preview');
    output.innerText = "Fetching geometry...";
    try {
        // Toggle between GenAI and NVQLink for demo
        const type = Math.random() > 0.5 ? 'genai' : 'nvqlink';
        const res = await fetch(`/api/geometry/${type}`);
        const data = await res.json();
        output.innerText = `${data.type}:\n` + JSON.stringify(data.data, null, 2);
    } catch (e) {
        output.innerText = "Error: " + e.message;
    }
}

async function loadEconomics() {
    const output = document.getElementById('economics-data');
    try {
        const res = await fetch('/api/economics');
        const data = await res.json();
        output.innerText = JSON.stringify(data.data, null, 2);
    } catch (e) {
        output.innerText = "Error loading economics.";
    }
}

// Load initial data
loadEconomics();
