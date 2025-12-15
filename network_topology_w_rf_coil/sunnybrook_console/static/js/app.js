let width, height;
let simulation;
let coreFreqChart, memoryChart, costChart;

document.addEventListener("DOMContentLoaded", () => {
    initClock();
    initCharts();

    // Default view
    switchTab('dashboard');

    document.getElementById("btn-rotate").addEventListener("click", rotateKeys);
    document.getElementById("btn-lockdown").addEventListener("click", initiateLockdown);

    // Poll for status updates
    setInterval(updateAllData, 1000);
});

function initClock() {
    setInterval(() => {
        const now = new Date();
        document.getElementById('clock').innerText = now.toLocaleTimeString();
    }, 1000);
}

function switchTab(tabId) {
    // Hide all views
    document.querySelectorAll('.view-section').forEach(el => el.style.display = 'none');
    // Show selected
    document.getElementById(`view-${tabId}`).style.display = 'block';

    // Update Sidebar Active State
    document.querySelectorAll('.nav-links li').forEach(li => li.classList.remove('active'));
    // Ideally map tabId to index, simple hack here:
    // This assumes order: Dashboard, Topology, Cluster, Cost, Security, Coil
    const map = { 'dashboard': 0, 'topology': 1, 'cluster': 2, 'cost': 3, 'security': 4, 'coil': 5 };
    document.querySelectorAll('.nav-links li')[map[tabId]].classList.add('active');

    // Update Title
    const titles = {
        'dashboard': 'Network Dashboard',
        'topology': 'Network Topology',
        'cluster': 'Cluster Schematics',
        'cost': 'Cost Optimization & Billing',
        'security': 'Security Operations',
        'coil': 'NVQLink Generator'
    };
    document.getElementById('page-title').innerText = titles[tabId];

    // Lazy load topology if needed
    if (tabId === 'topology') {
        renderGraph();
    }
}

async function updateAllData() {
    await fetchStatus();
    await fetchPerformance();

    // If cluster view is active, update cluster
    if (document.getElementById('view-cluster').style.display !== 'none') {
        fetchCluster();
    }
    // If cost view is active, update cost
    if (document.getElementById('view-cost').style.display !== 'none') {
        fetchCosts();
    }
}

async function fetchCosts() {
    try {
        const res = await fetch('/api/costs');
        const data = await res.json();

        document.getElementById('cost-total').innerText = data.current_monthly_total.toFixed(2);
        document.getElementById('cost-savings').innerText = data.savings.toFixed(2);
        document.getElementById('spot-usage').innerText = data.spot_instance_usage;

        // Update Chart
        const labels = Object.keys(data.department_breakdown);
        const values = Object.values(data.department_breakdown);

        costChart.data.labels = labels;
        costChart.data.datasets[0].data = values;
        costChart.update();

    } catch (e) { console.error(e); }
}

async function fetchStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        const healthEl = document.getElementById("system-health");
        healthEl.textContent = data.network_health;
        healthEl.style.color = data.network_health === "Optimal" ? "var(--success)" : "var(--danger)";

        // Log alerts
        const log = document.getElementById("event-log");
        if (data.alerts && data.alerts.length > 0) {
            log.innerHTML = "";
            [...data.alerts].reverse().forEach(a => {
                const li = document.createElement("li");
                li.innerText = `[${new Date(a.timestamp * 1000).toLocaleTimeString()}] ${a.msg}`;
                log.appendChild(li);
            });
        }

    } catch (e) { console.error(e); }
}

async function fetchPerformance() {
    try {
        const res = await fetch('/api/performance');
        const data = await res.json();

        document.getElementById('val-core-freq').innerText = `${data.core_frequency} GHz`;
        document.getElementById('val-memory').innerText = `${data.memory_usage} GB`;
        document.getElementById('val-threads').innerText = data.active_threads;

        // Update Charts
        updateChart(coreFreqChart, data.core_frequency);
        updateChart(memoryChart, data.memory_usage);

    } catch (e) { console.error(e); }
}

async function fetchCluster() {
    try {
        const res = await fetch('/api/cluster');
        const racks = await res.json();
        const container = document.getElementById('cluster-container');
        container.innerHTML = ''; // Re-render for simplicity (could optimize)

        racks.forEach((rack) => {
            const rackEl = document.createElement('div');
            rackEl.className = 'rack';

            const title = document.createElement('div');
            title.className = 'rack-title';
            title.innerText = rack.name;
            rackEl.appendChild(title);

            rack.units.forEach(u => {
                const unitEl = document.createElement('div');
                unitEl.className = `rack-unit ${u.status === 'active' ? 'unit-active' : 'unit-warning'}`;
                unitEl.innerText = `${u.id} | ${u.temp}°C`;
                rackEl.appendChild(unitEl);
            });
            container.appendChild(rackEl);
        });
    } catch (e) { console.error(e); }
}


function initCharts() {
    const ctx1 = document.getElementById('coreFreqChart').getContext('2d');
    coreFreqChart = new Chart(ctx1, {
        type: 'line',
        data: {
            labels: Array(20).fill(''),
            datasets: [{
                label: 'Frequency (GHz)',
                data: Array(20).fill(3.4),
                borderColor: '#38bdf8',
                tension: 0.4,
                borderWidth: 2,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            animation: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { min: 2.0, max: 4.5, grid: { color: '#334155' } },
                x: { display: false }
            }
        }
    });

    const ctx2 = document.getElementById('memoryChart').getContext('2d');
    memoryChart = new Chart(ctx2, {
        type: 'line',
        data: {
            labels: Array(20).fill(''),
            datasets: [{
                label: 'Memory (GB)',
                data: Array(20).fill(45),
                borderColor: '#22c55e',
                tension: 0.4,
                borderWidth: 2,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            animation: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { min: 30, max: 60, grid: { color: '#334155' } },
                x: { display: false }
            }
        }
    });

    const ctx3 = document.getElementById('costChart').getContext('2d');
    costChart = new Chart(ctx3, {
        type: 'doughnut',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: ['#38bdf8', '#22c55e', '#f59e0b', '#ef4444', '#a855f7'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right', labels: { color: '#94a3b8' } }
            }
        }
    });
}

function updateChart(chart, newVal) {
    const data = chart.data.datasets[0].data;
    data.push(newVal);
    data.shift();
    chart.update();
}

// ------ TOPOLOGY LOGIC ------
async function renderGraph() {
    const container = document.getElementById("graph-container");
    if (!container || container.querySelector('svg')) return; // already rendered

    const width = container.clientWidth;
    const height = container.clientHeight;

    const res = await fetch('/api/topology');
    const graph = await res.json();

    const svg = d3.select("#graph-container").append("svg")
        .attr("width", "100%")
        .attr("height", "100%")
        .attr("viewBox", [0, 0, width, height]);

    const simulation = d3.forceSimulation(graph.nodes)
        .force("link", d3.forceLink(graph.links).id(d => d.id).distance(120))
        .force("charge", d3.forceManyBody().strength(-400))
        .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(graph.links)
        .enter().append("line")
        .attr("class", "link")
        .attr("stroke-width", d => Math.sqrt(d.value));

    const node = svg.append("g")
        .attr("class", "nodes")
        .selectAll("g")
        .data(graph.nodes)
        .enter().append("g")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    node.append("circle")
        .attr("r", d => d.group === 6 ? 15 : 8)
        .attr("fill", d => colorByGroup(d.group));

    node.append("text")
        .attr("dx", 12)
        .attr("dy", ".35em")
        .text(d => d.id);

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("transform", d => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    function colorByGroup(g) {
        const colors = ["#ffffff", "#94a3b8", "#38bdf8", "#38bdf8", "#22c55e", "#f59e0b", "#a855f7"];
        return colors[g] || "#fff";
    }
}

async function rotateKeys() {
    await fetch('/api/security/rotate-keys', { method: 'POST' });
}

async function initiateLockdown() {
    await fetch('/api/security/lockdown', { method: 'POST' });
    alert("SYSTEM LOCKED DOWN. ALL TRAFFIC BLOCKED.");
}

async function runCoilOptimization() {
    const strength = document.getElementById('coil-field-strength').value;
    const btn = document.querySelector('button[onclick="runCoilOptimization()"]');
    const resultsDiv = document.getElementById('coil-results');
    const metricsDiv = document.getElementById('coil-metrics');

    btn.innerText = "Optimizing (NVQLink)...";
    btn.disabled = true;
    resultsDiv.style.display = 'none';

    try {
        const res = await fetch('/api/optimize-coil', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ field_strength: strength })
        });
        const data = await res.json();

        if (data.status === 'success') {
            metricsDiv.innerHTML = `
                <div style="margin-bottom: 10px;"><strong>Topology:</strong> ${data.topology.turns} turns, ${data.topology.length_mm.toFixed(1)}mm length</div>
                <div style="margin-bottom: 10px;"><strong>Inductance:</strong> ${data.circuit.inductance_uH.toFixed(2)} µH</div>
                <div style="margin-bottom: 10px;"><strong>Q-Factor:</strong> ${data.circuit.q_factor.toFixed(2)}</div>
                <div style="color: var(--success);"><strong>Resonant Freq:</strong> ${data.circuit.resonance_MHz.toFixed(2)} MHz</div>
            `;
            resultsDiv.style.display = 'block';
        } else {
            alert('Optimization failed: ' + data.message);
        }
    } catch (e) {
        console.error(e);
        alert('Optimization failed. Check console.');
    } finally {
        btn.innerText = "Run Optimization";
        btn.disabled = false;
    }
}
