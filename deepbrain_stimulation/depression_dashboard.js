
class DepressionDashboard {
    constructor() {
        this.module = document.getElementById('depressionModule');
        this.ctx = document.getElementById('neuroChart').getContext('2d');
        this.chart = null;
        this.initialized = false;
    }

    init() {
        if (this.initialized) return;

        this.setupChart();
        this.setupEventListeners();

        this.initialized = true;
    }

    setupChart() {
        this.chart = new Chart(this.ctx, {
            type: 'bar',
            data: {
                labels: ['Serotonin (5-HT)', 'Dopamine (DA)', 'Glutamate'],
                datasets: [{
                    label: 'Neurotransmitter Levels',
                    data: [0.3, 0.4, 0.8], // Initial mock data
                    backgroundColor: [
                        'rgba(75, 192, 192, 0.7)', // Serotonin - Teal
                        'rgba(54, 162, 235, 0.7)', // Dopamine - Blue
                        'rgba(255, 99, 132, 0.7)'  // Glutamate - Red
                    ],
                    borderColor: [
                        'rgba(75, 192, 192, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 99, 132, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.2,
                        grid: { color: '#333' },
                        ticks: { color: '#888' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#888' }
                    }
                },
                plugins: {
                    legend: { display: false },
                    title: { display: true, text: 'Synaptic Balance', color: '#aaa' }
                }
            }
        });
    }

    setupEventListeners() {
        document.getElementById('simDepressionBtn').addEventListener('click', () => this.runSimulation());
        document.getElementById('resetDepressionBtn').addEventListener('click', () => this.resetSimulation());

        // Input Sync
        const sync = (id, valId) => {
            document.getElementById(id).addEventListener('input', (e) => {
                document.getElementById(valId).textContent = e.target.value + (id.includes('Freq') ? ' Hz' : ' mA');
            });
        };
        sync('depFreq', 'depFreqVal');
        sync('depAmp', 'depAmpVal');
    }

    async runSimulation() {
        if (!window.app) return;

        const params = {
            target: document.getElementById('depTarget').value,
            frequency: document.getElementById('depFreq').value,
            amplitude: document.getElementById('depAmp').value
        };

        window.app.showLoading(document.getElementById('feaContainer'));

        try {
            const response = await window.app.post('/depression/simulate', params);
            if (response.success) {
                this.updateUI(response.results);
            }
        } catch (error) {
            console.error(error);
            window.app.showError(document.getElementById('feaContainer'), 'Simulation Failed');
        }
    }

    async resetSimulation() {
        if (!window.app) return;
        try {
            const response = await window.app.post('/depression/reset', {});
            if (response.success) {
                // Reset UI visualization manually or re-fetch basic state if endpoint existed
                // For now, reload chart with defaults
                this.chart.data.datasets[0].data = [0.3, 0.4, 0.8];
                this.chart.update();
                document.getElementById('feaContainer').innerHTML = '<span style="color: #666;">Ready</span>';
                document.getElementById('paradigmList').innerHTML = '<li>Baseline Restored.</li>';
            }
        } catch (e) { console.error(e); }
    }

    updateUI(results) {
        // 1. Update Chart
        const nt = results.neurotransmitters;
        this.chart.data.datasets[0].data = [nt.serotonin, nt.dopamine, nt.glutamate];
        this.chart.update();

        // Update Status Text
        const ratio = nt.serotonin / (nt.dopamine + 0.01);
        let status = "Imbalanced";
        if (ratio > 0.8 && ratio < 1.2) status = "Balanced";
        else if (ratio < 0.5) status = "Dopaminergic Dominance (Agitation Risk)";
        else if (ratio > 1.5) status = "Serotonergic Dominance (Sedation Risk)";

        document.getElementById('balanceStatus').textContent = status;

        // 2. Update FEA
        if (results.fea_heatmap) {
            const img = document.createElement('img');
            img.src = 'data:image/png;base64,' + results.fea_heatmap;
            img.style.maxWidth = '100%';
            img.style.maxHeight = '100%';
            img.style.borderRadius = '4px';

            const container = document.getElementById('feaContainer');
            container.innerHTML = '';
            container.appendChild(img);
        }

        document.getElementById('actVol').textContent = results.activation_stats.volume_mm3.toFixed(2);
        document.getElementById('maxField').textContent = results.activation_stats.max_field_v_mm.toFixed(2);

        // 3. Update Executive Metrics
        const exec = results.executive;
        document.getElementById('flexMeter').style.width = (exec.cognitive_flexibility * 100) + '%';
        document.getElementById('memMeter').style.width = (exec.working_memory * 100) + '%';
        document.getElementById('speedMeter').style.width = (exec.decision_speed * 100) + '%';

        // 4. Update Paradigms
        const pList = document.getElementById('paradigmList');
        pList.innerHTML = '';
        results.paradigms.forEach(p => {
            const li = document.createElement('li');
            li.textContent = p;
            li.style.marginBottom = '8px';
            pList.appendChild(li);
        });
    }
}
