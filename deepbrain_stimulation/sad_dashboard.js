
class SADDashboard {
    constructor() {
        this.initialized = false;
        this.chart = null;
    }

    init() {
        if (this.initialized) return;
        console.log("Initializing SAD Dashboard...");

        this.setupEventListeners();
        this.initialized = true;
    }

    setupEventListeners() {
        document.getElementById('run-sad-sim').addEventListener('click', () => this.runSimulation());
        document.getElementById('sad-target').addEventListener('change', () => this.updateParameters());
    }

    updateParameters() {
        // Logic to update recommended params based on target
        const target = document.getElementById('sad-target').value;
        const freqInput = document.getElementById('sad-frequency');

        if (target === 'Lateral Habenula') {
            freqInput.value = 135; // LHb requires high freq usually
        } else {
            freqInput.value = 60; // SCN might use lower freq
        }
    }

    async runSimulation() {
        const params = {
            target: document.getElementById('sad-target').value,
            frequency: parseFloat(document.getElementById('sad-frequency').value),
            amplitude: parseFloat(document.getElementById('sad-amplitude').value),
            paradigm: document.getElementById('sad-paradigm').value,
            duration: 1.0 // 1 second simulation
        };

        this.showLoading(true);

        try {
            const response = await fetch(`${API_BASE}/sad/treat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });

            const data = await response.json();

            if (data.success) {
                this.updateResults(data);
            } else {
                alert('Simulation failed: ' + data.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Simulation error');
        } finally {
            this.showLoading(false);
        }
    }

    updateResults(data) {
        const result = data.result;

        // Render DBS Profile
        const profileDiv = document.getElementById('dbs-profile-content');
        const electrical = result.dbs_profile.electrical_properties;
        const field = result.dbs_profile.field_distribution;
        const safety = result.dbs_profile.safety_check;

        profileDiv.innerHTML = `
            <div style="margin-bottom: 10px; border-bottom: 1px solid #444; padding-bottom: 5px;">
                <div style="display:flex; justify-content:space-between;"><span>Impedance:</span> <strong style="color:white">${electrical.impedance_ohms} Ω</strong></div>
                <div style="display:flex; justify-content:space-between;"><span>Voltage:</span> <strong style="color:white">${electrical.voltage_v} V</strong></div>
                <div style="display:flex; justify-content:space-between;"><span>Power:</span> <strong style="color:white">${electrical.power_consumption_uw} µW</strong></div>
            </div>
            <div style="margin-bottom: 10px; border-bottom: 1px solid #444; padding-bottom: 5px;">
                <div style="display:flex; justify-content:space-between;"><span>Charge Density:</span> <strong style="color:${safety.status === 'Safe' ? '#00ff88' : '#ff3366'}">${electrical.charge_density_uc_cm2} µC/cm²</strong></div>
                <div style="display:flex; justify-content:space-between;"><span>VTA Volume:</span> <strong style="color:white">${field.vta_volume_mm3} mm³</strong></div>
                <div style="display:flex; justify-content:space-between;"><span>Max E-Field:</span> <strong style="color:white">${field.max_e_field_v_mm} V/mm</strong></div>
            </div>
            <div style="text-align: center; margin-top: 10px;">
                <span class="stat-badge" style="background: ${safety.status === 'Safe' ? 'rgba(0,255,136,0.2)' : 'rgba(255,51,102,0.2)'}; color: ${safety.status === 'Safe' ? '#00ff88' : '#ff3366'};">
                    ${safety.status.toUpperCase()}
                </span>
            </div>
        `;

        // Render Post-Treatment Metrics
        const postDiv = document.getElementById('post-treatment-content');
        const outcome = result.post_treatment.clinical_outcome;
        const circadian = result.post_treatment.circadian_state;

        postDiv.innerHTML = `
            <div class="stat-card" style="border-left: 4px solid ${outcome.remission_probability > 50 ? '#00d4ff' : '#ffaa00'}; margin-bottom: 15px;">
                <div class="stat-value">${outcome.remission_probability}%</div>
                <div class="stat-label">Remission Probability</div>
                <div class="stat-badge">${outcome.classification}</div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div>
                    <div style="font-size:0.8em; color:#888;">Melatonin Peak</div>
                    <div style="font-size:1.1em; font-weight:bold; color:white;">${circadian.melatonin_peak} pg/mL</div>
                </div>
                <div>
                    <div style="font-size:0.8em; color:#888;">SCN Activity</div>
                    <div style="font-size:1.1em; font-weight:bold; color:white;">${circadian.scn_firing_rate} Hz</div>
                </div>
                <div>
                    <div style="font-size:0.8em; color:#888;">Neural Repair</div>
                    <div style="font-size:1.1em; font-weight:bold; color:#a78bfa;">${outcome.neural_repair_index}</div>
                </div>
                <div>
                    <div style="font-size:0.8em; color:#888;">Mood Index</div>
                    <div style="font-size:1.1em; font-weight:bold; color:white;">${outcome.mood_stabilization_index}/10</div>
                </div>
            </div>
        `;

        // Plot Neural Activity
        this.plotActivity(result.neural_activity, result.melatonin_trace);

        // Plot Recovery Trajectory
        this.plotRecovery(result.post_treatment.recovery_data);
    }

    plotActivity(neural, melatonin) {
        const ctx = document.getElementById('sad-chart').getContext('2d');

        if (this.chart) this.chart.destroy();

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({ length: neural.length }, (_, i) => i * 0.01),
                datasets: [{
                    label: 'Neural Oscillation',
                    data: neural,
                    borderColor: '#60a5fa',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0,
                    yAxisID: 'y'
                }, {
                    label: 'Melatonin',
                    data: melatonin,
                    borderColor: '#a78bfa',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    y: {
                        display: false,
                        grid: { display: false }
                    },
                    y1: {
                        display: false,
                        grid: { display: false }
                    },
                    x: {
                        display: false
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    plotRecovery(data) {
        const ctx = document.getElementById('sad-recovery-chart').getContext('2d');

        if (this.recoveryChart) this.recoveryChart.destroy();

        this.recoveryChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.weeks.map(w => `Wk ${w}`),
                datasets: [{
                    label: 'Recovery (%)',
                    data: data.trajectory,
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#888', font: { size: 9 } }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#888', font: { size: 9 }, maxTicksLimit: 6 }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    showLoading(show) {
        const btn = document.getElementById('run-sad-sim');
        if (show) {
            btn.innerHTML = '<span class="spinner"></span> Simulate...';
            btn.disabled = true;
        } else {
            btn.innerHTML = 'Run SAD Treatment Simulation';
            btn.disabled = false;
        }
    }
}

// Global instance
window.sadDashboard = new SADDashboard();
console.log("SAD Dashboard Loaded");
