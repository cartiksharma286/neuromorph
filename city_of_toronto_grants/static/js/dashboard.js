/* Dashboard Controller for City of Toronto Community Grants Optimization */

let pulseChart = null;
let revenueChart = null;
let sectorChart = null;
let pigeonChart = null;
let currentData = null;

document.addEventListener("DOMContentLoaded", () => {
    initControls();
    loadData();
});

function initControls() {
    const budgetInput = document.getElementById("budgetRange");
    const flipInput = document.getElementById("flipRange");
    const bwInput = document.getElementById("bwRange");
    const applicantsInput = document.getElementById("applicantsRange");

    budgetInput.addEventListener("input", (e) => {
        document.getElementById("budgetValue").textContent = `$${parseInt(e.target.value).toLocaleString()} CAD`;
    });
    flipInput.addEventListener("input", (e) => {
        document.getElementById("flipValue").textContent = `${e.target.value}°`;
    });
    bwInput.addEventListener("input", (e) => {
        document.getElementById("bwValue").textContent = `${parseFloat(e.target.value).toFixed(1)} kHz`;
    });
    applicantsInput.addEventListener("input", (e) => {
        document.getElementById("applicantsValue").textContent = parseInt(e.target.value).toLocaleString();
    });

    document.getElementById("btnApply").addEventListener("click", () => {
        loadData();
    });

    const btnAuto = document.getElementById("btnAutoOptimize");
    if (btnAuto) {
        btnAuto.addEventListener("click", () => {
            const budget = document.getElementById("budgetRange").value;
            const applicants = document.getElementById("applicantsRange").value;
            btnAuto.textContent = "⏳ Optimizing...";

            fetch("/api/auto_optimize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    budget: parseFloat(budget),
                    applicants: parseInt(applicants)
                })
            })
            .then(res => res.json())
            .then(data => {
                btnAuto.textContent = "💎 Maximize Yield";
                if (data.optimal_parameters) {
                    document.getElementById("flipRange").value = data.optimal_parameters.flip_angle;
                    document.getElementById("flipValue").textContent = `${data.optimal_parameters.flip_angle}°`;
                    document.getElementById("bwRange").value = data.optimal_parameters.bandwidth;
                    document.getElementById("bwValue").textContent = `${data.optimal_parameters.bandwidth.toFixed(1)} kHz`;
                }
                currentData = data;
                updateMetrics(data.summary);
                renderPulseChart(data.rf_pulse_profile);
                renderRevenueChart(data.yearly_projections);
                renderSectorChart(data.sector_allocations);
                renderPigeonChart(data.sector_allocations, data.pigeonhole_invariants);
                renderTables(data);
            })
            .catch(err => {
                btnAuto.textContent = "💎 Maximize Yield";
                console.error("Auto optimize error:", err);
            });
        });
    }
}

function loadData() {
    const budget = document.getElementById("budgetRange").value;
    const flip_angle = document.getElementById("flipRange").value;
    const bandwidth = document.getElementById("bwRange").value;
    const applicants = document.getElementById("applicantsRange").value;

    fetch("/api/allocate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            budget: parseFloat(budget),
            flip_angle: parseFloat(flip_angle),
            bandwidth: parseFloat(bandwidth),
            applicants: parseInt(applicants),
            years: 5
        })
    })
    .then(res => res.json())
    .then(data => {
        currentData = data;
        updateMetrics(data.summary);
        renderPulseChart(data.rf_pulse_profile);
        renderRevenueChart(data.yearly_projections);
        renderSectorChart(data.sector_allocations);
        renderPigeonChart(data.sector_allocations, data.pigeonhole_invariants);
        renderTables(data);
    })
    .catch(err => {
        console.error("Error loading allocation data:", err);
    });
}

function updateMetrics(s) {
    document.getElementById("metricYield").textContent = `$${(s["5yr_cumulative_municipal_yield_cad"] / 1e6).toFixed(2)}M`;
    document.getElementById("metricRoi").textContent = `+${s["5yr_net_municipal_roi_pct"].toFixed(1)}%`;
    document.getElementById("metricLeverage").textContent = `$${(s["total_leveraged_capital_cad"] / 1e6).toFixed(2)}M`;
    document.getElementById("metricSroi").textContent = `${s["weighted_average_sroi"].toFixed(2)}x`;
    document.getElementById("metricJobs").textContent = s["total_community_jobs_sustained"].toLocaleString();
    document.getElementById("metricCollisions").textContent = s["dirichlet_peak_occupancy"];
}

function renderPulseChart(profile) {
    const ctx = document.getElementById("pulseChartCanvas").getContext("2d");
    if (pulseChart) pulseChart.destroy();

    pulseChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: profile.frequencies,
            datasets: [
                {
                    label: "Excitation Probability |β(ω)|²",
                    data: profile.excitation_probability,
                    borderColor: "#7c3aed",
                    backgroundColor: "rgba(124, 58, 237, 0.1)",
                    fill: true,
                    tension: 0.35,
                    borderWidth: 2,
                    pointRadius: 0
                },
                {
                    label: "Transverse Coherence |M_xy|",
                    data: profile.transverse_magnetization,
                    borderColor: "#00A3E0",
                    borderDash: [5, 5],
                    borderWidth: 1.5,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: "Frequency Offset Δω (kHz)" },
                    grid: { color: "#f1f5f9" }
                },
                y: {
                    title: { display: true, text: "Excitation Amplitude [0, 1]" },
                    min: 0,
                    max: 1.05,
                    grid: { color: "#f1f5f9" }
                }
            },
            plugins: {
                legend: { position: "top" }
            }
        }
    });
}

function renderRevenueChart(projections) {
    const ctx = document.getElementById("revenueChartCanvas").getContext("2d");
    if (revenueChart) revenueChart.destroy();

    const labels = projections.map(p => `FY ${p.year}`);
    const taxLift = projections.map(p => p.tax_base_lift_cad / 1e6);
    const costAvoidance = projections.map(p => p.cost_avoidance_cad / 1e6);
    const cumulativeYield = projections.map(p => p.cumulative_municipal_yield_cad / 1e6);

    revenueChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Tax Base Expansion ($M)",
                    data: taxLift,
                    backgroundColor: "#002B49"
                },
                {
                    label: "Municipal Cost Avoidance ($M)",
                    data: costAvoidance,
                    backgroundColor: "#00A3E0"
                },
                {
                    type: "line",
                    label: "Cumulative Municipal Yield ($M)",
                    data: cumulativeYield,
                    borderColor: "#F2A900",
                    backgroundColor: "#F2A900",
                    borderWidth: 2.5,
                    tension: 0.2,
                    yAxisID: "yCumulative"
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: "#f1f5f9" } },
                y: {
                    title: { display: true, text: "Annual Yield ($M CAD)" },
                    stacked: true,
                    grid: { color: "#f1f5f9" }
                },
                yCumulative: {
                    position: "right",
                    title: { display: true, text: "Cumulative ($M CAD)" },
                    grid: { drawOnChartArea: false }
                }
            },
            plugins: {
                legend: { position: "top" }
            }
        }
    });
}

function renderSectorChart(sectors) {
    const ctx = document.getElementById("sectorChartCanvas").getContext("2d");
    if (sectorChart) sectorChart.destroy();

    const labels = sectors.map(s => s.sector_name);
    const dataVals = sectors.map(s => s.allocation_cad / 1e6);

    sectorChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: labels,
            datasets: [{
                data: dataVals,
                backgroundColor: [
                    "#002B49",
                    "#00A3E0",
                    "#F2A900",
                    "#7c3aed",
                    "#10b981",
                    "#f43f5e"
                ],
                borderWidth: 2,
                borderColor: "#ffffff"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "right" },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` $${ctx.raw.toFixed(2)}M CAD (${((ctx.raw / dataVals.reduce((a,b)=>a+b,0))*100).toFixed(1)}%)`
                    }
                }
            }
        }
    });
}

function renderPigeonChart(sectors, invariants) {
    const ctx = document.getElementById("pigeonChartCanvas").getContext("2d");
    if (pigeonChart) pigeonChart.destroy();

    const labels = sectors.map(s => s.sector_name.split(" ")[0]);
    const occupancies = sectors.map(s => s.grants_awarded);

    pigeonChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Assigned Grants (Pigeonhole Occupancy)",
                    data: occupancies,
                    backgroundColor: "#3b82f6",
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: { display: true, text: "Number of Vetted Grants (Occupancy n_k)" },
                    grid: { color: "#f1f5f9" }
                }
            },
            plugins: {
                annotation: {
                    annotations: {
                        line1: {
                            type: 'line',
                            yMin: invariants.pigeonhole_minimum_peak,
                            yMax: invariants.pigeonhole_minimum_peak,
                            borderColor: '#ef4444',
                            borderWidth: 2,
                            borderDash: [6, 6],
                            label: {
                                content: `Dirichlet Peak ⌈N/K⌉ = ${invariants.pigeonhole_minimum_peak}`,
                                enabled: true
                            }
                        }
                    }
                }
            }
        }
    });
}

function renderTables(data) {
    // 1. Sector Table
    const secTbody = document.getElementById("sectorTableBody");
    secTbody.innerHTML = "";
    data.sector_allocations.forEach(s => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><b>${s.sector_name}</b></td>
            <td><span class="badge badge-quantum">${(s.excitation_resonance * 100).toFixed(1)}%</span></td>
            <td><b>$${(s.allocation_cad / 1e6).toFixed(2)}M</b></td>
            <td>${s.allocation_pct.toFixed(1)}%</td>
            <td>${s.grants_awarded}</td>
            <td>$${s.avg_grant_size_cad.toLocaleString()}</td>
            <td><span style="color: #16a34a; font-weight:700;">$${(s.leveraged_capital_cad / 1e6).toFixed(2)}M</span></td>
        `;
        secTbody.appendChild(tr);
    });

    // 2. Ward Table
    const wardTbody = document.getElementById("wardTableBody");
    wardTbody.innerHTML = "";
    data.ward_allocations.forEach(w => {
        const priorityColor = w.priority_tier === "High" ? "#dc2626" : (w.priority_tier === "Medium" ? "#d97706" : "#2563eb");
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><b>Ward ${w.ward_id}: ${w.ward_name}</b></td>
            <td>${w.population.toLocaleString()}</td>
            <td>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span>${w.deprivation_index.toFixed(2)}</span>
                    <div style="flex: 1; height: 6px; background: #e2e8f0; border-radius: 3px; max-width: 60px;">
                        <div style="width: ${w.deprivation_index * 100}%; height: 100%; background: ${priorityColor}; border-radius: 3px;"></div>
                    </div>
                </div>
            </td>
            <td><span class="badge" style="background: ${priorityColor}15; color: ${priorityColor};">${w.priority_tier}</span></td>
            <td><b>$${(w.allocation_cad / 1e6).toFixed(2)}M</b></td>
            <td>${w.allocation_pct.toFixed(2)}%</td>
            <td><span class="badge" style="background:#dcfce7; color:#15803d;">${w.jobs_created} jobs</span></td>
            <td>$${(w.annual_cost_savings_cad / 1e6).toFixed(2)}M</td>
        `;
        wardTbody.appendChild(tr);
    });
}

function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.style.display = "none");

    if (tabId === "sectors") {
        document.getElementById("btnTabSectors").classList.add("active");
        document.getElementById("sectorTableContainer").style.display = "block";
    } else {
        document.getElementById("btnTabWards").classList.add("active");
        document.getElementById("wardTableContainer").style.display = "block";
    }
}
