/* Dashboard Controller for City of Toronto Community Grants Optimization */

let pulseChart = null;
let revenueChart = null;
let sectorChart = null;
let pigeonChart = null;
let jobsChart = null;
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
                renderJobsHorizonChart(data.yearly_projections);
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
        renderJobsHorizonChart(data.yearly_projections);
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
    if (document.getElementById("metricJobsMargin") && s["5yr_total_citywide_jobs_created"]) {
        document.getElementById("metricJobsMargin").textContent = s["5yr_total_citywide_jobs_created"].toLocaleString();
    }
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

function renderJobsHorizonChart(projections) {
    const canvas = document.getElementById("jobsHorizonChartCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (jobsChart) jobsChart.destroy();

    const labels = projections.map(p => `FY ${p.year}`);
    const communityJobs = projections.map(p => p.community_new_jobs);
    const biotechJobs = projections.map(p => p.biotech_new_jobs);
    const cumulativeMargin = projections.map(p => p.cumulative_total_jobs);

    jobsChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "New Community Grants Sector Jobs",
                    data: communityJobs,
                    backgroundColor: "#10b981",
                    borderRadius: 4
                },
                {
                    label: "New Downtown Biotech STEM Jobs",
                    data: biotechJobs,
                    backgroundColor: "#6366f1",
                    borderRadius: 4
                },
                {
                    type: "line",
                    label: "Cumulative Jobs Margin (Workers Sustained / Created)",
                    data: cumulativeMargin,
                    borderColor: "#d97706",
                    backgroundColor: "#f59e0b",
                    borderWidth: 3,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    tension: 0.25,
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
                    title: { display: true, text: "Annual New Jobs Created (Workers)" },
                    stacked: true,
                    grid: { color: "#f1f5f9" }
                },
                yCumulative: {
                    position: "right",
                    title: { display: true, text: "Cumulative Citywide Jobs Margin" },
                    grid: { drawOnChartArea: false }
                }
            },
            plugins: {
                legend: { position: "top" },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toLocaleString()} workers`
                    }
                }
            }
        }
    });
}

function renderTables(data) {
    // 1. Sector Table
    const secTbody = document.getElementById("sectorTableBody");
    if (secTbody) {
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
    }

    // 2. Ward Table
    const wardTbody = document.getElementById("wardTableBody");
    if (wardTbody) {
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

    // 3. Downtown Toronto Biotech Corridor Table
    if (data.biotech_corridor) {
        const bio = data.biotech_corridor;
        if (document.getElementById("bioRdStat")) {
            document.getElementById("bioRdStat").textContent = `$${(bio.summary.annual_rd_investment_cad / 1e6).toFixed(1)}M CAD`;
            document.getElementById("bioVcStat").textContent = `$${(bio.summary.vc_matched_capital_cad / 1e6).toFixed(1)}M CAD`;
            document.getElementById("bioJobsStat").textContent = `${bio.summary.five_year_total_stem_jobs.toLocaleString()} positions`;
            document.getElementById("bioTaxStat").textContent = `$${(bio.summary.annual_municipal_tax_cad / 1e6).toFixed(2)}M CAD`;
        }

        const bioTbody = document.getElementById("biotechTableBody");
        if (bioTbody) {
            bioTbody.innerHTML = "";
            bio.hubs.forEach(h => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><b>${h.hub_name}</b></td>
                    <td><span style="font-size:0.8rem; color:#475569;">${h.focus}</span></td>
                    <td><b>$${(h.annual_rd_investment_cad / 1e6).toFixed(1)}M</b></td>
                    <td><span class="badge" style="background:#e0e7ff; color:#4338ca;">${h.vc_leverage_ratio.toFixed(1)}x</span></td>
                    <td><span style="color:#16a34a; font-weight:700;">$${(h.vc_matched_capital_cad / 1e6).toFixed(1)}M</span></td>
                    <td>$${h.avg_stem_salary_cad.toLocaleString()}</td>
                    <td>${h.yr1_stem_jobs.toLocaleString()}</td>
                    <td><span class="badge" style="background:#dcfce7; color:#15803d; font-weight:700;">+${h.five_year_total_jobs.toLocaleString()} jobs</span></td>
                    <td>$${(h.annual_municipal_tax_cad / 1e6).toFixed(2)}M</td>
                    <td>$${(h.annual_gdp_impact_cad / 1e6).toFixed(1)}M</td>
                `;
                bioTbody.appendChild(tr);
            });
        }
    }

    // 4. Table of Values: 5-Year Balance Sheet & Jobs Margin
    const balTbody = document.getElementById("balanceTableBody");
    if (balTbody) {
        balTbody.innerHTML = "";
        data.yearly_projections.forEach(y => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><b>FY ${y.year}</b></td>
                <td>$${(y.tax_base_lift_cad / 1e6).toFixed(2)}M</td>
                <td>$${(y.cost_avoidance_cad / 1e6).toFixed(2)}M</td>
                <td><b style="color:#002B49;">$${(y.total_municipal_yield_cad / 1e6).toFixed(2)}M</b></td>
                <td><b style="color:#F2A900;">$${(y.cumulative_municipal_yield_cad / 1e6).toFixed(2)}M</b></td>
                <td><span class="badge" style="background:#dcfce7; color:#15803d;">+${y.community_new_jobs.toLocaleString()}</span></td>
                <td><span class="badge" style="background:#e0e7ff; color:#4338ca;">+${y.biotech_new_jobs.toLocaleString()}</span></td>
                <td><b style="color:#0f172a;">+${y.total_new_jobs_created.toLocaleString()} jobs</b></td>
                <td><span class="badge" style="background:#fef3c7; color:#92400e; font-weight:700; font-size:0.82rem;">${y.cumulative_total_jobs.toLocaleString()} total</span></td>
                <td>$${(y.leveraged_inflow_cad / 1e6).toFixed(2)}M</td>
                <td>$${(y.annual_sroi_value_cad / 1e6).toFixed(2)}M</td>
            `;
            balTbody.appendChild(tr);
        });
    }
}

function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.style.display = "none");

    if (tabId === "sectors") {
        document.getElementById("btnTabSectors").classList.add("active");
        document.getElementById("sectorTableContainer").style.display = "block";
    } else if (tabId === "wards") {
        document.getElementById("btnTabWards").classList.add("active");
        document.getElementById("wardTableContainer").style.display = "block";
    } else if (tabId === "biotech") {
        document.getElementById("btnTabBiotech").classList.add("active");
        document.getElementById("biotechTableContainer").style.display = "block";
    } else if (tabId === "balance") {
        document.getElementById("btnTabBalance").classList.add("active");
        document.getElementById("balanceTableContainer").style.display = "block";
    }
}
