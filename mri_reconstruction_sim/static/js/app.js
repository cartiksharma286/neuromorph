// Global state for simulation data
let lastSimulationData = null;

// Tab switching function
function switchView(viewId, buttonElement) {
    // Hide all view-content divs
    const allViews = document.querySelectorAll('.view-content');
    allViews.forEach(view => {
        view.classList.remove('active');
    });
    
    // Remove active class from all tab triggers
    const allTabs = document.querySelectorAll('.tab-trigger');
    allTabs.forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show the selected view
    const selectedView = document.getElementById('view-' + viewId);
    if (selectedView) {
        selectedView.classList.add('active');
    }
    
    // Mark the clicked button as active
    if (buttonElement) {
        buttonElement.classList.add('active');
    }
    
    // Load data for the specific tab
    setTimeout(() => {
        if (viewId === 'snr') {
            loadSNRMatrixData();
        } else if (viewId === 'analytics') {
            loadDistributionAnalytics();
        } else if (viewId === 'quantum-geometry') {
            loadQuantumGeometry();
        } else if (viewId === 'robotics-analytics') {
            loadRoboticsAnalytics();
        }
    }, 100);
}

// SNR Matrix Table Loader and Display
function loadSNRMatrix() {
    const insightsDiv = document.getElementById('snr-insights');
    if (!insightsDiv) return;
    
    insightsDiv.innerHTML = '<div style="color: #38bdf8; padding: 1rem;">Loading SNR matrix data...</div>';
    
    fetch('/api/snr_matrix')
        .then(res => res.json())
        .then(data => {
            if (!data.success) {
                insightsDiv.innerHTML = '<div style="color: #ef4444;">Error loading SNR matrix: ' + (data.error || 'Unknown error') + '</div>';
                return;
            }
            
            const matrix = data.snr_matrix;
            if (!matrix || matrix.length === 0) {
                insightsDiv.innerHTML = '<div style="color: #94a3b8;">No SNR data available.</div>';
                return;
            }
            
            // Extract unique sequences
            const sequences = matrix.length > 0 ? matrix[0].snr_values.map(v => v.sequence) : [];
            
            // Create HTML table
            let html = '<div style="overflow-x: auto; margin-top: 1rem;">';
            html += '<table style="width: 100%; border-collapse: collapse; background: #0f172a; border: 1px solid #334155;">';
            html += '<thead><tr style="background: #1e293b; border-bottom: 2px solid #38bdf8;">';
            html += '<th style="padding: 0.75rem; text-align: left; color: #38bdf8; font-weight: 600; border-right: 1px solid #334155;">Coil Type</th>';
            
            sequences.forEach(seq => {
                html += `<th style="padding: 0.75rem; text-align: center; color: #38bdf8; font-weight: 600; border-right: 1px solid #334155; font-size: 0.85rem;">${seq}</th>`;
            });
            
            html += '</tr></thead><tbody>';
            
            // Find min/max SNR for color scaling
            let minSnr = Infinity, maxSnr = -Infinity;
            matrix.forEach(coil => {
                coil.snr_values.forEach(val => {
                    if (val.snr !== undefined && val.snr > 0) {
                        minSnr = Math.min(minSnr, val.snr);
                        maxSnr = Math.max(maxSnr, val.snr);
                    }
                });
            });
            
            if (minSnr === Infinity) { minSnr = 0; maxSnr = 100; }
            
            // Add rows
            matrix.forEach((coil, idx) => {
                const rowColor = idx % 2 === 0 ? 'transparent' : '#0d1219';
                html += `<tr style="background: ${rowColor}; border-bottom: 1px solid #334155;">`;
                html += `<td style="padding: 0.75rem; color: #38bdf8; font-weight: 500; border-right: 1px solid #334155; white-space: nowrap;">${coil.coil}</td>`;
                
                coil.snr_values.forEach(val => {
                    const snr = val.snr || 0;
                    const normalized = (snr - minSnr) / (maxSnr - minSnr || 1);
                    const hue = normalized * 240; // Blue (240°) to Red (0°)
                    const bgColor = `hsl(${hue}, 70%, 45%)`;
                    const textColor = normalized > 0.5 ? '#000' : '#fff';
                    
                    html += `<td style="padding: 0.75rem; text-align: center; background: ${bgColor}; color: ${textColor}; font-weight: 600; border-right: 1px solid #334155; font-size: 0.9rem;" title="SNR: ${snr.toFixed(2)}">${snr.toFixed(1)}</td>`;
                });
                
                html += '</tr>';
            });
            
            html += '</tbody></table>';
            html += '</div>';
            
            // Add legend
            html += '<div style="margin-top: 1.5rem; padding: 1rem; background: #0f172a; border: 1px solid #334155; border-radius: 8px;">';
            html += '<div style="font-weight: 600; color: #38bdf8; margin-bottom: 0.75rem;">📊 SNR Heatmap Legend</div>';
            html += '<div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; font-size: 0.85rem;">';
            html += `<div style="display: flex; align-items: center; gap: 0.5rem;"><div style="width: 20px; height: 20px; background: hsl(240, 70%, 45%);"></div><span style="color: #94a3b8;">Lowest SNR (${minSnr.toFixed(2)})</span></div>`;
            html += `<div style="display: flex; align-items: center; gap: 0.5rem;"><div style="width: 20px; height: 20px; background: hsl(120, 70%, 45%);"></div><span style="color: #94a3b8;">Medium SNR</span></div>`;
            html += `<div style="display: flex; align-items: center; gap: 0.5rem;"><div style="width: 20px; height: 20px; background: hsl(0, 70%, 45%);"></div><span style="color: #94a3b8;">Highest SNR (${maxSnr.toFixed(2)})</span></div>`;
            html += '</div></div>';
            
            // Add summary statistics
            let totalSnr = 0, count = 0;
            matrix.forEach(coil => {
                coil.snr_values.forEach(val => {
                    if (val.snr && val.snr > 0) {
                        totalSnr += val.snr;
                        count++;
                    }
                });
            });
            const avgSnr = count > 0 ? (totalSnr / count).toFixed(2) : 0;
            const bestCoilSeq = findBestCombination(matrix);
            
            html += '<div style="margin-top: 1.5rem; padding: 1rem; background: #0f172a; border: 1px solid #38bdf8; border-radius: 8px;">';
            html += '<div style="font-weight: 600; color: #38bdf8; margin-bottom: 0.75rem;">📈 Summary Statistics</div>';
            html += `<div style="color: #94a3b8; font-family: monospace; line-height: 1.8;">`;
            html += `Average SNR: <span style="color: #38bdf8;">${avgSnr}</span><br>`;
            html += `Peak SNR: <span style="color: #38bdf8;">${maxSnr.toFixed(2)}</span><br>`;
            html += `Best Combination: <span style="color: #22c55e;">${bestCoilSeq.coil} + ${bestCoilSeq.sequence} (SNR: ${bestCoilSeq.snr.toFixed(2)})</span><br>`;
            html += `Total Combinations: <span style="color: #38bdf8;">${count}</span>`;
            html += '</div></div>';
            
            insightsDiv.innerHTML = html;
        })
        .catch(err => {
            insightsDiv.innerHTML = '<div style="color: #ef4444;">Failed to fetch SNR matrix: ' + err.message + '</div>';
            console.error('SNR Matrix Error:', err);
        });
}

function findBestCombination(matrix) {
    let best = { coil: 'N/A', sequence: 'N/A', snr: 0 };
    matrix.forEach(coil => {
        coil.snr_values.forEach(val => {
            if (val.snr && val.snr > best.snr) {
                best = { coil: coil.coil, sequence: val.sequence, snr: val.snr };
            }
        });
    });
    return best;
}

// Auto-load SNR matrix when the SNR view is accessed
document.addEventListener('DOMContentLoaded', function() {
    const snrTab = document.querySelector('[onclick*="view-snr"]');
    if (snrTab) {
        loadSNRMatrix();
    }
});

// Also allow programmatic triggering
window.loadSNRMatrixData = loadSNRMatrix;

// Distribution Analysis Loader
function loadDistributionAnalytics() {
    const detailsDiv = document.getElementById('analytics-details');
    const imgDiv = document.getElementById('img-distribution_curve');
    
    if (!detailsDiv || !imgDiv) return;
    
    detailsDiv.innerHTML = '<div style="color: #38bdf8;">Loading distribution analysis...</div>';
    
    fetch('/api/distribution_analysis')
        .then(res => res.json())
        .then(data => {
            if (data.success && data.image) {
                imgDiv.src = 'data:image/png;base64,' + data.image;
            }
            if (data.details) {
                detailsDiv.innerHTML = data.details;
            }
        })
        .catch(err => {
            detailsDiv.innerHTML = '<div style="color: #ef4444;">Error loading distribution analysis</div>';
            console.error('Distribution Analysis Error:', err);
        });
}

// Quantum Geometry Loader
function loadQuantumGeometry() {
    const insightsDiv = document.getElementById('quantum-insights');
    const rekImg = document.getElementById('img-quantum-recon');
    const kspaceImg = document.getElementById('img-quantum-kspace');
    const manifoldImg = document.getElementById('img-quantum-manifold');
    
    if (!rekImg) return;
    
    insightsDiv.innerHTML = '<div style="color: #38bdf8;">Loading quantum geometry analysis...</div>';
    
    fetch('/api/quantum_geometry')
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                if (data.recon_image) rekImg.src = 'data:image/png;base64,' + data.recon_image;
                if (data.kspace_image) kspaceImg.src = 'data:image/png;base64,' + data.kspace_image;
                if (data.manifold_image) manifoldImg.src = 'data:image/png;base64,' + data.manifold_image;
                
                // Update metrics
                if (data.metrics) {
                    const metrics = data.metrics;
                    document.getElementById('metricNormVal').textContent = (metrics.metric_norm || 0).toFixed(3);
                    document.getElementById('curvatureMax').textContent = (metrics.curvature || 0).toFixed(3);
                    document.getElementById('cfDepthVal').textContent = metrics.cf_depth || '32-bit';
                }
                
                // Update insights
                if (data.insights) {
                    insightsDiv.innerHTML = data.insights;
                }
            }
        })
        .catch(err => {
            insightsDiv.innerHTML = '<div style="color: #ef4444;">Error loading quantum geometry: ' + err.message + '</div>';
        });
}

// Robotics Analytics Loader
function loadRoboticsAnalytics() {
    const metricsDiv = document.getElementById('robotics-metrics');
    const imgDiv = document.getElementById('img-robotics-map');
    const statusDiv = document.getElementById('robotics-status-badge');
    
    if (!metricsDiv || !imgDiv) return;
    
    metricsDiv.innerHTML = '<div style="color: #38bdf8;">Loading robotics analytics...</div>';
    
    fetch('/api/robotics_analytics')
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                if (data.image) imgDiv.src = 'data:image/png;base64,' + data.image;
                
                // Update status
                if (data.status) {
                    statusDiv.textContent = data.status;
                    statusDiv.style.color = data.status === 'ACTIVE' ? '#22c55e' : '#f43f5e';
                }
                
                // Update metrics
                if (data.metrics) {
                    let metricsHtml = '';
                    Object.keys(data.metrics).forEach(key => {
                        metricsHtml += `
                            <div class="metric-mini">
                                <div class="metric-mini-val">${(data.metrics[key]).toFixed(2)}</div>
                                <div class="metric-mini-label">${key.replace(/_/g, ' ').toUpperCase()}</div>
                            </div>
                        `;
                    });
                    metricsDiv.innerHTML = metricsHtml;
                }
            }
        })
        .catch(err => {
            metricsDiv.innerHTML = '<div style="color: #ef4444;">Error loading robotics analytics</div>';
            console.error('Robotics Analytics Error:', err);
        });
}

