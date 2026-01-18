/**
 * Cell Tracking & Differentiation System - Frontend Application
 */

const API_BASE = 'http://localhost:5000/api';

// Global State
let currentFrame = 0;
let totalFrames = 0;
let isPlaying = false;
let playInterval = null;
let charts = {};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    initializeNavigation();
    initializeEventListeners();
    initializeCharts();
});

// Navigation
function initializeNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const view = item.dataset.view;
            switchView(view);
            
            // Update active state
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
        });
    });
}

function switchView(viewName) {
    const views = document.querySelectorAll('.view-container');
    views.forEach(view => view.classList.remove('active'));
    
    const targetView = document.getElementById(`view-${viewName}`);
    if (targetView) {
        targetView.classList.add('active');
        updateViewHeader(viewName);
    }
}

function updateViewHeader(viewName) {
    const titles = {
        'upload': {
            title: 'Upload Microscopy Data',
            subtitle: 'Begin by uploading time-lapse microscopy images or use sample data'
        },
        'tracking': {
            title: 'Cell Tracking',
            subtitle: 'Track cells across time-lapse frames with advanced algorithms'
        },
        'optimization': {
            title: 'Heuristic Optimization',
            subtitle: 'Optimize tracking and lineage reconstruction using evolutionary algorithms'
        },
        'differentiation': {
            title: 'Differentiation Analysis',
            subtitle: 'Analyze cell differentiation trajectories and predict cell fate'
        },
        'generative': {
            title: 'Generative AI',
            subtitle: 'Generate synthetic cell trajectories and morphologies'
        }
    };
    
    const config = titles[viewName];
    if (config) {
        document.getElementById('view-title').textContent = config.title;
        document.getElementById('view-subtitle').textContent = config.subtitle;
    }
}

// Event Listeners
function initializeEventListeners() {
    // Upload
    document.getElementById('load-sample-btn').addEventListener('click', loadSampleData);
    document.getElementById('file-input').addEventListener('change', handleFileUpload);
    
    // Tracking
    document.getElementById('run-tracking-btn').addEventListener('click', runTracking);
    document.getElementById('play-btn').addEventListener('click', togglePlayback);
    document.getElementById('prev-frame-btn').addEventListener('click', () => changeFrame(-1));
    document.getElementById('next-frame-btn').addEventListener('click', () => changeFrame(1));
    document.getElementById('frame-slider').addEventListener('input', (e) => {
        currentFrame = parseInt(e.target.value);
        updateFrame();
    });
    
    // Optimization
    document.getElementById('run-ga-btn').addEventListener('click', runGeneticAlgorithm);
    document.getElementById('run-pso-btn').addEventListener('click', runPSO);
    document.getElementById('run-aco-btn').addEventListener('click', runACO);
    document.getElementById('run-hybrid-btn').addEventListener('click', runHybridOptimization);
    
    // Differentiation
    document.getElementById('analyze-diff-btn').addEventListener('click', analyzeDifferentiation);
    document.getElementById('predict-fate-btn').addEventListener('click', predictCellFate);
    
    // Generative
    document.getElementById('gen-traj-btn').addEventListener('click', generateTrajectories);
    document.getElementById('gen-morph-btn').addEventListener('click', generateMorphologies);
}

// Initialize Charts
function initializeCharts() {
    const chartConfig = {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: {
                labels: {
                    color: '#a0aec0'
                }
            }
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#a0aec0'
                }
            },
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#a0aec0'
                }
            }
        }
    };
    
    // GA Chart
    charts.ga = new Chart(document.getElementById('ga-chart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Fitness',
                data: [],
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.4
            }]
        },
        options: chartConfig
    });
    
    // PSO Chart
    charts.pso = new Chart(document.getElementById('pso-chart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Cost',
                data: [],
                borderColor: '#f5576c',
                backgroundColor: 'rgba(245, 87, 108, 0.1)',
                tension: 0.4
            }]
        },
        options: chartConfig
    });
    
    // ACO Chart
    charts.aco = new Chart(document.getElementById('aco-chart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Quality',
                data: [],
                borderColor: '#00f2fe',
                backgroundColor: 'rgba(0, 242, 254, 0.1)',
                tension: 0.4
            }]
        },
        options: chartConfig
    });
    
    // Trajectory Chart
    charts.trajectory = new Chart(document.getElementById('trajectory-chart'), {
        type: 'scatter',
        data: {
            datasets: []
        },
        options: {
            ...chartConfig,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#a0aec0'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#a0aec0'
                    }
                }
            }
        }
    });
    
    // Generated Trajectories Chart
    charts.genTraj = new Chart(document.getElementById('gen-traj-chart'), {
        type: 'line',
        data: {
            datasets: []
        },
        options: chartConfig
    });
}

// Upload Functions
async function loadSampleData() {
    showToast('Loading sample data...', 'info');
    
    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ use_sample: true })
        });
        
        const data = await response.json();
        
        if (data.success) {
            totalFrames = data.n_frames;
            document.getElementById('frame-count').textContent = totalFrames;
            showToast(`Loaded ${totalFrames} frames successfully!`, 'success');
            switchView('tracking');
        } else {
            showToast('Failed to load sample data', 'error');
        }
    } catch (error) {
        showToast('Error loading sample data', 'error');
        console.error(error);
    }
}

async function handleFileUpload(event) {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;
    
    showToast(`Uploading ${files.length} images...`, 'info');
    
    // Convert files to base64
    const imagePromises = files.map(file => {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.readAsDataURL(file);
        });
    });
    
    const images = await Promise.all(imagePromises);
    
    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ images })
        });
        
        const data = await response.json();
        
        if (data.success) {
            totalFrames = data.n_frames;
            document.getElementById('frame-count').textContent = totalFrames;
            showToast(`Uploaded ${totalFrames} frames successfully!`, 'success');
            switchView('tracking');
        } else {
            showToast('Failed to upload images', 'error');
        }
    } catch (error) {
        showToast('Error uploading images', 'error');
        console.error(error);
    }
}

// Tracking Functions
async function runTracking() {
    const loadingOverlay = document.getElementById('tracking-loading');
    loadingOverlay.classList.add('active');
    
    showToast('Running cell tracking...', 'info');
    
    try {
        const response = await fetch(`${API_BASE}/track`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update statistics
            document.getElementById('total-tracks').textContent = data.n_tracks;
            document.getElementById('active-tracks').textContent = data.n_active_tracks;
            document.getElementById('track-count').textContent = data.n_active_tracks;
            
            if (data.tracks && data.tracks.length > 0) {
                const avgLength = data.tracks.reduce((sum, t) => sum + t.length, 0) / data.tracks.length;
                document.getElementById('avg-track-length').textContent = avgLength.toFixed(1);
                
                // Update track list
                updateTrackList(data.tracks);
            }
            
            // Display visualization
            if (data.visualization) {
                document.getElementById('tracking-canvas').src = data.visualization;
            }
            
            // Update frame controls
            document.getElementById('frame-slider').max = totalFrames - 1;
            currentFrame = 0;
            updateFrameDisplay();
            
            showToast(`Tracking complete! Found ${data.n_active_tracks} tracks`, 'success');
        } else {
            showToast('Tracking failed: ' + data.message, 'error');
        }
    } catch (error) {
        showToast('Error running tracking', 'error');
        console.error(error);
    } finally {
        loadingOverlay.classList.remove('active');
    }
}

function updateTrackList(tracks) {
    const trackList = document.getElementById('track-list');
    trackList.innerHTML = '';
    
    tracks.forEach(track => {
        const item = document.createElement('div');
        item.className = 'track-item';
        item.innerHTML = `
            <span class="track-id">Track ${track.id}</span>
            <span>${track.length} frames</span>
        `;
        trackList.appendChild(item);
    });
}

function togglePlayback() {
    isPlaying = !isPlaying;
    const playBtn = document.getElementById('play-btn');
    
    if (isPlaying) {
        playBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="4" width="4" height="16"/>
                <rect x="14" y="4" width="4" height="16"/>
            </svg>
        `;
        playInterval = setInterval(() => {
            changeFrame(1);
            if (currentFrame >= totalFrames - 1) {
                togglePlayback();
            }
        }, 200);
    } else {
        playBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
        `;
        clearInterval(playInterval);
    }
}

function changeFrame(delta) {
    currentFrame = Math.max(0, Math.min(totalFrames - 1, currentFrame + delta));
    updateFrame();
}

async function updateFrame() {
    document.getElementById('frame-slider').value = currentFrame;
    updateFrameDisplay();
    
    try {
        const response = await fetch(`${API_BASE}/get_frame/${currentFrame}`);
        const data = await response.json();
        
        if (data.success && data.visualization) {
            document.getElementById('tracking-canvas').src = data.visualization;
        }
    } catch (error) {
        console.error('Error updating frame:', error);
    }
}

function updateFrameDisplay() {
    document.getElementById('current-frame-display').textContent = 
        `Frame ${currentFrame} / ${totalFrames - 1}`;
}

// Optimization Functions
async function runGeneticAlgorithm() {
    const generations = parseInt(document.getElementById('ga-generations').value);
    const btn = document.getElementById('run-ga-btn');
    btn.disabled = true;
    btn.textContent = 'Running...';
    
    showToast('Running genetic algorithm...', 'info');
    
    try {
        const response = await fetch(`${API_BASE}/optimize/tracking`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ n_generations: generations })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update chart
            charts.ga.data.labels = Array.from({length: data.fitness_history.length}, (_, i) => i);
            charts.ga.data.datasets[0].data = data.fitness_history;
            charts.ga.update();
            
            // Display result
            document.getElementById('ga-result').innerHTML = `
                <strong>Best Fitness:</strong> ${data.best_fitness.toFixed(4)}<br>
                <strong>Generations:</strong> ${data.n_generations}
            `;
            
            showToast('GA optimization complete!', 'success');
        } else {
            showToast('GA optimization failed', 'error');
        }
    } catch (error) {
        showToast('Error running GA', 'error');
        console.error(error);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run GA Optimization';
    }
}

async function runPSO() {
    const iterations = parseInt(document.getElementById('pso-iterations').value);
    const btn = document.getElementById('run-pso-btn');
    btn.disabled = true;
    btn.textContent = 'Running...';
    
    showToast('Running PSO...', 'info');
    
    try {
        const response = await fetch(`${API_BASE}/optimize/parameters`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ max_iter: iterations })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update chart
            charts.pso.data.labels = Array.from({length: data.cost_history.length}, (_, i) => i);
            charts.pso.data.datasets[0].data = data.cost_history;
            charts.pso.update();
            
            // Display result
            document.getElementById('pso-result').innerHTML = `
                <strong>Final Cost:</strong> ${data.final_cost.toFixed(4)}<br>
                <strong>Best Parameters:</strong> [${data.best_parameters.map(p => p.toFixed(3)).join(', ')}]
            `;
            
            showToast('PSO optimization complete!', 'success');
        } else {
            showToast('PSO optimization failed', 'error');
        }
    } catch (error) {
        showToast('Error running PSO', 'error');
        console.error(error);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run PSO Optimization';
    }
}

async function runACO() {
    const iterations = parseInt(document.getElementById('aco-iterations').value);
    const btn = document.getElementById('run-aco-btn');
    btn.disabled = true;
    btn.textContent = 'Running...';
    
    showToast('Running ACO lineage reconstruction...', 'info');
    
    try {
        const response = await fetch(`${API_BASE}/optimize/lineage`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ iterations })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update chart
            charts.aco.data.labels = Array.from({length: data.quality_history.length}, (_, i) => i);
            charts.aco.data.datasets[0].data = data.quality_history;
            charts.aco.update();
            
            // Display result
            document.getElementById('aco-result').innerHTML = `
                <strong>Final Quality:</strong> ${data.final_quality.toFixed(4)}<br>
                <strong>Lineage Edges:</strong> ${data.lineage_edges.length}
            `;
            
            // Update lineage tree visualization
            visualizeLineageTree(data.lineage_edges);
            
            showToast('ACO lineage reconstruction complete!', 'success');
        } else {
            showToast('ACO optimization failed', 'error');
        }
    } catch (error) {
        showToast('Error running ACO', 'error');
        console.error(error);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run ACO Lineage';
    }
}

async function runHybridOptimization() {
    const btn = document.getElementById('run-hybrid-btn');
    btn.disabled = true;
    btn.textContent = 'Running...';
    
    showToast('Running hybrid optimization...', 'info');
    
    // Update status badges
    document.getElementById('hybrid-ga-status').textContent = 'Running';
    document.getElementById('hybrid-pso-status').textContent = 'Pending';
    document.getElementById('hybrid-aco-status').textContent = 'Pending';
    
    try {
        // Run GA
        await runGeneticAlgorithm();
        document.getElementById('hybrid-ga-status').textContent = 'Complete';
        document.getElementById('hybrid-pso-status').textContent = 'Running';
        
        // Run PSO
        await runPSO();
        document.getElementById('hybrid-pso-status').textContent = 'Complete';
        document.getElementById('hybrid-aco-status').textContent = 'Running';
        
        // Run ACO
        await runACO();
        document.getElementById('hybrid-aco-status').textContent = 'Complete';
        
        showToast('Hybrid optimization complete!', 'success');
    } catch (error) {
        showToast('Error in hybrid optimization', 'error');
        console.error(error);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Hybrid Optimization';
    }
}

// Differentiation Functions
async function analyzeDifferentiation() {
    const btn = document.getElementById('analyze-diff-btn');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';
    
    showToast('Analyzing differentiation trajectories...', 'info');
    
    try {
        const response = await fetch(`${API_BASE}/differentiation/analyze`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            const stats = data.statistics;
            
            // Update metrics
            const metrics = document.getElementById('diff-metrics');
            metrics.innerHTML = `
                <div class="metric-card">
                    <span class="metric-label">Mean Rate</span>
                    <span class="metric-value">${stats.mean_rate?.toFixed(3) || '--'}</span>
                </div>
                <div class="metric-card">
                    <span class="metric-label">Tortuosity</span>
                    <span class="metric-value">${stats.mean_tortuosity?.toFixed(3) || '--'}</span>
                </div>
                <div class="metric-card">
                    <span class="metric-label">Path Length</span>
                    <span class="metric-value">${stats.mean_path_length?.toFixed(2) || '--'}</span>
                </div>
                <div class="metric-card">
                    <span class="metric-label">Trajectories</span>
                    <span class="metric-value">${stats.n_trajectories || 0}</span>
                </div>
            `;
            
            // Update trajectory chart
            if (data.trajectories && data.trajectories.length > 0) {
                const colors = ['#667eea', '#f5576c', '#00f2fe', '#38f9d7', '#f093fb'];
                
                charts.trajectory.data.datasets = data.trajectories.map((traj, idx) => {
                    const cluster = data.clusters ? data.clusters[idx] : 0;
                    return {
                        label: `Track ${idx} (Type ${cluster})`,
                        data: traj.map(p => ({ x: p[0], y: p[1] })),
                        borderColor: colors[cluster % colors.length],
                        backgroundColor: colors[cluster % colors.length],
                        pointRadius: 2,
                        showLine: true,
                        tension: 0.4
                    };
                });
                charts.trajectory.update();
            }
            
            showToast('Analysis complete!', 'success');
        } else {
            showToast('Analysis failed: ' + data.message, 'error');
        }
    } catch (error) {
        showToast('Error analyzing differentiation', 'error');
        console.error(error);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Analyze Trajectories';
    }
}

async function predictCellFate() {
    const steps = parseInt(document.getElementById('predict-steps').value);
    const btn = document.getElementById('predict-fate-btn');
    btn.disabled = true;
    btn.textContent = 'Predicting...';
    
    showToast('Predicting cell fate...', 'info');
    
    try {
        const response = await fetch(`${API_BASE}/differentiation/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ n_steps: steps })
        });
        
        const data = await response.json();
        
        if (data.success) {
            const prediction = data.prediction;
            
            // Display prediction
            const resultDiv = document.getElementById('fate-prediction');
            resultDiv.innerHTML = '<h4>Predicted Cell Fate Trajectory:</h4>';
            
            prediction.cell_fate.forEach((fate, idx) => {
                const confidence = (prediction.confidence[idx] * 100).toFixed(1);
                resultDiv.innerHTML += `
                    <div style="padding: 0.5rem; margin: 0.5rem 0; background: var(--bg-tertiary); border-radius: 8px;">
                        Step ${idx + 1}: <strong>${fate}</strong> (${confidence}% confidence)
                    </div>
                `;
            });
            
            showToast('Prediction complete!', 'success');
        } else {
            showToast('Prediction failed', 'error');
        }
    } catch (error) {
        showToast('Error predicting cell fate', 'error');
        console.error(error);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Predict Cell Fate';
    }
}

function visualizeLineageTree(edges) {
    const treeDiv = document.getElementById('lineage-tree');
    treeDiv.innerHTML = '<h4>Lineage Tree Structure:</h4>';
    
    edges.forEach(([parent, daughter]) => {
        treeDiv.innerHTML += `
            <div style="padding: 0.5rem; margin: 0.5rem 0; background: var(--bg-tertiary); border-radius: 8px;">
                Cell ${parent} → Cell ${daughter}
            </div>
        `;
    });
}

// Generative Functions
async function generateTrajectories() {
    const samples = parseInt(document.getElementById('gen-traj-samples').value);
    const length = parseInt(document.getElementById('gen-traj-length').value);
    const btn = document.getElementById('gen-traj-btn');
    btn.disabled = true;
    btn.textContent = 'Generating...';
    
    showToast('Generating synthetic trajectories...', 'info');
    
    try {
        const response = await fetch(`${API_BASE}/generate/trajectories`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ n_samples: samples, length })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update chart with generated trajectories
            const colors = ['#667eea', '#f5576c', '#00f2fe', '#38f9d7', '#f093fb'];
            
            charts.genTraj.data.datasets = data.trajectories.map((traj, idx) => ({
                label: `Trajectory ${idx + 1}`,
                data: traj.map((point, t) => ({ x: t, y: point[0] })),
                borderColor: colors[idx % colors.length],
                backgroundColor: 'transparent',
                tension: 0.4
            }));
            
            charts.genTraj.update();
            
            showToast(`Generated ${samples} trajectories!`, 'success');
        } else {
            showToast('Generation failed', 'error');
        }
    } catch (error) {
        showToast('Error generating trajectories', 'error');
        console.error(error);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Generate Trajectories';
    }
}

async function generateMorphologies() {
    const samples = parseInt(document.getElementById('gen-morph-samples').value);
    const btn = document.getElementById('gen-morph-btn');
    btn.disabled = true;
    btn.textContent = 'Generating...';
    
    showToast('Generating cell morphologies...', 'info');
    
    try {
        const response = await fetch(`${API_BASE}/generate/morphology`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ n_samples: samples })
        });
        
        const data = await response.json();
        
        if (data.success) {
            const gallery = document.getElementById('morph-gallery');
            gallery.innerHTML = '';
            
            data.morphologies.forEach((morphImg, idx) => {
                const item = document.createElement('div');
                item.className = 'morphology-item';
                item.innerHTML = `<img src="${morphImg}" alt="Morphology ${idx + 1}">`;
                gallery.appendChild(item);
            });
            
            showToast(`Generated ${samples} morphologies!`, 'success');
        } else {
            showToast('Generation failed', 'error');
        }
    } catch (error) {
        showToast('Error generating morphologies', 'error');
        console.error(error);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Generate Morphologies';
    }
}

// Toast Notifications
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
