// ============================================
// Quantum Neuroimaging - Interactive UI
// ============================================

// State Management
const state = {
    currentSequence: 'gre',
    parameters: {
        TE: 30,
        TR: 500,
        FA: 30,
        matrix_size: 256
    },
    optimization: {
        quantum: true,
        adaptive: true,
        motionCorrection: true
    },
    isOptimizing: false,
    motionData: []
};

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initializeControls();
    initializeVisualizations();
    startRealtimeFeedback();
    updateMetrics();
});

// ============================================
// Control Initialization
// ============================================

function initializeControls() {
    // Sequence selection
    document.querySelectorAll('.sequence-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.sequence-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.currentSequence = btn.dataset.sequence;
            updateSequenceDefaults();
            updateMetrics();
        });
    });

    // Parameter sliders
    const sliders = [
        { id: 'te', param: 'TE' },
        { id: 'tr', param: 'TR' },
        { id: 'fa', param: 'FA' },
        { id: 'matrix', param: 'matrix_size' }
    ];

    sliders.forEach(({ id, param }) => {
        const slider = document.getElementById(`${id}-slider`);
        const valueDisplay = document.getElementById(`${id}-value`);

        slider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            state.parameters[param] = value;
            valueDisplay.textContent = value;
            updateMetrics();
            updateKSpaceVisualization();
        });
    });

    // Optimization toggles
    document.getElementById('quantum-opt').addEventListener('change', (e) => {
        state.optimization.quantum = e.target.checked;
    });

    document.getElementById('adaptive-opt').addEventListener('change', (e) => {
        state.optimization.adaptive = e.target.checked;
    });

    document.getElementById('motion-correction').addEventListener('change', (e) => {
        state.optimization.motionCorrection = e.target.checked;
    });

    // Optimize button
    document.getElementById('optimize-btn').addEventListener('click', runOptimization);
}

// ============================================
// Sequence Defaults
// ============================================

function updateSequenceDefaults() {
    const defaults = {
        gre: { TE: 10, TR: 500, FA: 30, matrix_size: 256 },
        se: { TE: 80, TR: 2000, FA: 90, matrix_size: 256 },
        epi: { TE: 30, TR: 2000, FA: 90, matrix_size: 64 },
        fmri: { TE: 30, TR: 2000, FA: 90, matrix_size: 64 }
    };

    const params = defaults[state.currentSequence];
    Object.keys(params).forEach(key => {
        state.parameters[key] = params[key];
        const slider = document.getElementById(`${key.toLowerCase().replace('_', '-')}-slider`);
        const valueDisplay = document.getElementById(`${key.toLowerCase().replace('_', '-')}-value`);
        if (slider) {
            slider.value = params[key];
            valueDisplay.textContent = params[key];
        }
    });
}

// ============================================
// Metrics Calculation
// ============================================

function calculateSNR() {
    const { TE, TR, FA } = state.parameters;
    // Simplified SNR model
    const snr = (1000 / TE) * Math.sin(FA * Math.PI / 180) * (TR / 1000);
    return snr.toFixed(1);
}

function calculateScanTime() {
    const { TR, matrix_size } = state.parameters;
    const timeSeconds = (TR * matrix_size) / 1000;
    const minutes = Math.floor(timeSeconds / 60);
    const seconds = Math.floor(timeSeconds % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function updateMetrics() {
    // Update SNR
    document.getElementById('snr-value').textContent = calculateSNR();
    
    // Update scan time
    document.getElementById('scan-time').textContent = calculateScanTime();
    
    // Update SNR mini chart
    updateSNRChart();
}

// ============================================
// Optimization
// ============================================

async function runOptimization() {
    if (state.isOptimizing) return;
    
    state.isOptimizing = true;
    const btn = document.getElementById('optimize-btn');
    btn.textContent = 'Optimizing...';
    btn.style.opacity = '0.6';
    
    // Show optimization progress
    showOptimizationProgress();
    
    // Simulate quantum optimization (in real app, would call Python backend)
    await simulateOptimization();
    
    state.isOptimizing = false;
    btn.innerHTML = '<span class="btn-icon">⚛️</span>Optimize Sequence';
    btn.style.opacity = '1';
    
    // Show success notification
    addFeedbackEvent('Optimization complete', 'success');
}

async function simulateOptimization() {
    const steps = 20;
    const canvas = document.getElementById('optimization-canvas');
    const ctx = canvas.getContext('2d');
    
    for (let i = 0; i <= steps; i++) {
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Gradually improve parameters
        if (state.optimization.quantum) {
            state.parameters.TE *= 0.98;
            state.parameters.FA = Math.min(state.parameters.FA * 1.01, 90);
        }
        
        updateMetrics();
        drawOptimizationProgress(ctx, i / steps);
    }
}

function drawOptimizationProgress(ctx, progress) {
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;
    
    ctx.clearRect(0, 0, width, height);
    
    // Draw convergence curve
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    for (let i = 0; i <= progress * 100; i++) {
        const x = (i / 100) * width;
        const y = height * (1 - Math.exp(-i / 20) * (1 - Math.random() * 0.1));
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.stroke();
}

function showOptimizationProgress() {
    const canvas = document.getElementById('optimization-canvas');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
}

// ============================================
// Visualizations
// ============================================

function initializeVisualizations() {
    updateKSpaceVisualization();
    initializeOptimizationChart();
    initializeSNRChart();
    initializeQuantumCircuit();
}

function updateKSpaceVisualization() {
    const canvas = document.getElementById('kspace-canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    
    ctx.clearRect(0, 0, width, height);
    
    // Draw k-space trajectory based on sequence type
    const trajectories = {
        gre: drawCartesianTrajectory,
        se: drawCartesianTrajectory,
        epi: drawEPITrajectory,
        fmri: drawEPITrajectory
    };
    
    trajectories[state.currentSequence](ctx, centerX, centerY, width, height);
}

function drawCartesianTrajectory(ctx, centerX, centerY, width, height) {
    const lines = state.parameters.matrix_size / 4;
    const spacing = Math.min(width, height) * 0.8 / lines;
    
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.6)';
    ctx.lineWidth = 2;
    
    for (let i = 0; i < lines; i++) {
        const y = centerY - (lines / 2 - i) * spacing;
        
        ctx.beginPath();
        ctx.moveTo(centerX - width * 0.4, y);
        ctx.lineTo(centerX + width * 0.4, y);
        ctx.stroke();
        
        // Add glow effect
        ctx.strokeStyle = `rgba(0, 212, 255, ${0.3 - i / lines * 0.2})`;
        ctx.lineWidth = 4;
        ctx.stroke();
        
        ctx.strokeStyle = 'rgba(0, 212, 255, 0.6)';
        ctx.lineWidth = 2;
    }
}

function drawEPITrajectory(ctx, centerX, centerY, width, height) {
    const lines = state.parameters.matrix_size / 4;
    const spacing = Math.min(width, height) * 0.8 / lines;
    
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let i = 0; i < lines; i++) {
        const y = centerY - (lines / 2 - i) * spacing;
        
        if (i % 2 === 0) {
            if (i === 0) ctx.moveTo(centerX - width * 0.4, y);
            else ctx.lineTo(centerX - width * 0.4, y);
            ctx.lineTo(centerX + width * 0.4, y);
        } else {
            ctx.lineTo(centerX + width * 0.4, y);
            ctx.lineTo(centerX - width * 0.4, y);
        }
    }
    
    ctx.stroke();
    
    // Add arrow at end
    ctx.fillStyle = '#00d4ff';
    ctx.beginPath();
    ctx.arc(centerX - width * 0.4, centerY + (lines / 2) * spacing, 5, 0, Math.PI * 2);
    ctx.fill();
}

function initializeOptimizationChart() {
    const canvas = document.getElementById('optimization-canvas');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
}

function initializeSNRChart() {
    updateSNRChart();
}

function updateSNRChart() {
    const container = document.getElementById('snr-chart');
    container.innerHTML = '';
    
    // Create mini sparkline
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '60');
    svg.style. cssText = 'display: block;';
    
    const points = [];
    for (let i = 0; i < 20; i++) {
        const x = (i / 19) * 100;
        const baseY = 50 - (calculateSNR() / 100 * 40);
        const y = baseY + Math.random() * 5;
        points.push(`${x},${y}`);
    }
    
    const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    polyline.setAttribute('points', points.join(' '));
    polyline.setAttribute('fill', 'none');
    polyline.setAttribute('stroke', '#00d4ff');
    polyline.setAttribute('stroke-width', '2');
    polyline.style.vectorEffect = 'non-scaling-stroke';
    
    svg.appendChild(polyline);
    container.appendChild(svg);
}

function initializeQuantumCircuit() {
    const container = document.getElementById('circuit-viz');
    
    // Create simple quantum circuit visualization
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '80');
    
    // Draw qubits
    for (let i = 0; i < 6; i++) {
        const y = 10 + i * 12;
        
        // Qubit line
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', '10');
        line.setAttribute('y1', y);
        line.setAttribute('x2', '90%');
        line.setAttribute('y2', y);
        line.setAttribute('stroke', 'rgba(0, 212, 255, 0.4)');
        line.setAttribute('stroke-width', '1');
        svg.appendChild(line);
        
        // Gates
        for (let j = 0; j < 3; j++) {
            const x = 30 + j * 60;
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', x);
            rect.setAttribute('y', y - 4);
            rect.setAttribute('width', '8');
            rect.setAttribute('height', '8');
            rect.setAttribute('fill', '#a855f7');
            rect.setAttribute('rx', '2');
            svg.appendChild(rect);
        }
    }
    
    container.appendChild(svg);
}

// ============================================
// Real-time Feedback
// ============================================

function startRealtimeFeedback() {
    setInterval(() => {
        simulateMotionDetection();
        updateMotionDisplay();
    }, 1000 / 10); // 10 Hz display update (simulating 200 Hz backend)
}

function simulateMotionDetection() {
    const motion = {
        timestamp: Date.now(),
        magnitude: Math.random() * 0.8,
        x: (Math.random() - 0.5) * 0.6,
        y: (Math.random() - 0.5) * 0.6,
        z: (Math.random() - 0.5) * 0.4
    };
    
    state.motionData.push(motion);
    if (state.motionData.length > 100) state.motionData.shift();
    
    // Detect significant motion events
    if (motion.magnitude > 0.5 && state.optimization.motionCorrection) {
        addFeedbackEvent(`Motion detected: ${motion.magnitude.toFixed(2)}mm - Correcting`, 'warning');
    }
}

function updateMotionDisplay() {
    if (state.motionData.length === 0) return;
    
    const latestMotion = state.motionData[state.motionData.length - 1];
    const magnitude = latestMotion.magnitude;
    
    document.getElementById('motion-magnitude').textContent = `${magnitude.toFixed(2)} mm`;
    
    const motionFill = document.querySelector('.motion-fill');
    motionFill.style.width = `${Math.min(magnitude / 2 * 100, 100)}%`;
    
    // Update status
    const statusEl = document.querySelector('.motion-status');
    if (magnitude < 0.5) {
        statusEl.textContent = 'Within Threshold';
        statusEl.className = 'motion-status status-good';
        motionFill.style.background = '#10b981';
    } else if (magnitude < 1.5) {
        statusEl.textContent = 'Correcting';
        statusEl.className = 'motion-status';
        statusEl.style.color = '#f59e0b';
        statusEl.style.background = 'rgba(245, 158, 11, 0.1)';
        motionFill.style.background = '#f59e0b';
    } else {
        statusEl.textContent = 'Reacquisition Required';
        statusEl.className = 'motion-status';
        statusEl.style.color = '#ef4444';
        statusEl.style.background = 'rgba(239, 68, 68, 0.1)';
        motionFill.style.background = '#ef4444';
    }
}

function addFeedbackEvent(message, type = 'info') {
    const timeline = document.getElementById('feedback-timeline');
    const event = document.createElement('div');
    event.className = 'feedback-event';
    event.style.cssText = `
        padding: 0.75rem;
        background: rgba(31, 41, 55, 0.6);
        border-left: 3px solid ${type === 'success' ? '#10b981' : type === 'warning' ? '#f59e0b' : '#00d4ff'};
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
        animation: slideIn 0.3s ease;
    `;
    
    const time = new Date().toLocaleTimeString();
    event.innerHTML = `
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span style="color: #d1d5db; font-weight: 500;">${message}</span>
            <span style="color: #9ca3af; font-size: 0.75rem;">${time}</span>
        </div>
    `;
    
    timeline.insertBefore(event, timeline.firstChild);
    
    // Keep only last 10 events
    while (timeline.children.length > 10) {
        timeline.removeChild(timeline.lastChild);
    }
}

// Add CSS animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
`;
document.head.appendChild(style);

// ============================================
// Initialize feedback events
// ============================================

setTimeout(() => {
    addFeedbackEvent('Quantum feedback system initialized', 'success');
    addFeedbackEvent('NVQLink connection established - <0.5ms latency', 'success');
    addFeedbackEvent('Motion tracking active at 200 Hz', 'info');
}, 500);

console.log('Quantum Neuroimaging UI initialized successfully!');
