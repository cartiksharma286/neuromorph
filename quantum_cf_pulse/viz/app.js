async function loadData() {
    const response = await fetch('pulse_data.json');
    const data = await response.json();
    return data;
}

function plotPulse(data) {
    const time = data.time.slice(0, -1); // Remove last time point for controls
    
    const traceX = {
        x: time,
        y: data.omega_x,
        mode: 'lines',
        name: 'Omega X',
        line: { color: '#1f77b4' }
    };
    
    const traceY = {
        x: time,
        y: data.omega_y,
        mode: 'lines',
        name: 'Omega Y',
        line: { color: '#ff7f0e' }
    };
    
    const layout = {
        title: 'Control Fields vs Time',
        xaxis: { title: 'Time (a.u.)' },
        yaxis: { title: 'Amplitude (rad/s)' }
    };
    
    Plotly.newPlot('pulse-shape', [traceX, traceY], layout);
}

function plotBlochSphere(data) {
    const x = data.bloch_trajectory.map(p => p[0]);
    const y = data.bloch_trajectory.map(p => p[1]);
    const z = data.bloch_trajectory.map(p => p[2]);
    
    // Create a sphere mesh
    const u = [];
    const v = [];
    for (let i = 0; i <= 20; i++) {
        u.push(i * Math.PI / 10);
        v.push(i * Math.PI / 10);
    }
    
    // Trajectory trace
    const traceTrajectory = {
        x: x,
        y: y,
        z: z,
        mode: 'lines+markers',
        marker: { size: 4, color: z, colorscale: 'Viridis' },
        line: { width: 5, color: 'black' },
        type: 'scatter3d',
        name: 'Spin Trajectory'
    };
    
    // Start point
    const traceStart = {
        x: [x[0]], y: [y[0]], z: [z[0]],
        mode: 'markers',
        marker: { size: 8, color: 'green' },
        type: 'scatter3d',
        name: 'Start'
    };
    
    // End point
    const traceEnd = {
        x: [x[x.length-1]], y: [y[y.length-1]], z: [z[z.length-1]],
        mode: 'markers',
        marker: { size: 8, color: 'red' },
        type: 'scatter3d',
        name: 'End'
    };

    const layout = {
        title: 'Bloch Sphere Trajectory',
        scene: {
            xaxis: { title: 'X', range: [-1, 1] },
            yaxis: { title: 'Y', range: [-1, 1] },
            zaxis: { title: 'Z', range: [-1, 1] },
            aspectmode: 'cube'
        }
    };
    
    Plotly.newPlot('bloch-sphere', [traceTrajectory, traceStart, traceEnd], layout);
}

async function init() {
    try {
        const data = await loadData();
        plotPulse(data);
        plotBlochSphere(data);
        console.log("Fidelity:", data.fidelity);
    } catch (e) {
        console.error("Error loading data:", e);
    }
}

init();
