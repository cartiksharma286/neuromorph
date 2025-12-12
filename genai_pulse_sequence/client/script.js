
// State
let blochScene, blochCamera, blochRenderer, mVector, mTrail;
let trailPoints = [];
const API_URL = 'http://localhost:8001';

// Initialize Three.js Scene
function initBlochSphere() {
    const container = document.getElementById('bloch-container');
    const w = container.clientWidth;
    const h = container.clientHeight;

    blochScene = new THREE.Scene();
    blochCamera = new THREE.PerspectiveCamera(50, w / h, 0.1, 1000);
    blochCamera.position.z = 2.5;
    blochCamera.position.y = 1;
    blochCamera.lookAt(0, 0, 0);

    blochRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    blochRenderer.setSize(w, h);
    container.appendChild(blochRenderer.domElement);

    // Sphere
    const geometry = new THREE.SphereGeometry(1, 32, 32);
    const material = new THREE.MeshBasicMaterial({ color: 0x444444, wireframe: true, transparent: true, opacity: 0.3 });
    const sphere = new THREE.Mesh(geometry, material);
    blochScene.add(sphere);

    // Axes
    const axesHelper = new THREE.AxesHelper(1.2);
    blochScene.add(axesHelper);

    // Magnetization Vector
    const arrowDir = new THREE.Vector3(0, 1, 0);
    const origin = new THREE.Vector3(0, 0, 0);
    const length = 1;
    const hex = 0xffff00; // Yellow
    mVector = new THREE.ArrowHelper(arrowDir, origin, length, hex, 0.2, 0.1);
    blochScene.add(mVector);

    // Trail
    const trailGeo = new THREE.BufferGeometry();
    const trailMat = new THREE.LineBasicMaterial({ color: 0x00ffff });
    mTrail = new THREE.Line(trailGeo, trailMat);
    blochScene.add(mTrail);

    animate();
}

function animate() {
    requestAnimationFrame(animate);
    blochRenderer.render(blochScene, blochCamera);
}

// Plotly Setup
function initPlots() {
    const layoutDefaults = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8' },
        margin: { t: 10, b: 30, l: 40, r: 10 },
        xaxis: { showgrid: false },
        yaxis: { showgrid: false }
    };

    Plotly.newPlot('diffusion-plot', [{ y: [], type: 'scatter', mode: 'lines', line: { color: '#5b21b6' } }],
        { ...layoutDefaults, title: { text: '' }, xaxis: { title: 'Time' }, yaxis: { range: [-3, 3] } });

    Plotly.newPlot('pulse-plot', [{ y: [], type: 'scatter', fill: 'tozeroy', line: { color: '#22d3ee' } }],
        { ...layoutDefaults, margin: { t: 5, b: 20, l: 30, r: 5 } });

    Plotly.newPlot('snr-plot', [{ y: [], type: 'scatter', mode: 'lines+markers', line: { color: '#10b981' } }],
        { ...layoutDefaults, margin: { t: 10, b: 30, l: 40, r: 10 } });
}

// Data Fetching & Animation
async function generatePulse() {
    const btn = document.getElementById('btn-generate');
    btn.disabled = true;
    btn.innerHTML = `<svg class="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24"></svg> Generating...`;

    try {
        const res = await fetch(`${API_URL}/generate`);
        const data = await res.json();

        // Update stats
        document.getElementById('val-snr').innerText = data.simulation.snr.toFixed(2);

        // Calculate Flip Angle (approx from last Mz)
        const finalMz = data.simulation.final_magnetization[2];
        const angle = Math.acos(finalMz) * (180 / Math.PI);
        document.getElementById('val-angle').innerText = angle.toFixed(1) + '°';

        document.getElementById('val-bw').innerText = data.metadata.target_bw.toFixed(2);
        document.getElementById('val-amp').innerText = data.metadata.target_amp.toFixed(2);

        // Animate Diffusion Process
        const steps = data.diffusion_steps;
        for (let i = 0; i < steps.length; i++) {
            Plotly.react('diffusion-plot', [{ y: steps[i], type: 'scatter', mode: 'lines', line: { color: '#818cf8', width: 2 } }],
                { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#94a3b8' }, xaxis: { showgrid: false }, yaxis: { range: [-3, 3] } });
            await new Promise(r => setTimeout(r, 50)); // Delay for animation
        }

        // Show final pulse
        Plotly.react('pulse-plot', [{ y: data.pulse, type: 'scatter', fill: 'tozeroy', line: { color: '#22d3ee' } }],
            { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#94a3b8' }, margin: { t: 5, b: 20, l: 30, r: 5 } });

        // Update Bloch Sphere
        updateBlochSphere(data.simulation.trajectory);

    } catch (e) {
        console.error(e);
        alert("Server Logic Error or Connection Failed");
    }

    btn.disabled = false;
    btn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg> Generate Pulse`;
}

function updateBlochSphere(trajectory) {
    // trajectory is Nx3 array
    trailPoints = [];

    let i = 0;
    function animateTrajectory() {
        if (i >= trajectory.length) return;

        const pt = trajectory[i];
        // Bloch coord swap: Z is up in ThreeJS usually Y
        // MRI Physics: Z is main field. So let's map MRI(x,y,z) -> Three(x,z,y) or similar.
        // Let's use MRI Z -> Three Y (Up). MRI X -> Three X. MRI Y -> Three Z.

        const vec = new THREE.Vector3(pt[0], pt[2], pt[1]);
        mVector.setDirection(vec.clone().normalize());

        trailPoints.push(vec);
        const geo = new THREE.BufferGeometry().setFromPoints(trailPoints);
        mTrail.geometry.dispose();
        mTrail.geometry = geo;

        i += Math.ceil(trajectory.length / 50); // Speed up

        blochRenderer.render(blochScene, blochCamera);
        requestAnimationFrame(animateTrajectory);
    }
    animateTrajectory();
}

// Optimization Loop
let snrHistory = [];
async function runOptimization() {
    const btn = document.getElementById('btn-optimize');
    btn.disabled = true;
    const bar = document.getElementById('opt-bar');
    const label = document.getElementById('opt-progress');

    label.innerText = "Running Adaptive Loop...";
    bar.style.width = "0%";

    try {
        const res = await fetch(`${API_URL}/optimize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ iterations: 10 })
        });
        const data = await res.json();

        bar.style.width = "100%";
        label.innerText = "Optimization Complete";

        // Plot SNR history
        data.logs.forEach(log => snrHistory.push(log.snr));
        if (snrHistory.length > 50) snrHistory = snrHistory.slice(-50);

        Plotly.react('snr-plot', [{ y: snrHistory, type: 'scatter', mode: 'lines+markers', line: { color: '#10b981' } }],
            { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#94a3b8' }, margin: { t: 10, b: 30, l: 40, r: 10 } });
        // Trigger a final generation to show the result
        generatePulse();

    } catch (e) {
        console.error(e);
    }

    setTimeout(() => {
        bar.style.width = "0%";
        label.innerText = "Idle";
        btn.disabled = false;
    }, 2000);
}

async function updateReconstruction() {
    try {
        const res = await fetch(`${API_URL}/reconstruct`);
        const data = await res.json();
        const img = document.getElementById('recon-img');
        img.src = `data:image/png;base64,${data.image}`;
    } catch (e) {
        console.error("Recon error", e);
    }
}

async function exportSeq() {
    try {
        const res = await fetch(`${API_URL}/export_seq`);
        const data = await res.json();

        // Create hidden link to download
        const element = document.createElement('a');
        element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(data.content));
        element.setAttribute('download', data.filename);
        element.style.display = 'none';
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);

    } catch (e) {
        console.error(e);
        alert("Export Failed");
    }
}

// Event Listeners
window.onload = function () {
    initBlochSphere();
    initPlots();

    document.getElementById('btn-generate').onclick = async () => {
        await generatePulse();
        updateReconstruction();
    };
    document.getElementById('btn-optimize').onclick = async () => {
        await runOptimization();
        updateReconstruction(); // Update image after optimization loop final generation
    };
    document.getElementById('btn-export').onclick = exportSeq;

    // Initial Run
    generatePulse().then(updateReconstruction);

    window.addEventListener('resize', () => {
        const container = document.getElementById('bloch-container');
        blochRenderer.setSize(container.clientWidth, container.clientHeight);
        blochCamera.aspect = container.clientWidth / container.clientHeight;
        blochCamera.updateProjectionMatrix();
        Plotly.Plots.resize('diffusion-plot');
        Plotly.Plots.resize('pulse-plot');
        Plotly.Plots.resize('snr-plot');
    });
};
