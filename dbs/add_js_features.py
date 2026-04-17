import os

js_file = "/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js"
with open(js_file, "r") as f:
    js = f.read()

new_js = """

// --- Added Dementia Optimization & FEA Functionality ---

let dementiaChartInstance = null;

function updateDementiaChart() {
    const dbsAmp = document.getElementById('voltage-range') ? document.getElementById('voltage-range').value : 30; // mapping voltage to amplitude 
    const declineRate = document.getElementById('dementia-decline-range') ? document.getElementById('dementia-decline-range').value : 0.05;
    const prompt = document.getElementById('dementia-prompt') ? document.getElementById('dementia-prompt').value : 'baseline';
    
    fetch('/api/dementia-staging', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            dbs_amplitude: dbsAmp, 
            decline_rate: declineRate,
            prompt: prompt
        })
    })
    .then(res => res.json())
    .then(data => {
        const ctx = document.getElementById('dementia-chart')?.getContext('2d');
        if (!ctx) return;
        
        if (dementiaChartInstance) {
            dementiaChartInstance.destroy();
        }
        
        const upperBounds = data.cognitive_trajectory.map((val, i) => val + data.clinical_distributions[i].std);
        const lowerBounds = data.cognitive_trajectory.map((val, i) => Math.max(0, val - data.clinical_distributions[i].std));
        
        dementiaChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.time_months,
                datasets: [
                    {
                        label: 'Mean Trajectory',
                        data: data.cognitive_trajectory,
                        borderColor: '#00ffcc',
                        backgroundColor: 'rgba(0, 255, 204, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: '+1 Std Dev',
                        data: upperBounds,
                        borderColor: 'rgba(255, 0, 127, 0.5)',
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0,
                    },
                    {
                        label: '-1 Std Dev',
                        data: lowerBounds,
                        borderColor: 'rgba(255, 0, 127, 0.5)',
                        borderDash: [5, 5],
                        fill: '-1',
                        backgroundColor: 'rgba(255, 0, 127, 0.1)',
                        pointRadius: 0,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Months' } },
                    y: { title: { display: true, text: 'Cognitive Score (MMSE)' }, min: 0 }
                }
            }
        });
    }).catch(e => console.error("Error charting dementia:", e));
}

let largeFeaScene, largeFeaCamera, largeFeaRenderer, largeFeaMesh;

function initLargerFEA() {
    const container = document.getElementById('fea-large-container');
    if (!container) return;
    
    // Clear past children
    container.innerHTML = '';
    
    largeFeaScene = new THREE.Scene();
    largeFeaCamera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
    largeFeaCamera.position.z = 15;
    
    largeFeaRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    largeFeaRenderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(largeFeaRenderer.domElement);
    
    // Build cortical manifold (Icosahedron as proxy)
    const feaNodes = document.getElementById('fea-nodes') ? parseInt(document.getElementById('fea-nodes').value) : 1000;
    const detail = Math.min(6, Math.max(1, Math.floor(feaNodes / 200)));
    
    const geo = new THREE.IcosahedronGeometry(7, detail);
    
    // Deform geometry
    const positions = geo.attributes.position;
    for(let i = 0; i < positions.count; i++) {
        let x = positions.getX(i);
        let y = positions.getY(i);
        let z = positions.getZ(i);
        let len = Math.sqrt(x*x + y*y + z*z);
        // bump noise
        let bump = 1 + 0.2 * Math.sin(x*2) * Math.cos(y*2);
        positions.setXYZ(i, x*bump, y*bump, z*bump);
    }
    geo.computeVertexNormals();
    
    const mat = new THREE.MeshPhongMaterial({
        color: 0x00ffcc,
        emissive: 0x111111,
        wireframe: true,
        side: THREE.DoubleSide
    });
    
    largeFeaMesh = new THREE.Mesh(geo, mat);
    largeFeaScene.add(largeFeaMesh);
    
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(10, 10, 10);
    largeFeaScene.add(light);
    
    const ambientLight = new THREE.AmbientLight(0x404040); // soft white light
    largeFeaScene.add(ambientLight);
    
    animateLargeFEA();
}

function animateLargeFEA() {
    if(!largeFeaRenderer) return;
    requestAnimationFrame(animateLargeFEA);
    if(largeFeaMesh) {
        largeFeaMesh.rotation.x += 0.002;
        largeFeaMesh.rotation.y += 0.003;
    }
    largeFeaRenderer.render(largeFeaScene, largeFeaCamera);
}

// Automatically load models if clicking their tab
window.addEventListener('click', (e) => {
    if(e.target.classList.contains('tab-btn')) {
        setTimeout(() => {
            if(document.getElementById('dementia-sidebar') && document.getElementById('dementia-sidebar').classList.contains('active')) {
                updateDementiaChart();
            }
            if(document.getElementById('fea-sidebar') && document.getElementById('fea-sidebar').classList.contains('active') && !largeFeaRenderer) {
                initLargerFEA();
            }
        }, 100);
    }
});

"""

if "updateDementiaChart()" not in js:
    with open(js_file, "a") as f:
        f.write(new_js)
    print("Added new features to main.js")

