import re

js_file = "/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/static/js/main.js"
with open(js_file, "r") as f:
    js = f.read()

replacement = """let largeFeaScene, largeFeaCamera, largeFeaRenderer, largeFeaMesh, rfCoilMesh, emFieldParticles;
let emParticlesGeo;

function initLargerFEA() {
    const container = document.getElementById('fea-large-container');
    if (!container) return;
    
    // Clear past children
    container.innerHTML = '';
    
    largeFeaScene = new THREE.Scene();
    largeFeaCamera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
    largeFeaCamera.position.z = 20;
    
    largeFeaRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    largeFeaRenderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(largeFeaRenderer.domElement);
    
    // --- CORTICAL SURFACE MANIFOLD
    const feaNodes = document.getElementById('fea-nodes') ? parseInt(document.getElementById('fea-nodes').value) : 1000;
    const detail = Math.min(6, Math.max(1, Math.floor(feaNodes / 200)));
    
    const geo = new THREE.IcosahedronGeometry(6, detail);
    const positions = geo.attributes.position;
    for(let i = 0; i < positions.count; i++) {
        let x = positions.getX(i);
        let y = positions.getY(i);
        let z = positions.getZ(i);
        let bump = 1 + 0.15 * Math.sin(x*2) * Math.cos(y*2) + 0.05 * Math.sin(z*4);
        positions.setXYZ(i, x*bump, y*bump, z*bump);
    }
    geo.computeVertexNormals();
    
    // Heatmap style coloring based on Z position for basic FEA visual
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(positions.count * 3), 3));
    const colors = geo.attributes.color;
    for(let i = 0; i < positions.count; i++) {
        const val = (positions.getY(i) + 6) / 12; // roughly 0 to 1
        colors.setXYZ(i, 1.0, val, 0.2); // yellow-red FEA heat gradient
    }

    const mat = new THREE.MeshPhongMaterial({
        vertexColors: true,
        emissive: 0x221100,
        wireframe: true,
        transparent: true,
        opacity: 0.8,
        side: THREE.DoubleSide
    });
    
    largeFeaMesh = new THREE.Mesh(geo, mat);
    largeFeaScene.add(largeFeaMesh);

    // --- RF COIL CIRCUITRY
    const coilGeo = new THREE.TorusKnotGeometry( 8.5, 0.3, 150, 16, 2, 5 );
    const coilMat = new THREE.MeshStandardMaterial({ 
        color: 0xaaaaaa, 
        metalness: 0.9, 
        roughness: 0.1,
        emissive: 0x001155
    });
    rfCoilMesh = new THREE.Mesh(coilGeo, coilMat);
    largeFeaScene.add(rfCoilMesh);

    // --- ELECTROMAGNETIC FIELD PARTICLES
    emParticlesGeo = new THREE.BufferGeometry();
    const pCount = 1000;
    const pArray = new Float32Array(pCount * 3);
    for(let i=0; i<pCount*3; i++) {
        pArray[i] = (Math.random() - 0.5) * 35; // Random spread
    }
    emParticlesGeo.setAttribute('position', new THREE.BufferAttribute(pArray, 3));
    const particleMat = new THREE.PointsMaterial({
        color: 0x00ffff,
        size: 0.25,
        transparent: true,
        opacity: 0.6,
        blending: THREE.AdditiveBlending
    });
    emFieldParticles = new THREE.Points(emParticlesGeo, particleMat);
    largeFeaScene.add(emFieldParticles);
    
    // LIGHTING
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(10, 20, 10);
    largeFeaScene.add(light);
    const ambientLight = new THREE.AmbientLight(0x404040); 
    largeFeaScene.add(ambientLight);
    
    animateLargeFEA();
}

function animateLargeFEA() {
    if(!largeFeaRenderer) return;
    requestAnimationFrame(animateLargeFEA);
    
    const time = Date.now() * 0.001;

    if(largeFeaMesh) {
        largeFeaMesh.rotation.y += 0.002;
    }
    if(rfCoilMesh) {
        rfCoilMesh.rotation.y -= 0.005;
        rfCoilMesh.rotation.x = Math.sin(time*0.5) * 0.2;
        // Pulse the emissive color of the coil to simulate RF activation
        rfCoilMesh.material.emissiveIntensity = 0.5 + 0.5 * Math.sin(time * 8);
    }
    
    // Animate EM Field particles to simulate magnetic flux toroidal vortex
    if(emFieldParticles) {
        const positions = emParticlesGeo.attributes.position.array;
        for(let i=0; i<positions.length; i+=3) {
            let x = positions[i];
            let y = positions[i+1];
            let z = positions[i+2];

            const r = Math.sqrt(x*x + z*z) + 0.0001;
            const theta = Math.atan2(z, x) + 0.015; 
            positions[i] = r * Math.cos(theta);
            positions[i+2] = r * Math.sin(theta);
            positions[i+1] += (15 / r) * 0.1 * Math.sin(time * 3 + r); 

            if(positions[i+1] > 17) positions[i+1] = -17; 
            if(positions[i+1] < -17) positions[i+1] = 17;
        }
        emParticlesGeo.attributes.position.needsUpdate = true;
    }

    largeFeaRenderer.render(largeFeaScene, largeFeaCamera);
}"""

# regex replace old initLargerFEA till before window.addEventListener
pattern = re.compile(r'let largeFeaScene, largeFeaCamera, largeFeaRenderer, largeFeaMesh;.*?function animateLargeFEA\(\) \{.*?largeFeaRenderer\.render\(largeFeaScene, largeFeaCamera\);\n\}', re.DOTALL)
js = pattern.sub(replacement, js)

with open(js_file, "w") as f:
    f.write(js)

