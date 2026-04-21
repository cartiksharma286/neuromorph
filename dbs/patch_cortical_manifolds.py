import re

# 1. Update JS
js_path = 'static/js/main.js'
with open(js_path, 'r') as f:
    js_text = f.read()

replacement_js = """function drawCorticalFEA() {
    const canvas = document.getElementById('fea-cortical-canvas');
    if (!canvas) return;

    if (!window.feaInitialized) {
        window.feaScene = new THREE.Scene();
        window.feaCamera = new THREE.PerspectiveCamera(60, canvas.clientWidth/canvas.clientHeight, 0.1, 100);
        window.feaCamera.position.z = 20;
        
        window.feaRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
        window.feaRenderer.setSize(canvas.clientWidth, canvas.clientHeight);

        const feaNodes = document.getElementById('fea-nodes') ? parseInt(document.getElementById('fea-nodes').value) : 10000;
        // High density detail for volumetric surface
        const detail = Math.min(15, Math.max(5, Math.floor(feaNodes / 200)));
        
        const geo = new THREE.IcosahedronGeometry(7.0, detail); 
        const positions = geo.attributes.position;
        const scalars = new Float32Array(positions.count);
        geo.setAttribute('scalar', new THREE.BufferAttribute(scalars, 1));
        
        for(let i = 0; i < positions.count; i++) {
            let x = positions.getX(i);
            let y = positions.getY(i);
            let z = positions.getZ(i);
            
            // Volumetric Human Cortex Procedural Approximation
            let fissure = 1.0;
            if (Math.abs(x) < 1.5) {
                fissure = 0.4 + 0.6 * (Math.abs(x) / 1.5);
            }
            
            let noise = Math.sin(x*2.0)*Math.cos(y*3.0)*Math.sin(z*2.5) + 
                        0.5*Math.sin(x*5.0+y)*Math.cos(z*4.0) +
                        0.25*Math.cos(x*10.0)*Math.sin(y*10.0+z);
            
            let gyri = Math.pow(Math.abs(noise), 0.6); 
            let baseRadius = 1.0 - 0.12 * gyri; 
            
            let nx = x * baseRadius * fissure * 0.85; 
            let ny = y * baseRadius * 0.8; 
            let nz = z * baseRadius * 1.15; 
            
            if (nz > 0) nx *= (1.0 - 0.1*(nz/7.0));
            if (ny < 0 && Math.abs(x) > 2.0 && nz < 2.0 && nz > -2.0) {
                nx *= 1.1;
                ny *= 1.05;
            }

            positions.setXYZ(i, nx, ny, nz);
        }
        geo.computeVertexNormals();

        // Custom BEM Boundary Element Contours Shader
        const vertexShader = `
            varying vec3 vNormal;
            varying vec3 vPosition;
            attribute float scalar;
            varying float vScalar;
            void main() {
                vNormal = normalize(normalMatrix * normal);
                vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
                vScalar = scalar;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;

        const fragmentShader = `
            varying vec3 vNormal;
            varying vec3 vPosition;
            varying float vScalar;
            
            vec3 colormap(float t) {
                float r = clamp(1.5 - abs(2.0 * t - 1.0), 0.0, 1.0);
                float g = clamp(1.5 - abs(2.0 * t - 1.5), 0.0, 1.0);
                float b = clamp(1.5 - abs(2.0 * t - 2.0), 0.0, 1.0);
                vec3 brainColor = vec3(0.6, 0.5, 0.5);
                return mix(brainColor, vec3(r, g, b), clamp(t * 3.0, 0.0, 1.0));
            }

            void main() {
                float t = clamp(vScalar / 1.5, 0.0, 1.0);
                vec3 color = colormap(t);
                
                // Isolines / Contours 
                float numContours = 15.0; 
                float contour = fract(t * numContours);
                float lineThick = 0.08; 
                
                if (t > 0.02 && (contour < lineThick || contour > 1.0 - lineThick)) {
                    color = vec3(1.0); // White Contour Band
                }
                
                vec3 lightDir = normalize(vec3(0.5, 1.0, 1.0));
                float diff = max(dot(vNormal, lightDir), 0.2);
                
                vec3 viewDir = normalize(-vPosition);
                float rim = 1.0 - max(dot(viewDir, vNormal), 0.0);
                rim = smoothstep(0.6, 1.0, rim);
                
                vec3 finalColor = color * diff + vec3(0.3) * rim;
                gl_FragColor = vec4(finalColor, 0.95);
            }
        `;

        const mat = new THREE.ShaderMaterial({
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            transparent: true,
            side: THREE.DoubleSide
        });
        
        const brainMesh = new THREE.Mesh(geo, mat);
        window.feaScene.add(brainMesh);
        window.brainMesh = brainMesh; 

        const rfEmitterGeo = new THREE.SphereGeometry(0.4, 16, 16);
        const rfEmitterMat = new THREE.MeshBasicMaterial({ color: 0xff00ff });
        const rfEmitter = new THREE.Mesh(rfEmitterGeo, rfEmitterMat);
        rfEmitter.position.set(2.5, 3.5, 2.5); 
        window.feaScene.add(rfEmitter);
        window.rfEmitter = rfEmitter;
        
        window.bemTime = 0;
        window.feaInitialized = true;
    }

    if (window.brainMesh && window.rfEmitter) {
        window.bemTime += 0.05;
        const sourcePos = window.rfEmitter.position;
        const posArray = window.brainMesh.geometry.attributes.position.array;
        const scalarArray = window.brainMesh.geometry.attributes.scalar.array;
        
        for(let i = 0, j = 0; i < posArray.length; i+=3, j++) {
            let dx = posArray[i] - sourcePos.x;
            let dy = posArray[i+1] - sourcePos.y;
            let dz = posArray[i+2] - sourcePos.z;
            let r = Math.sqrt(dx*dx + dy*dy + dz*dz);
            
            let rf_pulse = Math.max(0, Math.sin(window.bemTime * 3.0 - r * 1.5)); 
            let targetE = (1.0 / (r * r + 0.1)) * rf_pulse * 12.0;
            
            scalarArray[j] = scalarArray[j] * 0.85 + targetE * 0.15;
        }
        window.brainMesh.geometry.attributes.scalar.needsUpdate = true;
        
        window.brainMesh.rotation.y += 0.003;
        window.brainMesh.rotation.z = Math.sin(window.bemTime * 0.1) * 0.05;
    }
    
    window.feaRenderer.render(window.feaScene, window.feaCamera);
    requestAnimationFrame(drawCorticalFEA);
}"""

idx1 = js_text.find("function drawCorticalFEA()")
if idx1 != -1:
    idx2 = js_text.find("function", idx1 + 10)
    if idx2 == -1: idx2 = len(js_text)
    new_js = js_text[:idx1] + replacement_js + "\n\n" + js_text[idx2:]
    with open(js_path, "w") as f:
        f.write(new_js)
    print("JS volumetric manifold successful.")
else:
    print("Could not find JS function.")

# 2. Update HTML
html_path = 'templates/index.html'
with open(html_path, 'r') as f:
    html_text = f.read()

target1 = """<div style="font-size: 10px; color: var(--accent-pink); margin-bottom: 5px;">CORTICAL CURRENT DENSITY (BEM)</div>"""
target2_start = """<div style="position: absolute; bottom: 10px; left: 10px;"""

new_legend_html = """
                    <div style="position: absolute; bottom: 10px; left: 10px; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 4px; font-size: 10px; color: white; width: 220px; z-index: 10; border: 1px solid rgba(255,255,255,0.2); backdrop-filter: blur(4px);">
                        <div style="font-weight: bold; margin-bottom: 6px; color: #00f2ff; font-size: 10px;">BOUNDARY ELEMENT ISOLINES LUT</div>
                        <div style="display: flex; align-items: center; justify-content: space-between; font-size: 9px; margin-bottom: 3px;">
                            <span>0.0</span>
                            <span style="font-weight:bold;">E-Field (V/mm)</span>
                            <span>1.5+</span>
                        </div>
                        <div style="height: 12px; width: 100%; background: linear-gradient(to right, rgb(153,128,128), rgb(0,0,128), rgb(0,128,255), rgb(0,255,128), rgb(255,255,0), rgb(255,128,0), rgb(255,0,0)); border: 1px solid #444; border-radius: 2px; margin-bottom: 6px;"></div>
                        <div style="display: flex; align-items: center; gap: 6px; font-size: 9px;">
                            <div style="width:14px; height:2px; background: white;"></div> 
                            <span>Boundary Contours (Δ=0.1 step)</span>
                        </div>
                    </div>
"""

# Very simple replacement logic for the old legend
if target1 in html_text:
    idx_start = html_text.find(target1)
    idx_end = html_text.find("</div>", html_text.find(target2_start))
    
    # We just know the rough block. We will just use regex to replace `<div style="font-size: 10px; color: var(--accent-pink); margin-bottom: 5px;">...</div>...` completely
    pattern = re.compile(r'<div style="font-size: 10px; color: var\(--accent-pink\); margin-bottom: 5px;">.*?</div>\s*<div style="position: absolute; bottom: 10px; left: 10px; background: rgba\(0,0,0,0\.[67]\).*?</div>\s*</div>', re.DOTALL)
    
    new_html = pattern.sub(target1 + new_legend_html, html_text)
    with open(html_path, "w") as f:
        f.write(new_html)
    print("HTML LUT legend patched.")
else:
    print("HTML legend target not found.")
