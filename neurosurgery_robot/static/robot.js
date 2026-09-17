/**
 * Miniaturized Ultrasound Probe 3D Simulation & Acoustic Wave Propagation
 * Based on biosono.com micro-transducer array schematics and acoustic field propagation.
 * 
 * Features:
 * - Ø 3.5 mm Titanium/Ceramic Enclosure with Burr-Hole Positioning Mount
 * - 32x32 (1024-element) CMUT/PMUT Micro-Transducer Face with λ/4 Matching Layer
 * - Miniaturized ASIC SoC Interconnect with Polyimide Micro-Ribbon
 * - Dynamic Acoustic Radiation Sector Cone & Focus Reticle (8.5 MHz Imaging / 2.25 MHz HIFU)
 * - Micro-manipulator 3-Axis Actuation into Intracranial Target Volume
 */

class MiniaturizedProbeViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) return;

        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0f1d);
        this.scene.fog = new THREE.FogExp2(0x0a0f1d, 0.18);

        // Camera
        this.camera = new THREE.PerspectiveCamera(65, this.container.clientWidth / this.container.clientHeight, 0.05, 100);
        this.camera.position.set(1.4, 1.2, 1.6);
        this.camera.lookAt(0.35, 0.35, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio || 1);
        this.container.innerHTML = '';
        this.container.appendChild(this.renderer.domElement);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x1e293b, 2.5);
        this.scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0x38bdf8, 1.8);
        dirLight.position.set(4, 6, 3);
        this.scene.add(dirLight);

        const accentLight = new THREE.PointLight(0x10b981, 2.0, 8);
        accentLight.position.set(0.7, 1.2, 0.5);
        this.scene.add(accentLight);

        // Surgical Floor / Grid
        const gridHelper = new THREE.GridHelper(6, 24, 0x0284c7, 0x1e293b);
        gridHelper.position.y = -0.1;
        this.scene.add(gridHelper);

        // Build Environment & Cranial Target
        this.buildPatientEnvironment();

        // Build biosono.com Miniaturized Ultrasound Probe
        this.buildBiosonoProbe();

        // Path & Pulse Animation State
        this.simulating = false;
        this.pathTime = 0;
        this.hifuActive = false;
        this.steeringAngleDeg = 0;

        // Mouse Drag / Orbit interaction
        this.initMouseControls();

        // Animation Loop
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);

        // Resize handler
        window.addEventListener('resize', () => this.onWindowResize(), false);
    }

    buildPatientEnvironment() {
        // Surgical Table & Cranial Mount
        const tableMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.7, roughness: 0.3 });
        const tableTop = new THREE.Mesh(new THREE.BoxGeometry(2.4, 0.08, 0.75), tableMat);
        tableTop.position.set(0.3, -0.04, 0);
        this.scene.add(tableTop);

        // Patient Head (Semi-transparent Cranial Contour)
        const headGroup = new THREE.Group();
        headGroup.position.set(0.65, 0.28, 0);
        this.scene.add(headGroup);
        this.headWorldPos = headGroup.position;

        const skullGeo = new THREE.SphereGeometry(0.19, 32, 32);
        const skullMat = new THREE.MeshPhysicalMaterial({
            color: 0x64748b,
            transmission: 0.75,
            opacity: 0.45,
            transparent: true,
            roughness: 0.2,
            metalness: 0.1,
            clearcoat: 1.0
        });
        const skull = new THREE.Mesh(skullGeo, skullMat);
        headGroup.add(skull);

        // Burr-Hole Keyhole Cranial Ring (3.8 mm entry port)
        const ringGeo = new THREE.TorusGeometry(0.024, 0.005, 16, 32);
        const ringMat = new THREE.MeshStandardMaterial({ color: 0xf59e0b, metalness: 0.9, roughness: 0.1 });
        const burrHoleRing = new THREE.Mesh(ringGeo, ringMat);
        burrHoleRing.rotation.x = Math.PI / 2;
        burrHoleRing.position.set(0, 0.19, 0);
        headGroup.add(burrHoleRing);

        // Stereotactic Frame Arc (Burr-hole guidance arch)
        const arcCurve = new THREE.EllipseCurve(0, 0, 0.23, 0.23, 0, Math.PI, false, 0);
        const arcPoints = arcCurve.getPoints(32);
        const arcGeo = new THREE.BufferGeometry().setFromPoints(arcPoints.map(p => new THREE.Vector3(p.x, p.y, 0)));
        const arcLine = new THREE.Line(arcGeo, new THREE.LineBasicMaterial({ color: 0x0284c7, linewidth: 2 }));
        arcLine.rotation.y = Math.PI / 2;
        arcLine.position.set(0, 0.05, 0);
        headGroup.add(arcLine);

        // Intracranial Tumor Target (Target lesion)
        const tumorGeo = new THREE.IcosahedronGeometry(0.042, 3);
        this.tumorMat = new THREE.MeshStandardMaterial({
            color: 0xef4444,
            roughness: 0.6,
            emissive: 0x7f1d1d,
            emissiveIntensity: 0.4,
            transparent: true,
            opacity: 0.85
        });
        this.tumorMesh = new THREE.Mesh(tumorGeo, this.tumorMat);
        this.tumorMesh.position.set(0.0, -0.04, 0.0);
        headGroup.add(this.tumorMesh);

        // Surrounding Healthy Cortex Vascular Network (Curved Micro-vessels)
        const vesselCurve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(-0.06, -0.08, 0.02),
            new THREE.Vector3(-0.02, -0.02, 0.06),
            new THREE.Vector3(0.04, -0.06, -0.03),
            new THREE.Vector3(0.08, 0.02, 0.04)
        ]);
        const vesselGeo = new THREE.TubeGeometry(vesselCurve, 32, 0.006, 8, false);
        const vesselMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, roughness: 0.3, metalness: 0.2 });
        headGroup.add(new THREE.Mesh(vesselGeo, vesselMat));
    }

    buildBiosonoProbe() {
        // Probe Root Group
        this.probeRoot = new THREE.Group();
        this.probeRoot.position.set(0.65, 0.70, 0.0);
        this.scene.add(this.probeRoot);

        // 1. Probe Handle / Micro-Actuator Housing (Burr-hole Manipulator)
        const handleMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.85, roughness: 0.25 });
        const handleGeo = new THREE.CylinderGeometry(0.028, 0.032, 0.22, 32);
        const handle = new THREE.Mesh(handleGeo, handleMat);
        handle.position.y = 0.20;
        this.probeRoot.add(handle);

        // Depth Gauge Vernier Rings on Handle
        const ringMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, metalness: 0.9, roughness: 0.1 });
        for (let i = 0; i < 4; i++) {
            const r = new THREE.Mesh(new THREE.TorusGeometry(0.030, 0.002, 8, 32), ringMat);
            r.rotation.x = Math.PI / 2;
            r.position.y = 0.12 + i * 0.04;
            this.probeRoot.add(r);
        }

        // 2. Miniaturized Needle Shaft (Titanium Ø 3.5 mm biosono spec)
        const shaftMat = new THREE.MeshStandardMaterial({
            color: 0xe2e8f0,
            metalness: 0.95,
            roughness: 0.15
        });
        const shaftGeo = new THREE.CylinderGeometry(0.007, 0.007, 0.26, 32);
        const shaft = new THREE.Mesh(shaftGeo, shaftMat);
        shaft.position.y = -0.04;
        this.probeRoot.add(shaft);

        // 3. CMUT / PMUT 2D Transducer Tip (32x32 Matrix Array Face)
        const tipMat = new THREE.MeshStandardMaterial({
            color: 0x10b981, // Gold/emerald piezo array surface
            emissive: 0x064e3b,
            emissiveIntensity: 0.6,
            metalness: 0.9,
            roughness: 0.1
        });
        const tipGeo = new THREE.CylinderGeometry(0.008, 0.007, 0.015, 32);
        this.transducerTip = new THREE.Mesh(tipGeo, tipMat);
        this.transducerTip.position.y = -0.17;
        this.probeRoot.add(this.transducerTip);

        // 4. Acoustic Emission Sector Cone (70° B-Mode / HIFU Field)
        const coneGeo = new THREE.ConeGeometry(0.16, 0.26, 32, 1, true);
        coneGeo.translate(0, -0.13, 0); // apex at probe tip
        this.acousticConeMat = new THREE.MeshBasicMaterial({
            color: 0x38bdf8,
            transparent: true,
            opacity: 0.28,
            side: THREE.DoubleSide,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });
        this.acousticCone = new THREE.Mesh(coneGeo, this.acousticConeMat);
        this.acousticCone.position.copy(this.transducerTip.position);
        this.probeRoot.add(this.acousticCone);

        // 5. HIFU Focal Ablation Spot (High Intensity Center Reticle)
        const focalSpotGeo = new THREE.SphereGeometry(0.014, 16, 16);
        this.focalSpotMat = new THREE.MeshBasicMaterial({
            color: 0xf59e0b,
            transparent: true,
            opacity: 0.0,
            blending: THREE.AdditiveBlending
        });
        this.focalSpot = new THREE.Mesh(focalSpotGeo, this.focalSpotMat);
        this.focalSpot.position.set(0, -0.17 - 0.14, 0); // 28 mm depth from tip
        this.probeRoot.add(this.focalSpot);

        // 6. Pulse Wavefront Wave Ring Particles
        this.pulseRings = [];
        for (let i = 0; i < 4; i++) {
            const ring = new THREE.Mesh(
                new THREE.RingGeometry(0.01 + i * 0.03, 0.014 + i * 0.03, 32),
                new THREE.MeshBasicMaterial({
                    color: 0x38bdf8,
                    transparent: true,
                    opacity: 0.4,
                    side: THREE.DoubleSide,
                    blending: THREE.AdditiveBlending
                })
            );
            ring.rotation.x = Math.PI / 2;
            ring.position.set(0, -0.17 - (i + 1) * 0.04, 0);
            this.probeRoot.add(ring);
            this.pulseRings.push({ mesh: ring, offset: i * 0.25 });
        }
    }

    initMouseControls() {
        let isDragging = false;
        let prevMousePos = { x: 0, y: 0 };

        this.container.addEventListener('mousedown', (e) => {
            isDragging = true;
            prevMousePos = { x: e.clientX, y: e.clientY };
        });

        window.addEventListener('mouseup', () => { isDragging = false; });

        this.container.addEventListener('mousemove', (e) => {
            if (isDragging) {
                const deltaX = e.clientX - prevMousePos.x;
                const deltaY = e.clientY - prevMousePos.y;
                this.camera.position.x += deltaX * 0.005;
                this.camera.position.y -= deltaY * 0.005;
                this.camera.lookAt(0.65, 0.35, 0);
                prevMousePos = { x: e.clientX, y: e.clientY };
            }
        });
    }

    setHIFUPulse(enabled, powerWatts = 60.0) {
        this.hifuActive = enabled;
        if (enabled) {
            this.acousticConeMat.color.setHex(0xef4444);
            this.acousticConeMat.opacity = 0.55;
            this.focalSpotMat.opacity = 0.95;
            this.focalSpotMat.color.setHex(0xffffff);
            if (this.tumorMesh) {
                this.tumorMat.emissive.setHex(0xef4444);
                this.tumorMat.emissiveIntensity = 0.95;
            }
        } else {
            this.acousticConeMat.color.setHex(0x38bdf8);
            this.acousticConeMat.opacity = 0.28;
            this.focalSpotMat.opacity = 0.0;
            if (this.tumorMesh) {
                this.tumorMat.emissive.setHex(0x7f1d1d);
                this.tumorMat.emissiveIntensity = 0.4;
            }
        }
    }

    updateProbePosition(nx, nz, depthNorm = 0.5) {
        if (!this.probeRoot) return;
        // Normalised coords to micro-manipulator world displacement
        const targetX = 0.65 + (nx - 0.5) * 0.12;
        const targetZ = (nz - 0.5) * 0.12;
        const targetY = 0.62 - depthNorm * 0.12;

        this.probeRoot.position.x += (targetX - this.probeRoot.position.x) * 0.15;
        this.probeRoot.position.z += (targetZ - this.probeRoot.position.z) * 0.15;
        this.probeRoot.position.y += (targetY - this.probeRoot.position.y) * 0.15;
    }

    setBeamSteering(angleDeg) {
        this.steeringAngleDeg = angleDeg;
        if (this.acousticCone) {
            this.acousticCone.rotation.z = THREE.MathUtils.degToRad(angleDeg);
        }
    }

    startSimulation() {
        this.simulating = true;
        this.pathTime = 0;
    }

    onWindowResize() {
        if (!this.container || !this.renderer || !this.camera) return;
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    animate() {
        requestAnimationFrame(this.animate);
        const t = Date.now() * 0.001;

        // Acoustic Wave Propagation Pulse Animation
        if (this.pulseRings) {
            for (const item of this.pulseRings) {
                const phase = (t * 2.5 + item.offset) % 1.0;
                const scale = 0.3 + phase * 1.8;
                item.mesh.scale.set(scale, scale, scale);
                item.mesh.material.opacity = (1.0 - phase) * (this.hifuActive ? 0.75 : 0.35);
                if (this.hifuActive) {
                    item.mesh.material.color.setHex(0xf97316);
                } else {
                    item.mesh.material.color.setHex(0x38bdf8);
                }
            }
        }

        // Simulating Micro-Actuation Scan
        if (this.simulating) {
            this.pathTime += 0.02;
            const nx = 0.5 + 0.3 * Math.sin(this.pathTime);
            const nz = 0.5 + 0.3 * Math.cos(this.pathTime);
            this.updateProbePosition(nx, nz, 0.6 + 0.2 * Math.sin(this.pathTime * 2));
            if (this.pathTime > Math.PI * 4) {
                this.simulating = false;
            }
        }

        // Focal Spot Pulse
        if (this.hifuActive && this.focalSpot) {
            const s = 1.0 + 0.3 * Math.sin(t * 18.0);
            this.focalSpot.scale.set(s, s, s);
        }

        this.renderer.render(this.scene, this.camera);
    }
}

// Backward compatibility alias
class RobotViz extends MiniaturizedProbeViz {
    updateJoints(angles) {
        // Map kinematics angles to probe micro-actuation
        if (angles && angles.length >= 3) {
            const nx = (angles[0] + 45) / 90;
            const nz = (angles[1] + 45) / 90;
            const depth = (angles[2] + 45) / 90;
            this.updateProbePosition(nx, nz, depth);
        }
    }
    setLaser(enabled) {
        this.setHIFUPulse(enabled);
    }
    setEndEffectorBrain(nx, nz, active) {
        this.updateProbePosition(nx, nz, 0.6);
        this.setHIFUPulse(active);
    }
}

