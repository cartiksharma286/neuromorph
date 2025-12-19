
class RobotScene {
    constructor(containerId) {
        this.container = document.getElementById(containerId);

        // 1. Setup Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x020617); // Deepest Slate
        this.scene.fog = new THREE.Fog(0x020617, 10, 50);

        // 2. Camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 100);
        this.camera.position.set(1.5, 1.2, 1.5);
        this.camera.lookAt(0, 0, 0);

        // 3. Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.container.appendChild(this.renderer.domElement);

        // 4. Lights
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x0f172a, 0.4);
        this.scene.add(hemiLight);

        const mainLight = new THREE.DirectionalLight(0xffffff, 1.0);
        mainLight.position.set(5, 10, 5);
        mainLight.castShadow = true;
        mainLight.shadow.mapSize.width = 1024;
        mainLight.shadow.mapSize.height = 1024;
        this.scene.add(mainLight);

        const blueLight = new THREE.PointLight(0x3b82f6, 0.8, 10);
        blueLight.position.set(-2, 2, -2);
        this.scene.add(blueLight);

        // 5. Environment
        this.buildEnvironment();

        // 6. Robot
        this.buildRobot();

        // 7. Loop
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);

        // Resize
        window.addEventListener('resize', () => {
            if (!this.camera || !this.renderer) return;
            this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        });
    }

    buildEnvironment() {
        // Floor Grid (Glowing)
        const grid = new THREE.GridHelper(10, 50, 0x06b6d4, 0x1e293b);
        grid.position.y = -0.5;
        this.scene.add(grid);

        // MRI Bore (Glassy/Metallic)
        const boreGeo = new THREE.CylinderGeometry(0.7, 0.7, 2.5, 64, 1, true);
        const boreMat = new THREE.MeshPhysicalMaterial({
            color: 0xe2e8f0,
            metalness: 0.8,
            roughness: 0.2,
            side: THREE.BackSide,
            clearcoat: 1.0,
            clearcoatRoughness: 0.1
        });
        const bore = new THREE.Mesh(boreGeo, boreMat);
        bore.rotation.x = Math.PI / 2;
        bore.receiveShadow = true;
        this.scene.add(bore);

        // Patient Bed
        const bedGeo = new THREE.BoxGeometry(0.5, 0.05, 3.0);
        const bedMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.5 });
        const bed = new THREE.Mesh(bedGeo, bedMat);
        bed.position.y = -0.3;
        bed.receiveShadow = true;
        bed.castShadow = true;
        this.scene.add(bed);

        // PATIENT (Stylized)
        const patientGroup = new THREE.Group();
        patientGroup.position.set(0, -0.27, 0);
        this.scene.add(patientGroup);

        const skinMat = new THREE.MeshStandardMaterial({ color: 0xfbd38d, roughness: 0.6 });
        const torso = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.15, 0.7), skinMat);
        torso.position.set(0, 0.075, -0.2);
        torso.castShadow = true;
        patientGroup.add(torso);

        const head = new THREE.Mesh(new THREE.SphereGeometry(0.12, 32, 32), skinMat);
        head.position.set(0, 0.1, -0.65);
        head.castShadow = true;
        patientGroup.add(head);

        // Legs
        const legGeo = new THREE.CylinderGeometry(0.07, 0.06, 0.6);
        const leftLeg = new THREE.Mesh(legGeo, skinMat);
        leftLeg.position.set(-0.15, 0.25, 0.3);
        leftLeg.rotation.set(-Math.PI / 3, 0, 0.2);
        leftLeg.castShadow = true;
        patientGroup.add(leftLeg);

        const rightLeg = new THREE.Mesh(legGeo, skinMat);
        rightLeg.position.set(0.15, 0.25, 0.3);
        rightLeg.rotation.set(-Math.PI / 3, 0, -0.2);
        rightLeg.castShadow = true;
        patientGroup.add(rightLeg);
    }

    buildRobot() {
        // High-Tech Materials
        const armMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3, metalness: 0.5 });
        const jointMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.7, metalness: 0.2 });
        const probeMat = new THREE.MeshStandardMaterial({ color: 0xdc2626, emissive: 0x991b1b, emissiveIntensity: 0.5 });

        this.robotRoot = new THREE.Group();
        this.robotRoot.position.set(0, -0.2, 0.4);
        this.scene.add(this.robotRoot);

        // Base
        const base = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.1, 0.1), jointMat);
        base.position.y = 0.05;
        base.castShadow = true;
        this.robotRoot.add(base);

        // J1
        this.J1 = new THREE.Group();
        this.J1.position.y = 0.1;
        this.robotRoot.add(this.J1);
        const l1 = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.15, 0.1), armMat);
        l1.position.y = 0.075;
        l1.castShadow = true;
        this.J1.add(l1);

        // J2
        this.J2 = new THREE.Group();
        this.J2.position.set(0, 0.15, 0);
        this.J1.add(this.J2);
        const l2 = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.3, 0.08), armMat);
        l2.position.y = 0.15;
        l2.castShadow = true;
        this.J2.add(l2);

        // J3
        this.J3 = new THREE.Group();
        this.J3.position.set(0, 0.3, 0);
        this.J2.add(this.J3);
        const l3 = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.25, 0.06), armMat);
        l3.position.y = 0.125;
        l3.castShadow = true;
        this.J3.add(l3);

        // J4
        this.J4 = new THREE.Group();
        this.J4.position.set(0, 0.25, 0);
        this.J3.add(this.J4);
        const l4 = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.08), jointMat);
        l4.rotation.z = Math.PI / 2;
        l4.castShadow = true;
        this.J4.add(l4);

        // J5
        this.J5 = new THREE.Group();
        this.J4.add(this.J5);
        const ee = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 0.12), armMat);
        ee.position.z = -0.06;
        ee.castShadow = true;
        this.J5.add(ee);

        // Probe
        this.probe = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.001, 0.35), probeMat);
        this.probe.rotation.x = Math.PI / 2;
        this.probe.position.z = -0.25;
        this.probe.castShadow = true;
        this.J5.add(this.probe);

        // Glowing Laser Beam
        const laserGeo = new THREE.CylinderGeometry(0.005, 0.005, 1.0, 8, 1, true);
        laserGeo.translate(0, -0.5, 0);
        laserGeo.rotateX(-Math.PI / 2); // Point -Z
        const laserMat = new THREE.MeshBasicMaterial({
            color: 0xff4500,
            transparent: true,
            opacity: 0.6,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            side: THREE.DoubleSide
        });
        this.laserBeam = new THREE.Mesh(laserGeo, laserMat);
        this.laserBeam.position.z = -0.42;
        this.laserBeam.visible = false;

        // Core of beam (brighter)
        const coreGeo = new THREE.CylinderGeometry(0.001, 0.001, 1.0, 8, 1, true);
        coreGeo.translate(0, -0.5, 0);
        coreGeo.rotateX(-Math.PI / 2);
        const coreMat = new THREE.MeshBasicMaterial({ color: 0xffffff, blending: THREE.AdditiveBlending });
        const core = new THREE.Mesh(coreGeo, coreMat);
        this.laserBeam.add(core);

        this.J5.add(this.laserBeam);

        // Cryo Mist (Ice Cloud)
        const mistGeo = new THREE.SphereGeometry(0.1, 16, 16);
        const mistMat = new THREE.MeshBasicMaterial({
            color: 0xcffafe,
            transparent: true,
            opacity: 0.4,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });
        this.cryoMist = new THREE.Mesh(mistGeo, mistMat);
        this.cryoMist.position.z = -0.3;
        this.cryoMist.scale.set(1, 2, 1);
        this.cryoMist.visible = false;
        this.J5.add(this.cryoMist);
    }

    setLaser(enabled) {
        if (this.laserBeam) {
            this.laserBeam.visible = enabled;
            if (enabled) {
                if (this.probe) {
                    this.probe.material.color.setHex(0xdc2626);
                    this.probe.material.emissive.setHex(0x991b1b);
                }
                if (this.cryoMist) this.cryoMist.visible = false;
            }
        }
    }

    setCryo(enabled) {
        if (this.cryoMist) {
            this.cryoMist.visible = enabled;
            if (enabled) {
                if (this.probe) {
                    this.probe.material.color.setHex(0x06b6d4); // Cyan
                    this.probe.material.emissive.setHex(0x0891b2);
                }
                if (this.laserBeam) this.laserBeam.visible = false;
            } else if (!this.laserBeam?.visible) {
                // Idle
                if (this.probe) {
                    this.probe.material.color.setHex(0xef4444);
                    this.probe.material.emissive.setHex(0x500000);
                }
            }
        }
    }

    update(joints) {
        if (!joints || !this.J1) return;

        // Smooth interpolation could go here, but direct mapping for responsiveness
        const t = Date.now() * 0.002;

        if (joints.length >= 3) {
            this.J1.rotation.y = joints[0] * 2.0;
            this.J2.rotation.x = -0.5 + joints[1];
            this.J3.rotation.x = 1.0 + joints[2];
            this.J5.rotation.x = -1.5; // Always point generally towards patient
        } else {
            // Idle breathing animation
            this.J1.rotation.y = Math.sin(t) * 0.1;
            this.J3.rotation.x = 1.0 + Math.cos(t) * 0.05;
        }

        // Pulse laser if on
        if (this.laserBeam && this.laserBeam.visible) {
            const s = 1.0 + Math.sin(t * 10) * 0.2;
            this.laserBeam.scale.set(s, 1, s);
        }

        // Pulse Cryo if on
        if (this.cryoMist && this.cryoMist.visible) {
            const s = 1.2 + Math.sin(t * 5) * 0.1;
            this.cryoMist.rotation.y += 0.05;
            this.cryoMist.scale.set(s, s * 1.5, s);
        }
    }

    animate() {
        requestAnimationFrame(this.animate);
        this.renderer.render(this.scene, this.camera);
    }
}
