
class RobotScene {
    constructor(containerId) {
        this.container = document.getElementById(containerId);

        // 1. Setup Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0f172a); // Slate-900

        // 2. Camera: Wide shot to ensure visibility
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(50, aspect, 0.01, 10);
        this.camera.position.set(1.2, 0.8, 1.2); // Isometric-ish view
        this.camera.lookAt(0, 0, 0);

        // 3. Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);

        // 4. Lights
        const ambLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(2, 5, 2);
        dirLight.castShadow = true;
        this.scene.add(dirLight);

        // 5. Environment (Bore + Bed)
        this.buildEnvironment();

        // 6. Robot (6-DOF Serial Arm)
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
        // MRI Bore Shell
        const boreGeo = new THREE.CylinderGeometry(0.6, 0.6, 2.0, 32, 1, true);
        const boreMat = new THREE.MeshStandardMaterial({
            color: 0xcbd5e1,
            side: THREE.BackSide,
            roughness: 0.3
        });
        const bore = new THREE.Mesh(boreGeo, boreMat);
        bore.rotation.x = Math.PI / 2; // Horizontal Z-aligned
        this.scene.add(bore);

        // Transparent Coil (Copper)
        class HelixCurve extends THREE.Curve {
            getPoint(t) {
                const r = 0.58;
                const h = 1.8;
                const turns = 10;
                const angle = t * Math.PI * 2 * turns;
                return new THREE.Vector3(
                    r * Math.cos(angle),
                    r * Math.sin(angle),
                    (t - 0.5) * h
                );
            }
        }
        const coilGeo = new THREE.TubeGeometry(new HelixCurve(), 100, 0.01, 8, false);
        const coilMat = new THREE.MeshStandardMaterial({
            color: 0xb45309,
            transparent: true,
            opacity: 0.4
        });
        const coil = new THREE.Mesh(coilGeo, coilMat);
        coil.rotation.x = Math.PI / 2;
        this.scene.add(coil);

        // Patient Bed
        const bedGeo = new THREE.BoxGeometry(0.4, 0.05, 2.2);
        const bedMat = new THREE.MeshStandardMaterial({ color: 0x475569 });
        const bed = new THREE.Mesh(bedGeo, bedMat);
        bed.position.y = -0.25;
        bed.receiveShadow = true;
        this.scene.add(bed);

        const grid = new THREE.GridHelper(2, 20, 0x334155, 0x1e293b);
        grid.position.y = -0.4;
        this.scene.add(grid);

        // PATIENT (Lithotomy Position)
        const patientGroup = new THREE.Group();
        patientGroup.position.set(0, -0.22, 0); // On top of bed
        this.scene.add(patientGroup);

        const skinMat = new THREE.MeshStandardMaterial({ color: 0xf5d0b0, roughness: 0.6 });

        // Torso
        const torso = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.15, 0.6), skinMat);
        torso.position.set(0, 0.075, -0.2);
        patientGroup.add(torso);

        // Head
        const head = new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 16), skinMat);
        head.position.set(0, 0.1, -0.6);
        patientGroup.add(head);

        // Legs (Lithotomy - Up and Out)
        const legGeo = new THREE.CylinderGeometry(0.06, 0.05, 0.5);

        const leftLeg = new THREE.Mesh(legGeo, skinMat);
        leftLeg.position.set(-0.15, 0.2, 0.25);
        leftLeg.rotation.set(-Math.PI / 3, 0, 0.3); // Up 60deg, Out
        patientGroup.add(leftLeg);

        const rightLeg = new THREE.Mesh(legGeo, skinMat);
        rightLeg.position.set(0.15, 0.2, 0.25);
        rightLeg.rotation.set(-Math.PI / 3, 0, -0.3);
        patientGroup.add(rightLeg);

    }

    buildRobot() {
        // MATERIAL
        const armMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.2 });
        const jointMat = new THREE.MeshStandardMaterial({ color: 0xf1f5f9 });
        const probeMat = new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0x991b1b });

        // ROOT: Base (Fixed to world/bed)
        this.robotRoot = new THREE.Group();
        this.robotRoot.position.set(0, -0.2, 0.3); // Mounted between legs endpoint
        this.scene.add(this.robotRoot);

        // LINK 0: Base Pedestal
        const baseGeo = new THREE.CylinderGeometry(0.06, 0.08, 0.1);
        const baseMesh = new THREE.Mesh(baseGeo, jointMat);
        baseMesh.position.y = 0.05;
        this.robotRoot.add(baseMesh);

        // JOINT 1: Waist (Rotates Y)
        this.J1 = new THREE.Group();
        this.J1.position.y = 0.1;
        this.robotRoot.add(this.J1);

        // Link 1 Visual
        const l1Geo = new THREE.BoxGeometry(0.08, 0.15, 0.08);
        const l1Mesh = new THREE.Mesh(l1Geo, armMat);
        l1Mesh.position.y = 0.075;
        this.J1.add(l1Mesh);

        // JOINT 2: Shoulder (Rotates X)
        this.J2 = new THREE.Group();
        this.J2.position.set(0, 0.15, 0); // Top of L1
        this.J1.add(this.J2);

        // Link 2 Visual (Upper Arm)
        const l2Geo = new THREE.BoxGeometry(0.06, 0.25, 0.06);
        const l2Mesh = new THREE.Mesh(l2Geo, armMat);
        l2Mesh.position.y = 0.125;
        this.J2.add(l2Mesh);

        // JOINT 3: Elbow (Rotates X)
        this.J3 = new THREE.Group();
        this.J3.position.set(0, 0.25, 0);
        this.J2.add(this.J3);

        // Link 3 Visual (Forearm)
        const l3Geo = new THREE.BoxGeometry(0.05, 0.2, 0.05);
        const l3Mesh = new THREE.Mesh(l3Geo, armMat);
        l3Mesh.position.y = 0.1;
        this.J3.add(l3Mesh);

        // JOINT 4: Wrist Roll (Rotates Y - local axis)
        this.J4 = new THREE.Group();
        this.J4.position.set(0, 0.2, 0);
        this.J3.add(this.J4);

        // Link 4 Visual
        const l4Geo = new THREE.CylinderGeometry(0.03, 0.03, 0.05);
        const l4Mesh = new THREE.Mesh(l4Geo, jointMat);
        l4Mesh.rotation.z = Math.PI / 2;
        this.J4.add(l4Mesh);

        // JOINT 5: Wrist Pitch (Rotates X)
        this.J5 = new THREE.Group();
        this.J5.position.set(0, 0, 0);
        this.J4.add(this.J5);

        // End Effector Holder
        const eeGeo = new THREE.BoxGeometry(0.04, 0.04, 0.1);
        const eeMesh = new THREE.Mesh(eeGeo, armMat);
        eeMesh.position.z = -0.05; // Pointing forward
        this.J5.add(eeMesh);

        // The PROBE / NEEDLE (J6 extension)
        this.probe = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.002, 0.3), probeMat);
        this.probe.rotation.x = Math.PI / 2;
        this.probe.position.z = -0.2; // Extending out from holder
        this.J5.add(this.probe);

        // Laser Beam (Hidden line)
        const laserGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -1)]);
        const laserMat = new THREE.LineBasicMaterial({ color: 0xff0000, linewidth: 2 });
        this.laserLine = new THREE.Line(laserGeo, laserMat);
        this.laserLine.position.z = -0.35;
        this.laserLine.visible = false;
        this.J5.add(this.laserLine);
    }

    setLaser(enabled) {
        if (this.laserLine) this.laserLine.visible = enabled;
    }

    update(joints) {
        // Safe check
        if (!joints || !this.J1) return;

        // Apply Joint Angles (mock mapping for demo)
        // [0: Waist, 1: Shoulder, 2: Elbow, 3: Roll, 4: Pitch, 5: ProbeDepth]

        const t = Date.now() * 0.001;

        // If we have actual joint data from backend:
        if (joints.length >= 3) {
            // Basic inverse logic simulation for visuals
            // Assume backend sends 0..1 values or radians
            // Let's damp them

            this.J1.rotation.y = joints[0] * 2.0;
            this.J2.rotation.x = -0.5 + joints[1]; // Bias to lean forward
            this.J3.rotation.x = 1.0 + joints[2];  // Elbow bent

            // Auto-animate wrist to look 'busy' or track orientation
            this.J5.rotation.x = -1.5; // Point down/forward

            // Probe depth?
            // this.probe.position.z = ...
        } else {
            // Idle Animation
            this.J1.rotation.y = Math.sin(t) * 0.2;
        }
    }

    animate() {
        requestAnimationFrame(this.animate);
        this.renderer.render(this.scene, this.camera);
    }
}
