class RobotViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);

        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1d26);
        this.scene.fog = new THREE.FogExp2(0x1a1d26, 0.2);

        // Camera
        this.camera = new THREE.PerspectiveCamera(75, this.container.clientWidth / this.container.clientHeight, 0.1, 1000);
        this.camera.position.set(2, 2, 2);
        this.camera.lookAt(0, 0, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.container.appendChild(this.renderer.domElement);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 2);
        this.scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 1);
        dirLight.position.set(5, 5, 5);
        this.scene.add(dirLight);

        const pointLight = new THREE.PointLight(0x3b82f6, 2, 10);
        pointLight.position.set(0, 2, 0);
        this.scene.add(pointLight);

        // Grid
        const gridHelper = new THREE.GridHelper(10, 20, 0x3b82f6, 0x444444);
        this.scene.add(gridHelper);

        // Build Robot
        this.joints = [];
        this.buildRobot();

        // Initialize Environment
        this.buildEnvironment();

        // Laser visualization (existing)
        this.laserBeam = new THREE.Mesh(
            new THREE.CylinderGeometry(0.02, 0.02, 5, 8),
            new THREE.MeshBasicMaterial({
                color: 0xff0000,
                transparent: true,
                opacity: 0.0,
                blending: THREE.AdditiveBlending
            })
        );
        this.laserBeam.rotation.x = -Math.PI / 2;
        this.laserBeam.position.z = 2.5; // Extends from end effector
        // Attach laser to last joint
        this.joints[this.joints.length - 1].add(this.laserBeam);

        // Path Animation State
        this.simulating = false;
        this.pathTime = 0;

        // Animation Loop
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);

        // Resize handler
        window.addEventListener('resize', () => this.onWindowResize(), false);
    }

    buildEnvironment() {
        // 1. MRI Bore (Cylindrical Tunnel)
        // Radius ~0.8m, Length ~2m
        const boreGeo = new THREE.CylinderGeometry(1.2, 1.2, 4, 32, 1, true);
        const boreMat = new THREE.MeshStandardMaterial({
            color: 0xeeeeee,
            side: THREE.BackSide,
            metalness: 0.3,
            roughness: 0.7
        });
        const bore = new THREE.Mesh(boreGeo, boreMat);
        bore.rotation.z = Math.PI / 2;
        bore.position.set(0, 0.5, 0);
        this.scene.add(bore);

        const housingGeo = new THREE.BoxGeometry(3, 3, 4);
        const housingMat = new THREE.MeshStandardMaterial({ color: 0xe0e0e0 });

        const coilGeo = new THREE.TorusGeometry(1.15, 0.02, 16, 64);
        const coilMat = new THREE.MeshBasicMaterial({ color: 0x3b82f6 });
        for (let z = -1.5; z <= 1.5; z += 0.5) {
            const coil = new THREE.Mesh(coilGeo, coilMat);
            coil.position.set(0, 0.5, z);
            this.scene.add(coil);
        }

        // 2. Patient Visualization
        const patientGroup = new THREE.Group();
        this.scene.add(patientGroup);

        // Body (Simple Cylinder)
        const bodyGeo = new THREE.CylinderGeometry(0.25, 0.25, 1.8, 16);
        const bodyMat = new THREE.MeshStandardMaterial({ color: 0x8ecae6 }); // Hospital gown blue
        const body = new THREE.Mesh(bodyGeo, bodyMat);
        body.rotation.z = Math.PI / 2;
        body.position.set(-0.2, 0.25, 0); // Lying on table
        patientGroup.add(body);

        // Head (Sphere) - Transparent to see "inside"
        const headGeo = new THREE.SphereGeometry(0.18, 32, 32);
        const headMat = new THREE.MeshPhysicalMaterial({
            color: 0xffdbac, // Skin tone
            transmission: 0.4,  // Glass-like transmission
            opacity: 0.5,
            transparent: true,
            roughness: 0.2,
            metalness: 0.1,
            clearcoat: 1.0
        });
        const head = new THREE.Mesh(headGeo, headMat);
        head.position.set(0.7, 0.3, 0); // At end of body
        patientGroup.add(head);

        // Brain Surface (Inner Layer)
        const brainGeo = new THREE.SphereGeometry(0.16, 32, 32);
        const brainMat = new THREE.MeshStandardMaterial({
            color: 0xf4f4f5,
            roughness: 0.5,
            wireframe: true, // Grid-like structure
            transparent: true,
            opacity: 0.1
        });
        const brain = new THREE.Mesh(brainGeo, brainMat);
        brain.position.copy(head.position);
        patientGroup.add(brain);

        // 3. Neurovasculature (Inside the Head)
        // Positioned RELATIVE to Head
        const vesselGroup = new THREE.Group();
        vesselGroup.position.copy(head.position);
        this.scene.add(vesselGroup);

        const curve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(-0.05, -0.05, 0),
            new THREE.Vector3(0.0, 0.05, 0.05),
            new THREE.Vector3(0.05, 0.0, -0.05),
            new THREE.Vector3(0.1, 0.05, 0)
        ]);

        const tubeGeo = new THREE.TubeGeometry(curve, 64, 0.015, 8, false);
        const vesselMat = new THREE.MeshStandardMaterial({ color: 0xef4444, roughness: 0.3, metalness: 0.1 });
        const vessel = new THREE.Mesh(tubeGeo, vesselMat);
        vesselGroup.add(vessel);

        // 4. Tumor Tissue (Target) - Inside Head
        const tumorGeo = new THREE.IcosahedronGeometry(0.04, 2);
        const tumorMat = new THREE.MeshStandardMaterial({
            color: 0x8b5cf6, // Violet
            roughness: 0.9,
            emissive: 0x220044
        });
        this.tumor = new THREE.Mesh(tumorGeo, tumorMat);
        // Position relative to vessel
        this.tumor.position.set(0.05, 0.05, 0);
        vesselGroup.add(this.tumor);

        // Save path
        this.simPath = curve;
    }

    startSimulation() {
        this.simulating = true;
        this.pathTime = 0;
    }

    buildRobot() {
        const material = new THREE.MeshStandardMaterial({
            color: 0xcccccc,
            roughness: 0.2,
            metalness: 0.8
        });
        const jointMat = new THREE.MeshStandardMaterial({ color: 0x333333 });

        // Create a root group for position control
        this.robotRoot = new THREE.Group();
        // Position robot to the side of the patient (closer to head at x=0.7)
        this.robotRoot.position.set(0.4, 0.0, 0.4);
        // Rotate slightly to face patient
        this.robotRoot.rotation.y = Math.PI / 4;
        this.scene.add(this.robotRoot);

        // Base
        const baseGeo = new THREE.CylinderGeometry(0.2, 0.3, 0.2, 32);
        const base = new THREE.Mesh(baseGeo, material);
        base.position.y = 0.1;
        this.robotRoot.add(base);

        // Joint 1 (Waist) - Rotates around Y
        const j1 = new THREE.Group();
        j1.position.y = 0.2; // Top of base
        base.add(j1);
        this.addJointGeo(j1, jointMat);
        this.joints.push(j1);

        // Link 1
        const l1Geo = new THREE.BoxGeometry(0.1, 0.4, 0.1);
        const l1 = new THREE.Mesh(l1Geo, material);
        l1.position.y = 0.2;
        j1.add(l1);

        // Joint 2 (Shoulder) - Rotates around Z (or X depending on config)
        // Adjusting to match simplified kinematics logic
        const j2 = new THREE.Group();
        j2.position.y = 0.2; // Top of Link 1
        l1.add(j2);
        this.addJointGeo(j2, jointMat);
        this.joints.push(j2);

        // Link 2 (Upper Arm)
        const l2Geo = new THREE.BoxGeometry(0.08, 0.4, 0.08);
        const l2 = new THREE.Mesh(l2Geo, material);
        l2.position.y = 0.2; // Extends up (will rotate)
        j2.add(l2);

        // Joint 3 (Elbow)
        const j3 = new THREE.Group();
        j3.position.y = 0.2;
        l2.add(j3);
        this.addJointGeo(j3, jointMat);
        this.joints.push(j3);

        // Link 3 (Forearm)
        const l3Geo = new THREE.BoxGeometry(0.06, 0.3, 0.06);
        const l3 = new THREE.Mesh(l3Geo, material);
        l3.position.y = 0.15;
        j3.add(l3);

        // Joint 4 (Wrist 1)
        const j4 = new THREE.Group();
        j4.position.y = 0.15;
        l3.add(j4);
        this.addJointGeo(j4, jointMat);
        this.joints.push(j4);

        // Link 4
        const l4Geo = new THREE.BoxGeometry(0.05, 0.1, 0.05);
        const l4 = new THREE.Mesh(l4Geo, material);
        l4.position.y = 0.05;
        j4.add(l4);

        // Joint 5 (Wrist 2)
        const j5 = new THREE.Group();
        j5.position.y = 0.05;
        l4.add(j5);
        this.addJointGeo(j5, jointMat);
        this.joints.push(j5); // Fixed typo 'puhs'

        // Link 5
        const l5 = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.1, 0.05), material);
        l5.position.y = 0.05;
        j5.add(l5);

        // Joint 6 (Wrist 3 / Flange)
        const j6 = new THREE.Group();
        j6.position.y = 0.05;
        l5.add(j6);
        this.addJointGeo(j6, jointMat);
        this.joints.push(j6);

        // End Effector Probe
        const probeGeo = new THREE.CylinderGeometry(0.01, 0.02, 0.2);
        const probe = new THREE.Mesh(probeGeo, new THREE.MeshStandardMaterial({ color: 0xff0000 }));
        probe.position.y = 0.1;
        j6.add(probe);
    }

    addJointGeo(parent, material) {
        const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.09), material);
        parent.add(sphere);
    }

    updateJoints(angles) {
        if (!angles || angles.length !== 6) return;

        // This is a rough mapping for visual verification
        // Backend handles real kinematics math
        // We just visualize the angles

        // J1: Y axis rotation (Waist)
        this.joints[0].rotation.y = angles[0];

        // J2: X axis (Shoulder)
        this.joints[1].rotation.x = angles[1];

        // J3: X axis (Elbow)
        this.joints[2].rotation.x = angles[2];

        // J4: Y axis (Wrist 1)
        this.joints[3].rotation.y = angles[3];

        // J5: X axis (Wrist 2)
        this.joints[4].rotation.x = angles[4];

        // J6: X axis (Wrist 3)
        this.joints[5].rotation.x = angles[5];
    }

    setLaser(enabled) {
        this.laserBeam.material.opacity = enabled ? 0.6 : 0.0;
        // Make tumor pulse if laser is on
        if (enabled && this.tumor) {
            this.tumor.material.emissiveIntensity = 0.5 + Math.sin(Date.now() * 0.01) * 0.3;
        } else if (this.tumor) {
            this.tumor.material.emissiveIntensity = 0.1;
        }
    }

    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    animate() {
        requestAnimationFrame(this.animate);

        // Piezoelectric Micro-vibration Simulation
        // Add subtle high-freq vibration to end effector visual if simulating
        if (this.simulating) {
            // Move camera slightly
        }

        this.renderer.render(this.scene, this.camera);
    }
}
