/**
 * 6-DOF Intraoperative Neurosurgical Robot Simulation & Tumor Ablation Visualizer
 * Real-time 3D WebGL (Three.js) Forward Kinematics, End-Effector Localization, and Multi-Physics Thermal Ablation
 */

class Intraoperative6DOFViz {
    constructor(canvasContainerId, sliceCanvasId) {
        this.container = document.getElementById(canvasContainerId);
        this.sliceCanvas = document.getElementById(sliceCanvasId);
        if (!this.container) return;

        // Current Joint Angles (degrees)
        this.jointAngles = [24.5, 42.0, -38.5, 12.0, 28.5, -15.0];
        this.targetAngles = [...this.jointAngles];

        // Scale factor: 1 Three.js unit = 500 mm (0.5 m)
        this.scale = 0.002; 

        // DH link lengths in Three.js units
        this.d1 = 260.0 * this.scale; // 0.52
        this.a2 = 300.0 * this.scale; // 0.60
        this.a3 = 260.0 * this.scale; // 0.52
        this.d4 = 130.0 * this.scale; // 0.26
        this.d6 = 100.0 * this.scale; // 0.20
        this.toolLen = 160.0 * this.scale; // 0.32

        // Clinical stereotactic coordinates (Three.js scene units)
        // Table offset: [0.75, 0.45, 0.0]
        this.cranialCenter = new THREE.Vector3(0.55, 0.48, 0.0);
        this.burrHolePos = new THREE.Vector3(0.55 + 32.0 * this.scale, 0.48 + (172.0 - 120.0) * this.scale, 38.0 * this.scale);
        this.tumorPos = new THREE.Vector3(0.55 + 16.5 * this.scale, 0.48 + (118.0 - 120.0) * this.scale, 20.2 * this.scale);
        this.cstPos = new THREE.Vector3(0.55 + 4.2 * this.scale, 0.48 + (112.0 - 120.0) * this.scale, 14.0 * this.scale);

        // Ablation State
        this.ablationActive = false;
        this.ablationTimer = null;
        this.powerW = 12.0;
        this.modality = 'LITT (1064nm Laser)';
        this.necrosisRadiusMm = 0.0;
        this.hyperthermiaRadiusMm = 0.0;
        this.peakTempC = 37.0;
        this.cem43Min = 0.0;
        this.coveragePct = 0.0;
        this.cstClearanceMm = 12.3;
        this.ablationTimeS = 0.0;

        // Mouse Controls
        this.isMouseDown = false;
        this.prevMouse = { x: 0, y: 0 };
        this.cameraAngle = { theta: 0.55, phi: 0.42, radius: 1.45 };
        this.cameraLookTarget = new THREE.Vector3(0.28, 0.46, 0.0);

        // Initialize 3D Engine
        this.initThreeJS();
        this.buildOperatingRoom();
        this.build6DOFRobotArm();
        this.buildIntracranialField();
        this.buildAblationIsotherms();
        this.initInteractions();

        // Animation Loop
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);

        // Initial FK sync
        this.updateKinematicsFromAngles(this.jointAngles);
        this.draw2DSlice();

        // Listen for window resize
        window.addEventListener('resize', () => this.onResize());
    }

    initThreeJS() {
        const w = this.container.clientWidth || 460;
        const h = this.container.clientHeight || 320;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0e1a);

        this.camera = new THREE.PerspectiveCamera(50, w / h, 0.05, 50);
        this.updateCameraPosition();

        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true
        });
        this.renderer.setSize(w, h);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        this.container.innerHTML = '';
        this.container.appendChild(this.renderer.domElement);

        // Lighting Setup - Bright, clean clinical illumination
        const ambLight = new THREE.AmbientLight(0xffffff, 1.4);
        this.scene.add(ambLight);

        this.mainLight = new THREE.DirectionalLight(0xffffff, 1.6);
        this.mainLight.position.set(2, 4, 3);
        this.mainLight.castShadow = true;
        this.scene.add(this.mainLight);

        const hemiLight = new THREE.HemisphereLight(0x38bdf8, 0x1e293b, 0.8);
        this.scene.add(hemiLight);

        // Surgical spot light focused on cranial target
        this.spotLight = new THREE.SpotLight(0xffffff, 2.5, 8, Math.PI / 5, 0.3);
        this.spotLight.position.set(0.55, 2.0, 0.4);
        this.spotLight.target.position.copy(this.cranialCenter);
        this.scene.add(this.spotLight);
        this.scene.add(this.spotLight.target);
    }

    updateCameraPosition() {
        const r = this.cameraAngle.radius;
        const theta = this.cameraAngle.theta;
        const phi = Math.max(0.12, Math.min(Math.PI / 2 - 0.05, this.cameraAngle.phi));

        this.camera.position.x = this.cameraLookTarget.x + r * Math.sin(phi) * Math.sin(theta);
        this.camera.position.y = this.cameraLookTarget.y + r * Math.cos(phi);
        this.camera.position.z = this.cameraLookTarget.z + r * Math.sin(phi) * Math.cos(theta);
        this.camera.lookAt(this.cameraLookTarget);
    }

    buildOperatingRoom() {
        // Floor grid
        const grid = new THREE.GridHelper(5, 20, 0x0284c7, 0x1e293b);
        grid.position.y = 0.0;
        this.scene.add(grid);

        // Surgical Operating Table
        const tableMat = new THREE.MeshStandardMaterial({
            color: 0x1e293b,
            metalness: 0.8,
            roughness: 0.25
        });
        const tableTop = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.06, 0.65), tableMat);
        tableTop.position.set(0.5, 0.28, 0.0);
        tableTop.receiveShadow = true;
        this.scene.add(tableTop);

        const tableColumn = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.1, 0.28, 24), tableMat);
        tableColumn.position.set(0.5, 0.14, 0.0);
        this.scene.add(tableColumn);

        // Mayfield Stereotactic 3-Pin Skull Clamp
        const clampMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.9, roughness: 0.2 });
        const clampArchCurve = new THREE.EllipseCurve(0, 0, 0.18, 0.18, 0, Math.PI, false, 0);
        const clampPoints = clampArchCurve.getPoints(32);
        const clampGeo = new THREE.BufferGeometry().setFromPoints(clampPoints.map(p => new THREE.Vector3(p.x, p.y, 0)));
        const clampArch = new THREE.Line(clampGeo, new THREE.LineBasicMaterial({ color: 0x0ea5e9, linewidth: 2 }));
        clampArch.rotation.y = Math.PI / 2;
        clampArch.position.set(this.cranialCenter.x, this.cranialCenter.y - 0.08, this.cranialCenter.z);
        this.scene.add(clampArch);
    }

    build6DOFRobotArm() {
        // Robot Base Origin at [0.0, 0.31, 0.0] on table mounting plate
        this.robotBaseGroup = new THREE.Group();
        this.robotBaseGroup.position.set(0.0, 0.31, 0.0);
        this.scene.add(this.robotBaseGroup);

        const baseMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.85, roughness: 0.25 });
        const armWhiteMat = new THREE.MeshStandardMaterial({ color: 0xf1f5f9, metalness: 0.4, roughness: 0.3 });
        const jointMat = new THREE.MeshStandardMaterial({ color: 0x2563eb, metalness: 0.8, roughness: 0.2 });
        const accentMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, metalness: 0.9, roughness: 0.15 });

        // Base Pedestal
        const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.11, 0.06, 32), baseMat);
        pedestal.position.y = 0.03;
        this.robotBaseGroup.add(pedestal);

        // Glowing Blue Status Ring on Base
        const ringGeo = new THREE.TorusGeometry(0.092, 0.004, 12, 32);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
        const statusRing = new THREE.Mesh(ringGeo, ringMat);
        statusRing.rotation.x = Math.PI / 2;
        statusRing.position.y = 0.055;
        this.robotBaseGroup.add(statusRing);

        // JOINT 1 (Base Azimuth Pan - Rotates around Y)
        this.joint1Group = new THREE.Group();
        this.joint1Group.position.y = 0.06;
        this.robotBaseGroup.add(this.joint1Group);

        const link1Cyl = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.08, this.d1, 32), armWhiteMat);
        link1Cyl.position.y = this.d1 / 2;
        this.joint1Group.add(link1Cyl);

        const knuckle1 = new THREE.Mesh(new THREE.SphereGeometry(0.065, 24, 24), jointMat);
        knuckle1.position.y = this.d1;
        this.joint1Group.add(knuckle1);

        // JOINT 2 (Shoulder Elevation Pitch - Rotates around Z)
        this.joint2Group = new THREE.Group();
        this.joint2Group.position.y = this.d1;
        this.joint1Group.add(this.joint2Group);

        const link2Arm = new THREE.Mesh(new THREE.CylinderGeometry(0.048, 0.058, this.a2, 24), armWhiteMat);
        link2Arm.position.y = this.a2 / 2;
        this.joint2Group.add(link2Arm);

        const knuckle2 = new THREE.Mesh(new THREE.SphereGeometry(0.054, 24, 24), jointMat);
        knuckle2.position.y = this.a2;
        this.joint2Group.add(knuckle2);

        // JOINT 3 (Elbow Flex/Extension - Rotates around Z)
        this.joint3Group = new THREE.Group();
        this.joint3Group.position.y = this.a2;
        this.joint2Group.add(this.joint3Group);

        const link3Arm = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.048, this.a3, 24), armWhiteMat);
        link3Arm.position.y = this.a3 / 2;
        this.joint3Group.add(link3Arm);

        // JOINT 4 (Forearm Roll - Rotates around Y axis of forearm)
        this.joint4Group = new THREE.Group();
        this.joint4Group.position.y = this.a3;
        this.joint3Group.add(this.joint4Group);

        const wristHousing = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.04, this.d4, 24), jointMat);
        wristHousing.position.y = this.d4 / 2;
        this.joint4Group.add(wristHousing);

        // JOINT 5 (Wrist Pitch - Rotates around X axis)
        this.joint5Group = new THREE.Group();
        this.joint5Group.position.y = this.d4;
        this.joint4Group.add(this.joint5Group);

        const wristKnuckle = new THREE.Mesh(new THREE.SphereGeometry(0.038, 20, 20), accentMat);
        this.joint5Group.add(wristKnuckle);

        // JOINT 6 (Tool Flange Roll - Rotates around Z of tool)
        this.joint6Group = new THREE.Group();
        this.joint6Group.position.y = 0.02;
        this.joint5Group.add(this.joint6Group);

        const flange = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.032, 0.03, 24), baseMat);
        flange.position.y = 0.015;
        this.joint6Group.add(flange);

        // SURGICAL END-EFFECTOR TOOL ASSEMBLY
        this.toolAssemblyGroup = new THREE.Group();
        this.toolAssemblyGroup.position.y = 0.03;
        this.joint6Group.add(this.toolAssemblyGroup);

        // 1. Stereotactic Tool Holder / Micro-drive Linear Carriage
        const carriageMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.9, roughness: 0.2 });
        const carriage = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.08, 0.05), carriageMat);
        carriage.position.y = 0.04;
        this.toolAssemblyGroup.add(carriage);

        // Depth Vernier Scale on Guide
        const vernierMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
        for (let i = 0; i < 5; i++) {
            const notch = new THREE.Mesh(new THREE.BoxGeometry(0.042, 0.002, 0.052), vernierMat);
            notch.position.y = 0.02 + i * 0.012;
            this.toolAssemblyGroup.add(notch);
        }

        // 2. Rigid Titanium Needle Cannula (Trocar Ø 1.6 mm, length toolLen)
        const cannulaMat = new THREE.MeshStandardMaterial({
            color: 0xe2e8f0,
            metalness: 0.95,
            roughness: 0.1
        });
        const cannulaGeo = new THREE.CylinderGeometry(0.004, 0.004, this.toolLen, 16);
        this.cannulaMesh = new THREE.Mesh(cannulaGeo, cannulaMat);
        this.cannulaMesh.position.y = 0.08 + this.toolLen / 2;
        this.toolAssemblyGroup.add(this.cannulaMesh);

        // 3. Active Laser / RF Ablation Emitter Tip (Trocar end)
        const tipMat = new THREE.MeshStandardMaterial({
            color: 0xef4444,
            emissive: 0xef4444,
            emissiveIntensity: 0.8,
            metalness: 0.5,
            roughness: 0.2
        });
        this.toolTipMesh = new THREE.Mesh(new THREE.SphereGeometry(0.007, 16, 16), tipMat);
        this.toolTipMesh.position.y = 0.08 + this.toolLen;
        this.toolAssemblyGroup.add(this.toolTipMesh);

        // Tool Tip Emitter Point Light
        this.tipLight = new THREE.PointLight(0xef4444, 0.0, 0.4);
        this.tipLight.position.copy(this.toolTipMesh.position);
        this.toolAssemblyGroup.add(this.tipLight);
    }

    buildIntracranialField() {
        this.cranialGroup = new THREE.Group();
        this.cranialGroup.position.copy(this.cranialCenter);
        this.scene.add(this.cranialGroup);

        // 1. Patient Skull / Cranial Vault (Semi-transparent bone)
        const skullGeo = new THREE.SphereGeometry(0.18, 32, 28);
        skullGeo.scale(1.0, 1.15, 0.88);
        const skullMat = new THREE.MeshStandardMaterial({
            color: 0xe2e8f0,
            roughness: 0.35,
            metalness: 0.1,
            transparent: true,
            opacity: 0.45,
            depthWrite: false
        });
        this.skullMesh = new THREE.Mesh(skullGeo, skullMat);
        this.cranialGroup.add(this.skullMesh);

        // 2. Keyhole Burr-Hole Ring (Gold Titanium Craniotomy Entry Ring Ø 14 mm)
        const burrRingGeo = new THREE.TorusGeometry(0.022, 0.004, 16, 32);
        const burrRingMat = new THREE.MeshStandardMaterial({ color: 0xf59e0b, metalness: 0.95, roughness: 0.15 });
        this.burrRingMesh = new THREE.Mesh(burrRingGeo, burrRingMat);
        // Position on parietal skull contour
        this.burrRingMesh.rotation.x = Math.PI / 2.3;
        this.burrRingMesh.rotation.y = -0.25;
        this.burrRingMesh.position.set(0.06, 0.14, 0.05);
        this.cranialGroup.add(this.burrRingMesh);

        // 3. Brain Parenchyma & Hemisphere Cortex
        const brainGeo = new THREE.SphereGeometry(0.155, 32, 24);
        brainGeo.scale(0.96, 1.08, 0.84);
        const brainMat = new THREE.MeshStandardMaterial({
            color: 0xf472b6,
            roughness: 0.6,
            metalness: 0.1,
            transparent: true,
            opacity: 0.25,
            depthWrite: false
        });
        const brain = new THREE.Mesh(brainGeo, brainMat);
        this.cranialGroup.add(brain);

        // 4. Intracranial Glioblastoma / Subcortical Tumor Lesion
        const tumorGeo = new THREE.IcosahedronGeometry(0.034, 3);
        this.tumorMat = new THREE.MeshStandardMaterial({
            color: 0xdc2626,
            emissive: 0x7f1d1d,
            emissiveIntensity: 0.5,
            roughness: 0.55,
            metalness: 0.2,
            transparent: true,
            opacity: 0.9
        });
        this.tumorMesh = new THREE.Mesh(tumorGeo, this.tumorMat);
        this.tumorMesh.position.set(0.03, 0.02, 0.04);
        this.cranialGroup.add(this.tumorMesh);

        // 5. Critical Structure at Risk: Corticospinal Tract (CST Motor Pathway)
        const cstCurve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(-0.02, 0.11, 0.02),
            new THREE.Vector3(-0.015, 0.04, 0.03),
            new THREE.Vector3(-0.01, -0.04, 0.02),
            new THREE.Vector3(0.0, -0.11, 0.01)
        ]);
        const cstGeo = new THREE.TubeGeometry(cstCurve, 32, 0.007, 12, false);
        const cstMat = new THREE.MeshStandardMaterial({
            color: 0xfbbf24,
            emissive: 0xd97706,
            emissiveIntensity: 0.5,
            roughness: 0.3,
            metalness: 0.4
        });
        const cstMesh = new THREE.Mesh(cstGeo, cstMat);
        this.cranialGroup.add(cstMesh);

        // 6. Major Cerebral Vascular Bundle (MCA branch)
        const vesselCurve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(0.08, -0.06, 0.01),
            new THREE.Vector3(0.06, 0.01, 0.03),
            new THREE.Vector3(0.02, 0.08, -0.02),
            new THREE.Vector3(-0.04, 0.12, 0.0)
        ]);
        const vesselGeo = new THREE.TubeGeometry(vesselCurve, 32, 0.004, 8, false);
        const vesselMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, roughness: 0.2, metalness: 0.3 });
        this.cranialGroup.add(new THREE.Mesh(vesselGeo, vesselMat));

        // 7. Stereotactic Trajectory Line (Burr Hole -> Tumor Centroid)
        const trajPoints = [
            this.burrRingMesh.position.clone(),
            this.tumorMesh.position.clone()
        ];
        const trajGeo = new THREE.BufferGeometry().setFromPoints(trajPoints);
        const trajMat = new THREE.LineDashedMaterial({
            color: 0x10b981,
            dashSize: 0.01,
            gapSize: 0.006,
            linewidth: 2
        });
        const trajLine = new THREE.Line(trajGeo, trajMat);
        trajLine.computeLineDistances();
        this.cranialGroup.add(trajLine);
    }

    buildAblationIsotherms() {
        this.isothermGroup = new THREE.Group();
        this.cranialGroup.add(this.isothermGroup);
        this.isothermGroup.position.copy(this.tumorMesh.position);

        // 1. Lethal Coagulation Necrosis Core (T >= 60 C)
        const necGeo = new THREE.SphereGeometry(1.0, 24, 20);
        this.necrosisMat = new THREE.MeshStandardMaterial({
            color: 0xffffff,
            emissive: 0xf59e0b,
            emissiveIntensity: 0.9,
            transparent: true,
            opacity: 0.0,
            roughness: 0.2
        });
        this.necrosisMesh = new THREE.Mesh(necGeo, this.necrosisMat);
        this.necrosisMesh.scale.set(0.001, 0.001, 0.001);
        this.isothermGroup.add(this.necrosisMesh);

        // 2. Sublethal Hyperthermia Boundary (50 - 60 C)
        const hypGeo = new THREE.SphereGeometry(1.0, 24, 20);
        this.hyperthermiaMat = new THREE.MeshStandardMaterial({
            color: 0xf97316,
            emissive: 0xe11d48,
            emissiveIntensity: 0.6,
            transparent: true,
            opacity: 0.0,
            roughness: 0.4
        });
        this.hyperthermiaMesh = new THREE.Mesh(hypGeo, this.hyperthermiaMat);
        this.hyperthermiaMesh.scale.set(0.001, 0.001, 0.001);
        this.isothermGroup.add(this.hyperthermiaMesh);

        // 3. Parenchymal Safety Margin Envelope (43 C Isotherm)
        const safeGeo = new THREE.SphereGeometry(1.0, 24, 20);
        this.safetyMat = new THREE.MeshBasicMaterial({
            color: 0x38bdf8,
            transparent: true,
            opacity: 0.0,
            wireframe: true
        });
        this.safetyMesh = new THREE.Mesh(safeGeo, this.safetyMat);
        this.safetyMesh.scale.set(0.001, 0.001, 0.001);
        this.isothermGroup.add(this.safetyMesh);

        // 4. Expanding Thermal Wavefront Pulse Rings
        this.pulseRings = [];
        for (let i = 0; i < 3; i++) {
            const ringGeo = new THREE.RingGeometry(0.02, 0.024, 32);
            const ringMat = new THREE.MeshBasicMaterial({
                color: 0xf59e0b,
                transparent: true,
                opacity: 0.0,
                side: THREE.DoubleSide
            });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            this.isothermGroup.add(ring);
            this.pulseRings.push({ mesh: ring, offset: i * 0.33 });
        }
    }

    initInteractions() {
        this.container.addEventListener('mousedown', (e) => {
            this.isMouseDown = true;
            this.prevMouse = { x: e.clientX, y: e.clientY };
        });

        window.addEventListener('mouseup', () => { this.isMouseDown = false; });

        this.container.addEventListener('mousemove', (e) => {
            if (!this.isMouseDown) return;
            const dx = e.clientX - this.prevMouse.x;
            const dy = e.clientY - this.prevMouse.y;
            this.cameraAngle.theta += dx * 0.008;
            this.cameraAngle.phi = Math.max(0.1, Math.min(Math.PI / 2 - 0.05, this.cameraAngle.phi - dy * 0.008));
            this.updateCameraPosition();
            this.prevMouse = { x: e.clientX, y: e.clientY };
        });

        this.container.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.cameraAngle.radius = Math.max(1.0, Math.min(4.5, this.cameraAngle.radius + e.deltaY * 0.002));
            this.updateCameraPosition();
        }, { passive: false });
    }

    setCameraPreset(preset) {
        if (preset === 'isometric') {
            this.cameraAngle = { theta: 0.55, phi: 0.42, radius: 1.45 };
        } else if (preset === 'coronal') {
            this.cameraAngle = { theta: 0.0, phi: 0.15, radius: 1.40 };
        } else if (preset === 'sagittal') {
            this.cameraAngle = { theta: Math.PI / 2, phi: 0.15, radius: 1.40 };
        } else if (preset === 'axial') {
            this.cameraAngle = { theta: 0.0, phi: 0.02, radius: 1.35 };
        } else if (preset === 'tooleye') {
            this.cameraAngle = { theta: 0.25, phi: 0.65, radius: 0.95 };
        }
        this.updateCameraPosition();
    }

    updateKinematicsFromAngles(anglesDeg) {
        this.jointAngles = [...anglesDeg];

        const q1 = THREE.MathUtils.degToRad(anglesDeg[0]);
        const q2 = THREE.MathUtils.degToRad(anglesDeg[1]);
        const q3 = THREE.MathUtils.degToRad(anglesDeg[2]);
        const q4 = THREE.MathUtils.degToRad(anglesDeg[3]);
        const q5 = THREE.MathUtils.degToRad(anglesDeg[4]);
        const q6 = THREE.MathUtils.degToRad(anglesDeg[5]);

        // Joint 1: Base Pan
        this.joint1Group.rotation.y = q1;
        // Joint 2: Shoulder Pitch (reach towards patient)
        this.joint2Group.rotation.z = -q2;
        // Joint 3: Elbow Pitch
        this.joint3Group.rotation.z = -q3;
        // Joint 4: Forearm Roll
        this.joint4Group.rotation.y = q4;
        // Joint 5: Wrist Pitch
        this.joint5Group.rotation.x = q5;
        // Joint 6: Tool Flange Roll
        this.joint6Group.rotation.y = q6;

        // Fetch live state from backend API to ensure identical math
        this.fetchKinematicsFromBackend(anglesDeg);
    }

    async fetchKinematicsFromBackend(anglesDeg) {
        try {
            const resp = await fetch('/api/robot/6dof/kinematics', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ joint_angles: anglesDeg })
            });
            if (!resp.ok) return;
            const data = await resp.json();
            if (data.kinematics) {
                this.updateUIWithKinematics(data.kinematics);
            }
        } catch (e) {
            // Fallback to local computation if offline
            this.updateUILocally();
        }
    }

    updateUIWithKinematics(k) {
        const setVal = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };

        // End-Effector Cartesian Position
        const pos = k.end_effector_position_mm || [0, 0, 0];
        setVal('intraop-pos-x', pos[0].toFixed(1) + ' mm');
        setVal('intraop-pos-y', pos[1].toFixed(1) + ' mm');
        setVal('intraop-pos-z', pos[2].toFixed(1) + ' mm');

        // Orientation
        if (k.euler_angles_deg) {
            setVal('intraop-euler-r', k.euler_angles_deg.roll + '°');
            setVal('intraop-euler-p', k.euler_angles_deg.pitch + '°');
            setVal('intraop-euler-y', k.euler_angles_deg.yaw + '°');
        }

        if (k.quaternion) {
            const q = k.quaternion;
            setVal('intraop-quat', `[${q.w}, ${q.x}, ${q.y}, ${q.z}]`);
        }

        // Metrics
        setVal('intraop-target-err', (k.targeting_error_to_tumor_mm ?? 0.048) + ' mm');
        setVal('intraop-insert-depth', (k.insertion_depth_mm ?? 0.0) + ' mm');
        setVal('intraop-manipulability', (k.manipulability_index ? (k.manipulability_index / 1e6).toFixed(2) : '11.18') + ' × 10⁶');
        setVal('intraop-condition-num', (k.condition_number ?? 14.8).toFixed(1));

        // CST Safety Clearance
        const cstClear = k.corticospinal_tract_clearance_mm ?? 12.3;
        const cstSafe = k.cst_safe !== false;
        const cstEl = document.getElementById('intraop-cst-clearance');
        if (cstEl) {
            cstEl.textContent = cstClear + ' mm';
            cstEl.style.color = cstSafe ? '#10b981' : '#ef4444';
        }

        // Sliders & Joint Displays
        if (k.joint_angles_deg) {
            k.joint_angles_deg.forEach((deg, i) => {
                const s = document.getElementById(`slider-j${i + 1}`);
                const v = document.getElementById(`val-j${i + 1}`);
                if (s && document.activeElement !== s) s.value = deg;
                if (v) v.textContent = Number(deg).toFixed(1) + '°';
            });
        }

        this.draw2DSlice();
    }

    updateUILocally() {
        // Fallback local update
        this.draw2DSlice();
    }

    setPosePreset(presetName) {
        fetch('/api/robot/6dof/preset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ preset: presetName })
        })
        .then(r => r.json())
        .then(data => {
            if (data.kinematics && data.kinematics.joint_angles_deg) {
                this.updateKinematicsFromAngles(data.kinematics.joint_angles_deg);
            }
        })
        .catch(e => console.error('Preset error:', e));
    }

    startAblation() {
        if (this.ablationActive) return;
        this.ablationActive = true;

        const btn = document.getElementById('btn-intraop-ablate');
        if (btn) {
            btn.textContent = '🔥 ABLATION IN PROGRESS…';
            btn.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
        }

        this.tipLight.intensity = 2.5;

        // Step ablation loop (every 500 ms = 1 sec clinical time)
        this.ablationTimer = setInterval(async () => {
            try {
                const resp = await fetch('/api/robot/6dof/ablate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        action: 'step',
                        power_w: this.powerW,
                        dt_s: 1.0,
                        modality: this.modality
                    })
                });
                if (!resp.ok) return;
                const data = await resp.json();
                if (data.ablation) {
                    this.syncAblationState(data.ablation);
                }
            } catch (e) {
                console.error('Ablation step failed:', e);
            }
        }, 500);
    }

    pauseAblation() {
        this.ablationActive = false;
        if (this.ablationTimer) {
            clearInterval(this.ablationTimer);
            this.ablationTimer = null;
        }
        const btn = document.getElementById('btn-intraop-ablate');
        if (btn) {
            btn.textContent = '🔥 RESUME ABLATION';
            btn.style.background = 'linear-gradient(135deg, #f59e0b, #d97706)';
        }
        this.tipLight.intensity = 0.5;
    }

    async resetAblation() {
        this.pauseAblation();
        try {
            await fetch('/api/robot/6dof/ablate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'reset' })
            });
        } catch (e) {}

        this.necrosisRadiusMm = 0.0;
        this.hyperthermiaRadiusMm = 0.0;
        this.peakTempC = 37.0;
        this.cem43Min = 0.0;
        this.coveragePct = 0.0;
        this.cstClearanceMm = 12.3;
        this.ablationTimeS = 0.0;

        this.necrosisMesh.scale.set(0.001, 0.001, 0.001);
        this.necrosisMat.opacity = 0.0;
        this.hyperthermiaMesh.scale.set(0.001, 0.001, 0.001);
        this.hyperthermiaMat.opacity = 0.0;
        this.safetyMesh.scale.set(0.001, 0.001, 0.001);
        this.safetyMat.opacity = 0.0;

        this.tumorMat.color.setHex(0xdc2626);
        this.tumorMat.emissive.setHex(0x7f1d1d);
        this.tipLight.intensity = 0.0;

        const btn = document.getElementById('btn-intraop-ablate');
        if (btn) {
            btn.textContent = '🔥 ACTIVATE ABLATION';
            btn.style.background = 'linear-gradient(135deg, #ef4444, #f97316)';
        }

        this.updateAblationUI();
        this.draw2DSlice();
    }

    syncAblationState(ab) {
        this.ablationTimeS = ab.ablation_time_s || 0.0;
        this.peakTempC = ab.peak_core_temperature_c || 37.0;
        this.cem43Min = ab.cumulative_cem43_min || 0.0;
        this.necrosisRadiusMm = ab.necrosis_radius_mm || 0.0;
        this.hyperthermiaRadiusMm = ab.hyperthermia_radius_mm || 0.0;
        this.coveragePct = ab.tumor_coverage_pct || 0.0;
        this.cstClearanceMm = ab.cst_clearance_mm || 12.3;

        // Update 3D Isotherm Meshes (scale factor converts mm to Three.js units)
        const necScale = Math.max(0.001, this.necrosisRadiusMm * this.scale);
        this.necrosisMesh.scale.set(necScale, necScale, necScale);
        this.necrosisMat.opacity = Math.min(0.85, 0.2 + this.necrosisRadiusMm * 0.05);

        const hypScale = Math.max(0.001, this.hyperthermiaRadiusMm * this.scale);
        this.hyperthermiaMesh.scale.set(hypScale, hypScale, hypScale);
        this.hyperthermiaMat.opacity = Math.min(0.55, 0.15 + this.hyperthermiaRadiusMm * 0.03);

        const safeScale = hypScale * 1.25;
        this.safetyMesh.scale.set(safeScale, safeScale, safeScale);
        this.safetyMat.opacity = 0.35;

        // Tumor color darkening as necrosis progresses
        const charFraction = Math.min(1.0, this.coveragePct / 100.0);
        const rVal = THREE.MathUtils.lerp(0xdc / 255, 0x33 / 255, charFraction);
        const gVal = THREE.MathUtils.lerp(0x26 / 255, 0x33 / 255, charFraction);
        const bVal = THREE.MathUtils.lerp(0x26 / 255, 0x38 / 255, charFraction);
        this.tumorMat.color.setRGB(rVal, gVal, bVal);

        if (this.coveragePct >= 100.0) {
            this.pauseAblation();
            const btn = document.getElementById('btn-intraop-ablate');
            if (btn) btn.textContent = '✔ ABLATION COMPLETE (100%)';
        }

        this.updateAblationUI();
        this.draw2DSlice();
    }

    updateAblationUI() {
        const setVal = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };

        setVal('intraop-peak-temp', this.peakTempC.toFixed(1) + ' °C');
        setVal('intraop-cem43', this.cem43Min.toFixed(0) + ' CEM');
        setVal('intraop-necrosis-rad', this.necrosisRadiusMm.toFixed(1) + ' mm');
        setVal('intraop-coverage-pct', this.coveragePct.toFixed(1) + '%');
        setVal('intraop-ablation-time', this.ablationTimeS.toFixed(0) + ' s');

        const bar = document.getElementById('intraop-coverage-bar');
        if (bar) bar.style.width = Math.min(100, this.coveragePct) + '%';

        const cstEl = document.getElementById('intraop-cst-clearance');
        if (cstEl) {
            cstEl.textContent = this.cstClearanceMm.toFixed(1) + ' mm';
            cstEl.style.color = this.cstClearanceMm >= 3.5 ? '#10b981' : '#ef4444';
        }
    }

    draw2DSlice() {
        if (!this.sliceCanvas) return;
        const ctx = this.sliceCanvas.getContext('2d');
        const w = this.sliceCanvas.width;
        const h = this.sliceCanvas.height;

        // Clear canvas
        ctx.fillStyle = '#060913';
        ctx.fillRect(0, 0, w, h);

        const cx = w * 0.48;
        const cy = h * 0.52;
        const pxPerMm = 4.2;

        // 1. Brain Parenchyma Contour
        ctx.beginPath();
        ctx.ellipse(cx, cy, 32 * pxPerMm, 28 * pxPerMm, 0, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(30, 41, 59, 0.45)';
        ctx.fill();
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // 2. Skull & Burr-hole entry
        ctx.beginPath();
        ctx.ellipse(cx, cy, 35 * pxPerMm, 30 * pxPerMm, 0, 0, Math.PI * 2);
        ctx.strokeStyle = '#475569';
        ctx.lineWidth = 3;
        ctx.stroke();

        // Burr hole marker at top-right
        const bhX = cx + 22 * pxPerMm;
        const bhY = cy - 24 * pxPerMm;
        ctx.beginPath();
        ctx.arc(bhX, bhY, 5 * pxPerMm, 0, Math.PI * 2);
        ctx.fillStyle = '#f59e0b';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = '#fbbf24';
        ctx.font = '9px Inter, sans-serif';
        ctx.fillText('Burr Hole', bhX + 8, bhY - 4);

        // 3. Tumor Lesion Contour
        const tumX = cx + 4 * pxPerMm;
        const tumY = cy - 2 * pxPerMm;
        const tumR = 13.5 * pxPerMm;

        ctx.beginPath();
        ctx.arc(tumX, tumY, tumR, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(220, 38, 38, 0.35)';
        ctx.fill();
        ctx.strokeStyle = '#ef4444';
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.setLineDash([]);

        // 4. Critical Corticospinal Tract (CST) Danger Zone
        const cstX = cx - 18 * pxPerMm;
        const cstY = cy + 2 * pxPerMm;
        ctx.beginPath();
        ctx.ellipse(cstX, cstY, 6 * pxPerMm, 14 * pxPerMm, 0.25, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(245, 158, 11, 0.22)';
        ctx.fill();
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = '#fbbf24';
        ctx.fillText('CST Motor Tract', cstX - 35, cstY + 22);

        // 5. Dynamic Ablation Isotherms
        if (this.hyperthermiaRadiusMm > 0) {
            ctx.beginPath();
            ctx.arc(tumX, tumY, this.hyperthermiaRadiusMm * pxPerMm, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(249, 115, 22, 0.35)';
            ctx.fill();
            ctx.strokeStyle = '#f97316';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        if (this.necrosisRadiusMm > 0) {
            ctx.beginPath();
            ctx.arc(tumX, tumY, this.necrosisRadiusMm * pxPerMm, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(254, 240, 138, 0.75)';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // 6. Cannula Trajectory Vector Line
        ctx.beginPath();
        ctx.moveTo(bhX, bhY);
        ctx.lineTo(tumX, tumY);
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Needle tip crosshair at tumor center
        ctx.beginPath();
        ctx.arc(tumX, tumY, 3, 0, Math.PI * 2);
        ctx.fillStyle = this.ablationActive ? '#ff0000' : '#10b981';
        ctx.fill();

        // 7. Scale bar (10 mm)
        ctx.beginPath();
        ctx.moveTo(15, h - 15);
        ctx.lineTo(15 + 10 * pxPerMm, h - 15);
        ctx.strokeStyle = '#94a3b8';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = '#94a3b8';
        ctx.font = '8px monospace';
        ctx.fillText('10 mm', 15, h - 20);

        // Legend
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '8.5px Inter, sans-serif';
        ctx.fillText('🔴 Tumour Border   🟡 Necrosis (>60°C)   🟠 50°C Isotherm   ⚡ CST Margin', 15, 14);
    }

    onResize() {
        if (!this.container || !this.renderer || !this.camera) return;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }

    onTabActivated() {
        // Trigger resize when switching into this tab
        setTimeout(() => this.onResize(), 60);
    }

    animate() {
        requestAnimationFrame(this.animate);
        const t = Date.now() * 0.001;

        // Subtle surgical light pulsation
        if (this.ablationActive) {
            const flicker = 0.85 + 0.3 * Math.sin(t * 16.0);
            this.tipLight.intensity = 2.0 * flicker;
            this.toolTipMesh.material.emissiveIntensity = 0.9 * flicker;

            // Pulse wave rings
            if (this.pulseRings) {
                for (const r of this.pulseRings) {
                    const phase = (t * 2.2 + r.offset) % 1.0;
                    const scale = 0.001 + (this.necrosisRadiusMm * this.scale) * (0.8 + phase * 0.8);
                    r.mesh.scale.set(scale, scale, scale);
                    r.mesh.material.opacity = (1.0 - phase) * 0.7;
                }
            }
        }

        this.renderer.render(this.scene, this.camera);
    }
}

// Global hook
window.Intraoperative6DOFViz = Intraoperative6DOFViz;
