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
        // ─── Surgical OR Table ───────────────────────────────────────────
        const tableMat = new THREE.MeshStandardMaterial({
            color: 0x4a90d9,
            metalness: 0.55,
            roughness: 0.35,
        });
        const tableTop = new THREE.Mesh(new THREE.BoxGeometry(3.0, 0.08, 0.72), tableMat);
        tableTop.position.set(0, -0.08, 0);
        this.scene.add(tableTop);

        const legMat = new THREE.MeshStandardMaterial({ color: 0xb0b8c8, metalness: 0.8, roughness: 0.25 });
        const legData = [[-1.25, -0.34, 0.30], [1.25, -0.34, 0.30], [-1.25, -0.34, -0.30], [1.25, -0.34, -0.30]];
        for (const [lx, ly, lz] of legData) {
            const leg = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.52, 0.07), legMat);
            leg.position.set(lx, ly, lz);
            this.scene.add(leg);
        }

        // Table pad (foam mattress look)
        const padMat = new THREE.MeshStandardMaterial({ color: 0xecfeff, roughness: 0.9 });
        const pad = new THREE.Mesh(new THREE.BoxGeometry(2.8, 0.07, 0.65), padMat);
        pad.position.set(0, -0.005, 0);
        this.scene.add(pad);

        // 1. MRI Bore (Cylindrical Tunnel — no coil rings)
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

        // 2. 5G Signal Lines — outside the scanner bore
        this._build5GLines();

        // 3. Patient Visualization
        const patientGroup = new THREE.Group();
        this.scene.add(patientGroup);

        // Body (Simple Cylinder) lying on table
        const bodyGeo = new THREE.CylinderGeometry(0.25, 0.25, 1.8, 16);
        const bodyMat = new THREE.MeshStandardMaterial({ color: 0x8ecae6 }); // Hospital gown blue
        const body = new THREE.Mesh(bodyGeo, bodyMat);
        body.rotation.z = Math.PI / 2;
        body.position.set(-0.2, 0.21, 0); // Lying on pad
        patientGroup.add(body);

        // Head (Sphere) - Transparent to see "inside"
        const headGeo = new THREE.SphereGeometry(0.18, 32, 32);
        const headMat = new THREE.MeshPhysicalMaterial({
            color: 0xffdbac, // Skin tone
            transmission: 0.4,
            opacity: 0.5,
            transparent: true,
            roughness: 0.2,
            metalness: 0.1,
            clearcoat: 1.0
        });
        const head = new THREE.Mesh(headGeo, headMat);
        head.position.set(0.7, 0.26, 0); // On table at head end
        patientGroup.add(head);

        // ─── Stereotactic Head Frame (4-pin ring) ───────────────────────
        const frameMat = new THREE.MeshStandardMaterial({ color: 0xd4d4d4, metalness: 0.9, roughness: 0.1 });
        const ringGeo = new THREE.TorusGeometry(0.20, 0.012, 12, 64);
        const headRing = new THREE.Mesh(ringGeo, frameMat);
        headRing.position.copy(head.position);
        headRing.rotation.x = Math.PI / 2;
        patientGroup.add(headRing);

        // Four fixing pins
        const pinGeo = new THREE.CylinderGeometry(0.008, 0.008, 0.24, 8);
        const pinAngles = [0, Math.PI / 2, Math.PI, 3 * Math.PI / 2];
        for (const angle of pinAngles) {
            const pin = new THREE.Mesh(pinGeo, frameMat);
            pin.position.set(
                head.position.x + Math.cos(angle) * 0.20,
                head.position.y + 0.12,
                head.position.z + Math.sin(angle) * 0.20
            );
            patientGroup.add(pin);
        }

        // Brain Surface (Inner Layer)
        const brainGeo = new THREE.SphereGeometry(0.16, 32, 32);
        const brainMat = new THREE.MeshStandardMaterial({
            color: 0xf4f4f5,
            roughness: 0.5,
            wireframe: true,
            transparent: true,
            opacity: 0.1
        });
        const brain = new THREE.Mesh(brainGeo, brainMat);
        brain.position.copy(head.position);
        patientGroup.add(brain);

        // 3. Neurovasculature (Inside the Head)
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

        // ─── End-Effector Probe-Tip Indicator (inside brain) ────────────
        // A small glowing sphere that tracks where the laser tip is inside the brain volume
        const tipGeo = new THREE.SphereGeometry(0.012, 16, 16);
        this.endEffectorTipMat = new THREE.MeshStandardMaterial({
            color: 0x22d3ee,
            emissive: 0x22d3ee,
            emissiveIntensity: 0.0,
            transparent: true,
            opacity: 0.0,
        });
        this.endEffectorTip = new THREE.Mesh(tipGeo, this.endEffectorTipMat);
        this.endEffectorTip.position.copy(head.position);
        this.scene.add(this.endEffectorTip);

        // Targeting reticle ring around the probe tip
        const reticleGeo = new THREE.RingGeometry(0.016, 0.022, 32);
        const reticleMat = new THREE.MeshBasicMaterial({
            color: 0x22d3ee, side: THREE.DoubleSide, transparent: true, opacity: 0.0
        });
        this.endEffectorReticle = new THREE.Mesh(reticleGeo, reticleMat);
        this.endEffectorReticle.position.copy(head.position);
        this.endEffectorReticle.rotation.x = Math.PI / 2;
        this.scene.add(this.endEffectorReticle);

        // Store head world position for projecting effector coords
        this._headWorldPos = head.position.clone();

        // Save path
        this.simPath = curve;
    }

    /** Update the glowing probe tip position in brain space.
     *  @param {number} nx  – normalised robot X ∈ [0,1]
     *  @param {number} nz  – normalised robot Z ∈ [0,1]
     *  @param {boolean} active – true when laser is firing
     */
    setEndEffectorBrain(nx, nz, active) {
        if (!this.endEffectorTip) return;
        // Map normalised thermal-grid coords to 3D world space around the head
        const hp = this._headWorldPos;
        const wx = hp.x + (nx - 0.5) * 0.30;   // ±0.15 m inside head radius
        const wy = hp.y;
        const wz = hp.z + (nz - 0.5) * 0.28;

        this.endEffectorTip.position.set(wx, wy, wz);
        this.endEffectorReticle.position.set(wx, wy, wz);

        const intensity = active ? (0.8 + 0.2 * Math.sin(Date.now() * 0.025)) : 0.4;
        this.endEffectorTipMat.opacity    = active ? 0.95 : 0.55;
        this.endEffectorTipMat.emissiveIntensity = intensity;
        this.endEffectorReticle.material.opacity = active ? 0.85 : 0.35;

        // Pulse the reticle
        if (active) {
            const s = 1.0 + 0.3 * Math.abs(Math.sin(Date.now() * 0.018));
            this.endEffectorReticle.scale.set(s, s, s);
        }
    }

    _build5GLines() {
        // 5G transmission lines radiate outward from outside the bore
        // Placed at bore ends (x = ±2.5) as antenna towers
        this._5gLines = [];
        this._5gPhase = 0;

        const signalMat = new THREE.LineBasicMaterial({
            color: 0x06b6d4,
            transparent: true,
            opacity: 0.85,
        });

        const numAntennas = 8;
        const antennaPositions = [
            new THREE.Vector3(-2.5, 0.5, 0),
            new THREE.Vector3( 2.5, 0.5, 0),
        ];

        for (const origin of antennaPositions) {
            // Vertical tower
            const towerGeo = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(origin.x, -0.3, origin.z),
                new THREE.Vector3(origin.x,  1.8, origin.z),
            ]);
            this.scene.add(new THREE.Line(towerGeo, new THREE.LineBasicMaterial({ color: 0x64748b })));

            // Radial signal arcs from antenna top toward the bore
            for (let i = 0; i < numAntennas; i++) {
                const angle = (i / numAntennas) * Math.PI * 2;
                const dir = new THREE.Vector3(
                    Math.cos(angle) * 0.6 * (origin.x < 0 ? 1 : -1),
                    Math.sin(angle * 0.5) * 0.3,
                    Math.sin(angle) * 0.8
                );

                const pts = [];
                const steps = 20;
                for (let s = 0; s <= steps; s++) {
                    const t = s / steps;
                    pts.push(new THREE.Vector3(
                        origin.x + dir.x * t,
                        origin.y + 0.3 + dir.y * t + Math.sin(t * Math.PI) * 0.2,
                        origin.z + dir.z * t
                    ));
                }

                const geo = new THREE.BufferGeometry().setFromPoints(pts);
                const mat = new THREE.LineBasicMaterial({
                    color: new THREE.Color().setHSL(0.52 + i * 0.015, 1.0, 0.6),
                    transparent: true,
                    opacity: 0.7,
                });
                const line = new THREE.Line(geo, mat);
                this.scene.add(line);
                this._5gLines.push({ line, baseOpacity: 0.7, offset: i * 0.4 });
            }

            // Pulsing ring at antenna base
            const ringGeo = new THREE.RingGeometry(0.05, 0.12, 32);
            const ringMat = new THREE.MeshBasicMaterial({
                color: 0x06b6d4, side: THREE.DoubleSide, transparent: true, opacity: 0.9
            });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.position.copy(origin);
            ring.position.y += 0.35;
            ring.rotation.x = Math.PI / 2;
            this.scene.add(ring);
            this._5gLines.push({ line: ring, baseOpacity: 0.9, offset: 0, isRing: true });
        }
    }

    _animate5GLines(t) {
        if (!this._5gLines) return;
        for (const entry of this._5gLines) {
            const pulse = 0.3 + 0.7 * Math.abs(Math.sin(t * 1.8 + entry.offset));
            entry.line.material.opacity = entry.baseOpacity * pulse;
            if (entry.isRing) {
                const s = 0.8 + 0.5 * Math.abs(Math.sin(t * 2.2));
                entry.line.scale.set(s, s, s);
            }
        }
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
        // Neurosurgical stereotactic position:
        // Patient head is at world (0.7, 0.3, 0); robot is mounted directly above
        // on a cranial frame so its arm hangs down into the surgical field.
        this.robotRoot.position.set(0.7, 1.9, 0.0);
        // Flip arm downward for top-down cranial approach
        this.robotRoot.rotation.z = Math.PI;
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
        if (enabled) {
            // Enhanced laser animation
            this.laserBeam.material.opacity = 0.7;
            this.laserBeam.material.emissive = new THREE.Color(0xff4444);
            this.laserBeam.material.emissiveIntensity = 0.8 + Math.sin(Date.now() * 0.02) * 0.2;
            
            // Animated glow effect
            const glowScale = 1.0 + Math.sin(Date.now() * 0.015) * 0.15;
            this.laserBeam.scale.x = glowScale * 1.2;
            this.laserBeam.scale.z = glowScale * 1.2;
            
            // Tumor ablation effects
            if (this.tumor) {
                this.tumor.material.emissiveIntensity = 0.6 + Math.sin(Date.now() * 0.02) * 0.3;
                this.tumor.scale.set(1.0 + Math.sin(Date.now() * 0.03) * 0.1, 
                                    1.0 + Math.sin(Date.now() * 0.03) * 0.1,
                                    1.0 + Math.sin(Date.now() * 0.03) * 0.1);
                // Change color as if ablating
                const hue = 0.8 + Math.sin(Date.now() * 0.004) * 0.1;
                this.tumor.material.color.setHSL(hue, 0.8, 0.5);
            }
        } else {
            this.laserBeam.material.opacity = 0.0;
            this.laserBeam.material.emissiveIntensity = 0.0;
            this.laserBeam.scale.set(1.0, 1.0, 1.0);
            
            if (this.tumor) {
                this.tumor.material.emissiveIntensity = 0.1;
                this.tumor.scale.set(1.0, 1.0, 1.0);
                this.tumor.material.color.set(0x8b5cf6);
            }
        }
    }

    update5GGuidance(trajectoryPoints) {
        // Visualize 5G neural pathway guidance trajectory
        // Remove previous trajectory visualization if exists
        if (this.trajectoryPath) {
            this.scene.remove(this.trajectoryPath);
        }

        if (!trajectoryPoints || trajectoryPoints.length === 0) return;

        // Create path from trajectory points
        const pathGeometry = new THREE.BufferGeometry();
        const pathVertices = [];

        // Convert grid coordinates to 3D space
        for (let point of trajectoryPoints) {
            const x = (point[0] / 128.0) - 0.5;  // Convert from grid to robot space
            const z = point[1] / 128.0;
            pathVertices.push(x, 0.3, z);  // y=0.3 for height in 3D
        }

        pathGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pathVertices), 3));

        const pathMat = new THREE.LineBasicMaterial({
            color: 0x00ff00,
            linewidth: 3,
            transparent: true,
            opacity: 0.7,
            fog: false
        });

        this.trajectoryPath = new THREE.Line(pathGeometry, pathMat);
        this.scene.add(this.trajectoryPath);

        // Also add waypoint spheres along the path
        if (!this.waypointSpheres) {
            this.waypointSpheres = [];
        }

        // Clear old spheres
        this.waypointSpheres.forEach(sphere => this.scene.remove(sphere));
        this.waypointSpheres = [];

        // Add new waypoint spheres
        const sphereGeo = new THREE.SphereGeometry(0.02, 16, 16);
        const sphereMat = new THREE.MeshBasicMaterial({ color: 0x00ff00 });

        for (let i = 0; i < Math.min(trajectoryPoints.length, 30); i++) {
            const point = trajectoryPoints[i];
            const x = (point[0] / 128.0) - 0.5;
            const z = point[1] / 128.0;

            const sphere = new THREE.Mesh(sphereGeo, sphereMat);
            sphere.position.set(x, 0.3, z);
            this.scene.add(sphere);
            this.waypointSpheres.push(sphere);
        }
    }

    update5GProgress(progress) {
        // Update 5G guidance progress visualization
        if (this.waypointSpheres && this.waypointSpheres.length > 0) {
            const progressIndex = Math.floor(progress * this.waypointSpheres.length);
            
            for (let i = 0; i < this.waypointSpheres.length; i++) {
                if (i < progressIndex) {
                    // Completed waypoints - blue
                    this.waypointSpheres[i].material.color.set(0x0066ff);
                } else if (i === progressIndex) {
                    // Current waypoint - yellow/pulsing
                    const pulse = 0.5 + Math.sin(Date.now() * 0.005) * 0.5;
                    this.waypointSpheres[i].material.color.setHSL(0.15, 1.0, pulse);
                } else {
                    // Upcoming waypoints - green
                    this.waypointSpheres[i].material.color.set(0x00ff00);
                }
            }
        }
    }

    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    animate() {
        requestAnimationFrame(this.animate);

        const t = Date.now() * 0.001;

        // Animate 5G lines
        this._animate5GLines(t);

        this.renderer.render(this.scene, this.camera);
    }
}
