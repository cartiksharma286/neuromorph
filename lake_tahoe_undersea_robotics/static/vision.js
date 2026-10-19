/**
 * Underwater Robot Camera & Vision System Client
 * Handles live canvas rendering of simulated turbid benthic camera feeds,
 * SVG plotting of elliptic moduli, optical attenuation, Weibull-GEV statistics,
 * variational quantum circuit training metrics, and Nature preprint downloads.
 */

document.addEventListener('DOMContentLoaded', async () => {
    const byId = id => document.getElementById(id);
    const form = byId('vision-form');
    const status = byId('vision-status');
    const runBtn = byId('btn-run-vision');
    const pdfBtn = byId('btn-vision-pdf');
    const sceneSelect = byId('vision-scene-select');
    const canvas = byId('vision-camera-canvas');
    let lastInputs = null;
    let lastReport = null;

    const fields = {
        turbidity_ntu: 'Turbidity (NTU)',
        target_distance_m: 'Target Distance (m)',
        depth_m: 'Survey Depth (m)',
        ambient_lux: 'Ambient Illumination (lux)',
        exposure_time_ms: 'Exposure Time (ms)',
        sensor_gain_iso: 'Sensor Gain (ISO)',
        pqc_layers: 'PQC Quantum Circuit Layers',
        seed: 'Random Seed'
    };

    const palette = ['#06b6d4', '#fbbf24', '#c084fc', '#10b981'];

    async function request(path, inputs) {
        const response = await fetch(`/api/vision/${path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputs)
        });
        if (!response.ok) {
            const type = response.headers.get('content-type') || '';
            const message = type.includes('application/json') ? (await response.json()).message : `HTTP ${response.status}`;
            throw new Error(message);
        }
        return response;
    }

    function plot(id, series, xlabel, ylabel) {
        const target = byId(id);
        if (!target) return;
        const namespace = 'http://www.w3.org/2000/svg';
        target.replaceChildren();
        const svg = document.createElementNS(namespace, 'svg');
        svg.setAttribute('viewBox', '0 0 600 310');
        svg.setAttribute('role', 'img');
        svg.setAttribute('aria-label', `${ylabel} against ${xlabel}. ${series.map(s => s.name).join(', ')}`);

        function add(tag, attributes, content) {
            const node = document.createElementNS(namespace, tag);
            Object.entries(attributes).forEach(([k, v]) => node.setAttribute(k, String(v)));
            if (content !== undefined) node.textContent = content;
            svg.append(node);
            return node;
        }

        const points = series.flatMap(s => s.points);
        const xs = points.map(p => p[0]);
        const ys = points.map(p => p[1]);
        let xmin = Math.min(...xs), xmax = Math.max(...xs);
        let ymin = Math.min(0, ...ys), ymax = Math.max(...ys);
        if (xmax === xmin) xmax += 1;
        if (ymax === ymin) ymax += 1;

        const x = v => 65 + (v - xmin) / (xmax - xmin) * 510;
        const y = v => 225 - (v - ymin) / (ymax - ymin) * 200;

        add('title', {}, `${ylabel} against ${xlabel}`);
        for (let i = 0; i <= 4; i++) {
            const yy = ymin + (ymax - ymin) * i / 4;
            const xx = xmin + (xmax - xmin) * i / 4;
            add('line', { x1: 65, x2: 575, y1: y(yy), y2: y(yy), stroke: '#334155' });
            add('text', { x: 58, y: y(yy) + 4, 'text-anchor': 'end' }, yy.toFixed(2));
            add('text', { x: x(xx), y: 245, 'text-anchor': 'middle' }, xx.toFixed(1));
        }

        add('text', { x: 65, y: 15 }, ylabel);
        add('text', { x: 310, y: 266, 'text-anchor': 'middle' }, xlabel);

        series.forEach((s, i) => {
            const color = palette[i % palette.length];
            add('polyline', {
                points: s.points.map(p => `${x(p[0])},${y(p[1])}`).join(' '),
                fill: 'none', stroke: color, 'stroke-width': 2
            });
            add('text', { x: 65 + (i % 2) * 260, y: 285 + Math.floor(i / 2) * 17, fill: color }, s.name);
        });

        target.append(svg);
    }

    function renderCanvas(report) {
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        const inp = report.inputs;
        const opt = report.optics;
        const el = report.elliptic;
        const qm = report.qml;
        const preset = report.scene_preset;

        // Background water radiance gradient
        const turb = inp.turbidity_ntu;
        const depth = inp.depth_m;
        const grad = ctx.createRadialGradient(w / 2, h / 2, 40, w / 2, h / 2, w / 1.4);

        // Water color grading based on depth and turbidity
        const greenFactor = Math.min(65, Math.floor(turb * 2.2 + 25));
        const blueFactor = Math.max(15, Math.floor(60 - depth * 0.08));
        grad.addColorStop(0, `rgb(6, ${greenFactor + 15}, ${blueFactor + 30})`);
        grad.addColorStop(1, `rgb(2, ${Math.floor(greenFactor * 0.4)}, ${Math.floor(blueFactor * 0.5)})`);

        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);

        // Simulated particulate marine snow & backscatter speckle
        const rng = (seed) => {
            let s = seed % 2147483647;
            if (s <= 0) s += 2147483646;
            return () => (s = s * 16807 % 2147483647) / 2147483647;
        };
        const random = rng(inp.seed + Math.floor(turb * 10));
        const numParticles = Math.min(220, Math.floor(turb * 8 + 35));

        ctx.fillStyle = 'rgba(210, 245, 255, 0.45)';
        for (let p = 0; p < numParticles; p++) {
            const px = random() * w;
            const py = random() * h;
            const pr = random() * 2.5 + 0.8;
            ctx.beginPath();
            ctx.arc(px, py, pr, 0, Math.PI * 2);
            ctx.fill();
        }

        // Draw Target Object
        const obj = report.detected_objects[0];
        const [bx, by, bw, bh] = obj.bbox;

        // Target body
        ctx.save();
        ctx.translate(bx + bw / 2, by + bh / 2);
        ctx.strokeStyle = '#06b6d4';
        ctx.fillStyle = `rgba(14, 116, 144, ${Math.max(0.2, opt.contrast_transmission * 0.7)})`;
        ctx.lineWidth = 2.5;

        // Draw stylized geometric contour based on aspect ratio
        ctx.beginPath();
        if (preset.aspect_ratio > 2.0) {
            // Cylindrical pipeline / fissure
            ctx.roundRect(-bw / 2 + 10, -bh / 2 + 15, bw - 20, bh - 30, 12);
        } else {
            // Mound / beacon
            ctx.ellipse(0, 0, (bw - 25) / 2, (bh - 25) / 2, 0, 0, Math.PI * 2);
        }
        ctx.fill();
        ctx.stroke();
        ctx.restore();

        // Bounding Box Reticle & Targeting Brackets
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 1.8;
        const cornerLen = 16;

        // Top-left
        ctx.beginPath();
        ctx.moveTo(bx, by + cornerLen);
        ctx.lineTo(bx, by);
        ctx.lineTo(bx + cornerLen, by);
        ctx.stroke();

        // Top-right
        ctx.beginPath();
        ctx.moveTo(bx + bw - cornerLen, by);
        ctx.lineTo(bx + bw, by);
        ctx.lineTo(bx + bw, by + cornerLen);
        ctx.stroke();

        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(bx, by + bh - cornerLen);
        ctx.lineTo(bx, by + bh);
        ctx.lineTo(bx + cornerLen, by + bh);
        ctx.stroke();

        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(bx + bw - cornerLen, by + bh);
        ctx.lineTo(bx + bw, by + bh);
        ctx.lineTo(bx + bw, by + bh - cornerLen);
        ctx.stroke();

        // Target HUD Labels
        ctx.fillStyle = '#06b6d4';
        ctx.font = 'bold 12px monospace';
        ctx.fillText(`TARGET: ${obj.label}`, bx, by - 24);

        ctx.fillStyle = '#f8fafc';
        ctx.font = '11px monospace';
        ctx.fillText(`CONF: ${(obj.confidence * 100).toFixed(1)}% | Q-FID: ${qm.quantum_state_fidelity.toFixed(3)}`, bx, by - 10);

        ctx.fillStyle = '#fbbf24';
        ctx.fillText(`MODULUS k: ${el.modulus_k.toFixed(3)} | log10[j]: ${el.log10_j.toFixed(2)}`, bx, by + bh + 16);

        ctx.fillStyle = '#f43f5e';
        ctx.fillText(`EVENT: ${obj.event_status}`, bx, by + bh + 30);

        // Top HUD Status Bar
        ctx.fillStyle = 'rgba(2, 6, 23, 0.75)';
        ctx.fillRect(0, 0, w, 24);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px monospace';
        ctx.fillText(`AUV OPTICAL CAM [CH 01] • DEPTH: ${inp.depth_m.toFixed(1)}m • TURB: ${inp.turbidity_ntu.toFixed(1)} NTU • DIST: ${inp.target_distance_m.toFixed(1)}m • EXP: ${inp.exposure_time_ms.toFixed(0)}ms • 30 FPS`, 10, 16);
    }

    function render(report) {
        lastReport = report;
        const inp = report.inputs;
        const opt = report.optics;
        const el = report.elliptic;
        const qm = report.qml;
        const st = report.statistics;

        // Render Canvas
        renderCanvas(report);

        // Telemetry readouts
        byId('vis-transmittance').textContent = `${(opt.optical_transmittance * 100).toFixed(1)}%`;
        byId('vis-j-invariant').textContent = el.j_invariant > 9999 ? el.j_invariant.toExponential(2) : el.j_invariant.toFixed(1);
        byId('vis-qml-conf').textContent = `${(qm.detection_probability * 100).toFixed(1)}%`;
        byId('vis-fidelity').textContent = qm.quantum_state_fidelity.toFixed(4);

        byId('vis-k').textContent = el.modulus_k.toFixed(4);
        byId('vis-tau').textContent = el.tau_imag.toFixed(4);
        byId('vis-c').textContent = `${opt.attenuation_coeff_m_minus_1.toFixed(3)} m⁻¹`;
        byId('vis-range').textContent = `${report.telemetry.visibility_range_m.toFixed(1)} m`;
        byId('vis-snr').textContent = `${report.telemetry.snr_db.toFixed(1)} dB`;

        // SVG Plots
        // 1. Klein j-invariant curve
        const kAxis = [];
        const jVals = [];
        for (let ki = 0.05; ki <= 0.95; ki += 0.03) {
            kAxis.push(ki);
            const k2 = ki * ki;
            const num = 256 * Math.pow(1 - k2 + k2 * k2, 3);
            const den = Math.max(1e-12, k2 * k2 * Math.pow(1 - k2, 2));
            jVals.push(Math.log10(Math.max(1.0, num / den)));
        }
        plot('vis-plot-j', [{
            name: 'log10[j(k)] Modular Invariant',
            points: kAxis.map((k, i) => [k, jVals[i]])
        }], 'Elliptic Modulus k in (0, 1)', 'log10[j(k)]');

        // 2. Beam Attenuation Spectrum
        plot('vis-plot-attenuation', [{
            name: 'Attenuation c(lambda) [m^-1]',
            points: opt.wavelengths_nm.map((w, i) => [w, opt.attenuation_spectrum_m_minus_1[i]])
        }], 'Optical Wavelength lambda (nm)', 'Attenuation c(lambda) [m^-1]');

        // 3. Statistical Distributions
        plot('vis-plot-distributions', [
            {
                name: 'Weibull Speckle PDF',
                points: st.weibull_intensity_x.map((x, i) => [x, st.weibull_pdf_y[i]])
            },
            {
                name: 'GEV Anomaly PDF',
                points: st.gev_anomaly_x.slice(0, 35).map((x, i) => [x + 1.0, st.gev_pdf_y[i]])
            }
        ], 'Normalized Intensity / Anomaly Variable', 'Probability Density (PDF)');

        // 4. QML Variational Loss & Quantum State Fidelity
        plot('vis-plot-qml', [
            {
                name: 'Variational Loss L(alpha)',
                points: qm.iterations.map((it, i) => [it, qm.variational_loss[i]])
            },
            {
                name: 'Quantum State Fidelity',
                points: qm.iterations.map((it, i) => [it, qm.training_fidelity[i]])
            }
        ], 'QML Optimization Epoch (Iterations)', 'Loss / Fidelity Metric');

        // 5. 60-Second Real-Time Optical Anomaly Timeline
        plot('vis-plot-events', [{
            name: 'GEV Optical Anomaly Score',
            points: st.timeline_sec.map((t, i) => [t, st.event_anomaly_score[i]])
        }], 'Mission Elapsed Time (seconds)', 'GEV Anomaly Metric');

        // 6. Ecological Conservation & MTF Resolution
        const con = report.conservation || {};
        if (con.gsd_mm_per_px && byId('vis-gsd')) {
            byId('vis-gsd').textContent = `${con.gsd_mm_per_px.toFixed(3)} mm/px`;
            byId('vis-swath').textContent = `${con.geo_extents.swath_width_m.toFixed(2)} m`;
            byId('vis-shannon').textContent = con.shannon_diversity_index.toFixed(2);
            byId('vis-periphyton').textContent = `${con.periphyton_cover_pct.toFixed(1)}%`;

            // Plot MTF
            plot('vis-plot-mtf', [
                {
                    name: 'Lens MTF (Diffraction)',
                    points: con.spatial_freq_lp_mm.map((f, i) => [f, con.lens_mtf[i]])
                },
                {
                    name: 'Turbid Water System MTF',
                    points: con.spatial_freq_lp_mm.map((f, i) => [f, con.system_mtf[i]])
                }
            ], 'Spatial Frequency f_s (lp/mm)', 'Modulation Transfer (MTF)');

            // Plot Bathymetric Elevation Transect
            plot('vis-plot-transect', [{
                name: 'Bathymetric Depth (m)',
                points: con.transect_km.map((dist, i) => [dist, con.transect_depth_m[i]])
            }], 'Lake Transect Distance (km)', 'Bathymetric Depth (m)');

            // Plot Sand and Substrate Reflectance
            plot('vis-plot-sand', [
                {
                    name: 'Quartz Sand rho(lambda)',
                    points: con.wavelengths_nm.map((w, i) => [w, con.rho_quartz_sand[i]])
                },
                {
                    name: 'Diatomaceous Silt rho(lambda)',
                    points: con.wavelengths_nm.map((w, i) => [w, con.rho_diatom_silt[i]])
                },
                {
                    name: 'Relict Deep Moss (60-120m)',
                    points: con.wavelengths_nm.map((w, i) => [w, con.rho_deep_moss[i]])
                },
                {
                    name: 'Shell Druse Substrate',
                    points: con.wavelengths_nm.map((w, i) => [w, con.rho_calcite_shell[i]])
                }
            ], 'Wavelength lambda (nm)', 'Spectral Reflectance rho(lambda)');
        }
    }

    async function simulate() {
        runBtn.disabled = true;
        pdfBtn.disabled = true;
        status.textContent = 'Simulating optical radiative transfer, elliptic moduli invariants, and QML classification...';

        try {
            const inputs = Object.fromEntries(
                Object.keys(fields).map(key => [key, Number(byId(`vis-${key}`).value)])
            );
            inputs.chosen_scene = sceneSelect.value;
            lastInputs = inputs;

            const response = await request('simulate', inputs);
            const result = await response.json();
            render(result.report);
            pdfBtn.disabled = false;
            status.textContent = 'Optical perception simulation complete. Target classified under QML and elliptic moduli theory.';
        } catch (err) {
            status.textContent = `Simulation failed: ${err.message}`;
            console.error('Vision simulation error:', err);
        } finally {
            runBtn.disabled = false;
        }
    }

    // Form submission
    form.addEventListener('submit', event => {
        event.preventDefault();
        simulate();
    });

    sceneSelect.addEventListener('change', () => {
        simulate();
    });

    // PDF Download
    pdfBtn.addEventListener('click', async () => {
        pdfBtn.disabled = true;
        status.textContent = 'Generating Nature-style vision preprint PDF...';
        try {
            const response = await request('preprint', lastInputs || {});
            const url = URL.createObjectURL(await response.blob());
            const link = document.createElement('a');
            link.href = url;
            link.download = 'Nature_Underwater_Robot_Quantum_Vision_Elliptic_Preprint.pdf';
            link.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
            status.textContent = 'Nature vision preprint PDF downloaded successfully.';
        } catch (err) {
            status.textContent = `PDF generation failed: ${err.message}`;
            console.error('Preprint error:', err);
        } finally {
            pdfBtn.disabled = false;
        }
    });

    // Initialize defaults from backend
    try {
        const response = await fetch('/api/vision/defaults');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        const grid = byId('vision-inputs-grid');
        grid.replaceChildren();

        Object.entries(fields).forEach(([key, title]) => {
            const label = document.createElement('label');
            label.textContent = title;
            const input = document.createElement('input');
            input.id = `vis-${key}`;
            input.className = 'form-control';
            input.type = 'number';
            input.required = true;
            [input.min, input.max] = data.limits[key];
            input.step = key in ['pqc_layers', 'seed'] ? '1' : 'any';
            input.value = data.inputs[key];
            label.append(input);
            grid.append(label);
        });

        if (data.inputs.chosen_scene) {
            sceneSelect.value = data.inputs.chosen_scene;
        }

        await simulate();
    } catch (err) {
        status.textContent = `Initialization failed: ${err.message}. Please reload.`;
        console.error('Vision init error:', err);
    }
});
