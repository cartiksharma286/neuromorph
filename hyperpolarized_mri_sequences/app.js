// Main Application Controller
// Hyperpolarized Pulse Sequence Generator

class HyperpolarizedApp {
    constructor() {
        this.currentSequenceType = 'vfa';
        this.currentNuclei = 'c13-pyruvate';
        this.currentSequenceData = null;
        this.init();
    }

    init() {
        console.log('Initializing Hyperpolarized Pulse Sequence Generator...');

        // Initialize UI handlers
        this.initNavigation();
        this.initNucleiSelector();
        this.initSequenceHandlers();
        this.initExportHandlers();

        // Load default nuclei
        this.updateNucleiInfo(this.currentNuclei);

        console.log('Application ready');
    }

    // ===== Navigation =====

    initNavigation() {
        const navButtons = document.querySelectorAll('.nav-item');

        navButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const sequenceType = button.dataset.sequence;
                this.switchSequenceType(sequenceType);
            });
        });
    }

    switchSequenceType(type) {
        // Update navigation
        document.querySelectorAll('.nav-item').forEach(btn => {
            btn.classList.remove('active');
        });

        document.querySelector(`[data-sequence="${type}"]`).classList.add('active');

        // Update panels
        document.querySelectorAll('.sequence-panel').forEach(panel => {
            panel.classList.remove('active');
        });

        document.getElementById(`panel-${type}`).classList.add('active');

        this.currentSequenceType = type;
    }

    // ===== Nuclei Selector =====

    initNucleiSelector() {
        const nucleiSelect = document.getElementById('nuclei-select');

        nucleiSelect.addEventListener('change', (e) => {
            this.currentNuclei = e.target.value;
            this.updateNucleiInfo(e.target.value);
        });
    }

    updateNucleiInfo(nucleiId) {
        const props = NucleiHelper.getProperties(nucleiId);

        document.getElementById('gamma-value').textContent = `${Math.abs(props.gamma).toFixed(4)} MHz/T`;
        document.getElementById('t1-value').textContent = `${props.t1} s`;
        document.getElementById('t2-value').textContent = `${props.t2} s`;
    }

    // ===== VFA Sequence =====

    initSequenceHandlers() {
        // VFA Calculator
        document.getElementById('calculate-vfa')?.addEventListener('click', () => {
            this.calculateVFA();
        });

        // Spiral Designer
        document.getElementById('calculate-spiral')?.addEventListener('click', () => {
            this.designSpiral();
        });

        // EPI Configurator
        document.getElementById('calculate-epi')?.addEventListener('click', () => {
            this.designEPI();
        });

        // Dynamic Sequence
        document.getElementById('calculate-dynamic')?.addEventListener('click', () => {
            this.designDynamic();
        });

        // Metabolic Imaging
        document.getElementById('calculate-metabolic')?.addEventListener('click', () => {
            this.designMetabolic();
        });
    }

    calculateVFA() {
        const numFrames = parseInt(document.getElementById('vfa-num-frames').value);
        const tr = parseFloat(document.getElementById('vfa-tr').value);
        const t1 = parseFloat(document.getElementById('vfa-t1').value);
        const strategy = document.getElementById('vfa-strategy').value;

        console.log('Calculating VFA:', { numFrames, tr, t1, strategy });

        // Generate VFA schedule
        const schedule = VFACalculator.generateSchedule(numFrames, t1, tr, strategy);

        // Store current sequence data
        this.currentSequenceData = {
            type: 'vfa',
            schedule: schedule,
            nuclei: this.currentNuclei
        };

        // Visualize
        this.visualizeVFASchedule(schedule);
        this.visualizeSignalEvolution(schedule);
        this.displayVFATable(schedule);
    }

    visualizeVFASchedule(schedule) {
        const canvas = document.getElementById('vfa-chart');
        const ctx = canvas.getContext('2d');

        // Clear
        ctx.fillStyle = '#1a2235';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw chart
        const margin = { top: 30, right: 30, bottom: 50, left: 60 };
        const width = canvas.width - margin.left - margin.right;
        const height = canvas.height - margin.top - margin.bottom;

        const maxAngle = Math.max(...schedule.flipAngles);

        // Draw axes
        ctx.strokeStyle = '#8b949e';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(margin.left, margin.top);
        ctx.lineTo(margin.left, margin.top + height);
        ctx.lineTo(margin.left + width, margin.top + height);
        ctx.stroke();

        // Draw bars
        const barWidth = width / schedule.flipAngles.length;

        schedule.flipAngles.forEach((angle, i) => {
            const barHeight = (angle / maxAngle) * height;
            const x = margin.left + i * barWidth;
            const y = margin.top + height - barHeight;

            // Gradient fill
            const gradient = ctx.createLinearGradient(x, y, x, margin.top + height);
            gradient.addColorStop(0, '#58a6ff');
            gradient.addColorStop(1, '#1f6feb');

            ctx.fillStyle = gradient;
            ctx.fillRect(x + 2, y, barWidth - 4, barHeight);
        });

        // Labels
        ctx.fillStyle = '#e6edf3';
        ctx.font = '14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Frame Number', margin.left + width / 2, canvas.height - 10);

        ctx.save();
        ctx.translate(15, margin.top + height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Flip Angle (°)', 0, 0);
        ctx.restore();
    }

    visualizeSignalEvolution(schedule) {
        const canvas = document.getElementById('signal-chart');
        SignalCalculator.drawSignalEvolution(canvas, schedule.signals, schedule.times, {
            labels: ['Signal']
        });
    }

    displayVFATable(schedule) {
        const tableContainer = document.getElementById('vfa-table');
        const rows = VFACalculator.exportTable(schedule);

        let html = `
            <table>
                <thead>
                    <tr>
                        <th>Frame</th>
                        <th>Time (s)</th>
                        <th>Flip Angle (°)</th>
                        <th>Signal</th>
                    </tr>
                </thead>
                <tbody>
        `;

        rows.forEach(row => {
            html += `
                <tr>
                    <td>${row.frame}</td>
                    <td>${row.time}</td>
                    <td>${row.flipAngle}</td>
                    <td>${row.signal}</td>
                </tr>
            `;
        });

        html += '</tbody></table>';
        tableContainer.innerHTML = html;
    }

    // ===== Spiral Sequence =====

    designSpiral() {
        const fov = parseFloat(document.getElementById('spiral-fov').value);
        const resolution = parseFloat(document.getElementById('spiral-resolution').value);
        const interleaves = parseInt(document.getElementById('spiral-interleaves').value);
        const maxGrad = parseFloat(document.getElementById('spiral-max-grad').value);
        const maxSlew = parseFloat(document.getElementById('spiral-max-slew').value);
        const spiralType = document.getElementById('spiral-type').value;

        console.log('Designing spiral:', { fov, resolution, interleaves, maxGrad, maxSlew });

        // Design trajectory
        const trajectory = spiralType === 'uniform' ?
            SpiralDesigner.designUniformSpiral(fov, resolution, interleaves, maxGrad, maxSlew) :
            SpiralDesigner.designVariableDensitySpiral(fov, resolution, interleaves, maxGrad, maxSlew);

        this.currentSequenceData = {
            type: 'spiral',
            trajectory: trajectory,
            nuclei: this.currentNuclei
        };

        // Visualize
        this.visualizeSpiralTrajectory(trajectory);
        this.visualizeSpiralGradients(trajectory);
        this.displaySpiralMetrics(trajectory, 100); // Assume 100ms TR
    }

    visualizeSpiralTrajectory(trajectory) {
        const canvas = document.getElementById('kspace-canvas');
        KSpaceVisualizer.drawTrajectory(canvas, trajectory, {
            showAllInterleaves: true,
            gradient: true
        });
    }

    visualizeSpiralGradients(trajectory) {
        const canvas = document.getElementById('gradient-canvas');
        const ctx = canvas.getContext('2d');

        ctx.fillStyle = '#1a2235';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const margin = { top: 30, right: 30, bottom: 50, left: 60 };
        const width = canvas.width - margin.left - margin.right;
        const height = canvas.height - margin.top - margin.bottom;

        // Draw Gx and Gy waveforms
        const samplesTo Show = Math.min(500, trajectory.gx.length);
        const maxGrad = Math.max(...trajectory.gx.map(Math.abs), ...trajectory.gy.map(Math.abs));

        // Gx (blue)
        ctx.strokeStyle = '#58a6ff';
        ctx.lineWidth = 2;
        ctx.beginPath();

        for (let i = 0; i < samplesToShow; i++) {
            const x = margin.left + (i / samplesToShow) * width;
            const y = margin.top + height / 2 - (trajectory.gx[i] / maxGrad) * (height / 2 - 10);

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Gy (green)
        ctx.strokeStyle = '#3fb950';
        ctx.beginPath();

        for (let i = 0; i < samplesToShow; i++) {
            const x = margin.left + (i / samplesToShow) * width;
            const y = margin.top + height / 2 - (trajectory.gy[i] / maxGrad) * (height / 2 - 10);

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Labels
        ctx.fillStyle = '#e6edf3';
        ctx.font = '12px Inter';
        ctx.fillText('Gx', margin.left + 10, margin.top + 20);
        ctx.fillStyle = '#3fb950';
        ctx.fillText('Gy', margin.left + 40, margin.top + 20);
    }

    displaySpiralMetrics(trajectory, tr) {
        const metrics = SpiralDesigner.calculateMetrics(trajectory, tr, trajectory.interleaves);

        document.getElementById('spiral-duration').textContent = metrics.readoutDuration;
        document.getElementById('spiral-scan-time').textContent = metrics.totalScanTime;
        document.getElementById('spiral-kmax').textContent = metrics.kmaxValue;
    }

    // ===== EPI Sequence =====

    designEPI() {
        const matrixSize = parseInt(document.getElementById('epi-matrix').value);
        const fov = parseFloat(document.getElementById('epi-fov').value);
        const shots = parseInt(document.getElementById('epi-shots').value);
        const partialFourier = document.getElementById('epi-partial-fourier').value;
        const trajectory = document.getElementById('epi-trajectory').value;

        console.log('Designing EPI:', { matrixSize, fov, shots, partialFourier });

        const epiTrajectory = EPIConfigurator.designEPI(matrixSize, fov, shots, partialFourier, trajectory);

        this.currentSequenceData = {
            type: 'epi',
            trajectory: epiTrajectory,
            nuclei: this.currentNuclei
        };

        // Visualize
        this.visualizeEPITrajectory(epiTrajectory);
        this.displayEPIMetrics(epiTrajectory);
    }

    visualizeEPITrajectory(trajectory) {
        const canvas = document.getElementById('epi-kspace-canvas');
        KSpaceVisualizer.drawTrajectory(canvas, trajectory, {
            gradient: false
        });
    }

    displayEPIMetrics(trajectory) {
        const metrics = EPIConfigurator.calculateMetrics(trajectory);

        document.getElementById('epi-echo-spacing').textContent = metrics.echoSpacing;
        document.getElementById('epi-te').textContent = metrics.effectiveTE;
        document.getElementById('epi-acq-time').textContent = metrics.acquisitionTime;
        document.getElementById('epi-bandwidth').textContent = metrics.bandwidthPerPixel;
    }

    // ===== Dynamic Sequence =====

    designDynamic() {
        const numFrames = parseInt(document.getElementById('dynamic-frames').value);
        const temporalRes = parseFloat(document.getElementById('dynamic-temporal-res').value);
        const readoutType = document.getElementById('dynamic-readout').value;
        const vfaEnabled = document.getElementById('dynamic-vfa').checked;

        const nucleiProps = NucleiHelper.getProperties(this.currentNuclei);

        const sequence = DynamicSequence.designDynamic(numFrames, temporalRes, readoutType, vfaEnabled, nucleiProps);

        this.currentSequenceData = {
            type: 'dynamic',
            sequence: sequence,
            nuclei: this.currentNuclei
        };

        console.log('Dynamic sequence designed:', sequence);
    }

    // ===== Metabolic Sequence =====

    designMetabolic() {
        const selectedMetabolites = Array.from(document.querySelectorAll('#metabolic-metabolites input:checked'))
            .map(cb => cb.value);
        const b0Field = parseFloat(document.getElementById('metabolic-b0').value);
        const sliceThickness = parseFloat(document.getElementById('metabolic-slice-thickness').value);
        const pulseType = document.getElementById('metabolic-pulse-type').value;

        const pulse = MetabolicImaging.designSpectralSpatial(selectedMetabolites, b0Field, sliceThickness, pulseType);

        this.currentSequenceData = {
            type: 'metabolic',
            pulse: pulse,
            nuclei: this.currentNuclei
        };

        console.log('Metabolic pulse designed:', pulse);
    }

    // ===== Export System =====

    initExportHandlers() {
        document.getElementById('export-btn')?.addEventListener('click', () => {
            this.showExportModal();
        });

        document.querySelector('.modal-close')?.addEventListener('click', () => {
            this.hideExportModal();
        });

        document.querySelectorAll('input[name="export-format"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.updateExportPreview(e.target.value);
            });
        });

        document.getElementById('copy-export')?.addEventListener('click', () => {
            this.copyExportToClipboard();
        });

        document.getElementById('download-export')?.addEventListener('click', () => {
            this.downloadExport();
        });
    }

    showExportModal() {
        document.getElementById('export-modal').classList.add('active');
        this.updateExportPreview('pypulseq');
    }

    hideExportModal() {
        document.getElementById('export-modal').classList.remove('active');
    }

    updateExportPreview(format) {
        if (!this.currentSequenceData) {
            document.getElementById('export-preview-content').textContent = 'No sequence generated yet. Please design a sequence first.';
            return;
        }

        let code = '';

        switch (format) {
            case 'pypulseq':
                code = this.exportPyPulseq();
                break;
            case 'json':
                code = this.exportJSON();
                break;
            case 'siemens':
                code = this.exportSiemens();
                break;
            case 'ge':
                code = this.exportGE();
                break;
            case 'philips':
                code = this.exportPhilips();
                break;
        }

        document.getElementById('export-preview-content').textContent = code;
    }

    exportPyPulseq() {
        const data = this.currentSequenceData;

        if (data.type === 'vfa' && data.schedule) {
            const nucleiProps = NucleiHelper.getProperties(data.nuclei);
            return PyPulseqExporter.exportVFASequence(data.schedule, 'spiral', nucleiProps, {
                fov: 24,
                sliceThickness: 10
            });
        } else if (data.type === 'spiral' && data.trajectory) {
            return SpiralDesigner.exportForPyPulseq(data.trajectory, data.trajectory.interleaves);
        }

        return '# PyPulseq export for this sequence type is under development';
    }

    exportJSON() {
        return JSONFormat.exportJSON(this.currentSequenceData);
    }

    exportSiemens() {
        return VendorExporter.exportSiemens(this.prepareVendorData());
    }

    exportGE() {
        return VendorExporter.exportGE(this.prepareVendorData());
    }

    exportPhilips() {
        return VendorExporter.exportPhilips(this.prepareVendorData());
    }

    prepareVendorData() {
        const data = { ...this.currentSequenceData };
        const nucleiProps = NucleiHelper.getProperties(this.currentNuclei);

        data.nucleus = nucleiProps.nucleus;
        data.gamma = nucleiProps.gamma;
        data.t1 = nucleiProps.t1;
        data.b0Field = nucleiProps.b0Field;

        if (data.schedule) {
            data.flipAngles = data.schedule.flipAngles;
            data.tr = data.schedule.parameters.tr;
            data.numFrames = data.schedule.parameters.numFrames;
        }

        return data;
    }

    copyExportToClipboard() {
        const text = document.getElementById('export-preview-content').textContent;
        navigator.clipboard.writeText(text).then(() => {
            alert('Copied to clipboard!');
        });
    }

    downloadExport() {
        const text = document.getElementById('export-preview-content').textContent;
        const format = document.querySelector('input[name="export-format"]:checked').value;

        const extensions = {
            'pypulseq': '.py',
            'json': '.json',
            'siemens': '_siemens.txt',
            'ge': '_ge.c',
            'philips': '_philips.txt'
        };

        const filename = `hyperpolarized_sequence${extensions[format]}`;

        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new HyperpolarizedApp();
});
