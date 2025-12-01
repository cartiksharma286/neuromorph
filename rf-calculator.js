// RF Calculator for MRI Coil Design
class RFCalculator {
    constructor() {
        // Gyromagnetic ratios (MHz/Tesla)
        this.gyromagneticRatios = {
            '1H': 42.58,   // Hydrogen
            '13C': 10.71,  // Carbon-13
            '31P': 17.25,  // Phosphorus-31
            '23Na': 11.27  // Sodium-23
        };
    }

    // Calculate Larmor frequency
    calculateLarmorFrequency(fieldStrength, nucleus = '1H') {
        const gamma = this.gyromagneticRatios[nucleus];
        return (gamma * fieldStrength).toFixed(2);
    }

    // Calculate loop inductance (Wheeler's formula for circular loop)
    calculateLoopInductance(diameter, wireDiameter) {
        // diameter and wireDiameter in mm
        // Returns inductance in nH

        const d = diameter / 1000; // Convert to meters
        const a = wireDiameter / 2000; // Wire radius in meters

        // Wheeler's formula: L = μ₀ * R * [ln(8R/a) - 2]
        // where μ₀ = 4π × 10^-7 H/m
        const mu0 = 4 * Math.PI * 1e-7;
        const R = d / 2;

        const L = mu0 * R * (Math.log(8 * R / a) - 2);

        // Convert to nH
        return (L * 1e9).toFixed(1);
    }

    // Calculate required capacitance for resonance
    calculateResonanceCapacitance(inductance, frequency) {
        // inductance in nH, frequency in MHz
        // Returns capacitance in pF

        const L = inductance * 1e-9; // Convert to H
        const f = frequency * 1e6;   // Convert to Hz

        // f = 1 / (2π√(LC))
        // C = 1 / (4π²f²L)
        const C = 1 / (4 * Math.PI * Math.PI * f * f * L);

        // Convert to pF
        return (C * 1e12).toFixed(1);
    }

    // Estimate Q-factor for a coil
    estimateQFactor(diameter, frequency, wireDiameter) {
        // Simplified Q-factor estimation
        // Q ≈ 2πfL/R where R includes skin effect resistance

        const freq = frequency * 1e6; // Hz
        const d = diameter / 1000; // meters
        const wireD = wireDiameter / 1000; // meters

        // Skin depth for copper at given frequency
        const skinDepth = Math.sqrt(1 / (Math.PI * freq * 4 * Math.PI * 1e-7 * 5.8e7));

        // Effective resistance with skin effect
        const length = Math.PI * d; // Loop circumference
        const resistance = (1.68e-8 * length) / (2 * Math.PI * wireD * skinDepth);

        // Calculate inductance
        const inductance = this.calculateLoopInductance(diameter, wireDiameter * 1000) * 1e-9;

        // Q = ωL/R
        const Q = (2 * Math.PI * freq * inductance) / resistance;

        return Math.round(Q);
    }

    // Calculate SNR relative to reference coil
    calculateRelativeSNR(coilDiameter, depth) {
        // Simplified SNR calculation
        // SNR ∝ (Volume of coil) / (Distance from center)²

        const radius = coilDiameter / 2;

        // Filling factor approximation
        const ff = Math.exp(-depth / radius);

        // Relative SNR (normalized to reference)
        const snr = ff * (radius / 40); // Normalized to 80mm reference coil

        return snr.toFixed(2);
    }

    // Draw B1 field penetration visualization
    drawFieldMap(canvas, coilDiameter) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.fillStyle = '#1a1a24';
        ctx.fillRect(0, 0, width, height);

        // Parameters
        const centerX = width / 2;
        const centerY = height - 50;
        const scale = 2.5;
        const radius = coilDiameter * scale;

        // Draw field intensity map
        const imageData = ctx.createImageData(width, height);
        const data = imageData.data;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                // Distance from coil center
                const dx = x - centerX;
                const dy = y - centerY;
                const dist = Math.sqrt(dx * dx + dy * dy);

                // Simple B1 field approximation (decays with distance)
                let intensity = Math.exp(-dist / (radius * 0.8));

                // Inside coil region
                if (dy > 0 && Math.abs(dx) < radius / 2) {
                    intensity *= 1.5;
                }

                intensity = Math.min(1, intensity);

                // Color mapping (blue to red gradient)
                const idx = (y * width + x) * 4;
                const r = intensity * 255;
                const g = intensity * 100;
                const b = (1 - intensity) * 255;

                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
                data[idx + 3] = 255;
            }
        }

        ctx.putImageData(imageData, 0, 0);

        // Draw coil outline
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 3;

        // Coil loops
        ctx.beginPath();
        ctx.ellipse(centerX - radius / 4, centerY, radius / 8, radius / 4, 0, 0, Math.PI * 2);
        ctx.stroke();

        ctx.beginPath();
        ctx.ellipse(centerX + radius / 4, centerY, radius / 8, radius / 4, 0, 0, Math.PI * 2);
        ctx.stroke();

        // Add labels
        ctx.fillStyle = '#f8fafc';
        ctx.font = '14px Inter';
        ctx.fillText('High B₁ Field', centerX - 50, centerY - radius / 3);
        ctx.fillText('RF Coil', centerX - 30, centerY + 30);

        // Draw scale
        ctx.strokeStyle = '#94a3b8';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(20, height - 20);
        ctx.lineTo(120, height - 20);
        ctx.stroke();
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px Inter';
        ctx.fillText('0', 20, height - 5);
        ctx.fillText(`${coilDiameter}mm`, 80, height - 5);
    }
}

// Initialize RF calculator
let rfCalculator = null;

function initRFCalculator() {
    rfCalculator = new RFCalculator();
    updateRFCalculations();
}

function updateRFCalculations() {
    if (!rfCalculator) return;

    // Larmor frequency
    const fieldStrength = parseFloat(document.getElementById('calcFieldStrength')?.value || 3);
    const nucleus = document.getElementById('nucleus')?.value || '1H';
    const larmorFreq = rfCalculator.calculateLarmorFrequency(fieldStrength, nucleus);
    const larmorElem = document.getElementById('calcLarmorFreq');
    if (larmorElem) larmorElem.textContent = `${larmorFreq} MHz`;

    // Loop inductance
    const loopDiameter = parseFloat(document.getElementById('calcLoopDiameter')?.value || 80);
    const wireDiameter = parseFloat(document.getElementById('calcWireDiameter')?.value || 1.3);
    const inductance = rfCalculator.calculateLoopInductance(loopDiameter, wireDiameter);
    const inductanceElem = document.getElementById('calcInductance');
    if (inductanceElem) inductanceElem.textContent = `${inductance} nH`;

    // Resonance capacitance
    const resInductance = parseFloat(document.getElementById('resInductance')?.value || 180);
    const targetFreq = parseFloat(document.getElementById('targetFreq')?.value || 127.74);
    const capacitance = rfCalculator.calculateResonanceCapacitance(resInductance, targetFreq);
    const capacitanceElem = document.getElementById('calcCapacitance');
    if (capacitanceElem) capacitanceElem.textContent = `${capacitance} pF`;

    // SNR estimation
    const snrCoilDiameter = parseFloat(document.getElementById('snrCoilDiameter')?.value || 80);
    const snrDepth = parseFloat(document.getElementById('snrDepth')?.value || 50);
    const snr = rfCalculator.calculateRelativeSNR(snrCoilDiameter, snrDepth);
    const snrElem = document.getElementById('calcSNR');
    if (snrElem) snrElem.textContent = snr;

    // Draw field map
    const fieldCanvas = document.getElementById('fieldCanvas');
    if (fieldCanvas) {
        rfCalculator.drawFieldMap(fieldCanvas, loopDiameter);
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { RFCalculator };
}
