/**
 * Coil Generator and Physics Engine
 * Generates knee coil geometries and simulates B1 fields using Biot-Savart law
 */

class CoilGenerator {
    constructor() {
        this.coilElements = [];
        this.currentConfig = null;
    }

    /**
     * Generate phased array coil configuration
     */
    generateCoil(params) {
        const {
            numChannels = 8,
            radius = 60,        // mm
            gap = 10,           // mm between elements
            turns = 3,          // number of turns per element
            current = 1.0,      // Amperes
            targetHomogeneity = 0.85,
            targetSNR = 50,
            targetCoupling = 0.1
        } = params;

        this.currentConfig = params;
        this.coilElements = [];

        // Calculate element positions around a cylinder (knee geometry)
        const angleStep = (2 * Math.PI) / numChannels;
        const elementWidth = (2 * Math.PI * radius / numChannels) - gap;
        const elementHeight = 150; // mm (covers knee length)

        for (let i = 0; i < numChannels; i++) {
            const angle = i * angleStep;
            const centerX = radius * Math.cos(angle);
            const centerY = radius * Math.sin(angle);

            const element = {
                id: i,
                type: 'rectangular_loop',
                position: { x: centerX, y: centerY, z: 0 },
                rotation: { x: 0, y: 0, z: angle },
                dimensions: {
                    width: elementWidth,
                    height: elementHeight,
                    turns: turns
                },
                current: current,
                wireLoops: this.generateWireLoops(
                    { x: centerX, y: centerY, z: 0 },
                    angle,
                    elementWidth,
                    elementHeight,
                    turns,
                    radius
                )
            };

            this.coilElements.push(element);
        }

        return this.coilElements;
    }

    /**
     * Generate individual wire loops for an element
     */
    generateWireLoops(position, rotation, width, height, turns, radius) {
        const loops = [];
        const turnSpacing = 2; // mm between turns

        for (let t = 0; t < turns; t++) {
            const offset = (t - (turns - 1) / 2) * turnSpacing;
            const loop = [];
            const segments = 20; // Number of segments per loop

            // Create rectangular loop
            for (let s = 0; s <= segments; s++) {
                const u = s / segments;
                let localX, localY, localZ;

                if (u < 0.25) {
                    // Bottom edge
                    const t = u / 0.25;
                    localX = -width / 2 + t * width;
                    localY = offset;
                    localZ = -height / 2;
                } else if (u < 0.5) {
                    // Right edge
                    const t = (u - 0.25) / 0.25;
                    localX = width / 2;
                    localY = offset;
                    localZ = -height / 2 + t * height;
                } else if (u < 0.75) {
                    // Top edge
                    const t = (u - 0.5) / 0.25;
                    localX = width / 2 - t * width;
                    localY = offset;
                    localZ = height / 2;
                } else {
                    // Left edge
                    const t = (u - 0.75) / 0.25;
                    localX = -width / 2;
                    localY = offset;
                    localZ = height / 2 - t * height;
                }

                // Rotate and translate to world coordinates
                const cosR = Math.cos(rotation);
                const sinR = Math.sin(rotation);

                // Apply radial offset for cylindrical arrangement
                const radialOffset = radius + localY;
                const worldX = radialOffset * Math.cos(rotation) + localZ * sinR;
                const worldY = radialOffset * Math.sin(rotation) - localZ * cosR;
                const worldZ = localX;

                loop.push({
                    x: worldX + position.x,
                    y: worldY + position.y,
                    z: worldZ + position.z
                });
            }

            loops.push(loop);
        }

        return loops;
    }

    /**
     * Calculate B1 field using Biot-Savart law
     */
    calculateB1Field(point, element) {
        const mu0 = 4 * Math.PI * 1e-7; // Permeability of free space
        let Bx = 0, By = 0, Bz = 0;

        for (const loop of element.wireLoops) {
            for (let i = 0; i < loop.length - 1; i++) {
                const r1 = loop[i];
                const r2 = loop[i + 1];

                // Current element vector (dl)
                const dl = {
                    x: r2.x - r1.x,
                    y: r2.y - r1.y,
                    z: r2.z - r1.z
                };

                // Midpoint of segment
                const mid = {
                    x: (r1.x + r2.x) / 2,
                    y: (r1.y + r2.y) / 2,
                    z: (r1.z + r2.z) / 2
                };

                // Vector from wire element to field point (R)
                const R = {
                    x: point.x - mid.x,
                    y: point.y - mid.y,
                    z: point.z - mid.z
                };

                const r = Math.sqrt(R.x * R.x + R.y * R.y + R.z * R.z);

                if (r < 1e-6) continue; // Avoid singularity

                // Cross product: dl × R
                const cross = {
                    x: dl.y * R.z - dl.z * R.y,
                    y: dl.z * R.x - dl.x * R.z,
                    z: dl.x * R.y - dl.y * R.x
                };

                // Biot-Savart contribution: dB = (μ₀/4π) * I * (dl × R) / r³
                const factor = (mu0 / (4 * Math.PI)) * element.current / Math.pow(r, 3);

                Bx += factor * cross.x;
                By += factor * cross.y;
                Bz += factor * cross.z;
            }
        }

        return { x: Bx, y: By, z: Bz };
    }

    /**
     * Calculate total B1 field from all coil elements
     */
    calculateTotalB1Field(point) {
        let Bx = 0, By = 0, Bz = 0;

        for (const element of this.coilElements) {
            const B = this.calculateB1Field(point, element);
            Bx += B.x;
            By += B.y;
            Bz += B.z;
        }

        return { x: Bx, y: By, z: Bz };
    }

    /**
     * Generate B1 field map over a region
     */
    generateFieldMap(resolution = 20) {
        const fieldMap = [];
        const range = 80; // mm
        const step = (2 * range) / resolution;

        for (let ix = 0; ix < resolution; ix++) {
            for (let iy = 0; iy < resolution; iy++) {
                for (let iz = 0; iz < resolution; iz++) {
                    const point = {
                        x: -range + ix * step,
                        y: -range + iy * step,
                        z: -range + iz * step
                    };

                    const B = this.calculateTotalB1Field(point);
                    const magnitude = Math.sqrt(B.x * B.x + B.y * B.y + B.z * B.z);

                    fieldMap.push({
                        position: point,
                        field: B,
                        magnitude: magnitude
                    });
                }
            }
        }

        return fieldMap;
    }

    /**
     * Calculate field homogeneity
     */
    calculateHomogeneity(fieldMap) {
        if (fieldMap.length === 0) return 0;

        const magnitudes = fieldMap.map(f => f.magnitude);
        const mean = magnitudes.reduce((sum, m) => sum + m, 0) / magnitudes.length;
        const variance = magnitudes.reduce((sum, m) => sum + Math.pow(m - mean, 2), 0) / magnitudes.length;
        const stdDev = Math.sqrt(variance);

        // Homogeneity as inverse of coefficient of variation
        const cv = mean > 0 ? stdDev / mean : 1;
        return Math.max(0, 1 - cv);
    }

    /**
     * Estimate SNR based on coil geometry
     */
    calculateSNR(params) {
        // Simplified SNR model: SNR ∝ √(N_channels) * B1_strength * Q / noise
        const numChannels = params.numChannels;
        const turns = params.turns;
        const current = params.current;

        // Sample B1 at center
        const centerField = this.calculateTotalB1Field({ x: 0, y: 0, z: 0 });
        const B1_strength = Math.sqrt(centerField.x ** 2 + centerField.y ** 2 + centerField.z ** 2);

        // Quality factor (simplified)
        const Q = 100 * turns;

        // Noise (simplified thermal noise model)
        const noise = Math.sqrt(params.radius / 10);

        // SNR formula
        const snr = Math.sqrt(numChannels) * B1_strength * 1e8 * Q / noise;

        return Math.min(100, Math.max(0, snr));
    }

    /**
     * Calculate mutual coupling between coil elements
     */
    calculateCoupling() {
        if (this.coilElements.length < 2) return 0;

        let totalCoupling = 0;
        let pairCount = 0;

        for (let i = 0; i < this.coilElements.length; i++) {
            for (let j = i + 1; j < this.coilElements.length; j++) {
                const elem1 = this.coilElements[i];
                const elem2 = this.coilElements[j];

                // Calculate mutual inductance based on proximity
                const dx = elem1.position.x - elem2.position.x;
                const dy = elem1.position.y - elem2.position.y;
                const dz = elem1.position.z - elem2.position.z;
                const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

                // Coupling decreases with distance
                const coupling = Math.exp(-distance / 50);
                totalCoupling += coupling;
                pairCount++;
            }
        }

        return pairCount > 0 ? totalCoupling / pairCount : 0;
    }

    /**
     * Evaluate coil performance metrics
     */
    evaluateCoil(params) {
        // Generate coil with given parameters
        this.generateCoil(params);

        // Calculate field map (low resolution for speed)
        const fieldMap = this.generateFieldMap(10);

        // Calculate metrics
        const homogeneity = this.calculateHomogeneity(fieldMap);
        const snr = this.calculateSNR(params);
        const coupling = this.calculateCoupling();

        return {
            homogeneity: homogeneity,
            snr: snr,
            coupling: coupling,
            fieldMap: fieldMap
        };
    }

    /**
     * Get coil specifications for export
     */
    getSpecifications() {
        if (!this.currentConfig) return null;

        return {
            configuration: this.currentConfig,
            elements: this.coilElements.map(elem => ({
                id: elem.id,
                position: elem.position,
                rotation: elem.rotation,
                dimensions: elem.dimensions,
                current: elem.current
            })),
            totalElements: this.coilElements.length,
            generatedAt: new Date().toISOString()
        };
    }
}
