/**
 * Variational Measure Theory Engine for Orthopedic Coil Generation
 * Implements generative AI for knee MRI coil design optimization
 */

class VariationalMeasureTheory {
    constructor() {
        this.constants = {
            mu0: 4 * Math.PI * 1e-7, // Permeability of free space
            epsilon0: 8.854187817e-12, // Permittivity of free space
            gamma: 42.58e6, // Gyromagnetic ratio for 1H (MHz/T)
            B0: 3.0 // Field strength in Tesla
        };

        this.coilTemplates = {
            knee: this.initializeKneeTemplate()
        };
    }

    /**
     * Initialize knee coil template geometry
     */
    initializeKneeTemplate() {
        return {
            name: 'Knee Coil Array',
            type: 'phased_array',
            anatomy: 'knee',
            num_channels: 8,
            geometry: {
                // Cylindrical approximation for knee
                radius: 0.08, // 8cm radius
                length: 0.25, // 25cm length
                curvature: 0.15 // Slight curvature
            },
            coil_elements: []
        };
    }

    /**
     * Variational measure for optimal coil placement
     * Uses calculus of variations to minimize energy functional
     */
    calculateVariationalMeasure(positions, parameters) {
        let totalEnergy = 0;

        // Energy functional E[x] = ∫(L(x, x', t))dt
        // L = kinetic energy - potential energy + coupling terms

        for (let i = 0; i < positions.length; i++) {
            // Self energy (individual coil efficiency)
            const selfEnergy = this.calculateSelfEnergy(positions[i], parameters);

            // Coupling energy (mutual inductance between coils)
            let couplingEnergy = 0;
            for (let j = 0; j < positions.length; j++) {
                if (i !== j) {
                    couplingEnergy += this.calculateCouplingEnergy(positions[i], positions[j], parameters);
                }
            }

            // Coverage energy (how well coil covers anatomy)
            const coverageEnergy = this.calculateCoverageEnergy(positions[i], parameters);

            // SNR optimization term
            const snrPenalty = this.calculateSNRPenalty(positions[i], parameters);

            totalEnergy += selfEnergy + 0.3 * couplingEnergy - 0.5 * coverageEnergy + 0.2 * snrPenalty;
        }

        return totalEnergy;
    }

    /**
     * Calculate self energy of individual coil element
     */
    calculateSelfEnergy(position, params) {
        // Self inductance: L = μ₀ * N² * A / l
        const N = params.turns || 1; // Number of turns
        const A = params.area || 0.01; // Coil area in m²
        const l = params.length || 0.05; // Coil length

        const L = this.constants.mu0 * N * N * A / l;

        // Resistance (simplified)
        const R = params.resistance || 1.0;

        // Q-factor penalty
        const omega = 2 * Math.PI * this.constants.gamma * this.constants.B0;
        const Q = omega * L / R;

        return 1.0 / (1.0 + Q); // Lower energy for higher Q
    }

    /**
     * Calculate coupling energy between two coil elements
     */
    calculateCouplingEnergy(pos1, pos2, params) {
        // Distance between coil centers
        const dx = pos1.x - pos2.x;
        const dy = pos1.y - pos2.y;
        const dz = pos1.z - pos2.z;
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

        // Mutual inductance (simplified dipole approximation)
        const M = this.constants.mu0 * (params.area || 0.01) / (4 * Math.PI * Math.pow(distance, 3));

        // Coupling coefficient
        const k = M / Math.sqrt(this.calculateSelfInductance(params) * this.calculateSelfInductance(params));

        // Penalize high coupling (want minimal overlap)
        return k * k;
    }

    /**
     * Calculate coverage energy (reward good anatomical coverage)
     */
    calculateCoverageEnergy(position, params) {
        // Distance from optimal coverage points
        const targetPoints = params.targetAnatomyPoints || this.generateKneeAnatomyPoints();

        let coverageScore = 0;
        for (const target of targetPoints) {
            const dist = this.distanceToPoint(position, target);
            // Gaussian weighting for coverage
            coverageScore += Math.exp(-dist * dist / (2 * 0.05 * 0.05));
        }

        return coverageScore / targetPoints.length;
    }

    /**
     * Calculate SNR penalty
     */
    calculateSNRPenalty(position, params) {
        // Simplified SNR calculation
        // SNR ∝ √(B₁²/R)
        const B1 = this.calculateB1Field(position, params);
        const R = params.resistance || 1.0;
        const SNR = Math.sqrt(B1 * B1 / R);

        // Return penalty (want to maximize SNR, so minimize penalty)
        return 1.0 / (1.0 + SNR);
    }

    /**
     * Calculate B1 field strength at position
     */
    calculateB1Field(position, params) {
        // Biot-Savart law simplified
        // B₁ = (μ₀ * I * A) / (2 * d³)
        const current = params.current || 1.0;
        const area = params.area || 0.01;
        const distance = Math.sqrt(position.x * position.x + position.y * position.y + position.z * position.z) + 0.01;

        return (this.constants.mu0 * current * area) / (2 * Math.pow(distance, 3));
    }

    /**
     * Generate optimal coil geometry using gradient descent on variational measure
     */
    generateOptimalCoilGeometry(template, iterations = 1000) {
        // Initialize coil positions (8-channel array for knee)
        let positions = this.initializeCoilPositions(template);

        const learningRate = 0.001;
        const params = {
            turns: 1,
            area: 0.008, // 80 cm²
            length: 0.05,
            resistance: 1.2,
            current: 1.0,
            targetAnatomyPoints: this.generateKneeAnatomyPoints()
        };

        // Gradient descent optimization
        for (let iter = 0; iter < iterations; iter++) {
            const currentEnergy = this.calculateVariationalMeasure(positions, params);

            // Calculate gradients
            const gradients = positions.map((pos, i) => {
                const epsilon = 0.0001;
                const grad = { x: 0, y: 0, z: 0 };

                // Numerical gradient in x
                positions[i].x += epsilon;
                const ex1 = this.calculateVariationalMeasure(positions, params);
                positions[i].x -= 2 * epsilon;
                const ex2 = this.calculateVariationalMeasure(positions, params);
                positions[i].x += epsilon;
                grad.x = (ex1 - ex2) / (2 * epsilon);

                // Numerical gradient in y
                positions[i].y += epsilon;
                const ey1 = this.calculateVariationalMeasure(positions, params);
                positions[i].y -= 2 * epsilon;
                const ey2 = this.calculateVariationalMeasure(positions, params);
                positions[i].y += epsilon;
                grad.y = (ey1 - ey2) / (2 * epsilon);

                // Numerical gradient in z
                positions[i].z += epsilon;
                const ez1 = this.calculateVariationalMeasure(positions, params);
                positions[i].z -= 2 * epsilon;
                const ez2 = this.calculateVariationalMeasure(positions, params);
                positions[i].z += epsilon;
                grad.z = (ez1 - ez2) / (2 * epsilon);

                return grad;
            });

            // Update positions
            positions = positions.map((pos, i) => ({
                x: pos.x - learningRate * gradients[i].x,
                y: pos.y - learningRate * gradients[i].y,
                z: pos.z - learningRate * gradients[i].z
            }));

            // Constrain positions to anatomical bounds
            positions = positions.map(pos => this.constrainToAnatomy(pos, template));
        }

        return {
            positions,
            finalEnergy: this.calculateVariationalMeasure(positions, params),
            parameters: params
        };
    }

    /**
     * Initialize coil element positions
     */
    initializeCoilPositions(template) {
        const numChannels = template.num_channels;
        const radius = template.geometry.radius;
        const length = template.geometry.length;

        const positions = [];

        // Arrange coils in cylindrical array around knee
        for (let i = 0; i < numChannels; i++) {
            const angle = (2 * Math.PI * i) / numChannels;
            const z = (i % 2) * length / 4 - length / 8; // Stagger in z-direction

            positions.push({
                x: radius * Math.cos(angle),
                y: radius * Math.sin(angle),
                z: z
            });
        }

        return positions;
    }

    /**
     * Generate target anatomy points for knee
     */
    generateKneeAnatomyPoints() {
        const points = [];

        // Patella, femur, tibia, and soft tissue regions
        const regions = [
            { center: [0, 0.04, 0], radius: 0.03, density: 5 },      // Patella
            { center: [0, 0, 0.08], radius: 0.04, density: 8 },      // Femur
            { center: [0, 0, -0.08], radius: 0.038, density: 8 },    // Tibia
            { center: [0.02, 0, 0], radius: 0.025, density: 4 },     // Lateral meniscus
            { center: [-0.02, 0, 0], radius: 0.025, density: 4 }     // Medial meniscus
        ];

        for (const region of regions) {
            for (let i = 0; i < region.density; i++) {
                const theta = Math.random() * 2 * Math.PI;
                const phi = Math.random() * Math.PI;
                const r = Math.random() * region.radius;

                points.push({
                    x: region.center[0] + r * Math.sin(phi) * Math.cos(theta),
                    y: region.center[1] + r * Math.sin(phi) * Math.sin(theta),
                    z: region.center[2] + r * Math.cos(phi)
                });
            }
        }

        return points;
    }

    /**
     * Constrain positions to anatomical bounds
     */
    constrainToAnatomy(position, template) {
        const maxRadius = template.geometry.radius * 1.2;
        const maxZ = template.geometry.length / 2;

        const radius2d = Math.sqrt(position.x * position.x + position.y * position.y);
        if (radius2d > maxRadius) {
            const scale = maxRadius / radius2d;
            position.x *= scale;
            position.y *= scale;
        }

        position.z = Math.max(-maxZ, Math.min(maxZ, position.z));

        return position;
    }

    /**
     * Calculate self inductance
     */
    calculateSelfInductance(params) {
        const N = params.turns || 1;
        const A = params.area || 0.01;
        const l = params.length || 0.05;
        return this.constants.mu0 * N * N * A / l;
    }

    /**
     * Distance between two points
     */
    distanceToPoint(p1, p2) {
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        const dz = p1.z - p2.z;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /**
     * Calculate B1 field map over volume
     */
    calculateB1FieldMap(coilPositions, params, resolution = 20) {
        const fieldMap = [];
        const range = 0.15; // 15cm range
        const step = (2 * range) / resolution;

        for (let ix = 0; ix < resolution; ix++) {
            for (let iy = 0; iy < resolution; iy++) {
                for (let iz = 0; iz < resolution; iz++) {
                    const point = {
                        x: -range + ix * step,
                        y: -range + iy * step,
                        z: -range + iz * step
                    };

                    // Sum contributions from all coil elements
                    let totalB1 = 0;
                    for (const coilPos of coilPositions) {
                        const relPos = {
                            x: point.x - coilPos.x,
                            y: point.y - coilPos.y,
                            z: point.z - coilPos.z
                        };
                        totalB1 += this.calculateB1Field(relPos, params);
                    }

                    fieldMap.push({
                        position: point,
                        B1: totalB1
                    });
                }
            }
        }

        return fieldMap;
    }

    /**
     * Calculate SNR distribution over volume
     */
    calculateSNRMap(coilPositions, params, resolution = 20) {
        const fieldMap = this.calculateB1FieldMap(coilPositions, params, resolution);

        return fieldMap.map(point => {
            const B1 = point.B1;
            const noise = Math.sqrt(params.resistance || 1.0);
            const SNR = B1 / noise;

            return {
                position: point.position,
                SNR: SNR
            };
        });
    }

    /**
     * Generate coil performance metrics
     */
    generatePerformanceMetrics(coilGeometry) {
        const positions = coilGeometry.positions;
        const params = coilGeometry.parameters;

        // Calculate average coupling
        let totalCoupling = 0;
        let couplingCount = 0;
        for (let i = 0; i < positions.length; i++) {
            for (let j = i + 1; j < positions.length; j++) {
                totalCoupling += this.calculateCouplingEnergy(positions[i], positions[j], params);
                couplingCount++;
            }
        }
        const avgCoupling = totalCoupling / couplingCount;

        // Calculate coverage
        const targetPoints = this.generateKneeAnatomyPoints();
        let coverageScore = 0;
        for (const pos of positions) {
            coverageScore += this.calculateCoverageEnergy(pos, { ...params, targetAnatomyPoints: targetPoints });
        }
        const avgCoverage = coverageScore / positions.length;

        // Calculate average SNR
        const snrMap = this.calculateSNRMap(positions, params, 15);
        const avgSNR = snrMap.reduce((sum, point) => sum + point.SNR, 0) / snrMap.length;
        const maxSNR = Math.max(...snrMap.map(p => p.SNR));

        return {
            coupling: avgCoupling,
            coverage: avgCoverage,
            avgSNR: avgSNR,
            maxSNR: maxSNR,
            efficiency: avgCoverage / (1 + avgCoupling),
            qualityFactor: maxSNR / avgSNR
        };
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VariationalMeasureTheory;
}
