// LLM-powered Design Assistant for MRI Coils
class LLMAssistant {
    constructor() {
        this.designDatabase = this.buildDesignDatabase();
    }

    // Build knowledge base of design patterns
    buildDesignDatabase() {
        return {
            fieldStrengths: {
                1.5: {
                    frequency: 63.87,
                    applications: 'Clinical MRI, whole-body imaging',
                    channelCount: '8-16 channels typical',
                    challenges: 'Lower SNR, longer scan times'
                },
                3: {
                    frequency: 127.74,
                    applications: 'High-resolution clinical and research',
                    channelCount: '16-32 channels typical',
                    challenges: 'B1 inhomogeneity, SAR concerns'
                },
                7: {
                    frequency: 298.06,
                    applications: 'Ultra-high field research, neuroscience',
                    channelCount: '32-64 channels recommended',
                    challenges: 'Severe B1 inhomogeneity, high SAR, RF wavelength effects'
                },
                9.4: {
                    frequency: 400.13,
                    applications: 'Preclinical, small animal imaging',
                    channelCount: '16-32 channels',
                    challenges: 'Extreme B1 inhomogeneity, wavelength < head size'
                }
            },
            anatomy: {
                'brain': {
                    diameter: 200,
                    recommendedChannels: 32,
                    geodesicFreq: 2,
                    priority: 'Whole-brain coverage, temporal lobe sensitivity'
                },
                'brain-pediatric': {
                    diameter: 160,
                    recommendedChannels: 24,
                    geodesicFreq: 2,
                    priority: 'Smaller form factor, comfort'
                },
                'head-neck': {
                    diameter: 220,
                    recommendedChannels: 48,
                    geodesicFreq: 3,
                    priority: 'Extended FOV, anterior neck coverage'
                },
                'knee': {
                    diameter: 180,
                    recommendedChannels: 24,
                    geodesicFreq: 2,
                    priority: 'High SNR, cartilage visualization, deep penetration'
                },
                'wrist': {
                    diameter: 100,
                    recommendedChannels: 16,
                    geodesicFreq: 2,
                    priority: 'Ultra-high resolution, small FOV'
                },
                'ankle': {
                    diameter: 140,
                    recommendedChannels: 16,
                    geodesicFreq: 2,
                    priority: 'Ligament visualization, uniform coverage'
                },
                'shoulder': {
                    diameter: 200,
                    recommendedChannels: 16,
                    geodesicFreq: 2,
                    priority: 'Deep penetration, rotator cuff visualization'
                },
                'custom': {
                    diameter: 180,
                    recommendedChannels: 32,
                    geodesicFreq: 2,
                    priority: 'Application-specific optimization'
                }
            },
            designGoals: {
                'snr': {
                    strategy: 'Maximize coil density, optimize element size',
                    recommendation: 'Use smaller elements, increase channel count'
                },
                'coverage': {
                    strategy: 'Uniform spatial distribution',
                    recommendation: 'Higher geodesic frequency, ensure peripheral coverage'
                },
                'balanced': {
                    strategy: 'Balance SNR and parallel imaging performance',
                    recommendation: 'Moderate element size, geodesic distribution'
                },
                'parallel': {
                    strategy: 'Maximize spatial encoding capability',
                    recommendation: 'High channel count, diverse coil orientations'
                }
            }
        };
    }

    // Generate design recommendations
    generateRecommendations(config) {
        const {
            fieldStrength = 3,
            targetAnatomy = 'brain',
            designGoal = 'balanced'
        } = config;

        const fieldData = this.designDatabase.fieldStrengths[fieldStrength];
        const anatomyData = this.designDatabase.anatomy[targetAnatomy];
        const goalData = this.designDatabase.designGoals[designGoal];

        // Calculate optimal parameters
        const larmorFreq = fieldData.frequency;
        const channelCount = anatomyData.recommendedChannels;
        const geodesicFreq = anatomyData.geodesicFreq;
        const coilDiameter = anatomyData.diameter;

        // Adjust based on design goal
        let adjustedChannels = channelCount;
        let adjustedGeodesic = geodesicFreq;

        if (designGoal === 'snr') {
            adjustedChannels = Math.round(channelCount * 0.8);
        } else if (designGoal === 'parallel') {
            adjustedChannels = Math.round(channelCount * 1.5);
            adjustedGeodesic = Math.min(4, geodesicFreq + 1);
        } else if (designGoal === 'coverage') {
            adjustedGeodesic = Math.min(4, geodesicFreq + 1);
        }

        // Calculate component values
        const loopDiameter = Math.round(coilDiameter / Math.sqrt(adjustedChannels) * 2.5);
        const inductance = this.estimateInductance(loopDiameter);
        const tuningCap = this.calculateCapacitance(inductance, larmorFreq);
        const matchingCap = tuningCap * 2.5;

        // Generate recommendations
        const recommendations = {
            summary: this.generateSummary(fieldStrength, targetAnatomy, designGoal),
            parameters: {
                geodesicFrequency: adjustedGeodesic,
                channelCount: adjustedChannels,
                coilRadius: coilDiameter / 2,
                loopDiameter: loopDiameter,
                larmorFrequency: larmorFreq
            },
            components: {
                loopInductance: inductance,
                tuningCapacitor: tuningCap,
                matchingCapacitor: matchingCap,
                wireGauge: 16
            },
            performance: {
                estimatedQFactor: this.estimateQFactor(loopDiameter, larmorFreq),
                expectedSNR: this.estimateSNR(adjustedChannels, loopDiameter),
                parallelImagingCapability: this.estimateParallelImaging(adjustedChannels, adjustedGeodesic)
            },
            designConsiderations: this.generateConsiderations(fieldStrength, designGoal, targetAnatomy),
            buildingTips: this.generateBuildingTips(fieldStrength)
        };

        return recommendations;
    }

    // Generate summary text
    generateSummary(fieldStrength, anatomy, goal) {
        const fieldData = this.designDatabase.fieldStrengths[fieldStrength];
        const anatomyData = this.designDatabase.anatomy[anatomy];
        const goalData = this.designDatabase.designGoals[goal];

        return `
### Design Overview

**Target System:** ${fieldStrength}T MRI System (${fieldData.frequency} MHz)
**Application:** ${fieldData.applications}
**Target Anatomy:** ${anatomyData.priority}
**Design Goal:** ${goalData.strategy}

${goalData.recommendation}

---

### Key Features

- **Geodesic Array Architecture:** Optimized vertex distribution for uniform coverage
- **Phased Array Design:** Multiple independent receive channels
- **Decoupling Strategy:** Geometric overlap + preamplifier decoupling
- **Matching Network:** Individual element matching to 50Ω
        `.trim();
    }

    // Generate design considerations
    generateConsiderations(fieldStrength, goal, anatomy) {
        const considerations = [];

        // Anatomy-specific considerations
        if (['knee', 'ankle'].includes(anatomy)) {
            considerations.push({
                title: 'Mechanical Design',
                description: 'Consider a split-coil or transmit/receive design to allow for easy patient positioning and varying limb sizes.',
                severity: 'important'
            });
        }

        if (anatomy === 'wrist') {
            considerations.push({
                title: 'Form Factor',
                description: 'Ensure the coil housing is compact and comfortable for the patient, possibly allowing for arm resting at the side.',
                severity: 'important'
            });
        }

        if (anatomy === 'shoulder') {
            considerations.push({
                title: 'Flexibility',
                description: 'Shoulder anatomy varies significantly. A semi-flexible or form-fitting design is recommended for optimal coupling.',
                severity: 'important'
            });
        }

        if (fieldStrength >= 7) {
            considerations.push({
                title: 'B1 Inhomogeneity',
                description: 'At ultra-high fields, RF wavelength approaches head dimensions. Use B1+ shimming or parallel transmission.',
                severity: 'critical'
            });
            considerations.push({
                title: 'SAR Management',
                description: 'Specific Absorption Rate increases quadratically with field strength. Monitor local and global SAR.',
                severity: 'critical'
            });
        }

        if (fieldStrength >= 3) {
            considerations.push({
                title: 'Decoupling Requirements',
                description: 'Adjacent coil elements must be well-decoupled. Target > 15 dB isolation.',
                severity: 'important'
            });
        }

        considerations.push({
            title: 'Element Size Optimization',
            description: 'Element size affects penetration depth vs. SNR trade-off. Smaller elements = better surface SNR.',
            severity: 'important'
        });

        if (goal === 'parallel') {
            considerations.push({
                title: 'g-Factor Optimization',
                description: 'Maximize spatial diversity to minimize g-factor noise amplification in parallel imaging.',
                severity: 'important'
            });
        }

        return considerations;
    }

    // Generate building tips
    generateBuildingTips(fieldStrength) {
        return [
            'Use high-Q non-magnetic capacitors (porcelain or PTFE dielectric)',
            'Minimize conductor length between loop and capacitors',
            'Ensure symmetrical capacitor placement for balanced tuning',
            'Test each element individually before array assembly',
            'Use vector network analyzer for tuning and matching',
            'Apply geometric overlap carefully - small changes affect coupling significantly',
            fieldStrength >= 7 ? 'Consider transmission line effects at UHF frequencies' : null,
            'Build test phantom for SNR and g-factor measurements'
        ].filter(Boolean);
    }

    // Helper: estimate inductance
    estimateInductance(diameter) {
        return Math.round(0.002 * diameter * diameter * 0.5);
    }

    // Helper: calculate capacitance
    calculateCapacitance(inductance, frequency) {
        const L = inductance * 1e-9;
        const f = frequency * 1e6;
        const C = 1 / (4 * Math.PI * Math.PI * f * f * L);
        return (C * 1e12).toFixed(1);
    }

    // Helper: estimate Q-factor
    estimateQFactor(diameter, frequency) {
        return Math.round(250 * (100 / diameter) * (128 / frequency));
    }

    // Helper: estimate SNR
    estimateSNR(channels, diameter) {
        const baseline = Math.sqrt(channels) * (diameter / 80);
        return baseline.toFixed(1);
    }

    // Helper: estimate parallel imaging capability
    estimateParallelImaging(channels, geodesicFreq) {
        const maxR = Math.min(channels / 4, geodesicFreq * 2);
        return `R=${Math.floor(maxR)} (${channels} channels, geodesic freq ${geodesicFreq})`;
    }

    // Format recommendations as HTML
    formatRecommendationsHTML(recommendations) {
        const { summary, parameters, components, performance, designConsiderations, buildingTips } = recommendations;

        let html = `
            <div class="recommendation-section">
                <div style="white-space: pre-line; line-height: 1.8;">${summary}</div>
            </div>
            
            <div class="recommendation-section">
                <h3 style="color: #6366f1; margin-bottom: 1rem;">Recommended Parameters</h3>
                <div class="param-grid">
                    <div class="param-item">
                        <span class="param-label">Geodesic Frequency:</span>
                        <span class="param-value">${parameters.geodesicFrequency}</span>
                    </div>
                    <div class="param-item">
                        <span class="param-label">Channel Count:</span>
                        <span class="param-value">${parameters.channelCount}</span>
                    </div>
                    <div class="param-item">
                        <span class="param-label">Coil Radius:</span>
                        <span class="param-value">${parameters.coilRadius} mm</span>
                    </div>
                    <div class="param-item">
                        <span class="param-label">Element Diameter:</span>
                        <span class="param-value">${parameters.loopDiameter} mm</span>
                    </div>
                    <div class="param-item">
                        <span class="param-label">Larmor Frequency:</span>
                        <span class="param-value">${parameters.larmorFrequency} MHz</span>
                    </div>
                </div>
            </div>
            
            <div class="recommendation-section">
                <h3 style="color: #6366f1; margin-bottom: 1rem;">Component Values</h3>
                <div class="param-grid">
                    <div class="param-item">
                        <span class="param-label">Loop Inductance:</span>
                        <span class="param-value">${components.loopInductance} nH</span>
                    </div>
                    <div class="param-item">
                        <span class="param-label">Tuning Capacitor:</span>
                        <span class="param-value">${components.tuningCapacitor} pF</span>
                    </div>
                    <div class="param-item">
                        <span class="param-label">Matching Capacitor:</span>
                        <span class="param-value">${components.matchingCapacitor} pF</span>
                    </div>
                    <div class="param-item">
                        <span class="param-label">Wire Gauge:</span>
                        <span class="param-value">${components.wireGauge} AWG</span>
                    </div>
                </div>
            </div>
            
            <div class="recommendation-section">
                <h3 style="color: #10b981; margin-bottom: 1rem;">Expected Performance</h3>
                <div class="param-grid">
                    <div class="param-item">
                        <span class="param-label">Estimated Q-Factor:</span>
                        <span class="param-value">${performance.estimatedQFactor}</span>
                    </div>
                    <div class="param-item">
                        <span class="param-label">Relative SNR:</span>
                        <span class="param-value">${performance.expectedSNR}×</span>
                    </div>
                    <div class="param-item">
                        <span class="param-label">Parallel Imaging:</span>
                        <span class="param-value">${performance.parallelImagingCapability}</span>
                    </div>
                </div>
            </div>
            
            <div class="recommendation-section">
                <h3 style="color: #f59e0b; margin-bottom: 1rem;">Design Considerations</h3>
                ${designConsiderations.map(c => `
                    <div class="consideration-item ${c.severity}">
                        <strong>${c.title}:</strong> ${c.description}
                    </div>
                `).join('')}
            </div>
            
            <div class="recommendation-section">
                <h3 style="color: #6366f1; margin-bottom: 1rem;">Building Tips</h3>
                <ul style="margin-left: 1.5rem;">
                    ${buildingTips.map(tip => `<li style="margin-bottom: 0.5rem;">${tip}</li>`).join('')}
                </ul>
            </div>
        `;

        // Add CSS for formatting
        const style = `
            <style>
                .recommendation-section {
                    margin-bottom: 2rem;
                    padding-bottom: 1rem;
                    border-bottom: 1px solid rgba(148, 163, 184, 0.1);
                }
                .param-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 0.75rem;
                }
                .param-item {
                    background: rgba(99, 102, 241, 0.1);
                    padding: 0.75rem;
                    border-radius: 0.375rem;
                    display: flex;
                    justify-content: space-between;
                }
                .param-label {
                    color: #cbd5e1;
                }
                .param-value {
                    color: #6366f1;
                    font-weight: 600;
                    font-family: 'JetBrains Mono', monospace;
                }
                .consideration-item {
                    padding: 0.75rem;
                    margin-bottom: 0.5rem;
                    border-radius: 0.375rem;
                    border-left: 3px solid;
                }
                .consideration-item.critical {
                    background: rgba(239, 68, 68, 0.1);
                    border-color: #ef4444;
                }
                .consideration-item.important {
                    background: rgba(245, 158, 11, 0.1);
                    border-color: #f59e0b;
                }
            </style>
        `;

        return style + html;
    }
}

// Initialize LLM assistant
let llmAssistant = null;

function initLLMAssistant() {
    llmAssistant = new LLMAssistant();
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LLMAssistant };
}
