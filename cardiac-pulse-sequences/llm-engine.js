// LLM Engine for Protocol Generation and Optimization

const LLMEngine = {
    /**
     * Protocol knowledge base
     */
    protocolTemplates: {
        myocarditis: {
            description: 'Comprehensive myocarditis protocol',
            sequences: ['cine', 't2-mapping', 't1-mapping', 'lge'],
            parameters: {
                accelerationFactor: 2.0,
                technique: 'sense',
                cinePhases: 25,
                temporalRes: 40
            },
            rationale: 'T2 mapping for edema, LGE for fibrosis, T1 mapping for ECV quantification'
        },
        viability: {
            description: 'Viability assessment protocol',
            sequences: ['cine', 'perfusion', 'lge'],
            parameters: {
                accelerationFactor: 3.0,
                technique: 'hybrid',
                perfusionSlices: 3
            },
            rationale: 'Rest perfusion to assess microvascular obstruction, LGE for transmural extent'
        },
        function: {
            description: 'Cardiac function assessment',
            sequences: ['cine'],
            parameters: {
                accelerationFactor: 2.5,
                technique: 'sense',
                cinePhases: 30,
                temporalRes: 35
            },
            rationale: 'High temporal resolution CINE for accurate volumetric analysis'
        },
        cardiomyopathy: {
            description: 'Cardiomyopathy characterization',
            sequences: ['cine', 't1-mapping', 't2-mapping', 'lge', 'flow'],
            parameters: {
                accelerationFactor: 2.0,
                technique: 'grappa',
                mappingType: 't1-molli'
            },
            rationale: 'Comprehensive tissue characterization with parametric mapping'
        }
    },

    /**
     * Generate protocol from clinical indication
     */
    generateProtocol(clinicalText) {
        const keywords = Utils.parseClinicalIndication(clinicalText);

        // Match to template
        let selectedTemplate = null;
        let templateKey = 'function'; // default

        if (keywords.conditions.includes('myocarditis')) {
            templateKey = 'myocarditis';
        } else if (keywords.conditions.includes('viability') || keywords.conditions.includes('ischemia')) {
            templateKey = 'viability';
        } else if (keywords.conditions.includes('cardiomyopathy')) {
            templateKey = 'cardiomyopathy';
        }

        selectedTemplate = this.protocolTemplates[templateKey];

        // Generate response
        const response = {
            protocol: selectedTemplate,
            explanation: this.generateExplanation(selectedTemplate, keywords),
            parameters: selectedTemplate.parameters,
            estimatedTime: this.estimateTotalScanTime(selectedTemplate.sequences)
        };

        return response;
    },

    /**
     * Generate natural language explanation
     */
    generateExplanation(template, keywords) {
        let explanation = `Based on your clinical indication, I recommend a **${template.description}**.\n\n`;

        explanation += `**Recommended Sequences:**\n`;
        template.sequences.forEach(seq => {
            explanation += `• ${seq.toUpperCase()}\n`;
        });

        explanation += `\n**Key Parameters:**\n`;
        const params = template.parameters;
        explanation += `• Parallel Imaging: ${params.technique.toUpperCase()} with R=${params.accelerationFactor}x\n`;

        if (params.cinePhases) {
            explanation += `• CINE: ${params.cinePhases} cardiac phases, ${params.temporalRes}ms temporal resolution\n`;
        }

        explanation += `\n**Clinical Rationale:**\n${template.rationale}\n`;

        explanation += `\n**Estimated Total Scan Time:** ${this.estimateTotalScanTime(template.sequences)} minutes`;

        return explanation;
    },

    /**
     * Optimize parameters for specific scenario
     */
    optimizeParameters(scenario) {
        const optimizations = {
            'high-heart-rate': {
                recommendation: 'Reduce cardiac phases, increase temporal resolution',
                parameters: { cinePhases: 20, temporalRes: 30, accelerationFactor: 2.5 }
            },
            'poor-breath-hold': {
                recommendation: 'Increase parallel imaging, reduce scan time per sequence',
                parameters: { accelerationFactor: 3.0, technique: 'compressed-sensing' }
            },
            'claustrophobia': {
                recommendation: 'Minimize total scan time, use SMS for multiple slices',
                parameters: { accelerationFactor: 3.0, technique: 'sms' }
            },
            'high-quality': {
                recommendation: 'Reduce acceleration, optimize SNR',
                parameters: { accelerationFactor: 1.5, coilElements: 32 }
            }
        };

        return optimizations[scenario] || {
            recommendation: 'Standard balanced protocol',
            parameters: { accelerationFactor: 2.0, technique: 'sense' }
        };
    },

    /**
     * Explain trade-offs
     */
    explainTradeoffs(parameter, value) {
        const explanations = {
            accelerationFactor: {
                increase: 'Higher acceleration reduces scan time but increases noise and potential artifacts. G-factor increases, reducing SNR.',
                decrease: 'Lower acceleration improves image quality and SNR but increases scan time. May exceed patient breath-hold capacity.'
            },
            temporalResolution: {
                increase: 'Better temporal resolution captures rapid cardiac motion more accurately but requires more k-space lines, increasing scan time.',
                decrease: 'Lower temporal resolution reduces scan time but may introduce temporal blurring, especially at high heart rates.'
            },
            coilElements: {
                increase: 'More coil elements allow higher acceleration with lower g-factor penalty. Improves parallel imaging performance.',
                decrease: 'Fewer coils limit maximum safe acceleration and increase g-factor, degrading SNR.'
            }
        };

        return explanations[parameter] || { increase: 'Parameter increased', decrease: 'Parameter decreased' };
    },

    /**
     * Estimate total scan time
     */
    estimateTotalScanTime(sequences) {
        const timePerSequence = {
            'cine': 8,      // minutes
            'perfusion': 2,
            'lge': 10,
            't1-mapping': 6,
            't2-mapping': 5,
            'flow': 7
        };

        let totalTime = 0;
        sequences.forEach(seq => {
            totalTime += timePerSequence[seq] || 5;
        });

        // Add setup and positioning overhead
        totalTime += 5;

        return Math.round(totalTime);
    },

    /**
     * Generate conversational response
     */
    generateResponse(userMessage) {
        const lowerMessage = userMessage.toLowerCase();

        // Protocol generation request
        if (lowerMessage.includes('protocol') || lowerMessage.includes('suspected') ||
            lowerMessage.includes('age') || lowerMessage.includes('yo ')) {
            const protocolResult = this.generateProtocol(userMessage);
            return protocolResult.explanation + '\n\nWould you like me to apply these settings?';
        }

        // Parameter optimization
        if (lowerMessage.includes('optimize') || lowerMessage.includes('improve')) {
            if (lowerMessage.includes('speed') || lowerMessage.includes('faster')) {
                const opt = this.optimizeParameters('poor-breath-hold');
                return `To optimize for speed:\n\n${opt.recommendation}\n\n` +
                    `I suggest: ${opt.parameters.technique.toUpperCase()} with R=${opt.parameters.accelerationFactor}x acceleration.`;
            } else if (lowerMessage.includes('quality') || lowerMessage.includes('snr')) {
                const opt = this.optimizeParameters('high-quality');
                return `To optimize for image quality:\n\n${opt.recommendation}\n\n` +
                    `I suggest: R=${opt.parameters.accelerationFactor}x with ${opt.parameters.coilElements} coil elements.`;
            }
        }

        // Explain parameters
        if (lowerMessage.includes('explain') || lowerMessage.includes('why')) {
            if (lowerMessage.includes('acceleration') || lowerMessage.includes('r =') || lowerMessage.includes('parallel')) {
                return `Parallel imaging acceleration works by:\n\n` +
                    `1. **Undersampling k-space**: Skipping phase encode lines to reduce scan time\n` +
                    `2. **Using coil arrays**: Multiple receiver coils with different spatial sensitivities\n` +
                    `3. **Reconstruction**: Algorithms (SENSE/GRAPPA/CS) recover missing data\n\n` +
                    `**Trade-off**: Higher R → faster scans but lower SNR (by √R × g-factor)`;
            } else if (lowerMessage.includes('g-factor') || lowerMessage.includes('g factor')) {
                return `The **g-factor** quantifies SNR penalty from parallel imaging:\n\n` +
                    `• g=1: Perfect (no penalty)\n` +
                    `• g=1.2-1.5: Excellent\n` +
                    `• g>2: Poor geometry, high noise\n\n` +
                    `It depends on coil arrangement, acceleration direction, and R-factor.`;
            }
        }

        // General cardiac imaging question
        if (lowerMessage.includes('cine')) {
            return `CINE imaging captures dynamic cardiac motion through the cardiac cycle.\n\n` +
                `**Key considerations:**\n` +
                `• Temporal resolution: 30-50ms typical (lower for high HR)\n` +
                `• 20-30 cardiac phases for smooth motion\n` +
                `• Balanced SSFP (TrueFISP) provides best blood-myocardium contrast`;
        }

        if (lowerMessage.includes('lge') || lowerMessage.includes('gadolinium')) {
            return `Late Gadolinium Enhancement (LGE) detects myocardial scar/fibrosis.\n\n` +
                `**Protocol tips:**\n` +
                `• Image 10-20 minutes post-contrast\n` +
                `• TI scout to null normal myocardium (typically 250-350ms at 1.5T)\n` +
                `• Phase-sensitive IR (PSIR) reduces TI sensitivity`;
        }

        // Default helpful response
        return `I can help you with:\n\n` +
            `• **Protocol generation**: Describe clinical scenario (e.g., "45yo male, suspected myocarditis")\n` +
            `• **Parameter optimization**: "Optimize for speed" or "optimize for quality"\n` +
            `• **Explanations**: "Explain g-factor" or "Why use SENSE?"\n` +
            `• **Sequence advice**: Ask about CINE, LGE, mapping, perfusion, or flow\n\n` +
            `What would you like to know?`;
    }
};

// Export
window.LLMEngine = LLMEngine;
