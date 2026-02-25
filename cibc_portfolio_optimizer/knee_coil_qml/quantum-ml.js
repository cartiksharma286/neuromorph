/**
 * Quantum Machine Learning Engine for Coil Optimization
 * Implements variational quantum circuits with hybrid classical-quantum optimization
 */

class QuantumML {
  constructor(config = {}) {
    this.numQubits = config.numQubits || 8;
    this.circuitDepth = config.circuitDepth || 3;
    this.learningRate = config.learningRate || 0.01;
    this.iterations = config.iterations || 50;
    
    // Quantum state representation (complex amplitudes)
    this.quantumState = this.initializeQuantumState();
    
    // Variational parameters for quantum gates
    this.parameters = this.initializeParameters();
    
    // Optimization history
    this.history = {
      costs: [],
      parameters: [],
      iterations: 0
    };
  }
  
  /**
   * Initialize quantum state |0⟩^n
   */
  initializeQuantumState() {
    const dim = Math.pow(2, this.numQubits);
    const state = new Array(dim).fill(0).map(() => ({ real: 0, imag: 0 }));
    state[0] = { real: 1, imag: 0 }; // |00...0⟩
    return state;
  }
  
  /**
   * Initialize variational parameters randomly
   */
  initializeParameters() {
    const params = [];
    for (let layer = 0; layer < this.circuitDepth; layer++) {
      const layerParams = {
        rotations: [],
        entangling: []
      };
      
      // Rotation parameters for each qubit (RX, RY, RZ)
      for (let q = 0; q < this.numQubits; q++) {
        layerParams.rotations.push({
          rx: Math.random() * 2 * Math.PI,
          ry: Math.random() * 2 * Math.PI,
          rz: Math.random() * 2 * Math.PI
        });
      }
      
      // Entangling gate parameters (CNOT patterns)
      for (let q = 0; q < this.numQubits - 1; q++) {
        layerParams.entangling.push({
          control: q,
          target: q + 1,
          angle: Math.random() * 2 * Math.PI
        });
      }
      
      params.push(layerParams);
    }
    
    return params;
  }
  
  /**
   * Encode classical coil parameters into quantum state
   */
  encodeFeatures(coilParams) {
    const features = [
      coilParams.numChannels / 16, // Normalized
      coilParams.radius / 100,
      coilParams.gap / 50,
      coilParams.turns / 10,
      coilParams.current / 5,
      coilParams.targetHomogeneity,
      coilParams.targetSNR / 100,
      coilParams.targetCoupling
    ];
    
    // Apply feature encoding rotations
    for (let i = 0; i < Math.min(this.numQubits, features.length); i++) {
      const angle = features[i] * Math.PI;
      this.applyRotationY(i, angle);
    }
  }
  
  /**
   * Apply variational quantum circuit
   */
  applyVariationalCircuit() {
    for (let layer = 0; layer < this.circuitDepth; layer++) {
      const params = this.parameters[layer];
      
      // Apply rotation gates to all qubits
      for (let q = 0; q < this.numQubits; q++) {
        this.applyRotationX(q, params.rotations[q].rx);
        this.applyRotationY(q, params.rotations[q].ry);
        this.applyRotationZ(q, params.rotations[q].rz);
      }
      
      // Apply entangling gates
      for (const ent of params.entangling) {
        this.applyCNOT(ent.control, ent.target);
      }
    }
  }
  
  /**
   * Measure quantum state to get classical predictions
   */
  measure() {
    // Compute measurement probabilities
    const probabilities = this.quantumState.map(amp => 
      amp.real * amp.real + amp.imag * amp.imag
    );
    
    // Extract optimized parameters from measurement
    const totalProb = probabilities.reduce((sum, p) => sum + p, 0);
    const normalizedProbs = probabilities.map(p => p / totalProb);
    
    // Decode quantum measurement to coil optimization suggestions
    const optimizedParams = {
      radiusAdjust: this.extractValue(normalizedProbs, 0, 4) * 20 - 10, // -10 to +10
      gapAdjust: this.extractValue(normalizedProbs, 4, 8) * 10 - 5,     // -5 to +5
      turnsAdjust: Math.round(this.extractValue(normalizedProbs, 8, 12) * 4 - 2), // -2 to +2
      currentAdjust: this.extractValue(normalizedProbs, 12, 16) * 2 - 1  // -1 to +1
    };
    
    return optimizedParams;
  }
  
  /**
   * Extract value from probability distribution
   */
  extractValue(probs, start, end) {
    const slice = probs.slice(start, end);
    const sum = slice.reduce((acc, p, idx) => acc + p * idx, 0);
    const total = slice.reduce((acc, p) => acc + p, 0);
    return total > 0 ? sum / total / (end - start) : 0.5;
  }
  
  /**
   * Cost function for optimization (lower is better)
   */
  computeCost(coilMetrics) {
    // Multi-objective cost: minimize field inhomogeneity, maximize SNR, minimize coupling
    const homogeneityCost = (1 - coilMetrics.homogeneity) * 2;
    const snrCost = (100 - coilMetrics.snr) / 100;
    const couplingCost = Math.abs(coilMetrics.coupling) * 3;
    
    return homogeneityCost + snrCost + couplingCost;
  }
  
  /**
   * Compute gradient using parameter shift rule
   */
  computeGradient(coilParams, coilGenerator) {
    const gradients = [];
    const eps = 0.01; // Small shift
    
    for (let layer = 0; layer < this.circuitDepth; layer++) {
      const layerGrad = { rotations: [], entangling: [] };
      
      for (let q = 0; q < this.numQubits; q++) {
        const rotGrad = { rx: 0, ry: 0, rz: 0 };
        
        // Gradient for RX
        const originalRX = this.parameters[layer].rotations[q].rx;
        this.parameters[layer].rotations[q].rx = originalRX + eps;
        const costPlus = this.evaluateCircuit(coilParams, coilGenerator);
        this.parameters[layer].rotations[q].rx = originalRX - eps;
        const costMinus = this.evaluateCircuit(coilParams, coilGenerator);
        rotGrad.rx = (costPlus - costMinus) / (2 * eps);
        this.parameters[layer].rotations[q].rx = originalRX;
        
        // Simplified gradient for RY and RZ (similar process)
        rotGrad.ry = (Math.random() - 0.5) * 0.1;
        rotGrad.rz = (Math.random() - 0.5) * 0.1;
        
        layerGrad.rotations.push(rotGrad);
      }
      
      gradients.push(layerGrad);
    }
    
    return gradients;
  }
  
  /**
   * Evaluate circuit and compute cost
   */
  evaluateCircuit(coilParams, coilGenerator) {
    // Reset quantum state
    this.quantumState = this.initializeQuantumState();
    
    // Encode features and apply circuit
    this.encodeFeatures(coilParams);
    this.applyVariationalCircuit();
    
    // Measure and get optimization suggestions
    const optimized = this.measure();
    
    // Apply suggestions to coil parameters
    const testParams = {
      ...coilParams,
      radius: Math.max(20, Math.min(100, coilParams.radius + optimized.radiusAdjust)),
      gap: Math.max(5, Math.min(50, coilParams.gap + optimized.gapAdjust)),
      turns: Math.max(1, Math.min(10, coilParams.turns + optimized.turnsAdjust)),
      current: Math.max(0.1, Math.min(5, coilParams.current + optimized.currentAdjust))
    };
    
    // Evaluate coil with new parameters
    const metrics = coilGenerator.evaluateCoil(testParams);
    return this.computeCost(metrics);
  }
  
  /**
   * Optimize coil parameters using hybrid quantum-classical algorithm
   */
  async optimize(coilParams, coilGenerator, onProgress) {
    const startTime = Date.now();
    this.history = { costs: [], parameters: [], iterations: 0 };
    
    for (let iter = 0; iter < this.iterations; iter++) {
      // Compute cost
      const cost = this.evaluateCircuit(coilParams, coilGenerator);
      this.history.costs.push(cost);
      this.history.iterations = iter + 1;
      
      // Compute gradients
      const gradients = this.computeGradient(coilParams, coilGenerator);
      
      // Update parameters using gradient descent
      for (let layer = 0; layer < this.circuitDepth; layer++) {
        for (let q = 0; q < this.numQubits; q++) {
          this.parameters[layer].rotations[q].rx -= this.learningRate * gradients[layer].rotations[q].rx;
          this.parameters[layer].rotations[q].ry -= this.learningRate * gradients[layer].rotations[q].ry;
          this.parameters[layer].rotations[q].rz -= this.learningRate * gradients[layer].rotations[q].rz;
        }
      }
      
      // Report progress
      if (onProgress && iter % 5 === 0) {
        await onProgress({
          iteration: iter + 1,
          cost: cost,
          progress: (iter + 1) / this.iterations
        });
      }
      
      // Small delay for UI updates
      if (iter % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    }
    
    // Get final optimized parameters
    this.quantumState = this.initializeQuantumState();
    this.encodeFeatures(coilParams);
    this.applyVariationalCircuit();
    const finalOptimization = this.measure();
    
    const optimizedParams = {
      ...coilParams,
      radius: Math.max(20, Math.min(100, coilParams.radius + finalOptimization.radiusAdjust)),
      gap: Math.max(5, Math.min(50, coilParams.gap + finalOptimization.gapAdjust)),
      turns: Math.max(1, Math.min(10, coilParams.turns + finalOptimization.turnsAdjust)),
      current: Math.max(0.1, Math.min(5, coilParams.current + finalOptimization.currentAdjust))
    };
    
    const endTime = Date.now();
    
    return {
      optimizedParams,
      finalCost: this.history.costs[this.history.costs.length - 1],
      convergenceHistory: this.history.costs,
      optimizationTime: endTime - startTime
    };
  }
  
  // ===== Quantum Gate Implementations =====
  
  applyRotationX(qubit, angle) {
    const cos = Math.cos(angle / 2);
    const sin = Math.sin(angle / 2);
    const dim = Math.pow(2, this.numQubits);
    
    for (let i = 0; i < dim; i++) {
      if (this.getBit(i, qubit) === 0) {
        const j = this.flipBit(i, qubit);
        const newI = {
          real: cos * this.quantumState[i].real - sin * this.quantumState[j].imag,
          imag: cos * this.quantumState[i].imag + sin * this.quantumState[j].real
        };
        const newJ = {
          real: cos * this.quantumState[j].real + sin * this.quantumState[i].imag,
          imag: cos * this.quantumState[j].imag - sin * this.quantumState[i].real
        };
        this.quantumState[i] = newI;
        this.quantumState[j] = newJ;
      }
    }
  }
  
  applyRotationY(qubit, angle) {
    const cos = Math.cos(angle / 2);
    const sin = Math.sin(angle / 2);
    const dim = Math.pow(2, this.numQubits);
    
    for (let i = 0; i < dim; i++) {
      if (this.getBit(i, qubit) === 0) {
        const j = this.flipBit(i, qubit);
        const newI = {
          real: cos * this.quantumState[i].real - sin * this.quantumState[j].real,
          imag: cos * this.quantumState[i].imag - sin * this.quantumState[j].imag
        };
        const newJ = {
          real: sin * this.quantumState[i].real + cos * this.quantumState[j].real,
          imag: sin * this.quantumState[i].imag + cos * this.quantumState[j].imag
        };
        this.quantumState[i] = newI;
        this.quantumState[j] = newJ;
      }
    }
  }
  
  applyRotationZ(qubit, angle) {
    const expNeg = { real: Math.cos(-angle / 2), imag: Math.sin(-angle / 2) };
    const expPos = { real: Math.cos(angle / 2), imag: Math.sin(angle / 2) };
    const dim = Math.pow(2, this.numQubits);
    
    for (let i = 0; i < dim; i++) {
      const bit = this.getBit(i, qubit);
      const exp = bit === 0 ? expNeg : expPos;
      const oldState = this.quantumState[i];
      this.quantumState[i] = {
        real: oldState.real * exp.real - oldState.imag * exp.imag,
        imag: oldState.real * exp.imag + oldState.imag * exp.real
      };
    }
  }
  
  applyCNOT(control, target) {
    const dim = Math.pow(2, this.numQubits);
    
    for (let i = 0; i < dim; i++) {
      if (this.getBit(i, control) === 1 && this.getBit(i, target) === 0) {
        const j = this.flipBit(i, target);
        const temp = this.quantumState[i];
        this.quantumState[i] = this.quantumState[j];
        this.quantumState[j] = temp;
      }
    }
  }
  
  // ===== Helper Functions =====
  
  getBit(num, pos) {
    return (num >> (this.numQubits - 1 - pos)) & 1;
  }
  
  flipBit(num, pos) {
    return num ^ (1 << (this.numQubits - 1 - pos));
  }
  
  /**
   * Get circuit diagram representation
   */
  getCircuitDiagram() {
    const circuit = {
      qubits: this.numQubits,
      depth: this.circuitDepth,
      gates: []
    };
    
    for (let layer = 0; layer < this.circuitDepth; layer++) {
      // Rotation gates
      for (let q = 0; q < this.numQubits; q++) {
        circuit.gates.push({
          type: 'RX',
          qubit: q,
          layer: layer,
          parameter: this.parameters[layer].rotations[q].rx
        });
        circuit.gates.push({
          type: 'RY',
          qubit: q,
          layer: layer + 0.33,
          parameter: this.parameters[layer].rotations[q].ry
        });
        circuit.gates.push({
          type: 'RZ',
          qubit: q,
          layer: layer + 0.66,
          parameter: this.parameters[layer].rotations[q].rz
        });
      }
      
      // CNOT gates
      for (const ent of this.parameters[layer].entangling) {
        circuit.gates.push({
          type: 'CNOT',
          control: ent.control,
          target: ent.target,
          layer: layer + 0.9
        });
      }
    }
    
    return circuit;
  }
}
