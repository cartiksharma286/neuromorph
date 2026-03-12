#!/usr/bin/env python3
"""
DejaVu Simulator: Neural Circuitry for Memory Pattern Recognition
Using Continued Fractions and Optimal Pattern Indexing
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict
from fractions import Fraction
import math

class ContinuedFraction:
    """Represent patterns using continued fractions for memory encoding"""
    
    def __init__(self, value: float, max_terms: int = 10):
        """Initialize continued fraction representation"""
        self.value = value
        self.max_terms = max_terms
        self.terms = self._compute_cf(value)
    
    def _compute_cf(self, x: float) -> List[int]:
        """Compute continued fraction representation of a number"""
        terms = []
        x = float(x)
        for _ in range(self.max_terms):
            a = int(np.floor(x))
            terms.append(a)
            x = x - a
            if x < 1e-10:
                break
            x = 1.0 / x
        return terms
    
    def convergents(self) -> List[Tuple[int, int]]:
        """Compute convergents (rational approximations) of continued fraction"""
        convergents = []
        for i in range(len(self.terms)):
            h_prev, k_prev = 1, 0
            h_curr, k_curr = self.terms[0], 1
            
            for j in range(1, i + 1):
                h_next = self.terms[j] * h_curr + h_prev
                k_next = self.terms[j] * k_curr + k_prev
                h_prev, k_prev = h_curr, k_curr
                h_curr, k_curr = h_next, k_next
            
            if i == 0:
                convergents.append((self.terms[0], 1))
            else:
                convergents.append((h_curr, k_curr))
        
        return convergents
    
    def similarity_score(self, other: 'ContinuedFraction') -> float:
        """Compute similarity between two continued fractions"""
        # Align lengths
        len1, len2 = len(self.terms), len(other.terms)
        min_len = min(len1, len2)
        
        # Compare term-by-term with decay
        similarity = 0.0
        for i in range(min_len):
            if self.terms[i] == other.terms[i]:
                similarity += (1.0 / (i + 1))
        
        return similarity / min_len if min_len > 0 else 0.0


class OptimalPatternIndex:
    """Efficient pattern indexing using hashing and clustering"""
    
    def __init__(self, embedding_dim: int = 64):
        """Initialize pattern index"""
        self.embedding_dim = embedding_dim
        self.patterns = {}  # pattern_id -> pattern_data
        self.index = defaultdict(list)  # hash -> list of pattern_ids
        self.pattern_count = 0
    
    def hash_pattern(self, pattern: np.ndarray) -> int:
        """Create locality-sensitive hash of pattern"""
        # Quantize pattern to discrete space for hashing
        quantized = np.round(pattern * 100).astype(int)
        hash_val = hash(tuple(quantized[:min(16, len(quantized))]))
        return abs(hash_val) % 10000
    
    def add_pattern(self, pattern: np.ndarray, metadata: Dict = None) -> int:
        """Add pattern to index"""
        pat_id = self.pattern_count
        self.pattern_count += 1
        
        # Store pattern
        self.patterns[pat_id] = {
            'vector': pattern.copy(),
            'metadata': metadata or {},
            'cf': ContinuedFraction(float(np.mean(pattern)))
        }
        
        # Index by hash
        h = self.hash_pattern(pattern)
        self.index[h].append(pat_id)
        
        return pat_id
    
    def query_similar(self, query_pattern: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """Find k most similar patterns"""
        h = self.hash_pattern(query_pattern)
        
        # Get candidates from hash bucket and neighbors
        candidates = set()
        for hash_offset in range(-2, 3):
            bucket = self.index[h + hash_offset]
            candidates.update(bucket)
        
        # Compute similarities
        similarities = []
        query_cf = ContinuedFraction(float(np.mean(query_pattern)))
        
        for pat_id in candidates:
            pat_data = self.patterns[pat_id]
            
            # Cosine similarity
            cos_sim = np.dot(query_pattern, pat_data['vector']) / (
                np.linalg.norm(query_pattern) * np.linalg.norm(pat_data['vector']) + 1e-8)
            
            # Continued fraction similarity
            cf_sim = query_cf.similarity_score(pat_data['cf'])
            
            # Combined score
            combined = 0.7 * cos_sim + 0.3 * cf_sim
            similarities.append((pat_id, float(combined)))
        
        # Return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]


class NeuralCircuit:
    """Simulates neural circuitry for deja vu memory processing"""
    
    def __init__(self, num_neurons: int = 256):
        """Initialize neural circuit"""
        self.num_neurons = num_neurons
        
        # Neural populations
        self.sensory_layer = np.random.randn(num_neurons) * 0.1
        self.memory_layer = np.random.randn(num_neurons) * 0.1
        self.recognition_layer = np.random.randn(num_neurons) * 0.1
        
        # Connection matrices
        self.W_sensory_memory = np.random.randn(num_neurons, num_neurons) * 0.1
        self.W_memory_recognition = np.random.randn(num_neurons, num_neurons) * 0.1
        self.W_recognition_memory = np.random.randn(num_neurons, num_neurons) * 0.05
        
        # Learning parameters
        self.learning_rate = 0.01
        self.decay_rate = 0.95
    
    def activation_function(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax normalization"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-8)
    
    def encode_stimulus(self, stimulus: np.ndarray) -> np.ndarray:
        """Encode external stimulus to sensory layer"""
        # Normalize stimulus
        stim_norm = (stimulus - np.mean(stimulus)) / (np.std(stimulus) + 1e-8)
        
        # Project to sensory layer
        if len(stim_norm) < self.num_neurons:
            padded = np.zeros(self.num_neurons)
            padded[:len(stim_norm)] = stim_norm
            stim_norm = padded
        else:
            stim_norm = stim_norm[:self.num_neurons]
        
        self.sensory_layer = stim_norm
        return self.sensory_layer.copy()
    
    def forward_pass(self, num_steps: int = 5) -> Tuple[float, np.ndarray]:
        """Execute forward pass through neural circuit"""
        deja_vu_scores = []
        
        for step in range(num_steps):
            # Sensory -> Memory
            memory_input = np.dot(self.W_sensory_memory, self.sensory_layer)
            self.memory_layer = self.activation_function(
                0.8 * self.memory_layer + 0.2 * memory_input
            )
            self.memory_layer = self.memory_layer / (np.linalg.norm(self.memory_layer) + 1e-8)
            
            # Memory -> Recognition
            recognition_input = np.dot(self.W_memory_recognition, self.memory_layer)
            self.recognition_layer = self.activation_function(
                0.8 * self.recognition_layer + 0.2 * recognition_input
            )
            self.recognition_layer = self.recognition_layer / (np.linalg.norm(self.recognition_layer) + 1e-8)
            
            # Recognition -> Memory feedback
            feedback = np.dot(self.W_recognition_memory, self.recognition_layer) * 0.05
            self.memory_layer += feedback
            self.memory_layer = self.memory_layer / (np.linalg.norm(self.memory_layer) + 1e-8)
            
            # Compute deja vu score
            deja_vu_score = self._compute_deja_vu_signal()
            deja_vu_scores.append(deja_vu_score)
        
        final_score = np.mean(deja_vu_scores)
        return final_score, self.recognition_layer.copy()
    
    def _compute_deja_vu_signal(self) -> float:
        """Compute deja vu signal strength"""
        # Deja vu occurs when recognition layer strongly activates memory layer
        recognition_strength = np.mean(np.abs(self.recognition_layer))
        memory_strength = np.mean(np.abs(self.memory_layer))
        
        # Cross-correlation measures familiarity
        norm_rec = np.linalg.norm(self.recognition_layer) + 1e-8
        norm_mem = np.linalg.norm(self.memory_layer) + 1e-8
        
        correlation = np.dot(self.recognition_layer, self.memory_layer) / (norm_rec * norm_mem)
        
        # Deja vu = sigmoid(high recognition * memory correlation)
        # This prevents explosion while maintaining sensitivity
        raw_score = recognition_strength * memory_strength * max(0, correlation)
        
        # Apply sigmoid to bound between 0 and 1
        deja_vu = 1.0 / (1.0 + np.exp(-raw_score))
        
        return float(deja_vu)
    
    def learn_pattern(self, stimulus: np.ndarray):
        """Learn a pattern through Hebbian plasticity"""
        self.encode_stimulus(stimulus)
        self.forward_pass(num_steps=3)
        
        # Normalize layers
        self.sensory_layer = self.sensory_layer / (np.linalg.norm(self.sensory_layer) + 1e-8)
        self.memory_layer = self.memory_layer / (np.linalg.norm(self.memory_layer) + 1e-8)
        self.recognition_layer = self.recognition_layer / (np.linalg.norm(self.recognition_layer) + 1e-8)
        
        # Hebbian learning: w_ij += eta * x_i * x_j with smaller learning rate
        delta_W1 = self.learning_rate * 0.1 * np.outer(self.sensory_layer, self.memory_layer)
        delta_W2 = self.learning_rate * 0.1 * np.outer(self.memory_layer, self.recognition_layer)
        
        self.W_sensory_memory += delta_W1
        self.W_memory_recognition += delta_W2
        
        # L2 normalization to prevent weight explosion
        w1_norm = np.linalg.norm(self.W_sensory_memory)
        w2_norm = np.linalg.norm(self.W_memory_recognition)
        
        if w1_norm > 1.0:
            self.W_sensory_memory /= w1_norm
        if w2_norm > 1.0:
            self.W_memory_recognition /= w2_norm


class DejaVuSimulator:
    """Complete deja vu simulation system"""
    
    def __init__(self, num_neurons: int = 256):
        """Initialize deja vu simulator"""
        self.circuit = NeuralCircuit(num_neurons=num_neurons)
        self.pattern_index = OptimalPatternIndex(embedding_dim=num_neurons)
        self.memory_buffer = []
        self.deja_vu_history = []
    
    def experience_stimulus(self, stimulus: np.ndarray, is_new_learning: bool = False) -> Dict:
        """Process a stimulus and detect deja vu"""
        # Encode stimulus
        self.circuit.encode_stimulus(stimulus)
        
        # If learning mode, strengthen memory
        if is_new_learning:
            self.circuit.learn_pattern(stimulus)
        
        # Forward pass through circuit
        deja_vu_score, activation_pattern = self.circuit.forward_pass()
        
        # Add to pattern index
        pattern_id = self.pattern_index.add_pattern(
            activation_pattern,
            metadata={'stimulus_mean': float(np.mean(stimulus))}
        )
        
        # Query for similar patterns
        similar_patterns = self.pattern_index.query_similar(activation_pattern, k=3)
        
        # Compute stimulus-based variation
        # Use stimulus statistics to add natural variation
        stimulus_var = float(np.var(stimulus))
        stimulus_entropy = float(np.sum(-np.maximum(stimulus, 0.01) * np.log(np.maximum(stimulus, 0.01)+1e-8) + 1e-8))
        
        # Top match similarity
        top_match_sim = similar_patterns[0][1] if similar_patterns else 0.0
        
        # Combine multiple factors
        # Base circuit signal + pattern similarity + stimulus information
        combined_deja_vu = (0.5 * deja_vu_score + 
                           0.3 * top_match_sim + 
                           0.2 * (stimulus_var / (stimulus_entropy + 1.0)))
        combined_deja_vu = min(1.0, max(0.0, combined_deja_vu))  # Clip to [0, 1]
        
        # Add memory buffer influence (repeated same pattern increases deja vu)
        if len(self.memory_buffer) > 0:
            # Check if this stimulus is similar to recent memories
            recent_memory_score = 0.0
            for i in range(max(0, len(self.memory_buffer) - 5), len(self.memory_buffer)):
                recent_pattern = self.memory_buffer[i]['activation']
                similarity = np.dot(activation_pattern, recent_pattern) / (
                    np.linalg.norm(activation_pattern) * np.linalg.norm(recent_pattern) + 1e-8
                )
                recent_memory_score += similarity
            recent_memory_score /= min(5, len(self.memory_buffer))
            
            # Boost deja vu if similar to recent memories
            combined_deja_vu = 0.7 * combined_deja_vu + 0.3 * recent_memory_score
            combined_deja_vu = min(1.0, combined_deja_vu)
        
        # Store in memory buffer
        memory_entry = {
            'stimulus': stimulus.copy(),
            'activation': activation_pattern.copy(),
            'pattern_id': pattern_id,
            'deja_vu_score': combined_deja_vu,
            'similarity_matches': similar_patterns
        }
        self.memory_buffer.append(memory_entry)
        self.deja_vu_history.append(combined_deja_vu)
        
        return {
            'deja_vu_score': float(combined_deja_vu),
            'is_deja_vu': combined_deja_vu > 0.3,
            'pattern_id': pattern_id,
            'similar_patterns': len(similar_patterns),
            'top_match': top_match_sim
        }
    
    def generate_stimulus_sequence(self, num_stimuli: int = 10, pattern_type: str = 'sine') -> List[np.ndarray]:
        """Generate a sequence of stimuli with repeating patterns"""
        stimuli = []
        
        for i in range(num_stimuli):
            if pattern_type == 'sine':
                # Sinusoidal patterns with variable frequency
                freq = 0.5 + (i % 3) * 0.3
                t = np.linspace(0, 2 * np.pi, 64)
                stimulus = np.sin(freq * t) + 0.1 * np.random.randn(64)
            
            elif pattern_type == 'gaussian':
                # Gaussian bumps
                center = (i % 4) * 16
                x = np.arange(64)
                stimulus = np.exp(-((x - center) ** 2) / 50.0) + 0.05 * np.random.randn(64)
            
            elif pattern_type == 'random':
                # Random noise (no pattern)
                stimulus = np.random.randn(64)
            
            stimuli.append(stimulus)
        
        return stimuli
    
    def simulate_familiarization(self, stimulus: np.ndarray, num_exposures: int = 5) -> List[float]:
        """Simulate repeated exposure to stimulus (familiarization effect)"""
        deja_vu_scores = []
        
        for exposure in range(num_exposures):
            # Add noise to simulate variation
            noisy_stimulus = stimulus + 0.05 * exposure * np.random.randn(*stimulus.shape)
            result = self.experience_stimulus(noisy_stimulus, is_new_learning=(exposure == 0))
            deja_vu_scores.append(result['deja_vu_score'])
        
        return deja_vu_scores
    
    def get_statistics(self) -> Dict:
        """Get simulator statistics"""
        if not self.deja_vu_history:
            return {}
        
        scores = np.array(self.deja_vu_history)
        return {
            'mean_deja_vu': float(np.mean(scores)),
            'std_deja_vu': float(np.std(scores)),
            'max_deja_vu': float(np.max(scores)),
            'min_deja_vu': float(np.min(scores)),
            'num_experiences': len(self.deja_vu_history),
            'deja_vu_frequency': float(np.sum(scores > 0.3) / len(scores))
        }


if __name__ == '__main__':
    # Example usage
    sim = DejaVuSimulator(num_neurons=256)
    
    print("="*80)
    print("DejaVu Simulator: Neural Circuitry with Continued Fractions")
    print("="*80)
    
    # Generate stimulus sequence
    print("\n1. Generating sinusoidal stimulus sequence...")
    stimuli = sim.generate_stimulus_sequence(num_stimuli=15, pattern_type='sine')
    
    # Process stimuli
    print("2. Processing stimuli through neural circuit...")
    for i, stimulus in enumerate(stimuli):
        result = sim.experience_stimulus(stimulus, is_new_learning=(i < 3))
        print(f"   Stimulus {i+1}: DejaVu={result['deja_vu_score']:.4f}, "
              f"IsDejaVu={result['is_deja_vu']}, TopMatch={result['top_match']:.4f}")
    
    # Simulate familiarization
    print("\n3. Testing familiarization effect...")
    test_stimulus = stimuli[0]
    familiarity_curve = sim.simulate_familiarization(test_stimulus, num_exposures=8)
    print(f"   Familiarization curve: {[f'{x:.3f}' for x in familiarity_curve]}")
    
    # Statistics
    print("\n4. Simulator Statistics:")
    stats = sim.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
    
    print("\n" + "="*80)
