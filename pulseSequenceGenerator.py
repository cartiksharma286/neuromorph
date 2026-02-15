from dataclasses import dataclass
from typing import List, Tuple
import json

@dataclass
class PulseSequence:
    """Represents a single pulse in a sequence"""
    frequency: float  # MHz
    duration: float   # microseconds
    amplitude: float  # normalized 0-1
    phase: float      # degrees

class PulseSequenceGenerator:
    """Generate optimal pulse sequences using LLM assistance"""
    
    def __init__(self, target_goal: str):
        self.target_goal = target_goal
        self.sequences: List[PulseSequence] = []
    
    def create_sequence(self, pulses: List[dict]) -> List[PulseSequence]:
        """Convert pulse parameters to PulseSequence objects"""
        return [PulseSequence(**pulse) for pulse in pulses]
    
    def format_for_llm(self) -> str:
        """Format sequences for LLM optimization"""
        return json.dumps([
            {
                'frequency': p.frequency,
                'duration': p.duration,
                'amplitude': p.amplitude,
                'phase': p.phase
            }
            for p in self.sequences
        ], indent=2)
    
    def add_pulse(self, freq: float, duration: float, 
                  amplitude: float, phase: float) -> None:
        """Add a pulse to the sequence"""
        self.sequences.append(PulseSequence(freq, duration, amplitude, phase))

# Example usage
if __name__ == "__main__":
    gen = PulseSequenceGenerator("Spin excitation optimization")
    gen.add_pulse(frequency=100.0, duration=2.5, amplitude=1.0, phase=0)
    gen.add_pulse(frequency=100.0, duration=5.0, amplitude=0.5, phase=90)
    print(gen.format_for_llm())