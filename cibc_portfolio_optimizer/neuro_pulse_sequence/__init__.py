"""
Neuroimaging Quantum ML Framework

A comprehensive framework for MRI pulse sequence generation with
quantum machine learning optimization and proprioceptive feedback.
"""

__version__ = "1.0.0"
__author__ = "Neuroimaging Quantum ML Team"

from . import quantum_optimizer
from . import pulse_sequences
from . import quantum_proprioception
from . import adaptive_learning

__all__ = [
    'quantum_optimizer',
    'pulse_sequences',
    'quantum_proprioception',
    'adaptive_learning'
]
