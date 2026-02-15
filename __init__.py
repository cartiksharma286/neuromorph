"""
RF Coil Designer with Generative AI

A comprehensive package for designing and optimizing RF coils
using parametric modeling and generative algorithms.
"""

__version__ = "1.0.0"

from .coil_designer import (RFCoilDesigner, GenerativeCoilDesigner, 
                            CoilParameters)
from .coil_visualizer import CoilVisualizer

__all__ = [
    'RFCoilDesigner',
    'GenerativeCoilDesigner', 
    'CoilParameters',
    'CoilVisualizer'
]
