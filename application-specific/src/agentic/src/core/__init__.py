"""
Core Module - Foundational Components

This module contains the core components of the NN Generator system:
- State definition
- Graph construction
- System configuration

Exports:
    NNGeneratorState: Complete state type definition
    update_dialog_stack: Dialog stack management function
    build_graph: Main graph construction function
    SystemConfig: System configuration class
"""

from .state import (
    NNGeneratorState,
)


from .config import SystemConfig
from .graph import build_graph

__all__ = [
    "NNGeneratorState",
    "SystemConfig",
    "build_graph",
]