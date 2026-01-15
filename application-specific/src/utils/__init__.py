"""Utility modules for the Neural Network Generator.

This package contains various utility classes and functions used throughout
the application for auto-extraction, file generation, and other common tasks.
"""

# Make key classes available at package level
try:
    from .AutoExtraction import AutoExtraction
    from .FileGenerator import FileGenerator
    from .Singleton import Singleton
    from .TemplateRenderer import TemplateRenderer

    __all__ = [
        "AutoExtraction",
        "FileGenerator",
        "Singleton",
        "TemplateRenderer",
    ]
except ImportError:
    # Handle cases where some modules might not be available
    __all__ = []
