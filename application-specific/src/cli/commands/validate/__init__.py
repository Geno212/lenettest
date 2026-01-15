"""Validation Commands.

Commands for validating neural network architectures and checking for issues.
"""

import typer

from .validate_architecture import app as validate_architecture_app
from .validate_compatibility import app as validate_compatibility_app
from .validate_layers import app as validate_layers_app
from .validate_performance import app as validate_performance_app

app = typer.Typer(help="Architecture validation commands")

# Add commands from individual modules
app.add_typer(validate_architecture_app)
app.add_typer(validate_layers_app)
app.add_typer(validate_performance_app)
app.add_typer(validate_compatibility_app)
