"""Code Generation Commands.

Commands for generating PyTorch and SystemC code from architectures.
"""

import typer

from .generate_pytorch import app as generate_pytorch_app
from .list_templates import app as list_templates_app

app = typer.Typer(help="Code generation commands")

# Add commands from individual modules
app.add_typer(generate_pytorch_app)
app.add_typer(list_templates_app)
