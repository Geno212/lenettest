"""Architecture Design Commands.

Commands for designing and managing neural network architectures.
"""

import typer

from .add_layer import app as add_layer_app
from .add_residual_block import app as add_residual_block_app
from .create_architecture import app as create_architecture_app
from .create_complex_architecture import app as create_complex_architecture_app
from .list_available_layers import app as list_available_layers_app
from .list_layers import app as list_layers_app
from .list_pretrained_models import app as list_pretrained_app
from .load_arch import app as load_arch_app
from .move_layer import app as move_layer_app
from .remove_layer import app as remove_layer_app
from .remove_residual_block import app as remove_residual_block_app
from .save_arch import app as save_arch_app
from .save_complex_architecture import app as save_complex_arch_app
from .show_architecture import app as show_architecture_app

app = typer.Typer(help="Architecture design commands")

# Add commands from individual modules
app.add_typer(create_architecture_app)
app.add_typer(create_complex_architecture_app)
app.add_typer(save_complex_arch_app)
app.add_typer(add_layer_app)
app.add_typer(add_residual_block_app)
app.add_typer(remove_layer_app)
app.add_typer(remove_residual_block_app)
app.add_typer(move_layer_app)
app.add_typer(list_layers_app)
app.add_typer(list_available_layers_app)
app.add_typer(list_pretrained_app)
app.add_typer(show_architecture_app)
app.add_typer(save_arch_app)
app.add_typer(load_arch_app)
