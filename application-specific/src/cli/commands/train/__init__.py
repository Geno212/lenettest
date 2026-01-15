"""Training Commands.

Commands for training neural network models.
"""

import typer

from .train_manual import app as train_manual_app
from .train_pretrained import app as train_pretrained_app
from .train_yolox import app as train_yolox_app

app = typer.Typer(help="Model training commands")

# Add commands from individual modules
app.add_typer(train_manual_app)
app.add_typer(train_pretrained_app)
app.add_typer(train_yolox_app)
