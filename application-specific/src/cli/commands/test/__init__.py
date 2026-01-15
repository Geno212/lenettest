"""Test Commands.

Commands for testing neural network models with images and datasets.
"""

import typer

from .benchmark_model import app as benchmark_model_app
from .compare_models import app as compare_models_app
from .test_batch import app as test_batch_app
from .test_image import app as test_image_app

app = typer.Typer(help="Model testing commands")

# Add commands from individual modules
app.add_typer(test_image_app)
app.add_typer(test_batch_app)
app.add_typer(benchmark_model_app)
app.add_typer(compare_models_app)
