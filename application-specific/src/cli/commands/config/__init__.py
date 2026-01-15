"""Configuration Commands.

Commands for configuring model parameters, optimizer, loss function, and scheduler.
"""

import typer

from .list_complex_models import app as list_complex_models_app
from .list_loss_functions import app as list_loss_functions_app
from .list_optimizers import app as list_optimizers_app
from .list_pretrained_models import app as list_pretrained_models_app
from .list_schedulers import app as list_schedulers_app
from .reset_config import app as reset_config_app
from .set_complex_params import app as set_complex_params_app
from .set_loss_function import app as set_loss_function_app
from .set_model_params import app as set_model_params_app
from .set_optimizer import app as set_optimizer_app
from .set_pretrained_model import app as set_pretrained_model_app
from .set_scheduler import app as set_scheduler_app
from .show_config import app as show_config_app

app = typer.Typer(help="Configuration and parameters commands")

# Add commands from individual modules
app.add_typer(show_config_app)
app.add_typer(set_model_params_app)
app.add_typer(set_complex_params_app)
app.add_typer(list_complex_models_app)
app.add_typer(set_optimizer_app)
app.add_typer(set_loss_function_app)
app.add_typer(set_scheduler_app)
app.add_typer(set_pretrained_model_app)
app.add_typer(list_optimizers_app)
app.add_typer(list_loss_functions_app)
app.add_typer(list_schedulers_app)
app.add_typer(list_pretrained_models_app)
app.add_typer(reset_config_app)
