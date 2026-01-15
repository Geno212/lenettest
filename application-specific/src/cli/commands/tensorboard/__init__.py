"""TensorBoard Commands for CLI.

Commands for managing TensorBoard integration with training.
"""

import typer

from .clean_tensorboard_logs import app as clean_tensorboard_logs_app
from .open_tensorboard import app as open_tensorboard_app
from .show_tensorboard_logs import app as show_tensorboard_logs_app
from .start_tensorboard import app as start_tensorboard_app
from .stop_tensorboard import app as stop_tensorboard_app
from .tensorboard_status import app as tensorboard_status_app

app = typer.Typer(help="TensorBoard management commands")

# Add commands from individual modules
app.add_typer(start_tensorboard_app)
app.add_typer(stop_tensorboard_app)
app.add_typer(tensorboard_status_app)
app.add_typer(open_tensorboard_app)
app.add_typer(show_tensorboard_logs_app)
app.add_typer(clean_tensorboard_logs_app)
