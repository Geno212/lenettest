"""Project Management Commands.

Commands for creating, managing, and organizing neural network projects.
"""

import typer

from .create_project import app as create_project_app
from .list_projects import app as list_projects_app
from .load_project import app as load_project_app
from .project_info import app as project_info_app

app = typer.Typer(help="Project management commands")

# Add commands from individual modules
app.add_typer(create_project_app)
app.add_typer(list_projects_app)
app.add_typer(load_project_app)
app.add_typer(project_info_app)
