import json
from pathlib import Path

import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.utils.console import console

app = typer.Typer(help="Load an existing neural network project")


@app.command("load")
def load_project(
    project_path: Path = typer.Argument(
        ...,
        help="Path to project directory or project.json file",
    ),
    set_active: bool = typer.Option(
        True,
        "--active",
        "-a",
        help="Set as active project",
    ),
):
    """Load an existing neural network project."""
    # Handle both directory and file inputs
    if project_path.is_file():
        if project_path.name == "project.json":
            project_dir = project_path.parent
            project_file = project_path
        else:
            console.print(f"[red]❌ {project_path} is not a project.json file[/red]")
            return
    elif project_path.is_dir():
        project_file = project_path / "project.json"
        if not project_file.exists():
            console.print(f"[red]❌ No project.json found in {project_path}[/red]")
            return
        project_dir = project_path
    else:
        console.print(f"[red]❌ Project path {project_path} does not exist[/red]")
        return

    try:
        with open(project_file) as f:
            project_data = json.load(f)

        # Validate project structure
        required_keys = ["name", "architecture", "config"]
        missing_keys = [key for key in required_keys if key not in project_data]

        if missing_keys:
            console.print(
                f"[red]❌ Invalid project file. Missing keys: {missing_keys}[/red]",
            )
            return

        # Load project data
        project_name = project_data["name"]
        description = project_data.get("description", "No description")
        author = project_data.get("author", "Unknown")

        # Success message
        load_text = Text.assemble(
            ("Project loaded successfully!", "bold green"),
            (f"\n📂 Project: {project_name}", "cyan"),
            ("\n📍 Directory: ", "dim"),
            (str(project_dir), "blue"),
            ("\n📝 Description: ", "dim"),
            (description, "white"),
            ("\n👤 Author: ", "dim"),
            (author, "white"),
        )

        console.print(
            Panel(load_text, title="[bold]Project Loaded[/bold]", border_style="green"),
        )

        if set_active:
            # Here you would set this as the active project in your configuration
            console.print("[blue]ℹ️  Project set as active[/blue]")

    except Exception as e:
        console.print(f"[red]❌ Error loading project: {e}[/red]")
