import json
from pathlib import Path

import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.utils.console import console

app = typer.Typer(help="Show project information")


@app.command("info")
def project_info(
    project_path: Path = typer.Argument(
        ...,
        help="Path to project directory or project.json file",
    ),
):
    """Show detailed information about a project."""
    # Handle both directory and file inputs
    if project_path.is_file():
        if project_path.name == "project.json":
            project_file = project_path
        else:
            console.print(f"[red]❌ {project_path} is not a project.json file[/red]")
            return
    elif project_path.is_dir():
        project_file = project_path / "project.json"
        if not project_file.exists():
            console.print(f"[red]❌ No project.json found in {project_path}[/red]")
            return
    else:
        console.print(f"[red]❌ Project path {project_path} does not exist[/red]")
        return

    try:
        with open(project_file) as f:
            project_data = json.load(f)

        # Get project statistics
        project_dir = project_file.parent
        num_configs = (
            len(list((project_dir / "configs").glob("*.json")))
            if (project_dir / "configs").exists()
            else 0
        )
        num_architectures = (
            len(list((project_dir / "architectures").glob("*.json")))
            if (project_dir / "architectures").exists()
            else 0
        )
        num_outputs = (
            len(list((project_dir / "outputs").iterdir()))
            if (project_dir / "outputs").exists()
            else 0
        )

        # Create detailed info panel
        info_text = Text.assemble(
            ("Project Information", "bold blue"),
            (f"\n🏷️  Name: {project_data.get('name', 'Unknown')}", "cyan"),
            (
                f"\n📝 Description: {project_data.get('description', 'No description')}",
                "white",
            ),
            (f"\n👤 Author: {project_data.get('author', 'Unknown')}", "green"),
            (f"\n📅 Created: {project_data.get('created_at', 'Unknown')}", "yellow"),
            (f"\n📊 Version: {project_data.get('version', 'Unknown')}", "magenta"),
            (f"\n📁 Directory: {project_dir}", "blue"),
            (
                f"\n🔧 Status: {project_data.get('status', 'unknown')}",
                "green" if project_data.get("status") == "created" else "yellow",
            ),
            ("\n\n📊 Statistics:", "bold green"),
            (f"\n📄 Configuration files: {num_configs}", "cyan"),
            (f"\n🏗️  Architecture files: {num_architectures}", "cyan"),
            (f"\n📦 Output files: {num_outputs}", "cyan"),
        )

        console.print(
            Panel(info_text, title="[bold]Project Details[/bold]", border_style="blue"),
        )

    except Exception as e:
        console.print(f"[red]❌ Error reading project info: {e}[/red]")
