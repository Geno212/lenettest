import json
import os
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.cli.utils.console import console

app = typer.Typer(help="List neural network projects in a directory")


@app.command("list")
def list_projects(
    directory: Path = typer.Option(
        Path.cwd(),
        "--dir",
        "-d",
        help="Directory to search for projects",
    ),
    show_details: bool = typer.Option(
        False,
        "--details",
        help="Show detailed project information",
    ),
):
    """List all neural network projects in a directory."""
    projects = []

    if directory.is_file():
        console.print(f"[red]❌ {directory} is not a directory[/red]")
        return

    # Find all project.json files
    for root, dirs, files in os.walk(directory):
        if "project.json" in files:
            project_file = Path(root) / "project.json"
            try:
                with open(project_file) as f:
                    project_data = json.load(f)

                project_info = {
                    "name": project_data.get("name", "Unknown"),
                    "path": str(Path(root).relative_to(directory)),
                    "description": project_data.get("description", "No description"),
                    "author": project_data.get("author", "Unknown"),
                    "status": project_data.get("status", "unknown"),
                    "created": project_data.get("created_at", "Unknown"),
                }
                projects.append(project_info)

            except Exception as e:
                console.print(
                    f"[yellow]⚠️  Could not read project at {project_file}: {e}[/yellow]",
                )

    if not projects:
        console.print(f"[yellow]📁 No projects found in {directory}[/yellow]")
        return

    if show_details:
        # Detailed view
        for project in projects:
            details = Text.assemble(
                ("Project: ", "bold blue"),
                (project["name"], "cyan"),
                ("\n📍 Path: ", "dim"),
                (project["path"], "white"),
                ("\n📝 Description: ", "dim"),
                (project["description"], "white"),
                ("\n👤 Author: ", "dim"),
                (project["author"], "white"),
                ("\n📊 Status: ", "dim"),
                (
                    project["status"],
                    "green" if project["status"] == "created" else "yellow",
                ),
                ("\n📅 Created: ", "dim"),
                (project["created"], "white"),
            )

            console.print(
                Panel(
                    details,
                    title=f"[bold]{project['name']}[/bold]",
                    border_style="blue",
                ),
            )
            console.print()  # Empty line between projects
    else:
        # Table view
        table = Table(title="Neural Network Projects")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Path", style="white")
        table.add_column("Author", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Description", style="white", max_width=40)

        for project in sorted(projects, key=lambda x: x["name"]):
            table.add_row(
                project["name"],
                project["path"],
                project["author"],
                project["status"],
                project["description"],
            )

        console.print(table)
