import json
from pathlib import Path

import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.utils.console import console

app = typer.Typer(help="Create a new neural network project")


@app.command("create")
def create_project(
    name: str = typer.Argument(..., help="Project name"),
    output_dir: Path = typer.Option(
        Path.cwd(),
        "--output",
        "-o",
        help="Output directory",
    ),
    description: str | None = typer.Option(
        None,
        "--desc",
        "-d",
        help="Project description",
    ),
    author: str | None = typer.Option(None, "--author", "-a", help="Project author"),
):
    """Create a new neural network project."""
    project_dir = output_dir / name

    if project_dir.exists():
        console.print(f"[red]❌ Project '{name}' already exists in {output_dir}[/red]")
        return

    # Create project structure
    try:
        project_dir.mkdir(parents=True)
        (project_dir / "configs").mkdir()
        (project_dir / "architectures").mkdir()
        (project_dir / "outputs").mkdir()
        (project_dir / "logs").mkdir()

        # Create project configuration
        project_config = {
            "name": name,
            "description": description or f"Neural network project: {name}",
            "author": author or "NN Generator CLI",
            "created_at": str(Path.cwd()),
            "version": "1.0.0",
            "architecture": {},
            "config": {},
            "status": "created",
        }

        config_file = project_dir / "project.json"
        with open(config_file, "w") as f:
            json.dump(project_config, f, indent=2)

        # Create default architecture file
        default_arch = {
            "layers": [],
            "misc_params": {
                "width": 224,
                "height": 224,
                "channels": 3,
                "num_epochs": 10,
                "batch_size": 32,
                "device": "cpu",
            },
            "optimizer": {},
            "loss_func": {},
            "scheduler": {},
            "pretrained": {"value": None, "index": -1},
        }

        arch_file = project_dir / "configs" / "default_architecture.json"
        with open(arch_file, "w") as f:
            json.dump(default_arch, f, indent=2)

        # Success message
        success_text = Text.assemble(
            ("Project created successfully!", "bold green"),
            (f"\n📁 Location: {project_dir}", "cyan"),
            ("\n📄 Project file: ", "dim"),
            (str(config_file), "blue"),
            ("\n🏗️  Default architecture: ", "dim"),
            (str(arch_file), "blue"),
        )

        console.print(
            Panel(
                success_text,
                title="[bold]Project Created[/bold]",
                border_style="green",
            ),
        )

    except Exception as e:
        console.print(f"[red]❌ Error creating project: {e}[/red]")
