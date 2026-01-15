#!/usr/bin/env python3
"""Neural Network Generator CLI Tool

A command-line interface for the Application-Specific Deep Learning Accelerator Designer.
This tool provides all the functionality of the GUI version through a comprehensive CLI.

"""

import sys
from pathlib import Path

from .commands.architecture import app as architecture_app
from .commands.config import app as config_app
from .commands.generate import app as generate_app
from .commands.project import app as project_app
from .commands.tensorboard import app as tensorboard_app
from .commands.test import app as test_app
from .commands.train import app as train_app
from .commands.validate import app as validate_app

# Add the src directory to Python path for proper imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))


import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.utils.console import console

# Import command groups

# Create the main CLI app
app = typer.Typer(
    name="nn-generator",
    help="Neural Network Generator - CLI tool for deep learning accelerator design",
    add_completion=False,
    rich_markup_mode="rich",
)

# Add command groups
app.add_typer(project_app, name="project", help="Project management commands")
app.add_typer(architecture_app, name="arch", help="Architecture design commands")
app.add_typer(config_app, name="config", help="Configuration and parameters commands")
app.add_typer(train_app, name="train", help="Model training commands")
app.add_typer(generate_app, name="gen", help="Code generation commands")
app.add_typer(validate_app, name="validate", help="Architecture validation commands")
app.add_typer(test_app, name="test", help="Model testing commands")
app.add_typer(
    tensorboard_app,
    name="tensorboard",
    help="TensorBoard management commands",
)

from cli.shell import main as shell_main


@app.command()
def shell():
    """Start interactive shell for running multiple commands in a single session."""
    console.print("[bold blue]🐚 Starting NN Generator Shell...[/bold blue]")
    shell_main()


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    config_file: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    """Neural Network Generator CLI

    A comprehensive command-line tool for designing, configuring, and generating
    neural network accelerators with full PyTorch and SystemC support.
    """
    # Store settings in context for use by subcommands
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug
    ctx.obj["config_file"] = config_file

    if verbose:
        console.print("[bold blue]🔧 Verbose mode enabled[/bold blue]")
    if debug:
        console.print("[bold yellow]🐛 Debug mode enabled[/bold yellow]")


@app.command()
def gui():
    """Launch the GUI application."""
    console.print(
        "[bold green]🖥️  Launching Neural Network Generator GUI...[/bold green]",
    )
    launch_gui()


def launch_gui():
    """Launch the GUI application."""
    try:
        # Import and run the GUI main function
        from src.main import main as gui_main

        gui_main()
    except ImportError as e:
        console.print(f"[red]❌ Failed to import GUI: {e}[/red]")
        console.print(
            "[yellow]⚠️  Make sure PySide6 and GUI dependencies are installed[/yellow]",
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ Failed to launch GUI: {e}[/red]")
        sys.exit(1)


@app.command()
def version():
    """Show version information."""
    version_info = Text.assemble(
        ("Neural Network Generator CLI", "bold blue"),
        ("\nVersion: ", "dim"),
        ("1.0.0", "green"),
        ("\nBuilt with: ", "dim"),
        ("Typer + Rich", "cyan"),
    )

    console.print(
        Panel(
            version_info,
            title="[bold]NN Generator CLI[/bold]",
            border_style="blue",
        ),
    )


@app.command()
def info():
    """Show system information and available features."""
    try:
        from cli.utils.system import get_system_info

        system_info = get_system_info()
    except ImportError:
        # Fallback if system utils not available
        import platform

        system_info = {
            "os": platform.system(),
            "python_version": platform.python_version(),
            "torch_version": "Not available",
            "cuda_available": False,
            "gpu_count": 0,
        }

    info_text = Text.assemble(
        ("System Information", "bold blue"),
        ("\nOS: ", "dim"),
        (system_info["os"], "green"),
        ("\nPython: ", "dim"),
        (system_info["python_version"], "green"),
        ("\nPyTorch: ", "dim"),
        (system_info["torch_version"], "green"),
        ("\nCUDA Available: ", "dim"),
        (
            str(system_info["cuda_available"]),
            "yellow" if system_info["cuda_available"] else "red",
        ),
        ("\nGPU Count: ", "dim"),
        (str(system_info["gpu_count"]), "cyan"),
    )

    console.print(
        Panel(
            info_text,
            title="[bold]System Info[/bold]",
            border_style="blue",
        ),
    )

    features_text = Text.assemble(
        ("Available Features", "bold green"),
        ("\n• Project Management", "cyan"),
        ("\n• Architecture Design", "cyan"),
        ("\n• Model Configuration", "cyan"),
        ("\n• Training & Validation", "cyan"),
        ("\n• Code Generation (PyTorch + SystemC)", "cyan"),
        ("\n• Transfer Learning", "cyan"),
        ("\n• Model Testing", "cyan"),
        ("\n• Pretrained Models (YOLOX, Torchvision)", "cyan"),
        ("\n• GUI Mode", "magenta"),
    )

    console.print(
        Panel(
            features_text,
            title="[bold]Features[/bold]",
            border_style="green",
        ),
    )


if __name__ == "__main__":
    app()
