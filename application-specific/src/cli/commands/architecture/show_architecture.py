from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console
from src.cli.utils.file_validator import validate_json_path_param

from .list_layers import list_layers

app = typer.Typer(help="Show the current architecture")


@app.command("show")
def show_architecture(
    file_path: Annotated[
        Path | None,
        typer.Option(
            "--file",
            "-f",
            help="Path to architecture file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            callback=validate_json_path_param,
        ),
    ] = None,
):
    """Show the current architecture or load from file."""
    if file_path:
        arch_manager.load_architecture(file_path)

    architecture = arch_manager.current_architecture

    # Architecture overview
    overview_text = Text.assemble(
        ("Architecture Overview", "bold blue"),
        (f"\n📊 Layers: {len(architecture['layers'])}", "cyan"),
        (
            f"\n📏 Input Size: {architecture['misc_params']['height']}x{architecture['misc_params']['width']}x{architecture['misc_params']['channels']}",
            "green",
        ),
        (f"\n⏱️  Epochs: {architecture['misc_params']['num_epochs']}", "yellow"),
        (f"\n📦 Batch Size: {architecture['misc_params']['batch_size']}", "magenta"),
        (f"\n🖥️  Device: {architecture['misc_params']['device']['value']}", "blue"),
        (f"\n📁   Dataset: {architecture['misc_params']['dataset']['value']}", "dim"),
    )

    console.print(
        Panel(overview_text, title="[bold]Overview[/bold]", border_style="blue"),
    )

    # Show layers if any
    if architecture["layers"]:
        console.print("\n[bold]Layers:[/bold]")
        list_layers()
    else:
        console.print("[yellow]📁 No layers defined[/yellow]")

    # Show configuration status
    config_status = Text()
    config_status.append("\nConfiguration Status:\n", style="bold green")

    optimizer = architecture.get("optimizer", {})
    loss_func = architecture.get("loss_func", {})
    scheduler = architecture.get("scheduler", {})
    pretrained = architecture.get("pretrained", {})

    config_status.append("  • Optimizer: ", style="dim")
    config_status.append(
        "✅ Configured" if optimizer else "❌ Not set",
        style="green" if optimizer else "red",
    )
    config_status.append("\n  • Loss Function: ", style="dim")
    config_status.append(
        "✅ Configured" if loss_func else "❌ Not set",
        style="green" if loss_func else "red",
    )
    config_status.append("\n  • Scheduler: ", style="dim")
    config_status.append(
        "✅ Configured" if scheduler else "❌ Not set",
        style="green" if scheduler else "red",
    )
    config_status.append("\n  • Pretrained Model: ", style="dim")
    config_status.append(
        "✅ Set" if pretrained.get("value") else "❌ Not set",
        style="green" if pretrained.get("value") else "red",
    )

    console.print(config_status)
