from pathlib import Path

import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

from .save_arch import save_arch

app = typer.Typer(help="Create a new neural network architecture.")


@app.command("create")
def create_architecture(
    name: str = typer.Argument(..., help="Architecture name"),
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory for the architecture file",
    ),
    base_model: str | None = typer.Option(
        None,
        "--base",
        "-b",
        help="Base model to start with",
    ),
):
    """Create a new neural network architecture."""
    arch_file = output_dir / f"{name}_architecture.json"

    if arch_file.exists():
        console.print(f"[red]❌ Architecture '{name}' already exists[/red]")
        return

    # Preserve all keys except layers and pretrained
    preserved_config = {}
    for key, value in arch_manager.current_architecture.items():
        if key not in ["layers", "pretrained"]:
            preserved_config[key] = value

    pretrained_index = -1
    if base_model and base_model.lower() in arch_manager.pretrained_models:
        pretrained_index = arch_manager.pretrained_models.index(base_model.lower())

    arch_manager.current_architecture = {
        **preserved_config,
        "layers": [],
        "pretrained": {"value": base_model, "index": pretrained_index},
    }

    # Add base model layers if specified
    if base_model:
        if base_model.lower() in arch_manager.pretrained_models:
            console.print(f"[green]✅ Added base model: {base_model}[/green]")
        else:
            console.print(
                f"[yellow]⚠️  Base model '{base_model}' not found in available models[/yellow]",
            )

    # Save the architecture
    save_arch(arch_file)

    success_text = Text.assemble(
        ("Architecture created successfully!", "bold green"),
        (f"\n📁 File: {arch_file}", "cyan"),
        (f"\n🧠 Base model: {base_model or 'None'}", "blue"),
        ("\n📊 Layers: ", "dim"),
        (str(len(arch_manager.current_architecture["layers"])), "white"),
    )

    console.print(
        Panel(
            success_text,
            title="[bold]Architecture Created[/bold]",
            border_style="green",
        ),
    )
