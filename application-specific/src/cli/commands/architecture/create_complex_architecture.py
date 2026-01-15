from pathlib import Path

import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(
    help="Create a new complex neural network architecture with pretrained model.",
)


@app.command("create-complex")
def create_complex_architecture(
    name: str = typer.Argument(..., help="Complex architecture name"),
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory for the architecture file",
    ),
    model: str = typer.Argument(
        ...,
        help="Complex pretrained model (e.g., YOLOX-Nano, YOLOX-Small)",
    ),
):
    """Create a new complex neural network architecture with pretrained model."""
    arch_file = output_dir / f"{name}_complex_architecture.json"

    if arch_file.exists():
        console.print(f"[red]❌ Complex architecture '{name}' already exists[/red]")
        return

    # Check if model is available
    if model not in arch_manager.complex_arch_models:
        console.print(f"[red]❌ Complex model '{model}' not available[/red]")
        available = ", ".join(arch_manager.complex_arch_models[:5])
        console.print(f"[blue]ℹ️  Available models: {available}...[/blue]")
        return

    # Create complex architecture
    arch_manager.create_complex_architecture(model)

    # Save the architecture
    arch_manager.save_complex_architecture(arch_file)

    success_text = Text.assemble(
        ("Complex architecture created successfully!", "bold green"),
        (f"\n📁 File: {arch_file}", "cyan"),
        (f"\n🤖 Model: {model}", "blue"),
        (
            f"\n📊 Depth: {arch_manager.current_architecture['pretrained']['depth']}",
            "yellow",
        ),
        (
            f"\n📏 Width: {arch_manager.current_architecture['pretrained']['width']}",
            "yellow",
        ),
    )

    console.print(
        Panel(
            success_text,
            title="[bold]Complex Architecture Created[/bold]",
            border_style="green",
        ),
    )
