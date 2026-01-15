from pathlib import Path

import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(help="Save the current complex architecture to a file.")


@app.command("save-complex")
def save_complex_architecture(
    file_path: Path = typer.Argument(
        ...,
        help="Path to save complex architecture file",
    ),
):
    """Save the current complex architecture to a file."""
    # Save complex architecture
    arch_manager.save_complex_architecture(file_path)

    success_text = Text.assemble(
        ("Complex architecture saved successfully!", "bold green"),
        (f"\n📁 File: {file_path}", "cyan"),
        (f"\n📊 Layers: {len(arch_manager.current_architecture['layers'])}", "blue"),
        (
            f"\n🤖 Model: {arch_manager.current_architecture['pretrained'].get('value', 'None')}",
            "yellow",
        ),
    )

    console.print(
        Panel(
            success_text,
            title="[bold]Complex Architecture Saved[/bold]",
            border_style="green",
        ),
    )
