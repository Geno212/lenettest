from pathlib import Path

import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(help="Save the current architecture to a file.")


@app.command("save")
def save_arch(
    file_path: Path = typer.Argument(..., help="Path to save architecture file"),
):
    """Save the current architecture to a file."""
    # Validate and prepare architecture using validation logic
    try:
        # Get current architecture for validation
        current_arch = arch_manager.current_architecture.copy()

        # Apply validation and model loading logic
        validated_arch = arch_manager.validate_architecture(current_arch)
        prepared_arch = arch_manager.prepare_architecture(validated_arch)

        # Update the architecture manager with validated architecture
        arch_manager.current_architecture = prepared_arch

    except Exception as e:
        console.print(
            f"[yellow]⚠️  Validation/Model loading failed: {e}. Saving without validation.[/yellow]",
        )

    arch_manager.save_architecture(file_path)

    success_text = Text.assemble(
        ("Architecture saved successfully!", "bold green"),
        (f"\n📁 File: {file_path}", "cyan"),
        (f"\n📊 Layers: {len(arch_manager.current_architecture['layers'])}", "blue"),
    )

    console.print(
        Panel(
            success_text,
            title="[bold]Architecture Saved[/bold]",
            border_style="green",
        ),
    )
