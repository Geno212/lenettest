from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console
from src.cli.utils.file_validator import validate_json_path_param

app = typer.Typer(help="Load an architecture from a file")


@app.command("load")
def load_arch(
    file_path: Annotated[
        Path,
        typer.Argument(
            help="Path to architecture file to load",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            callback=validate_json_path_param,
        ),
    ],
):
    """Load an architecture from a file."""
    arch_manager.load_architecture(file_path)

    success_text = Text.assemble(
        ("Architecture loaded successfully!", "bold green"),
        (f"\n📁 File: {file_path}", "cyan"),
        (f"\n📊 Layers: {len(arch_manager.current_architecture['layers'])}", "blue"),
    )

    console.print(
        Panel(
            success_text,
            title="[bold]Architecture Loaded[/bold]",
            border_style="green",
        ),
    )
