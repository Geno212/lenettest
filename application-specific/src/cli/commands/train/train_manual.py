import json
from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel

from src.cli.utils.console import console
from src.cli.utils.file_validator import validate_json_path_param
from src.cli.utils.run_training_with_gui_flow import run_training_with_gui_flow

app = typer.Typer(help="Train a manual (custom architecture) model")


@app.command("manual")
def train_manual(
    config_file: Annotated[
        Path,
        typer.Argument(
            help="Path to manual training JSON configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            callback=validate_json_path_param,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for manual training",
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ] = Path("./outputs/manual"),
    verbose: Annotated[  # noqa: FBT002
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """Train a manual (custom architecture) model using JSON configuration."""
    console.print(Panel.fit("🔧 Manual Training", style="bold blue"))

    # Load configuration file
    with Path.open(config_file) as f:
        config = json.load(f)

    console.print(f"[green]✅ Config file: {config_file}[/green]")
    console.print(f"[blue]📂 Output directory: {output_dir}[/blue]")

    # Run training using GUI flow
    success = run_training_with_gui_flow(config, output_dir, "manual", verbose)

    if success:
        console.print(
            Panel.fit("✅ Manual training completed successfully!", style="bold green"),
        )
    else:
        console.print(Panel.fit("❌ Manual training failed", style="bold red"))
        raise typer.Exit(1)
