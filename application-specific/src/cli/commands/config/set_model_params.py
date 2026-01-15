from pathlib import Path

import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils import get_available_devices
from src.cli.utils.console import console

app = typer.Typer(help="Set model parameters.")


@app.command("model-params")
def set_model_params(
    height: int | None = typer.Option(None, "--height", "-h", help="Input height"),
    width: int | None = typer.Option(None, "--width", "-w", help="Input width"),
    channels: int | None = typer.Option(
        None,
        "--channels",
        "-c",
        help="Input channels",
    ),
    epochs: int | None = typer.Option(
        None,
        "--epochs",
        "-e",
        help="Number of epochs",
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Batch size",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        "-d",
        help="Device (cpu, cuda, cuda:0, etc.)",
    ),
    dataset: str = typer.Option(..., "--dataset", help="Dataset name"),
    dataset_path: Path = typer.Option(..., "--dataset-path", help="Path to dataset"),
):
    """Set model parameters."""
    architecture = arch_manager.current_architecture
    misc_params = architecture["misc_params"]

    # Update parameters
    updates = []
    if height is not None:
        misc_params["height"] = height
        updates.append(f"height={height}")
    if width is not None:
        misc_params["width"] = width
        updates.append(f"width={width}")
    if channels is not None:
        misc_params["channels"] = channels
        updates.append(f"channels={channels}")
    if epochs is not None:
        misc_params["num_epochs"] = epochs
        updates.append(f"epochs={epochs}")
    if batch_size is not None:
        misc_params["batch_size"] = batch_size
        updates.append(f"batch_size={batch_size}")
    if device is not None:
        device_index = -1
        devices = get_available_devices()
        if device in devices:
            device_index = devices.index(device)
        misc_params["device"] = {"value": device, "index": device_index}
    if dataset is not None:
        dataset_index = -1
        if dataset in arch_manager.datasets:
            dataset_index = arch_manager.datasets.index(dataset)
        misc_params["dataset"] = {"value": dataset, "index": dataset_index}
        updates.append(f"dataset={dataset}")
    if dataset_path is not None:
        misc_params["dataset_path"] = str(dataset_path)
        updates.append(f"dataset_path={dataset_path}")

    if not updates:
        console.print(
            "[yellow]⚠️  No parameters specified. Use --help to see available options.[/yellow]",
        )
        return

    success_text = Text.assemble(
        ("Model parameters updated!", "bold green"),
        ("\n✅ Updated:", "cyan"),
    )
    for update in updates:
        success_text.append(f"\n  • {update}", "white")

    console.print(
        Panel(
            success_text,
            title="[bold]Model Parameters[/bold]",
            border_style="green",
        ),
    )
