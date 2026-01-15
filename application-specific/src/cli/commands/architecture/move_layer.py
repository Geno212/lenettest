import typer

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils import Analyze
from src.cli.utils.console import console

app = typer.Typer(help="Move a layer within the current architecture")


@app.command("move-layer")
def move_layer(
    from_index: int = typer.Argument(..., help="Current index of layer"),
    to_index: int = typer.Argument(..., help="New index for layer"),
):
    """Move a layer from one position to another."""
    layers_list = arch_manager.current_architecture["layers"]

    if not layers_list:
        console.print("[red]❌ No layers in current architecture[/red]")
        return

    if not (0 <= from_index < len(layers_list)):
        console.print(
            f"[red]❌ Invalid from_index {from_index}. Must be between 0 and {len(layers_list) - 1}[/red]",
        )
        return

    if not (0 <= to_index < len(layers_list)):
        console.print(
            f"[red]❌ Invalid to_index {to_index}. Must be between 0 and {len(layers_list) - 1}[/red]",
        )
        return

    if from_index == to_index:
        console.print("[yellow]⚠️  Layer is already at the target position[/yellow]")
        return

    success = arch_manager.move_layer(from_index, to_index)

    if success:
        console.print(
            f"[green]✅ Layer moved from position {from_index} to {to_index}[/green]",
        )
    else:
        console.print("[red]❌ Failed to move layer[/red]")
    Analyze(arch_manager.current_architecture["layers"])
