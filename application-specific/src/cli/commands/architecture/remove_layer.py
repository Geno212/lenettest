import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils import Analyze
from src.cli.utils.console import console

app = typer.Typer(help="Remove a layer from the current architecture")


@app.command("remove-layer")
def remove_layer(
    index: int = typer.Argument(..., help="Index of layer to remove (0-based)"),
):
    """Remove a layer from the current architecture."""
    layers_list = arch_manager.current_architecture["layers"]

    if not layers_list:
        console.print("[red]❌ No layers in current architecture[/red]")
        return

    if not (0 <= index < len(layers_list)):
        console.print(
            f"[red]❌ Invalid layer index {index}. Must be between 0 and {len(layers_list) - 1}[/red]",
        )
        return

    removed_layer = arch_manager.remove_layer(index)

    success_text = Text.assemble(
        ("Layer removed successfully!", "bold green"),
        (f"\n🗑️  Type: {removed_layer['type']}", "cyan"),
        (f"\n📍 Index: {index}", "blue"),
    )

    console.print(
        Panel(success_text, title="[bold]Layer Removed[/bold]", border_style="green"),
    )
    Analyze(arch_manager.current_architecture["layers"])
