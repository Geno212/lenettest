import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils import Analyze
from src.cli.utils.console import console

app = typer.Typer(help="Remove a residual block from the current architecture")


@app.command("remove-residual-block")
def remove_residual_block(
    index: int = typer.Argument(
        ...,
        help="Index of residual block to remove (0-based)",
    ),
):
    """Remove a residual block from the current architecture."""
    layers_list = arch_manager.current_architecture["layers"]

    if not layers_list:
        console.print("[red]❌ No layers in current architecture[/red]")
        return

    if not (0 <= index < len(layers_list)):
        console.print(
            f"[red]❌ Invalid layer index {index}. Must be between 0 and {len(layers_list) - 1}[/red]",
        )
        return

    layer = layers_list[index]
    if layer["type"] != "Residual_Block":
        console.print(
            f"[red]❌ Layer at index {index} is not a residual block (type: {layer['type']})[/red]",
        )
        return

    removed_block = arch_manager.remove_layer(index)

    success_text = Text.assemble(
        ("Residual block removed successfully!", "bold green"),
        (f"\n🗑️  Type: {removed_block['type']}", "cyan"),
        (f"\n📍 Index: {index}", "blue"),
        (f"\n📦 Internal Layers: {len(removed_block.get('layers', []))}", "yellow"),
    )

    if removed_block.get("layers"):
        layer_types = [l["type"] for l in removed_block["layers"][:5]]  # Show first 5
        success_text.append("\n" + "\n".join(f"  • {lt}" for lt in layer_types))
        if len(removed_block["layers"]) > 5:
            success_text.append(
                f"\n  ... and {len(removed_block['layers']) - 5} more layers",
            )

    console.print(
        Panel(
            success_text,
            title="[bold]Residual Block Removed[/bold]",
            border_style="green",
        ),
    )
    Analyze(arch_manager.current_architecture["layers"])
