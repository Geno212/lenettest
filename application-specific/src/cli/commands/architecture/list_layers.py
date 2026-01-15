import typer
from rich.table import Table

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(help="List all layers in the current architecture")


@app.command("list-layers")
def list_layers():
    """List all layers in the current architecture."""
    layers_list = arch_manager.current_architecture["layers"]

    if not layers_list:
        console.print("[yellow]📁 No layers in current architecture[/yellow]")
        return

    table = Table(title="Current Architecture Layers")
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("Type", style="green", no_wrap=True)
    table.add_column("Parameters", style="white", max_width=50)
    table.add_column("Name", style="yellow")

    for i, layer in enumerate(layers_list):
        params_str = ", ".join(
            [f"{k}={v}" for k, v in list(layer.get("params", {}).items())[:3]],
        )
        if len(layer.get("params", {})) > 3:
            params_str += "..."

        # Handle residual blocks specially
        layer_type = layer["type"]
        if layer_type == "Residual_Block":
            internal_layers = layer.get("layers", [])
            params_str = f"Internal: {len(internal_layers)} layers"
            if internal_layers:
                layer_types = [l["type"] for l in internal_layers[:3]]  # Show first 3
                params_str += f" ({', '.join(layer_types)}"
                if len(internal_layers) > 3:
                    params_str += "..."
                params_str += ")"

        table.add_row(
            str(i),
            layer_type,
            params_str,
            layer.get("name", "unnamed"),
        )

    console.print(table)
