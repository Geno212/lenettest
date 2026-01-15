import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils import Analyze
from src.cli.utils.console import console

app = typer.Typer(help="Add a layer to the current architecture")


@app.command("add-layer")
def add_layer(
    layer_type: str = typer.Argument(..., help="Type of layer to add"),
    params: list[str] = typer.Option(
        [],
        "--param",
        "-p",
        help="Layer parameters in key=value format",
    ),
    position: int | None = typer.Option(
        None,
        "--pos",
        help="Position to insert layer (default: end)",
    ),
):
    """Add a layer to the current architecture."""
    if layer_type not in arch_manager.layers:
        console.print(f"[red]❌ Layer type '{layer_type}' not available[/red]")
        available = ", ".join(
            arch_manager.list_available_layers()[:10],
        )  # Show first 10
        console.print(f"[blue]ℹ️  Available layers: {available}...[/blue]")
        return

    layer_params = {}
    layer_info = arch_manager.get_layer_info(layer_type)

    if layer_info is None:
        console.print(
            f"[red]❌ Could not get layer information for '{layer_type}'[/red]",
        )
        return

    # Get valid parameter names for this layer
    valid_param_names = {param["name"] for param in layer_info}

    for param in params:
        if "=" in param:
            key, value = param.split("=", 1)

            if key not in valid_param_names:
                console.print(
                    f"[red]❌ Parameter '{key}' is not valid for layer type '{layer_type}'[/red]",
                )
                console.print(
                    f"[blue]ℹ️  Valid parameters: {', '.join(sorted(valid_param_names))}[/blue]",
                )
                return

            # Check parameter type matches expected type
            expected_type = None
            for param_info in layer_info:
                if param_info["name"] == key:
                    expected_type = param_info.get("type")
                    break

            if expected_type:
                try:
                    # Try to convert to expected type
                    if expected_type == bool:
                        if value.lower() not in ["true", "false"]:
                            console.print(
                                f"[red]❌ Parameter '{key}' expects boolean value (true/false), got: {value}[/red]",
                            )
                            return
                        converted_value = value.lower() == "true"
                    elif expected_type == int:
                        if not value.isdigit() and not (
                            value.startswith("-") and value[1:].isdigit()
                        ):
                            console.print(
                                f"[red]❌ Parameter '{key}' expects integer value, got: {value}[/red]",
                            )
                            return
                        converted_value = int(value)
                    elif expected_type == float:
                        if not value.replace(".", "", 1).replace(
                            "-",
                            "",
                            1,
                        ).isdigit() and value not in ["inf", "-inf"]:
                            console.print(
                                f"[red]❌ Parameter '{key}' expects float value, got: {value}[/red]",
                            )
                            return
                        converted_value = float(value)
                    # For other types (like str), try to eval if it looks like a list/tuple
                    elif value.startswith("[") and value.endswith("]"):
                        converted_value = eval(value)
                    else:
                        converted_value = value
                    layer_params[key] = converted_value
                except Exception as e:
                    console.print(
                        f"[red]❌ Error converting parameter '{key}' value '{value}': {e!s}[/red]",
                    )
                    return
            else:
                # Fallback to original type conversion if no type info
                try:
                    # Try boolean
                    if value.lower() in ["true", "false"]:
                        layer_params[key] = value.lower() == "true"
                    # Try int
                    elif value.isdigit():
                        layer_params[key] = int(value)
                    # Try float
                    elif value.replace(".", "", 1).isdigit():
                        layer_params[key] = float(value)
                    # Try list (for tuples like [0.9, 0.999])
                    elif value.startswith("[") and value.endswith("]"):
                        layer_params[key] = eval(value)
                    else:
                        layer_params[key] = value
                except:
                    layer_params[key] = value
        else:
            console.print(
                f"[red]❌ Invalid parameter format: {param}. Use key=value[/red]",
            )
            return

    # Validate position parameter
    layers_list = arch_manager.current_architecture["layers"]
    if position is not None:
        if not (0 <= position <= len(layers_list)):
            console.print(
                f"[red]❌ Invalid position {position}. Must be between 0 and {len(layers_list)} (or omit for end)[/red]",
            )
            return
    else:
        position = len(layers_list)  # Default to end

    # Add layer
    layer = arch_manager.add_layer(layer_type, layer_params, position)

    success_text = Text.assemble(
        ("Layer added successfully!", "bold green"),
        (f"\n🧠 Type: {layer_type}", "cyan"),
        (f"\n📍 Position: {position}", "blue"),
        (f"\n⚙️  Parameters: {len(layer_params)}", "yellow"),
    )

    if layer_params:
        success_text.append(
            "\n" + "\n".join(f"  • {k}: {v}" for k, v in layer_params.items()),
        )

    console.print(
        Panel(success_text, title="[bold]Layer Added[/bold]", border_style="green"),
    )
    Analyze(arch_manager.current_architecture["layers"])
