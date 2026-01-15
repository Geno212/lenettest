import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils import Analyze
from src.cli.utils.console import console

app = typer.Typer(help="Add a residual block to the current architecture")


@app.command("add-residual-block")
def add_residual_block(
    in_channels: int = typer.Argument(..., help="Input channels for residual block"),
    out_channels: int = typer.Argument(..., help="Output channels for residual block"),
    layers: list[str] = typer.Argument(
        ...,
        help="Layer types to include in residual block (e.g., Conv2d BatchNorm2d ReLU)",
    ),
    layer_params: list[str] = typer.Option(
        [],
        "--param",
        "-p",
        help="Parameters for layers in residual block",
    ),
):
    """Add a residual block to the current architecture."""
    # Validate input parameters
    if in_channels <= 0 or out_channels <= 0:
        console.print(
            "[red]❌ Input and output channels must be positive integers[/red]",
        )
        return

    # Validate that specified layers are available
    available_layers = arch_manager.list_available_layers()
    invalid_layers = [layer for layer in layers if layer not in available_layers]
    if invalid_layers:
        console.print(f"[red]❌ Invalid layer types: {', '.join(invalid_layers)}[/red]")
        console.print(
            f"[blue]ℹ️  Available layers: {', '.join(available_layers[:10])}...[/blue]",
        )
        return

    # Parse layer parameters
    residual_layers = []
    current_param_index = 0

    for layer_type in layers:
        layer_info = arch_manager.get_layer_info(layer_type)
        if not layer_info:
            console.print(
                f"[red]❌ Could not get layer information for '{layer_type}'[/red]",
            )
            return

        valid_param_names = {param["name"] for param in layer_info}
        layer_params_dict = {}

        # Get parameters for this specific layer
        params_for_this_layer = []
        if current_param_index < len(layer_params):
            # Look for parameters that start with layer_type:
            for i in range(current_param_index, len(layer_params)):
                if ":" in layer_params[i]:
                    layer_name, param = layer_params[i].split(":", 1)
                    if layer_name == layer_type:
                        params_for_this_layer.append(param)
                        current_param_index = i + 1
                    else:
                        break
                else:
                    current_param_index = i + 1
                    break

        # Process parameters for this layer
        for param in params_for_this_layer:
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

                # Type validation and conversion
                expected_type = None
                for param_info in layer_info:
                    if param_info["name"] == key:
                        expected_type = param_info.get("type")
                        break

                if expected_type:
                    try:
                        if expected_type == bool:
                            if value.lower() not in ["true", "false"]:
                                console.print(
                                    f"[red]❌ Parameter '{key}' expects boolean value (true/false), got: {value}[/red]",
                                )
                                return
                            layer_params_dict[key] = value.lower() == "true"
                        elif expected_type == int:
                            if not value.isdigit() and not (
                                value.startswith("-") and value[1:].isdigit()
                            ):
                                console.print(
                                    f"[red]❌ Parameter '{key}' expects integer value, got: {value}[/red]",
                                )
                                return
                            layer_params_dict[key] = int(value)
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
                            layer_params_dict[key] = float(value)
                        elif value.startswith("[") and value.endswith("]"):
                            layer_params_dict[key] = eval(value)
                        else:
                            layer_params_dict[key] = value
                    except Exception as e:
                        console.print(
                            f"[red]❌ Error converting parameter '{key}' value '{value}': {e!s}[/red]",
                        )
                        return
                else:
                    # Fallback conversion
                    try:
                        if value.lower() in ["true", "false"]:
                            layer_params_dict[key] = value.lower() == "true"
                        elif value.isdigit():
                            layer_params_dict[key] = int(value)
                        elif value.replace(".", "", 1).isdigit():
                            layer_params_dict[key] = float(value)
                        elif value.startswith("[") and value.endswith("]"):
                            layer_params_dict[key] = eval(value)
                        else:
                            layer_params_dict[key] = value
                    except:
                        layer_params_dict[key] = value

        # Create the layer
        layer = {
            "type": layer_type,
            "params": layer_params_dict,
            "name": f"{layer_type.lower()}_{len(residual_layers) + 1}",
        }
        residual_layers.append(layer)

    # Create residual block
    residual_block = {
        "type": "Residual_Block",
        "params": {
            "in_channels": in_channels,
            "out_channels": out_channels,
        },
        "layers": residual_layers,
        "name": f"residual_block_{len(arch_manager.current_architecture['layers']) + 1}",
    }

    # Add residual block to architecture
    arch_manager.current_architecture["layers"].append(residual_block)

    success_text = Text.assemble(
        ("Residual block added successfully!", "bold green"),
        ("\n🧠 Type: Residual_Block", "cyan"),
        (f"\n📍 Position: {len(arch_manager.current_architecture['layers'])}", "blue"),
        (f"\n🔗 Input Channels: {in_channels}", "yellow"),
        (f"\n🔗 Output Channels: {out_channels}", "yellow"),
        (f"\n📦 Internal Layers: {len(residual_layers)}", "magenta"),
    )

    if residual_layers:
        success_text.append(
            "\n" + "\n".join(f"  • {layer['type']}" for layer in residual_layers),
        )

    console.print(
        Panel(
            success_text,
            title="[bold]Residual Block Added[/bold]",
            border_style="green",
        ),
    )
    Analyze(arch_manager.current_architecture["layers"])
