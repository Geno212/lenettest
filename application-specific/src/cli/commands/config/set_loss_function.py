import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(help="Set the loss function for the current architecture")


@app.command("loss")
def set_loss_function(
    loss_type: str = typer.Argument(..., help="Loss function type"),
    params: list[str] = typer.Option(
        [],
        "--param",
        "-p",
        help="Loss function parameters in key=value format",
    ),
):
    """Set the loss function configuration."""
    if loss_type not in arch_manager.loss_funcs:
        console.print(f"[red]❌ Loss function '{loss_type}' not available[/red]")
        available = ", ".join(
            list(arch_manager.loss_funcs.keys())[:10],
        )  # Show first 10
        console.print(f"[blue]ℹ️  Available loss functions: {available}...[/blue]")
        return

    # Get valid parameter names for this loss function
    loss_info = arch_manager.loss_funcs[loss_type]
    valid_param_names = {param["name"] for param in loss_info}

    # Parse parameters
    loss_params = {}
    for param in params:
        if "=" in param:
            key, value = param.split("=", 1)

            # Check if parameter is valid for this loss function
            if key not in valid_param_names:
                console.print(
                    f"[red]❌ Parameter '{key}' is not valid for loss function '{loss_type}'[/red]",
                )
                console.print(
                    f"[blue]ℹ️  Valid parameters: {', '.join(sorted(valid_param_names))}[/blue]",
                )
                return

            # Type validation and conversion
            expected_type = None
            for param_info in loss_info:
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
                        loss_params[key] = value.lower() == "true"
                    elif expected_type == int:
                        if not value.isdigit() and not (
                            value.startswith("-") and value[1:].isdigit()
                        ):
                            console.print(
                                f"[red]❌ Parameter '{key}' expects integer value, got: {value}[/red]",
                            )
                            return
                        loss_params[key] = int(value)
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
                        loss_params[key] = float(value)
                    elif value.startswith("[") and value.endswith("]"):
                        loss_params[key] = eval(value)
                    else:
                        loss_params[key] = value
                except Exception as e:
                    console.print(
                        f"[red]❌ Error converting parameter '{key}' value '{value}': {e!s}[/red]",
                    )
                    return
            else:
                # Fallback conversion
                try:
                    if value.lower() in ["true", "false"]:
                        loss_params[key] = value.lower() == "true"
                    elif value.isdigit():
                        loss_params[key] = int(value)
                    elif value.replace(".", "", 1).isdigit():
                        loss_params[key] = float(value)
                    elif value.startswith("[") and value.endswith("]"):
                        loss_params[key] = eval(value)
                    else:
                        loss_params[key] = value
                except:
                    loss_params[key] = value
        else:
            console.print(
                f"[red]❌ Invalid parameter format: {param}. Use key=value[/red]",
            )
            return

    # Add default values for missing parameters or show error for required parameters without defaults
    missing_params = []
    for param_info in loss_info:
        param_name = param_info["name"]
        if param_name not in loss_params:
            default_value = param_info.get("defaultvalue")
            if default_value is not None:
                if isinstance(default_value, int) and default_value < 0:
                    loss_params[param_name] = 1
                elif isinstance(default_value, float) and default_value < 0:
                    loss_params[param_name] = 1.0
                else:
                    loss_params[param_name] = default_value
            else:
                missing_params.append(param_name)

    if missing_params:
        console.print(
            f"[red]❌ Missing required parameters for loss function '{loss_type}': {', '.join(missing_params)}[/red]",
        )
        console.print(
            "[blue]ℹ️  Please provide values for these parameters using --param key=value[/blue]",
        )
        return

    # Set loss function in architecture
    arch_manager.current_architecture["loss_func"] = {
        "type": loss_type,
        "params": loss_params,
    }

    success_text = Text.assemble(
        ("Loss function configured!", "bold green"),
        (f"\n📉 Type: {loss_type}", "cyan"),
    )

    if loss_params:
        success_text.append("\n⚙️  Parameters:", "dim")
        for key, value in loss_params.items():
            success_text.append(f"\n  • {key}: {value}", "white")

    console.print(
        Panel(success_text, title="[bold]Loss Function[/bold]", border_style="green"),
    )
