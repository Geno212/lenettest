import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(help="Set optimizer configuration for the architecture")


@app.command("optimizer")
def set_optimizer(
    optimizer_type: str = typer.Argument(..., help="Optimizer type"),
    params: list[str] = typer.Option(
        [],
        "--param",
        "-p",
        help="Optimizer parameters in key=value format",
    ),
):
    """Set the optimizer configuration."""
    if optimizer_type not in arch_manager.optimizers:
        console.print(f"[red]❌ Optimizer '{optimizer_type}' not available[/red]")
        available = ", ".join(
            list(arch_manager.optimizers.keys())[:10],
        )  # Show first 10
        console.print(f"[blue]ℹ️  Available optimizers: {available}...[/blue]")
        return

    # Get valid parameter names for this optimizer
    optimizer_info = arch_manager.optimizers[optimizer_type]
    valid_param_names = {param["name"] for param in optimizer_info}

    # Parse parameters
    optimizer_params = {}
    for param in params:
        if "=" in param:
            key, value = param.split("=", 1)

            # Check if parameter is valid for this optimizer
            if key not in valid_param_names:
                console.print(
                    f"[red]❌ Parameter '{key}' is not valid for optimizer '{optimizer_type}'[/red]",
                )
                console.print(
                    f"[blue]ℹ️  Valid parameters: {', '.join(sorted(valid_param_names))}[/blue]",
                )
                return

            # Type validation and conversion
            expected_type = None
            for param_info in optimizer_info:
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
                        optimizer_params[key] = value.lower() == "true"
                    elif expected_type == int:
                        if not value.isdigit() and not (
                            value.startswith("-") and value[1:].isdigit()
                        ):
                            console.print(
                                f"[red]❌ Parameter '{key}' expects integer value, got: {value}[/red]",
                            )
                            return
                        optimizer_params[key] = int(value)
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
                        optimizer_params[key] = float(value)
                    elif value.startswith("[") and value.endswith("]"):
                        optimizer_params[key] = eval(value)
                    else:
                        optimizer_params[key] = value
                except Exception as e:
                    console.print(
                        f"[red]❌ Error converting parameter '{key}' value '{value}': {e!s}[/red]",
                    )
                    return
            else:
                # Fallback conversion
                try:
                    if value.lower() in ["true", "false"]:
                        optimizer_params[key] = value.lower() == "true"
                    elif value.isdigit():
                        optimizer_params[key] = int(value)
                    elif value.replace(".", "", 1).isdigit():
                        optimizer_params[key] = float(value)
                    elif value.startswith("[") and value.endswith("]"):
                        optimizer_params[key] = eval(value)
                    else:
                        optimizer_params[key] = value
                except:
                    optimizer_params[key] = value
        else:
            console.print(
                f"[red]❌ Invalid parameter format: {param}. Use key=value[/red]",
            )
            return

    # Add default values for missing parameters or show error for required parameters without defaults
    missing_params = []
    for param_info in optimizer_info:
        param_name = param_info["name"]
        if param_name not in optimizer_params:
            default_value = param_info.get("defaultvalue")
            if default_value is not None:
                optimizer_params[param_name] = default_value
            else:
                missing_params.append(param_name)

    if missing_params:
        console.print(
            f"[red]❌ Missing required parameters for optimizer '{optimizer_type}': {', '.join(missing_params)}[/red]",
        )
        console.print(
            "[blue]ℹ️  Please provide values for these parameters using --param key=value[/blue]",
        )
        return

    # Set optimizer in architecture
    arch_manager.current_architecture["optimizer"] = {
        "type": optimizer_type,
        "params": optimizer_params,
    }

    success_text = Text.assemble(
        ("Optimizer configured!", "bold green"),
        (f"\n🔧 Type: {optimizer_type}", "cyan"),
    )

    if optimizer_params:
        success_text.append("\n⚙️  Parameters:", "dim")
        for key, value in optimizer_params.items():
            success_text.append(f"\n  • {key}: {value}", "white")

    console.print(
        Panel(success_text, title="[bold]Optimizer[/bold]", border_style="green"),
    )
