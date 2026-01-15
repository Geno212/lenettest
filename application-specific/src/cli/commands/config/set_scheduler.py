import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(help="Set the learning rate scheduler configuration")


@app.command("scheduler")
def set_scheduler(
    scheduler_type: str = typer.Argument(
        ...,
        help="Scheduler type (use 'None' for no scheduler)",
    ),
    params: list[str] = typer.Option(
        [],
        "--param",
        "-p",
        help="Scheduler parameters in key=value format",
    ),
):
    """Set the learning rate scheduler configuration."""
    if scheduler_type not in arch_manager.schedulers and scheduler_type != "None":
        console.print(f"[red]❌ Scheduler '{scheduler_type}' not available[/red]")
        available = ", ".join(
            list(arch_manager.schedulers.keys())[:10],
        )  # Show first 10
        console.print(f"[blue]ℹ️  Available schedulers: {available}...[/blue]")
        console.print("[blue]ℹ️  Use 'None' for no scheduler[/blue]")
        return

    # Handle "None" scheduler case
    if scheduler_type == "None":
        arch_manager.current_architecture["scheduler"] = {
            "type": "None",
            "params": {},
        }
        console.print(
            Panel(
                "Scheduler disabled (None)",
                title="[bold]Scheduler[/bold]",
                border_style="green",
            ),
        )
        return

    # Get valid parameter names for this scheduler
    scheduler_info = arch_manager.schedulers[scheduler_type]
    valid_param_names = {param["name"] for param in scheduler_info}

    # Parse parameters
    scheduler_params = {}
    for param in params:
        if "=" in param:
            key, value = param.split("=", 1)

            # Check if parameter is valid for this scheduler
            if key not in valid_param_names:
                console.print(
                    f"[red]❌ Parameter '{key}' is not valid for scheduler '{scheduler_type}'[/red]",
                )
                console.print(
                    f"[blue]ℹ️  Valid parameters: {', '.join(sorted(valid_param_names))}[/blue]",
                )
                return

            # Type validation and conversion
            expected_type = None
            for param_info in scheduler_info:
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
                        scheduler_params[key] = value.lower() == "true"
                    elif expected_type == int:
                        if not value.isdigit() and not (
                            value.startswith("-") and value[1:].isdigit()
                        ):
                            console.print(
                                f"[red]❌ Parameter '{key}' expects integer value, got: {value}[/red]",
                            )
                            return
                        scheduler_params[key] = int(value)
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
                        scheduler_params[key] = float(value)
                    elif value.startswith("[") and value.endswith("]"):
                        scheduler_params[key] = eval(value)
                    else:
                        scheduler_params[key] = value
                except Exception as e:
                    console.print(
                        f"[red]❌ Error converting parameter '{key}' value '{value}': {e!s}[/red]",
                    )
                    return
            else:
                # Fallback conversion
                try:
                    if value.lower() in ["true", "false"]:
                        scheduler_params[key] = value.lower() == "true"
                    elif value.isdigit():
                        scheduler_params[key] = int(value)
                    elif value.replace(".", "", 1).isdigit():
                        scheduler_params[key] = float(value)
                    elif value.startswith("[") and value.endswith("]"):
                        scheduler_params[key] = eval(value)
                    else:
                        scheduler_params[key] = value
                except:
                    scheduler_params[key] = value
        else:
            console.print(
                f"[red]❌ Invalid parameter format: {param}. Use key=value[/red]",
            )
            return

    # Add default values for missing parameters or show error for required parameters without defaults
    missing_params = []
    for param_info in scheduler_info:
        param_name = param_info["name"]
        if param_name not in scheduler_params:
            default_value = param_info.get("defaultvalue")
            if default_value is not None:
                scheduler_params[param_name] = default_value
            else:
                missing_params.append(param_name)

    if missing_params:
        console.print(
            f"[red]❌ Missing required parameters for scheduler '{scheduler_type}': {', '.join(missing_params)}[/red]",
        )
        console.print(
            "[blue]ℹ️  Please provide values for these parameters using --param key=value[/blue]",
        )
        return

    # Set scheduler in architecture
    arch_manager.current_architecture["scheduler"] = {
        "type": scheduler_type,
        "params": scheduler_params,
    }

    scheduler_display = "None" if scheduler_type == "None" else scheduler_type
    success_text = Text.assemble(
        ("Scheduler configured!", "bold green"),
        (f"\n📅 Type: {scheduler_display}", "cyan"),
    )

    if scheduler_params:
        success_text.append("\n⚙️  Parameters:", "dim")
        for key, value in scheduler_params.items():
            success_text.append(f"\n  • {key}: {value}", "white")

    console.print(
        Panel(success_text, title="[bold]Scheduler[/bold]", border_style="green"),
    )
