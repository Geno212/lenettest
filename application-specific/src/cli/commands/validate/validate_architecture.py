from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console
from src.cli.utils.file_validator import validate_json_path_param

app = typer.Typer(help="Validate the current architecture for issues.")


@app.command("architecture")
def validate_architecture(
    architecture_file: Annotated[
        Path,
        typer.Option(
            "--arch",
            "-a",
            help="Path to architecture file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            callback=validate_json_path_param,
        ),
    ],
    strict: bool = typer.Option(
        False,
        "--strict",
        "-s",
        help="Enable strict validation",
    ),
    show_suggestions: bool = typer.Option(
        True,
        "--suggestions",
        help="Show improvement suggestions",
    ),
):
    """Validate the current architecture for issues."""
    # Load architecture if specified
    if architecture_file:
        arch_manager.load_architecture(architecture_file)

    architecture = arch_manager.current_architecture

    console.print("[bold blue]🔍 Validating Architecture...[/bold blue]")

    validation_results = []
    warnings = []
    errors = []

    # Basic validation checks
    layers = architecture.get("layers", {}).get("list", [])

    # Check 1: Layers exist
    if not layers:
        errors.append("❌ No layers defined in architecture")
    else:
        validation_results.append(f"✅ Found {len(layers)} layers")

    # Check 2: Required components
    if not architecture.get("optimizer"):
        errors.append("❌ No optimizer configured")
    else:
        validation_results.append("✅ Optimizer configured")

    if not architecture.get("loss_func"):
        errors.append("❌ No loss function configured")
    else:
        validation_results.append("✅ Loss function configured")

    if (
        not architecture.get("scheduler")
        or architecture.get("scheduler", {}).get("type") == "None"
    ):
        warnings.append("⚠️  No scheduler configured (using default)")
    else:
        validation_results.append("✅ Scheduler configured")

    # Check 3: Model parameters
    misc_params = architecture.get("misc_params", {})

    required_params = [
        "width",
        "height",
        "channels",
        "num_epochs",
        "batch_size",
        "device",
    ]
    for param in required_params:
        if param not in misc_params or misc_params[param] is None:
            errors.append(f"❌ Missing required parameter: {param}")
        else:
            validation_results.append(f"✅ Parameter {param}: {misc_params[param]}")

    # Check 4: Layer connectivity
    if layers:
        # Check for input/output compatibility between layers
        prev_channels = misc_params.get("channels", 3)
        prev_height = misc_params.get("height", 224)
        prev_width = misc_params.get("width", 224)

        for i, layer in enumerate(layers):
            layer_type = layer.get("type", "Unknown")

            # Check layer-specific parameters
            if layer_type == "Conv2d":
                if "in_channels" not in layer.get("params", {}):
                    layer["params"]["in_channels"] = prev_channels
                    warnings.append(
                        f"⚠️  Layer {i + 1} ({layer_type}): Auto-set in_channels to {prev_channels}",
                    )
                if "out_channels" not in layer.get("params", {}):
                    errors.append(
                        f"❌ Layer {i + 1} ({layer_type}): Missing out_channels parameter",
                    )
                else:
                    prev_channels = layer["params"]["out_channels"]

            elif layer_type == "Linear":
                # Calculate input features for Linear layers
                input_features = prev_channels * prev_height * prev_width
                if "in_features" not in layer.get("params", {}):
                    layer["params"]["in_features"] = input_features
                    warnings.append(
                        f"⚠️  Layer {i + 1} ({layer_type}): Auto-set in_features to {input_features}",
                    )
                if "out_features" not in layer.get("params", {}):
                    errors.append(
                        f"❌ Layer {i + 1} ({layer_type}): Missing out_features parameter",
                    )

            elif layer_type in ["MaxPool2d", "AvgPool2d"]:
                # Update spatial dimensions
                kernel_size = layer.get("params", {}).get("kernel_size", 2)
                stride = layer.get("params", {}).get("stride", 2)
                padding = layer.get("params", {}).get("padding", 0)

                prev_height = (prev_height - kernel_size + 2 * padding) // stride + 1
                prev_width = (prev_width - kernel_size + 2 * padding) // stride + 1

                if prev_height <= 0 or prev_width <= 0:
                    errors.append(
                        f"❌ Layer {i + 1} ({layer_type}): Invalid output dimensions ({prev_height}x{prev_width})",
                    )

            elif layer_type == "BatchNorm2d":
                if "num_features" not in layer.get("params", {}):
                    layer["params"]["num_features"] = prev_channels
                    warnings.append(
                        f"⚠️  Layer {i + 1} ({layer_type}): Auto-set num_features to {prev_channels}",
                    )

    # Display results
    console.print(
        Panel(
            "\n".join(validation_results),
            title="[bold]Validation Results[/bold]",
            border_style="green",
        ),
    )

    if errors:
        console.print(
            Panel("\n".join(errors), title="[bold]Errors[/bold]", border_style="red"),
        )

    if warnings:
        console.print(
            Panel(
                "\n".join(warnings),
                title="[bold]Warnings[/bold]",
                border_style="yellow",
            ),
        )

    # Overall status
    if errors:
        console.print(
            Panel(
                "❌ Architecture has errors that need to be fixed",
                title="[bold]Status[/bold]",
                border_style="red",
            ),
        )
        return False
    if warnings:
        console.print(
            Panel(
                "⚠️  Architecture has warnings but is valid",
                title="[bold]Status[/bold]",
                border_style="yellow",
            ),
        )
    else:
        console.print(
            Panel(
                "✅ Architecture is valid",
                title="[bold]Status[/bold]",
                border_style="green",
            ),
        )

    # Suggestions
    if show_suggestions:
        console.print("\n[bold blue]💡 Suggestions for Improvement:[/bold blue]")

        suggestions = []

        # Suggest adding dropout for regularization
        conv_layers = [l for l in layers if l.get("type") == "Conv2d"]
        linear_layers = [l for l in layers if l.get("type") == "Linear"]

        if len(conv_layers) >= 3 and not any(
            l.get("type") == "Dropout2d" for l in layers
        ):
            suggestions.append("• Consider adding Dropout2d layers for regularization")

        # Suggest batch normalization
        if len(conv_layers) >= 2 and not any(
            l.get("type") == "BatchNorm2d" for l in layers
        ):
            suggestions.append(
                "• Consider adding BatchNorm2d layers for better training stability",
            )

        # Suggest data augmentation
        if misc_params.get("num_epochs", 10) > 50:
            suggestions.append(
                "• For long training runs, consider data augmentation techniques",
            )

        # Suggest learning rate scheduling
        if (
            not architecture.get("scheduler")
            or architecture["scheduler"].get("type") == "None"
        ):
            suggestions.append(
                "• Consider using a learning rate scheduler for better convergence",
            )

        if suggestions:
            for suggestion in suggestions:
                console.print(f"  {suggestion}")
        else:
            console.print("  • Architecture looks well-configured!")

    return len(errors) == 0
