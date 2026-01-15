from pathlib import Path
from typing import Annotated

import typer

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console
from src.cli.utils.file_validator import validate_json_path_param

app = typer.Typer(help="Validate individual layers or specific layer configurations.")


@app.command("layers")
def validate_layers(
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
    layer_index: int | None = typer.Option(
        None,
        "--layer",
        "-l",
        help="Validate specific layer (0-based index)",
    ),
):
    """Validate individual layers or specific layer configurations."""
    # Load architecture if specified
    if architecture_file:
        arch_manager.load_architecture(architecture_file)

    architecture = arch_manager.current_architecture
    layers = architecture.get("layers", {}).get("list", [])

    if not layers:
        console.print("[red]❌ No layers defined in architecture[/red]")
        return

    if layer_index is not None:
        if not (0 <= layer_index < len(layers)):
            console.print(
                f"[red]❌ Invalid layer index {layer_index}. Must be between 0 and {len(layers) - 1}[/red]",
            )
            return
        layers = [layers[layer_index]]

    console.print(f"[bold blue]🔍 Validating {len(layers)} layer(s)...[/bold blue]")

    for i, layer in enumerate(layers):
        if layer_index is not None:
            display_index = layer_index
        else:
            display_index = i

        layer_type = layer.get("type", "Unknown")
        params = layer.get("params", {})

        console.print(f"\n[bold cyan]Layer {display_index}: {layer_type}[/bold cyan]")

        # Layer-specific validation
        if layer_type == "Conv2d":
            issues = []

            if "in_channels" not in params:
                issues.append("❌ Missing in_channels parameter")
            if "out_channels" not in params:
                issues.append("❌ Missing out_channels parameter")
            if "kernel_size" not in params:
                issues.append("❌ Missing kernel_size parameter")

            if not issues:
                issues.append("✅ All required parameters present")

            # Check parameter values
            if "kernel_size" in params:
                ks = params["kernel_size"]
                if isinstance(ks, int) and (ks <= 0 or ks > 15):
                    issues.append("⚠️  kernel_size seems unusual (recommended: 1-7)")
                elif isinstance(ks, (list, tuple)) and (
                    len(ks) != 2 or ks[0] <= 0 or ks[1] <= 0
                ):
                    issues.append("⚠️  kernel_size tuple seems invalid")

            if "stride" in params and params["stride"] <= 0:
                issues.append("⚠️  stride should be positive")

            for issue in issues:
                console.print(f"  {issue}")

        elif layer_type == "Linear":
            issues = []

            if "in_features" not in params:
                issues.append("❌ Missing in_features parameter")
            if "out_features" not in params:
                issues.append("❌ Missing out_features parameter")

            if not issues:
                issues.append("✅ All required parameters present")

            # Check parameter values
            if "in_features" in params and params["in_features"] <= 0:
                issues.append("⚠️  in_features should be positive")
            if "out_features" in params and params["out_features"] <= 0:
                issues.append("⚠️  out_features should be positive")

            for issue in issues:
                console.print(f"  {issue}")

        elif layer_type in ["MaxPool2d", "AvgPool2d"]:
            issues = []

            if "kernel_size" not in params:
                issues.append("❌ Missing kernel_size parameter")

            if not issues:
                issues.append("✅ All required parameters present")

            # Check parameter values
            if "kernel_size" in params:
                ks = params["kernel_size"]
                if isinstance(ks, int) and (ks <= 0 or ks > 10):
                    issues.append("⚠️  kernel_size seems unusual (recommended: 2-5)")
                elif isinstance(ks, (list, tuple)) and (
                    len(ks) != 2 or ks[0] <= 0 or ks[1] <= 0
                ):
                    issues.append("⚠️  kernel_size tuple seems invalid")

            for issue in issues:
                console.print(f"  {issue}")

        else:
            console.print(f"  ℹ️  Layer type '{layer_type}' - basic validation only")
            console.print(f"  ✅ Layer exists with {len(params)} parameters")
