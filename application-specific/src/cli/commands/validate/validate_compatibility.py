from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console
from src.cli.utils.file_validator import validate_json_path_param

app = typer.Typer(help="Validate architecture compatibility with target hardware")


@app.command("compatibility")
def validate_compatibility(
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
    target_hardware: str = typer.Option(
        "gpu",
        "--hardware",
        "-h",
        help="Target hardware (cpu, gpu, edge)",
    ),
):
    """Validate architecture compatibility with target hardware."""
    # Load architecture if specified
    if architecture_file:
        arch_manager.load_architecture(architecture_file)

    architecture = arch_manager.current_architecture
    layers = architecture.get("layers", {}).get("list", [])
    misc_params = architecture.get("misc_params", {})

    if not layers:
        console.print("[red]❌ No layers defined in architecture[/red]")
        return

    console.print(
        f"[bold blue]🔧 Checking {target_hardware.upper()} Compatibility...[/bold blue]",
    )

    compatibility_issues = []
    recommendations = []

    # Calculate model statistics
    total_params = 0
    channels = misc_params.get("channels", 3)
    height = misc_params.get("height", 224)
    width = misc_params.get("width", 224)

    for layer in layers:
        layer_type = layer.get("type", "Unknown")
        params = layer.get("params", {})

        if layer_type == "Conv2d":
            in_ch = params.get("in_channels", channels)
            out_ch = params.get("out_channels", 64)
            kernel = params.get("kernel_size", 3)

            if isinstance(kernel, int):
                kernel_h = kernel_w = kernel
            else:
                kernel_h, kernel_w = kernel

            conv_params = in_ch * out_ch * kernel_h * kernel_w
            total_params += conv_params

            # Update dimensions
            stride = params.get("stride", 1)
            padding = params.get("padding", 0)
            height = (height - kernel_h + 2 * padding) // stride + 1
            width = (width - kernel_w + 2 * padding) // stride + 1
            channels = out_ch

        elif layer_type == "Linear":
            in_features = params.get("in_features", channels * height * width)
            out_features = params.get("out_features", 1000)
            total_params += in_features * out_features

    # Hardware-specific checks
    if target_hardware.lower() == "cpu":
        if total_params > 100000000:  # > 100M parameters
            compatibility_issues.append(
                "❌ Model too large for CPU training (very slow)",
            )
            recommendations.append("• Use model compression or smaller architecture")
        elif total_params > 50000000:  # > 50M parameters
            compatibility_issues.append("⚠️  Large model for CPU - training may be slow")
            recommendations.append("• Consider using GPU for faster training")

        if misc_params.get("batch_size", 32) > 64:
            compatibility_issues.append(
                "⚠️  Large batch size may cause memory issues on CPU",
            )
            recommendations.append("• Reduce batch size for CPU training")

    elif target_hardware.lower() == "gpu":
        if misc_params.get("batch_size", 32) < 16:
            recommendations.append(
                "• Consider increasing batch size for better GPU utilization",
            )

        if total_params > 1000000000:  # > 1B parameters
            compatibility_issues.append("⚠️  Very large model - may need multiple GPUs")
            recommendations.append(
                "• Consider model parallelism or gradient checkpointing",
            )

    elif target_hardware.lower() == "edge":
        if total_params > 5000000:  # > 5M parameters
            compatibility_issues.append("❌ Model too large for edge deployment")
            recommendations.append("• Use model compression, quantization, or pruning")

        if height * width > 50 * 50:
            compatibility_issues.append("⚠️  Large input resolution for edge device")
            recommendations.append("• Consider reducing input resolution")

        # Check for unsupported operations on edge devices
        unsupported_layers = []
        for layer in layers:
            layer_type = layer.get("type", "Unknown")
            if layer_type in ["BatchNorm2d", "Dropout2d", "MaxPool2d"]:
                unsupported_layers.append(layer_type)

        if unsupported_layers:
            compatibility_issues.append(
                f"⚠️  Edge devices may not support: {', '.join(unsupported_layers)}",
            )
            recommendations.append("• Replace with edge-optimized alternatives")

    # Display results
    if compatibility_issues:
        console.print(
            Panel(
                "\n".join(compatibility_issues),
                title="[bold]Compatibility Issues[/bold]",
                border_style="red",
            ),
        )

    if recommendations:
        console.print(
            Panel(
                "\n".join(recommendations),
                title="[bold]Recommendations[/bold]",
                border_style="yellow",
            ),
        )

    if not compatibility_issues:
        console.print(
            Panel(
                "✅ Architecture is compatible with target hardware",
                title="[bold]Compatibility Status[/bold]",
                border_style="green",
            ),
        )

    # Hardware requirements summary
    hardware_reqs = Text.assemble(
        ("Hardware Requirements", "bold green"),
        (f"\n💾 Memory: ~{total_params * 4 / (1024 * 1024):.1f} MB", "cyan"),
        (f"\n🖥️  Target: {target_hardware.upper()}", "blue"),
        (f"\n📊 Parameters: {total_params:,}", "yellow"),
        (
            f"\n📏 Input Size: {misc_params.get('height', 224)}×{misc_params.get('width', 224)}",
            "white",
        ),
    )

    console.print(Panel(hardware_reqs, border_style="green"))
