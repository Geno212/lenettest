from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console
from src.cli.utils.file_validator import validate_json_path_param

app = typer.Typer(help="Analyze architecture performance characteristics.")


@app.command("performance")
def validate_performance(
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
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Override batch size for analysis",
    ),
    input_size: str | None = typer.Option(
        None,
        "--input-size",
        "-i",
        help="Override input size (HxWxC)",
    ),
):
    """Analyze architecture performance characteristics."""
    # Load architecture if specified
    if architecture_file:
        arch_manager.load_architecture(architecture_file)

    architecture = arch_manager.current_architecture
    layers = architecture.get("layers", {}).get("list", [])
    misc_params = architecture.get("misc_params", {})

    if not layers:
        console.print("[red]❌ No layers defined in architecture[/red]")
        return

    # Override parameters if specified
    if batch_size:
        misc_params["batch_size"] = batch_size
    if input_size:
        try:
            h, w, c = map(int, input_size.split("x"))
            misc_params["height"] = h
            misc_params["width"] = w
            misc_params["channels"] = c
        except:
            console.print(
                "[red]❌ Invalid input size format. Use HxWxC (e.g., 224x224x3)[/red]",
            )
            return

    console.print("[bold blue]📊 Analyzing Performance Characteristics...[/bold blue]")

    # Calculate model complexity
    total_params = 0
    total_flops = 0
    memory_mb = 0

    channels = misc_params.get("channels", 3)
    height = misc_params.get("height", 224)
    width = misc_params.get("width", 224)
    batch = misc_params.get("batch_size", 32)

    console.print(f"\n[bold]Input: {batch}×{channels}×{height}×{width}[/bold]")

    for i, layer in enumerate(layers):
        layer_type = layer.get("type", "Unknown")
        params = layer.get("params", {})

        console.print(f"\n[cyan]Layer {i + 1}: {layer_type}[/cyan]")

        if layer_type == "Conv2d":
            in_ch = params.get("in_channels", channels)
            out_ch = params.get("out_channels", 64)
            kernel = params.get("kernel_size", 3)
            stride = params.get("stride", 1)
            padding = params.get("padding", 0)

            if isinstance(kernel, int):
                kernel_h = kernel_w = kernel
            else:
                kernel_h, kernel_w = kernel

            # Calculate output dimensions
            out_h = (height - kernel_h + 2 * padding) // stride + 1
            out_w = (width - kernel_w + 2 * padding) // stride + 1

            # Calculate parameters and FLOPs
            conv_params = in_ch * out_ch * kernel_h * kernel_w
            conv_flops = batch * conv_params * out_h * out_w

            total_params += conv_params
            total_flops += conv_flops

            console.print(f"  Input: {in_ch}×{height}×{width}")
            console.print(f"  Output: {out_ch}×{out_h}×{out_w}")
            console.print(f"  Params: {conv_params:,}")
            console.print(f"  FLOPs: {conv_flops:,}")

            # Update dimensions for next layer
            channels = out_ch
            height = out_h
            width = out_w

        elif layer_type == "Linear":
            in_features = params.get("in_features", channels * height * width)
            out_features = params.get("out_features", 1000)

            linear_params = in_features * out_features
            linear_flops = batch * linear_params

            total_params += linear_params
            total_flops += linear_flops

            console.print(f"  Input features: {in_features:,}")
            console.print(f"  Output features: {out_features:,}")
            console.print(f"  Params: {linear_params:,}")
            console.print(f"  FLOPs: {linear_flops:,}")

            # Linear layers typically don't change spatial dimensions for this analysis
            # But they do change the channel/feature dimension

        elif layer_type in ["MaxPool2d", "AvgPool2d"]:
            kernel = params.get("kernel_size", 2)
            stride = params.get("stride", 2)
            padding = params.get("padding", 0)

            if isinstance(kernel, int):
                kernel_h = kernel_w = kernel
            else:
                kernel_h, kernel_w = kernel

            # Calculate output dimensions
            out_h = (height - kernel_h + 2 * padding) // stride + 1
            out_w = (width - kernel_w + 2 * padding) // stride + 1

            console.print(f"  Input: {channels}×{height}×{width}")
            console.print(f"  Output: {channels}×{out_h}×{out_w}")
            console.print("  Params: 0 (no learnable parameters)")
            console.print(f"  FLOPs: ~{batch * channels * out_h * out_w:,} (minimal)")

            # Update dimensions for next layer
            height = out_h
            width = out_w

        else:
            console.print(f"  [dim]{layer_type} - analysis not implemented[/dim]")

    # Calculate memory requirements
    memory_mb = (total_params * 4) / (
        1024 * 1024
    )  # Assuming 4 bytes per parameter (float32)

    # Display summary
    summary_text = Text.assemble(
        ("Performance Summary", "bold green"),
        (f"\n📊 Total Parameters: {total_params:,}", "cyan"),
        (f"\n💾 Memory Usage: {memory_mb:.2f} MB", "yellow"),
        (f"\n🔥 FLOPs per forward pass: {total_flops:,}", "magenta"),
        (f"\n⚡ FLOPs per sample: {total_flops // batch:,}", "blue"),
        (f"\n📏 Final feature map: {channels}×{height}×{width}", "white"),
    )

    console.print(
        Panel(summary_text, title="[bold]Summary[/bold]", border_style="green"),
    )

    # Performance classification
    if total_params < 1000000:  # < 1M parameters
        perf_class = "Lightweight"
        color = "green"
    elif total_params < 10000000:  # < 10M parameters
        perf_class = "Medium"
        color = "yellow"
    else:  # >= 10M parameters
        perf_class = "Heavy"
        color = "red"

    console.print(f"\n[bold {color}]Model Classification: {perf_class}[/bold {color}]")

    # Recommendations
    console.print("\n[bold blue]💡 Recommendations:[/bold blue]")

    if memory_mb > 1000:  # > 1GB
        console.print(
            "  ⚠️  Consider model compression techniques (pruning, quantization)",
        )
    if total_params > 50000000:  # > 50M parameters
        console.print("  ⚠️  Consider using depthwise separable convolutions")
    if height * width < 10:  # Very small feature maps
        console.print("  ⚠️  Feature maps are very small - check for over-pooling")
