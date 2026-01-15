import typer
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(help="List all available layer types")


@app.command("layers-available")
def list_available_layers(
    filter_type: str | None = typer.Option(
        None,
        "--filter",
        "-f",
        help="Filter by layer type (conv, pool, linear, etc.)",
    ),
):
    """List all available layer types."""
    available_layers = arch_manager.list_available_layers()

    if filter_type:
        filter_lower = filter_type.lower()
        available_layers = [
            layer for layer in available_layers if filter_lower in layer.lower()
        ]

    if not available_layers:
        console.print(
            f"[yellow]📁 No layers found matching filter '{filter_type}'[/yellow]",
        )
        return

    # Group layers by category for better display
    conv_layers = [l for l in available_layers if "conv" in l.lower()]
    pool_layers = [l for l in available_layers if "pool" in l.lower()]
    linear_layers = [l for l in available_layers if "linear" in l.lower()]
    norm_layers = [
        l for l in available_layers if "norm" in l.lower() or "batch" in l.lower()
    ]
    activation_layers = [
        l
        for l in available_layers
        if any(x in l.lower() for x in ["relu", "tanh", "sigmoid", "softmax"])
    ]
    dropout_layers = [l for l in available_layers if "dropout" in l.lower()]
    other_layers = [
        l
        for l in available_layers
        if l
        not in conv_layers
        + pool_layers
        + linear_layers
        + norm_layers
        + activation_layers
        + dropout_layers
    ]

    # Display in organized sections
    sections = [
        ("Convolutional", conv_layers),
        ("Pooling", pool_layers),
        ("Linear", linear_layers),
        ("Normalization", norm_layers),
        ("Activation", activation_layers),
        ("Dropout", dropout_layers),
        ("Other", other_layers),
    ]

    for section_name, layers in sections:
        if layers:
            layers_text = Text()
            layers_text.append(f"\n{section_name} Layers:\n", style="bold blue")
            for layer in sorted(layers):
                layers_text.append(f"  • {layer}\n", style="cyan")

            console.print(layers_text)
