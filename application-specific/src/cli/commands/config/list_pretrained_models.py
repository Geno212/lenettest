import typer
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(help="Configuration commands for pretrained models")


@app.command("list-pretrained")
def list_pretrained_models():
    """List all available pretrained models."""
    # Group models by type
    torchvision_models = [
        m for m in arch_manager.pretrained_models if not m.startswith("yolox")
    ]
    yolox_models = [m for m in arch_manager.pretrained_models if m.startswith("yolox")]

    if yolox_models:
        yolox_text = Text("YOLOX Models:", style="bold blue")
        for model in sorted(yolox_models):
            yolox_text.append(f"\n  • {model}", "cyan")
        console.print(yolox_text)

    if torchvision_models:
        tv_text = Text("\nTorchvision Models:", style="bold blue")
        # Show first 20 models to avoid overwhelming output
        for model in sorted(torchvision_models[:20]):
            tv_text.append(f"\n  • {model}", "cyan")
        if len(torchvision_models) > 20:
            tv_text.append(f"\n  ... and {len(torchvision_models) - 20} more", "dim")
        console.print(tv_text)
