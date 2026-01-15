import typer
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(help="List all pretrained models available")


@app.command("list-pretrained")
def list_pretrained_models():
    """List all available pretrained models."""
    available_models = arch_manager.list_available_pretrained_models()

    if not available_models:
        console.print("[yellow]📁 No pretrained models available[/yellow]")
        return

    # Group models by category for better display
    torchvision_models = [
        m
        for m in available_models
        if not any(
            x in m.lower()
            for x in ["yolox", "efficientnet", "mobilenet", "resnet", "densenet", "vgg"]
        )
    ]
    yolox_models = [m for m in available_models if "yolox" in m.lower()]
    efficientnet_models = [m for m in available_models if "efficientnet" in m.lower()]
    mobilenet_models = [m for m in available_models if "mobilenet" in m.lower()]
    resnet_models = [m for m in available_models if "resnet" in m.lower()]
    densenet_models = [m for m in available_models if "densenet" in m.lower()]
    vgg_models = [m for m in available_models if "vgg" in m.lower()]
    other_models = [
        m
        for m in available_models
        if m
        not in torchvision_models
        + yolox_models
        + efficientnet_models
        + mobilenet_models
        + resnet_models
        + densenet_models
        + vgg_models
    ]

    # Display in organized sections
    sections = [
        ("Torchvision Models", torchvision_models),
        ("YOLOX Models", yolox_models),
        ("EfficientNet Models", efficientnet_models),
        ("MobileNet Models", mobilenet_models),
        ("ResNet Models", resnet_models),
        ("DenseNet Models", densenet_models),
        ("VGG Models", vgg_models),
        ("Other Models", other_models),
    ]

    for section_name, models in sections:
        if models:
            models_text = Text()
            models_text.append(f"\n{section_name}:\n", style="bold blue")
            for model in sorted(models):
                models_text.append(f"  • {model}\n", style="cyan")

            console.print(models_text)
