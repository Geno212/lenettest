import typer
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(help="List available complex pretrained models")


@app.command("list-complex-models")
def list_complex_models():
    """List all available complex pretrained models."""
    if arch_manager.complex_arch_models:
        models_text = Text("Available Complex Models:", style="bold blue")
        for model in sorted(arch_manager.complex_arch_models):
            models_text.append(f"\n  • {model}", "cyan")
        console.print(models_text)
    else:
        console.print("[yellow]📁 No complex models available[/yellow]")
