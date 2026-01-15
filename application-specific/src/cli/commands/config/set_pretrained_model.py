import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(help="Set pretrained model for the architecture")


@app.command("pretrained")
def set_pretrained_model(
    model_name: str = typer.Argument(..., help="Pretrained model name"),
):
    """Set the pretrained model."""
    if model_name not in arch_manager.pretrained_models:
        console.print(f"[red]❌ Pretrained model '{model_name}' not available[/red]")
        console.print(
            f"[blue]ℹ️  Available models: {', '.join(arch_manager.pretrained_models[:10])}...[/blue]",
        )
        return

    # Set pretrained model in architecture
    arch_manager.current_architecture["pretrained"] = {
        "value": model_name,
        "index": arch_manager.pretrained_models.index(model_name),
    }

    success_text = Text.assemble(
        ("Pretrained model configured!", "bold green"),
        (f"\n🤖 Model: {model_name}", "cyan"),
        (f"\n📍 Index: {arch_manager.pretrained_models.index(model_name)}", "blue"),
    )

    console.print(
        Panel(
            success_text,
            title="[bold]Pretrained Model[/bold]",
            border_style="green",
        ),
    )
