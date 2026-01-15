import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.utils.console import console

app = typer.Typer(help="List available code generation templates")


@app.command("list-templates")
def list_templates():
    """List available code generation templates."""
    console.print("[bold blue]📋 Available Templates[/bold blue]")

    templates = [
        {
            "name": "PyTorch Training",
            "description": "Complete PyTorch training setup with model, trainer, and utilities",
            "type": "Python",
            "features": [
                "Model definition",
                "Training loop",
                "Validation",
                "Checkpointing",
                "Logging",
            ],
        },
        {
            "name": "SystemC Hardware",
            "description": "SystemC implementation for hardware acceleration",
            "type": "SystemC/C++",
            "features": [
                "Hardware modules",
                "Testbench",
                "CMake build",
                "Simulation setup",
            ],
        },
        {
            "name": "Transfer Learning",
            "description": "Pretrained model fine-tuning setup",
            "type": "Python",
            "features": [
                "Pretrained model loading",
                "Transfer learning",
                "Custom classifier",
                "Fine-tuning",
            ],
        },
        {
            "name": "YOLOX Deployment",
            "description": "YOLOX model deployment and optimization",
            "type": "Python",
            "features": [
                "YOLOX integration",
                "Model conversion",
                "Optimization",
                "Deployment",
            ],
        },
    ]

    for template in templates:
        template_text = Text.assemble(
            (f"{template['name']}", "bold cyan"),
            (f"\n  Type: {template['type']}", "dim"),
            (f"\n  {template['description']}", "white"),
            ("\n  Features:", "green"),
        )

        for feature in template["features"]:
            template_text.append(f"\n    • {feature}", "blue")

        console.print(Panel(template_text, border_style="blue"))
        console.print()  # Empty line between templates
