import typer

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(help="Configuration commands")


@app.command("reset")
def reset_config():
    """Reset all configuration to defaults."""
    arch_manager.current_architecture = {
        "layers": [],
        "misc_params": {
            "device": {
                "value": "cpu",
                "index": 0,
            },
            "width": 224,
            "height": 224,
            "channels": 3,
            "num_epochs": 10,
            "batch_size": 32,
            "dataset": {
                "value": None,
                "index": -1,
            },
            "dataset_path": None,
        },
        "optimizer": {},
        "loss_func": {},
        "scheduler": {},
        "pretrained": {"value": None, "index": -1},
    }
    console.print("[green]✅ Configuration reset to defaults[/green]")
