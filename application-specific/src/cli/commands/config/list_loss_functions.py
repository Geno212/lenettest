import typer
from rich.table import Table

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(help="Configuration commands for model parameters")


@app.command("list-losses")
def list_loss_functions():
    """List all available loss functions."""
    table = Table(title="Available Loss Functions")
    table.add_column("Loss Function", style="cyan", no_wrap=True)
    table.add_column("Parameters", style="white", max_width=50)

    for loss_name, params in arch_manager.loss_funcs.items():
        params_str = ", ".join([f"{p['name']}" for p in params[:3]])
        if len(params) > 3:
            params_str += "..."

        table.add_row(loss_name, params_str)

    console.print(table)
