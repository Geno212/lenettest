import typer
from rich.table import Table

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(
    help="Configuration commands for model parameters, optimizer, loss function, and scheduler",
)


@app.command("list-optimizers")
def list_optimizers():
    """List all available optimizers."""
    table = Table(title="Available Optimizers")
    table.add_column("Optimizer", style="cyan", no_wrap=True)
    table.add_column("Parameters", style="white", max_width=50)

    for optimizer_name, params in arch_manager.optimizers.items():
        params_str = ", ".join([f"{p['name']}" for p in params[:3]])
        if len(params) > 3:
            params_str += "..."

        table.add_row(optimizer_name, params_str)

    console.print(table)
