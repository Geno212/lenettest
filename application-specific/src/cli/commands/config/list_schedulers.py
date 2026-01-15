import typer
from rich.table import Table

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(help="Configuration commands for model training")


@app.command("list-schedulers")
def list_schedulers():
    """List all available schedulers."""
    table = Table(title="Available Schedulers")
    table.add_column("Scheduler", style="cyan", no_wrap=True)
    table.add_column("Parameters", style="white", max_width=50)

    for scheduler_name, params in arch_manager.schedulers.items():
        params_str = ", ".join([f"{p['name']}" for p in params[:3]])
        if len(params) > 3:
            params_str += "..."

        table.add_row(scheduler_name, params_str)

    console.print(table)
