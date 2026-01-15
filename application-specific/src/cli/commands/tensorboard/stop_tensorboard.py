import typer

from src.cli.core.tensorboard_manager import tensorboard_manager
from src.cli.utils.console import console

app = typer.Typer(help="Stop TensorBoard server.")


@app.command("stop")
def stop_tensorboard():
    """Stop TensorBoard server."""
    if tensorboard_manager is None:
        console.print("[yellow]⚠️  No active TensorBoard server found[/yellow]")
        return

    if tensorboard_manager.stop_tensorboard():
        console.print("[green]✅ TensorBoard stopped successfully[/green]")
    else:
        console.print("[red]❌ Failed to stop TensorBoard[/red]")
