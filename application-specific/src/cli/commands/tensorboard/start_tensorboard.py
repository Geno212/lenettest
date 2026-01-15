import webbrowser

import typer

from src.cli.core.tensorboard_manager import TensorBoardManager
from src.cli.utils.console import console

app = typer.Typer(help="Start TensorBoard server.")


@app.command("start")
def start_tensorboard(
    logdir: str = typer.Option(
        "./data/tensorboardlogs",
        "--logdir",
        "-l",
        help="Log directory to serve",
    ),
    port: int = typer.Option(6006, "--port", "-p", help="Port to run TensorBoard on"),
    open_browser: bool = typer.Option(
        False,
        "--open",
        "-o",
        help="Open TensorBoard in browser",
    ),
):
    """Start TensorBoard server."""
    manager = TensorBoardManager(logdir, port)

    if manager.start_tensorboard():
        if open_browser:
            console.print(
                f"[blue]🌐 Opening TensorBoard in browser: {manager.url}[/blue]",
            )
            webbrowser.open(manager.url)

        # Save manager instance for other commands
        global tensorboard_manager
        tensorboard_manager = manager

        console.print("[green]✅ TensorBoard started successfully![/green]")
        console.print("[dim]💡 Use 'tensorboard stop' to stop the server[/dim]")
    else:
        console.print("[red]❌ Failed to start TensorBoard[/red]")
        raise typer.Exit(1)
