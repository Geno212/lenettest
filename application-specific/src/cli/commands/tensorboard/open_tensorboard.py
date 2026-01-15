import webbrowser

import typer

from src.cli.core.tensorboard_manager import TensorBoardManager, tensorboard_manager
from src.cli.utils.console import console

app = typer.Typer(help="Open TensorBoard in web browser.")


@app.command("open")
def open_tensorboard(
    logdir: str = typer.Option(
        "./data/tensorboardlogs",
        "--logdir",
        "-l",
        help="Log directory",
    ),
    port: int = typer.Option(6006, "--port", "-p", help="Port (if server not running)"),
):
    """Open TensorBoard in web browser."""
    global tensorboard_manager
    manager = TensorBoardManager(logdir, port)

    # Check if server is already running
    if tensorboard_manager and tensorboard_manager.get_status()["running"]:
        url = tensorboard_manager.url
    else:
        # Try to start server
        console.print("[blue]🔄 TensorBoard not running, starting server...[/blue]")
        if manager.start_tensorboard():
            tensorboard_manager = manager
            url = manager.url
        else:
            console.print("[red]❌ Could not start TensorBoard server[/red]")
            return

    console.print(f"[blue]🌐 Opening TensorBoard: {url}[/blue]")
    webbrowser.open(url)
