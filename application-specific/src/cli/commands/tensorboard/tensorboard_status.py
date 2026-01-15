import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.tensorboard_manager import tensorboard_manager
from src.cli.utils.console import console

app = typer.Typer(help="Show TensorBoard server status.")


@app.command("status")
def tensorboard_status():
    """Show TensorBoard server status."""
    if tensorboard_manager is None:
        console.print("[yellow]⚠️  No TensorBoard manager initialized[/yellow]")
        console.print("[blue]💡 Use 'tensorboard start' to start a server[/blue]")
        return

    status = tensorboard_manager.get_status()

    # Create status panel
    if status["running"]:
        status_text = Text.assemble(
            ("TensorBoard Status", "bold green"),
            (f"\n🌐 URL: {status['url']}", "cyan"),
            (f"\n📁 Log Directory: {status['logdir']}", "white"),
            (f"\n🔌 Port: {status['port']}", "yellow"),
            (f"\n⚙️  PID: {status['pid']}", "magenta"),
            ("\n✅ Status: Running", "green"),
        )
        border_style = "green"
    else:
        status_text = Text.assemble(
            ("TensorBoard Status", "bold red"),
            (f"\n📁 Log Directory: {status['logdir']}", "white"),
            (f"\n🔌 Port: {status['port']}", "yellow"),
            ("\n❌ Status: Not Running", "red"),
            ("\n💡 Use 'tensorboard start' to start the server", "dim"),
        )
        border_style = "red"

    console.print(
        Panel(status_text, title="[bold]TensorBoard[/bold]", border_style=border_style),
    )
