import time
from pathlib import Path

import typer
from rich.table import Table

from src.cli.utils.console import console

app = typer.Typer(help="Show available TensorBoard log files.")


@app.command("logs")
def show_tensorboard_logs(
    logdir: str = typer.Option(
        "./data/tensorboardlogs",
        "--logdir",
        "-l",
        help="Log directory to examine",
    ),
    max_files: int = typer.Option(
        10,
        "--max-files",
        "-n",
        help="Maximum number of log files to show",
    ),
):
    """Show available TensorBoard log files."""
    log_path = Path(logdir)

    if not log_path.exists():
        console.print(f"[red]❌ Log directory {logdir} not found[/red]")
        return

    # Find log directories (TensorBoard creates subdirectories with timestamps)
    log_dirs = []
    for item in log_path.iterdir():
        if item.is_dir():
            log_dirs.append(item)

    if not log_dirs:
        console.print(f"[yellow]📁 No log directories found in {logdir}[/yellow]")
        return

    # Sort by modification time (newest first)
    log_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Create table
    table = Table(title=f"TensorBoard Logs in {logdir}")
    table.add_column("Directory", style="cyan")
    table.add_column("Modified", style="yellow")
    table.add_column("Size", style="green")

    for log_dir in log_dirs[:max_files]:
        # Calculate directory size
        total_size = sum(f.stat().st_size for f in log_dir.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)

        # Format modification time
        mod_time = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(log_dir.stat().st_mtime),
        )

        table.add_row(
            log_dir.name,
            mod_time,
            f"{size_mb:.1f} MB",
        )

    console.print(table)

    if len(log_dirs) > max_files:
        console.print(
            f"[dim]... and {len(log_dirs) - max_files} more directories[/dim]",
        )
