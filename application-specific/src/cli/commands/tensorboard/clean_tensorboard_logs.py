import time
from pathlib import Path

import typer

from src.cli.utils.console import console

app = typer.Typer(help="Clean old TensorBoard log files.")


@app.command("clean")
def clean_tensorboard_logs(
    logdir: str = typer.Option(
        "./data/tensorboardlogs",
        "--logdir",
        "-l",
        help="Log directory to clean",
    ),
    older_than_days: int = typer.Option(
        7,
        "--older-than",
        "-d",
        help="Remove logs older than N days",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force removal without confirmation",
    ),
):
    """Clean old TensorBoard log files."""
    log_path = Path(logdir)

    if not log_path.exists():
        console.print(f"[red]❌ Log directory {logdir} not found[/red]")
        return

    # Find old log directories

    current_time = time.time()
    cutoff_time = current_time - (older_than_days * 24 * 60 * 60)

    old_dirs = []
    for item in log_path.iterdir():
        if item.is_dir() and item.stat().st_mtime < cutoff_time:
            old_dirs.append(item)

    if not old_dirs:
        console.print(
            f"[green]✅ No log directories older than {older_than_days} days found[/green]",
        )
        return

    # Show what will be deleted
    total_size = sum(
        sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) for d in old_dirs
    )
    size_mb = total_size / (1024 * 1024)

    console.print(
        f"[yellow]🗑️  Found {len(old_dirs)} old log directories ({size_mb:.1f} MB) to remove:[/yellow]",
    )
    for old_dir in old_dirs:
        console.print(f"  • {old_dir.name}")

    if not force:
        confirm = typer.confirm("Are you sure you want to delete these logs?")
        if not confirm:
            console.print("[blue]ℹ️  Operation cancelled[/blue]")
            return

    # Delete old directories
    deleted_count = 0
    for old_dir in old_dirs:
        try:
            import shutil

            shutil.rmtree(old_dir)
            deleted_count += 1
        except Exception as e:
            console.print(f"[red]❌ Error deleting {old_dir.name}: {e}[/red]")

    console.print(
        f"[green]✅ Successfully deleted {deleted_count}/{len(old_dirs)} log directories[/green]",
    )
