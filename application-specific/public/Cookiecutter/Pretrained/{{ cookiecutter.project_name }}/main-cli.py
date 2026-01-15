#!/usr/bin/env python3
"""
CLI Training Script for Pretrained Models
Provides a command-line interface with rich progress bars for model training.
"""
import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from rich.text import Text

# Add python directory to path
basedir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(basedir, 'python'))

from python.pretrained import train

# Path configurations
model_output = os.path.normpath(os.path.join(basedir, "./SystemC/Pt/model.pt"))
build_output = os.path.normpath(os.path.join(basedir, "./SystemC/build"))
source_output = os.path.normpath(os.path.join(basedir, "./SystemC"))

console = Console()


def _is_port_available(port: int) -> bool:
    """Check if a TCP port is available on localhost."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("localhost", port))
            return True
    except OSError:
        return False


def _find_open_port(start_port: int = 6006, max_tries: int = 20) -> int:
    """Find an available port, searching sequentially from start_port."""
    port = start_port
    tries = 0
    while tries < max_tries and not _is_port_available(port):
        port += 1
        tries += 1
    return port


def start_tensorboard_server(logdir: str, port: int = 6006, open_browser: bool = False):
    """Start a TensorBoard server as a subprocess.

    Returns a tuple (process, url). The process will keep running after this script ends.
    """
    # Ensure logdir exists
    os.makedirs(logdir, exist_ok=True)

    chosen_port = port if _is_port_available(port) else _find_open_port(port)
    if chosen_port != port:
        console.print(
            f"[yellow]⚠️  Port {port} is busy, using port {chosen_port} instead[/yellow]",
        )

    url = f"http://localhost:{chosen_port}"
    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "serve",
        "--logdir",
        os.path.normpath(logdir),
        "--port",
        str(chosen_port),
    ]

    console.print(f"[blue]🚀 Starting TensorBoard on {url}[/blue]")
    console.print(f"[dim]📁 Log directory: {os.path.normpath(logdir)}[/dim]")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # Give TB a moment to come up (best-effort)
    time.sleep(1.5)

    if open_browser:
        console.print(f"[blue]🌐 Opening TensorBoard in browser: {url}[/blue]")
        try:
            webbrowser.open(url)
        except Exception:
            pass

    console.print("[green]✅ TensorBoard started[/green]")
    console.print("[dim]💡 It will keep running after this script exits[/dim]")
    return proc, url


def train_with_progress(
    logdir: str,
    wrap: bool = False,
    tensorboard: bool = True,
    tb_port: int = 6006,
    open_tb: bool = False,
):
    """
    Train the model with a rich progress bar display.
    
    Args:
        logdir: Directory for TensorBoard logs
        wrap: Whether to perform SystemC wrapping after training
    """
    console.print(Panel.fit(
        "[bold cyan]Starting Pretrained Model Training[/bold cyan]",
        border_style="cyan"
    ))
    
    # Optionally start TensorBoard just before training
    tb_proc = None
    tb_url = None
    if tensorboard:
        try:
            tb_proc, tb_url = start_tensorboard_server(logdir, tb_port, open_tb)
            console.print(f"[dim]🌐 TensorBoard: {tb_url}[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not start TensorBoard: {e}[/yellow]")

    # Create progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        
        training_task = progress.add_task(
            "[cyan]Training model...", 
            total=100
        )
        
        def progress_callback(value):
            """Callback to update progress bar"""
            progress.update(training_task, completed=min(value, 100))
        
        try:
            # Call the training function
            train(callback=progress_callback, logdir=logdir)
            progress.update(training_task, completed=100)
            console.print("\n[bold green]✓[/bold green] Training completed successfully!")
            
            # Check if model was saved
            if os.path.isfile(model_output):
                console.print(f"[bold green]✓[/bold green] Model saved to: {model_output}")
            else:
                console.print("[bold yellow]⚠[/bold yellow] Warning: Model file not found")
                return False
            
            # Perform wrapping if requested
            if wrap:
                console.print("\n" + "="*50)
                return perform_cmake_wrap()
            
            return True
            
        except Exception as e:
            console.print(f"\n[bold red]✗[/bold red] Training failed: {str(e)}")
            return False


def perform_cmake_wrap():
    """
    Perform CMake configuration and build for SystemC wrapping.
    
    Returns:
        bool: True if successful, False otherwise
    """
    console.print(Panel.fit(
        "[bold magenta]Starting SystemC Wrapping[/bold magenta]",
        border_style="magenta"
    ))
    
    # Check if model exists
    if not os.path.isfile(model_output):
        console.print("[bold red]✗[/bold red] Model file not found. Cannot wrap.")
        return False
    
    try:
        # Clean build directory
        if os.path.exists(build_output):
            console.print(f"[yellow]Cleaning build directory...[/yellow]")
            shutil.rmtree(build_output)
        
        # CMake configuration
        console.print("[cyan]Running CMake configuration...[/cyan]")
        result = subprocess.run(
            ["cmake", "-S", source_output, "-B", build_output],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout:
            console.print(f"[dim]{result.stdout}[/dim]")
        
        console.print("[bold green]✓[/bold green] CMake configuration completed")
        
        # CMake build
        console.print("[cyan]Running CMake build...[/cyan]")
        result = subprocess.run(
            ["cmake", "--build", build_output, "--clean-first"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout:
            console.print(f"[dim]{result.stdout}[/dim]")
        
        console.print("[bold green]✓[/bold green] CMake build completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]✗[/bold red] CMake process failed:")
        if e.stderr:
            console.print(f"[red]{e.stderr}[/red]")
        return False
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Wrapping failed: {str(e)}")
        return False


def main():
    """Main entry point for CLI training."""
    parser = argparse.ArgumentParser(
        description="Train a pretrained model via CLI with progress tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--logdir",
        type=str,
        default=r"{{cookiecutter.log_dir}}",
        help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Do not start a local TensorBoard server on training start",
    )
    parser.add_argument(
        "--tb-port",
        type=int,
        default=6006,
        help="Port to run the TensorBoard server on",
    )
    parser.add_argument(
        "--open-tb",
        action="store_true",
        help="Open TensorBoard URL in the default browser",
    )
    
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="Perform SystemC wrapping after training"
    )
    
    parser.add_argument(
        "--no-wrap",
        action="store_true",
        help="Skip SystemC wrapping (train only)"
    )
    
    args = parser.parse_args()
    
    # Display configuration
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Log Directory: [cyan]{args.logdir}[/cyan]")
    console.print(f"  SystemC Wrap: [cyan]{'Yes' if args.wrap else 'No'}[/cyan]")
    console.print(f"  Start TensorBoard: [cyan]{'No' if args.no_tensorboard else 'Yes'}[/cyan]")
    console.print(f"  TensorBoard Port: [cyan]{args.tb_port}[/cyan]")
    console.print()
    
    # Execute training
    success = train_with_progress(
        logdir=args.logdir,
        wrap=args.wrap and not args.no_wrap,
        tensorboard=not args.no_tensorboard,
        tb_port=args.tb_port,
        open_tb=args.open_tb,
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
