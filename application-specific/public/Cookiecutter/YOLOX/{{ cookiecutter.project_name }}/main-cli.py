#!/usr/bin/env python3
"""
CLI Training Script for YOLOX Models
Provides a command-line interface with rich progress bars for YOLOX model training.
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
import datetime
import json
import re
import ast

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.panel import Panel
from rich.live import Live
from rich.table import Table

# Path configurations
basedir = os.path.dirname(__file__)
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


def train_yolox(logdir: str, wrap: bool = False, tensorboard: bool = True, tb_port: int = 6006, open_tb: bool = False):
    """
    Train the YOLOX model with live output display.
    
    Args:
        logdir: Directory for TensorBoard logs
        wrap: Whether to perform SystemC wrapping after training
    """
    console.print(Panel.fit(
        "[bold cyan]Starting YOLOX Model Training[/bold cyan]",
        border_style="cyan"
    ))
    
    # Dynamically construct file paths
    train_script = os.path.join(basedir, r"python\train.py")
    yolox_custom = os.path.join(basedir, r"python\yolox_custom.py")
    weight_file = r"{{cookiecutter.pretrained_weights}}"
    
    # Display configuration
    console.print("\n[bold]Training Configuration:[/bold]")
    console.print(f"  Train Script: [cyan]{train_script}[/cyan]")
    console.print(f"  Config File: [cyan]{yolox_custom}[/cyan]")
    console.print(f"  Pretrained Weights: [cyan]{weight_file}[/cyan]")
    console.print(f"  Log Directory: [cyan]{logdir}[/cyan]")
    console.print()
    
    # Create a unique run directory for logs and metrics
    unique_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_log_dir = logdir
    log_dir = os.path.join(base_log_dir, unique_name)
    os.makedirs(log_dir, exist_ok=True)
    metrics_file = os.path.join(log_dir, "training_metrics.json")
    metrics = {"epochs": []}

    # Optionally start TensorBoard just before training
    tb_proc = None
    tb_url = None
    if tensorboard:
        try:
            tb_proc, tb_url = start_tensorboard_server(logdir, tb_port, open_tb)
            console.print(f"[dim]🌐 TensorBoard: {tb_url}[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not start TensorBoard: {e}[/yellow]")

    # Build the training command
    command = [
        sys.executable,  # Use current Python interpreter
        train_script,
        "-f", yolox_custom,
        "-d", "1",
        "-b", "2",
        "--fp16",
        "-o",
        "-c", weight_file,
    ]
    

    env = os.environ.copy()
    yolox_root = str(Path(basedir).parent.parent.absolute()) 
    # Add to PYTHONPATH
    env["PYTHONPATH"] = yolox_root + os.pathsep + env.get("PYTHONPATH", "")
    # -----------------------------------------

    try:
        console.print("[cyan]Starting training process...[/cyan]\n")
        
        # Run the training process with live output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env  # <--- Pass the modified environment here
        )
        
        # Display live output and attempt to extract any metrics logged by YOLOX
        if process.stdout is None:
            console.print("[yellow]⚠️  Live output is unavailable (no stdout)[/yellow]")
        else:
            for line in iter(process.stdout.readline, ''):
                if not line:
                    continue
                raw = line.rstrip("\n")
                lowered = raw.lower()

                # Print with highlighting
                if "error" in lowered or "exception" in lowered:
                    console.print(f"[bold red]{raw}[/bold red]")
                elif "epoch" in lowered or "iter" in lowered:
                    console.print(f"[bold cyan]{raw}[/bold cyan]")
                elif any(k in lowered for k in ("loss", "map", "acc", "ap")):
                    console.print(f"[yellow]{raw}[/yellow]")
                else:
                    console.print(f"[dim]{raw}[/dim]")

                parsed_metrics = {}
                parsed_epoch = None

                try:
                    # 1) Try to find a JSON/dict segment and parse it
                    if "{" in raw and "}" in raw:
                        try:
                            start = raw.index("{")
                            end = raw.rindex("}")
                            snippet = raw[start:end+1]
                            obj = ast.literal_eval(snippet)
                            if isinstance(obj, dict):
                                for k, v in obj.items():
                                    try:
                                        parsed_metrics[str(k).lower()] = float(v)
                                    except Exception:
                                        parsed_metrics[str(k).lower()] = v
                        except Exception:
                            pass

                    # 2) Extract epoch if present
                    m_epoch = re.search(r"epoch[^0-9]*(\d+)", lowered)
                    if m_epoch:
                        parsed_epoch = int(m_epoch.group(1))

                    # 3) Find key:value or key=value pairs
                    for m in re.finditer(r"([A-Za-z0-9_/%-]+)\s*[:=]\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", raw):
                        k = m.group(1).strip().lower()
                        v = m.group(2)
                        try:
                            parsed_metrics[k] = float(v)
                        except Exception:
                            parsed_metrics[k] = v

                    # 4) Also capture 'key value' patterns
                    for m in re.finditer(r"\b([a-zA-Z_/%-]{2,})\b\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", raw):
                        k = m.group(1).strip().lower()
                        v = m.group(2)
                        if k in ("epoch", "iter", "step", "batch"):
                            continue
                        try:
                            parsed_metrics.setdefault(k, float(v))
                        except Exception:
                            parsed_metrics.setdefault(k, v)

                    # If we found metrics, record them.
                    if parsed_metrics:
                        # Fix for deprecation warning: use timezone-aware UTC
                        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                        
                        if parsed_epoch is not None:
                            entry = {"epoch": int(parsed_epoch)}
                            for k, v in parsed_metrics.items():
                                entry[k] = v
                            if metrics.get("epochs") and metrics["epochs"][-1].get("epoch") == entry["epoch"]:
                                metrics["epochs"][-1].update(entry)
                            else:
                                metrics.setdefault("epochs", []).append(entry)
                        else:
                            metrics.setdefault("entries", []).append({"timestamp": timestamp, "line": raw, "metrics": parsed_metrics})

                        # flush to disk (best-effort)
                        try:
                            tmp_path = metrics_file + ".tmp"
                            with open(tmp_path, "w", encoding="utf-8") as f:
                                json.dump(metrics, f, indent=2)
                            os.replace(tmp_path, metrics_file)
                        except Exception:
                            pass
                except Exception:
                    pass
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            console.print("\n[bold green]✓[/bold green] Training completed successfully!")
            
            # Check if model was saved
            if os.path.isfile(model_output):
                console.print(f"[bold green]✓[/bold green] Model saved to: {model_output}")
            else:
                console.print("[bold yellow]⚠[/bold yellow] Warning: Model file not found")

            # Finalize metrics file
            try:
                tmp_path = metrics_file + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                os.replace(tmp_path, metrics_file)
            except Exception:
                pass
            
            # Perform wrapping if requested
            if wrap:
                console.print("\n" + "="*50)
                return perform_cmake_wrap()
            
            return True
        else:
            console.print(f"\n[bold red]✗[/bold red] Training failed with exit code: {return_code}")
            return False
            
    except FileNotFoundError:
        console.print(f"[bold red]✗[/bold red] Training script not found: {train_script}")
        return False
    except Exception as e:
        console.print(f"\n[bold red]✗[/bold red] Training failed: {str(e)}")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
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
        description="Train a YOLOX model via CLI with progress tracking",
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
    
    # Execute training
    success = train_yolox(
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