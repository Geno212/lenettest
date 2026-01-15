#!/usr/bin/env python3
"""TensorBoard Manager for CLI

Manages TensorBoard processes for CLI training integration.
"""

import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.cli.utils.console import console


class TensorBoardManager:
    """Manages TensorBoard processes for CLI training."""

    def __init__(self, logdir: str, port: int = 6006):
        self.logdir = Path(logdir)
        self.port = port
        self.process = None
        self.pid = None
        self.url = f"http://localhost:{port}"

    def is_port_available(self) -> bool:
        """Check if the specified port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", self.port))
                return True
        except OSError:
            return False

    def find_available_port(self, start_port: int = 6006) -> int:
        """Find an available port starting from the specified port."""
        port = start_port
        while not self.is_port_available_port(port):
            port += 1
            if port > start_port + 100:  # Prevent infinite loop
                raise RuntimeError(
                    f"Could not find available port in range {start_port}-{start_port + 100}",
                )
        return port

    def is_port_available_port(self, port: int) -> bool:
        """Check if a specific port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                return True
        except OSError:
            return False

    def start_tensorboard(self) -> bool:
        """Start TensorBoard process."""
        try:
            # Ensure log directory exists
            self.logdir.mkdir(parents=True, exist_ok=True)

            # Check if port is available, find alternative if not
            if not self.is_port_available_port(self.port):
                self.port = self.find_available_port(self.port)
                self.url = f"http://localhost:{self.port}"
                console.print(
                    f"[yellow]⚠️  Port {self.port} in use, using port {self.port} instead[/yellow]",
                )

            # Start TensorBoard process
            cmd = [
                sys.executable,
                "-m",
                "tensorboard.main",
                "serve",
                "--logdir",
                str(self.logdir),
                "--port",
                str(self.port),
            ]

            console.print(f"[blue]🚀 Starting TensorBoard on {self.url}[/blue]")
            console.print(f"[dim]📁 Log directory: {self.logdir}[/dim]")

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            self.pid = self.process.pid
            console.print(f"[green]✅ TensorBoard started with PID {self.pid}[/green]")

            # Wait a moment for TensorBoard to start up
            time.sleep(2)

            return True

        except Exception as e:
            console.print(f"[red]❌ Failed to start TensorBoard: {e}[/red]")
            return False

    def stop_tensorboard(self) -> bool:
        """Stop TensorBoard process."""
        if self.process is None:
            console.print("[yellow]⚠️  No TensorBoard process to stop[/yellow]")
            return False

        try:
            console.print(
                f"[yellow]🛑 Stopping TensorBoard (PID: {self.pid})...[/yellow]",
            )

            # Try graceful termination first
            self.process.terminate()

            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                console.print(
                    "[yellow]⚠️  Graceful shutdown failed, force killing...[/yellow]",
                )
                self.process.kill()
                self.process.wait()

            console.print("[green]✅ TensorBoard stopped successfully[/green]")
            self.process = None
            self.pid = None
            return True

        except Exception as e:
            console.print(f"[red]❌ Error stopping TensorBoard: {e}[/red]")
            return False

    def get_status(self) -> dict[str, Any]:
        """Get TensorBoard status information."""
        status = {
            "running": False,
            "pid": self.pid,
            "port": self.port,
            "url": self.url,
            "logdir": str(self.logdir),
            "process": self.process,
        }

        if self.process is not None:
            if self.process.poll() is None:
                status["running"] = True
            else:
                status["exit_code"] = self.process.poll()
                self.process = None
                self.pid = None

        return status

    def cleanup(self):
        """Clean up any running TensorBoard processes."""
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except:
                if self.process.poll() is None:
                    self.process.kill()
            finally:
                self.process = None
                self.pid = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


tensorboard_manager = None


def get_tensorboard_manager(logdir: str) -> TensorBoardManager:
    """Get or create a TensorBoard manager instance."""
    global tensorboard_manager
    if tensorboard_manager is None:
        tensorboard_manager = TensorBoardManager(logdir)
    return tensorboard_manager
