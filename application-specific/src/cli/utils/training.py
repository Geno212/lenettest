#!/usr/bin/env python3
"""Training Utilities for CLI

Utilities for integrating CLI training with GUI training system.
"""

import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from rich.console import Console

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

console = Console()


class TrainingProgress:
    """Handles training progress tracking for CLI."""

    def __init__(self):
        self.progress_value = 0
        self.current_epoch = 0
        self.total_epochs = 1
        self.training_active = False
        self._lock = threading.Lock()

    def update_progress(self, value: float):
        """Update progress value (0-100)."""
        with self._lock:
            self.progress_value = max(0, min(100, value))

    def set_epochs(self, current: int, total: int):
        """Set current and total epoch information."""
        with self._lock:
            self.current_epoch = current
            self.total_epochs = total

    def get_status(self) -> dict[str, Any]:
        """Get current progress status."""
        with self._lock:
            return {
                "progress": self.progress_value,
                "epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "active": self.training_active,
            }


def create_progress_callback(progress_tracker: TrainingProgress) -> Callable:
    """Create a progress callback function for the GUI training system."""

    def progress_callback(value: float):
        """Progress callback that integrates with Rich progress bars."""
        progress_tracker.update_progress(value)

        # Calculate which epoch we're in
        if hasattr(progress_callback, "total_steps"):
            steps_per_epoch = (
                progress_callback.total_steps // progress_callback.total_epochs
            )
            current_epoch = int(value * progress_callback.total_epochs / 100) + 1
            progress_tracker.set_epochs(current_epoch, progress_callback.total_epochs)

    # Store metadata for epoch calculation
    progress_callback.total_epochs = 1  # Will be set by caller
    progress_callback.total_steps = 100  # Will be set by caller

    return progress_callback


def setup_training_environment(
    output_dir: Path,
    architecture_config: dict[str, Any],
) -> dict[str, Any]:
    """Set up the training environment with proper paths and configurations."""
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use default tensorboard logs directory unless user specifies otherwise
    project_root = Path(__file__).parent.parent.parent.parent  # Go up to project root
    default_tensorboard_dir = project_root / "data" / "tensorboardlogs"
    logs_dir = default_tensorboard_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Set up SystemC output structure
    systemc_dir = output_dir / "SystemC"
    systemc_dir.mkdir(exist_ok=True)
    pt_dir = systemc_dir / "Pt"
    pt_dir.mkdir(exist_ok=True)

    # Create training configuration
    training_config = {
        "architecture": architecture_config,
        "output_dir": str(output_dir),
        "logs_dir": str(logs_dir),
        "models_dir": str(models_dir),
        "systemc_dir": str(systemc_dir),
        "model_output": str(pt_dir / "model.pt"),
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    }

    # Save configuration
    config_file = output_dir / "training_config.json"
    import json

    with open(config_file, "w") as f:
        json.dump(training_config, f, indent=2)

    return training_config


def import_training_module(model_type: str = "pretrained"):
    """Import the correct training module based on model type."""
    try:
        project_root = Path(__file__).parent.parent.parent.parent

        if model_type.lower() == "pretrained":
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "pretrained",
                str(
                    project_root
                    / "public/Cookiecutter/Pretrained/python/pretrained.py",
                ),
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.train

        if model_type.lower() == "manual":
            spec = importlib.util.spec_from_file_location(
                "manual",
                str(project_root / "public/Cookiecutter/Manual/python/manual.py"),
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.train

        if model_type.lower() == "yolox":
            sys.path.append(str(project_root / "YOLOX"))
            from YOLOX.yolox.core.trainer import Trainer

            return Trainer

    except ImportError as e:
        console.print(f"[red]❌ Could not import training module: {e}[/red]")
        raise


def save_checkpoint(state: dict[str, Any], filepath: str):
    """Save model checkpoint."""
    console.print(f"[blue]💾 Saving checkpoint to {filepath}...[/blue]")
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model: Any, optimizer: Any) -> dict[str, Any]:
    """Load model checkpoint."""
    if not os.path.exists(filepath):
        console.print(f"[red]❌ Checkpoint file not found: {filepath}[/red]")
        return None

    console.print(f"[blue]🔄 Loading checkpoint from {filepath}...[/blue]")
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def run_training(
    architecture_config: dict[str, Any],
    training_config: dict[str, Any],
    progress_callback: Callable,
    verbose: bool = False,
    resume: bool = False,
    checkpoint_path: str | None = None,
) -> bool:
    """Run model training using the unified training backend."""
    try:
        # Determine model type from architecture config
        model_type = architecture_config.get("model_type", "pretrained")

        # Import the correct training function
        train_func = import_training_module(model_type)

        # Extract training parameters
        logdir = Path(training_config["logs_dir"])
        model_output = Path(training_config["model_output"])

        if verbose:
            console.print(
                f"[blue]📊 Starting training with {model_type} backend[/blue]",
            )
            console.print(f"[dim]📁 Log directory: {logdir}[/dim]")
            console.print(f"[dim]🤖 Model output: {model_output}[/dim]")

        # Create necessary directories
        logdir.mkdir(parents=True, exist_ok=True)
        model_output.parent.mkdir(parents=True, exist_ok=True)

        # Run training with the appropriate backend
        if model_type.lower() == "yolox":
            trainer = train_func(architecture_config)
            trainer.train()
        else:
            train_func(
                callback=progress_callback,
                logdir=str(logdir),
                model_output=str(model_output),
                checkpoint_path=checkpoint_path if resume else None,
                architecture=architecture_config,
            )

        if verbose:
            console.print("[green]✅ Training completed successfully[/green]")

        return True

    except Exception as e:
        console.print(f"[red]❌ Training failed: {e}[/red]")
        if verbose:
            import traceback

            console.print(f"[red]Stack trace: {traceback.format_exc()}[/red]")
        return False


class TrainingMonitor:
    """Monitors training progress and provides real-time updates."""

    def __init__(self, progress_tracker: TrainingProgress):
        self.progress_tracker = progress_tracker
        self.monitoring_active = False
        self._monitor_thread = None

    def start_monitoring(self, update_interval: float = 1.0):
        """Start monitoring training progress."""
        self.monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(update_interval,),
            daemon=True,
        )
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring training progress."""
        self.monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)

    def _monitor_loop(self, update_interval: float):
        """Monitor loop that runs in background thread."""
        while self.monitoring_active:
            status = self.progress_tracker.get_status()
            # Could add real-time logging here if needed
            time.sleep(update_interval)


def validate_training_requirements(architecture_config: dict[str, Any]) -> bool:
    """Validate that all training requirements are met."""
    required_fields = [
        "layers",
        "optimizer",
        "loss_func",
    ]

    # Check for required architecture components
    for field in required_fields:
        if field not in architecture_config:
            console.print(f"[red]❌ Missing required field: {field}[/red]")
            return False

    # Check for layers
    if not architecture_config["layers"]:
        console.print("[red]❌ No layers defined in architecture[/red]")
        return False

    return True
