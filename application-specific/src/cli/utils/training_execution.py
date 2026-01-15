#!/usr/bin/env python3
"""Training Execution Utilities

Utilities for executing generated training scripts from CLI commands.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

console = Console()


def transform_to_cookiecutter_format(
    config: dict[str, Any], training_type: str,
) -> dict[str, Any]:
    """Transform simple JSON config to cookiecutter template format."""

    def create_value_index_dict(value, index=0):
        """Create the value/index dict format expected by cookiecutter."""
        if isinstance(value, dict) and "value" in value:
            return value  # Already in correct format
        return {"value": value, "index": index}

    # Base cookiecutter config
    cookiecutter_config = {
        "project_name": f"{training_type.title()}Training_{int(time.time())}",
    }

    # Transform pretrained model
    if "pretrained" in config:
        if isinstance(config["pretrained"], dict) and "value" in config["pretrained"]:
            cookiecutter_config["pretrained"] = config["pretrained"]
        else:
            cookiecutter_config["pretrained"] = create_value_index_dict(
                config["pretrained"], 0,
            )

    # Transform misc_params
    if "misc_params" in config:
        misc = config["misc_params"]
        cookiecutter_config["misc_params"] = {
            "device": create_value_index_dict(misc.get("device", "cpu"), 0),
            "width": misc.get("width", 224),
            "height": misc.get("height", 224),
            "channels": misc.get("channels", 3),
            "num_epochs": misc.get("epoch", misc.get("num_epochs", 10)),
            "batch_size": misc.get("batch_size", 32),
            "dataset": create_value_index_dict(misc.get("dataset", "MNIST"), 0),
            "dataset_path": misc.get("dataset_path", ""),
        }

    # Transform optimizer
    if "optimizer" in config:
        opt = config["optimizer"]
        if isinstance(opt, dict) and "name" in opt:
            cookiecutter_config["optimizer"] = {
                "type": opt["name"],
                "params": opt.get("params", {}),
            }
        else:
            cookiecutter_config["optimizer"] = {"type": str(opt), "params": {}}

    # Transform loss function
    if "loss_func" in config:
        loss = config["loss_func"]
        if isinstance(loss, dict) and "name" in loss:
            cookiecutter_config["loss_func"] = {
                "type": loss["name"],
                "params": loss.get("params", {}),
            }
        else:
            cookiecutter_config["loss_func"] = {"type": str(loss), "params": {}}

    # Transform scheduler
    if "scheduler" in config:
        sched = config["scheduler"]
        if isinstance(sched, dict) and "name" in sched:
            cookiecutter_config["scheduler"] = {
                "type": sched["name"],
                "params": sched.get("params", {}),
            }
        else:
            cookiecutter_config["scheduler"] = {"type": str(sched), "params": {}}

    # Transform layers (for manual training)
    if config.get("layers"):
        cookiecutter_config["layers"] = {"list": []}
        for i, layer in enumerate(config["layers"]):
            layer_dict = {
                "type": layer.get("layer_type", "Conv2d"),
                "name": f"{layer.get('layer_type', 'layer').lower()}_{i + 1}",
                "params": {k: v for k, v in layer.items() if k != "layer_type"},
            }
            cookiecutter_config["layers"]["list"].append(layer_dict)

    # Set default log directory
    cookiecutter_config["log_dir"] = str(
        Path(__file__).parent.parent.parent.parent / "data" / "tensorboardlogs",
    )

    return cookiecutter_config


def execute_manual_training(
    config_path: Path,
    output_dir: Path,
    verbose: bool = False,
) -> bool:
    """Execute manual training using the generated configuration."""
    try:
        # Load configuration
        with open(config_path) as f:
            config = json.load(f)

        # Path to the manual training template
        project_root = Path(__file__).parent.parent.parent.parent
        manual_template = project_root / "public" / "Cookiecutter" / "Manual"

        # Generate the training project using cookiecutter
        from cookiecutter.main import cookiecutter

        generated_project = cookiecutter(
            str(manual_template),
            extra_context=config,
            output_dir=str(output_dir),
            no_input=True,
        )

        # Execute the generated training script
        training_script = Path(generated_project) / "python" / "manual.py"

        if not training_script.exists():
            console.print(f"[red]❌ Training script not found: {training_script}[/red]")
            return False

        console.print(f"[green]🚀 Executing training script: {training_script}[/green]")

        # Run the training script
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(generated_project) / "python")

        result = subprocess.run(
            [sys.executable, str(training_script)],
            check=False, cwd=str(Path(generated_project)),
            env=env,
            capture_output=not verbose,
            text=True,
        )

        if result.returncode == 0:
            console.print("[green]✅ Training completed successfully![/green]")
            return True
        console.print(
            f"[red]❌ Training failed with return code: {result.returncode}[/red]",
        )
        if result.stderr and not verbose:
            console.print(f"[red]Error: {result.stderr}[/red]")
        return False

    except Exception as e:
        console.print(f"[red]❌ Error executing manual training: {e}[/red]")
        if verbose:
            console.print_exception()
        return False


def execute_pretrained_training(
    config_path: Path,
    output_dir: Path,
    verbose: bool = False,
) -> bool:
    """Execute pretrained training using the generated configuration."""
    try:
        # Load configuration
        with open(config_path) as f:
            config = json.load(f)

        # Transform the simple JSON format to cookiecutter format
        cookiecutter_config = transform_to_cookiecutter_format(config, "pretrained")

        # Path to the pretrained training template
        project_root = Path(__file__).parent.parent.parent.parent
        pretrained_template = project_root / "public" / "Cookiecutter" / "Pretrained"

        # Generate the training project using cookiecutter
        from cookiecutter.main import cookiecutter

        generated_project = cookiecutter(
            str(pretrained_template),
            extra_context=cookiecutter_config,
            output_dir=str(output_dir),
            no_input=True,
        )

        # Execute the generated training script
        training_script = Path(generated_project) / "python" / "pretrained.py"

        if not training_script.exists():
            console.print(f"[red]❌ Training script not found: {training_script}[/red]")
            return False

        console.print(f"[green]🚀 Executing training script: {training_script}[/green]")

        # Run the training script
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(generated_project) / "python")

        result = subprocess.run(
            [sys.executable, str(training_script)],
            check=False, cwd=str(Path(generated_project)),
            env=env,
            capture_output=not verbose,
            text=True,
        )

        if result.returncode == 0:
            console.print("[green]✅ Training completed successfully![/green]")
            return True
        console.print(
            f"[red]❌ Training failed with return code: {result.returncode}[/red]",
        )
        if result.stderr and not verbose:
            console.print(f"[red]Error: {result.stderr}[/red]")
        return False

    except Exception as e:
        console.print(f"[red]❌ Error executing pretrained training: {e}[/red]")
        if verbose:
            console.print_exception()
        return False


def execute_yolox_training(
    config_path: Path,
    output_dir: Path,
    train_args: list,
    verbose: bool = False,
) -> bool:
    """Execute YOLOX training using the generated configuration."""
    try:
        # Load configuration
        with open(config_path) as f:
            config = json.load(f)

        # Path to the YOLOX training template
        project_root = Path(__file__).parent.parent.parent.parent
        yolox_template = project_root / "public" / "Cookiecutter" / "YOLOX"

        # Generate the training project using cookiecutter
        from cookiecutter.main import cookiecutter

        generated_project = cookiecutter(
            str(yolox_template),
            extra_context=config,
            output_dir=str(output_dir),
            no_input=True,
        )

        # Execute the generated training script
        training_script = Path(generated_project) / "python" / "train.py"

        if not training_script.exists():
            console.print(f"[red]❌ Training script not found: {training_script}[/red]")
            return False

        console.print(
            f"[green]🚀 Executing YOLOX training script: {training_script}[/green]",
        )

        # Prepare the command with arguments
        cmd = [sys.executable, str(training_script)] + train_args

        console.print(f"[blue]🔧 Command: {' '.join(cmd)}[/blue]")

        # Run the training script
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(generated_project) / "python")

        result = subprocess.run(
            cmd,
            check=False, cwd=str(Path(generated_project)),
            env=env,
            capture_output=not verbose,
            text=True,
        )

        if result.returncode == 0:
            console.print("[green]✅ YOLOX training completed successfully![/green]")
            return True
        console.print(
            f"[red]❌ YOLOX training failed with return code: {result.returncode}[/red]",
        )
        if result.stderr and not verbose:
            console.print(f"[red]Error: {result.stderr}[/red]")
        return False

    except Exception as e:
        console.print(f"[red]❌ Error executing YOLOX training: {e}[/red]")
        if verbose:
            console.print_exception()
        return False


def setup_training_callback(progress_callback=None):
    """Set up a training callback function that can be used with the training scripts."""

    def callback(value: float, epoch: int = 0, loss: float = 0.0, acc: float = 0.0):
        """Training progress callback."""
        if progress_callback:
            progress_callback(value, epoch, loss, acc)
        else:
            # Default console output
            console.print(
                f"Epoch {epoch} - Progress: {value:.1f}% - Loss: {loss:.4f} - Acc: {acc:.4f}",
            )

    return callback


def create_training_progress() -> Progress:
    """Create a Rich progress bar for training monitoring."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )
