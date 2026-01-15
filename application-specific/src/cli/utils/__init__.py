#!/usr/bin/env python3
"""CLI Utilities

Utility functions and helpers for the CLI.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from src.Tests.StaticAnalysis import StaticAnalysis

console = Console()

StaticAnalysis = StaticAnalysis("public/Rules/warning_rules.txt", False)



def validate_file_exists(file_path: Path, description: str = "File") -> bool:
    """Validate that a file exists."""
    if not file_path.exists():
        console.print(f"[red]❌ {description} not found: {file_path}[/red]")
        return False
    return True


def validate_directory_exists(dir_path: Path, description: str = "Directory") -> bool:
    """Validate that a directory exists."""
    if not dir_path.exists():
        console.print(f"[red]❌ {description} not found: {dir_path}[/red]")
        return False
    return True


def create_directory(dir_path: Path, description: str = "Directory") -> bool:
    """Create a directory if it doesn't exist."""
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        console.print(f"[red]❌ Error creating {description}: {e}[/red]")
        return False


def save_json_file(
    data: dict[str, Any],
    file_path: Path,
    description: str = "File",
) -> bool:
    """Save data to a JSON file."""
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        console.print(f"[red]❌ Error saving {description}: {e}[/red]")
        return False


def load_json_file(
    file_path: Path,
    description: str = "File",
) -> dict[str, Any] | None:
    """Load data from a JSON file."""
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[red]❌ Error loading {description}: {e}[/red]")
        return None


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"


def count_files_in_directory(dir_path: Path) -> int:
    """Count total files in a directory recursively."""
    if not dir_path.exists():
        return 0

    count = 0
    for root, dirs, files in os.walk(dir_path):
        count += len(files)
    return count


def get_directory_size(dir_path: Path) -> int:
    """Get total size of a directory in bytes."""
    if not dir_path.exists():
        return 0

    total_size = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.exists():
                total_size += file_path.stat().st_size

    return total_size


def validate_architecture(architecture: dict[str, Any]) -> list[str]:
    """Validate an architecture dictionary."""
    issues = []

    # Check required keys
    required_keys = ["layers", "misc_params", "optimizer", "loss_func"]
    for key in required_keys:
        if key not in architecture:
            issues.append(f"❌ Missing required key: {key}")

    # Check layers
    if "layers" in architecture:
        layers = architecture["layers"]
        if not isinstance(layers, list):
            issues.append("❌ Invalid layers format")
        else:
            for i, layer in enumerate(layers):
                if not isinstance(layer, dict) or "type" not in layer:
                    issues.append(f"❌ Layer {i} missing 'type' field")

    # Check misc_params
    if "misc_params" in architecture:
        misc_params = architecture["misc_params"]
        required_params = [
            "width",
            "height",
            "channels",
            "num_epochs",
            "batch_size",
            "device",
        ]
        for param in required_params:
            if param not in misc_params:
                issues.append(f"❌ Missing required parameter: {param}")

    return issues


def format_validation_issues(issues: list[str]) -> str:
    """Format validation issues for display."""
    if not issues:
        return "✅ No issues found"

    return "\n".join(issues)


def get_available_devices() -> list[str]:
    """Get list of available devices."""
    devices = ["cpu"]

    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
    except ImportError:
        pass

    return devices


def get_device_info(device: str) -> dict[str, Any]:
    """Get information about a specific device."""
    if device == "cpu":
        import multiprocessing

        return {
            "type": "CPU",
            "cores": multiprocessing.cpu_count(),
            "memory": "System RAM",
        }

    if device.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                device_id = 0 if device == "cuda" else int(device.split(":")[1])
                if device_id < torch.cuda.device_count():
                    props = torch.cuda.get_device_properties(device_id)
                    return {
                        "type": "CUDA GPU",
                        "name": props.name,
                        "memory_gb": props.total_memory / (1024**3),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
        except:
            pass

    return {"type": "Unknown", "error": "Device not available"}


def print_status(message: str, status: str = "info"):
    """Print a status message with appropriate styling."""
    if status == "success":
        console.print(f"[green]✅ {message}[/green]")
    elif status == "error":
        console.print(f"[red]❌ {message}[/red]")
    elif status == "warning":
        console.print(f"[yellow]⚠️  {message}[/yellow]")
    elif status == "info":
        console.print(f"[blue]ℹ️  {message}[/blue]")
    else:
        console.print(message)


def Analyze(layers):
    violations_list = StaticAnalysis.analyze(
        layers,
    )
    if violations_list:
        console.print(f"[yellow]⚠️  Found {len(violations_list)} violations:[/yellow]")
        for i, violation in enumerate(violations_list, 1):
            console.print(f"  {i}. {violation}")


def Analyze_for_MCP(layers):
    """Analyze layers for MCP compatibility and return structured results.

    Args:
        layers: List of layer dictionaries to analyze

    Returns:
        dict: Analysis results with violations and summary

    """
    violations_list = StaticAnalysis.analyze(
        layers,
    )

    return {
        "violations_count": len(violations_list),
        "violations": violations_list,
        "has_violations": len(violations_list) > 0,
        "summary": f"Found {len(violations_list)} violations"
        if violations_list
        else "No violations found",
    }


def confirm_action(message: str) -> bool:
    """Ask user to confirm an action."""
    return typer.confirm(message)


def progress_callback(current: int, total: int, message: str = "Processing..."):
    """Simple progress callback for use with rich progress bars."""
    return f"{message}: {current}/{total}"


# Export commonly used functions
__all__ = [
    "Analyze",
    "confirm_action",
    "count_files_in_directory",
    "create_directory",
    "format_file_size",
    "format_validation_issues",
    "get_available_devices",
    "get_device_info",
    "get_directory_size",
    "load_json_file",
    "print_status",
    "progress_callback",
    "save_json_file",
    "validate_architecture",
    "validate_directory_exists",
    "validate_file_exists",
]
