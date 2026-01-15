#!/usr/bin/env python3
"""System Utilities

System information and management utilities.
"""

import os
import platform
from pathlib import Path
from typing import Any

import psutil
import torch
from rich.console import Console

console = Console()


def get_system_info() -> dict[str, Any]:
    """Get comprehensive system information."""
    try:
        # Basic system info
        system_info = {
            "os": platform.system(),
            "os_version": platform.release(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor() or "Unknown",
            "python_version": platform.python_version(),
        }

        # Hardware info
        system_info["cpu_count"] = os.cpu_count() or 1
        system_info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)

        # PyTorch info
        system_info["torch_version"] = torch.__version__
        system_info["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            system_info["cuda_version"] = torch.version.cuda
            system_info["gpu_count"] = torch.cuda.device_count()
            system_info["current_device"] = (
                torch.cuda.get_device_name(0)
                if torch.cuda.device_count() > 0
                else "No GPU"
            )
        else:
            system_info["cuda_version"] = "Not available"
            system_info["gpu_count"] = 0
            system_info["current_device"] = "CPU"

        return system_info

    except Exception as e:
        console.print(f"[red]❌ Error getting system info: {e}[/red]")
        return {
            "os": "Unknown",
            "os_version": "Unknown",
            "architecture": "Unknown",
            "processor": "Unknown",
            "python_version": "Unknown",
            "cpu_count": 1,
            "memory_gb": 0,
            "torch_version": "Unknown",
            "cuda_available": False,
            "cuda_version": "Unknown",
            "gpu_count": 0,
            "current_device": "Unknown",
        }


def get_gpu_info() -> list[dict[str, Any]]:
    """Get detailed GPU information."""
    if not torch.cuda.is_available():
        return []

    gpu_info = []
    try:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append(
                {
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count,
                    "max_threads_per_multi_processor": props.max_threads_per_multi_processor,
                },
            )
    except Exception as e:
        console.print(f"[red]❌ Error getting GPU info: {e}[/red]")

    return gpu_info


def get_memory_info() -> dict[str, Any]:
    """Get detailed memory information."""
    try:
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "usage_percent": memory.percent,
            "swap_total_gb": round(psutil.swap_memory().total / (1024**3), 2),
            "swap_used_gb": round(psutil.swap_memory().used / (1024**3), 2),
        }
    except Exception as e:
        console.print(f"[red]❌ Error getting memory info: {e}[/red]")
        return {}


def get_disk_info(path: Path = Path.cwd()) -> dict[str, Any]:
    """Get disk usage information for a path."""
    try:
        usage = psutil.disk_usage(path)
        return {
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "usage_percent": usage.percent,
        }
    except Exception as e:
        console.print(f"[red]❌ Error getting disk info: {e}[/red]")
        return {}


def check_dependencies() -> dict[str, Any]:
    """Check if all required dependencies are installed."""
    required_packages = [
        "torch",
        "torchvision",
        "torchaudio",
        "rich",
        "typer",
        "jinja2",
        "cookiecutter",
        "pillow",
        "numpy",
        "matplotlib",
        "tensorboard",
        "psutil",
    ]

    dependencies = {}

    for package in required_packages:
        try:
            __import__(package)
            dependencies[package] = {
                "installed": True,
                "version": "Unknown",
            }

            # Try to get version of each package
            try:
                module = __import__(package)
                if hasattr(module, "__version__"):
                    dependencies[package]["version"] = module.__version__
                elif hasattr(module, "version"):
                    dependencies[package]["version"] = module.version
            except:
                pass

        except ImportError:
            dependencies[package] = {
                "installed": False,
                "version": "Not installed",
            }

    return dependencies


def diagnose_system() -> dict[str, Any]:
    """Perform comprehensive system diagnosis."""
    diagnosis = {
        "system": get_system_info(),
        "memory": get_memory_info(),
        "disk": get_disk_info(),
        "gpus": get_gpu_info(),
        "dependencies": check_dependencies(),
        "issues": [],
        "warnings": [],
        "recommendations": [],
    }

    # Analyze system for issues
    system = diagnosis["system"]
    memory = diagnosis["memory"]
    disk = diagnosis["disk"]

    # Check Python version
    python_version = tuple(map(int, system["python_version"].split(".")[:2]))
    if python_version < (3, 8):
        diagnosis["issues"].append("Python version too old (need >= 3.8)")
    elif python_version < (3, 9):
        diagnosis["warnings"].append("Python version < 3.9 may have limited features")

    # Check memory
    if memory.get("total_gb", 0) < 4:
        diagnosis["issues"].append(
            f"Insufficient RAM: {memory.get('total_gb', 0):.1f}GB (need >= 4GB)",
        )
    elif memory.get("total_gb", 0) < 8:
        diagnosis["warnings"].append(
            f"Low RAM: {memory.get('total_gb', 0):.1f}GB (recommended >= 8GB)",
        )

    # Check disk space
    if disk.get("free_gb", 0) < 5:
        diagnosis["issues"].append(
            f"Insufficient disk space: {disk.get('free_gb', 0):.1f}GB free (need >= 5GB)",
        )
    elif disk.get("free_gb", 0) < 10:
        diagnosis["warnings"].append(
            f"Low disk space: {disk.get('free_gb', 0):.1f}GB free (recommended >= 10GB)",
        )

    # Check PyTorch CUDA
    if system["cuda_available"]:
        diagnosis["recommendations"].append(
            "CUDA is available - consider using GPU for faster training",
        )
    else:
        diagnosis["recommendations"].append(
            "CUDA not available - training will use CPU (slower)",
        )

    # Check dependencies
    missing_deps = [
        pkg for pkg, info in diagnosis["dependencies"].items() if not info["installed"]
    ]
    if missing_deps:
        diagnosis["issues"].extend(
            [f"Missing dependency: {pkg}" for pkg in missing_deps],
        )
        diagnosis["recommendations"].append(
            f"Install missing dependencies: pip install {' '.join(missing_deps)}",
        )

    return diagnosis


def print_diagnosis_report(diagnosis: dict[str, Any]):
    """Print a formatted diagnosis report."""
    console.print("[bold blue]🔍 System Diagnosis Report[/bold blue]")

    # System info
    system = diagnosis["system"]
    console.print("\n[bold]System:[/bold]")
    console.print(f"  OS: {system['os']} {system['os_version']}")
    console.print(f"  CPU: {system['processor']} ({system['cpu_count']} cores)")
    console.print(f"  Memory: {diagnosis['memory'].get('total_gb', 0):.1f}GB")
    console.print(f"  Disk: {diagnosis['disk'].get('free_gb', 0):.1f}GB free")

    # PyTorch info
    console.print("\n[bold]PyTorch:[/bold]")
    console.print(f"  Version: {system['torch_version']}")
    console.print(
        f"  CUDA: {'✅ Available' if system['cuda_available'] else '❌ Not available'}",
    )

    if diagnosis["gpus"]:
        console.print(f"  GPUs: {len(diagnosis['gpus'])}")
        for gpu in diagnosis["gpus"]:
            console.print(f"    • {gpu['name']} ({gpu['total_memory_gb']:.1f}GB)")

    # Dependencies
    console.print("\n[bold]Dependencies:[/bold]")
    deps = diagnosis["dependencies"]
    for package, info in deps.items():
        status = "✅" if info["installed"] else "❌"
        version = (
            f" ({info['version']})"
            if info["installed"] and info["version"] != "Unknown"
            else ""
        )
        console.print(f"  {status} {package}{version}")

    # Issues and warnings
    if diagnosis["issues"]:
        console.print(
            f"\n[bold red]❌ Critical Issues ({len(diagnosis['issues'])}):[/bold red]",
        )
        for issue in diagnosis["issues"]:
            console.print(f"  • {issue}")

    if diagnosis["warnings"]:
        console.print(
            f"\n[bold yellow]⚠️  Warnings ({len(diagnosis['warnings'])}):[/bold yellow]",
        )
        for warning in diagnosis["warnings"]:
            console.print(f"  • {warning}")

    if diagnosis["recommendations"]:
        console.print(
            f"\n[bold blue]💡 Recommendations ({len(diagnosis['recommendations'])}):[/bold blue]",
        )
        for rec in diagnosis["recommendations"]:
            console.print(f"  • {rec}")

    # Overall status
    if not diagnosis["issues"]:
        console.print(
            "\n[bold green]🎉 System is ready for neural network development![/bold green]",
        )
    else:
        console.print(
            f"\n[bold red]⚠️  System has {len(diagnosis['issues'])} critical issues that need to be resolved[/bold red]",
        )
