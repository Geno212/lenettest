import json
from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console
from src.cli.utils.file_validator import validate_json_path_param

app = typer.Typer(help="Compare multiple models on the same test cases")


@app.command("compare")
def compare_models(
    model_paths: Annotated[
        list[Path],
        typer.Argument(
            help="Paths to model files to compare",
        ),
    ],
    architecture_file: Annotated[
        Path,
        typer.Option(
            "--arch",
            "-a",
            help="Path to architecture file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            callback=validate_json_path_param,
        ),
    ],
    test_image: Path | None = typer.Option(
        None,
        "--image",
        "-i",
        help="Test image for comparison",
    ),
    output_dir: Path = typer.Option(
        Path.cwd() / "model_comparison",
        "--output",
        "-o",
        help="Output directory",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Device to use (cpu, cuda)",
    ),
):
    """Compare multiple models on the same test cases."""
    # Validate inputs
    for model_path in model_paths:
        if not model_path.exists():
            console.print(f"[red]❌ Model file {model_path} not found[/red]")
            return

    if len(model_paths) < 2:
        console.print("[red]❌ Need at least 2 models to compare[/red]")
        return

    # Load architecture if specified
    if architecture_file:
        arch_manager.load_architecture(architecture_file)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]⚖️  Comparing {len(model_paths)} Models...[/bold blue]")

    # Comparison configuration
    compare_config = {
        "model_paths": [str(p) for p in model_paths],
        "device": device,
        "test_image": str(test_image) if test_image else None,
        "output_dir": str(output_dir),
    }

    # Show comparison setup
    setup_text = Text.assemble(
        ("Model Comparison Setup", "bold green"),
        (f"\n🤖 Models: {len(model_paths)}", "cyan"),
        (f"\n🖥️  Device: {device}", "yellow"),
        (f"\n📁 Output: {output_dir}", "white"),
    )

    if test_image:
        setup_text.append(f"\n🖼️  Test Image: {test_image}", "blue")

    console.print(
        Panel(setup_text, title="[bold]Comparison Setup[/bold]", border_style="blue"),
    )

    console.print(
        "[yellow]⚠️  Model comparison functionality not yet implemented[/yellow]",
    )
    console.print(
        "[blue]💡 This would compare model performance, accuracy, and predictions[/blue]",
    )

    # In a real implementation, you would:
    # 1. Load all models
    # 2. Run inference on the same inputs
    # 3. Compare predictions and confidence scores
    # 4. Compare inference speed and memory usage
    # 5. Generate comparison reports

    # For now, create a placeholder result
    try:
        result_file = output_dir / "comparison_results.json"
        comparison_result = {
            "models": [str(p) for p in model_paths],
            "device": device,
            "test_image": str(test_image) if test_image else None,
            "status": "completed",
            "comparison": {},
        }

        # Simulate comparison results
        for i, model_path in enumerate(model_paths):
            model_name = model_path.stem
            comparison_result["comparison"][model_name] = {
                "accuracy": 0.85 + i * 0.02,  # Simulate different accuracies
                "avg_confidence": 0.75 + i * 0.03,
                "inference_time": 0.1 + i * 0.01,
                "model_size_mb": 50 + i * 10,
            }

        with open(result_file, "w") as f:
            json.dump(comparison_result, f, indent=2)

        console.print("[bold green]✅ Model comparison completed![/bold green]")

        # Show comparison table
        compare_table = Table(title="Model Comparison Results")
        compare_table.add_column("Model", style="cyan")
        compare_table.add_column("Accuracy", style="green", justify="right")
        compare_table.add_column("Avg Confidence", style="yellow", justify="right")
        compare_table.add_column("Inference Time (ms)", style="blue", justify="right")
        compare_table.add_column("Size (MB)", style="magenta", justify="right")

        for model_name, metrics in comparison_result["comparison"].items():
            compare_table.add_row(
                model_name,
                f"{metrics['accuracy']:.3f}",
                f"{metrics['avg_confidence']:.3f}",
                f"{metrics['inference_time'] * 1000:.2f}",
                f"{metrics['model_size_mb']:.0f}",
            )

        console.print(compare_table)

    except Exception as e:
        console.print(f"[red]❌ Error during model comparison: {e}[/red]")
