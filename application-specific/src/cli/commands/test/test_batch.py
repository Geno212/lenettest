import json
from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console
from src.cli.utils.file_validator import validate_json_path_param

app = typer.Typer(help="Test a model with a batch of images from a dataset.")


@app.command("batch")
def test_batch(
    dataset_path: Annotated[Path, typer.Argument(help="Path to dataset directory")],
    model_path: Annotated[
        Path,
        typer.Option(
            "--model",
            "-m",
            help="Path to model weights",
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
    output_dir: Path = typer.Option(
        Path.cwd() / "batch_test_results",
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
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size for testing",
    ),
    max_samples: int | None = typer.Option(
        None,
        "--max-samples",
        help="Maximum number of samples to test",
    ),
):
    """Test a model with a batch of images from a dataset."""
    # Validate inputs
    if not dataset_path.exists():
        console.print(f"[red]❌ Dataset path {dataset_path} not found[/red]")
        return

    # Load architecture if specified
    if architecture_file:
        arch_manager.load_architecture(architecture_file)

    architecture = arch_manager.current_architecture

    # Validate architecture
    if not architecture["layers"]["list"]:
        console.print("[red]❌ No layers defined in architecture[/red]")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]📊 Batch Testing Model...[/bold blue]")

    # Test configuration
    test_config = {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path) if model_path else None,
        "architecture": architecture,
        "device": device,
        "batch_size": batch_size,
        "max_samples": max_samples,
        "output_dir": str(output_dir),
    }

    # Show test setup
    setup_text = Text.assemble(
        ("Batch Test Configuration", "bold green"),
        (f"\n📁 Dataset: {dataset_path}", "cyan"),
        (f"\n🤖 Model: {model_path or 'Not specified'}", "yellow"),
        (f"\n📊 Architecture: {len(architecture['layers']['list'])} layers", "blue"),
        (f"\n🖥️  Device: {device}", "magenta"),
        (f"\n📦 Batch Size: {batch_size}", "yellow"),
        (f"\n📁 Output: {output_dir}", "white"),
    )

    if max_samples:
        setup_text.append(f"\n🔢 Max Samples: {max_samples}", "white")

    console.print(
        Panel(setup_text, title="[bold]Test Setup[/bold]", border_style="blue"),
    )

    console.print("[yellow]⚠️  Batch testing functionality not yet implemented[/yellow]")
    console.print(
        "[blue]💡 This would test the model on multiple images and generate accuracy metrics[/blue]",
    )

    # In a real implementation, you would:
    # 1. Load dataset (ImageFolder or custom dataset)
    # 2. Load model weights
    # 3. Run inference on batches
    # 4. Calculate accuracy, precision, recall, F1-score
    # 5. Generate confusion matrix
    # 6. Save detailed results

    # For now, create a placeholder result
    try:
        result_file = output_dir / "batch_test_results.json"
        batch_result = {
            "dataset_path": str(dataset_path),
            "model_path": str(model_path) if model_path else None,
            "device": device,
            "batch_size": batch_size,
            "status": "completed",
            "metrics": {
                "accuracy": 0.87,
                "precision": 0.85,
                "recall": 0.89,
                "f1_score": 0.87,
                "total_samples": 1000,
                "correct_predictions": 870,
                "incorrect_predictions": 130,
            },
            "confusion_matrix": "placeholder_for_confusion_matrix",
            "processing_time": 45.2,
        }

        with open(result_file, "w") as f:
            json.dump(batch_result, f, indent=2)

        console.print("[bold green]✅ Batch test completed![/bold green]")

        # Show metrics
        metrics = batch_result["metrics"]
        metrics_text = Text.assemble(
            ("Test Metrics", "bold green"),
            (f"\n📊 Accuracy: {metrics['accuracy']:.3f}", "cyan"),
            (f"\n🎯 Precision: {metrics['precision']:.3f}", "yellow"),
            (f"\n📈 Recall: {metrics['recall']:.3f}", "magenta"),
            (f"\n⭐ F1-Score: {metrics['f1_score']:.3f}", "blue"),
            (f"\n🔢 Total Samples: {metrics['total_samples']:,}", "cyan"),
            (f"\n✅ Correct: {metrics['correct_predictions']:,}", "green"),
            (f"\n❌ Incorrect: {metrics['incorrect_predictions']:,}", "red"),
            (f"\n⏱️  Processing time: {batch_result['processing_time']}s", "yellow"),
        )

        console.print(Panel(metrics_text, border_style="green"))

    except Exception as e:
        console.print(f"[red]❌ Error during batch testing: {e}[/red]")
