import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(help="Set complex model parameters.")


@app.command("complex-params")
def set_complex_params(
    data_num_workers: int | None = typer.Option(
        None,
        "--data-num-workers",
        help="Number of data workers",
    ),
    eval_interval: int | None = typer.Option(
        None,
        "--eval-interval",
        help="Evaluation interval",
    ),
    warmup_epochs: int | None = typer.Option(
        None,
        "--warmup-epochs",
        help="Warmup epochs",
    ),
    scheduler: str | None = typer.Option(
        None,
        "--scheduler",
        help="Scheduler type (cos, linear, etc.)",
    ),
    num_classes: int | None = typer.Option(
        None,
        "--num-classes",
        help="Number of classes",
    ),
    pretrained_weights: str | None = typer.Option(
        None,
        "--weights",
        help="Path to pretrained weights",
    ),
):
    """Set complex model parameters."""
    # Ensure complex fields exist
    arch_manager._ensure_complex_fields()

    architecture = arch_manager.current_architecture
    complex_params = architecture["complex_misc_params"]

    # Update parameters
    updates = []
    if data_num_workers is not None:
        complex_params["data_num_workers"] = data_num_workers
        updates.append(f"data_num_workers={data_num_workers}")
    if eval_interval is not None:
        complex_params["eval_interval"] = eval_interval
        updates.append(f"eval_interval={eval_interval}")
    if warmup_epochs is not None:
        complex_params["warmup_epochs"] = warmup_epochs
        updates.append(f"warmup_epochs={warmup_epochs}")
    if scheduler is not None and scheduler in arch_manager.complex_schedulers:
        complex_params["scheduler"] = scheduler
        updates.append(f"scheduler={scheduler}")
    if num_classes is not None:
        complex_params["num_classes"] = num_classes
        updates.append(f"num_classes={num_classes}")
    if pretrained_weights is not None:
        architecture["pretrained_weights"] = pretrained_weights
        updates.append(f"pretrained_weights={pretrained_weights}")

    if not updates:
        console.print(
            "[yellow]⚠️  No parameters specified. Use --help to see available options.[/yellow]",
        )
        return

    success_text = Text.assemble(
        ("Complex parameters updated!", "bold green"),
        ("\n✅ Updated:", "cyan"),
    )
    for update in updates:
        success_text.append(f"\n  • {update}", "white")

    console.print(
        Panel(
            success_text,
            title="[bold]Complex Parameters[/bold]",
            border_style="green",
        ),
    )
