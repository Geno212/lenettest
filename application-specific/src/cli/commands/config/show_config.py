import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.utils.console import console

app = typer.Typer(help="Show current configuration and parameters.")


@app.command("show")
def show_config():
    """Show current configuration."""
    architecture = arch_manager.current_architecture

    # Model parameters
    misc_params = architecture.get("misc_params", {})

    config_text = Text.assemble(
        ("Model Configuration", "bold blue"),
        ("\n\n📊 Model Parameters:", "bold green"),
        (f"\n  • Input Height: {misc_params.get('height', 'Not set')}", "cyan"),
        (f"\n  • Input Width: {misc_params.get('width', 'Not set')}", "cyan"),
        (f"\n  • Input Channels: {misc_params.get('channels', 'Not set')}", "cyan"),
        (f"\n  • Number of Epochs: {misc_params.get('num_epochs', 'Not set')}", "cyan"),
        (f"\n  • Batch Size: {misc_params.get('batch_size', 'Not set')}", "cyan"),
        (
            f"\n  • Device: {misc_params.get('device', {}).get('value', 'Not set')}",
            "cyan",
        ),
        (
            f"\n  • Dataset: {misc_params.get('dataset', {}).get('value', 'Not set')}",
            "cyan",
        ),
        (f"\n  • Dataset Path: {misc_params.get('dataset_path', 'Not set')}", "cyan"),
    )

    console.print(
        Panel(config_text, title="[bold]Configuration[/bold]", border_style="blue"),
    )

    # Complex parameters (if they exist)
    complex_params = architecture.get("complex_misc_params")
    if complex_params:
        complex_text = Text.assemble(
            ("Complex Parameters", "bold magenta"),
            (
                f"\n  • Data Workers: {complex_params.get('data_num_workers', 'Not set')}",
                "cyan",
            ),
            (
                f"\n  • Eval Interval: {complex_params.get('eval_interval', 'Not set')}",
                "cyan",
            ),
            (
                f"\n  • Warmup Epochs: {complex_params.get('warmup_epochs', 'Not set')}",
                "cyan",
            ),
            (f"\n  • Scheduler: {complex_params.get('scheduler', 'Not set')}", "cyan"),
            (
                f"\n  • Num Classes: {complex_params.get('num_classes', 'Not set')}",
                "cyan",
            ),
            (
                f"\n  • Pretrained Weights: {architecture.get('pretrained_weights', 'Not set')}",
                "cyan",
            ),
            (f"\n  • Log Dir: {architecture.get('log_dir', 'Not set')}", "cyan"),
        )
        console.print(
            Panel(
                complex_text,
                title="[bold]Complex Configuration[/bold]",
                border_style="magenta",
            ),
        )

    # Optimizer
    optimizer = architecture.get("optimizer", {})
    if optimizer:
        optimizer_text = Text.assemble(
            ("Optimizer Configuration", "bold green"),
            (f"\n  • Type: {optimizer.get('type', 'Not set')}", "cyan"),
        )
        if optimizer.get("params"):
            optimizer_text.append("\n  • Parameters:", "dim")
            for key, value in optimizer["params"].items():
                optimizer_text.append(f"\n    - {key}: {value}", "white")
        console.print(Panel(optimizer_text, border_style="green"))
    else:
        console.print(
            Panel(
                "❌ No optimizer configured",
                title="[bold]Optimizer[/bold]",
                border_style="red",
            ),
        )

    # Loss Function
    loss_func = architecture.get("loss_func", {})
    if loss_func:
        loss_text = Text.assemble(
            ("Loss Function Configuration", "bold yellow"),
            (f"\n  • Type: {loss_func.get('type', 'Not set')}", "cyan"),
        )
        if loss_func.get("params"):
            loss_text.append("\n  • Parameters:", "dim")
            for key, value in loss_func["params"].items():
                loss_text.append(f"\n    - {key}: {value}", "white")
        console.print(Panel(loss_text, border_style="yellow"))
    else:
        console.print(
            Panel(
                "❌ No loss function configured",
                title="[bold]Loss Function[/bold]",
                border_style="red",
            ),
        )

    # Scheduler
    scheduler = architecture.get("scheduler", {})
    if scheduler and scheduler.get("type") != "None":
        scheduler_text = Text.assemble(
            ("Scheduler Configuration", "bold magenta"),
            (f"\n  • Type: {scheduler.get('type', 'Not set')}", "cyan"),
        )
        if scheduler.get("params"):
            scheduler_text.append("\n  • Parameters:", "dim")
            for key, value in scheduler["params"].items():
                scheduler_text.append(f"\n    - {key}: {value}", "white")
        console.print(Panel(scheduler_text, border_style="magenta"))
    else:
        console.print(
            Panel(
                "❌ No scheduler configured",
                title="[bold]Scheduler[/bold]",
                border_style="red",
            ),
        )

    # Pretrained Model
    pretrained = architecture.get("pretrained", {})
    if pretrained and pretrained.get("value"):
        pretrained_text = Text.assemble(
            ("Pretrained Model Configuration", "bold blue"),
            (f"\n  • Model: {pretrained.get('value', 'Not set')}", "cyan"),
        )
        # Show complex model details if available
        if "depth" in pretrained and "width" in pretrained:
            pretrained_text.append(
                f"\n  • Depth: {pretrained.get('depth', 'Not set')}",
                "cyan",
            )
            pretrained_text.append(
                f"\n  • Width: {pretrained.get('width', 'Not set')}",
                "cyan",
            )
        console.print(Panel(pretrained_text, border_style="blue"))
    else:
        console.print(
            Panel(
                "❌ No pretrained model configured",
                title="[bold]Pretrained Model[/bold]",
                border_style="red",
            ),
        )
