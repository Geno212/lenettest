import sys
from pathlib import Path

import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.cli.utils.console import console
from src.paths.SystemPaths import SystemPaths
from src.utils.Cookiecutter import Cookiecutter


def setup_training_environment(
    config: dict,
    output_dir: Path,
    training_type: str,
) -> tuple:
    """Set up the training environment exactly like the GUI does."""
    # Initialize the same classes the GUI uses
    sys_paths = SystemPaths()
    cookiecutter = Cookiecutter(sys_paths.jinja_templates, debug=False)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set log directory in config (same as GUI)
    config["log_dir"] = str(sys_paths.log_path)

    # Update paths based on training type
    if training_type == "manual":
        template_dir = sys_paths.manual_template_dir
        jinja_json_path = sys_paths.manual_jinja_json
        cookie_json_path = sys_paths.manual_cookie_json
    elif training_type == "pretrained":
        # Handle complex model selection like GUI does
        if "pretrained" in config and "value" in config["pretrained"]:
            model_name = config["pretrained"]["value"]
            sys_paths.update_paths(model_name=model_name)
        template_dir = sys_paths.transfer_template_dir
        jinja_json_path = sys_paths.transfer_jinja_json
        cookie_json_path = sys_paths.transfer_cookie_json
    else:
        console.print(f"[red]❌ Unknown training type: {training_type}[/red]")
        raise typer.Exit(1)

    return cookiecutter, template_dir, jinja_json_path, cookie_json_path


def run_training_with_gui_flow(
    config: dict,
    output_dir: Path,
    training_type: str,
    verbose: bool = False,
) -> bool:
    """Run training using the exact same flow as the GUI."""
    try:
        # Set up training environment
        cookiecutter, template_dir, jinja_json_path, cookie_json_path = (
            setup_training_environment(config, output_dir, training_type)
        )

        console.print(
            f"[blue]🚀 Setting up {training_type} training environment...[/blue]",
        )

        # Generate project using cookiecutter template (same as GUI)
        with console.status("[bold green]Generating training project..."):
            project_output = cookiecutter.render_cookiecutter_template_cli(
                data=config,
                src_cookie_json_path=jinja_json_path,
                output_cookie_json_path=cookie_json_path,
                template_dir=template_dir,
                output_path=str(output_dir),
            )

        if not project_output:
            console.print("[red]❌ Failed to generate training project[/red]")
            return False

        console.print(
            f"[green]✅ Training project generated at: {project_output}[/green]",
        )

        # Find the generated training script
        if training_type == "manual":
            training_script = (
                Path(project_output) / "Manual_Output" / "python" / "manual.py"
            )
        elif training_type == "pretrained":
            training_script = (
                Path(project_output) / "Pretrained_Output" / "python" / "pretrained.py"
            )

        if not training_script.exists():
            console.print(
                f"[red]❌ Training script not found at: {training_script}[/red]",
            )
            return False

        console.print(f"[blue]📊 Starting {training_type} training...[/blue]")
        console.print(f"[dim]Training script: {training_script}[/dim]")

        # Import and run the training module directly (same as GUI)
        sys.path.insert(0, str(training_script.parent))

        try:
            if training_type == "manual":
                from manual import train
            elif training_type == "pretrained":
                from pretrained import train

            # Create a progress callback like the GUI does
            progress_tracker = {"current": 0, "total": 100}

            def progress_callback(progress_value):
                progress_tracker["current"] = min(progress_value, 100)
                if verbose:
                    console.print(
                        f"[dim]Training progress: {progress_value:.1f}%[/dim]",
                    )

            # Start training with progress monitoring
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Training model...", total=100)

                def training_callback(value):
                    progress_callback(value)
                    progress.update(task, completed=value)

                # Run training with same parameters as GUI
                log_dir = config.get("log_dir", "data/tensorboardlogs")
                train(callback=training_callback, logdir=log_dir)

                progress.update(task, completed=100)

            console.print(
                f"[green]✅ {training_type.title()} training completed successfully![/green]",
            )
            console.print(f"[blue]📊 TensorBoard logs available at: {log_dir}[/blue]")
            console.print(
                f"[dim]To view training graphs: tensorboard --logdir={log_dir}[/dim]",
            )

            return True

        except ImportError as e:
            console.print(f"[red]❌ Failed to import training module: {e}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]❌ Training failed: {e}[/red]")
            if verbose:
                import traceback

                traceback.print_exc()
            return False
        finally:
            # Clean up sys.path
            if str(training_script.parent) in sys.path:
                sys.path.remove(str(training_script.parent))

    except Exception as e:
        console.print(f"[red]❌ Training setup failed: {e}[/red]")
        if verbose:
            import traceback

            traceback.print_exc()
        return False
