import os
from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.text import Text

from src.cli.core.architecture_manager import arch_manager
from src.cli.core.generate import generate_manual, generate_pretrained, generate_yolox
from src.cli.utils.console import console
from src.cli.utils.file_validator import validate_json_path_param
from src.paths.SystemPaths import SystemPaths
from src.utils.Cookiecutter import Cookiecutter

app = typer.Typer(help="Generate code from the current architecture")


@app.command("pytorch")
def generate_pytorch(
    architecture_file: Annotated[
        Path,
        typer.Argument(
            help="Path to architecture file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            callback=validate_json_path_param,
        ),
    ],
    output_dir: Path = typer.Option(
        Path.cwd() / "generated",
        "--output",
        "-o",
        help="Output directory",
    ),
    model_name: str = typer.Option(
        "GeneratedModel",
        "--name",
        "-n",
        help="Name for the generated model",
    ),
    include_requirements: bool = typer.Option(
        True,
        "--requirements",
        help="Include requirements.txt",
    ),
):
    """Generate PyTorch code from architecture."""
    # Load architecture
    arch_manager.load_architecture(architecture_file)

    # Show generation plan
    plan_text = Text.assemble(
        ("Generation Plan", "bold green"),
        (f"\n📦 Model Name: {model_name}", "cyan"),
        (f"\n📊 Layers: {len(arch_manager.current_architecture['layers'])}", "yellow"),
        (f"\n📁 Output: {output_dir}", "blue"),
        (
            f"\n Requirements: {'✅' if include_requirements else '❌'}",
            "green" if include_requirements else "red",
        ),
    )

    console.print(
        Panel(plan_text, title="[bold]Generation Plan[/bold]", border_style="blue"),
    )

    try:
        sys_paths = SystemPaths()
        cookiecutter = Cookiecutter(sys_paths.jinja_templates, debug=False)

        # Get correct template paths based on model type (YOLOX vs Pretrained vs Manual)
        pretrained_model = arch_manager.current_architecture.get("pretrained", {}).get(
            "value",
        )

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        sys_paths.jsondir = str(output_dir)

        if pretrained_model and pretrained_model.lower().startswith("yolo"):
            # YOLOX template
            generate_yolox(sys_paths, pretrained_model, cookiecutter, output_dir)
        elif pretrained_model:
            # Pretrained/Transfer Learning template
            generate_pretrained(sys_paths, pretrained_model, cookiecutter, output_dir)
        else:
            # Manual template (no pretrained model)
            generate_manual(sys_paths, cookiecutter, output_dir)

        # Generate the project using cookiecutter
        try:
            console.print(
                "[bold green]✅ PyTorch code generated successfully![/bold green]",
            )
            # Show generated files
            generated_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    rel_path = Path(root) / file
                    generated_files.append(rel_path.relative_to(output_dir))

            if generated_files:
                files_text = Text("Generated Files:", style="bold green")
                for file in sorted(generated_files)[:10]:  # Show first 10
                    files_text.append(f"\n  📄 {file}", "cyan")
                if len(generated_files) > 10:
                    files_text.append(
                        f"\n  ... and {len(generated_files) - 10} more",
                        "dim",
                    )

                console.print(Panel(files_text, border_style="green"))

        except Exception as e:
            console.print(f"[red]❌ Error during code generation: {e}[/red]")
            import traceback

            console.print(f"[red]Full traceback: {traceback.format_exc()}[/red]")
            return

    except ImportError as e:
        console.print(f"[red]❌ Could not import generation modules: {e}[/red]")
        console.print(
            "[yellow]💡 Make sure you're running from the project root directory[/yellow]",
        )
        return
    except Exception as e:
        console.print(f"[red]❌ Code generation failed: {e}[/red]")
        import traceback

        console.print(f"[red]Full traceback: {traceback.format_exc()}[/red]")
        return
