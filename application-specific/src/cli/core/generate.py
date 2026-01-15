from pathlib import Path

from src.cli.utils.console import console
from src.paths import PathsFactory
from src.utils.Cookiecutter import Cookiecutter

from .architecture_manager import arch_manager


def generate_pretrained(
    sys_paths,
    pretrained_model,
    cookiecutter: Cookiecutter,
    output_dir: Path,
):
    """Generate pretrained/transfer learning project and run it."""
    console.print(
        f"[blue]🔍 Using Pretrained template for model: {pretrained_model}[/blue]",
    )

    try:
        # Update paths for the specific pretrained model
        sys_paths.update_paths(pretrained_model)

        # Use transfer learning template paths
        jinja_path = sys_paths.transfer_jinja_json
        template_dir = sys_paths.transfer_template_dir
        cookie_json_path = sys_paths.transfer_cookie_json

        output_path = cookiecutter.render_cookiecutter_template_cli(
            arch_manager.current_architecture,
            jinja_path,
            cookie_json_path,
            template_dir,
            str(output_dir),
        )

        # Verify the main.py file exists before running
        main_py_path = Path(output_path) / "Pretrained_Output" / "main.py"
        if not main_py_path.exists():
            console.print(
                f"[red]❌ Generated main.py not found at: {main_py_path}[/red]",
            )
            return

    except Exception as e:
        console.print(f"[red]❌ Error during pretrained code generation: {e}[/red]")


def generate_yolox(
    sys_paths,
    pretrained_model,
    cookiecutter: Cookiecutter,
    output_dir: Path,
):
    jinja_path, template_dir, cookie_json_path = PathsFactory.PathFactory.get_paths(
        sys_paths,
        pretrained_model,
    )
    console.print(f"[blue]🔍 Using YOLOX template for model: {pretrained_model}[/blue]")
    try:
        output_path = cookiecutter.render_cookiecutter_template_cli(
            arch_manager.current_architecture,
            jinja_path,
            cookie_json_path,
            template_dir,
            str(output_dir),
        )

        # Verify the main.py file exists before running
        main_py_path = Path(output_path) / "YOLOX" / "main.py"
        if not main_py_path.exists():
            console.print(
                f"[red]❌ Generated main.py not found at: {main_py_path}[/red]",
            )
            return

    except Exception as e:
        console.print(f"[red]❌ Error during code generation: {e}[/red]")


def generate_manual(sys_paths, cookiecutter: Cookiecutter, output_dir: Path):
    """Generate manual project and run it."""
    console.print("[blue]🔍 Using Manual template[/blue]")

    try:
        jinja_path = sys_paths.manual_jinja_json
        template_dir = sys_paths.manual_template_dir
        cookie_json_path = sys_paths.manual_cookie_json

        output_path = cookiecutter.render_cookiecutter_template_cli(
            arch_manager.current_architecture,
            jinja_path,
            cookie_json_path,
            template_dir,
            str(output_dir),
        )

        # Verify the main.py file exists before running
        main_py_path = Path(output_path) / "Manual_Output" / "main.py"
        if not main_py_path.exists():
            console.print(
                f"[red]❌ Generated main.py not found at: {main_py_path}[/red]",
            )
            return

    except Exception as e:
        console.print(f"[red]❌ Error during manual code generation: {e}[/red]")


def handle_stderr(process):
    """Handle stderr output from the subprocess"""
    if process and process.stderr:
        for line in process.stderr:
            print("STDERR:", line.strip())


def handle_stdout(process):
    """Handle stdout output from the subprocess"""
    if process and process.stdout:
        for line in process.stdout:
            print("STDOUT:", line.strip())
