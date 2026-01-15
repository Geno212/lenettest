import os
from pathlib import Path
from typing import Any

from fastmcp import Context

from src.cli.core.architecture_manager import ArchitectureManager
from src.paths import PathsFactory
from src.paths.SystemPaths import SystemPaths
from src.utils.Cookiecutter import Cookiecutter

from .main import arch_mcp


@arch_mcp.tool()
async def generate_pytorch(
    ctx: Context,
    architecture_file: Path,
    output_dir: Path = Path.cwd() / "generated",
    model_name: str = "GeneratedModel",
    include_requirements: bool = True,
) -> dict[str, Any]:
    """Generate PyTorch code from a neural network architecture file.

    This function takes an architecture file and generates complete PyTorch training code
    including model definition, training loop, validation, and utilities. It supports
    different model types (YOLOX, pretrained/transfer learning, or manual).

    Args:
        ctx: The FastMCP context containing session state.
        architecture_file: Path to the JSON architecture file to generate code from.
        output_dir: Directory where the generated code will be saved.
        model_name: Name for the generated model/project.
        include_requirements: Whether to include requirements.txt file.

    Returns:
        dict: A dictionary containing status information and generation details.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with generation information including:
                - output_dir: Where files were generated
                - model_name: Name of the generated model
                - layers_count: Number of layers in the architecture
                - generated_files: List of generated files
                - plan_details: Original generation plan

    Raises:
        FileNotFoundError: If architecture_file does not exist.
        ImportError: If required generation modules cannot be imported.
        Exception: If code generation fails for any other reason.

    """
    arch_manager = ctx.get_state("arch_manager")

    if not architecture_file.exists():
        return {
            "status": "error",
            "message": f"Architecture file {architecture_file} not found",
            "details": {"file_path": str(architecture_file)},
        }

    # Load architecture
    arch_manager.load_architecture(architecture_file)

    # Show generation plan
    plan_details = {
        "model_name": model_name,
        "layers_count": len(arch_manager.current_architecture["layers"]),
        "output_dir": str(output_dir),
        "include_requirements": include_requirements,
    }

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
            result = generate_yolox(
                arch_manager,
                sys_paths,
                pretrained_model,
                cookiecutter,
                output_dir,
            )
        elif pretrained_model:
            # Pretrained/Transfer Learning template
            result = generate_pretrained(
                arch_manager,
                sys_paths,
                pretrained_model,
                cookiecutter,
                output_dir,
            )
        else:
            # Manual template (no pretrained model)
            result = generate_manual(arch_manager, sys_paths, cookiecutter, output_dir)

        if result["status"] != "success":
            return result

        # Generate the project using cookiecutter
        try:
            # Show generated files
            generated_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    rel_path = Path(root) / file
                    generated_files.append(rel_path.relative_to(output_dir))

            return {
                "status": "success",
                "message": "PyTorch code generated successfully",
                "details": {
                    "output_dir": str(output_dir),
                    "model_name": model_name,
                    "layers_count": len(arch_manager.current_architecture["layers"]),
                    "generated_files": [str(f) for f in sorted(generated_files)[:10]],
                    "plan_details": plan_details,
                },
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during code generation: {e}",
                "details": {"error": str(e)},
            }

    except ImportError as e:
        return {
            "status": "error",
            "message": f"Could not import generation modules: {e}",
            "details": {"error": str(e)},
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Code generation failed: {e}",
            "details": {"error": str(e)},
        }


def generate_pretrained(
    arch_manager: ArchitectureManager,
    sys_paths,
    pretrained_model: str,
    cookiecutter: Cookiecutter,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate pretrained/transfer learning project.

    This function generates PyTorch code for transfer learning using a pretrained model
    as the base and adding custom layers on top for fine-tuning.

    Args:
        arch_manager: The ArchitectureManager instance containing the current architecture.
        sys_paths: SystemPaths instance for template path management.
        pretrained_model: Name of the pretrained model to use as base.
        cookiecutter: Cookiecutter instance for template rendering.
        output_dir: Directory where the generated code will be saved.

    Returns:
        dict: A dictionary containing status information and generation details.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with generation information including:
                - output_path: Where files were generated
                - main_py_path: Path to the main training script

    """
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
            return {
                "status": "error",
                "message": f"Generated main.py not found at: {main_py_path}",
                "details": {"expected_path": str(main_py_path)},
            }

        return {
            "status": "success",
            "message": "Pretrained code generated successfully",
            "details": {"output_path": output_path, "main_py_path": str(main_py_path)},
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during pretrained code generation: {e}",
            "details": {"error": str(e)},
        }


def generate_yolox(
    arch_manager: ArchitectureManager,
    sys_paths,
    pretrained_model: str,
    cookiecutter: Cookiecutter,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate YOLOX object detection project.

    This function generates PyTorch code for YOLOX object detection model training
    and deployment with optimized performance for real-time applications.

    Args:
        arch_manager: The ArchitectureManager instance containing the current architecture.
        sys_paths: SystemPaths instance for template path management.
        pretrained_model: Name of the YOLOX model variant to use.
        cookiecutter: Cookiecutter instance for template rendering.
        output_dir: Directory where the generated code will be saved.

    Returns:
        dict: A dictionary containing status information and generation details.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with generation information including:
                - output_path: Where files were generated
                - main_py_path: Path to the main training script

    """
    try:
        jinja_path, template_dir, cookie_json_path = PathsFactory.PathFactory.get_paths(
            sys_paths,
            pretrained_model,
        )

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
            return {
                "status": "error",
                "message": f"Generated main.py not found at: {main_py_path}",
                "details": {"expected_path": str(main_py_path)},
            }

        return {
            "status": "success",
            "message": "YOLOX code generated successfully",
            "details": {"output_path": output_path, "main_py_path": str(main_py_path)},
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during YOLOX code generation: {e}",
            "details": {"error": str(e)},
        }


def generate_manual(
    arch_manager: ArchitectureManager,
    sys_paths,
    cookiecutter: Cookiecutter,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate manual/custom neural network project.

    This function generates PyTorch code for a custom neural network architecture
    without using pretrained models, allowing full control over the model design.

    Args:
        arch_manager: The ArchitectureManager instance containing the current architecture.
        sys_paths: SystemPaths instance for template path management.
        cookiecutter: Cookiecutter instance for template rendering.
        output_dir: Directory where the generated code will be saved.

    Returns:
        dict: A dictionary containing status information and generation details.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with generation information including:
                - output_path: Where files were generated
                - main_py_path: Path to the main training script

    """
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
            return {
                "status": "error",
                "message": f"Generated main.py not found at: {main_py_path}",
                "details": {"expected_path": str(main_py_path)},
            }

        return {
            "status": "success",
            "message": "Manual code generated successfully",
            "details": {"output_path": output_path, "main_py_path": str(main_py_path)},
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during manual code generation: {e}",
            "details": {"error": str(e)},
        }


def handle_stderr(process) -> dict[str, Any]:
    """Handle stderr output from subprocess.

    This function processes error output from a subprocess execution,
    printing each line to the console for debugging purposes.

    Args:
        process: The subprocess object containing stderr stream.

    Returns:
        dict: Status information about stderr handling.
            - status: "completed"
            - output_type: "stderr"

    """
    if process and process.stderr:
        for line in process.stderr:
            print("STDERR:", line.strip())
    return {"status": "completed", "output_type": "stderr"}


def handle_stdout(process) -> dict[str, Any]:
    """Handle stdout output from subprocess.

    This function processes standard output from a subprocess execution,
    printing each line to the console for monitoring purposes.

    Args:
        process: The subprocess object containing stdout stream.

    Returns:
        dict: Status information about stdout handling.
            - status: "completed"
            - output_type: "stdout"

    """
    if process and process.stdout:
        for line in process.stdout:
            print("STDOUT:", line.strip())
    return {"status": "completed", "output_type": "stdout"}


@arch_mcp.tool()
async def list_templates(ctx: Context) -> dict[str, Any]:
    """List available code generation templates.

    This function provides a comprehensive list of all available code generation
    templates including PyTorch training, SystemC hardware, transfer learning,
    and YOLOX deployment templates with their features and descriptions.

    Args:
        ctx: The FastMCP context containing session state.

    Returns:
        dict: A dictionary containing template information.
            - status: "success"
            - message: Human-readable status message
            - details: Dictionary containing:
                - templates_count: Number of available templates
                - templates: List of template dictionaries with name, description, type, and features

    """
    templates = [
        {
            "name": "PyTorch Training",
            "description": "Complete PyTorch training setup with model, trainer, and utilities",
            "type": "Python",
            "features": [
                "Model definition",
                "Training loop",
                "Validation",
                "Checkpointing",
                "Logging",
            ],
        },
        {
            "name": "SystemC Hardware",
            "description": "SystemC implementation for hardware acceleration",
            "type": "SystemC/C++",
            "features": [
                "Hardware modules",
                "Testbench",
                "CMake build",
                "Simulation setup",
            ],
        },
        {
            "name": "Transfer Learning",
            "description": "Pretrained model fine-tuning setup",
            "type": "Python",
            "features": [
                "Pretrained model loading",
                "Transfer learning",
                "Custom classifier",
                "Fine-tuning",
            ],
        },
        {
            "name": "YOLOX Deployment",
            "description": "YOLOX model deployment and optimization",
            "type": "Python",
            "features": [
                "YOLOX integration",
                "Model conversion",
                "Optimization",
                "Deployment",
            ],
        },
    ]

    return {
        "status": "success",
        "message": "Templates retrieved successfully",
        "details": {"templates_count": len(templates), "templates": templates},
    }
