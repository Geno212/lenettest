"""Training MCP Server

MCP tools for training neural network models (manual, pretrained, and YOLOX).
Converted from CLI commands to MCP tools following the standardized pattern.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import sys
import json
import logging
from fastmcp import Context
from src.utils.Cookiecutter import Cookiecutter
from src.paths.SystemPaths import SystemPaths
from .main import arch_mcp, arch_manager_store

logger = logging.getLogger(__name__)

def setup_training_environment(config: dict, output_dir: Path, training_type: str) -> tuple:
    """Set up the training environment exactly like the GUI does.
    
    Args:
        config: The architecture configuration dictionary.
        output_dir: Directory where training output will be saved.
        training_type: Type of training ("manual", "pretrained", or "yolox").
    
    Returns:
        tuple: (cookiecutter, template_dir, jinja_json_path, cookie_json_path)
    
    Raises:
        ValueError: If training_type is unknown.
    """
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
    elif training_type in ["pretrained", "yolox"]:
        # Handle complex model selection like GUI does
        if "pretrained" in config and "value" in config["pretrained"]:
            model_name = config["pretrained"]["value"]
            sys_paths.update_paths(model_name=model_name)
        template_dir = sys_paths.transfer_template_dir
        jinja_json_path = sys_paths.transfer_jinja_json
        cookie_json_path = sys_paths.transfer_cookie_json
    else:
        raise ValueError(f"Unknown training type: {training_type}")
    
    return cookiecutter, template_dir, jinja_json_path, cookie_json_path


@arch_mcp.tool()
async def train_manual_model(
    ctx: Context,
    project_output: Path,
    output_dir: Path = Path.cwd() / "outputs" / "manual",
    log_dir: Path = Path("data/tensorboardlogs"),
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train a manual (custom architecture) model using JSON configuration.

    This function trains a neural network model with a custom architecture defined in the configuration file.
    It sets up the training environment, generates the training code using cookiecutter templates,
    and executes the training process with progress monitoring.

    Args:
        ctx: FastMCP context containing the architecture manager state.
        config_file: Path to manual training JSON configuration file. If None, uses current architecture.
        output_dir: Output directory for manual training (default: current_dir/outputs/manual).
        verbose: Enable verbose output with detailed progress information.

    Returns:
        dict: A dictionary containing operation status and training details.
            - status: "success", "error", or "warning"
            - message: Human-readable status message
            - details: Dictionary with training information including:
                - output_dir: Where training files were generated
                - log_dir: TensorBoard logs directory
                - training_script: Path to the generated training script
                - config_file: Configuration file used (if provided)
                - progress: Training progress information

    Raises:
        FileNotFoundError: If config_file doesn't exist.
        Exception: If training setup or execution fails.
    """
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    # project_output is expected to be the path to the generated project directory.
    training_script = Path(project_output) / "Manual_Output" / "main.py"

    if not training_script.exists():
        return {
            "status": "error",
            "message": f"Training script not found at: {training_script}",
            "details": {
                "expected_path": str(training_script),
                "project_output": str(project_output),
            },
        }

    # Use provided log_dir (default: data/tensorboardlogs)
    log_dir_str = str(log_dir)

    # Import and run the training module directly (same as GUI)
    sys.path.insert(0, str(training_script.parent))
    try:
        # Dynamic import of the training module
        import importlib.util
        spec = importlib.util.spec_from_file_location("manual_train", training_script)
        if spec and spec.loader:
            manual_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(manual_module)

            # Create a progress callback
            progress_tracker = {"current": 0, "total": 100}

            def progress_callback(progress_value):
                progress_tracker["current"] = min(progress_value, 100)
                if verbose:
                    logger.info(f"Training progress: {progress_value:.1f}%")

            # Run training
            if hasattr(manual_module, "train"):
                manual_module.train(callback=progress_callback, logdir=log_dir_str)
            else:
                return {
                    "status": "error",
                    "message": "Training module does not have a 'train' function",
                    "details": {"training_script": str(training_script)},
                }

            return {
                "status": "success",
                "message": "Manual training completed successfully!",
                "details": {
                    "output_dir": str(output_dir),
                    "log_dir": log_dir_str,
                    "training_script": str(training_script),
                    "project_output": str(project_output),
                    "progress": progress_tracker["current"],
                },
            }
        else:
            return {
                "status": "error",
                "message": "Failed to load training module",
                "details": {"training_script": str(training_script)},
            }
    except ImportError as e:
        return {
            "status": "error",
            "message": f"Failed to import training module: {e}",
            "details": {"error": str(e), "training_script": str(training_script)},
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Training failed: {e}",
            "details": {"error": str(e), "training_script": str(training_script)},
        }
    finally:
        # Clean up sys.path
        if str(training_script.parent) in sys.path:
            sys.path.remove(str(training_script.parent))


@arch_mcp.tool()
async def train_pretrained_model(
    ctx: Context,
    project_output: Path,
    output_dir: Path = Path.cwd() / "outputs" / "pretrained",
    log_dir: Path = Path("data/tensorboardlogs"),
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train a pretrained model with transfer learning using JSON configuration.

    This function trains a neural network model using transfer learning with a pretrained base model.
    It sets up the training environment, generates the training code using cookiecutter templates,
    and executes the training process with progress monitoring. The pretrained model is fine-tuned
    on your specific dataset.

    Args:
        ctx: FastMCP context containing the architecture manager state.
        config_file: Path to pretrained training JSON configuration file. If None, uses current architecture.
        output_dir: Output directory for pretrained training (default: current_dir/outputs/pretrained).
        verbose: Enable verbose output with detailed progress information.

    Returns:
        dict: A dictionary containing operation status and training details.
            - status: "success", "error", or "warning"
            - message: Human-readable status message
            - details: Dictionary with training information including:
                - output_dir: Where training files were generated
                - log_dir: TensorBoard logs directory
                - training_script: Path to the generated training script
                - pretrained_model: Name of the pretrained model used
                - config_file: Configuration file used (if provided)
                - progress: Training progress information

    Raises:
        FileNotFoundError: If config_file doesn't exist.
        Exception: If training setup or execution fails.
    """
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    # project_output is expected to be the path to the generated project directory.
    training_script = Path(project_output) / "Pretrained_Output" / "main.py"

    if not training_script.exists():
        return {
            "status": "error",
            "message": f"Training script not found at: {training_script}",
            "details": {
                "expected_path": str(training_script),
                "project_output": str(project_output),
            },
        }

    # Use provided log_dir (default: data/tensorboardlogs)
    log_dir_str = str(log_dir)


    # Import and run the training module directly (same as GUI)
    sys.path.insert(0, str(training_script.parent))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("pretrained_train", training_script)
        if spec and spec.loader:
            pretrained_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pretrained_module)

            progress_tracker = {"current": 0, "total": 100}

            def progress_callback(progress_value):
                progress_tracker["current"] = min(progress_value, 100)
                if verbose:
                    logger.info(f"Training progress: {progress_value:.1f}%")

            if hasattr(pretrained_module, "train"):
                pretrained_module.train(callback=progress_callback, logdir=log_dir_str)
            else:
                return {
                    "status": "error",
                    "message": "Training module does not have a 'train' function",
                    "details": {"training_script": str(training_script)},
                }

            return {
                "status": "success",
                "message": "Pretrained training completed successfully!",
                "details": {
                    "output_dir": str(output_dir),
                    "log_dir": log_dir_str,
                    "training_script": str(training_script),
                    "project_output": str(project_output),
                    "progress": progress_tracker["current"],
                },
            }
        else:
            return {
                "status": "error",
                "message": "Failed to load training module",
                "details": {"training_script": str(training_script)},
            }
    except ImportError as e:
        return {
            "status": "error",
            "message": f"Failed to import training module: {e}",
            "details": {"error": str(e), "training_script": str(training_script)},
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Training failed: {e}",
            "details": {"error": str(e), "training_script": str(training_script)},
        }
    finally:
        if str(training_script.parent) in sys.path:
            sys.path.remove(str(training_script.parent))


@arch_mcp.tool()
async def train_yolox_model(
    ctx: Context,
    project_output: Path,
    output_dir: Path = Path.cwd() / "outputs" / "yolox",
    log_dir: Path = Path("data/tensorboardlogs"),
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train a YOLOX object detection model using JSON configuration.

    This function trains a YOLOX model for object detection tasks. YOLOX is a high-performance
    anchor-free object detector. It sets up the training environment, generates the training code
    using cookiecutter templates, and executes the training process with progress monitoring.

    Args:
        ctx: FastMCP context containing the architecture manager state.
        config_file: Path to YOLOX training JSON configuration file. If None, uses current architecture.
        output_dir: Output directory for YOLOX training (default: current_dir/outputs/yolox).
        verbose: Enable verbose output with detailed progress information.

    Returns:
        dict: A dictionary containing operation status and training details.
            - status: "success", "error", or "warning"
            - message: Human-readable status message
            - details: Dictionary with training information including:
                - output_dir: Where training files were generated
                - log_dir: TensorBoard logs directory
                - training_script: Path to the generated training script
                - yolox_model: Name of the YOLOX model variant used
                - config_file: Configuration file used (if provided)
                - progress: Training progress information

    Raises:
        FileNotFoundError: If config_file doesn't exist.
        Exception: If training setup or execution fails.
    """
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    # project_output is expected to be the path to the generated project directory.
    training_script = Path(project_output) / "Pretrained_Output" / "main.py"

    if not training_script.exists():
        return {
            "status": "error",
            "message": f"Training script not found at: {training_script}",
            "details": {
                "expected_path": str(training_script),
                "project_output": str(project_output),
            },
        }

    # Use provided log_dir (default: data/tensorboardlogs)
    log_dir_str = str(log_dir)


    # Import and run the training module directly (same as GUI)
    sys.path.insert(0, str(training_script.parent))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("yolox_train", training_script)
        if spec and spec.loader:
            yolox_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(yolox_module)

            progress_tracker = {"current": 0, "total": 100}

            def progress_callback(progress_value):
                progress_tracker["current"] = min(progress_value, 100)
                if verbose:
                    logger.info(f"Training progress: {progress_value:.1f}%")

            if hasattr(yolox_module, "train"):
                yolox_module.train(callback=progress_callback, logdir=log_dir_str)
            else:
                return {
                    "status": "error",
                    "message": "Training module does not have a 'train' function",
                    "details": {"training_script": str(training_script)},
                }

            return {
                "status": "success",
                "message": "YOLOX training completed successfully!",
                "details": {
                    "output_dir": str(output_dir),
                    "log_dir": log_dir_str,
                    "training_script": str(training_script),
                    "project_output": str(project_output),
                    "progress": progress_tracker["current"],
                },
            }
        else:
            return {
                "status": "error",
                "message": "Failed to load training module",
                "details": {"training_script": str(training_script)},
            }
    except ImportError as e:
        return {
            "status": "error",
            "message": f"Failed to import training module: {e}",
            "details": {"error": str(e), "training_script": str(training_script)},
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Training failed: {e}",
            "details": {"error": str(e), "training_script": str(training_script)},
        }
    finally:
        if str(training_script.parent) in sys.path:
            sys.path.remove(str(training_script.parent))


@arch_mcp.tool()
async def get_training_status(
    ctx: Context,
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Get the status of the most recent training run.

    This function checks the TensorBoard logs directory to retrieve information about
    the most recent training run, including the log directory location and available
    training metrics.

    Args:
        ctx: FastMCP context containing the architecture manager state.
        log_dir: Optional path to the TensorBoard logs directory. If None, uses default location.

    Returns:
        dict: A dictionary containing operation status and training information.
            - status: "success", "error", or "info"
            - message: Human-readable status message
            - details: Dictionary with training status including:
                - log_dir: Path to TensorBoard logs
                - has_logs: Whether training logs exist
                - tensorboard_command: Command to view logs in TensorBoard
    """
    if log_dir is None:
        sys_paths = SystemPaths()
        log_dir = Path(sys_paths.log_path)
    
    if not log_dir.exists():
        return {
            "status": "info",
            "message": "No training logs found",
            "details": {
                "log_dir": str(log_dir),
                "has_logs": False
            }
        }
    
    # Check if there are any log files
    has_logs = any(log_dir.rglob("events.out.tfevents.*"))
    
    return {
        "status": "success",
        "message": "Training status retrieved successfully",
        "details": {
            "log_dir": str(log_dir),
            "has_logs": has_logs,
            "tensorboard_command": f"tensorboard --logdir={log_dir}"
        }
    }
