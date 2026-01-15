"""Configuration MCP Server

MCP tools for configuring model parameters, optimizer, loss function, and scheduler.
Converted from CLI commands to MCP tools following the standardized pattern.
"""

from typing import Any

from fastmcp import Context

from src.cli.utils import get_available_devices

from .main import arch_mcp


@arch_mcp.tool()
async def show_config(ctx: Context) -> dict[str, Any]:
    """Show current configuration including model parameters, optimizer, loss function, and scheduler.

    Args:
        ctx: FastMCP context containing the architecture manager state.

    Returns:
        dict: A dictionary containing configuration information.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with all configuration sections:
                - model_params: Basic model configuration
                - complex_params: Complex model parameters (if configured)
                - optimizer: Optimizer configuration
                - loss_function: Loss function configuration
                - scheduler: Learning rate scheduler configuration
                - pretrained: Pretrained model configuration

    """
    arch_manager = ctx.get_state("arch_manager")
    architecture = arch_manager.current_architecture

    # Model parameters
    misc_params = architecture.get("misc_params", {})
    model_params = {
        "height": misc_params.get("height", "Not set"),
        "width": misc_params.get("width", "Not set"),
        "channels": misc_params.get("channels", "Not set"),
        "num_epochs": misc_params.get("num_epochs", "Not set"),
        "batch_size": misc_params.get("batch_size", "Not set"),
        "device": misc_params.get("device", {}).get("value", "Not set"),
        "dataset": misc_params.get("dataset", {}).get("value", "Not set"),
        "dataset_path": misc_params.get("dataset_path", "Not set"),
    }

    # Complex parameters (if they exist)
    complex_params_data = None
    complex_params = architecture.get("complex_misc_params")
    if complex_params:
        complex_params_data = {
            "data_num_workers": complex_params.get("data_num_workers", "Not set"),
            "eval_interval": complex_params.get("eval_interval", "Not set"),
            "warmup_epochs": complex_params.get("warmup_epochs", "Not set"),
            "scheduler": complex_params.get("scheduler", "Not set"),
            "num_classes": complex_params.get("num_classes", "Not set"),
            "pretrained_weights": architecture.get("pretrained_weights", "Not set"),
            "log_dir": architecture.get("log_dir", "Not set"),
        }

    # Optimizer
    optimizer = architecture.get("optimizer", {})
    optimizer_data = {
        "configured": bool(optimizer),
        "type": optimizer.get("type", "Not set") if optimizer else "Not set",
        "params": optimizer.get("params", {}) if optimizer else {},
    }

    # Loss Function
    loss_func = architecture.get("loss_func", {})
    loss_func_data = {
        "configured": bool(loss_func),
        "type": loss_func.get("type", "Not set") if loss_func else "Not set",
        "params": loss_func.get("params", {}) if loss_func else {},
    }

    # Scheduler
    scheduler = architecture.get("scheduler", {})
    scheduler_configured = scheduler and scheduler.get("type") != "None"
    scheduler_data = {
        "configured": scheduler_configured,
        "type": scheduler.get("type", "Not set") if scheduler else "Not set",
        "params": scheduler.get("params", {}) if scheduler else {},
    }

    # Pretrained Model
    pretrained = architecture.get("pretrained", {})
    pretrained_data = {
        "configured": bool(pretrained and pretrained.get("value")),
        "model": pretrained.get("value", "Not set") if pretrained else "Not set",
    }
    if pretrained and "depth" in pretrained and "width" in pretrained:
        pretrained_data["depth"] = pretrained.get("depth")
        pretrained_data["width"] = pretrained.get("width")

    return {
        "status": "success",
        "message": "Configuration retrieved successfully",
        "details": {
            "model_params": model_params,
            "complex_params": complex_params_data,
            "optimizer": optimizer_data,
            "loss_function": loss_func_data,
            "scheduler": scheduler_data,
            "pretrained": pretrained_data,
        },
    }


@arch_mcp.tool()
async def set_model_params(
    ctx: Context,
    height: int | None = None,
    width: int | None = None,
    channels: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    dataset: str | None = None,
    dataset_path: str | None = None,
) -> dict[str, Any]:
    """Set model parameters.

    Args:
        ctx: FastMCP context containing the architecture manager state.
        height: Input height for the model.
        width: Input width for the model.
        channels: Number of input channels.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        device: Device to use (cpu, cuda, cuda:0, etc.).
        dataset: Dataset name.
        dataset_path: Path to the dataset directory.

    Returns:
        dict: A dictionary containing operation status.
            - status: "success", "error", or "warning"
            - message: Human-readable status message
            - details: Dictionary with updated parameters

    """
    arch_manager = ctx.get_state("arch_manager")
    architecture = arch_manager.current_architecture
    misc_params = architecture["misc_params"]

    # Update parameters
    updates = {}
    if height is not None:
        misc_params["height"] = height
        updates["height"] = height
    if width is not None:
        misc_params["width"] = width
        updates["width"] = width
    if channels is not None:
        misc_params["channels"] = channels
        updates["channels"] = channels
    if epochs is not None:
        misc_params["num_epochs"] = epochs
        updates["epochs"] = epochs
    if batch_size is not None:
        misc_params["batch_size"] = batch_size
        updates["batch_size"] = batch_size
    if device is not None:
        device_index = -1
        devices = get_available_devices()
        if device in devices:
            device_index = devices.index(device)
        misc_params["device"] = {"value": device, "index": device_index}
        updates["device"] = device
    if dataset is not None:
        dataset_index = -1
        if dataset in arch_manager.datasets:
            dataset_index = arch_manager.datasets.index(dataset)
        misc_params["dataset"] = {"value": dataset, "index": dataset_index}
        updates["dataset"] = dataset
    if dataset_path is not None:
        misc_params["dataset_path"] = str(dataset_path)
        updates["dataset_path"] = dataset_path

    if not updates:
        return {
            "status": "warning",
            "message": "No parameters specified to update",
            "details": {"updates": {}},
        }

    return {
        "status": "success",
        "message": "Model parameters updated successfully",
        "details": {"updates": updates},
    }


@arch_mcp.tool()
async def set_complex_params(
    ctx: Context,
    data_workers: int | None = None,
    eval_interval: int | None = None,
    warmup_epochs: int | None = None,
    scheduler: str | None = None,
    num_classes: int | None = None,
    pretrained_weights: str | None = None,
) -> dict[str, Any]:
    """Set complex model parameters.

    Args:
        ctx: FastMCP context containing the architecture manager state.
        data_workers: Number of data workers for data loading.
        eval_interval: Interval for model evaluation during training.
        warmup_epochs: Number of warmup epochs.
        scheduler: Scheduler type (cos, linear, etc.).
        num_classes: Number of output classes.
        pretrained_weights: Path to pretrained weights file.

    Returns:
        dict: A dictionary containing operation status.
            - status: "success", "error", or "warning"
            - message: Human-readable status message
            - details: Dictionary with updated parameters

    """
    arch_manager = ctx.get_state("arch_manager")

    # Ensure complex fields exist
    arch_manager._ensure_complex_fields()

    architecture = arch_manager.current_architecture
    complex_params = architecture["complex_misc_params"]

    # Update parameters
    updates = {}
    if data_workers is not None:
        complex_params["data_num_workers"] = data_workers
        updates["data_workers"] = data_workers
    if eval_interval is not None:
        complex_params["eval_interval"] = eval_interval
        updates["eval_interval"] = eval_interval
    if warmup_epochs is not None:
        complex_params["warmup_epochs"] = warmup_epochs
        updates["warmup_epochs"] = warmup_epochs
    if scheduler is not None:
        if scheduler in arch_manager.complex_schedulers:
            complex_params["scheduler"] = scheduler
            updates["scheduler"] = scheduler
        else:
            return {
                "status": "error",
                "message": f"Invalid scheduler: {scheduler}",
                "details": {
                    "scheduler": scheduler,
                    "available_schedulers": arch_manager.complex_schedulers,
                },
            }
    if num_classes is not None:
        complex_params["num_classes"] = num_classes
        updates["num_classes"] = num_classes
    if pretrained_weights is not None:
        architecture["pretrained_weights"] = pretrained_weights
        updates["pretrained_weights"] = pretrained_weights

    if not updates:
        return {
            "status": "warning",
            "message": "No parameters specified to update",
            "details": {"updates": {}},
        }

    return {
        "status": "success",
        "message": "Complex parameters updated successfully",
        "details": {"updates": updates},
    }


@arch_mcp.tool()
async def list_complex_models(ctx: Context) -> dict[str, Any]:
    """List all available complex pretrained models.

    Args:
        ctx: FastMCP context containing the architecture manager state.

    Returns:
        dict: A dictionary containing available complex models.
            - status: "success"
            - message: Human-readable status message
            - details: Dictionary with list of available models

    """
    arch_manager = ctx.get_state("arch_manager")

    if arch_manager.complex_arch_models:
        models = sorted(arch_manager.complex_arch_models)
        return {
            "status": "success",
            "message": f"Found {len(models)} complex model(s)",
            "details": {
                "count": len(models),
                "models": models,
            },
        }
    return {
        "status": "success",
        "message": "No complex models available",
        "details": {
            "count": 0,
            "models": [],
        },
    }


@arch_mcp.tool()
async def set_optimizer(
    ctx: Context,
    optimizer_type: str,
    params: list[str] | None = None,
) -> dict[str, Any]:
    """Set the optimizer configuration.

    Args:
        ctx: FastMCP context containing the architecture manager state.
        optimizer_type: Type of optimizer (Adam, SGD, AdamW, etc.).
        params: List of optimizer parameters in key=value format.

    Returns:
        dict: A dictionary containing operation status.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with optimizer configuration

    """
    arch_manager = ctx.get_state("arch_manager")

    if optimizer_type not in arch_manager.optimizers:
        available = list(arch_manager.optimizers.keys())[:10]
        return {
            "status": "error",
            "message": f"Optimizer '{optimizer_type}' not available",
            "details": {
                "optimizer_type": optimizer_type,
                "available_optimizers": available,
                "note": "Showing first 10 available optimizers",
            },
        }

    # Get valid parameter names for this optimizer
    optimizer_info = arch_manager.optimizers[optimizer_type]
    valid_param_names = {param["name"] for param in optimizer_info}

    # Parse parameters
    optimizer_params = {}
    parse_errors = []

    if params:
        for param in params:
            if "=" not in param:
                parse_errors.append(f"Invalid parameter format: {param}. Use key=value")
                continue

            key, value = param.split("=", 1)

            # Check if parameter is valid for this optimizer
            if key not in valid_param_names:
                return {
                    "status": "error",
                    "message": f"Parameter '{key}' is not valid for optimizer '{optimizer_type}'",
                    "details": {
                        "invalid_parameter": key,
                        "valid_parameters": sorted(valid_param_names),
                    },
                }

            # Type validation and conversion
            expected_type = None
            for param_info in optimizer_info:
                if param_info["name"] == key:
                    expected_type = param_info.get("type")
                    break

            try:
                if expected_type == bool:
                    if value.lower() not in ["true", "false"]:
                        return {
                            "status": "error",
                            "message": f"Parameter '{key}' expects boolean value (true/false), got: {value}",
                            "details": {
                                "parameter": key,
                                "value": value,
                                "expected_type": "bool",
                            },
                        }
                    optimizer_params[key] = value.lower() == "true"
                elif expected_type == int:
                    if not value.lstrip("-").isdigit():
                        return {
                            "status": "error",
                            "message": f"Parameter '{key}' expects integer value, got: {value}",
                            "details": {
                                "parameter": key,
                                "value": value,
                                "expected_type": "int",
                            },
                        }
                    optimizer_params[key] = int(value)
                elif expected_type == float:
                    if not value.replace(".", "", 1).lstrip("-").replace(
                        "e", "", 1,
                    ).replace("E", "", 1).replace(
                        "+", "", 1,
                    ).isdigit() and value not in ["inf", "-inf"]:
                        return {
                            "status": "error",
                            "message": f"Parameter '{key}' expects float value, got: {value}",
                            "details": {
                                "parameter": key,
                                "value": value,
                                "expected_type": "float",
                            },
                        }
                    optimizer_params[key] = float(value)
                elif value.startswith("[") and value.endswith("]"):
                    optimizer_params[key] = eval(value)
                else:
                    optimizer_params[key] = value
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error converting parameter '{key}' value '{value}': {e!s}",
                    "details": {"parameter": key, "value": value, "error": str(e)},
                }

    # Add default values for missing parameters or show error for required parameters without defaults
    missing_params = []
    for param_info in optimizer_info:
        param_name = param_info["name"]
        if param_name not in optimizer_params:
            default_value = param_info.get("defaultvalue")
            if default_value is not None:
                optimizer_params[param_name] = default_value
            else:
                missing_params.append(param_name)

    if missing_params:
        return {
            "status": "error",
            "message": f"Missing required parameters for optimizer '{optimizer_type}'",
            "details": {
                "missing_parameters": missing_params,
                "note": "Please provide values for these parameters using params=['key=value', ...]",
            },
        }

    # Set optimizer in architecture
    arch_manager.current_architecture["optimizer"] = {
        "type": optimizer_type,
        "params": optimizer_params,
    }

    return {
        "status": "success",
        "message": "Optimizer configured successfully",
        "details": {
            "optimizer_type": optimizer_type,
            "parameters": optimizer_params,
        },
    }


@arch_mcp.tool()
async def set_loss_function(
    ctx: Context,
    loss_type: str,
    params: list[str] | None = None,
) -> dict[str, Any]:
    """Set the loss function configuration.

    Args:
        ctx: FastMCP context containing the architecture manager state.
        loss_type: Type of loss function (CrossEntropyLoss, MSELoss, etc.).
        params: List of loss function parameters in key=value format.

    Returns:
        dict: A dictionary containing operation status.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with loss function configuration

    """
    arch_manager = ctx.get_state("arch_manager")

    if loss_type not in arch_manager.loss_funcs:
        available = list(arch_manager.loss_funcs.keys())[:10]
        return {
            "status": "error",
            "message": f"Loss function '{loss_type}' not available",
            "details": {
                "loss_type": loss_type,
                "available_loss_functions": available,
                "note": "Showing first 10 available loss functions",
            },
        }

    # Get valid parameter names for this loss function
    loss_info = arch_manager.loss_funcs[loss_type]
    valid_param_names = {param["name"] for param in loss_info}

    # Parse parameters
    loss_params = {}

    if params:
        for param in params:
            if "=" not in param:
                return {
                    "status": "error",
                    "message": f"Invalid parameter format: {param}. Use key=value",
                    "details": {"invalid_parameter": param},
                }

            key, value = param.split("=", 1)

            # Check if parameter is valid for this loss function
            if key not in valid_param_names:
                return {
                    "status": "error",
                    "message": f"Parameter '{key}' is not valid for loss function '{loss_type}'",
                    "details": {
                        "invalid_parameter": key,
                        "valid_parameters": sorted(valid_param_names),
                    },
                }

            # Type validation and conversion
            expected_type = None
            for param_info in loss_info:
                if param_info["name"] == key:
                    expected_type = param_info.get("type")
                    break

            try:
                if expected_type == bool:
                    if value.lower() not in ["true", "false"]:
                        return {
                            "status": "error",
                            "message": f"Parameter '{key}' expects boolean value (true/false), got: {value}",
                            "details": {
                                "parameter": key,
                                "value": value,
                                "expected_type": "bool",
                            },
                        }
                    loss_params[key] = value.lower() == "true"
                elif expected_type == int:
                    if not value.lstrip("-").isdigit():
                        return {
                            "status": "error",
                            "message": f"Parameter '{key}' expects integer value, got: {value}",
                            "details": {
                                "parameter": key,
                                "value": value,
                                "expected_type": "int",
                            },
                        }
                    loss_params[key] = int(value)
                elif expected_type == float:
                    if not value.replace(".", "", 1).lstrip("-").replace(
                        "e", "", 1,
                    ).replace("E", "", 1).replace(
                        "+", "", 1,
                    ).isdigit() and value not in ["inf", "-inf"]:
                        return {
                            "status": "error",
                            "message": f"Parameter '{key}' expects float value, got: {value}",
                            "details": {
                                "parameter": key,
                                "value": value,
                                "expected_type": "float",
                            },
                        }
                    loss_params[key] = float(value)
                elif value.startswith("[") and value.endswith("]"):
                    loss_params[key] = eval(value)
                else:
                    loss_params[key] = value
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error converting parameter '{key}' value '{value}': {e!s}",
                    "details": {"parameter": key, "value": value, "error": str(e)},
                }

    # Add default values for missing parameters
    missing_params = []
    for param_info in loss_info:
        param_name = param_info["name"]
        if param_name not in loss_params:
            default_value = param_info.get("defaultvalue")
            if default_value is not None:
                if isinstance(default_value, int) and default_value < 0:
                    loss_params[param_name] = 1
                elif isinstance(default_value, float) and default_value < 0:
                    loss_params[param_name] = 1.0
                else:
                    loss_params[param_name] = default_value
            else:
                missing_params.append(param_name)

    if missing_params:
        return {
            "status": "error",
            "message": f"Missing required parameters for loss function '{loss_type}'",
            "details": {
                "missing_parameters": missing_params,
                "note": "Please provide values for these parameters using params=['key=value', ...]",
            },
        }

    # Set loss function in architecture
    arch_manager.current_architecture["loss_func"] = {
        "type": loss_type,
        "params": loss_params,
    }

    return {
        "status": "success",
        "message": "Loss function configured successfully",
        "details": {
            "loss_type": loss_type,
            "parameters": loss_params,
        },
    }


@arch_mcp.tool()
async def set_scheduler(
    ctx: Context,
    scheduler_type: str,
    params: list[str] | None = None,
) -> dict[str, Any]:
    """Set the learning rate scheduler configuration.

    Args:
        ctx: FastMCP context containing the architecture manager state.
        scheduler_type: Type of scheduler (StepLR, CosineAnnealingLR, None, etc.).
        params: List of scheduler parameters in key=value format.

    Returns:
        dict: A dictionary containing operation status.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with scheduler configuration

    """
    arch_manager = ctx.get_state("arch_manager")

    # Handle "None" scheduler case
    if scheduler_type == "None":
        arch_manager.current_architecture["scheduler"] = {
            "type": "None",
            "params": {},
        }
        return {
            "status": "success",
            "message": "Scheduler disabled (set to None)",
            "details": {
                "scheduler_type": "None",
                "parameters": {},
            },
        }

    if scheduler_type not in arch_manager.schedulers:
        available = list(arch_manager.schedulers.keys())[:10]
        return {
            "status": "error",
            "message": f"Scheduler '{scheduler_type}' not available",
            "details": {
                "scheduler_type": scheduler_type,
                "available_schedulers": available,
                "note": "Showing first 10 available schedulers. Use 'None' for no scheduler",
            },
        }

    # Get valid parameter names for this scheduler
    scheduler_info = arch_manager.schedulers[scheduler_type]
    valid_param_names = {param["name"] for param in scheduler_info}

    # Parse parameters
    scheduler_params = {}

    if params:
        for param in params:
            if "=" not in param:
                return {
                    "status": "error",
                    "message": f"Invalid parameter format: {param}. Use key=value",
                    "details": {"invalid_parameter": param},
                }

            key, value = param.split("=", 1)

            # Check if parameter is valid for this scheduler
            if key not in valid_param_names:
                return {
                    "status": "error",
                    "message": f"Parameter '{key}' is not valid for scheduler '{scheduler_type}'",
                    "details": {
                        "invalid_parameter": key,
                        "valid_parameters": sorted(valid_param_names),
                    },
                }

            # Type validation and conversion
            expected_type = None
            for param_info in scheduler_info:
                if param_info["name"] == key:
                    expected_type = param_info.get("type")
                    break

            try:
                if expected_type == bool:
                    if value.lower() not in ["true", "false"]:
                        return {
                            "status": "error",
                            "message": f"Parameter '{key}' expects boolean value (true/false), got: {value}",
                            "details": {
                                "parameter": key,
                                "value": value,
                                "expected_type": "bool",
                            },
                        }
                    scheduler_params[key] = value.lower() == "true"
                elif expected_type == int:
                    if not value.lstrip("-").isdigit():
                        return {
                            "status": "error",
                            "message": f"Parameter '{key}' expects integer value, got: {value}",
                            "details": {
                                "parameter": key,
                                "value": value,
                                "expected_type": "int",
                            },
                        }
                    scheduler_params[key] = int(value)
                elif expected_type == float:
                    if not value.replace(".", "", 1).lstrip("-").replace(
                        "e", "", 1,
                    ).replace("E", "", 1).replace(
                        "+", "", 1,
                    ).isdigit() and value not in ["inf", "-inf"]:
                        return {
                            "status": "error",
                            "message": f"Parameter '{key}' expects float value, got: {value}",
                            "details": {
                                "parameter": key,
                                "value": value,
                                "expected_type": "float",
                            },
                        }
                    scheduler_params[key] = float(value)
                elif value.startswith("[") and value.endswith("]"):
                    scheduler_params[key] = eval(value)
                else:
                    scheduler_params[key] = value
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error converting parameter '{key}' value '{value}': {e!s}",
                    "details": {"parameter": key, "value": value, "error": str(e)},
                }

    # Add default values for missing parameters
    missing_params = []
    for param_info in scheduler_info:
        param_name = param_info["name"]
        if param_name not in scheduler_params:
            default_value = param_info.get("defaultvalue")
            if default_value is not None:
                scheduler_params[param_name] = default_value
            else:
                missing_params.append(param_name)

    if missing_params:
        return {
            "status": "error",
            "message": f"Missing required parameters for scheduler '{scheduler_type}'",
            "details": {
                "missing_parameters": missing_params,
                "note": "Please provide values for these parameters using params=['key=value', ...]",
            },
        }

    # Set scheduler in architecture
    arch_manager.current_architecture["scheduler"] = {
        "type": scheduler_type,
        "params": scheduler_params,
    }

    return {
        "status": "success",
        "message": "Scheduler configured successfully",
        "details": {
            "scheduler_type": scheduler_type,
            "parameters": scheduler_params,
        },
    }


@arch_mcp.tool()
async def list_optimizers(ctx: Context, limit: int = 20) -> dict[str, Any]:
    """List available optimizer types.

    Args:
        ctx: FastMCP context containing the architecture manager state.
        limit: Maximum number of optimizers to return (default: 20).

    Returns:
        dict: A dictionary containing available optimizers.
            - status: "success"
            - message: Human-readable status message
            - details: Dictionary with optimizer information

    """
    arch_manager = ctx.get_state("arch_manager")
    optimizers = list(arch_manager.optimizers.keys())

    return {
        "status": "success",
        "message": f"Found {len(optimizers)} optimizer(s)",
        "details": {
            "total_count": len(optimizers),
            "returned_count": min(limit, len(optimizers)),
            "optimizers": optimizers[:limit],
            "note": f"Showing first {limit} optimizers"
            if len(optimizers) > limit
            else "Showing all optimizers",
        },
    }


@arch_mcp.tool()
async def list_loss_functions(ctx: Context, limit: int = 20) -> dict[str, Any]:
    """List available loss function types.

    Args:
        ctx: FastMCP context containing the architecture manager state.
        limit: Maximum number of loss functions to return (default: 20).

    Returns:
        dict: A dictionary containing available loss functions.
            - status: "success"
            - message: Human-readable status message
            - details: Dictionary with loss function information

    """
    arch_manager = ctx.get_state("arch_manager")
    loss_funcs = list(arch_manager.loss_funcs.keys())

    return {
        "status": "success",
        "message": f"Found {len(loss_funcs)} loss function(s)",
        "details": {
            "total_count": len(loss_funcs),
            "returned_count": min(limit, len(loss_funcs)),
            "loss_functions": loss_funcs[:limit],
            "note": f"Showing first {limit} loss functions"
            if len(loss_funcs) > limit
            else "Showing all loss functions",
        },
    }


@arch_mcp.tool()
async def list_schedulers(ctx: Context, limit: int = 20) -> dict[str, Any]:
    """List available learning rate scheduler types.

    Args:
        ctx: FastMCP context containing the architecture manager state.
        limit: Maximum number of schedulers to return (default: 20).

    Returns:
        dict: A dictionary containing available schedulers.
            - status: "success"
            - message: Human-readable status message
            - details: Dictionary with scheduler information

    """
    arch_manager = ctx.get_state("arch_manager")
    schedulers = list(arch_manager.schedulers.keys())

    return {
        "status": "success",
        "message": f"Found {len(schedulers)} scheduler(s)",
        "details": {
            "total_count": len(schedulers),
            "returned_count": min(limit, len(schedulers)),
            "schedulers": schedulers[:limit],
            "note": f"Showing first {limit} schedulers. Use 'None' to disable scheduler"
            if len(schedulers) > limit
            else "Showing all schedulers. Use 'None' to disable scheduler",
        },
    }


@arch_mcp.tool()
async def list_datasets(ctx: Context) -> dict[str, Any]:
    """List available dataset types.

    Args:
        ctx: FastMCP context containing the architecture manager state.

    Returns:
        dict: A dictionary containing available datasets.
            - status: "success"
            - message: Human-readable status message
            - details: Dictionary with dataset information

    """
    arch_manager = ctx.get_state("arch_manager")
    datasets = arch_manager.datasets

    return {
        "status": "success",
        "message": f"Found {len(datasets)} dataset(s)",
        "details": {
            "count": len(datasets),
            "datasets": datasets,
        },
    }
