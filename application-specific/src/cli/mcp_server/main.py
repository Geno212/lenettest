import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import Any
from fastmcp import Context, FastMCP
from src.cli.core.architecture_manager import ArchitectureManager
from src.cli.utils import Analyze_for_MCP
from src.paths.SystemPaths import SystemPaths
from src.utils.Cookiecutter import Cookiecutter
from src.cli.utils import get_available_devices
from typing import Dict, Any, Optional
import sys
import logging
from fastmcp import Context
from src.paths.SystemPaths import SystemPaths
import cv2
import numpy as np
import torch
from YOLOX.yolox.exp import get_exp
from src.cli.utils.test import (
    detect_and_visualize,
    get_HGD_model,
    validate_and_parse_class_names,
)
import os
from pathlib import Path
from typing import Any
from src.paths import PathsFactory

# Configure logging with session_id filter
class SessionContextFilter(logging.Filter):
    """Adds session_id to log records if not present."""
    def filter(self, record):
        if not hasattr(record, 'session_id'):
            record.session_id = 'N/A'
        return True

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [Session: %(session_id)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add the filter to the root logger and all handlers
root_logger = logging.getLogger()
session_filter = SessionContextFilter()
root_logger.addFilter(session_filter)
for handler in root_logger.handlers:
    handler.addFilter(session_filter)

arch_manager_store: dict[str, ArchitectureManager] = {}
session_lock = asyncio.Lock()


async def cleanup():
    async with session_lock:
        for sid in list(arch_manager_store.keys()):
            arch_manager_store.pop(sid, None)


@asynccontextmanager
async def mcp_lifespan(app):
    print("Starting MCP server resources")
    arch_manager_store.clear()
    yield
    print("Shutting down MCP server resources")
    await cleanup()


arch_mcp = FastMCP(name="ArchService", lifespan=mcp_lifespan)


# Helper: create a Manager for a session
def _create_manager_for_session(session_id: str) -> ArchitectureManager:
    # create persistent state and manager (pass the state to each component)
    manager = ArchitectureManager()
    arch_manager_store[session_id] = manager
    return manager


# For stdio or clients that don't get automatic session_id, provide a start_session tool:
@arch_mcp.tool(
    name="initialize_session",
    description="Create a new session and return session_id",
)
async def initialize_session_tool(ctx: Context) -> dict:
    # create manager and store
    logger.info(f"Initializing session", extra={'session_id': ctx.session_id})
    _create_manager_for_session(ctx.session_id)
    await ctx.info(f"Created manager for session {ctx.session_id}")
    logger.info(f"Session initialized successfully", extra={'session_id': ctx.session_id})
    return {"session_id": ctx.session_id}


# Tool to explicitly close a session
@arch_mcp.tool(name="close_session")
async def close_session_tool(ctx: Context) -> dict:
    logger.info(f"Closing session", extra={'session_id': ctx.session_id})
    async with session_lock:
        mgr = arch_manager_store.pop(ctx.session_id, None)
    if mgr and hasattr(mgr, "close"):
        del mgr
    logger.info(f"Session closed successfully", extra={'session_id': ctx.session_id})
    return {"status": "closed", "session_id": ctx.session_id}


# ========== Tools ==========


@arch_mcp.tool()
async def create_architecture(
    ctx: Context,
    name: str,
    output_dir: Path,
    base_model: str | None = None,
    run_number: int | None = None,
):
    """Create a new neural network architecture.

    This function creates a new neural network architecture with the specified name and output directory.
    It can optionally start with a base model if provided. The architecture is initialized with empty layers
    and the specified pretrained model if available.

    Args:
        ctx: The FastMCP context containing session state.
        name: The name of the architecture to create.
        output_dir: The directory where the architecture file will be saved.
        base_model: Optional base model to start the architecture with.

    Returns:
        dict: A dictionary containing status information and architecture details.

    """
    logger.info(f"Creating architecture: {name}, base_model: {base_model}", extra={'session_id': ctx.session_id})
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    arch_file = output_dir / f"{name}_architecture_run_{run_number}.json"

    if arch_file.exists():
        return {
            "status": "error",
            "message": f"Architecture '{name}' already exists",
            "details": {"file_path": str(arch_file)},
        }

    # Preserve all keys except layers and pretrained
    preserved_config = {}
    for key, value in arch_manager.current_architecture.items():
        if key not in ["layers", "pretrained"]:
            preserved_config[key] = value

    pretrained_index = -1
    if base_model and base_model.lower() in arch_manager.pretrained_models:
        pretrained_index = arch_manager.pretrained_models.index(base_model.lower())

    arch_manager.current_architecture = {
        **preserved_config,
        "layers": [],
        "pretrained": {"value": base_model.lower() if base_model else None, "index": pretrained_index},
    }

    # Add base model layers if specified
    if base_model:
        if base_model.lower() in arch_manager.pretrained_models:
            return {
                "status": "success",
                "message": f"Added base model: {base_model}",
                "details": {"base_model": base_model},
            }
        return {
            "status": "warning",
            "message": f"Base model '{base_model}' not found in available models",
            "details": {"base_model": base_model},
        }

    # Save the architecture
    arch_manager.save_architecture(arch_file)

    status = "success"
    message = "Architecture created successfully!"
    details = {
        "file_path": str(arch_file),
        "base_model": base_model or "None",
        "layers_count": len(arch_manager.current_architecture["layers"]),
    }

    return {"status": status, "message": message, "details": details}


@arch_mcp.tool()
async def set_pretrained(
    ctx: Context,
    base_model: str | None = None,
    file_path: Path | None = None,
):
    """Set the pretrained model for the current architecture.

    This function updates the pretrained model configuration for the current neural network architecture.
    For YOLOX models, it automatically configures depth and width parameters for complex architecture.
    Optionally saves the architecture to file after updating.

    Args:
        ctx: The FastMCP context containing session state.
        base_model: Optional base model to set as pretrained.
        file_path: Optional path to save the architecture after updating.

    Returns:
        dict: A dictionary containing status information and pretrained model details.

    """
    logger.info(f"Setting pretrained model: {base_model}, file_path: {file_path}", extra={'session_id': ctx.session_id})
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    # Check if this is a YOLOX complex model
    is_complex_model = (
        base_model and 
        base_model.lower().startswith("yolox") and 
        base_model.lower() in [m.lower() for m in arch_manager.complex_arch_models]
    )

    if is_complex_model:
        # Handle complex architecture (YOLOX models with depth/width)
        try:
            arch_manager.create_complex_architecture(base_model)
            
            # Save if file path provided
            if file_path:
                arch_manager.save_complex_architecture(file_path)
                logger.info(f"Complex architecture saved to {file_path}", extra={'session_id': ctx.session_id})
            
            return {
                "status": "success",
                "message": f"Set complex pretrained model: {base_model}",
                "details": {
                    "base_model": base_model,
                    "depth": arch_manager.current_architecture["pretrained"]["depth"],
                    "width": arch_manager.current_architecture["pretrained"]["width"],
                    "file_saved": str(file_path) if file_path else None,
                },
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to set complex architecture: {e}",
                "details": {"base_model": base_model, "error": str(e)},
            }
    
    # Handle regular pretrained models
    # Preserve all keys except layers and pretrained
    preserved_config = {}
    for key, value in arch_manager.current_architecture.items():
        if key not in ["layers", "pretrained"]:
            preserved_config[key] = value

    pretrained_index = -1
    if base_model and base_model.lower() in arch_manager.pretrained_models:
        pretrained_index = arch_manager.pretrained_models.index(base_model.lower())

    arch_manager.current_architecture = {
        **preserved_config,
        "pretrained": {"value": base_model, "index": pretrained_index},
    }

    # Save if file path provided
    if file_path:
        arch_manager.save_architecture(file_path)
        logger.info(f"Architecture saved to {file_path}", extra={'session_id': ctx.session_id})

    # Return status
    if base_model:
        if base_model.lower() in arch_manager.pretrained_models:
            return {
                "status": "success",
                "message": f"Set pretrained model: {base_model}",
                "details": {
                    "base_model": base_model,
                    "file_saved": str(file_path) if file_path else None,
                },
            }
        return {
            "status": "warning",
            "message": f"Base model '{base_model}' not found in available models",
            "details": {
                "base_model": base_model,
                "file_saved": str(file_path) if file_path else None,
            },
        }

    return {
        "status": "success",
        "message": "Pretrained model configuration updated",
        "details": {
            "base_model": base_model or "None",
            "file_saved": str(file_path) if file_path else None,
        },
    }


@arch_mcp.tool()
async def create_complex_architecture(
    ctx: Context,
    name: str,
    output_dir: Path,
    model: str,
):
    """Create a new complex neural network architecture with pretrained model.

    This function creates a new complex neural network architecture with the specified name,
    output directory, and a required pretrained model. The architecture is built using the
    specified complex model and saved to the output directory.

    Args:
        ctx: The FastMCP context containing session state.
        name: The name of the complex architecture to create.
        output_dir: The directory where the architecture file will be saved.
        model: The pretrained model to use for the architecture (e.g., YOLOX-Nano).

    Returns:
        dict: A dictionary containing status information and architecture details.

    """
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    arch_file = output_dir / f"{name}_complex_architecture.json"

    if arch_file.exists():
        return {
            "status": "error",
            "message": f"Complex architecture '{name}' already exists",
            "details": {"file_path": str(arch_file)},
        }

    # Check if model is available (case insensitive)
    if model.lower() not in [m.lower() for m in arch_manager.complex_arch_models]:
        logger.error(f"Complex model '{model}' not available", extra={'session_id': ctx.session_id})
        return {
            "status": "error",
            "message": f"Complex model '{model}' not available",
            "details": {
                "model": model,
                "available_models": list(arch_manager.complex_arch_models)[:5],
            },
        }

    # Create complex architecture
    arch_manager.create_complex_architecture(model)

    # Save the architecture
    arch_manager.save_complex_architecture(arch_file)

    return {
        "status": "success",
        "message": "Complex architecture created successfully!",
        "details": {
            "file_path": str(arch_file),
            "model": model,
            "depth": arch_manager.current_architecture["pretrained"]["depth"],
            "width": arch_manager.current_architecture["pretrained"]["width"],
        },
    }


@arch_mcp.tool()
async def save_complex_architecture(
    ctx: Context,
    file_path: Path,
):
    """Save the current complex architecture to a file.

    This function saves the current complex neural network architecture to the specified file path.
    It displays a success message with details about the saved architecture including the number of layers
    and the pretrained model used.

    Args:
        ctx: The FastMCP context containing session state.
        file_path: The file path where the complex architecture will be saved.

    Returns:
        dict: A dictionary containing status information and save details.

    """
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    # Save complex architecture
    arch_manager.save_complex_architecture(file_path)

    return {
        "status": "success",
        "message": "Complex architecture saved successfully!",
        "details": {
            "file_path": str(file_path),
            "layers_count": len(arch_manager.current_architecture["layers"]),
            "model": arch_manager.current_architecture["pretrained"].get(
                "value",
                "None",
            ),
        },
    }


@arch_mcp.tool()
async def add_layer(
    ctx: Context,
    layer_type: str,
    params: dict[str, Any],
    position: int | None = None,
    file_path: Path | None = None,
):
    """Add a layer to the current architecture.

    This function adds a new layer to the current neural network architecture. It validates that the layer type
    is available, processes the layer parameters with proper type conversion, and adds the layer to the architecture.
    The function displays detailed information about the added layer including its type, position, and parameters.

    Args:
        ctx: The FastMCP context containing session state.
        layer_type: The type of layer to add (e.g., 'Conv2d', 'Linear', 'ReLU').
        params: Dictionary of parameter names to values. The parameter name is the key
        position: Optional position to insert the layer (default: end of architecture).
        file_path: Optional path to save the architecture after adding the layer.

    Returns:
        dict: A dictionary containing status information and layer details.

    """
    logger.info(f"Adding layer: {layer_type} at position {position}, file_path: {file_path}", extra={'session_id': ctx.session_id})
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    if layer_type not in arch_manager.layers:
        return {
            "status": "error",
            "message": f"Layer type '{layer_type}' not available",
            "details": {
                "layer_type": layer_type,
                "available_layers": arch_manager.list_available_layers()[:10],
            },
        }

    layer_params = {}
    layer_info = arch_manager.get_layer_info(layer_type)
    unnecessary_params = {
            "dilation",
            "divisor_override",
            "approximate",
    }

    if layer_info is None:
        return {
            "status": "error",
            "message": f"Could not get layer information for '{layer_type}'",
            "details": {"layer_type": layer_type},
        }

    # Get valid parameter names for this layer
    valid_param_names = {param["name"] for param in layer_info}
    valid_param_names.difference_update(unnecessary_params)

    # Expect params to be a dict mapping parameter name -> value
    if not isinstance(params, dict):
        return {
            "status": "error",
            "message": "Invalid params format: expected a dictionary of parameter names to values",
            "details": {"received_type": type(params).__name__},
        }

    for key, value in params.items():
        if key not in valid_param_names:
            return {
                "status": "error",
                "message": f"Parameter '{key}' is not valid for layer type '{layer_type}'",
                "details": {
                    "parameter": key,
                    "layer_type": layer_type,
                    "valid_parameters": sorted(valid_param_names),
                },
            }

        # Find expected type for this parameter (if available)
        expected_type = None
        for param_info in layer_info:
            if param_info["name"] == key:
                expected_type = param_info.get("type")
                break

        try:
            converted_value = value
            # Convert strings to the expected type where appropriate
            if expected_type is not None:
                # Boolean
                if expected_type == bool:
                    if isinstance(value, str):
                        if value.lower() not in ("true", "false"):
                            return {
                                "status": "error",
                                "message": f"Parameter '{key}' expects boolean value (true/false), got: {value}",
                                "details": {"parameter": key, "expected_type": "boolean", "received_value": value},
                            }
                        converted_value = value.lower() == "true"
                    else:
                        converted_value = bool(value)

                # Integer
                elif expected_type == int:
                    if isinstance(value, int):
                        converted_value = value
                    elif isinstance(value, str):
                        converted_value = int(value)
                    else:
                        converted_value = int(value)

                # Float
                elif expected_type == float:
                    if isinstance(value, float):
                        converted_value = value
                    elif isinstance(value, str):
                        converted_value = float(value)
                    else:
                        converted_value = float(value)

                else:
                    # For other expected types (e.g., list/tuple), allow passing native types or string repr
                    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                        converted_value = eval(value)
                    else:
                        converted_value = value
            else:
                # No expected type available: try sensible conversions for strings
                if isinstance(value, str):
                    low = value.lower()
                    if low in ("true", "false"):
                        converted_value = low == "true"
                    else:
                        # try int then float then list
                        try:
                            converted_value = int(value)
                        except Exception:
                            try:
                                converted_value = float(value)
                            except Exception:
                                if value.startswith("[") and value.endswith("]"):
                                    converted_value = eval(value)
                                else:
                                    converted_value = value
                else:
                    converted_value = value

            layer_params[key] = converted_value
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error converting parameter '{key}' value '{value}': {e}",
                "details": {"parameter": key, "value": value, "error": str(e)},
            }

    # Validate position parameter
    layers_list = arch_manager.current_architecture["layers"]
    if position is not None:
        if not (0 <= position <= len(layers_list)):
            return {
                "status": "error",
                "message": f"Invalid position {position}. Must be between 0 and {len(layers_list)} (or omit for end)",
                "details": {
                    "position": position,
                    "valid_range": f"0-{len(layers_list)}",
                    "layers_count": len(layers_list),
                },
            }
    else:
        position = len(layers_list)  # Default to end

    # Add layer
    layer = arch_manager.add_layer(layer_type, layer_params, position)

    # Get analysis results for Layer compatibility
    analysis_results = Analyze_for_MCP(arch_manager.current_architecture["layers"])

    # Save architecture if file_path is provided
    if file_path:
        arch_manager.save_architecture(file_path)
        logger.info(f"Architecture saved to {file_path} after adding layer", extra={'session_id': ctx.session_id})

    return {
        "status": "success",
        "message": "Layer added successfully!",
        "details": {
            "layer_type": layer_type,
            "position": position,
            "parameters_count": len(layer_params),
            "parameters": layer_params if layer_params else {},
            "analysis": analysis_results,
        },
    }


@arch_mcp.tool()
async def set_layers(
    ctx: Context,
    layers_config: list[dict[str, Any]],
    file_path: Path | None = None,
):
    """Replace all layers in the current architecture with the provided layers.

    This function clears all existing layers from the architecture and then adds the new layers
    in sequence using the same validation and parameter processing logic as the add_layer function.
    It provides a convenient way to completely replace the layer configuration while maintaining
    all the validation and error handling.

    Args:
        ctx: The FastMCP context containing session state.
        layers_config: List of layer configuration dictionaries, each containing:
            - layer_type: The type of layer to add (e.g., 'Conv2d', 'Linear', 'ReLU')
            - params: Dictionary of parameter names to values
            - position: Optional position to insert the layer (default: end of architecture)
        file_path: Optional path to save the architecture after setting all layers.

    Returns:
        dict: A dictionary containing status information and details about all set layers.

    """
    logger.info(f"Replacing all layers with {len(layers_config)} new layers", extra={'session_id': ctx.session_id})
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    if not layers_config:
        return {
            "status": "error",
            "message": "No layers configuration provided",
            "details": {"layers_count": 0},
        }

    # Clear all existing layers first
    arch_manager.current_architecture["layers"] = []

    added_layers = []
    failed_layers = []
    total_analysis = []

    for i, layer_config in enumerate(layers_config):
        # Validate layer configuration
        if not isinstance(layer_config, dict):
            failed_layers.append({
                "index": i,
                "error": "Invalid layer configuration format",
                "config": layer_config,
            })
            continue

        layer_type = layer_config.get("layer_type")
        params = layer_config.get("params", {})
        position = layer_config.get("position")

        if not layer_type:
            failed_layers.append({
                "index": i,
                "error": "Missing layer_type in configuration",
                "config": layer_config,
            })
            continue

        # Validate layer type availability
        if layer_type not in arch_manager.layers:
            failed_layers.append({
                "index": i,
                "layer_type": layer_type,
                "error": f"Layer type '{layer_type}' not available",
                "available_layers": arch_manager.list_available_layers()[:10],
            })
            continue

        # Process layer parameters (reuse logic from add_layer)
        layer_params = {}
        layer_info = arch_manager.get_layer_info(layer_type)
        unnecessary_params = {
            "dilation",
            "divisor_override",
            "approximate",
        }

        if layer_info is None:
            failed_layers.append({
                "index": i,
                "layer_type": layer_type,
                "error": f"Could not get layer information for '{layer_type}'",
            })
            continue

        # Get valid parameter names for this layer
        valid_param_names = {param["name"] for param in layer_info}
        valid_param_names.difference_update(unnecessary_params)

        # Validate params format
        if not isinstance(params, dict):
            failed_layers.append({
                "index": i,
                "layer_type": layer_type,
                "error": "Invalid params format: expected a dictionary of parameter names to values",
                "received_type": type(params).__name__,
            })
            continue

        # Process each parameter
        param_validation_failed = False
        for key, value in params.items():
            if key not in valid_param_names:
                failed_layers.append({
                    "index": i,
                    "layer_type": layer_type,
                    "error": f"Parameter '{key}' is not valid for layer type '{layer_type}'",
                    "parameter": key,
                    "valid_parameters": sorted(valid_param_names),
                })
                param_validation_failed = True
                break

            # Find expected type for this parameter
            expected_type = None
            for param_info in layer_info:
                if param_info["name"] == key:
                    expected_type = param_info.get("type")
                    break

            try:
                converted_value = value
                # Convert strings to the expected type where appropriate
                if expected_type is not None:
                    # Boolean
                    if expected_type == bool:
                        if isinstance(value, str):
                            if value.lower() not in ("true", "false"):
                                failed_layers.append({
                                    "index": i,
                                    "layer_type": layer_type,
                                    "error": f"Parameter '{key}' expects boolean value (true/false), got: {value}",
                                    "parameter": key,
                                    "expected_type": "boolean",
                                    "received_value": value,
                                })
                                param_validation_failed = True
                                break
                            converted_value = value.lower() == "true"
                        else:
                            converted_value = bool(value)

                    # Integer
                    elif expected_type == int:
                        if isinstance(value, int):
                            converted_value = value
                        elif isinstance(value, str):
                            converted_value = int(value)
                        else:
                            converted_value = int(value)

                    # Float
                    elif expected_type == float:
                        if isinstance(value, float):
                            converted_value = value
                        elif isinstance(value, str):
                            converted_value = float(value)
                        else:
                            converted_value = float(value)

                    else:
                        # For other expected types, allow passing native types or string repr
                        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                            converted_value = eval(value)
                        else:
                            converted_value = value
                else:
                    # No expected type available: try sensible conversions for strings
                    if isinstance(value, str):
                        low = value.lower()
                        if low in ("true", "false"):
                            converted_value = low == "true"
                        else:
                            # try int then float then list
                            try:
                                converted_value = int(value)
                            except Exception:
                                try:
                                    converted_value = float(value)
                                except Exception:
                                    if value.startswith("[") and value.endswith("]"):
                                        converted_value = eval(value)
                                    else:
                                        converted_value = value
                    else:
                        converted_value = value

                layer_params[key] = converted_value
            except Exception as e:
                failed_layers.append({
                    "index": i,
                    "layer_type": layer_type,
                    "error": f"Error converting parameter '{key}' value '{value}': {e}",
                    "parameter": key,
                    "value": value,
                    "error_details": str(e),
                })
                param_validation_failed = True
                break

        if param_validation_failed:
            continue

        # Validate position parameter (relative to current build state)
        layers_list = arch_manager.current_architecture["layers"]
        if position is not None:
            if not (0 <= position <= len(layers_list)):
                failed_layers.append({
                    "index": i,
                    "layer_type": layer_type,
                    "error": f"Invalid position {position}. Must be between 0 and {len(layers_list)} (or omit for end)",
                    "position": position,
                    "valid_range": f"0-{len(layers_list)}",
                    "layers_count": len(layers_list),
                })
                continue
        else:
            position = len(layers_list)  # Default to end

        # Add layer
        try:
            layer = arch_manager.add_layer(layer_type, layer_params, position)
            
            # Get analysis results for this layer
            analysis_results = Analyze_for_MCP(arch_manager.current_architecture["layers"])
            total_analysis.extend(analysis_results)

            added_layers.append({
                "index": i,
                "layer_type": layer_type,
                "position": position,
                "parameters_count": len(layer_params),
                "parameters": layer_params,
                "analysis": analysis_results,
            })
        except Exception as e:
            failed_layers.append({
                "index": i,
                "layer_type": layer_type,
                "error": f"Failed to add layer: {e}",
                "error_details": str(e),
            })

    # Save architecture if file_path is provided
    if file_path and added_layers:
        arch_manager.save_architecture(file_path)
        logger.info(f"Architecture saved to {file_path} after setting layers", extra={'session_id': ctx.session_id})

    # Return comprehensive results
    status = "success" if not failed_layers else "partial_success" if added_layers else "error"
    message = f"Successfully set {len(added_layers)} layers" if added_layers else "No layers were set"
    
    if failed_layers:
        message += f", {len(failed_layers)} layers failed"

    return {
        "status": status,
        "message": message,
        "details": {
            "total_layers": len(layers_config),
            "successful_layers": len(added_layers),
            "failed_layers": len(failed_layers),
            "added_layers": added_layers,
            "failed_layers": failed_layers,
            "total_analysis": total_analysis,
            "file_saved": str(file_path) if file_path and added_layers else None,
        },
    }


@arch_mcp.tool()
async def add_residual_block(
    ctx: Context,
    in_channels: int,
    out_channels: int,
    layers: list[str],
    layer_params: list[str] = [],
):
    """Add a residual block to the current architecture.

    This function creates and adds a residual block to the current neural network architecture. A residual block
    consists of multiple layers (e.g., Conv2d, BatchNorm2d, ReLU) that form a residual connection pattern.
    The function validates input parameters, processes layer configurations, and creates the residual block structure.

    Args:
        ctx: The FastMCP context containing session state.
        in_channels: Number of input channels for the residual block.
        out_channels: Number of output channels for the residual block.
        layers: List of layer types to include in the residual block.
        layer_params: List of parameter strings for configuring layers in the residual block.

    Returns:
        dict: A dictionary containing status information and layer details.

    """
    logger.info(f"Adding residual block: in_channels={in_channels}, out_channels={out_channels}, layers={layers}", extra={'session_id': ctx.session_id})
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    # Validate input parameters
    if in_channels <= 0 or out_channels <= 0:
        return {
            "status": "error",
            "message": "Input and output channels must be positive integers",
            "details": {"in_channels": in_channels, "out_channels": out_channels},
        }

    # Validate that specified layers are available
    available_layers = arch_manager.list_available_layers()
    invalid_layers = [layer for layer in layers if layer not in available_layers]
    if invalid_layers:
        return {
            "status": "error",
            "message": f"Invalid layer types: {', '.join(invalid_layers)}",
            "details": {
                "invalid_layers": invalid_layers,
                "available_layers": available_layers[:10],
            },
        }

    # Parse layer parameters
    residual_layers = []
    current_param_index = 0

    for layer_type in layers:
        layer_info = arch_manager.get_layer_info(layer_type)
        if layer_info is None:
            return {
                "status": "error",
                "message": f"Could not get layer information for '{layer_type}'",
                "details": {"layer_type": layer_type},
            }

        valid_param_names = {param["name"] for param in layer_info}
        layer_params_dict = {}

        # Get parameters for this specific layer
        params_for_this_layer = []
        if current_param_index < len(layer_params):
            # Look for parameters that start with layer_type:
            for i in range(current_param_index, len(layer_params)):
                if ":" in layer_params[i]:
                    layer_name, param = layer_params[i].split(":", 1)
                    if layer_name == layer_type:
                        params_for_this_layer.append(param)
                        current_param_index = i + 1
                    else:
                        break
                else:
                    current_param_index = i + 1
                    break

        # Process parameters for this layer
        for param in params_for_this_layer:
            if "=" in param:
                key, value = param.split("=", 1)

                if key not in valid_param_names:
                    return {
                        "status": "error",
                        "message": f"Parameter '{key}' is not valid for layer type '{layer_type}'",
                        "details": {
                            "parameter": key,
                            "layer_type": layer_type,
                            "valid_parameters": sorted(valid_param_names),
                        },
                    }

                # Type validation and conversion
                expected_type = None
                for param_info in layer_info:
                    if param_info["name"] == key:
                        expected_type = param_info.get("type")
                        break
                if expected_type:
                    try:
                        if expected_type == bool:
                            if value.lower() not in ["true", "false"]:
                                return {
                                    "status": "error",
                                    "message": f"Parameter '{key}' expects boolean value (true/false), got: {value}",
                                    "details": {
                                        "parameter": key,
                                        "expected_type": "boolean",
                                        "received_value": value,
                                    },
                                }
                            layer_params_dict[key] = value.lower() == "true"
                        elif expected_type == int:
                            if not value.isdigit() and not (
                                value.startswith("-") and value[1:].isdigit()
                            ):
                                return {
                                    "status": "error",
                                    "message": f"Parameter '{key}' expects integer value, got: {value}",
                                    "details": {
                                        "parameter": key,
                                        "expected_type": "integer",
                                        "received_value": value,
                                    },
                                }
                            layer_params_dict[key] = int(value)
                        elif expected_type == float:
                            if not value.replace(".", "", 1).replace(
                                "-",
                                "",
                                1,
                            ).isdigit() and value not in ["inf", "-inf"]:
                                return {
                                    "status": "error",
                                    "message": f"Parameter '{key}' expects float value, got: {value}",
                                    "details": {
                                        "parameter": key,
                                        "expected_type": "float",
                                        "received_value": value,
                                    },
                                }
                            layer_params_dict[key] = float(value)
                        elif value.startswith("[") and value.endswith("]"):
                            layer_params_dict[key] = eval(value)
                        else:
                            layer_params_dict[key] = value
                    except Exception as e:
                        return {
                            "status": "error",
                            "message": f"Error converting parameter '{key}' value '{value}': {e}",
                            "details": {
                                "parameter": key,
                                "value": value,
                                "error": str(e),
                            },
                        }
                else:
                    # Fallback conversion
                    try:
                        if value.lower() in ["true", "false"]:
                            layer_params_dict[key] = value.lower() == "true"
                        elif value.isdigit():
                            layer_params_dict[key] = int(value)
                        elif value.replace(".", "", 1).isdigit():
                            layer_params_dict[key] = float(value)
                        elif value.startswith("[") and value.endswith("]"):
                            layer_params_dict[key] = eval(value)
                        else:
                            layer_params_dict[key] = value
                    except:
                        layer_params_dict[key] = value

        # Create the layer
        layer = {
            "type": layer_type,
            "params": layer_params_dict,
            "name": f"{layer_type.lower()}_{len(residual_layers) + 1}",
        }
        residual_layers.append(layer)

    # Create residual block
    residual_block = {
        "type": "Residual_Block",
        "params": {
            "in_channels": in_channels,
            "out_channels": out_channels,
        },
        "layers": residual_layers,
        "name": f"residual_block_{len(arch_manager.current_architecture['layers']) + 1}",
    }

    # Add residual block to architecture
    arch_manager.current_architecture["layers"].append(residual_block)

    # Get analysis results for Layer compatibility
    analysis_results = Analyze_for_MCP(arch_manager.current_architecture["layers"])

    return {
        "status": "success",
        "message": "Residual block added successfully!",
        "details": {
            "residual_block_type": "Residual_Block",
            "position": len(arch_manager.current_architecture["layers"]),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "internal_layers_count": len(residual_layers),
            "internal_layers": [layer["type"] for layer in residual_layers],
            "analysis": analysis_results,
        },
    }


@arch_mcp.tool()
async def remove_layer(
    ctx: Context,
    index: int,
):
    """Remove a layer from the current architecture.

    This function removes a specified layer from the current neural network architecture by its index.
    It validates that the architecture has layers and that the provided index is valid before removing the layer.

    Args:
        ctx: The FastMCP context containing session state.
        index: The zero-based index of the layer to remove from the architecture.

    Returns:
        dict: A dictionary containing status information and layer details.

    """
    logger.info(f"Removing layer at index: {index}", extra={'session_id': ctx.session_id})
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    layers_list = arch_manager.current_architecture["layers"]

    if not layers_list:
        return {"status": "error", "message": "No layers in current architecture"}

    if not (0 <= index < len(layers_list)):
        return {
            "status": "error",
            "message": f"Invalid layer index {index}. Must be between 0 and {len(layers_list) - 1}",
            "details": {
                "index": index,
                "valid_range": f"0-{len(layers_list) - 1}",
                "layers_count": len(layers_list),
            },
        }

    arch_manager.remove_layer(index)

    return {
        "status": "success",
        "message": "Layer removed successfully!",
        "details": {
            "index": index,
            "remaining_layers": len(arch_manager.current_architecture["layers"]),
        },
    }


@arch_mcp.tool()
async def remove_residual_block(
    ctx: Context,
    index: int,
):
    """Remove a residual block from the current architecture.

    This function removes a residual block from the current neural network architecture by its index.
    It validates that the architecture has layers, that the provided index is valid, and that the layer
    at the specified index is actually a residual block before removing it.

    Args:
        ctx: The FastMCP context containing session state.
        index: The zero-based index of the residual block to remove from the architecture.

    Returns:
        dict: A dictionary containing status information and layer details.

    """
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    layers_list = arch_manager.current_architecture["layers"]

    if not layers_list:
        return {"status": "error", "message": "No layers in current architecture"}

    if not (0 <= index < len(layers_list)):
        return {
            "status": "error",
            "message": f"Invalid layer index {index}. Must be between 0 and {len(layers_list) - 1}",
            "details": {
                "index": index,
                "valid_range": f"0-{len(layers_list) - 1}",
                "layers_count": len(layers_list),
            },
        }

    layer = layers_list[index]
    if layer["type"] != "Residual_Block":
        return {
            "status": "error",
            "message": f"Layer at index {index} is not a residual block (type: {layer['type']})",
            "details": {
                "index": index,
                "layer_type": layer["type"],
                "expected_type": "Residual_Block",
            },
        }

    arch_manager.remove_layer(index)

    return {
        "status": "success",
        "message": "Residual block removed successfully!",
    }


@arch_mcp.tool()
async def move_layer(
    ctx: Context,
    from_index: int,
    to_index: int,
):
    """Move a layer from one position to another.

    This function moves a layer from one position to another within the current neural network architecture.
    It validates that the architecture has layers and that both the source and destination indices are valid.
    If the source and destination indices are the same, it prints a warning message.

    Args:
        ctx: The FastMCP context containing session state.
        from_index: The current zero-based index of the layer to move.
        to_index: The new zero-based index where the layer should be moved.

    Returns:
        dict: A dictionary containing status information and layer details.

    """
    logger.info(f"Moving layer from {from_index} to {to_index}", extra={'session_id': ctx.session_id})
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    layers_list = arch_manager.current_architecture["layers"]

    if not layers_list:
        return {"status": "error", "message": "No layers in current architecture"}

    if not (0 <= from_index < len(layers_list)):
        return {
            "status": "error",
            "message": f"Invalid from_index {from_index}. Must be between 0 and {len(layers_list) - 1}",
        }

    if not (0 <= to_index < len(layers_list)):
        return {
            "status": "error",
            "message": f"Invalid to_index {to_index}. Must be between 0 and {len(layers_list) - 1}",
        }

    if from_index == to_index:
        return {
            "status": "warning",
            "message": "Layer is already at the target position",
        }

    success = arch_manager.move_layer(from_index, to_index)

    if success:
        return {
            "status": "success",
            "message": f"Layer moved from position {from_index} to {to_index}",
            "details": {
                "from_index": from_index,
                "to_index": to_index,
                "layers_count": len(layers_list),
            },
        }
    return {
        "status": "error",
        "message": "Failed to move layer",
        "details": {"from_index": from_index, "to_index": to_index},
    }


@arch_mcp.tool()
async def get_layer_info_batch(
    ctx: Context,
    layer_types: list[str],
):
    """Get layer information for multiple layer types.

    This function retrieves detailed information for multiple layer types in a single call.
    For each requested layer type, it returns the parameter definitions, types, and other
    metadata. If a layer type is not found, it includes an error status for that specific layer.

    Args:
        ctx: The FastMCP context containing session state.
        layer_types: List of layer type names to retrieve information for.

    Returns:
        dict: A dictionary containing layer information for each requested type.

    """
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    if not layer_types:
        return {
            "status": "error",
            "message": "No layer types provided",
            "details": {"requested_count": 0},
        }

    layers_info = {}
    found_count = 0

    for layer_type in layer_types:
        layer_info = arch_manager.get_layer_info(layer_type)

        if layer_info:
            layers_info[layer_type] = {
                "status": "found",
                "parameters": layer_info,
                "parameters_count": len(layer_info),
            }
            found_count += 1
        else:
            layers_info[layer_type] = {
                "status": "error",
                "message": f"Layer type '{layer_type}' not found",
                "available_layers": arch_manager.list_available_layers()[:10],  # Show first 10 available
            }

    return {
        "status": "success",
        "message": f"Layer info retrieved for {found_count} of {len(layer_types)} requested types",
        "details": {
            "layers_info": layers_info,
            "requested_count": len(layer_types),
            "found_count": found_count,
            "available_layers_count": len(arch_manager.layers),
        },
    }


@arch_mcp.tool()
async def list_layers(
    ctx: Context,
):
    """List all layers in the current architecture.

    This function displays a formatted table of all layers currently in the neural network architecture.
    It shows each layer's index, type, parameters, and name. For residual blocks, it also displays
    information about the internal layers. If no layers exist, it prints a message indicating an empty architecture.

    Args:
        ctx: The FastMCP context containing session state.

    Returns:
        dict: A dictionary containing status information and layer details.

    """
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    layers_list = arch_manager.current_architecture["layers"]

    if not layers_list:
        return {
            "status": "info",
            "message": "No layers in current architecture",
            "details": {"layers_count": len(layers_list)},
        }

    return {
        "status": "success",
        "message": "Layers retrieved successfully",
        "details": {"layers": layers_list, "layers_count": len(layers_list)},
    }


@arch_mcp.tool()
async def list_available_layers(ctx: Context):
    """List all available layer types.

    This function retrieves and organizes all available layer types from the ArchitectureManager,
    grouping them by category for better display. It categorizes layers into Convolutional, Pooling,
    Linear, Normalization, Activation, Dropout, and Other categories. Returns the organized sections
    for potential use in other functions or displays.

    Args:
        ctx: The FastMCP context containing session state.

    Returns:
        list: A list of tuples containing (category_name, layer_list) for each layer category.

    """
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    available_layers = arch_manager.list_available_layers()

    if not available_layers:
        return None

    # Group layers by category for better display
    conv_layers = [l for l in available_layers if "conv" in l.lower()]
    pool_layers = [l for l in available_layers if "pool" in l.lower()]
    linear_layers = [l for l in available_layers if "linear" in l.lower()]
    norm_layers = [
        l for l in available_layers if "norm" in l.lower() or "batch" in l.lower()
    ]
    activation_layers = [
        l
        for l in available_layers
        if any(x in l.lower() for x in ["relu", "tanh", "sigmoid", "softmax"])
    ]
    dropout_layers = [l for l in available_layers if "dropout" in l.lower()]
    other_layers = [
        l
        for l in available_layers
        if l
        not in conv_layers
        + pool_layers
        + linear_layers
        + norm_layers
        + activation_layers
        + dropout_layers
    ]

    # Display in organized sections
    sections = [
        ("Convolutional", conv_layers),
        ("Pooling", pool_layers),
        ("Linear", linear_layers),
        ("Normalization", norm_layers),
        ("Activation", activation_layers),
        ("Dropout", dropout_layers),
        ("Other", other_layers),
    ]

    return sections


@arch_mcp.tool()
async def list_pretrained_models(ctx: Context):
    """List all available pretrained models.

    This function retrieves and organizes all available pretrained models from the ArchitectureManager,
    grouping them by category for better display. It categorizes models into Torchvision Models, YOLOX Models,
    EfficientNet Models, MobileNet Models, ResNet Models, DenseNet Models, VGG Models, and Other Models.
    Displays each category with its available models if any exist.

    Args:
        ctx: The FastMCP context containing session state.

    Returns:
        dict: A dictionary containing status information and model categories.
            - status: "success"
            - details: Dictionary with model_categories containing organized model sections

    """
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    available_models = arch_manager.list_available_pretrained_models()

    if not available_models:
        return {"status": "info", "message": "No pretrained models available"}

    # Group models by category for better display
    torchvision_models = [
        m
        for m in available_models
        if not any(
            x in m.lower()
            for x in ["yolox", "efficientnet", "mobilenet", "resnet", "densenet", "vgg"]
        )
    ]
    yolox_models = [m for m in available_models if "yolox" in m.lower()]
    efficientnet_models = [m for m in available_models if "efficientnet" in m.lower()]
    mobilenet_models = [m for m in available_models if "mobilenet" in m.lower()]
    resnet_models = [m for m in available_models if "resnet" in m.lower()]
    densenet_models = [m for m in available_models if "densenet" in m.lower()]
    vgg_models = [m for m in available_models if "vgg" in m.lower()]
    other_models = [
        m
        for m in available_models
        if m
        not in torchvision_models
        + yolox_models
        + efficientnet_models
        + mobilenet_models
        + resnet_models
        + densenet_models
        + vgg_models
    ]

    # Display in organized sections
    sections = [
        ("Torchvision Models", torchvision_models),
        ("YOLOX Models", yolox_models),
        ("EfficientNet Models", efficientnet_models),
        ("MobileNet Models", mobilenet_models),
        ("ResNet Models", resnet_models),
        ("DenseNet Models", densenet_models),
        ("VGG Models", vgg_models),
        ("Other Models", other_models),
    ]

    return {"status": "success", "details": {"model_categories": sections}}


@arch_mcp.tool()
async def show_architecture(
    ctx: Context,
    file_path: Path | None = None,
):
    """Show the current architecture or load from file.

    This function displays the current neural network architecture with detailed information including
    the number of layers, input dimensions, training parameters, and device configuration. If a file path
    is provided, it loads the architecture from that file first. It also shows the configuration status
    for optimizer, loss function, scheduler, and pretrained model.

    Args:
        arch_manager: The ArchitectureManager instance containing the current architecture.
        file_path: Optional path to an architecture file to load before displaying.

    Returns:
        dict: A dictionary containing status information and architecture details.

    """
    logger.info(f"Showing architecture, file_path: {file_path}", extra={'session_id': ctx.session_id})
    if file_path:
        if not file_path.exists():
            return {
                "status": "error",
                "message": f"Architecture file {file_path} not found",
                "details": {"file_path": str(file_path)},
            }

    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }
    # Load architecture from file only if a file_path was provided
    if file_path:
        arch_manager.load_architecture(file_path)
    architecture = arch_manager.current_architecture

    # Build architecture overview
    architecture_overview = {
        "layers_count": len(architecture["layers"]),
        "input_size": {
            "height": architecture["misc_params"]["height"],
            "width": architecture["misc_params"]["width"],
            "channels": architecture["misc_params"]["channels"],
        },
        "training_config": {
            "epochs": architecture["misc_params"]["num_epochs"],
            "batch_size": architecture["misc_params"]["batch_size"],
        },
        "device": architecture["misc_params"]["device"]["value"],
        "dataset": architecture["misc_params"]["dataset"]["value"],
    }

    # Build configuration status
    config_status = {}
    optimizer = architecture.get("optimizer", {})
    loss_func = architecture.get("loss_func", {})
    scheduler = architecture.get("scheduler", {})
    pretrained = architecture.get("pretrained", {})

    config_status["optimizer"] = "configured" if optimizer else "not_set"
    config_status["loss_function"] = "configured" if loss_func else "not_set"
    config_status["scheduler"] = "configured" if scheduler else "not_set"
    config_status["pretrained_model"] = "set" if pretrained.get("value") else "not_set"

    return {
        "status": "success",
        "message": "Architecture information retrieved successfully",
        "details": {
            "architecture_overview": architecture_overview,
            "configuration_status": config_status,
            "layers": architecture["layers"] if architecture["layers"] else [],
        },
    }


@arch_mcp.tool()
async def save_arch(
    ctx: Context,
    file_path: Path,
):
    """Save the current architecture to a file.

    This function saves the current neural network architecture to the specified file path after
    validating and preparing it. It attempts to validate the architecture and apply any necessary
    model loading logic. If validation fails, it saves the architecture without validation while
    displaying a warning message. The function provides detailed success feedback including the file path
    and number of layers saved.

    Args:
        arch_manager: The ArchitectureManager instance containing the current architecture to save.
        file_path: The file path where the architecture will be saved.

    Returns:
        dict: A dictionary containing status information and save details.

    """
    logger.info(f"Saving architecture to: {file_path}", extra={'session_id': ctx.session_id})
    # Validate and prepare architecture using validation logic
    # Get current architecture for validation
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }
    try:
        current_arch = arch_manager.current_architecture.copy()

        # Apply validation and model loading logic
        validated_arch = arch_manager.validate_architecture(current_arch)
        prepared_arch = arch_manager.prepare_architecture(validated_arch)

        # Update the architecture manager with validated architecture
        arch_manager.current_architecture = prepared_arch

    except Exception as e:
        arch_manager.save_architecture(file_path)
        return {
            "status": "warning",
            "message": f"Validation/Model loading failed: {e}. Saving without validation.",
            "details": {"error": str(e), "action": "saved_without_validation"},
        }

    arch_manager.save_architecture(file_path)

    return {
        "status": "success",
        "message": "Architecture saved successfully!",
        "details": {
            "file_path": str(file_path),
            "layers_count": len(arch_manager.current_architecture["layers"]),
        },
    }


@arch_mcp.tool()
async def load_arch(
    ctx: Context,
    file_path: Path,
):
    """Load an architecture from a file.

    This function loads a neural network architecture from the specified file path into the current
    ArchitectureManager instance. It validates that the file exists before attempting to load it.
    Upon successful loading, it displays detailed information about the loaded architecture including
    the file path and number of layers.

    Args:
        arch_manager: The ArchitectureManager instance where the architecture will be loaded.
        file_path: The file path from which to load the architecture.

    Returns:
        dict: A dictionary containing status information and load details.

    """
    logger.info(f"Loading architecture from: {file_path}", extra={'session_id': ctx.session_id})
    if not file_path.exists():
        return {
            "status": "error",
            "message": f"Architecture file {file_path} not found",
            "details": {"file_path": str(file_path)},
        }

    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }
    arch_manager.load_architecture(file_path)

    return {
        "status": "success",
        "message": "Architecture loaded successfully!",
        "details": {
            "file_path": str(file_path),
            "layers_count": len(arch_manager.current_architecture["layers"]),
        },
    }





# ============= Configuration Server Tools =============

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
    logger.info(f"Showing configuration", extra={'session_id': ctx.session_id})
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }
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
            "pretrained_weights": complex_params.get("pretrained_weights", "Not set"),
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
        pretrained_data.update({"depth": pretrained["depth"], "width": pretrained["width"]})

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
    target_accuracy: float | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    dataset: str | None = None,
    dataset_path: str | None = None,
    file_path: Path | None = None,
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
        file_path: Optional path to save the architecture after setting model parameters.

    Returns:
        dict: A dictionary containing operation status.
            - status: "success", "error", or "warning"
            - message: Human-readable status message
            - details: Dictionary with updated parameters

    """
    logger.info(
        "Setting model params: height=%s, width=%s, channels=%s, epochs=%s, target_accuracy=%s, "
        "batch_size=%s,device=%s,dataset=%s,dataset_path=%s, file_path=%s",
        height, width, channels, epochs, target_accuracy, batch_size, device, dataset, dataset_path, file_path,
        extra={'session_id': ctx.session_id}
    )
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }
    architecture = arch_manager.current_architecture
    misc_params = architecture["misc_params"]

    # Update parameters
    updates = {}
    if height is not None:
        updates["height"] = height
        misc_params["height"] = height
    if width is not None:
        updates["width"] = width
        misc_params["width"] = width
    if channels is not None:
        updates["channels"] = channels
        misc_params["channels"] = channels
    if epochs is not None:
        updates["num_epochs"] = epochs
        misc_params["num_epochs"] = epochs
    if target_accuracy is not None:
        updates["target_accuracy"] = target_accuracy
        misc_params["target_accuracy"] = target_accuracy
    if batch_size is not None:
        updates["batch_size"] = batch_size
        misc_params["batch_size"] = batch_size
    if device is not None:
        #if user has multiple gpus and specifies "gpu", default to "cuda:0"
        if device == "gpu":
            device = "cuda:0"
        updates["device"] = device
        misc_params["device"] = {"value": device, "index": 1 if device.startswith("cuda") else 0}
    if dataset is not None:
        # Get available datasets from architecture manager to find the index
        available_datasets = arch_manager.datasets if hasattr(arch_manager, 'datasets') else []
        dataset_index = 0
        dataset_value = dataset
        # Exact match first
        if dataset in available_datasets:
            dataset_index = available_datasets.index(dataset)
            dataset_value = available_datasets[dataset_index]
        else:
            # Try case-insensitive match
            if isinstance(dataset, str):
                for i, d in enumerate(available_datasets):
                    if d.lower() == dataset.lower():
                        dataset_index = i
                        dataset_value = d
                        break
        updates["dataset"] = dataset_value
        misc_params["dataset"] = {"value": dataset_value, "index": dataset_index}
    if dataset_path is not None:
        updates["dataset_path"] = dataset_path
        misc_params["dataset_path"] = dataset_path

    if not updates:
        logger.info("No model parameters provided to update", extra={'session_id': ctx.session_id})
        return {
            "status": "warning",
            "message": "No parameters provided to update",
            "details": {"updates": {}},
        }

    # Save architecture if file_path is provided
    logger.info(f"set_model_params: file_path={file_path}, type={type(file_path)}, bool={bool(file_path)}", extra={'session_id': ctx.session_id})
    if file_path:
        arch_manager.save_architecture(file_path)
        logger.info(f"Architecture saved to {file_path} after setting model params", extra={'session_id': ctx.session_id})

    return {
        "status": "success",
        "message": "Model parameters updated successfully",
        "details": {"updates": updates},
    }


@arch_mcp.tool()
async def set_complex_params(
    ctx: Context,
    data_num_workers: int | None = None,
    eval_interval: int | None = None,
    warmup_epochs: int | None = None,
    scheduler: str | None = None,
    num_classes: int | None = None,
    pretrained_weights: str | None = None,
    file_path: Path | None = None,
) -> dict[str, Any]:
    """Set complex model parameters.

    Args:
        ctx: FastMCP context containing the architecture manager state.
        data_num_workers: Number of data workers for data loading.
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
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    logger.info(f"Setting complex params: data_num_workers={data_num_workers}, eval_interval={eval_interval}, warmup_epochs={warmup_epochs}, scheduler={scheduler}, num_classes={num_classes}, pretrained_weights={pretrained_weights}", extra={'session_id': ctx.session_id})
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    # Ensure complex fields exist
    arch_manager._ensure_complex_fields()

    architecture = arch_manager.current_architecture
    complex_params = architecture["complex_misc_params"]

    # Update parameters
    updates = {}
    if data_num_workers is not None:
        updates["data_num_workers"] = data_num_workers
        complex_params["data_num_workers"] = data_num_workers
    if eval_interval is not None:
        updates["eval_interval"] = eval_interval
        complex_params["eval_interval"] = eval_interval
    if warmup_epochs is not None:
        updates["warmup_epochs"] = warmup_epochs
        complex_params["warmup_epochs"] = warmup_epochs
    if scheduler is not None:
        valid_schedulers = ["cos", "linear"]
        if scheduler not in valid_schedulers:
            return {
                "status": "error",
                "message": f"Invalid scheduler type. Must be one of: {', '.join(valid_schedulers)}",
                "details": {"valid_schedulers": valid_schedulers},
            }
        updates["scheduler"] = scheduler
        complex_params["scheduler"] = scheduler
    if num_classes is not None:
        updates["num_classes"] = num_classes
        complex_params["num_classes"] = num_classes
    if pretrained_weights is not None:
        updates["pretrained_weights"] = pretrained_weights
        architecture["pretrained_weights"] = pretrained_weights
        
    if file_path is not None:
        arch_manager.save_architecture(file_path)
        logger.info(f"Architecture saved to {file_path} after setting complex params", extra={'session_id': ctx.session_id})

    if not updates:
        return {
            "status": "warning",
            "message": "No parameters provided to update",
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
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    if arch_manager.complex_arch_models:
        return {
            "status": "success",
            "message": f"Found {len(arch_manager.complex_arch_models)} complex models",
            "details": {
                "count": len(arch_manager.complex_arch_models),
                "models": arch_manager.complex_arch_models,
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
    params: dict[str, Any] | None = None,
    file_path: Path | None = None,
) -> dict[str, Any]:
    """Set the optimizer configuration.

    Args:
        ctx: FastMCP context containing the architecture manager state.
        optimizer_type: Type of optimizer (Adam, SGD, AdamW, etc.).
        params: Dictionary of optimizer parameter names to values. Values may be native
            Python types or strings (e.g. "0.001", "[0.9,0.999]"). Strings will be
            converted based on the optimizer metadata when possible.
        file_path: Optional path to save the architecture after setting optimizer.

    Returns:
        dict: A dictionary containing operation status.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with optimizer configuration

    """
    logger.info(f"Setting optimizer: {optimizer_type}, file_path: {file_path}", extra={'session_id': ctx.session_id})
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    # Case-insensitive match for optimizer type
    optimizer_key = None
    for k in arch_manager.optimizers.keys():
        if k.lower() == optimizer_type.lower():
            optimizer_key = k
            break
    if optimizer_key is None:
        return {
            "status": "error",
            "message": f"Invalid optimizer type: {optimizer_type}",
            "details": {
                "valid_optimizers": list(arch_manager.optimizers.keys()),
            },
        }
    unnecessary_optimizer_params = {"params"}

    # Get valid parameter names for this optimizer
    optimizer_info = arch_manager.optimizers[optimizer_key]
    valid_param_names = {param["name"] for param in optimizer_info}
    valid_param_names.difference_update(unnecessary_optimizer_params)

    # Parse parameters (expect a dict now)
    optimizer_params = {}
    parse_errors = []

    if params is not None:
        if not isinstance(params, dict):
            return {
                "status": "error",
                "message": "Invalid params format: expected a dictionary of parameter names to values",
                "details": {"received_type": type(params).__name__},
            }

        for key, raw_value in params.items():
            # Check if parameter is valid for this optimizer
            if key not in valid_param_names:
                parse_errors.append(f"Unknown parameter '{key}' for optimizer '{optimizer_type}'")
                continue

            # Get parameter info
            param_info = next(p for p in optimizer_info if p["name"] == key)
            param_type = param_info.get("type")

            try:
                value = raw_value
                # Normalize param_type to string when necessary
                tname = None
                if isinstance(param_type, str):
                    tname = param_type.lower()
                elif param_type in (int, float, bool, list):
                    tname = param_type.__name__

                if tname == "float":
                    if isinstance(value, str):
                        value = float(value)
                    else:
                        value = float(value)
                elif tname == "int":
                    if isinstance(value, str):
                        value = int(value)
                    else:
                        value = int(value)
                elif tname == "bool":
                    if isinstance(value, str):
                        value = value.lower() in ("true", "yes", "1", "t")
                    else:
                        value = bool(value)
                elif tname == "list":
                    if isinstance(value, str):
                        if not value.startswith("[") or not value.endswith("]"):
                            parse_errors.append(
                                f"List parameter '{key}' must be in format [item1,item2,...]"
                            )
                            continue
                        value = eval(value)
                    elif isinstance(value, (list, tuple)):
                        value = list(value)
                    else:
                        parse_errors.append(f"Unsupported list parameter value for '{key}': {value}")
                        continue
                else:
                    # keep as-is (string or native type)
                    value = value

                optimizer_params[key] = value

            except Exception:
                parse_errors.append(f"Could not convert parameter '{key}' value '{raw_value}' to type {param_type}")
                continue

    # Add default values for missing parameters or show error for required parameters without defaults
    missing_params = []
    for param_info in optimizer_info:
        param_name = param_info["name"]
        if param_name not in optimizer_params:
            # Try multiple keys for default value (different extraction formats)
            default_val = param_info.get("defaultvalue") or param_info.get("default") or param_info.get("defaultValue")
            if default_val is not None:
                optimizer_params[param_name] = default_val
            elif not param_info.get("required", False):
                # If parameter is not explicitly required, skip it (optional param with no default)
                pass
            else:
                missing_params.append(param_name)

    if missing_params:
        logger.error(f"Missing required optimizer params: {missing_params}, param_info sample: {optimizer_info[0] if optimizer_info else 'none'}", extra={'session_id': ctx.session_id})
        return {
            "status": "error",
            "message": "Missing required parameters",
            "details": {
                "missing_params": missing_params,
                "parse_errors": parse_errors,
            },
        }

    # Set optimizer in architecture
    arch_manager.current_architecture["optimizer"] = {
        "type": optimizer_key,
        "params": optimizer_params,
    }

    # Save architecture if file_path is provided
    logger.info(f"set_optimizer: file_path={file_path}, type={type(file_path)}, bool={bool(file_path)}", extra={'session_id': ctx.session_id})
    if file_path:
        arch_manager.save_architecture(file_path)
        logger.info(f"Architecture saved to {file_path} after setting optimizer", extra={'session_id': ctx.session_id})

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
    params: dict[str, Any] | None = None,
    file_path: Path | None = None,
) -> dict[str, Any]:
    """Set the loss function configuration.

    Args:
        ctx: FastMCP context containing the architecture manager state.
        loss_type: Type of loss function (CrossEntropyLoss, MSELoss, etc.).
        params: Dictionary of loss parameter names to values. Values may be native types
            or strings (e.g. "0.1", "True"). Strings will be converted based on
            loss function metadata when possible.
        file_path: Optional path to save the architecture after setting loss function.

    Returns:
        dict: A dictionary containing operation status.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with loss function configuration

    """
    logger.info(f"Setting loss function: {loss_type}, file_path: {file_path}", extra={'session_id': ctx.session_id})
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    # Case-insensitive match for loss function type
    loss_key = None
    for k in arch_manager.loss_funcs.keys():
        if k.lower() == loss_type.lower():
            loss_key = k
            break
    if loss_key is None:
        return {
            "status": "error",
            "message": f"Invalid loss function type: {loss_type}",
            "details": {
                "valid_loss_functions": list(arch_manager.loss_funcs.keys()),
            },
        }
    unnecessary_loss_params = {"reduce", "size_average", "weight"}
    # Get valid parameter names for this loss function
    loss_info = arch_manager.loss_funcs[loss_key]
    valid_param_names = {param["name"] for param in loss_info}
    valid_param_names.difference_update(unnecessary_loss_params)

    # Parse parameters (expect dict)
    loss_params = {}

    if params is not None:
        if not isinstance(params, dict):
            return {
                "status": "error",
                "message": "Invalid params format: expected a dictionary of parameter names to values",
                "details": {"received_type": type(params).__name__},
            }

        for key, raw_value in params.items():
            # Check if parameter is valid for this loss function
            if key not in valid_param_names:
                return {
                    "status": "error",
                    "message": f"Unknown parameter '{key}' for loss function '{loss_type}'",
                    "details": {"valid_parameters": list(valid_param_names)},
                }

            # Get parameter info
            param_info = next(p for p in loss_info if p["name"] == key)
            param_type = param_info.get("type")

            try:
                value = raw_value
                tname = None
                if isinstance(param_type, str):
                    tname = param_type.lower()
                elif param_type in (int, float, bool, list, str):
                    tname = param_type.__name__

                if tname == "float":
                    value = float(value)
                elif tname == "int":
                    value = int(value)
                elif tname == "bool":
                    if isinstance(value, str):
                        value = value.lower() in ("true", "yes", "1", "t")
                    else:
                        value = bool(value)
                elif tname == "list":
                    if isinstance(value, str):
                        if not value.startswith("[") or not value.endswith("]"):
                            return {
                                "status": "error",
                                "message": f"List parameter '{key}' must be in format [item1,item2,...]",
                            }
                        value = eval(value)
                    elif isinstance(value, (list, tuple)):
                        value = list(value)
                    else:
                        return {
                            "status": "error",
                            "message": f"Unsupported list parameter value for '{key}': {value}",
                        }
                elif tname == "str" or tname is None:
                    # keep as string or original
                    value = value
                else:
                    value = value

                loss_params[key] = value

            except Exception:
                return {
                    "status": "error",
                    "message": f"Could not convert parameter '{key}' value '{raw_value}' to type {param_type}",
                }

    # Add default values for missing parameters
    missing_params = []
    for param_info in loss_info:
        param_name = param_info["name"]
        if param_name not in loss_params:
            if "defaultvalue" in param_info:
                loss_params[param_name] = param_info["defaultvalue"]
            else:
                # Convert type to string for JSON serialization
                param_type = param_info["type"]
                type_str = param_type.__name__ if isinstance(param_type, type) else str(param_type)
                missing_params.append(
                    {
                        "name": param_name,
                        "type": type_str,
                        "description": param_info.get("description", "No description available"),
                    }
                )

    if missing_params:
        return {
            "status": "error",
            "message": "Missing required parameters",
            "details": {
                "missing_params": missing_params,
            },
        }

    # Set loss function in architecture
    arch_manager.current_architecture["loss_func"] = {
        "type": loss_key,
        "params": loss_params,
    }

    # Save architecture if file_path is provided
    if file_path:
        arch_manager.save_architecture(file_path)
        logger.info(f"Architecture saved to {file_path} after setting loss function", extra={'session_id': ctx.session_id})

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
    params: dict[str, Any] | None = None,
    file_path: Path | None = None,
) -> dict[str, Any]:
    """Set the learning rate scheduler configuration.

    Args:
        ctx: FastMCP context containing the architecture manager state.
        scheduler_type: Type of scheduler (StepLR, CosineAnnealingLR, None, etc.).
        params: Dictionary of scheduler parameter names to values. Values may be native
            Python types or strings (e.g. "10", "0.1", "[0.1,0.2]"). Strings will be
            converted based on scheduler metadata when possible.
        file_path: Optional path to save the architecture after setting scheduler.

    Returns:
        dict: A dictionary containing operation status.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with scheduler configuration

    """
    logger.info(f"Setting scheduler: {scheduler_type}, file_path: {file_path}", extra={'session_id': ctx.session_id})
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    # Handle "None" scheduler case (case-insensitive)
    if isinstance(scheduler_type, str) and scheduler_type.lower() == "none":
        arch_manager.current_architecture["scheduler"] = {
            "type": "None",
            "params": {},
        }
        # Save architecture if file_path is provided
        if file_path:
            arch_manager.save_architecture(file_path)
            logger.info(f"Architecture saved to {file_path} after setting scheduler", extra={'session_id': ctx.session_id})
        return {
            "status": "success",
            "message": "Scheduler disabled (set to None)",
            "details": {
                "scheduler_type": "None",
                "parameters": {},
            },
        }

    # Case-insensitive match for scheduler type
    scheduler_key = None
    for k in arch_manager.schedulers.keys():
        if k.lower() == scheduler_type.lower():
            scheduler_key = k
            break
    if scheduler_key is None:
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
    
    unnecessary_schedulers_params = {
            "optimizer",
            "lr_lambda",
            "milestones",
            "schedulers",
            "verbose",
            "scale_fn",
        }

    # Get valid parameter names for this scheduler
    scheduler_info = arch_manager.schedulers[scheduler_key]
    valid_param_names = {param["name"] for param in scheduler_info}
    valid_param_names.difference_update(unnecessary_schedulers_params)

    # Parse parameters (expect dict)
    scheduler_params = {}

    if params is not None:
        if not isinstance(params, dict):
            return {
                "status": "error",
                "message": "Invalid params format: expected a dictionary of parameter names to values",
                "details": {"received_type": type(params).__name__},
            }

        for key, raw_value in params.items():
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
                value = raw_value
                tname = None
                if isinstance(expected_type, str):
                    tname = expected_type.lower()
                elif expected_type in (int, float, bool, list):
                    tname = expected_type.__name__

                if tname == "bool":
                    if isinstance(value, str):
                        if value.lower() not in ["true", "false"]:
                            return {
                                "status": "error",
                                "message": f"Parameter '{key}' expects boolean value (true/false), got: {value}",
                                "details": {"parameter": key, "value": value, "expected_type": "bool"},
                            }
                        value = value.lower() == "true"
                    else:
                        value = bool(value)
                elif tname == "int":
                    value = int(value)
                elif tname == "float":
                    value = float(value)
                elif tname == "list":
                    if isinstance(value, str):
                        if not value.startswith("[") or not value.endswith("]"):
                            return {
                                "status": "error",
                                "message": f"List parameter '{key}' must be in format [item1,item2,...]",
                            }
                        value = eval(value)
                    elif isinstance(value, (list, tuple)):
                        value = list(value)
                    else:
                        return {
                            "status": "error",
                            "message": f"Unsupported list parameter value for '{key}': {value}",
                        }
                else:
                    # try to interpret string lists or keep as-is
                    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                        value = eval(value)
                    else:
                        value = value

                scheduler_params[key] = value
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error converting parameter '{key}' value '{raw_value}': {e!s}",
                    "details": {"parameter": key, "value": raw_value, "error": str(e)},
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
        "type": scheduler_key,
        "params": scheduler_params,
    }

    # Save architecture if file_path is provided
    if file_path:
        arch_manager.save_architecture(file_path)
        logger.info(f"Architecture saved to {file_path} after setting scheduler", extra={'session_id': ctx.session_id})

    return {
        "status": "success",
        "message": "Scheduler configured successfully",
        "details": {
            "scheduler_type": scheduler_type,
            "parameters": scheduler_params,
        },
    }



# ============= Project Server Tools =============

@arch_mcp.tool()
def create_project(
    name: str,
    output_dir: Path = Path.cwd(),
    description: str | None = None,
    author: str | None = None,
) -> dict[str, Any]:
    """Create a new neural network project.

    This function creates a complete project structure with directories for
    configurations, architectures, outputs, and logs, along with default
    project configuration and architecture files.

    Args:
        name: Name of the project to create.
        output_dir: Directory where the project will be created.
        description: Optional description for the project.
        author: Optional author name for the project.

    Returns:
        dict: A dictionary containing status information and project details.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with project information including:
                - project_dir: Path where project was created
                - config_file: Path to project.json file
                - arch_file: Path to default architecture file
                - project_config: Complete project configuration

    """
    logger.info(f"Creating project: {name} in {output_dir}", extra={'session_id': 'system'})
    project_dir = output_dir / name

    if project_dir.exists():
        return {
            "status": "error",
            "message": f"Project '{name}' already exists in {output_dir}",
            "details": {"project_name": name, "output_dir": str(output_dir)},
        }

    # Create project structure
    try:
        project_dir.mkdir(parents=True)
        (project_dir / "architectures").mkdir()
        (project_dir / "outputs").mkdir()

        # Create project configuration
        project_config = {
            "name": name,
            "description": description or f"Neural network project: {name}",
            "author": author or "NN Generator CLI",
            "created_at": str(Path.cwd()),
            "version": "1.0.0",
            "architecture": {},
            "config": {},
            "status": "created",
        }

        config_file = project_dir / "project.json"
        with open(config_file, "w") as f:
            json.dump(project_config, f, indent=2)

        return {
            "status": "success",
            "message": "Project created successfully",
            "details": {
                "project_dir": str(project_dir),
                "config_file": str(config_file),
                "project_config": project_config,
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error creating project: {e}",
            "details": {"error": str(e)},
        }


@arch_mcp.tool()
def list_projects(
    directory: Path = Path.cwd(),
) -> dict[str, Any]:
    """List all neural network projects in a directory.

    This function searches for project.json files in the specified directory
    and its subdirectories, returning information about each discovered project.

    Args:
        directory: Directory to search for projects.


    Returns:
        dict: A dictionary containing project listing information.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary containing:
                - projects_count: Number of projects found
                - projects: List of project information dictionaries
                - search_directory: Directory that was searched

    """
    projects = []

    if directory.is_file():
        return {
            "status": "error",
            "message": f"{directory} is not a directory",
            "details": {"provided_path": str(directory)},
        }

    # Find all project.json files
    for root, dirs, files in os.walk(directory):
        if "project.json" in files:
            project_file = Path(root) / "project.json"
            try:
                with open(project_file) as f:
                    project_data = json.load(f)

                project_info = {
                    "name": project_data.get("name", "Unknown"),
                    "path": str(Path(root).relative_to(directory)),
                    "description": project_data.get("description", "No description"),
                    "author": project_data.get("author", "Unknown"),
                    "status": project_data.get("status", "unknown"),
                    "created": project_data.get("created_at", "Unknown"),
                }
                projects.append(project_info)

            except Exception:
                # Log error but continue with other projects
                pass

    if not projects:
        return {
            "status": "success",
            "message": f"No projects found in {directory}",
            "details": {
                "projects_count": 0,
                "projects": [],
                "search_directory": str(directory),
            },
        }

    return {
        "status": "success",
        "message": f"Found {len(projects)} project(s)",
        "details": {
            "projects_count": len(projects),
            "projects": projects,
            "search_directory": str(directory),
        },
    }


@arch_mcp.tool()
def load_project(
    project_path: Path,
    set_active: bool = True,
) -> dict[str, Any]:
    """Load an existing neural network project.

    This function loads project configuration from a project.json file
    and validates the project structure. Optionally sets the project as active.

    Args:
        project_path: Path to project directory or project.json file.
        set_active: Whether to set this project as the active project.

    Returns:
        dict: A dictionary containing status information and project details.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with project information including:
                - project_name: Name of the loaded project
                - project_dir: Directory containing the project
                - project_file: Path to project.json file
                - project_data: Complete project configuration
                - set_as_active: Whether project was set as active

    """
    logger.info(f"Loading project from: {project_path}", extra={'session_id': 'system'})
    # Handle both directory and file inputs
    if project_path.is_file():
        if project_path.name == "project.json":
            project_dir = project_path.parent
            project_file = project_path
        else:
            return {
                "status": "error",
                "message": f"{project_path} is not a project.json file",
                "details": {"provided_path": str(project_path)},
            }
    elif project_path.is_dir():
        project_file = project_path / "project.json"
        if not project_file.exists():
            return {
                "status": "error",
                "message": f"No project.json found in {project_path}",
                "details": {"project_path": str(project_path)},
            }
        project_dir = project_path
    else:
        return {
            "status": "error",
            "message": f"Project path {project_path} does not exist",
            "details": {"project_path": str(project_path)},
        }

    try:
        with open(project_file) as f:
            project_data = json.load(f)

        # Validate project structure
        required_keys = ["name", "architecture", "config"]
        missing_keys = [key for key in required_keys if key not in project_data]

        if missing_keys:
            return {
                "status": "error",
                "message": f"Invalid project file. Missing keys: {missing_keys}",
                "details": {
                    "missing_keys": missing_keys,
                    "project_file": str(project_file),
                },
            }

        # Load project data
        project_name = project_data["name"]
        description = project_data.get("description", "No description")
        author = project_data.get("author", "Unknown")

        return {
            "status": "success",
            "message": "Project loaded successfully",
            "details": {
                "project_name": project_name,
                "project_dir": str(project_dir),
                "project_file": str(project_file),
                "project_data": project_data,
                "set_as_active": set_active,
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading project: {e}",
            "details": {"error": str(e)},
        }


@arch_mcp.tool()
def project_info(
    project_path: Path,
) -> dict[str, Any]:
    """Show detailed information about a project.

    This function provides comprehensive information about a neural network project
    including statistics about configurations, architectures, and outputs.

    Args:
        project_path: Path to project directory or project.json file.

    Returns:
        dict: A dictionary containing detailed project information.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with project information including:
                - project_name: Name of the project
                - project_dir: Directory containing the project
                - project_file: Path to project.json file
                - project_data: Complete project configuration
                - statistics: Dictionary with counts of configs, architectures, and outputs

    """
    # Handle both directory and file inputs
    if project_path.is_file():
        if project_path.name == "project.json":
            project_file = project_path
        else:
            return {
                "status": "error",
                "message": f"{project_path} is not a project.json file",
                "details": {"provided_path": str(project_path)},
            }
    elif project_path.is_dir():
        project_file = project_path / "project.json"
        if not project_file.exists():
            return {
                "status": "error",
                "message": f"No project.json found in {project_path}",
                "details": {"project_path": str(project_path)},
            }
    else:
        return {
            "status": "error",
            "message": f"Project path {project_path} does not exist",
            "details": {"project_path": str(project_path)},
        }

    try:
        with open(project_file) as f:
            project_data = json.load(f)

        # Get project statistics
        project_dir = project_file.parent
        num_configs = (
            len(list((project_dir / "configs").glob("*.json")))
            if (project_dir / "configs").exists()
            else 0
        )
        num_architectures = (
            len(list((project_dir / "architectures").glob("*.json")))
            if (project_dir / "architectures").exists()
            else 0
        )
        num_outputs = (
            len(list((project_dir / "outputs").iterdir()))
            if (project_dir / "outputs").exists()
            else 0
        )

        statistics = {
            "configs_count": num_configs,
            "architectures_count": num_architectures,
            "outputs_count": num_outputs,
        }

        return {
            "status": "success",
            "message": "Project information retrieved successfully",
            "details": {
                "project_name": project_data.get("name", "Unknown"),
                "project_dir": str(project_dir),
                "project_file": str(project_file),
                "project_data": project_data,
                "statistics": statistics,
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading project info: {e}",
            "details": {"error": str(e)},
        }

# ============= Code Generation Server Tools =============





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
    logger.info(f"Generating PyTorch code: {model_name} from {architecture_file}", extra={'session_id': ctx.session_id})
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {
            "status": "error",
            "message": f"No architecture manager found for session {ctx.session_id}",
            "details": {"session_id": ctx.session_id},
        }

    if not architecture_file.exists():
        return {
            "status": "error",
            "message": f"Architecture file {architecture_file} not found",
            "details": {"file_path": str(architecture_file)},
        }

    # Load architecture
    arch_manager.load_architecture(architecture_file)



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
                    "generated_files": [str(f) for f in sorted(generated_files)[:10]],
                    "model_name": model_name
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

        cookiecutter.render_cookiecutter_template_cli(
            arch_manager.current_architecture,
            jinja_path,
            cookie_json_path,
            template_dir,
            str(output_dir),
        )

        # Cookiecutter creates a subfolder with the project name inside output_dir
        actual_output_path = output_dir / "Pretrained_Output"
        main_py_path = actual_output_path / "main.py"
        
        if not main_py_path.exists():
            return {
                "status": "error",
                "message": f"Generated main.py not found at: {main_py_path}",
                "details": {"expected_path": str(main_py_path)},
            }

        return {
            "status": "success",
            "message": "Pretrained code generated successfully",
            "details": {"output_path": str(actual_output_path), "main_py_path": str(main_py_path)},
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


        if isinstance(arch_manager.current_architecture.get("layers"), list):
            arch_manager.current_architecture["layers"] = {
                "list": arch_manager.current_architecture["layers"]
            }
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


@arch_mcp.tool()
async def get_available_loss_functions(ctx: Context) -> dict:
    """Get all available loss functions from the architecture manager.

    Args:
        ctx: The FastMCP context containing session state.

    Returns:
        dict: A dictionary containing status information and available loss functions.
            - status: "success"
            - loss_functions: List of available loss function names
    """
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {"status": "error", "message": "No active session. Call initialize_session first."}
    
    return {
        "status": "success",
        "loss_functions": list(arch_manager.loss_funcs.keys()) if hasattr(arch_manager, 'loss_funcs') else []
    }


@arch_mcp.tool()
async def get_available_optimizers(ctx: Context) -> dict:
    """Get all available optimizers from the architecture manager.

    Args:
        ctx: The FastMCP context containing session state.

    Returns:
        dict: A dictionary containing status information and available optimizers.
            - status: "success"
            - optimizers: List of available optimizer names
    """
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {"status": "error", "message": "No active session. Call initialize_session first."}
    
    return {
        "status": "success",
        "optimizers": list(arch_manager.optimizers.keys()) if hasattr(arch_manager, 'optimizers') else []
    }


@arch_mcp.tool()
async def get_available_schedulers(ctx: Context) -> dict:
    """Get all available learning rate schedulers from the architecture manager.

    Args:
        ctx: The FastMCP context containing session state.

    Returns:
        dict: A dictionary containing status information and available schedulers.
            - status: "success"
            - schedulers: List of available scheduler names
            - complex_schedulers: List of available complex scheduler names
    """
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {"status": "error", "message": "No active session. Call initialize_session first."}
    
    return {
        "status": "success",
        "schedulers": list(arch_manager.schedulers.keys()) if hasattr(arch_manager, 'schedulers') else [],
        "complex_schedulers": arch_manager.complex_schedulers if hasattr(arch_manager, 'complex_schedulers') else []
    }


@arch_mcp.tool()
async def get_available_pretrained_models(ctx: Context) -> dict:
    """Get all available pretrained models from the architecture manager.

    Args:
        ctx: The FastMCP context containing session state.

    Returns:
        dict: A dictionary containing status information and available pretrained models.
            - status: "success"
            - pretrained_models: List of available pretrained model names
    """
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {"status": "error", "message": "No active session. Call initialize_session first."}
    
    return {
        "status": "success",
        "pretrained_models": arch_manager.pretrained_models if hasattr(arch_manager, 'pretrained_models') else []
    }


@arch_mcp.tool()
async def get_available_datasets(ctx: Context) -> dict:
    """Get all available datasets from the architecture manager.

    Args:
        ctx: The FastMCP context containing session state.

    Returns:
        dict: A dictionary containing status information and available datasets.
            - status: "success"
            - datasets: List of available dataset names
    """
    arch_manager = arch_manager_store.get(ctx.session_id, None)
    if not arch_manager:
        return {"status": "error", "message": "No active session. Call initialize_session first."}
    
    return {
        "status": "success",
        "datasets": arch_manager.datasets if hasattr(arch_manager, 'datasets') else []
    }
    
# ============= Train Server Tools =============



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


# ----------------- Image / Model Helpers -----------------
def is_grayscale_image(img: np.ndarray, tol: int = 2) -> bool:
    """Return True if image is grayscale or effectively grayscale (channels nearly identical).

    Args:
        img: Image as a numpy array in OpenCV BGR ordering or 2D grayscale.
        tol: Maximum allowed difference between color channels to consider image grayscale.
    """
    if img is None:
        return False
    if img.ndim == 2:
        return True
    if img.ndim == 3 and img.shape[2] == 1:
        return True
    if img.ndim == 3 and img.shape[2] >= 3:
        # Compare channels with a small tolerance
        r = img[..., 0].astype(np.int16)
        g = img[..., 1].astype(np.int16)
        b = img[..., 2].astype(np.int16)
        if (int(np.max(np.abs(r - g))) <= tol
                and int(np.max(np.abs(r - b))) <= tol
                and int(np.max(np.abs(g - b))) <= tol):
            return True
    return False


def get_model_in_channels(model_or_state: Any) -> int:
    """Attempt to infer model input channels from a model instance or state_dict.

    Strategy:
    - If `model_or_state` is a nn.Module, find first Conv2d and return `in_channels`.
    - If it's a state_dict (dict of tensors), find first weight tensor with >=4 dims and use its shape[1].
    - For TorchScript models, inspect state_dict from graph.
    - Fall back to 3.
    """
    import torch.nn as nn

    # If we have a module instance (not a dict), inspect its modules for Conv2d
    try:
        if not isinstance(model_or_state, dict) and hasattr(model_or_state, "modules"):
            for m in model_or_state.modules():
                if isinstance(m, nn.Conv2d):
                    return int(m.in_channels)
    except Exception:
        pass

    # TorchScript model: try to get state_dict and inspect weights
    try:
        if hasattr(model_or_state, "state_dict") and callable(getattr(model_or_state, "state_dict")):
            state = model_or_state.state_dict()
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 4:
                    # Conv weight shape: (out_channels, in_channels, h, w)
                    return int(v.shape[1])
    except Exception:
        pass

    # If this is a dict, it may be a state_dict or a wrapper containing a state dict
    try:
        if isinstance(model_or_state, dict):
            # Common wrapper keys that may hold the real state or model
            for wrapper_key in ("model", "state_dict", "model_state_dict", "module", "net"):
                nested = model_or_state.get(wrapper_key)
                if isinstance(nested, dict):
                    model_or_state = nested
                    break
                if nested is not None and not isinstance(nested, dict) and hasattr(nested, "modules"):
                    # Nested module instance
                    for m in nested.modules():
                        if isinstance(m, nn.Conv2d):
                            return int(m.in_channels)
    except Exception:
        pass

    # State dict inspection: find first weight tensor with shape (out, in, h, w)
    try:
        if isinstance(model_or_state, dict):
            for k, v in model_or_state.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 4:
                    return int(v.shape[1])
    except Exception:
        pass

    # Last resort: check conv1 attribute only when it's not a dict
    try:
        if not isinstance(model_or_state, dict) and hasattr(model_or_state, "conv1") and hasattr(model_or_state.conv1, "in_channels"):
            return int(model_or_state.conv1.in_channels)
    except Exception:
        pass

    return 3


def adapt_image_channels(img: np.ndarray, expected_channels: int) -> np.ndarray:
    """Adapt `img` to have `expected_channels` channels.

    - If expected 1: collapse identical channels or convert to grayscale.
    - If expected 3: ensure 3-channel BGR output (suitable for cv2.cvtColor -> RGB later).
    """
    if img is None:
        return img

    # Already single-channel 2D
    if expected_channels == 1:
        if img.ndim == 2:
            return img
        if img.ndim == 3 and img.shape[2] == 1:
            return img.squeeze(2)
        # If channels look identical, collapse
        if is_grayscale_image(img):
            return img[..., 0]
        # Convert color to grayscale
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Expected 3 channels
    if expected_channels == 3:
        if img.ndim == 3 and img.shape[2] >= 3:
            return img[..., :3]
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 1:
            return cv2.cvtColor(img.squeeze(2), cv2.COLOR_GRAY2BGR)

    # Fallback: replicate or trim channels to match expected
    if img.ndim == 2:
        return np.stack([img] * expected_channels, axis=-1)
    if img.ndim == 3 and img.shape[2] < expected_channels:
        reps = expected_channels // img.shape[2]
        out = np.concatenate([img] * reps, axis=2)
        if out.shape[2] < expected_channels:
            # pad by repeating first channel
            pad_c = expected_channels - out.shape[2]
            out = np.concatenate([out, img[..., :1].repeat(pad_c, axis=2)], axis=2)
        return out[..., :expected_channels]

    return img

# ---------------------------------------------------------


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

    # project_output can be either:
    # 1. The project root directory (e.g., my_first_project)
    # 2. The Manual_Output directory itself (e.g., my_first_project/Manual_Output)
    # Handle both cases
    project_path = Path(project_output)
    
    # Check if project_output already ends with Manual_Output
    if project_path.name == "Manual_Output" and (project_path / "main-cli.py").exists():
        training_script_cli = project_path / "main-cli.py"
    else:
        # Assume project_output is the project root
        training_script_cli = project_path / "Manual_Output" / "main-cli.py"
    
    if not training_script_cli.exists():
        return {
            "status": "error",
            "message": f"CLI training script not found at: {training_script_cli}",
            "details": {
                "expected_path": str(training_script_cli),
                "project_output": str(project_output),
                "note": "Checked both project_root/Manual_Output and direct Manual_Output paths"
            },
        }

    # Use provided log_dir (default: data/tensorboardlogs)
    log_dir_str = str(log_dir)
    # Determine expected saved model path (created by cookiecutter templates)
    expected_model_path = training_script_cli.parent / "SystemC" / "Pt" / "model.pt"

    # Run training as a subprocess using the CLI script
    try:
        import subprocess
        
        # Build command to run the CLI training script
        command = [
            sys.executable,  # Use current Python interpreter
            str(training_script_cli),
            "--logdir", log_dir_str,
            "--no-wrap",  # Don't perform SystemC wrapping from CLI
        ]
        
        if verbose:
            logger.info(f"Running training command: {' '.join(command)}")
        
        # Run the subprocess without redirecting stdout/stderr
        # This allows Rich progress bars to display properly in the terminal
        process = subprocess.Popen(
            command,
            cwd=str(training_script_cli.parent)
            # No stdout/stderr redirection - output goes directly to terminal
        )
        
        # Wait for process to complete
        return_code = process.wait()

        # After training attempt, check for model existence
        model_exists = expected_model_path.exists()
        
        if return_code == 0:
            return {
                "status": "success",
                "message": "Manual training completed successfully!",
                "details": {
                    "output_dir": str(output_dir),
                    "log_dir": log_dir_str,
                    "training_script": str(training_script_cli),
                    "project_output": str(project_output),
                    "saved_model_path": str(expected_model_path),
                    "saved_model_exists": model_exists,
                    "note": "Training output displayed in terminal with Rich progress bars"
                },
            }
        else:
            return {
                "status": "error",
                "message": f"Training failed with exit code: {return_code}",
                "details": {
                    "exit_code": return_code,
                    "training_script": str(training_script_cli),
                    "saved_model_path": str(expected_model_path),
                    "saved_model_exists": model_exists,
                },
            }
    except FileNotFoundError as e:
        return {
            "status": "error",
            "message": f"Python interpreter or script not found: {e}",
            "details": {"error": str(e), "training_script": str(training_script_cli)},
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Training failed: {e}",
            "details": {"error": str(e), "training_script": str(training_script_cli)},
        }


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

    # project_output can be either:
    # 1. The project root directory (e.g., my_first_project)
    # 2. The Pretrained_Output directory itself (e.g., my_first_project/Pretrained_Output)
    # Handle both cases
    project_path = Path(project_output)
    
    # Check if project_output already ends with Pretrained_Output
    if project_path.name == "Pretrained_Output" and (project_path / "main-cli.py").exists():
        training_script_cli = project_path / "main-cli.py"
    else:
        # Assume project_output is the project root
        training_script_cli = project_path / "Pretrained_Output" / "main-cli.py"

    if not training_script_cli.exists():
        return {
            "status": "error",
            "message": f"CLI training script not found at: {training_script_cli}",
            "details": {
                "expected_path": str(training_script_cli),
                "project_output": str(project_output),
                "note": "Checked both project_root/Pretrained_Output and direct Pretrained_Output paths"
            },
        }

    # Use provided log_dir (default: data/tensorboardlogs)
    log_dir_str = str(log_dir)
    # Determine expected saved model path
    expected_model_path = training_script_cli.parent / "SystemC" / "Pt" / "model.pt"

    # Run training as a subprocess using the CLI script
    try:
        import subprocess
        
        # Build command to run the CLI training script
        command = [
            sys.executable,  # Use current Python interpreter
            str(training_script_cli),
            "--logdir", log_dir_str,
            "--no-wrap",  # Don't perform SystemC wrapping from CLI
        ]
        
        if verbose:
            logger.info(f"Running training command: {' '.join(command)}")
        
        # Run the subprocess without redirecting stdout/stderr
        # This allows Rich progress bars to display properly in the terminal
        process = subprocess.Popen(
            command,
            cwd=str(training_script_cli.parent)
            # No stdout/stderr redirection - output goes directly to terminal
        )
        
        # Wait for process to complete
        return_code = process.wait()

        # After training attempt, check for model existence
        model_exists = expected_model_path.exists()
        
        if return_code == 0:
            return {
                "status": "success",
                "message": "Pretrained training completed successfully!",
                "details": {
                    "output_dir": str(output_dir),
                    "log_dir": log_dir_str,
                    "training_script": str(training_script_cli),
                    "project_output": str(project_output),
                    "saved_model_path": str(expected_model_path),
                    "saved_model_exists": model_exists,
                    "note": "Training output displayed in terminal with Rich progress bars"
                },
            }
        else:
            return {
                "status": "error",
                "message": f"Training failed with exit code: {return_code}",
                "details": {
                    "exit_code": return_code,
                    "training_script": str(training_script_cli),
                    "saved_model_path": str(expected_model_path),
                    "saved_model_exists": model_exists,
                },
            }
    except FileNotFoundError as e:
        return {
            "status": "error",
            "message": f"Python interpreter or script not found: {e}",
            "details": {"error": str(e), "training_script": str(training_script_cli)},
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Training failed: {e}",
            "details": {"error": str(e), "training_script": str(training_script_cli)},
        }


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

    # project_output can be either:
    # 1. The project root directory (e.g., my_first_project)
    # 2. The Pretrained_Output directory itself (e.g., my_first_project/Pretrained_Output)
    # YOLOX uses the Pretrained_Output directory, handle both cases
    project_path = Path(project_output)
    
    # Check if project_output already ends with Pretrained_Output
    if project_path.name == "YOLOX" and (project_path / "main-cli.py").exists():
        training_script_cli = project_path / "main-cli.py"
    else:
        # Assume project_output is the project root
        training_script_cli = project_path / "YOLOX" / "main-cli.py"

    if not training_script_cli.exists():
        return {
            "status": "error",
            "message": f"CLI training script not found at: {training_script_cli}",
            "details": {
                "expected_path": str(training_script_cli),
                "project_output": str(project_output),
                "note": "Checked both project_root/Pretrained_Output and direct Pretrained_Output paths"
            },
        }

    # Use provided log_dir (default: data/tensorboardlogs)
    log_dir_str = str(log_dir)
    
    # YOLOX saves model to YOLOX_outputs/yolox_base/best_ckpt.pth (relative to training script dir)
    expected_model_dir = training_script_cli.parent / "YOLOX_outputs" / "yolox_base"
    expected_best_model = expected_model_dir / "best_ckpt.pth"
    expected_latest_model = expected_model_dir / "latest_ckpt.pth"

    # Run training as a subprocess using the CLI script
    try:
        import subprocess
        
        # Build command to run the CLI training script
        command = [
            sys.executable,  # Use current Python interpreter
            str(training_script_cli),
            "--logdir", log_dir_str,
            "--no-wrap",  # Don't perform SystemC wrapping from CLI
        ]
        
        if verbose:
            logger.info(f"Running training command: {' '.join(command)}")
        
        # Run the subprocess without redirecting stdout/stderr
        # This allows Rich progress bars to display properly in the terminal
        process = subprocess.Popen(
            command,
            cwd=str(training_script_cli.parent)
            # No stdout/stderr redirection - output goes directly to terminal
        )
        
        # Wait for process to complete
        return_code = process.wait()
        
        # After training, check for model files
        best_model_exists = expected_best_model.exists()
        latest_model_exists = expected_latest_model.exists()
        
        if return_code == 0:
            return {
                "status": "success",
                "message": "YOLOX training completed successfully!",
                "details": {
                    "output_dir": str(output_dir),
                    "log_dir": log_dir_str,
                    "training_script": str(training_script_cli),
                    "project_output": str(project_output),
                    "model_output_dir": str(expected_model_dir),
                    "best_model_path": str(expected_best_model),
                    "best_model_exists": best_model_exists,
                    "latest_model_path": str(expected_latest_model),
                    "latest_model_exists": latest_model_exists,
                    "note": "Training output displayed in terminal with Rich progress bars"
                },
            }
        else:
            return {
                "status": "error",
                "message": f"Training failed with exit code: {return_code}",
                "details": {
                    "exit_code": return_code,
                    "training_script": str(training_script_cli),
                    "model_output_dir": str(expected_model_dir),
                    "best_model_path": str(expected_best_model),
                    "best_model_exists": best_model_exists,
                    "latest_model_path": str(expected_latest_model),
                    "latest_model_exists": latest_model_exists,
                },
            }
    except FileNotFoundError as e:
        return {
            "status": "error",
            "message": f"Python interpreter or script not found: {e}",
            "details": {"error": str(e), "training_script": str(training_script_cli)},
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Training failed: {e}",
            "details": {"error": str(e), "training_script": str(training_script_cli)},
        }


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


# ============= Test Server Tools =============

@arch_mcp.tool()
async def test_manual_image(
    ctx: Context,
    image_path: Path,
    model_path: Path,
    input_height: int = 224,
    input_width: int = 224,
    output_dir: Path = Path("./test_results"),
) -> Dict[str, Any]:
    """Test a manual/custom model with a single image for classification.

    This function performs inference on a single image using a trained manual model,
    returning the predicted class label and confidence score.

    Args:
        ctx: FastMCP context containing session state.
        image_path: Path to image file to test.
        model_path: Path to trained model .pt file.
        input_height: Expected input height (default: 224).
        input_width: Expected input width (default: 224).
        output_dir: Output directory for test results (default: ./test_results).

    Returns:
        dict: A dictionary containing operation status and test details.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with test information including:
                - predicted_class: Index of predicted class
                - confidence: Confidence score
                - probabilities: List of all class probabilities

    """
    logger.info(
        f"Testing manual model with image: {image_path}",
        extra={'session_id': ctx.session_id}
    )

    try:
        # Validate paths
        if not image_path.exists():
            return {
                "status": "error",
                "message": f"Image file not found: {image_path}",
                "details": {"image_path": str(image_path)},
            }

        if not model_path.exists():
            return {
                "status": "error",
                "message": f"Model file not found: {model_path}",
                "details": {"model_path": str(model_path)},
            }

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}", extra={'session_id': ctx.session_id})

        # Load model
        logger.info("Loading model...", extra={'session_id': ctx.session_id})
        model = torch.load(model_path, map_location=device)
        model.eval()

        # Load and preprocess image
        logger.info(f"Loading image: {image_path}", extra={'session_id': ctx.session_id})
        img = cv2.imread(str(image_path))
        if img is None:
            return {
                "status": "error",
                "message": f"Failed to load image: {image_path}",
                "details": {"image_path": str(image_path)},
            }
        # Preprocess: adapt channels based on model expectation, resize, normalize, convert to tensor
        try:
            expected_channels = get_model_in_channels(model)
        except Exception:
            expected_channels = 3
        logger.info(f"Model expects {expected_channels} input channels", extra={'session_id': ctx.session_id})

        # Adapt image to expected channels (returns 2D gray or 3-channel BGR)
        img = adapt_image_channels(img, expected_channels)

        # Build tensor depending on whether image is grayscale or color
        if is_grayscale_image(img):
            img_resized = cv2.resize(img, (input_width, input_height))
            img_tensor = torch.from_numpy(img_resized).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
            logger.info(f"Processing grayscale image with shape: {img_tensor.shape}", extra={'session_id': ctx.session_id})
        else:
            # img is BGR 3-channel; convert to RGB then resize
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (input_width, input_height))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            logger.info(f"Processing color image with shape: {img_tensor.shape}", extra={'session_id': ctx.session_id})

        # Run inference
        logger.info("Running inference...", extra={'session_id': ctx.session_id})
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        predicted_class = predicted_class.item()
        confidence = confidence.item()
        all_probs = probabilities[0].cpu().numpy().tolist()

        logger.info(
            f"Prediction: class {predicted_class} with confidence {confidence:.4f}",
            extra={'session_id': ctx.session_id}
        )

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        return {
            "status": "success",
            "message": "Manual model inference completed successfully!",
            "details": {
                "image_path": str(image_path),
                "model_path": str(model_path),
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": all_probs,
                "device": str(device),
                "output_dir": str(output_dir),
            },
        }

    except Exception as e:
        logger.error(
            f"Error during manual model inference: {e}",
            extra={'session_id': ctx.session_id},
            exc_info=True
        )
        return {
            "status": "error",
            "message": f"Inference failed: {e}",
            "details": {
                "error": str(e),
                "image_path": str(image_path),
                "model_path": str(model_path),
            },
        }


@arch_mcp.tool()
async def test_pretrained_image(
    ctx: Context,
    image_path: Path,
    model_path: Path,
    input_height: int = 224,
    input_width: int = 224,
    output_dir: Path = Path("./test_results"),
) -> Dict[str, Any]:
    """Test a pretrained/transfer learning model with a single image for classification.

    This function performs inference on a single image using a trained pretrained model,
    returning the predicted class label and confidence score.

    Args:
        ctx: FastMCP context containing session state.
        image_path: Path to image file to test.
        model_path: Path to trained model .pt file.
        input_height: Expected input height (default: 224).
        input_width: Expected input width (default: 224).
        output_dir: Output directory for test results (default: ./test_results).

    Returns:
        dict: A dictionary containing operation status and test details.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with test information including:
                - predicted_class: Index of predicted class
                - confidence: Confidence score
                - probabilities: List of all class probabilities

    """
    logger.info(
        f"Testing pretrained model with image: {image_path}",
        extra={'session_id': ctx.session_id}
    )

    try:
        # Validate paths
        if not image_path.exists():
            return {
                "status": "error",
                "message": f"Image file not found: {image_path}",
                "details": {"image_path": str(image_path)},
            }

        if not model_path.exists():
            return {
                "status": "error",
                "message": f"Model file not found: {model_path}",
                "details": {"model_path": str(model_path)},
            }

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}", extra={'session_id': ctx.session_id})

        # Load model
        logger.info("Loading model...", extra={'session_id': ctx.session_id})
        model = torch.load(model_path, map_location=device)
        model.eval()

        # Load and preprocess image
        logger.info(f"Loading image: {image_path}", extra={'session_id': ctx.session_id})
        img = cv2.imread(str(image_path))
        if img is None:
            return {
                "status": "error",
                "message": f"Failed to load image: {image_path}",
                "details": {"image_path": str(image_path)},
            }
        # Preprocess: adapt channels based on model expectation, resize, normalize, convert to tensor
        try:
            expected_channels = get_model_in_channels(model)
        except Exception:
            expected_channels = 3
        logger.info(f"Model expects {expected_channels} input channels", extra={'session_id': ctx.session_id})

        # Adapt image to expected channels (returns 2D gray or 3-channel BGR)
        img = adapt_image_channels(img, expected_channels)

        # Build tensor depending on whether image is grayscale or color
        if is_grayscale_image(img):
            img_resized = cv2.resize(img, (input_width, input_height))
            img_tensor = torch.from_numpy(img_resized).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
            logger.info(f"Processing grayscale image with shape: {img_tensor.shape}", extra={'session_id': ctx.session_id})
        else:
            # img is BGR 3-channel; convert to RGB then resize
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (input_width, input_height))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            logger.info(f"Processing color image with shape: {img_tensor.shape}", extra={'session_id': ctx.session_id})

        # Run inference
        logger.info("Running inference...", extra={'session_id': ctx.session_id})
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        predicted_class = predicted_class.item()
        confidence = confidence.item()
        all_probs = probabilities[0].cpu().numpy().tolist()

        logger.info(
            f"Prediction: class {predicted_class} with confidence {confidence:.4f}",
            extra={'session_id': ctx.session_id}
        )

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        return {
            "status": "success",
            "message": "Pretrained model inference completed successfully!",
            "details": {
                "image_path": str(image_path),
                "model_path": str(model_path),
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": all_probs,
                "output_dir": str(output_dir),
            },
        }

    except Exception as e:
        logger.error(
            f"Error during pretrained model inference: {e}",
            extra={'session_id': ctx.session_id},
            exc_info=True
        )
        return {
            "status": "error",
            "message": f"Inference failed: {e}",
            "details": {
                "error": str(e),
                "image_path": str(image_path),
                "model_path": str(model_path),
            },
        }


@arch_mcp.tool()
async def test_image(
    ctx: Context,
    image_path: Path,
    model_path: Path,
    hgd_ckpt_path: Path,
    class_names_path: Path,
    output_dir: Path = Path("./test_results"),
) -> Dict[str, Any]:
    """Test a YOLOX model with a single image for object detection.

    This function tests a trained YOLOX model with HGD denoising on a single image.
    It performs object detection on both the original (perturbed) image and the denoised image,
    then saves the visualization results to the output directory.

    Args:
        ctx: FastMCP context containing session state.
        image_path: Path to image file to test.
        model_path: Path to model weights .pth file.
        hgd_ckpt_path: Path to HGD checkpoint .pt file.
        class_names_path: Path to file containing class names (one per line).
        output_dir: Output directory for test results (default: ./test_results).

    Returns:
        dict: A dictionary containing operation status and test details.
            - status: "success" or "error"
            - message: Human-readable status message
            - details: Dictionary with test information including:
                - image_path: Path to the tested image
                - output_dir: Where results were saved
                - num_classes: Number of classes in the model
                - device: Device used for inference

    """
    logger.info(
        f"Testing image: {image_path} with model: {model_path}",
        extra={'session_id': ctx.session_id}
    )

    try:
        # Validate paths
        if not image_path.exists():
            return {
                "status": "error",
                "message": f"Image file not found: {image_path}",
                "details": {"image_path": str(image_path)},
            }

        if not model_path.exists():
            return {
                "status": "error",
                "message": f"Model weights file not found: {model_path}",
                "details": {"model_path": str(model_path)},
            }

        if not hgd_ckpt_path.exists():
            return {
                "status": "error",
                "message": f"HGD checkpoint file not found: {hgd_ckpt_path}",
                "details": {"hgd_ckpt_path": str(hgd_ckpt_path)},
            }

        if not class_names_path.exists():
            return {
                "status": "error",
                "message": f"Class names file not found: {class_names_path}",
                "details": {"class_names_path": str(class_names_path)},
            }

        # Parse class names
        try:
            class_names_list = validate_and_parse_class_names(class_names_path)
        except ValueError as e:
            return {
                "status": "error",
                "message": f"Error parsing class names: {e}",
                "details": {"class_names_path": str(class_names_path), "error": str(e)},
            }

        num_classes = len(class_names_list)
        logger.info(
            f"Loaded {num_classes} classes from {class_names_path}",
            extra={'session_id': ctx.session_id}
        )

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}", extra={'session_id': ctx.session_id})

        # Load YOLOX model
        logger.info("Loading YOLOX model...", extra={'session_id': ctx.session_id})
        exp = get_exp(None, "yolox-s")
        exp.num_classes = num_classes
        model = exp.get_model()
        model.eval()

        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to(device)
        logger.info("YOLOX model loaded successfully!", extra={'session_id': ctx.session_id})

        # Load image
        logger.info(f"Loading image: {image_path}", extra={'session_id': ctx.session_id})
        img = cv2.imread(str(image_path))
        if img is None:
            return {
                "status": "error",
                "message": f"Failed to load image: {image_path}",
                "details": {"image_path": str(image_path)},
            }

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Detect on perturbed image
        logger.info("Running detection on perturbed image...", extra={'session_id': ctx.session_id})
        detect_and_visualize(
            model,
            img,
            class_names_list,
            "Perturbed Image",
            num_classes,
            device,
            output_dir,
        )

        # Load HGD model
        logger.info("Loading HGD model...", extra={'session_id': ctx.session_id})
        hgd_model = get_HGD_model(device, hgd_ckpt_path)
        hgd_model.eval()

        # Preprocess for HGD
        img_tensor = (
            torch.from_numpy(img.astype(np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
            .float()
        )

        # Denoise image
        logger.info("Denoising image with HGD...", extra={'session_id': ctx.session_id})
        with torch.no_grad():
            noise = hgd_model(img_tensor)
            denoised_tensor = img_tensor - noise
            denoised_tensor = torch.clamp(denoised_tensor, 0.0, 255.0)

        denoised_tensor = (
            denoised_tensor.squeeze(0).cpu().byte().permute(1, 2, 0).numpy()
        )

        # Detect on denoised image
        logger.info("Running detection on denoised image...", extra={'session_id': ctx.session_id})
        detect_and_visualize(
            model,
            denoised_tensor,
            class_names_list,
            "Denoised Image",
            num_classes,
            device,
            output_dir,
        )

        # Display images and wait for user to close them (like CLI version)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        logger.info("Test completed successfully!", extra={'session_id': ctx.session_id})

        return {
            "status": "success",
            "message": "Image test completed successfully!",
            "details": {
                "image_path": str(image_path),
                "output_dir": str(output_dir),
                "num_classes": num_classes,
                "class_names": class_names_list,
                "device": str(device),
                "model_path": str(model_path),
                "hgd_ckpt_path": str(hgd_ckpt_path),
            },
        }

    except Exception as e:
        logger.error(
            f"Error during image test: {e}",
            extra={'session_id': ctx.session_id},
            exc_info=True
        )
        return {
            "status": "error",
            "message": f"Test failed: {e}",
            "details": {
                "error": str(e),
                "image_path": str(image_path),
                "model_path": str(model_path),
            },
        }


if __name__ == "__main__":
    arch_mcp.run("http")
