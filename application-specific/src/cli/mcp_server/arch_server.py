from pathlib import Path

from fastmcp import Context

from src.cli.utils import Analyze_for_MCP

from .main import arch_mcp

# ========== Tools ==========


@arch_mcp.tool()
async def create_architecture(
    ctx: Context,
    name: str,
    output_dir: Path,
    base_model: str | None = None,
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
    arch_manager = ctx.get_state("arch_manager")

    arch_file = output_dir / f"{name}_architecture.json"

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
        "pretrained": {"value": base_model, "index": pretrained_index},
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
    arch_manager = ctx.get_state("arch_manager")

    arch_file = output_dir / f"{name}_complex_architecture.json"

    if arch_file.exists():
        return {
            "status": "error",
            "message": f"Complex architecture '{name}' already exists",
            "details": {"file_path": str(arch_file)},
        }

    # Check if model is available
    if model not in arch_manager.complex_arch_models:
        return {
            "status": "error",
            "message": f"Complex model '{model}' not available",
            "details": {
                "model": model,
                "available_models": arch_manager.complex_arch_models[:5],
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
    arch_manager = ctx.get_state("arch_manager")

    # Save complex architecture
    arch_manager.save_complex_architecture(file_path)

    return {
        "status": "success",
        "message": "Complex architecture saved successfully!",
        "details": {
            "file_path": str(file_path),
            "layers_count": len(arch_manager.current_architecture["layers"]),
            "model": arch_manager.current_architecture["pretrained"].get(
                "value", "None",
            ),
        },
    }


@arch_mcp.tool()
async def add_layer(
    ctx: Context,
    layer_type: str,
    params: list[str] = [],
    position: int | None = None,
):
    """Add a layer to the current architecture.

    This function adds a new layer to the current neural network architecture. It validates that the layer type
    is available, processes the layer parameters with proper type conversion, and adds the layer to the architecture.
    The function displays detailed information about the added layer including its type, position, and parameters.

    Args:
        ctx: The FastMCP context containing session state.
        layer_type: The type of layer to add (e.g., 'Conv2d', 'Linear', 'ReLU').
        params: List of parameter strings in key=value format for layer configuration.
        position: Optional position to insert the layer (default: end of architecture).

    Returns:
        dict: A dictionary containing status information and layer details.

    """
    arch_manager = ctx.get_state("arch_manager")

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

    if layer_info is None:
        return {
            "status": "error",
            "message": f"Could not get layer information for '{layer_type}'",
            "details": {"layer_type": layer_type},
        }

    # Get valid parameter names for this layer
    valid_param_names = {param["name"] for param in layer_info}

    for param in params:
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

            # Check parameter type matches expected type
            expected_type = None
            for param_info in layer_info:
                if param_info["name"] == key:
                    expected_type = param_info.get("type")
                    break

            if expected_type:
                try:
                    # Try to convert to expected type
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
                        converted_value = value.lower() == "true"
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
                        converted_value = int(value)
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
                        converted_value = float(value)
                    # For other types (like str), try to eval if it looks like a list/tuple
                    elif value.startswith("[") and value.endswith("]"):
                        converted_value = eval(value)
                    else:
                        converted_value = value
                    layer_params[key] = converted_value
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error converting parameter '{key}' value '{value}': {e}",
                        "details": {"parameter": key, "value": value, "error": str(e)},
                    }
            else:
                # Fallback to original type conversion if no type info
                try:
                    # Try boolean
                    if value.lower() in ["true", "false"]:
                        layer_params[key] = value.lower() == "true"
                    # Try int
                    elif value.isdigit():
                        layer_params[key] = int(value)
                    # Try float
                    elif value.replace(".", "", 1).isdigit():
                        layer_params[key] = float(value)
                    # Try list (for tuples like [0.9, 0.999])
                    elif value.startswith("[") and value.endswith("]"):
                        layer_params[key] = eval(value)
                    else:
                        layer_params[key] = value
                except:
                    layer_params[key] = value
        else:
            return {
                "status": "error",
                "message": f"Invalid parameter format: {param}. Use key=value",
                "details": {"parameter": param},
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
    arch_manager = ctx.get_state("arch_manager")

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
    arch_manager = ctx.get_state("arch_manager")

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

    removed_layer = arch_manager.remove_layer(index)

    return {
        "status": "success",
        "message": "Layer removed successfully!",
        "details": {
            "layer_type": removed_layer["type"],
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
    arch_manager = ctx.get_state("arch_manager")

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

    removed_block = arch_manager.remove_layer(index)

    return {
        "status": "success",
        "message": "Residual block removed successfully!",
        "details": {
            "residual_block_type": removed_block["type"],
            "index": index,
            "internal_layers_count": len(removed_block.get("layers", [])),
            "internal_layer_types": [
                l["type"] for l in removed_block.get("layers", [])
            ],
        },
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
    arch_manager = ctx.get_state("arch_manager")

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
    arch_manager = ctx.get_state("arch_manager")

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
    arch_manager = ctx.get_state("arch_manager")

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
    arch_manager = ctx.get_state("arch_manager")

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
    if file_path:
        if not file_path.exists():
            return {
                "status": "error",
                "message": f"Architecture file {file_path} not found",
                "details": {"file_path": str(file_path)},
            }

    arch_manager = ctx.get_state("arch_manager")
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
    # Validate and prepare architecture using validation logic
    try:
        # Get current architecture for validation
        arch_manager = ctx.get_state("arch_manager")
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
    if not file_path.exists():
        return {
            "status": "error",
            "message": f"Architecture file {file_path} not found",
            "details": {"file_path": str(file_path)},
        }

    arch_manager = ctx.get_state("arch_manager")
    arch_manager.load_architecture(file_path)

    return {
        "status": "success",
        "message": "Architecture loaded successfully!",
        "details": {
            "file_path": str(file_path),
            "layers_count": len(arch_manager.current_architecture["layers"]),
        },
    }
