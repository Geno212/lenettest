# src/schemas/layers.py
"""Layer type schemas and validation rules."""

from typing import Dict, Any, List, Optional


LAYER_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "Conv2d": {
        "description": "2D Convolution layer",
        "required": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
        "optional": ["dilation", "groups", "bias"],
        "defaults": {
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "groups": 1,
            "bias": True
        },
        "param_types": {
            "in_channels": int,
            "out_channels": int,
            "kernel_size": [int, tuple],
            "stride": [int, tuple],
            "padding": [int, tuple, str],
            "dilation": [int, tuple],
            "groups": int,
            "bias": bool
        },
        "notes": "in_channels will be inferred from previous layer except for first layer"
    },
    
    "MaxPool2d": {
        "description": "Max pooling layer",
        "required": ["kernel_size", "stride", "padding"],
        "optional": ["dilation"],
        "defaults": {
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
            "dilation": 1
        },
        "param_types": {
            "kernel_size": [int, tuple],
            "stride": [int, tuple],
            "padding": [int, tuple],
            "dilation": [int, tuple]
        }
    },
    
    "AvgPool2d": {
        "description": "Average pooling layer",
        "required": ["kernel_size", "stride", "padding"],
        "optional": [],
        "defaults": {
            "kernel_size": 2,
            "stride": 2,
            "padding": 0
        },
        "param_types": {
            "kernel_size": [int, tuple],
            "stride": [int, tuple],
            "padding": [int, tuple]
        }
    },
    
    "Linear": {
        "description": "Fully connected layer",
        "required": ["in_features", "out_features"],
        "optional": ["bias"],
        "defaults": {
            "bias": True
        },
        "param_types": {
            "in_features": int,
            "out_features": int,
            "bias": bool
        },
        "notes": "in_features will be inferred from flattened shape except for first layer"
    },
    
    "ReLU": {
        "description": "ReLU activation function",
        "required": [],
        "optional": ["inplace"],
        "defaults": {
            "inplace": False
        },
        "param_types": {
            "inplace": bool
        }
    },
    
    "LeakyReLU": {
        "description": "Leaky ReLU activation",
        "required": [],
        "optional": ["negative_slope", "inplace"],
        "defaults": {
            "negative_slope": 0.01,
            "inplace": False
        },
        "param_types": {
            "negative_slope": float,
            "inplace": bool
        }
    },
    
    "Sigmoid": {
        "description": "Sigmoid activation",
        "required": [],
        "optional": [],
        "defaults": {},
        "param_types": {}
    },
    
    "Tanh": {
        "description": "Tanh activation",
        "required": [],
        "optional": [],
        "defaults": {},
        "param_types": {}
    },
    
    "Softmax": {
        "description": "Softmax activation",
        "required": [],
        "optional": ["dim"],
        "defaults": {
            "dim": 1
        },
        "param_types": {
            "dim": int
        }
    },
    
    "Dropout": {
        "description": "Dropout regularization",
        "required": [],
        "optional": ["p", "inplace"],
        "defaults": {
            "p": 0.5,
            "inplace": False
        },
        "param_types": {
            "p": float,
            "inplace": bool
        }
    },
    
    "BatchNorm2d": {
        "description": "Batch normalization for 2D data",
        "required": ["num_features"],
        "optional": ["eps", "momentum", "affine", "track_running_stats"],
        "defaults": {
            "eps": 1e-5,
            "momentum": 0.1,
            "affine": True,
            "track_running_stats": True
        },
        "param_types": {
            "num_features": int,
            "eps": float,
            "momentum": float,
            "affine": bool,
            "track_running_stats": bool
        },
        "notes": "num_features should match out_channels of previous conv layer"
    },
    
    "Flatten": {
        "description": "Flatten spatial dimensions",
        "required": [],
        "optional": ["start_dim", "end_dim"],
        "defaults": {
            "start_dim": 1,
            "end_dim": -1
        },
        "param_types": {
            "start_dim": int,
            "end_dim": int
        },
        "notes": "Required before first Linear layer when coming from Conv layers"
    },
    
    "AdaptiveAvgPool2d": {
        "description": "Adaptive average pooling to fixed output size",
        "required": ["output_size"],
        "optional": [],
        "defaults": {},
        "param_types": {
            "output_size": [int, tuple]
        }
    }
    ,
    "Conv1d": {
        "description": "1D Convolution layer",
        "required": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
        "optional": ["dilation", "groups", "bias"],
        "defaults": {
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "groups": 1,
            "bias": True
        },
        "param_types": {
            "in_channels": int,
            "out_channels": int,
            "kernel_size": [int, tuple],
            "stride": [int, tuple],
            "padding": [int, tuple, str],
            "dilation": [int, tuple],
            "groups": int,
            "bias": bool
        }
    },
    "Conv3d": {
        "description": "3D Convolution layer",
        "required": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
        "optional": ["dilation", "groups", "bias"],
        "defaults": {
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "groups": 1,
            "bias": True
        },
        "param_types": {
            "in_channels": int,
            "out_channels": int,
            "kernel_size": [int, tuple],
            "stride": [int, tuple],
            "padding": [int, tuple, str],
            "dilation": [int, tuple],
            "groups": int,
            "bias": bool
        }
    },
    "ConvTranspose2d": {
        "description": "2D transposed convolution (deconvolution)",
        "required": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
        "optional": ["output_padding", "groups", "bias", "dilation"],
        "defaults": {
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "output_padding": 0,
            "groups": 1,
            "bias": True,
            "dilation": 1
        },
        "param_types": {
            "in_channels": int,
            "out_channels": int,
            "kernel_size": [int, tuple],
            "stride": [int, tuple],
            "padding": [int, tuple],
            "output_padding": [int, tuple],
            "groups": int,
            "bias": bool,
            "dilation": [int, tuple]
        }
    },
    "ConvTranspose1d": {
        "description": "1D transposed convolution",
        "required": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
        "optional": ["output_padding", "groups", "bias", "dilation"],
        "defaults": {
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "output_padding": 0,
            "groups": 1,
            "bias": True,
            "dilation": 1
        },
        "param_types": {
            "in_channels": int,
            "out_channels": int,
            "kernel_size": [int, tuple],
            "stride": [int, tuple],
            "padding": [int, tuple],
            "output_padding": [int, tuple],
            "groups": int,
            "bias": bool,
            "dilation": [int, tuple]
        }
    },
    "ConvTranspose3d": {
        "description": "3D transposed convolution",
        "required": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
        "optional": ["output_padding", "groups", "bias", "dilation"],
        "defaults": {
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "output_padding": 0,
            "groups": 1,
            "bias": True,
            "dilation": 1
        },
        "param_types": {
            "in_channels": int,
            "out_channels": int,
            "kernel_size": [int, tuple],
            "stride": [int, tuple],
            "padding": [int, tuple],
            "output_padding": [int, tuple],
            "groups": int,
            "bias": bool,
            "dilation": [int, tuple]
        }
    },
    "MaxPool1d": {
        "description": "Max pooling 1D",
        "required": ["kernel_size", "stride", "padding"],
        "optional": ["dilation"],
        "defaults": {
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
            "dilation": 1
        },
        "param_types": {
            "kernel_size": [int, tuple],
            "stride": [int, tuple],
            "padding": [int, tuple],
            "dilation": [int, tuple]
        }
    },
    "MaxPool3d": {
        "description": "Max pooling 3D",
        "required": ["kernel_size", "stride", "padding"],
        "optional": ["dilation"],
        "defaults": {
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
            "dilation": 1
        },
        "param_types": {
            "kernel_size": [int, tuple],
            "stride": [int, tuple],
            "padding": [int, tuple],
            "dilation": [int, tuple]
        }
    },
    "AvgPool1d": {
        "description": "Average pooling 1D",
        "required": ["kernel_size", "stride", "padding"],
        "optional": [],
        "defaults": {
            "kernel_size": 2,
            "stride": 2,
            "padding": 0
        },
        "param_types": {
            "kernel_size": [int, tuple],
            "stride": [int, tuple],
            "padding": [int, tuple]
        }
    },
    "AvgPool3d": {
        "description": "Average pooling 3D",
        "required": ["kernel_size", "stride", "padding"],
        "optional": [],
        "defaults": {
            "kernel_size": 2,
            "stride": 2,
            "padding": 0
        },
        "param_types": {
            "kernel_size": [int, tuple],
            "stride": [int, tuple],
            "padding": [int, tuple]
        }
    },
    "AdaptiveAvgPool1d": {
        "description": "Adaptive average pooling 1D",
        "required": ["output_size"],
        "optional": [],
        "defaults": {},
        "param_types": {"output_size": [int, tuple]}
    },
    "AdaptiveAvgPool3d": {
        "description": "Adaptive average pooling 3D",
        "required": ["output_size"],
        "optional": [],
        "defaults": {},
        "param_types": {"output_size": [int, tuple]}
    },
    "AdaptiveMaxPool1d": {
        "description": "Adaptive max pooling 1D",
        "required": ["output_size"],
        "optional": [],
        "defaults": {},
        "param_types": {"output_size": [int, tuple]}
    },
    "AdaptiveMaxPool2d": {
        "description": "Adaptive max pooling 2D",
        "required": ["output_size"],
        "optional": [],
        "defaults": {},
        "param_types": {"output_size": [int, tuple]}
    },
    "AdaptiveMaxPool3d": {
        "description": "Adaptive max pooling 3D",
        "required": ["output_size"],
        "optional": [],
        "defaults": {},
        "param_types": {"output_size": [int, tuple]}
    },
    "BatchNorm1d": {
        "description": "Batch normalization for 1D data",
        "required": ["num_features"],
        "optional": ["eps", "momentum", "affine", "track_running_stats"],
        "defaults": {
            "eps": 1e-5,
            "momentum": 0.1,
            "affine": True,
            "track_running_stats": True
        },
        "param_types": {
            "num_features": int,
            "eps": float,
            "momentum": float,
            "affine": bool,
            "track_running_stats": bool
        }
    },
    "BatchNorm3d": {
        "description": "Batch normalization for 3D data",
        "required": ["num_features"],
        "optional": ["eps", "momentum", "affine", "track_running_stats"],
        "defaults": {
            "eps": 1e-5,
            "momentum": 0.1,
            "affine": True,
            "track_running_stats": True
        },
        "param_types": {
            "num_features": int,
            "eps": float,
            "momentum": float,
            "affine": bool,
            "track_running_stats": bool
        }
    },
    "InstanceNorm1d": {
        "description": "Instance normalization for 1D data",
        "required": ["num_features"],
        "optional": ["eps", "momentum", "affine", "track_running_stats"],
        "defaults": {"eps": 1e-5, "momentum": 0.1, "affine": False, "track_running_stats": False},
        "param_types": {"num_features": int, "eps": float, "momentum": float, "affine": bool, "track_running_stats": bool}
    },
    "InstanceNorm2d": {
        "description": "Instance normalization for 2D data",
        "required": ["num_features"],
        "optional": ["eps", "momentum", "affine", "track_running_stats"],
        "defaults": {"eps": 1e-5, "momentum": 0.1, "affine": False, "track_running_stats": False},
        "param_types": {"num_features": int, "eps": float, "momentum": float, "affine": bool, "track_running_stats": bool}
    },
    "InstanceNorm3d": {
        "description": "Instance normalization for 3D data",
        "required": ["num_features"],
        "optional": ["eps", "momentum", "affine", "track_running_stats"],
        "defaults": {"eps": 1e-5, "momentum": 0.1, "affine": False, "track_running_stats": False},
        "param_types": {"num_features": int, "eps": float, "momentum": float, "affine": bool, "track_running_stats": bool}
    },
    "LayerNorm": {
        "description": "Layer normalization",
        "required": ["normalized_shape"],
        "optional": ["eps", "elementwise_affine"],
        "defaults": {"eps": 1e-5, "elementwise_affine": True},
        "param_types": {"normalized_shape": [int, tuple, list], "eps": float, "elementwise_affine": bool}
    },
    "GroupNorm": {
        "description": "Group normalization",
        "required": ["num_groups", "num_channels"],
        "optional": ["eps", "affine"],
        "defaults": {"eps": 1e-5, "affine": True},
        "param_types": {"num_groups": int, "num_channels": int, "eps": float, "affine": bool}
    },
    "LocalResponseNorm": {
        "description": "Local Response Normalization (LRN)",
        "required": ["size"],
        "optional": ["alpha", "beta", "k"],
        "defaults": {"alpha": 1e-4, "beta": 0.75, "k": 1.0},
        "param_types": {"size": int, "alpha": float, "beta": float, "k": float}
    },
    "Dropout2d": {
        "description": "2D Dropout (channels)",
        "required": [],
        "optional": ["p", "inplace"],
        "defaults": {"p": 0.5, "inplace": False},
        "param_types": {"p": float, "inplace": bool}
    },
    "Dropout3d": {
        "description": "3D Dropout (channels)",
        "required": [],
        "optional": ["p", "inplace"],
        "defaults": {"p": 0.5, "inplace": False},
        "param_types": {"p": float, "inplace": bool}
    },
    "AlphaDropout": {
        "description": "Alpha Dropout",
        "required": [],
        "optional": ["p"],
        "defaults": {"p": 0.5},
        "param_types": {"p": float}
    },
    "Embedding": {
        "description": "Embedding layer for discrete tokens",
        "required": ["num_embeddings", "embedding_dim"],
        "optional": ["padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse"],
        "defaults": {"padding_idx": None, "max_norm": None, "norm_type": 2.0, "scale_grad_by_freq": False, "sparse": False},
        "param_types": {"num_embeddings": int, "embedding_dim": int, "padding_idx": [int, type(None)], "max_norm": [float, type(None)], "norm_type": float, "scale_grad_by_freq": bool, "sparse": bool}
    },
    "Upsample": {
        "description": "Upsampling layer",
        "required": [],
        "optional": ["size", "scale_factor", "mode", "align_corners"],
        "defaults": {"mode": "nearest", "align_corners": None},
        "param_types": {"size": [int, tuple, type(None)], "scale_factor": [int, float, tuple, type(None)], "mode": str, "align_corners": [bool, type(None)]}
    },
    "PixelShuffle": {
        "description": "Pixel shuffle for sub-pixel convolution upsampling",
        "required": ["upscale_factor"],
        "optional": [],
        "defaults": {},
        "param_types": {"upscale_factor": int}
    },
    "ReflectionPad2d": {
        "description": "Reflection padding for 2D inputs",
        "required": ["padding"],
        "optional": [],
        "defaults": {},
        "param_types": {"padding": [int, tuple]}
    },
    "ReplicationPad2d": {
        "description": "Replication padding for 2D inputs",
        "required": ["padding"],
        "optional": [],
        "defaults": {},
        "param_types": {"padding": [int, tuple]}
    },
    "ZeroPad2d": {
        "description": "Zero padding for 2D inputs",
        "required": ["padding"],
        "optional": [],
        "defaults": {},
        "param_types": {"padding": [int, tuple]}
    },
    "ELU": {
        "description": "ELU activation",
        "required": [],
        "optional": ["alpha", "inplace"],
        "defaults": {"alpha": 1.0, "inplace": False},
        "param_types": {"alpha": float, "inplace": bool}
    },
    "SELU": {
        "description": "SELU activation",
        "required": [],
        "optional": [],
        "defaults": {},
        "param_types": {}
    },
    "PReLU": {
        "description": "PReLU activation",
        "required": [],
        "optional": ["num_parameters", "init"],
        "defaults": {"num_parameters": 1, "init": 0.25},
        "param_types": {"num_parameters": int, "init": float}
    },
    "GELU": {
        "description": "GELU activation",
        "required": [],
        "optional": [],
        "defaults": {},
        "param_types": {}
    },
    "Softplus": {
        "description": "Softplus activation",
        "required": [],
        "optional": ["beta", "threshold"],
        "defaults": {"beta": 1.0, "threshold": 20.0},
        "param_types": {"beta": float, "threshold": float}
    },
    "Hardtanh": {
        "description": "Hardtanh activation",
        "required": [],
        "optional": ["min_val", "max_val", "inplace"],
        "defaults": {"min_val": -1.0, "max_val": 1.0, "inplace": False},
        "param_types": {"min_val": float, "max_val": float, "inplace": bool}
    },
    "ReLU6": {
        "description": "ReLU6 activation",
        "required": [],
        "optional": ["inplace"],
        "defaults": {"inplace": False},
        "param_types": {"inplace": bool}
    }
}


def get_layer_schema(layer_type: str) -> Optional[Dict[str, Any]]:
    """Get schema for a layer type.

    Returns None if the layer type is not available.
    """
    return LAYER_SCHEMAS.get(layer_type)


def list_available_layers() -> List[str]:
    """List all available layer types."""
    return list(LAYER_SCHEMAS.keys())


def get_required_params(layer_type: str) -> List[str]:
    """Get required parameters for a layer type."""
    schema = get_layer_schema(layer_type)
    return schema.get("required", []) if schema else []


def get_optional_params(layer_type: str) -> List[str]:
    """Get optional parameters for a layer type."""
    schema = get_layer_schema(layer_type)
    return schema.get("optional", []) if schema else []


def get_default_params(layer_type: str) -> Dict[str, Any]:
    """Get default parameter values for a layer type."""
    schema = get_layer_schema(layer_type)
    return schema.get("defaults", {}) if schema else {}