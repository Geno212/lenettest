# src/schemas/optimizers.py
"""Optimizer configurations and defaults."""

from typing import Dict, Any, Optional


OPTIMIZER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ADAM": {
        "description": "Adam optimizer - adaptive learning rate",
        "required": [],
        "optional": [
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "foreach",
            "maximize",
            "capturable",
            "differentiable",
        ],
        "defaults": {
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "foreach": None,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
        },
        "param_ranges": {
            "lr": (1e-6, 1.0),
            "weight_decay": (0, 0.1)
        },
        "use_cases": [
            "General purpose",
            "Works well out of the box",
            "Good for most deep learning tasks",
            "Recommended for beginners"
        ],
        "notes": "Most popular optimizer, good default choice"
    },
    
    "SGD": {
        "description": "Stochastic Gradient Descent",
        "required": [],
        "optional": [
            "lr",
            "momentum",
            "dampening",
            "weight_decay",
            "nesterov",
            "foreach",
            "maximize",
        ],
        "defaults": {
            "lr": 0.01,
            "momentum": 0,
            "weight_decay": 0.0,
            "dampening": 0,
            "nesterov": False,
            "foreach": None,
            "maximize": False,
        },
        "param_ranges": {
            "lr": (1e-4, 1.0),
            "momentum": (0, 1.0),
            "weight_decay": (0, 0.1)
        },
        "use_cases": [
            "Training from scratch",
            "Large batch sizes",
            "Often better final accuracy than Adam",
            "Requires more tuning"
        ],
        "notes": "Classic optimizer, often used with momentum=0.9"
    },
    
    "ADAMW": {
        "description": "Adam with decoupled weight decay",
        "required": [],
        "optional": [
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "foreach",
            "maximize",
            "capturable",
            "differentiable",
        ],
        "defaults": {
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
            "amsgrad": False,
            "foreach": None,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
        },
        "param_ranges": {
            "lr": (1e-6, 1.0),
            "weight_decay": (0, 0.1)
        },
        "use_cases": [
            "Better generalization than Adam",
            "Transformers and modern architectures",
            "When weight decay is important"
        ],
        "notes": "Improved version of Adam with better weight decay"
    },
    
    "RMSPROP": {
        "description": "RMSprop optimizer",
        "required": [],
        "optional": [
            "lr",
            "alpha",
            "eps",
            "weight_decay",
            "momentum",
            "centered",
            "foreach",
            "maximize",
        ],
        "defaults": {
            "lr": 0.01,
            "alpha": 0.99,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "momentum": 0,
            "centered": False,
            "foreach": None,
            "maximize": False,
        },
        "param_ranges": {
            "lr": (1e-6, 0.1),
            "weight_decay": (0, 0.1)
        },
        "use_cases": [
            "Recurrent neural networks",
            "Non-stationary objectives",
            "Online learning"
        ],
        "notes": "Good for RNNs, less common for CNNs"
    },
    # src/schemas/optimizers.py (continued)

    "ADAGRAD": {
        "description": "Adaptive Gradient optimizer",
        "required": [],
        "optional": [
            "lr",
            "lr_decay",
            "weight_decay",
            "eps",
            "foreach",
            "maximize",
        ],
        "defaults": {
            "lr": 0.01,
            "lr_decay": 0,
            "weight_decay": 0.0,
            "eps": 1e-10,
            "foreach": None,
            "maximize": False,
        },
        "param_ranges": {
            "lr": (1e-4, 0.1),
            "weight_decay": (0, 0.1)
        },
        "use_cases": [
            "Sparse data",
            "NLP tasks",
            "When features appear at different frequencies"
        ],
        "notes": "Learning rate decreases over time, can be too aggressive"
    }
}


def get_optimizer_config(optimizer_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for an optimizer."""
    return OPTIMIZER_CONFIGS.get(optimizer_name.upper())


def list_available_optimizers() -> list:
    """List all available optimizers."""
    return list(OPTIMIZER_CONFIGS.keys())


def get_optimizer_defaults(optimizer_name: str) -> Dict[str, Any]:
    """Get default parameters for an optimizer."""
    config = get_optimizer_config(optimizer_name)
    return config.get("defaults", {}) if config else {}


def get_recommended_lr(optimizer_name: str, training_from_scratch: bool = True) -> float:
    """
    Get recommended learning rate for optimizer.
    
    Args:
        optimizer_name: Name of optimizer
        training_from_scratch: True if training from scratch, False if fine-tuning
        
    Returns:
        Recommended learning rate
    """
    config = get_optimizer_config(optimizer_name)
    if not config:
        return 0.001
    
    default_lr = config["defaults"]["lr"]
    
    # Fine-tuning typically uses 10x lower learning rate
    if not training_from_scratch:
        return default_lr / 10
    
    return default_lr