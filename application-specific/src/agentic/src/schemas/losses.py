# src/schemas/losses.py
"""Loss function configurations."""

from typing import Dict, Any, List, Optional


LOSS_FUNCTIONS: Dict[str, Dict[str, Any]] = {
    "CROSSENTROPYLOSS": {
        "description": "Cross entropy loss for multi-class classification",
        "task_types": ["classification"],
        "required": [],
        "optional": ["ignore_index", "reduction", "label_smoothing"],
        "defaults": {
            # Match torch.nn.CrossEntropyLoss defaults
            "reduction": "mean",
            "label_smoothing": 0.0,
            "ignore_index": -100,
        },
        "param_types": {
            "ignore_index": int,
            "reduction": str,
            "label_smoothing": float
        },
        "use_cases": [
            "Multi-class classification (>2 classes)",
            "Mutually exclusive classes",
            "Most common classification loss"
        ],
        "notes": "Combines LogSoftmax and NLLLoss. Use for classification tasks.",
        "compatible_architectures": ["CNN", "ResNet", "VGG", "MobileNet"]
    },
    
    "BCELOSS": {
        "description": "Binary Cross Entropy loss",
        "task_types": ["classification"],
        "required": [],
        "optional": ["reduction"],
        "defaults": {
            "reduction": "mean"
        },
        "param_types": {
            "reduction": str
        },
        "use_cases": [
            "Binary classification (2 classes)",
            "Multi-label classification",
            "Requires sigmoid activation on output"
        ],
        "notes": "Use BCEWithLogitsLoss for numerical stability",
        "compatible_architectures": ["CNN", "ResNet", "Custom"]
    },
    
    "BCEWITHLOGITSLOSS": {
        "description": "BCE loss with built-in sigmoid",
        "task_types": ["classification"],
        "required": [],
        "optional": [ "reduction", "pos_weight"],
        "defaults": {
            "reduction": "mean"
        },
        "param_types": {
            "reduction": str,
            "pos_weight": "Tensor"
        },
        "use_cases": [
            "Binary classification",
            "Multi-label classification",
            "Imbalanced datasets (use pos_weight)"
        ],
        "notes": "More numerically stable than BCELoss. Preferred over BCELoss.",
        "compatible_architectures": ["CNN", "ResNet", "Custom"]
    },
    
    "MSELOSS": {
        "description": "Mean Squared Error loss",
        "task_types": ["regression"],
        "required": [],
        "optional": ["reduction"],
        "defaults": {
            "reduction": "mean"
        },
        "param_types": {
            "reduction": str
        },
        "use_cases": [
            "Regression tasks",
            "Predicting continuous values",
            "Image reconstruction"
        ],
        "notes": "L2 loss. Sensitive to outliers.",
        "compatible_architectures": ["CNN", "ResNet", "Custom"]
    },
    
    "L1LOSS": {
        "description": "Mean Absolute Error loss",
        "task_types": ["regression"],
        "required": [],
        "optional": ["reduction"],
        "defaults": {
            "reduction": "mean"
        },
        "param_types": {
            "reduction": str
        },
        "use_cases": [
            "Regression tasks",
            "More robust to outliers than MSE",
            "Image reconstruction"
        ],
        "notes": "L1 loss. Less sensitive to outliers than MSE.",
        "compatible_architectures": ["CNN", "ResNet", "Custom"]
    },
    
    "SMOOTHL1LOSS": {
        "description": "Smooth L1 loss (Huber loss)",
        "task_types": ["regression"],
        "required": [],
        "optional": ["reduction", "beta"],
        "defaults": {
            "reduction": "mean",
            "beta": 1.0
        },
        "param_types": {
            "reduction": str,
            "beta": float
        },
        "use_cases": [
            "Regression tasks",
            "Object detection (bounding box regression)",
            "Combines benefits of L1 and L2"
        ],
        "notes": "Less sensitive to outliers than MSE, smoother than L1.",
        "compatible_architectures": ["CNN", "ResNet", "Detection models"]
    },
    
    "NLLLOSS": {
        "description": "Negative Log Likelihood loss",
        "task_types": ["classification"],
        "required": [],
        "optional": [ "ignore_index", "reduction"],
        "defaults": {
            "reduction": "mean"
        },
        "param_types": {
            "ignore_index": int,
            "reduction": str
        },
        "use_cases": [
            "Multi-class classification",
            "When using LogSoftmax activation",
            "Used internally by CrossEntropyLoss"
        ],
        "notes": "Requires LogSoftmax activation. Usually use CrossEntropyLoss instead.",
        "compatible_architectures": ["CNN", "ResNet", "Custom"]
    },
    
    "KLDIVLOSS": {
        "description": "Kullback-Leibler divergence loss",
        "task_types": ["classification", "regression"],
        "required": [],
        "optional": ["reduction", "log_target"],
        "defaults": {
            "reduction": "mean",
            "log_target": False
        },
        "param_types": {
            "reduction": str,
            "log_target": bool
        },
        "use_cases": [
            "Distillation",
            "Matching probability distributions",
            "Semi-supervised learning"
        ],
        "notes": "Measures divergence between distributions.",
        "compatible_architectures": ["CNN", "ResNet", "Custom"]
    },
    
    "HingeLoss": {
        "description": "Hinge loss for SVMs",
        "task_types": ["classification"],
        "required": [],
        "optional": ["margin", "reduction"],
        "defaults": {
            "margin": 1.0,
            "reduction": "mean"
        },
        "param_types": {
            "margin": float,
            "reduction": str
        },
        "use_cases": [
            "Binary classification",
            "SVM-style classifiers",
            "Maximum margin classification"
        ],
        "notes": "Used in SVM classifiers. Less common in deep learning.",
        "compatible_architectures": ["Custom"]
    }
}


def get_loss_config(loss_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a loss function."""
    return LOSS_FUNCTIONS.get(loss_name.upper())


def list_available_losses() -> List[str]:
    """List all available loss functions."""
    return list(LOSS_FUNCTIONS.keys())


def get_losses_for_task(task_type: str) -> List[str]:
    """
    Get recommended loss functions for a task type.
    
    Args:
        task_type: Type of task (classification, regression, etc.)
        
    Returns:
        List of suitable loss function names
    """
    suitable_losses = []
    
    for loss_name, config in LOSS_FUNCTIONS.items():
        if task_type in config["task_types"]:
            suitable_losses.append(loss_name)
    
    return suitable_losses


def get_default_loss_for_task(task_type: str) -> str:
    """
    Get default loss function for a task type.
    
    Args:
        task_type: Type of task
        
    Returns:
        Default loss function name
    """
    defaults = {
        "classification": "CrossEntropyLoss",
        "regression": "MSELoss",
        "detection": "SmoothL1Loss",
        "segmentation": "CrossEntropyLoss"
    }
    
    return defaults.get(task_type, "CrossEntropyLoss")


def validate_loss_for_task(loss_name: str, task_type: str) -> Dict[str, Any]:
    """
    Validate if loss function is appropriate for task.
    
    Args:
        loss_name: Name of loss function
        task_type: Type of task
        
    Returns:
        {
            "valid": bool,
            "warning": Optional[str]
        }
    """
    config = get_loss_config(loss_name)
    
    if not config:
        return {
            "valid": False,
            "warning": f"Unknown loss function: {loss_name}"
        }
    
    if task_type not in config["task_types"]:
        return {
            "valid": False,
            "warning": f"{loss_name} is not suitable for {task_type} tasks. "
                      f"Recommended: {', '.join(get_losses_for_task(task_type))}"
        }
    
    return {"valid": True}