# src/schemas/requirements_schema.py
"""JSON schema for requirements extraction output validation."""

from typing import Dict, Any

# JSON schema for requirements extraction
REQUIREMENTS_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "task_type": {
            "type": "string",
            "enum": ["classification", "detection", "segmentation", "regression"],
            "description": "Type of machine learning task"
        },
        "dataset": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Dataset name (e.g., MNIST, CIFAR10, or custom name)"
                },
                "path": {
                    "type": "string",
                    "description": "Path to dataset on filesystem"
                },
                "input_shape": {
                    "type": "object",
                    "properties": {
                        "height": {"type": "integer", "minimum": 1},
                        "width": {"type": "integer", "minimum": 1},
                        "channels": {"type": "integer", "minimum": 1, "maximum": 4}
                    },
                    "required": ["height", "width", "channels"]
                },
                "output_classes": {
                    "type": "integer",
                    "minimum": 2,
                    "description": "Number of classes for classification tasks"
                }
            },
            "required": ["name"]
        },
        "architecture": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["pretrained", "custom"],
                    "description": "Whether to use pretrained model or custom architecture"
                },
                "pretrained_model": {
                    "type": "string",
                    "description": "Name of pretrained model (if type='pretrained')"
                },
                "layers": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Layer definitions (if type='custom')"
                }
            },
            "required": ["type"]
        },
        "target_metrics": {
            "type": "object",
            "properties": {
                "accuracy": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Target accuracy (0.0 to 1.0)"
                }
            },
            "required": ["accuracy"]
        },
        "training_config": {
            "type": "object",
            "properties": {
                "optimizer": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "lr": {"type": "number", "minimum": 0, "exclusiveMinimum": True}
                    },
                    "required": ["type"]
                },
                "loss_function": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"}
                    },
                    "required": ["type"]
                },
                "scheduler": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"}
                    }
                },
                "epochs": {
                    "type": "integer",
                    "minimum": 1
                },
                "batch_size": {
                    "type": "integer",
                    "minimum": 1
                },
                "device": {
                    "type": "string",
                }
            }
        }
    },
    "required": ["task_type", "dataset", "architecture", "target_metrics"]
}


def get_schema_example() -> str:
    """
    Get example JSON output matching the schema.
    
    Returns:
        Example JSON string for LLM prompt
    """
    return """{
  "task_type": "classification",
  "dataset": {
    "name": "MNIST",
    "path": "~/data/mnist"
  },
  "architecture": {
    "type": "pretrained",
    "pretrained_model": "resnet18"
  },
  "target_metrics": {
    "accuracy": 0.95
  },
  "training_config": {
    "optimizer": {
      "type": "Adam",
      "lr": 0.001
    },
    "loss_function": {
      "type": "CrossEntropyLoss"
    },
    "epochs": 50,
    "batch_size": 64
  }
}"""


def get_minimal_schema_example() -> str:
    """
    Get minimal required JSON output.
    
    Returns:
        Minimal example JSON string
    """
    return """{
  "task_type": "classification",
  "dataset": {
    "name": "CIFAR10"
  },
  "architecture": {
    "type": "pretrained",
    "pretrained_model": "resnet18"
  },
  "target_metrics": {
    "accuracy": 0.90
  }
}"""


# Export public API
__all__ = [
    "REQUIREMENTS_JSON_SCHEMA",
    "get_schema_example",
    "get_minimal_schema_example"
]
