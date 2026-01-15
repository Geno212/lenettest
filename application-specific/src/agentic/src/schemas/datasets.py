# src/schemas/datasets.py
"""Standard dataset definitions and metadata."""

from typing import Dict, Any


STANDARD_DATASETS: Dict[str, Dict[str, Any]] = {
    "MNIST": {
        "name": "MNIST",
        "description": "Handwritten digits (0-9)",
        "task_type": "classification",
        "input_shape": {
            "height": 28,
            "width": 28,
            "channels": 1
        },
        "output_classes": 10,
        "download_url": "http://yann.lecun.com/exdb/mnist/",
        "size_mb": 11,
        "num_train": 60000,
        "num_test": 10000,
        "recommended_models": ["custom_cnn", "lenet", "simple_mlp"],
        "typical_accuracy": {
            "custom_cnn": 0.99,
            "pretrained": 0.995
        }
    },
    
    "CIFAR10": {
        "name": "CIFAR10",
        "description": "10 classes of 32x32 RGB images",
        "task_type": "classification",
        "input_shape": {
            "height": 32,
            "width": 32,
            "channels": 3
        },
        "output_classes": 10,
        "classes": ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"],
        "download_url": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "size_mb": 170,
        "num_train": 50000,
        "num_test": 10000,
        "recommended_models": ["resnet18", "vgg16", "mobilenet_v2"],
        "typical_accuracy": {
            "custom_cnn": 0.75,
            "resnet18": 0.90,
            "resnet50": 0.93
        }
    },
    
    "CIFAR100": {
        "name": "CIFAR100",
        "description": "100 classes of 32x32 RGB images",
        "task_type": "classification",
        "input_shape": {
            "height": 32,
            "width": 32,
            "channels": 3
        },
        "output_classes": 100,
        "download_url": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "size_mb": 170,
        "num_train": 50000,
        "num_test": 10000,
        "recommended_models": ["resnet50", "efficientnet_b0"],
        "typical_accuracy": {
            "custom_cnn": 0.45,
            "resnet50": 0.70,
            "efficientnet_b0": 0.73
        }
    },
    
    "IMAGENET": {
        "name": "ImageNet",
        "description": "1000 classes of high-resolution images",
        "task_type": "classification",
        "input_shape": {
            "height": 224,
            "width": 224,
            "channels": 3
        },
        "output_classes": 1000,
        "download_url": "https://image-net.org/",
        "size_mb": 150000,  # ~150GB
        "num_train": 1281167,
        "num_test": 50000,
        "recommended_models": ["resnet50", "efficientnet_b0", "vgg16"],
        "typical_accuracy": {
            "resnet50": 0.76,
            "efficientnet_b0": 0.77,
            "vgg16": 0.71
        },
        "notes": "Very large dataset, requires significant storage and training time"
    },
    
    "FASHION_MNIST": {
        "name": "Fashion-MNIST",
        "description": "Fashion products (10 classes)",
        "task_type": "classification",
        "input_shape": {
            "height": 28,
            "width": 28,
            "channels": 1
        },
        "output_classes": 10,
        "classes": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
        "download_url": "https://github.com/zalandoresearch/fashion-mnist",
        "size_mb": 30,
        "num_train": 60000,
        "num_test": 10000,
        "recommended_models": ["custom_cnn", "resnet18"],
        "typical_accuracy": {
            "custom_cnn": 0.90,
            "resnet18": 0.94
        }
    },
    
    "COCO": {
        "name": "COCO",
        "description": "Common Objects in Context - detection/segmentation",
        "task_type": "detection",
        "input_shape": {
            "height": 640,
            "width": 640,
            "channels": 3
        },
        "output_classes": 80,
        "download_url": "https://cocodataset.org/",
        "size_mb": 25000,  # ~25GB
        "num_train": 118287,
        "num_val": 5000,
        "recommended_models": ["yolov5", "fasterrcnn", "maskrcnn"],
        "notes": "Requires specialized detection/segmentation models"
    }
}


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get metadata for a standard dataset.
    
    Args:
        dataset_name: Name of dataset (case-insensitive)
        
    Returns:
        Dataset metadata dictionary or None if not found
    """
    return STANDARD_DATASETS.get(dataset_name.upper())


def is_standard_dataset(dataset_name: str) -> bool:
    """Check if dataset is a standard dataset."""
    return dataset_name.upper() in STANDARD_DATASETS


def list_standard_datasets() -> list:
    """List all available standard datasets."""
    return list(STANDARD_DATASETS.keys())


def get_recommended_models(dataset_name: str) -> list:
    """Get recommended models for a dataset."""
    info = get_dataset_info(dataset_name)
    return info.get("recommended_models", []) if info else []


def get_typical_accuracy(dataset_name: str, model_name: str) -> float:
    """Get typical accuracy for a model on a dataset."""
    info = get_dataset_info(dataset_name)
    if not info:
        return None
    return info.get("typical_accuracy", {}).get(model_name)