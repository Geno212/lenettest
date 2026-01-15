# src/schemas/pretrained_models.py
"""Pretrained model metadata and configurations."""

from typing import Dict, Any, List, Optional


PRETRAINED_MODELS: Dict[str, Dict[str, Any]] = {
    "resnet18": {
        "name": "ResNet18",
        "family": "ResNet",
        "description": "18-layer residual network",
        "parameters": 11689512,
        "input_size": 224,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 44.7,
        "top1_accuracy": 69.76,
        "top5_accuracy": 89.08,
        "use_cases": ["Image classification", "Transfer learning", "Feature extraction"],
        "pros": ["Fast training", "Good balance of accuracy and speed", "Relatively small"],
        "cons": ["Lower accuracy than deeper variants"],
        "recommended_for": ["General purpose", "Quick prototyping", "Limited compute"]
    },
    
    "resnet34": {
        "name": "ResNet34",
        "family": "ResNet",
        "description": "34-layer residual network",
        "parameters": 21797672,
        "input_size": 224,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 83.3,
        "top1_accuracy": 73.31,
        "top5_accuracy": 91.42,
        "use_cases": ["Image classification", "Transfer learning"],
        "pros": ["Better accuracy than ResNet18", "Still relatively fast"],
        "cons": ["Larger than ResNet18"],
        "recommended_for": ["When accuracy matters more than speed"]
    },
    
    "resnet50": {
        "name": "ResNet50",
        "family": "ResNet",
        "description": "50-layer residual network with bottleneck blocks",
        "parameters": 25557032,
        "input_size": 224,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 97.8,
        "top1_accuracy": 76.13,
        "top5_accuracy": 92.86,
        "use_cases": ["Image classification", "Object detection", "Feature extraction"],
        "pros": ["High accuracy", "Well-established", "Many pretrained versions"],
        "cons": ["Slower than ResNet18/34", "Larger model"],
        "recommended_for": ["Production systems", "High accuracy requirements"]
    },
    
    "resnet101": {
        "name": "ResNet101",
        "family": "ResNet",
        "description": "101-layer residual network",
        "parameters": 44549160,
        "input_size": 224,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 170.5,
        "top1_accuracy": 77.37,
        "top5_accuracy": 93.56,
        "use_cases": ["High accuracy classification", "Complex tasks"],
        "pros": ["Very high accuracy"],
        "cons": ["Large and slow", "Requires significant compute"],
        "recommended_for": ["When maximum accuracy is needed"]
    },
    
    "resnet152": {
        "name": "ResNet152",
        "family": "ResNet",
        "description": "152-layer residual network",
        "parameters": 60192808,
        "input_size": 224,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 230.4,
        "top1_accuracy": 78.31,
        "top5_accuracy": 94.05,
        "use_cases": ["Maximum accuracy classification"],
        "pros": ["Highest accuracy in ResNet family"],
        "cons": ["Very large and slow", "Diminishing returns vs ResNet101"],
        "recommended_for": ["Research", "When every percentage point matters"]
    },
    
    "vgg11": {
        "name": "VGG11",
        "family": "VGG",
        "description": "11-layer VGG network",
        "parameters": 132863336,
        "input_size": 224,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 506.8,
        "top1_accuracy": 69.02,
        "top5_accuracy": 88.63,
        "use_cases": ["Image classification", "Teaching/learning"],
        "pros": ["Simple architecture", "Easy to understand"],
        "cons": ["Very large", "Outdated"],
        "recommended_for": ["Educational purposes", "Simple tasks"]
    },
    
    "vgg16": {
        "name": "VGG16",
        "family": "VGG",
        "description": "16-layer VGG network",
        "parameters": 138357544,
        "input_size": 224,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 527.8,
        "top1_accuracy": 71.59,
        "top5_accuracy": 90.38,
        "use_cases": ["Image classification", "Feature extraction", "Style transfer"],
        "pros": ["Simple architecture", "Good features for style transfer"],
        "cons": ["Extremely large", "Slow", "Outdated"],
        "recommended_for": ["Feature extraction", "Legacy applications"]
    },
    
    "vgg19": {
        "name": "VGG19",
        "family": "VGG",
        "description": "19-layer VGG network",
        "parameters": 143667240,
        "input_size": 224,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 548.1,
        "top1_accuracy": 72.38,
        "top5_accuracy": 90.88,
        "use_cases": ["Feature extraction", "Style transfer"],
        "pros": ["Rich features", "Good for style transfer"],
        "cons": ["Extremely large", "Very slow"],
        "recommended_for": ["Style transfer applications"]
    },
    
    "mobilenet_v2": {
        "name": "MobileNetV2",
        "family": "MobileNet",
        "description": "Efficient mobile architecture with inverted residuals",
        "parameters": 3504872,
        "input_size": 224,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 13.6,
        "top1_accuracy": 71.88,
        "top5_accuracy": 90.29,
        "use_cases": ["Mobile deployment", "Edge devices", "Real-time applications"],
        "pros": ["Very small", "Fast inference", "Mobile-optimized"],
        "cons": ["Lower accuracy than ResNet"],
        "recommended_for": ["Mobile apps", "Embedded systems", "Limited resources"]
    },
    
    "mobilenet_v3_small": {
        "name": "MobileNetV3-Small",
        "family": "MobileNet",
        "description": "Smallest MobileNetV3 variant",
        "parameters": 2542856,
        "input_size": 224,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 9.8,
        "top1_accuracy": 67.67,
        "top5_accuracy": 87.40,
        "use_cases": ["Mobile deployment", "Ultra-lightweight applications"],
        "pros": ["Extremely small and fast"],
        "cons": ["Lower accuracy"],
        "recommended_for": ["Severely constrained environments"]
    },
    
    "mobilenet_v3_large": {
        "name": "MobileNetV3-Large",
        "family": "MobileNet",
        "description": "Larger MobileNetV3 variant",
        "parameters": 5483032,
        "input_size": 224,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 21.1,
        "top1_accuracy": 74.04,
        "top5_accuracy": 91.34,
        "use_cases": ["Mobile deployment with higher accuracy"],
        "pros": ["Good balance of size and accuracy", "Mobile-optimized"],
        "cons": ["Still lower accuracy than ResNet50"],
        "recommended_for": ["Mobile apps needing better accuracy"]
    },
    
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "family": "EfficientNet",
        "description": "Baseline EfficientNet with compound scaling",
        "parameters": 5288548,
        "input_size": 224,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 20.3,
        "top1_accuracy": 77.69,
        "top5_accuracy": 93.53,
        "use_cases": ["Efficient high-accuracy classification"],
        "pros": ["Best accuracy-to-size ratio", "State-of-the-art efficiency"],
        "cons": ["More complex architecture"],
        "recommended_for": ["Production systems", "Best performance per parameter"]
    },
    
    "efficientnet_b1": {
        "name": "EfficientNet-B1",
        "family": "EfficientNet",
        "description": "Scaled up EfficientNet",
        "parameters": 7794184,
        "input_size": 240,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 30.1,
        "top1_accuracy": 79.84,
        "top5_accuracy": 94.93,
        "use_cases": ["High accuracy with efficiency"],
        "pros": ["Excellent accuracy for size"],
        "cons": ["Slightly larger input size"],
        "recommended_for": ["When you can afford slightly more compute"]
    },
    
    "efficientnet_b3": {
        "name": "EfficientNet-B3",
        "family": "EfficientNet",
        "description": "Mid-range EfficientNet",
        "parameters": 12233232,
        "input_size": 300,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 47.2,
        "top1_accuracy": 82.01,
        "top5_accuracy": 96.08,
        "use_cases": ["High accuracy applications"],
        "pros": ["Very high accuracy", "Still efficient"],
        "cons": ["Larger input size (300×300)"],
        "recommended_for": ["Production systems prioritizing accuracy"]
    },
    
    "densenet121": {
        "name": "DenseNet121",
        "family": "DenseNet",
        "description": "121-layer densely connected network",
        "parameters": 7978856,
        "input_size": 224,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 30.8,
        "top1_accuracy": 74.43,
        "top5_accuracy": 91.97,
        "use_cases": ["Image classification", "Dense feature connections"],
        "pros": ["Feature reuse", "Parameter efficient"],
        "cons": ["Memory intensive during training"],
        "recommended_for": ["When feature reuse is important"]
    },
    
    "densenet161": {
        "name": "DenseNet161",
        "family": "DenseNet",
        "description": "161-layer densely connected network",
        "parameters": 28681000,
        "input_size": 224,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 110.4,
        "top1_accuracy": 77.14,
        "top5_accuracy": 93.56,
        "use_cases": ["High accuracy classification"],
        "pros": ["High accuracy", "Rich features"],
        "cons": ["Large and memory intensive"],
        "recommended_for": ["When accuracy is priority and memory available"]
    },
    
    "squeezenet1_0": {
        "name": "SqueezeNet 1.0",
        "family": "SqueezeNet",
        "description": "Extremely compact architecture",
        "parameters": 1248424,
        "input_size": 224,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 4.8,
        "top1_accuracy": 58.09,
        "top5_accuracy": 80.42,
        "use_cases": ["Ultra-lightweight deployment"],
        "pros": ["Extremely small"],
        "cons": ["Low accuracy"],
        "recommended_for": ["When size is absolutely critical"]
    },
    
    "alexnet": {
        "name": "AlexNet",
        "family": "AlexNet",
        "description": "Historic CNN that won ImageNet 2012",
        "parameters": 61100840,
        "input_size": 224,
        "pretrained_on": "ImageNet",
        "num_classes": 1000,
        "size_mb": 233.1,
        "top1_accuracy": 56.52,
        "top5_accuracy": 79.07,
        "use_cases": ["Educational", "Historical"],
        "pros": ["Historic significance", "Simple architecture"],
        "cons": ["Outdated", "Low accuracy", "Large"],
        "recommended_for": ["Educational purposes only"]
    },

    "yolox-nano": {
        "name": "YOLOX-Nano",
        "family": "YOLOX",
        "description": "Ultra-lightweight YOLOX variant for object detection",
        "parameters": 915000,
        "input_size": 416,
        "pretrained_on": "COCO",
        "num_classes": 80,
        "size_mb": 3.5,
        "top1_accuracy": None,
        "top5_accuracy": None,
        "mAP": 25.3,
        "use_cases": ["Object detection", "Edge devices", "Real-time applications"],
        "pros": ["Extremely small", "Fast inference", "Low power consumption"],
        "cons": ["Lower accuracy"],
        "recommended_for": ["Mobile and embedded systems", "Resource-constrained environments"]
    },

    "yolox-tiny": {
        "name": "YOLOX-Tiny",
        "family": "YOLOX",
        "description": "Tiny YOLOX variant for object detection",
        "parameters": 5100000,
        "input_size": 416,
        "pretrained_on": "COCO",
        "num_classes": 80,
        "size_mb": 19.5,
        "top1_accuracy": None,
        "top5_accuracy": None,
        "mAP": 32.8,
        "use_cases": ["Object detection", "Real-time applications"],
        "pros": ["Small size", "Good speed-accuracy balance"],
        "cons": ["Moderate accuracy"],
        "recommended_for": ["Mobile applications", "Quick prototyping"]
    },

    "yolox-small": {
        "name": "YOLOX-S",
        "family": "YOLOX",
        "description": "Small YOLOX variant for object detection",
        "parameters": 9000000,
        "input_size": 640,
        "pretrained_on": "COCO",
        "num_classes": 80,
        "size_mb": 34.3,
        "top1_accuracy": None,
        "top5_accuracy": None,
        "mAP": 40.5,
        "use_cases": ["Object detection", "Production systems"],
        "pros": ["Good accuracy", "Reasonable size"],
        "cons": ["Larger than tiny variants"],
        "recommended_for": ["General object detection tasks"]
    },

    "yolox-medium": {
        "name": "YOLOX-M",
        "family": "YOLOX",
        "description": "Medium YOLOX variant for object detection",
        "parameters": 25300000,
        "input_size": 640,
        "pretrained_on": "COCO",
        "num_classes": 80,
        "size_mb": 96.9,
        "top1_accuracy": None,
        "top5_accuracy": None,
        "mAP": 46.9,
        "use_cases": ["Object detection", "High accuracy applications"],
        "pros": ["High accuracy", "Balanced performance"],
        "cons": ["Larger model size"],
        "recommended_for": ["Production systems", "Accuracy-critical tasks"]
    },

    "yolox-large": {
        "name": "YOLOX-L",
        "family": "YOLOX",
        "description": "Large YOLOX variant for object detection",
        "parameters": 54200000,
        "input_size": 640,
        "pretrained_on": "COCO",
        "num_classes": 80,
        "size_mb": 207.4,
        "top1_accuracy": None,
        "top5_accuracy": None,
        "mAP": 49.7,
        "use_cases": ["Object detection", "Maximum accuracy"],
        "pros": ["Very high accuracy"],
        "cons": ["Large and slow", "High compute requirements"],
        "recommended_for": ["Research", "High-end applications"]
    },

    "yolox-x-large": {
        "name": "YOLOX-X",
        "family": "YOLOX",
        "description": "Extra large YOLOX variant for object detection",
        "parameters": 99100000,
        "input_size": 640,
        "pretrained_on": "COCO",
        "num_classes": 80,
        "size_mb": 379.0,
        "top1_accuracy": None,
        "top5_accuracy": None,
        "mAP": 51.1,
        "use_cases": ["Object detection", "Benchmarking"],
        "pros": ["Highest accuracy in YOLOX family"],
        "cons": ["Very large and slow", "Requires significant compute"],
        "recommended_for": ["Research", "When maximum accuracy is needed"]
    }
}


def get_pretrained_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a pretrained model (case-insensitive)."""
    for key, value in PRETRAINED_MODELS.items():
        if key.lower() == model_name.lower():
            return value
    return {}


def list_pretrained_models_by_family(family: str) -> list:
    """List pretrained models, optionally filtered by family."""
    if family:
        return [
            name for name, info in PRETRAINED_MODELS.items()
            if info.get("family", "").lower() == family.lower()
        ]
    return list(PRETRAINED_MODELS.keys())


def get_recommended_models_for_task(
    task_type: str,
    constraint: Optional[str] 
) -> list:
    """
    Get recommended models for a task with optional constraints.
    
    Args:
        task_type: Type of task
        constraint: "mobile", "accuracy", "speed", "balanced"
        
    Returns:
        List of recommended model names
    """
    if constraint == "mobile":
        return ["mobilenet_v2", "mobilenet_v3_large", "efficientnet_b0"]
    elif constraint == "accuracy":
        return ["efficientnet_b3", "resnet152", "densenet161"]
    elif constraint == "speed":
        return ["resnet18", "mobilenet_v2", "efficientnet_b0"]
    elif constraint == "balanced":
        return ["resnet50", "efficientnet_b0", "mobilenet_v3_large"]
    else:
        # General recommendations
        return ["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v2"]


def compare_models(model_names: List[str]) -> Dict[str, Any]:
    """
    Compare multiple models.
    
    Returns comparison table with key metrics.
    """
    comparison = {
        "models": [],
        "parameters": [],
        "size_mb": [],
        "top1_accuracy": [],
        "recommended_for": []
    }
    
    for name in model_names:
        info = get_pretrained_model_info(name)
        if info:
            comparison["models"].append(info["name"])
            comparison["parameters"].append(info["parameters"])
            comparison["size_mb"].append(info["size_mb"])
            comparison["top1_accuracy"].append(info["top1_accuracy"])
            comparison["recommended_for"].append(info["recommended_for"])
    
    return comparison