# src/logic/configuration_logic.py
"""Deterministic logic for training configuration."""

from typing import Dict, Any, List
import copy
from src.agentic.src.schemas.optimizers import OPTIMIZER_CONFIGS, get_optimizer_defaults, get_recommended_lr
from src.agentic.src.schemas.losses import LOSS_FUNCTIONS, get_default_loss_for_task, validate_loss_for_task


class ConfigurationSpecialistLogic:
    """
    Deterministic logic for configuration operations.
    
    Handles:
    - Merging configurations from multiple sources
    - Applying intelligent defaults
    - Validation of parameters
    - Context-aware recommendations
    """
    
    def __init__(self):
        self.optimizer_configs = OPTIMIZER_CONFIGS
        self.loss_functions = LOSS_FUNCTIONS
    
    # ==================== MERGING METHODS ====================
    
    def merge_params(
        self,
        preferences: Dict[str, Any],
        received: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge training preferences with received parameters.
        
        Priority: received > preferences
        
        Args:
            preferences: From task_specification.training_config
            received: From ToConfigurationSpecialist tool call
            
        Returns:
            Merged dictionary
        """
        merged = preferences.copy()
        
        # Merge model_params
        if "model_params" in received:
            for key, value in received["model_params"].items():
                if key=="device" and value=="gpu":
                    merged[key] = "cuda:0"
                else:
                    merged[key] = value
        
        # Merge optimizer_info
        if "optimizer_info" in received:
            opt = received["optimizer_info"]
            if isinstance(opt, str):
                merged["optimizer"] = {"type": opt}
            elif isinstance(opt, dict):
                # Flatten params if nested
                if "params" in opt and isinstance(opt["params"], dict):
                    # Merge type and params at same level
                    merged["optimizer"] = {"type": opt.get("type")}
                    merged["optimizer"].update(opt["params"])
                else:
                    merged["optimizer"] = opt
        
        # Merge loss_info
        if "loss_info" in received:
            loss = received["loss_info"]
            if isinstance(loss, str):
                merged["loss_function"] = {"type": loss}
            elif isinstance(loss, dict):
                merged["loss_function"] = loss
        
        # Merge scheduler_info
        if "scheduler_info" in received:
            merged["scheduler"] = received["scheduler_info"]
        
        return merged
    
    def complete_config(
        self,
        defaults: Dict[str, Any],
        provided: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Complete configuration with defaults.
        
        Priority: provided > defaults
        
        Args:
            defaults: Default configuration
            provided: User-provided configuration
            
        Returns:
            Complete configuration
        """
        # Start from a deep copy of defaults to avoid mutating caller data.
        complete = copy.deepcopy(defaults)

        def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively merge override into base and return a new dict.

            Rules:
            - Keys present in override take precedence.
            - If both values are dicts, merge recursively.
            - None in override is treated as an explicit override (i.e. sets value to None).
            """
            result = copy.deepcopy(base)
            for k, v in (override or {}).items():
                if v is None:
                    # Explicitly set to None
                    result[k] = None
                elif k in result and isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = _deep_merge(result[k], v)
                else:
                    result[k] = copy.deepcopy(v)
            return result

        if provided:
            complete = _deep_merge(complete, provided)

        return complete
    
    # ==================== DEFAULTS METHODS ====================
    
    def get_default_config(
        self,
        task_type: str,
        dataset_name: str,
        architecture_type: str,
        is_pretrained: bool = False
    ) -> Dict[str, Any]:
        """
        Get intelligent default configuration based on context.
        
        Args:
            task_type: Type of task
            dataset_name: Dataset name
            architecture_type: "pretrained" or "custom"
            is_pretrained: Whether using pretrained model
            
        Returns:
            Default configuration dictionary
        """
        # Base defaults
        config = {
            "epochs": 50,
            "batch_size": 32,
            "device": "cpu"
        }
        
        # Adjust for pretrained models (fine-tuning)
        if is_pretrained:
            config["epochs"] = 20  # Fewer epochs for fine-tuning
            config["optimizer"] = {
                "type": "Adam",
                "lr": 0.0001,  # Lower LR for fine-tuning
                "weight_decay": 0.0001
            }
        else:
            # Training from scratch
            config["optimizer"] = {
                "type": "Adam",
                "lr": 0.001,
                "weight_decay": 0.0
            }
        
        # Set loss function based on task
        loss_type = get_default_loss_for_task(task_type)
        config["loss_function"] = {"type": loss_type, "params": {}}
        
        # Dataset-specific adjustments
        if dataset_name.upper() == "MNIST":
            config["batch_size"] = 64  # MNIST is small, can use larger batches
            config["epochs"] = 10  # MNIST trains quickly
        elif dataset_name.upper() in ["CIFAR10", "CIFAR100"]:
            config["batch_size"] = 64
            config["epochs"] = 100 if not is_pretrained else 30
            # Add scheduler for longer training
            if not is_pretrained:
                config["scheduler"] = {
                    "type": "StepLR",
                    "step_size": 30,
                    "gamma": 0.1
                }
        elif dataset_name.upper() == "IMAGENET":
            config["batch_size"] = 128  # Large dataset
            config["epochs"] = 90 if not is_pretrained else 20
            if not is_pretrained:
                config["optimizer"]["type"] = "SGD"
                config["optimizer"]["lr"] = 0.01
                config["optimizer"]["momentum"] = 0.9
                config["scheduler"] = {
                    "type": "StepLR",
                    "step_size": 30,
                    "gamma": 0.1
                }
        
        return config
    
    def recommend_hyperparameters(
        self,
        architecture_type: str,
        dataset_size: str,
        task_type: str,
        is_pretrained: bool = False
    ) -> Dict[str, Any]:
        """
        Recommend hyperparameters based on context.
        
        Args:
            architecture_type: "pretrained" or "custom"
            dataset_size: "small" (<10k), "medium" (10k-100k), "large" (>100k)
            task_type: Type of task
            is_pretrained: Whether using pretrained model
            
        Returns:
            Recommendations with reasoning
        """
        recommendations = {}
        
        # Optimizer recommendations
        if is_pretrained:
            recommendations["optimizer"] = {
                "type": "Adam",
                "lr": 0.0001,
                "reasoning": "Lower learning rate to preserve pretrained weights"
            }
        elif dataset_size == "large":
            recommendations["optimizer"] = {
                "type": "SGD",
                "lr": 0.01,
                "momentum": 0.9,
                "reasoning": "SGD with momentum works better for large datasets"
            }
        else:
            recommendations["optimizer"] = {
                "type": "Adam",
                "lr": 0.001,
                "reasoning": "Adam is robust for most tasks"
            }
        
        # Batch size recommendations
        if dataset_size == "small":
            recommendations["batch_size"] = {
                "value": 16,
                "reasoning": "Smaller batches for small datasets to avoid overfitting"
            }
        elif dataset_size == "large":
            recommendations["batch_size"] = {
                "value": 128,
                "reasoning": "Larger batches for efficiency on large datasets"
            }
        else:
            recommendations["batch_size"] = {
                "value": 32,
                "reasoning": "Balanced batch size for medium datasets"
            }
        
        # Epochs recommendations
        if is_pretrained:
            recommendations["epochs"] = {
                "value": 20,
                "reasoning": "Fewer epochs needed for fine-tuning"
            }
        elif dataset_size == "small":
            recommendations["epochs"] = {
                "value": 100,
                "reasoning": "More epochs with small data to ensure convergence"
            }
        else:
            recommendations["epochs"] = {
                "value": 50,
                "reasoning": "Standard training duration"
            }
        
        # Regularization recommendations
        if dataset_size == "small":
            recommendations["regularization"] = {
                "weight_decay": 0.001,
                "reasoning": "Higher regularization to prevent overfitting on small datasets"
            }
        
        return recommendations
    
    # ==================== VALIDATION METHODS ====================
    
    def validate_config(
        self,
        config: Dict[str, Any],
        task_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate configuration completeness and reasonableness.
        
        Args:
            config: Configuration to validate
            task_spec: Task specification for context
            
        Returns:
            {
                "valid": bool,
                "errors": List[str],
                "warnings": List[str]
            }
        """
        errors = []
        warnings = []
        
        # Check required fields
        if "epochs" not in config or config["epochs"] is None:
            errors.append("Number of epochs not specified")
        elif config["epochs"] <= 0:
            errors.append("Epochs must be positive")
        elif config["epochs"] > 500:
            warnings.append(f"Training for {config['epochs']} epochs may take a very long time")
        
        if "batch_size" not in config or config["batch_size"] is None:
            errors.append("Batch size not specified")
        elif config["batch_size"] <= 0:
            errors.append("Batch size must be positive")
        elif config["batch_size"] > 512:
            warnings.append(f"Batch size {config['batch_size']} is very large and may cause memory issues")
        elif config["batch_size"] < 8:
            warnings.append(f"Batch size {config['batch_size']} is very small and may cause unstable training")
        
        # Validate optimizer
        if "optimizer" not in config or not config["optimizer"]:
            errors.append("Optimizer not specified")
        else:
            opt = config["optimizer"]
            if "type" not in opt:
                errors.append("Optimizer type not specified")
            else:
                # Validate learning rate
                lr = opt.get("lr")
                if lr is not None:
                    if lr <= 0:
                        errors.append("Learning rate must be positive")
                    elif lr > 1.0:
                        warnings.append(f"Learning rate {lr} is very high and may cause training instability")
                    elif lr < 1e-6:
                        warnings.append(f"Learning rate {lr} is very low and may cause extremely slow training")
                    elif lr > 0.1:
                        warnings.append(f"Learning rate {lr} is high. Typical values: 0.001 (Adam) or 0.01 (SGD)")
                
                # Validate weight decay
                wd = opt.get("weight_decay")
                if wd is not None and wd > 0.1:
                    warnings.append(f"Weight decay {wd} is very high and may over-regularize")
        
        # Validate loss function
        if "loss_function" not in config or not config["loss_function"]:
            errors.append("Loss function not specified")
        else:
            loss = config["loss_function"]
            if "type" not in loss:
                errors.append("Loss function type not specified")
            else:
                # Validate loss matches task
                task_type = task_spec.get("task_type")
                loss_validation = validate_loss_for_task(loss["type"], task_type)
                if not loss_validation["valid"]:
                    errors.append(loss_validation.get("warning", "Loss function incompatible with task"))
        
        # Validate dataset path
        if not config.get("dataset_path"):
            warnings.append("Dataset path not specified - make sure dataset is available before training")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }