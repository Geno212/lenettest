# src/logic/requirements_logic.py
"""Deterministic logic for requirements extraction and validation."""

from typing import Dict, Any, List
import re
from src.agentic.src.schemas.datasets import STANDARD_DATASETS
from src.agentic.src.schemas.layers import LAYER_SCHEMAS
from src.agentic.src.schemas.optimizers import OPTIMIZER_CONFIGS
from src.agentic.src.schemas.losses import LOSS_FUNCTIONS
import json
import logging
from langchain_core.messages import SystemMessage, AIMessage



class RequirementsAnalystLogic:
    """
    Deterministic logic for requirements analysis.
    
    This class handles:
    - Inference of missing fields from standard datasets
    - Validation of requirements
    - Merging specifications
    - Checking completeness
    - Generating targeted questions
    """
    def __init__(self):
        self.STANDARD_DATASETS = STANDARD_DATASETS
        self.LAYER_SCHEMAS = LAYER_SCHEMAS
        self.OPTIMIZER_CONFIGS = OPTIMIZER_CONFIGS
        self.LOSS_FUNCTIONS = LOSS_FUNCTIONS
    
    @staticmethod
    def normalize_input_shape(input_shape: Any) -> Dict[str, int]:
        """
        Normalize input_shape to dictionary format.
        
        Handles:
        - Dict: {"height": 28, "width": 28, "channels": 1}
        - String: "28×28×1" or "28x28x1" or "28,28,1"
        - Tuple/List: (28, 28, 1) or [28, 28, 1]
        
        Args:
            input_shape: Input shape in any format
            
        Returns:
            Dictionary with height, width, channels keys
        """
        if isinstance(input_shape, dict):
            # Already a dict, ensure it has the right keys
            return {
                "height": input_shape.get("height", 224),
                "width": input_shape.get("width", 224),
                "channels": input_shape.get("channels", 3)
            }
        
        if isinstance(input_shape, str):
            # Parse string format like "28×28×1" or "28x28x1" or "28,28,1"
            # Replace common separators with a standard one
            normalized = input_shape.replace('×', 'x').replace(',', 'x').replace(' ', '')
            parts = normalized.split('x')
            
            try:
                if len(parts) == 3:
                    h, w, c = [int(p.strip()) for p in parts]
                    return {"height": h, "width": w, "channels": c}
                elif len(parts) == 2:
                    # Assume grayscale if only 2 dimensions
                    h, w = [int(p.strip()) for p in parts]
                    return {"height": h, "width": w, "channels": 1}
            except (ValueError, IndexError):
                pass
        
        if isinstance(input_shape, (list, tuple)):
            # Convert list/tuple to dict
            try:
                if len(input_shape) == 3:
                    return {"height": int(input_shape[0]), "width": int(input_shape[1]), "channels": int(input_shape[2])}
                elif len(input_shape) == 2:
                    return {"height": int(input_shape[0]), "width": int(input_shape[1]), "channels": 1}
            except (ValueError, IndexError, TypeError):
                pass
        
        # Default fallback
        return {"height": 224, "width": 224, "channels": 3}
    
    def infer_missing_fields(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infer missing fields based on standard datasets.
        
        For standard datasets (MNIST, CIFAR10, etc), we know the input shapes
        and number of classes, so we can fill these automatically.
        
        Args:
            spec: Current specification
            
        Returns:
            Dictionary with inferred fields
        """
        inferred = {}
        
        dataset = spec.get("dataset", {})
        # Defensive: ensure dataset is a dict
        if not isinstance(dataset, dict):
            dataset = {}
        
        # Normalize input_shape if it exists but is not a dict
        if "input_shape" in dataset and not isinstance(dataset["input_shape"], dict):
            inferred.setdefault("dataset", {})["input_shape"] = self.normalize_input_shape(dataset["input_shape"])
        
        dataset_name = dataset.get("name", "").upper()
        
        # Check if standard dataset
        if dataset_name in self.STANDARD_DATASETS:
            standard = self.STANDARD_DATASETS[dataset_name]
            
            # Infer input shape
            if "input_shape" not in dataset:
                inferred.setdefault("dataset", {})["input_shape"] = standard["input_shape"]
            
            # Infer output classes
            if "output_classes" not in dataset and "output_classes" in standard:
                inferred.setdefault("dataset", {})["output_classes"] = standard["output_classes"]
            
            # Infer task type
            if "task_type" not in spec:
                inferred["task_type"] = standard["task_type"]
            

        
        return inferred
    
    def merge_specifications(
        self,
        current: Dict[str, Any],
        new: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge current and new specifications.
        
        Handles nested dictionaries properly.
        Priority: new > current
        
        Args:
            current: Existing specification
            new: New specification to merge
            
        Returns:
            Merged specification
        """
        merged = current.copy()
        
        for key, value in new.items():
            if key == "request":
                continue  # Skip generic request field
            
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                # Deep merge dictionaries
                merged[key] = {**merged[key], **value}
            else:
                # Direct assignment
                merged[key] = value
        
        return merged
    
    def check_completeness(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if specification has all REQUIRED fields.
        
        Required fields:
        - task_type
        - dataset.name
        - dataset.path
        - dataset.input_shape (for custom datasets)
        - dataset.output_classes (for classification with custom datasets)
        - target_metrics.accuracy
        - architecture.type
        - architecture.pretrained_model (if type="pretrained")
        - architecture.layers (if type="custom")
        
        Args:
            spec: Specification to check
            
        Returns:
            {
                "is_complete": bool,
                "missing": List[str]  # List of missing field names
            }
        """
        missing = []
        
        # Check task type
        if not spec.get("task_type"):
            missing.append("task_type")
        
        # Check dataset requirements
        dataset = spec.get('dataset', {})
        # Defensive: ensure dataset is a dict
        if not isinstance(dataset, dict):
            dataset = {}
        if not dataset.get('name'):
            missing.append('dataset.name')
            
        # Require path for all dataset types
        if not dataset.get('path'):
            missing.append('dataset.path')
        
        dataset_name = dataset.get("name", "").upper()
        is_standard = dataset_name in self.STANDARD_DATASETS
        
        # For custom datasets, need input_shape
        if not is_standard:
            if not dataset.get("input_shape"):
                missing.append("input_shape")
        
        # For classification tasks, need output_classes
        if spec.get("task_type") == "classification":
            if not is_standard and not dataset.get("output_classes"):
                missing.append("output_classes")
        
        # Check target metrics
        if not spec.get("target_metrics", {}).get("accuracy"):
            missing.append("target_accuracy")
        
        # Check architecture
        arch = spec.get("architecture", {})
        # Defensive: ensure arch is a dict
        if not isinstance(arch, dict):
            arch = {}
        if not arch.get("type"):
            missing.append("architecture_type")
        elif arch.get("type") == "pretrained":
            if not arch.get("pretrained_model"):
                missing.append("pretrained_model")
        elif arch.get("type") == "custom":
            if not arch.get("layers") or len(arch.get("layers", [])) == 0:
                missing.append("architecture_layers")
        
        return {
            "is_complete": len(missing) == 0,
            "missing": missing
        }
    
    def validate_specification(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate completed specification.
        
        Checks:
        - Target accuracy in valid range
        - No conflicts (pretrained + custom layers)
        - Dataset path exists (warning if not)
        - Training config values reasonable (if provided)
        
        Args:
            spec: Complete specification
            
        Returns:
            {
                "valid": bool,
                "errors": List[str],
                "warnings": List[str]
            }
        """
        errors = []
        warnings = []
        
        # Validate target accuracy
        target_metrics = spec.get("target_metrics", {})
        # Defensive: ensure target_metrics is a dict
        if not isinstance(target_metrics, dict):
            target_metrics = {}
        target_acc = target_metrics.get("accuracy")
        if target_acc is not None:
            if target_acc < 0 or target_acc > 1:
                errors.append("Target accuracy must be between 0 and 1")
            elif target_acc > 0.99:
                warnings.append("Target accuracy > 99% is very ambitious and may be difficult to achieve")
        
        # Validate architecture conflicts
        arch = spec.get("architecture", {})
        # Defensive: ensure arch is a dict
        if not isinstance(arch, dict):
            arch = {}
        if arch.get("pretrained_model") and arch.get("layers"):
            errors.append(
                "Cannot specify both pretrained model and custom layers. "
                "Please choose one approach: pretrained OR custom."
            )
        
        # Validate dataset path
        dataset = spec.get("dataset", {})
        # Defensive: ensure dataset is a dict
        if not isinstance(dataset, dict):
            dataset = {}
        if dataset.get("path"):
            import os
            path = os.path.expanduser(dataset["path"])
            if not os.path.exists(path):
                warnings.append(
                    f"Dataset path does not exist: {dataset['path']}. "
                    f"Make sure to create it before training."
                )
        
        # Validate training config (if provided)
        training = spec.get("training_config", {})
        # Defensive: ensure training is a dict
        if not isinstance(training, dict):
            training = {}
        if training:
            # Epochs
            epochs = training.get("epochs")
            if epochs is not None:
                if epochs <= 0:
                    errors.append("Epochs must be positive")
                elif epochs > 500:
                    warnings.append(f"Training for {epochs} epochs may take a very long time")
            
            # Batch size
            batch_size = training.get("batch_size")
            if batch_size is not None:
                if batch_size <= 0:
                    errors.append("Batch size must be positive")
                elif batch_size > 512:
                    warnings.append(f"Batch size {batch_size} is very large and may cause memory issues")
            
            # Learning rate
            optimizer = training.get("optimizer", {})
            # Defensive: ensure optimizer is a dict
            if not isinstance(optimizer, dict):
                optimizer = {}
            lr = optimizer.get("lr")
            if lr is not None:
                if lr <= 0:
                    errors.append("Learning rate must be positive")
                elif lr > 1:
                    warnings.append(f"Learning rate {lr} is very high and may cause training instability")
                elif lr < 1e-6:
                    warnings.append(f"Learning rate {lr} is very low and may cause slow convergence")
        
        # Validate input shape
        input_shape = dataset.get("input_shape", {})
        # Defensive: normalize input_shape if it's not a dict
        if not isinstance(input_shape, dict):
            input_shape = self.normalize_input_shape(input_shape)
        if input_shape:
            h, w, c = input_shape.get("height"), input_shape.get("width"), input_shape.get("channels")
            if any(v is None or v <= 0 for v in [h, w, c]):
                errors.append("Input shape dimensions must be positive integers")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def generate_questions(
        self,
        missing: List[str],
        current_spec: Dict[str, Any],
        available_options: Dict[str, Any],
        llm: Any
    ) -> str:
        """
        Generate natural language questions for missing fields.

        If an LLM client is provided (llm), use few-shot prompting to produce clear,
        numbered, and example-rich questions. If no llm is provided, fall back to
        the original deterministic question builder.

        Args:
            missing: List of missing field names
            current_spec: Current specification (for context)
            available_options: Available datasets, models, optimizers from MCP
            llm: Optional LLM client with async `ainvoke(messages)` API

        Returns:
            Formatted question string (as produced by the LLM or deterministic fallback)
        """

        # Normalize missing keys (e.g., 'dataset.name' -> 'dataset_name')
        normalized = [m.replace('.', '_') for m in missing]

        if llm is None:
            raise ValueError("llm client is required for generate_questions")
        else:


            logger = logging.getLogger(__name__)

            system_prompt = (
                "You are an assistant that asks the user concise, unambiguous, and numbered "
                "follow-up questions to collect missing configuration fields needed to create and train "
                "a neural network project. For each missing item, ask for the exact value, provide an "
                "example or acceptable formats, and show available options when relevant. Keep questions "
                "short (1-3 sentences) and include an example for each field. Group related fields in the "
                "same numbered question if it improves clarity."
            )

            few_shot_examples = []
            few_shot_examples.append(
                {
                    "missing": ["task_type", "dataset.path"],
                    "current_spec": {"dataset": {"name": "CUSTOM_DS"}},
                    "questions": (
                        "1) Task type: What is the task for this dataset? Example responses: `classification`, "
                        "`detection`, `segmentation`.\n2) Dataset path: Where is `CUSTOM_DS` located on your system? "
                        "Provide an absolute or home path, e.g. `C:/data/CUSTOM_DS` or `~/datasets/custom_ds`."
                    )
                }
            )

            few_shot_examples.append(
                {
                    "missing": ["input_shape", "output_classes"],
                    "current_spec": {"dataset": {"name": "MyImages"}},
                    "questions": (
                        "1) Input shape: What are the image dimensions? Provide as `height×width×channels`, "
                        "e.g. `224×224×3` or `28×28×1`.\n2) Number of classes: How many output classes? "
                        "Give an integer, e.g. `10` or `2`."
                    )
                }
            )

            prompt_parts = [
                f"Missing fields: {json.dumps(normalized)}",
                "Current spec (for context):",
                json.dumps(current_spec, indent=2),
                "Available options (show first few where applicable):",
                json.dumps({k: available_options.get(k, [])[:10] for k in ['datasets', 'pretrained_models', 'optimizers', 'loss_functions', 'schedulers'] if k in available_options}, indent=2),
                "\nPlease produce a user-facing message that asks for the missing fields. Number the questions and include short examples or acceptable formats."
            ]

            prompt_parts.append("\nFew-shot examples (input -> desired output style):")
            for ex in few_shot_examples:
                prompt_parts.append(f"Input missing: {ex['missing']}\nCurrent spec: {json.dumps(ex['current_spec'])}\nDesired questions:\n{ex['questions']}\n")

            user_prompt = "\n\n".join(prompt_parts)

            messages = []
            messages.append(SystemMessage(content=system_prompt))
            messages.append(AIMessage(content=user_prompt))

            # If the LLM call fails, propagate the exception — callers must provide a working LLM.
            response = await llm.ainvoke(messages)
            content = getattr(response, 'content', response)
            if not isinstance(content, str):
                content = str(content)
            return content
        # If execution reaches here something went wrong; raise to notify caller
        raise RuntimeError("generate_questions: failed to generate questions with provided llm")