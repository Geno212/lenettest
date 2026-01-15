# src/logic/architecture_logic.py
"""Deterministic logic for architecture design and validation."""

from typing import Dict, Any, List, Optional, Tuple
from src.agentic.src.schemas.layers import LAYER_SCHEMAS
from src.agentic.src.schemas.pretrained_models import PRETRAINED_MODELS
import copy


class ArchitectureDesignerLogic:
    """
    Deterministic logic for architecture operations.
    
    Handles:
    - Layer specification validation
    - Dimension inference through layer stack
    - Architecture flow validation
    - Parameter completion
    """
    
    def __init__(self):
        self.layer_schemas = LAYER_SCHEMAS
        self.pretrained_models = PRETRAINED_MODELS
        
    
    # ==================== PRETRAINED MODEL METHODS ====================
    
    def is_valid_pretrained_model(self, model_name: str) -> bool:
        """Check if pretrained model name is valid."""
        return model_name.lower() in [m.lower() for m in self.pretrained_models.keys()]
    
    def get_available_pretrained_models(self) -> List[str]:
        """Get list of available pretrained models."""
        return list(self.pretrained_models.keys())
    
    def get_pretrained_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a pretrained model."""
        for key, value in self.pretrained_models.items():
            if key.lower() == model_name.lower():
                return value
        return {}
    
    # ==================== LAYER VALIDATION METHODS ====================
    
    def validate_layer_specification(self, layer_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single layer specification.
        
        Args:
            layer_spec: {"type": "Conv2d", "out_channels": 64, ...}
            
        Returns:
            {
                "valid": bool,
                "missing_required": List[str],
                "complete_params": Dict[str, Any],
                "suggested_defaults": Dict[str, Any]
            }
        """
        layer_type = layer_spec.get("type")
        
        if not layer_type or layer_type not in self.layer_schemas:
            return {
                "valid": False,
                "error": f"Unknown layer type: {layer_type}",
                "missing_required": [],
                "complete_params": {},
                "suggested_defaults": {}
            }
        
        schema = self.layer_schemas[layer_type]
        required = schema.get("required", [])
        optional = schema.get("optional", [])
        defaults = schema.get("defaults", {})
        
        # Check required parameters
        missing_required = []
        for param in required:
            if param not in layer_spec:
                missing_required.append(param)
        
        # Build complete parameters (provided + defaults)
        complete_params = defaults.copy()
        for key, value in layer_spec.items():
            if key != "type":
                complete_params[key] = value
        
        # Generate suggested defaults for missing required
        suggested_defaults = {}
        for param in missing_required:
            suggested_defaults[param] = self._suggest_param_value(layer_type, param)
        
        return {
            "valid": len(missing_required) == 0,
            "missing_required": missing_required,
            "complete_params": complete_params,
            "suggested_defaults": suggested_defaults
        }
    
    def _suggest_param_value(self, layer_type: str, param_name: str) -> Any:
        """Suggest a reasonable default value for a parameter."""
        # Prefer schema-provided defaults when available
        schema = self.layer_schemas.get(layer_type, {})
        defaults = schema.get("defaults", {})

        if param_name in defaults:
            return defaults[param_name]

        # Generic sensible fallbacks for commonly required params across many layer types
        generic = {
            # channel/feature sizes
            "out_channels": 64,
            "in_channels": None,  # typically inferred from previous layer/input
            "out_features": 512,
            "in_features": None,  # inferred from flatten
            "num_features": 64,

            # pooling/conv params
            "kernel_size": 3,
            "stride": 1,
            "padding": 0,
            "dilation": 1,

            # transposed conv
            "output_padding": 0,

            # adaptive/upsample
            "output_size": 1,
            "upscale_factor": 2,

            # embedding
            "num_embeddings": 1000,
            "embedding_dim": 128,

            # dropout
            "p": 0.5,

            # activation / inplace flags
            "inplace": False,

            # others
            "bias": True,
        }

        # If schema explicitly lists the param as required but has no default, try reasonable guess
        if param_name in generic:
            return generic[param_name]

        # As a last resort, return None to indicate no suggestion available
        return None
    
    def complete_layer_params(self, layer_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Complete layer parameters with defaults."""
        validation = self.validate_layer_specification(layer_spec)
        return validation["complete_params"]
    
    # ==================== DIMENSION INFERENCE METHODS ====================
    
    def infer_layer_dimensions(
        self,
        layers: List[Dict[str, Any]],
        input_shape: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """
        Infer dimensions through layer stack.
        
        Args:
            layers: List of layer specifications
            input_shape: {"height": 28, "width": 28, "channels": 1}
            
        Returns:
            List of layers annotated with input_shape and output_shape
            
        Raises:
            ValueError: If dimensions cannot be inferred
        """
        annotated_layers = []
        current_shape = input_shape.copy()
        is_spatial = True  # Track if we're in spatial (Conv) or linear (Dense) domain
        
        for idx, layer in enumerate(layers):
            layer_type = layer.get("type")
            params = layer.copy()
            
            # Add input shape
            params["input_shape"] = current_shape.copy()
            
            # Calculate output shape based on layer type
            try:
                if layer_type == "Conv2d":
                    current_shape = self._infer_conv2d_output(current_shape, params, idx)
                    is_spatial = True
                
                elif layer_type == "MaxPool2d" or layer_type == "AvgPool2d":
                    current_shape = self._infer_pool2d_output(current_shape, params)
                    is_spatial = True
                
                elif layer_type == "Flatten":
                    current_shape = self._infer_flatten_output(current_shape)
                    is_spatial = False
                
                elif layer_type == "Linear":
                    if is_spatial:
                        raise ValueError(
                            f"Cannot connect Linear layer at position {idx} directly to spatial layers. "
                            f"Add a Flatten layer first."
                        )
                    current_shape = self._infer_linear_output(current_shape, params, idx)
                
                elif layer_type == "BatchNorm2d":
                    # BatchNorm doesn't change dimensions
                    if not is_spatial:
                        raise ValueError(f"BatchNorm2d at position {idx} requires spatial input")
                
                elif layer_type in ["ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Dropout", "Softmax"]:
                    # Activation and dropout don't change dimensions
                    pass
                
                elif layer_type == "AdaptiveAvgPool2d":
                    output_size = params.get("output_size", 1)
                    # Normalize output_size to a pair of ints
                    output_size = self._as_pair(output_size)
                    current_shape = {
                        "height": output_size[0],
                        "width": output_size[1],
                        "channels": current_shape["channels"]
                    }
                    is_spatial = True
                
                else:
                    # Unknown layer - pass through dimensions
                    pass
                
                params["output_shape"] = current_shape.copy()
                annotated_layers.append(params)
                
            except Exception as e:
                raise ValueError(f"Dimension inference failed at layer {idx} ({layer_type}): {str(e)}")
        
        return annotated_layers
    
    # -------------------- helpers --------------------
    def _as_pair(self, value: Any) -> Tuple[int, int]:
        """Normalize int/float/sequence to a length-2 tuple of ints.

        Accepts:
        - int or float -> (int(v), int(v))
        - list/tuple of length 1 -> (int(v[0]), int(v[0]))
        - list/tuple of length 2 -> (int(v[0]), int(v[1]))
        Raises ValueError for other shapes/types.
        """
        # int or float
        if isinstance(value, (int, float)):
            iv = int(round(value))
            return (iv, iv)
        # list/tuple
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                iv = int(round(value[0]))
                return (iv, iv)
            if len(value) == 2:
                return (int(round(value[0])), int(round(value[1])))
            raise ValueError(f"Expected pair or scalar, got sequence of length {len(value)}")
        raise ValueError(f"Expected int/float or pair, got {type(value).__name__}")

    def _to_int(self, value: Any, name: str = "value") -> int:
        """Coerce numeric to int; raise for unsupported types."""
        if isinstance(value, (int, float)):
            return int(round(value))
        raise ValueError(f"Expected numeric for {name}, got {type(value).__name__}")

    def _infer_conv2d_output(
        self,
        input_shape: Dict[str, int],
        params: Dict[str, Any],
        layer_idx: int
    ) -> Dict[str, int]:
        """Calculate Conv2d output dimensions."""
        h_in = input_shape.get("height")
        w_in = input_shape.get("width")
        c_in = input_shape.get("channels")
        
        if h_in is None or w_in is None or c_in is None:
            raise ValueError("Input shape must have height, width, and channels")
        # Coerce to ints in case floats slipped through
        h_in, w_in, c_in = int(h_in), int(w_in), int(c_in)
        
        # Get parameters
        out_channels = params.get("out_channels")
        if out_channels is None:
            raise ValueError(f"Conv2d at layer {layer_idx} missing required parameter 'out_channels'")
        out_channels = self._to_int(out_channels, "out_channels")
        
        kernel_size = self._as_pair(params.get("kernel_size", 3))
        stride = self._as_pair(params.get("stride", 1))
        padding = self._as_pair(params.get("padding", 0))
        dilation = self._as_pair(params.get("dilation", 1))
        
        # Calculate output dimensions
        h_out = int((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        w_out = int((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
        
        if h_out <= 0 or w_out <= 0:
            raise ValueError(
                f"Conv2d output dimensions invalid: {h_out}×{w_out}. "
                f"Input too small or kernel/stride too large."
            )
        
        return {
            "height": h_out,
            "width": w_out,
            "channels": out_channels
        }
    
    def _infer_pool2d_output(
        self,
        input_shape: Dict[str, int],
        params: Dict[str, Any]
    ) -> Dict[str, int]:
        """Calculate pooling output dimensions."""
        h_in = int(input_shape["height"])  # ensure ints
        w_in = int(input_shape["width"])   
        c_in = int(input_shape["channels"])
        
        kernel_size = self._as_pair(params.get("kernel_size", 2))
        # Default stride = kernel_size
        stride = self._as_pair(params.get("stride", kernel_size))
        padding = self._as_pair(params.get("padding", 0))
        
        h_out = int((h_in + 2 * padding[0] - kernel_size[0]) / stride[0] + 1)
        w_out = int((w_in + 2 * padding[1] - kernel_size[1]) / stride[1] + 1)
        
        if h_out <= 0 or w_out <= 0:
            raise ValueError(f"Pooling output dimensions invalid: {h_out}×{w_out}")
        
        return {
            "height": h_out,
            "width": w_out,
            "channels": c_in
        }
    
    def _infer_flatten_output(self, input_shape: Dict[str, int]) -> Dict[str, int]:
        """Calculate Flatten output dimensions."""
        h = input_shape.get("height")
        w = input_shape.get("width")
        c = input_shape.get("channels")
        
        if h and w and c:
            # Spatial to linear
            features = h * w * c
            return {"features": features}
        elif input_shape.get("features"):
            # Already flattened
            return input_shape.copy()
        else:
            raise ValueError("Cannot flatten: invalid input shape")
    
    def _infer_linear_output(
        self,
        input_shape: Dict[str, int],
        params: Dict[str, Any],
        layer_idx: int
    ) -> Dict[str, int]:
        """Calculate Linear layer output dimensions."""
        in_features = input_shape.get("features")
        
        if in_features is None:
            raise ValueError(f"Linear layer at {layer_idx} requires flattened input with 'features' dimension")
        
        out_features = params.get("out_features")
        if out_features is None:
            raise ValueError(f"Linear layer at {layer_idx} missing required parameter 'out_features'")
        
        return {"features": out_features}
    
    # ==================== VALIDATION METHODS ====================
    
    def validate_architecture_flow(
        self,
        annotated_layers: List[Dict[str, Any]],
        task_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that architecture flow is correct.
        
        Checks:
        - Spatial to linear transitions have Flatten
        - Final output matches task requirements
        - No dimension mismatches
        
        Args:
            annotated_layers: Layers with inferred dimensions
            task_spec: Task specification
            
        Returns:
            {
                "valid": bool,
                "errors": List[str],
                "warnings": List[str]
            }
        """
        errors = []
        warnings = []
        
        if not annotated_layers:
            errors.append("Architecture has no layers")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        # Check for Flatten before first Linear
        found_spatial = False
        found_flatten_before_linear = False
        
        for idx, layer in enumerate(annotated_layers):
            layer_type = layer.get("type")
            
            if layer_type in ["Conv2d", "MaxPool2d", "AvgPool2d"]:
                found_spatial = True
            
            if layer_type == "Flatten":
                found_flatten_before_linear = True
            
            if layer_type == "Linear" and found_spatial and not found_flatten_before_linear:
                errors.append(
                    f"Layer {idx} (Linear) follows spatial layers without Flatten. "
                    f"Add Flatten layer before first Linear layer."
                )
        
        # Check output layer
        last_layer = annotated_layers[-1]
        output_shape = last_layer.get("output_shape", {})
        
        task_type = task_spec.get("task_type")
        num_classes = task_spec.get("dataset", {}).get("output_classes")
        
        if task_type == "classification" and num_classes:
            output_features = output_shape.get("features")
            
            if output_features and output_features != num_classes:
                warnings.append(
                    f"Output layer has {output_features} features but task has {num_classes} classes. "
                    f"Consider adjusting final Linear layer to Linear(out_features={num_classes})."
                )
        
        # Check for activation functions
        has_activations = any(
            layer.get("type") in ["ReLU", "LeakyReLU", "Sigmoid", "Tanh"]
            for layer in annotated_layers
        )
        
        if not has_activations:
            warnings.append(
                "No activation functions detected. Consider adding ReLU after Conv2d layers."
            )
        
        # Check for regularization
        has_dropout = any(layer.get("type") == "Dropout" for layer in annotated_layers)
        has_batchnorm = any(layer.get("type") in ["BatchNorm2d", "BatchNorm1d"] for layer in annotated_layers)
        
        if not (has_dropout or has_batchnorm):
            warnings.append(
                "No regularization detected. Consider adding Dropout or BatchNorm for better generalization."
            )
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def format_dimension_flow(self, annotated_layers: List[Dict[str, Any]]) -> str:
        """
        Format dimension flow through layers for display.
        
        Args:
            annotated_layers: Layers with dimensions
            
        Returns:
            Formatted string showing dimension flow
        """
        lines = []
        
        for idx, layer in enumerate(annotated_layers):
            layer_type = layer.get("type")
            output_shape = layer.get("output_shape", {})
            
            # Format shape
            if "height" in output_shape:
                shape_str = f"({output_shape['height']}×{output_shape['width']}×{output_shape['channels']})"
            elif "features" in output_shape:
                shape_str = f"({output_shape['features']})"
            else:
                shape_str = "(?)"
            
            lines.append(f"{idx:2d}. {layer_type:20s} → {shape_str}")
        
        return "\n".join(lines)
    
    # ==================== MISSING PARAMS VALIDATION ====================
    
    def get_missing_required_params(self, layer_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Check which required params are missing for a layer.
        
        Args:
            layer_spec: Layer specification with 'layer_type' and 'params'
        
        Returns:
            Dict with:
                - missing: List of missing required param names
                - layer_type: The layer type
                - layer_index: Optional index if provided
        """
        layer_type = layer_spec.get("layer_type")
        if not layer_type or layer_type not in self.layer_schemas:
            return {
                "layer_type": layer_type or "unknown",
                "missing": [],
                "error": f"Unknown layer type: {layer_type}"
            }
        
        schema = self.layer_schemas[layer_type]
        required_params = schema.get("required", [])
        provided_params = layer_spec.get("params", {})
        
        missing = [param for param in required_params if param not in provided_params]
        
        return {
            "layer_type": layer_type,
            "missing": missing,
            "position": layer_spec.get("position"),
        }