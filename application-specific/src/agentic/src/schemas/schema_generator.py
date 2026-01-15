"""
Dynamic Pydantic schema generator for explicit layer parameter extraction.

This module provides:
- Explicit Pydantic models for core PyTorch layer types (Conv2d, Linear, etc.)
- A function to dynamically build a Pydantic model for only the relevant layer types
- Utilities for extracting all parameters for each layer type
"""

from typing import Dict, Optional, Type
from pydantic import BaseModel, Field, create_model

# Explicit schemas for core layer types (expand as needed)

class Conv2dParams(BaseModel):
    out_channels: Optional[int] = Field(None, description="Number of output channels")
    kernel_size: Optional[int]  = Field(None, description="Size of the convolving kernel")
    stride: int  = Field(1, description="Stride of the convolution")
    padding: int  = Field(0, description="Zero-padding added to both sides")
    groups: int = Field(1, description="Number of blocked connections")
    bias: bool = Field(True, description="If True, adds a learnable bias")

class LinearParams(BaseModel):
    out_features: Optional[int] = Field(None, description="Size of each output sample")
    bias: bool = Field(True, description="If True, adds a learnable bias")

class MaxPool2dParams(BaseModel):
    kernel_size: Optional[int]  = Field(None, description="Size of the window")
    stride: int  = Field(1, description="Stride of the window")
    padding: int  = Field(0, description="Implicit zero padding")
    return_indices: bool = Field(False, description="If True, will return the max indices")
    ceil_mode: bool = Field(False, description="When True, will use ceil instead of floor")

class ReLUParams(BaseModel):
    inplace: bool = Field(False, description="Can optionally do the operation in-place")

class BatchNorm2dParams(BaseModel):
    num_features: Optional[int] = Field(None, description="Number of features")
    eps: float = Field(1e-5, description="A value added to the denominator for numerical stability")
    momentum: float = Field(0.1, description="Value used for the running_mean and running_var computation")
    affine: bool = Field(True, description="If True, this module has learnable affine parameters")
    track_running_stats: bool = Field(True, description="If True, this module tracks running mean and var")

class DropoutParams(BaseModel):
    p: float = Field(0.5, description="Probability of an element to be zeroed")
    inplace: bool = Field(False, description="If set to True, will do this operation in-place")

class FlattenParams(BaseModel):
    start_dim: int = Field(1, description="First dim to flatten")
    end_dim: int = Field(-1, description="Last dim to flatten")

class AdaptiveAvgPool2dParams(BaseModel):
    output_size: Optional[int]  = Field(None, description="Target output size")

# Map layer type names to their explicit param schemas
LAYER_TYPE_TO_SCHEMA: Dict[str, Type[BaseModel]] = {
    "Conv2d": Conv2dParams,
    "Linear": LinearParams,
    "MaxPool2d": MaxPool2dParams,
    "ReLU": ReLUParams,
    "BatchNorm2d": BatchNorm2dParams,
    "Dropout": DropoutParams,
    "Flatten": FlattenParams,
    "AdaptiveAvgPool2d": AdaptiveAvgPool2dParams,
}

def build_explicit_layer_schema(layer_types: list[str]) -> Type[BaseModel]:
    """
    Dynamically build a Pydantic model with explicit fields for each relevant layer type.
    Each field is a list of the corresponding explicit param schema.
    """
    fields = {}
    for layer_type in layer_types:
        schema = LAYER_TYPE_TO_SCHEMA.get(layer_type)
        if schema:
            fields[layer_type] = (list[schema], Field(default_factory=list))
    if not fields:
        raise ValueError("No valid layer types provided for explicit schema generation.")
    return create_model(
        "ExplicitManualLayersExtraction",
        **fields,
        __base__=BaseModel
    )
