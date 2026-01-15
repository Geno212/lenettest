"""
Dynamic model builder and conversion utility for explicit layer extraction.

This module provides:
- Utility to build a dynamic extraction model for only the relevant layer types
- Conversion from explicit schema instances to ManualLayerConfig format
"""
from typing import List, Dict, Any, Type
import logging
from pydantic import BaseModel, Field
from src.agentic.src.core.state import ManualLayerConfig
from src.agentic.src.schemas.schema_generator import LAYER_TYPE_TO_SCHEMA



def build_dynamic_extraction_model(layer_types: List[str]) -> Type[BaseModel]:
    """
    Build a dynamic Pydantic model for extracting parameters for the given layer types.
    Each field is named after the layer type and uses the explicit param model.
    """
    from pydantic import create_model
    fields = {}
    for lt in layer_types:
        param_model = LAYER_TYPE_TO_SCHEMA.get(lt)
        if param_model:
            # Request a list of param objects for each layer type so multiple
            # occurrences are represented as ordered arrays in the LLM output.
            fields[lt] = (list[param_model], Field(default_factory=list))
    if not fields:
        return create_model("EmptyExtraction")
    return create_model("DynamicLayerExtraction", **fields)


def explicit_to_manual_layer_configs(
    extracted: Dict[str, Any],
    partial_layers: List[ManualLayerConfig]
) -> List[ManualLayerConfig]:
    """
    Convert explicit-schema extraction output to a list of ManualLayerConfig instances.

    Implementation notes:
    - `extracted` is expected to map each layer type to an ordered list of
      parameter objects (one per occurrence). We transform these into per-layer
      configs by consuming the per-type lists in the same order as `partial_layers`.
    - If the model returns a single object for a type, it's treated as a one-item list.
    - If there are fewer entries than occurrences, the missing occurrences fall back
      to the original `partial_layers` params. If there are extra entries, we log a
      warning and ignore the extras.
    """
    result: List[ManualLayerConfig] = []

    # Build per-type queues from the extracted output (copy lists to avoid mutating input)
    queues: Dict[str, List[Any]] = {}
    for t, v in extracted.items():
        if isinstance(v, list):
            queues[t] = list(v)
        else:
            queues[t] = [v]

    # Iterate original layers and pop the next params entry for each occurrence
    for orig_idx, orig_layer in enumerate(partial_layers):
        layer_type = orig_layer.layer_type
        queue = queues.get(layer_type, [])

        params_entry = queue.pop(0) if queue else None

        # Normalize params_entry into a dict
        if params_entry is None:
            params_dict: Dict[str, Any] = orig_layer.params or {}
        elif isinstance(params_entry, BaseModel):
            params_dict = params_entry.model_dump(exclude_none=True)
        elif isinstance(params_entry, dict):
            params_dict = params_entry
        else:
            raise ValueError(f"Unexpected params entry type: {type(params_entry)}")

        result.append(
            ManualLayerConfig(
                layer_type=layer_type,
                params=params_dict,
                position=orig_idx,
            )
        )

    # Warn if the LLM returned more entries than occurrences
    for t, q in queues.items():
        if q:
            logging.warning(
                "explicit_to_manual_layer_configs: extra parameter entries for type %s (extra=%d)",
                t,
                len(q),
            )

    return result
