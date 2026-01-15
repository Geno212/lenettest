"""Cross-phase extraction utilities for capturing hints across workflow phases.

This module provides:
1. Shared Pydantic models for extracting cross-phase hints
2. Smart merge functions that handle additive vs replacement semantics
3. LLM-based intent classification for robust merge decisions
4. Helper utilities for consistent extraction across assistants
"""

from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field
from src.agentic.src.core.state import (
    ManualLayerConfig,
    OptimizerConfig,
    LossFunctionConfig,
    SchedulerConfig,
    ModelParams,
    ComplexParams,
)

# Global LLM instance for intent classification (set by assistants)
_intent_classifier_llm = None


def set_intent_classifier_llm(llm):
    """Set the LLM to use for intent classification.
    
    This should be called by assistants during initialization.
    """
    global _intent_classifier_llm
    _intent_classifier_llm = llm


class MergeIntent(BaseModel):
    """User's intent for merging new information with existing state.
    
    This is used to determine whether user wants to add/update or completely replace.
    """
    action: Literal["add", "replace"] = Field(
        ...,
        description=(
            "User's intended action:\n"
            "- 'add': Append new items to existing or update existing items (e.g., add more layers, update parameters)\n"
            "- 'replace': Completely replace existing with new (e.g., start over, forget previous)\n"
        )
    )


class PartialModelParams(BaseModel):
    """Flexible version of ModelParams for extraction - all fields optional.
    
    This allows LLM to extract partial training parameters across multiple turns.
    Fields will be merged with existing state using smart merge logic.
    """
    height: Optional[int] = None
    width: Optional[int] = None
    channels: Optional[int] = None
    epochs: Optional[int] = None
    target_accuracy: Optional[float] = None
    batch_size: Optional[int] = None
    device: Optional[str] = None
    dataset: Optional[str] = None
    dataset_path: Optional[str] = None

class PartialComplexParams(BaseModel):
    """Advanced or optional training parameters."""

    data_workers: Optional[int] = Field(None, description="Number of dataloader workers")
    eval_interval: Optional[int] = Field(None, description="Evaluation cadence in epochs")
    warmup_epochs: Optional[int] = Field(None, description="Warmup epochs before LR schedule")
    scheduler: Optional[str] = Field(None, description="Scheduler strategy name. Available options: cos, warmcos, yoloxwarmcos, yoloxsemiwarmcos, multistep")
    num_classes: Optional[int] = Field(None, description="Output classes for the task")
    pretrained_weights: Optional[str] = Field(None, description="FileSystem Path to pretrained weights  Extract ONLY if the value is a valid filesystem path. It MUST contain directory separators (/ or \\) OR end with a file extension (e.g., .pth, .pt, .ckpt).")


class CrossPhaseExtraction(BaseModel):
    """Extraction model capturing hints from any phase about any concern.
    
    All fields are optional. An assistant can extract whatever the user mentions,
    regardless of which phase is currently active.
    
    SPECIAL CASE - SCHEDULER: If the user explicitly states they don't want to use a scheduler,
    set scheduler_type='None' 
    This is ONLY for scheduler, not other components.
    """
    
    # Project hints
    project_name: Optional[str] = None
    output_dir: Optional[str] = None
    
    # Architecture hints
    pretrained_model: Optional[str] = None
    manual_layers: Optional[List[ManualLayerConfig]] = None
    
    # Training configuration hints
    model_params: Optional[PartialModelParams] = None
    optimizer_config: Optional[OptimizerConfig] = None
    loss_config: Optional[LossFunctionConfig] = None
    scheduler_config: Optional[SchedulerConfig] = None


class ComplexParamsExtraction(BaseModel):
    """Extraction model for YOLOX complex parameters only."""
    complex_params: Optional[PartialComplexParams] = None


class DesignPhaseExtraction(BaseModel):
    """Extraction model for design phase assistants (architecture & configuration).
    
    Similar to CrossPhaseExtraction but excludes project-level fields (project_name, output_dir)
    since those are handled by ProjectManager.
    
    All fields are optional. Extract whatever the user mentions.
    
    SPECIAL CASE - SCHEDULER: If the user explicitly states they don't want to use a scheduler,
    set scheduler_type='None'. This is ONLY for scheduler, not other components.
    """
    
    # Architecture hints
    pretrained_model: Optional[str] = None
    manual_layers: Optional[List[ManualLayerConfig]] = None
    
    # Training configuration hints
    model_params: Optional[PartialModelParams] = None
    optimizer_config: Optional[OptimizerConfig] = None
    loss_config: Optional[LossFunctionConfig] = None
    scheduler_config: Optional[SchedulerConfig] = None


async def classify_merge_intent(
    user_message: str,
    context: str,
    existing_state: Any = None
) -> MergeIntent:
    """Classify user's intent for merging using LLM.
    
    Args:
        user_message: The user's latest message
        context: What is being merged (e.g., "manual layers", "optimizer config")
        existing_state: The actual existing state object/value (not a description)
        
    Returns:
        MergeIntent with action (add/replace/update)
    """
    if _intent_classifier_llm is None:
        # Fallback to heuristic if no LLM available
        return _classify_intent_heuristic(user_message)
    
    prompt = f"""Analyze the user's intent for how to merge their new information with existing state.

Context: {context}
Existing state: {existing_state}
User message: "{user_message}"

Determine if the user wants to:
- "add": Append/add new items to what exists (e.g., "add another layer", "also include")
- "replace": Completely replace/discard existing (e.g., "start over", "forget that", "change to")
- "update": Modify/refine existing items (e.g., "change the learning rate", "increase epochs")


If nothing exists yet, use "add" by default.
If user is making their first specification, use "add".
"""
    
    try:
        structured_llm = _intent_classifier_llm.with_structured_output(MergeIntent)
        intent = await structured_llm.ainvoke(prompt)
        if isinstance(intent, MergeIntent):
            return intent
        if isinstance(intent, dict):
            return MergeIntent.model_validate(intent)
    except Exception as e:
        # Fallback to heuristic on error
        return _classify_intent_heuristic(user_message)
    
    return _classify_intent_heuristic(user_message)


def _classify_intent_heuristic(user_message: str) -> MergeIntent:
    """Fallback heuristic-based intent classification.
    
    Used when LLM is unavailable or fails.
    """
    msg_lower = user_message.lower()
    
    # Strong replacement signals
    # Special case: only scheduler supports "no X" / "without X" pattern
    replace_signals = [
        "replace all", "change all", "start over", "forget",
        "discard", "instead of", "scratch", "new architecture",
        "switch to", "change to"
    ]
    
    # Scheduler-specific "no/without" patterns
    scheduler_no_patterns = [
        "no scheduler", "without scheduler", "don't use scheduler",
        "do not use scheduler", "no learning rate scheduler"
    ]
    
    if any(signal in msg_lower for signal in replace_signals):
        return MergeIntent(action="replace")
    
    if any(pattern in msg_lower for pattern in scheduler_no_patterns):
        return MergeIntent(action="replace")
    
    # Update signals
    update_signals = [
        "change", "modify", "update", "adjust", "set", "increase", "decrease"
    ]
    if any(signal in msg_lower for signal in update_signals):
        return MergeIntent(action="add")
    
    # Default to add (additive is safest)
    return MergeIntent(action="add")


async def merge_manual_layers(
    existing: List[ManualLayerConfig],
    new: Optional[List[ManualLayerConfig]],
    user_message: str = ""
) -> List[ManualLayerConfig]:
    """Merge manual layers with LLM-based intent classification.
    
    Uses LLM to determine if user wants to add or replace layers.
    
    Args:
        existing: Current manual_layers in state
        new: Newly extracted layers
        user_message: Latest user message (for intent classification)
        
    Returns:
        Merged list of ManualLayerConfig
    """
    if new is None or len(new) == 0:
        return existing
    
    if len(existing) == 0:
        return new
    
    # Classify intent using LLM
    intent = await classify_merge_intent(
        user_message=user_message,
        context="manual architecture layers",
        existing_state=existing
    )
    
    # Apply merge based on intent
    if intent.action == "replace":
        return new
    
    # For "add" intent: update existing layers at same positions, insert only truly new ones
    result = [layer.model_copy(deep=True) for layer in sorted(existing, key=lambda x: x.position if x.position is not None else float('inf'))]
    
    # Build a map of existing positions for quick lookup
    existing_positions = {layer.position: idx for idx, layer in enumerate(result)}
    
    # Sort new layers before processing
    new_sorted = sorted(new, key=lambda x: x.position if x.position is not None else float('inf'))
    
    for layer in new_sorted:
        layer_copy = layer.model_copy(deep=True)
        pos = layer_copy.position
        
        # Check if this position already exists - if so, UPDATE instead of INSERT
        if pos is not None and pos in existing_positions:
            existing_idx = existing_positions[pos]
            existing_layer = result[existing_idx]
            # Same layer type at same position = update params
            if existing_layer.layer_type == layer_copy.layer_type:
                # Merge params: new params override existing
                merged_params = {**existing_layer.params, **layer_copy.params}
                result[existing_idx] = ManualLayerConfig(
                    layer_type=layer_copy.layer_type,
                    params=merged_params,
                    position=pos
                )
                continue
        
        # Truly new layer - insert at position
        if pos is None:
            target_pos = len(result)
        else:
            try:
                target_pos = int(pos)
            except Exception:
                target_pos = 0

            if target_pos < 0:
                target_pos = 0
            if target_pos > len(result):
                target_pos = len(result)

        insert_idx = min(target_pos, len(result))
        result.insert(insert_idx, layer_copy)
        # Update position map after insert
        existing_positions = {layer.position: idx for idx, layer in enumerate(result)}
        
    # Final Re-indexing
    for idx, layer in enumerate(result):
        layer.position = idx
    
    return result


def merge_pretrained_model(
    existing: Optional[str],
    new: Optional[str],
    user_intent: str = ""
) -> Optional[str]:
    """Merge pretrained model selection.
    
    Logic:
    - New value always replaces (pretrained model is singular choice)
    - If new is None, keep existing
    
    Args:
        existing: Current pretrained_model in state
        new: Newly extracted pretrained model
        user_intent: Latest user message
        
    Returns:
        Final pretrained model name
    """
        
    if new is not None:
        if new.startswith("yolo"):
            new = new.replace("_", "-")
        return new
    return existing


async def merge_optimizer_config(
    existing: Optional[OptimizerConfig],
    new: Optional[OptimizerConfig],
    user_message: str = ""
) -> Optional[OptimizerConfig]:
    """Merge optimizer configuration with LLM-based intent classification.
    
    Logic:
    - If new optimizer_type differs -> full replacement (always)
    - If same optimizer_type -> use LLM to decide merge vs replace params
    - If new is None -> keep existing
    
    Args:
        existing: Current optimizer_config in state
        new: Newly extracted optimizer config
        user_message: Latest user message
        
    Returns:
        Final OptimizerConfig
    """
    if new is None:
        return existing
    
    if existing is None:
        return new
    
    # Get optimizer_type from existing (handle both dict and Pydantic)
    existing_type = existing.get("optimizer_type") if isinstance(existing, dict) else existing.optimizer_type
    new_type = new.get("optimizer_type") if isinstance(new, dict) else new.optimizer_type
    
    # Different optimizer type -> full replacement (no ambiguity)
    if new_type != existing_type:
        return new if not isinstance(new, dict) else OptimizerConfig(**new)
    
    # Same optimizer type -> check intent for params
    intent = await classify_merge_intent(
        user_message=user_message,
        context=f"optimizer configuration ({existing_type})",
        existing_state=existing
    )
    
    if intent.action == "replace":
        return new if not isinstance(new, dict) else OptimizerConfig(**new)
    
    # For "add" or "update", merge params: take new for overlapping, keep existing for unique, add new for unique
    existing_params = existing.get("params") if isinstance(existing, dict) else existing.params
    new_params = new.get("params") if isinstance(new, dict) else new.params
    merged_params = {**(existing_params or {}), **(new_params or {})}
    return OptimizerConfig(
        optimizer_type=str(new_type),
        params=merged_params
    )


async def merge_loss_config(
    existing: Optional[LossFunctionConfig],
    new: Optional[LossFunctionConfig],
    user_message: str = ""
) -> Optional[LossFunctionConfig]:
    """Merge loss function configuration with LLM-based intent classification.
    
    Logic: Same as optimizer - replace if type changes, use LLM for param merge if same type.
    """
    if new is None:
        return existing
    
    if existing is None:
        return new
    
    # Get loss_type from existing (handle both dict and Pydantic)
    existing_type = existing.get("loss_type") if isinstance(existing, dict) else existing.loss_type
    new_type = new.get("loss_type") if isinstance(new, dict) else new.loss_type
    
    # Different loss type -> full replacement (no ambiguity)
    if new_type != existing_type:
        return new if not isinstance(new, dict) else LossFunctionConfig(**new)
    
    # Same loss type -> check intent for params
    intent = await classify_merge_intent(
        user_message=user_message,
        context=f"loss function configuration ({existing_type})",
        existing_state=existing
    )
    
    if intent.action == "replace":
        return new if not isinstance(new, dict) else LossFunctionConfig(**new)
    
    # For "add" or "update", merge params: take new for overlapping, keep existing for unique, add new for unique
    existing_params = existing.get("params") if isinstance(existing, dict) else existing.params
    new_params = new.get("params") if isinstance(new, dict) else new.params
    merged_params = {**(existing_params or {}), **(new_params or {})}
    return LossFunctionConfig(
        loss_type=str(new_type),
        params=merged_params
    )


async def merge_scheduler_config(
    existing: Optional[SchedulerConfig],
    new: Optional[SchedulerConfig],
    user_message: str = ""
) -> Optional[SchedulerConfig]:
    """Merge scheduler configuration with LLM-based intent classification.
    
    Logic: Same as optimizer/loss.
    """
    if new is None:
        return existing
    
    if existing is None:
        return new
    
    # Get scheduler_type from existing (handle both dict and Pydantic)
    existing_type = existing.get("scheduler_type") if isinstance(existing, dict) else existing.scheduler_type
    new_type = new.get("scheduler_type") if isinstance(new, dict) else new.scheduler_type
    
    # Different scheduler type -> full replacement (no ambiguity)
    if new_type != existing_type:
        return new if not isinstance(new, dict) else SchedulerConfig(**new)
    
    # Same scheduler type -> check intent for params
    intent = await classify_merge_intent(
        user_message=user_message,
        context=f"scheduler configuration ({existing_type})",
        existing_state=existing
    )
    
    if intent.action == "replace":
        return new if not isinstance(new, dict) else SchedulerConfig(**new)
    
    # For "add" or "update", merge params: take new for overlapping, keep existing for unique, add new for unique
    existing_params = existing.get("params") if isinstance(existing, dict) else existing.params
    new_params = new.get("params") if isinstance(new, dict) else new.params
    merged_params = {**(existing_params or {}), **(new_params or {})}
    return SchedulerConfig(
        scheduler_type=str(new_type),
        params=merged_params
    )


async def merge_model_params(
    existing: "Optional[ModelParams] | Dict[str, Any]",
    new: Optional["ModelParams | PartialModelParams"]
) -> "Optional[ModelParams] | Dict[str, Any]":
    """Merge model parameters with field-wise updates.
    
    Logic:
    - Always merge: new non-None values override existing, keep existing for fields not in new
    - Handles both complete ModelParams and partial PartialModelParams from extraction
    - Returns dict when fields are incomplete (to preserve partial data across turns)
    
    Args:
        existing: Current model_params in state (ModelParams object or dict)
        new: Newly extracted model params (ModelParams, PartialModelParams, or dict)
        
    Returns:
        Final ModelParams if all required fields present, or dict with partial data, or None
    """
    if new is None:
        return existing
    
    if existing is None:
        # If new is PartialModelParams, try to create ModelParams (may fail if required fields missing)
        if isinstance(new, PartialModelParams):
            new_dict = new.model_dump(exclude_none=True)
            try:
                return ModelParams(**new_dict)
            except Exception:
                # Can't create valid ModelParams yet with partial data - return dict to preserve
                return new_dict
        return new
    
    # Always merge field by field: new values override existing
    # Handle existing as dict or Pydantic object
    if isinstance(existing, dict):
        merged_dict = existing.copy()
    else:
        merged_dict = existing.model_dump()
    
    # Handle new as dict, PartialModelParams, or ModelParams
    if isinstance(new, dict):
        new_dict = {k: v for k, v in new.items() if v is not None}
    else:
        new_dict = new.model_dump(exclude_none=True)
    
    # Merge: new values override existing
    merged_dict.update(new_dict)
    
    # Try to create valid ModelParams
    try:
        return ModelParams(**merged_dict)
    except Exception:
        # Still missing required fields - return merged dict to preserve partial data
        # This allows accumulation across multiple turns
        return merged_dict


async def merge_complex_params(
    existing: "Optional[ComplexParams] | Dict[str, Any]",
    new: Optional["ComplexParams | Dict[str, Any]"]
) -> "Optional[ComplexParams] | Dict[str, Any]":
    """Merge complex training parameters with field-wise updates."""
    if new is None:
        return existing
    
    if existing is None:
        # If new is PartialModelParams, try to create ModelParams (may fail if required fields missing)
        if isinstance(new, PartialComplexParams):
            new_dict = new.model_dump(exclude_none=True)
            try:
                return ComplexParams(**new_dict)
            except Exception:
                # Can't create valid ModelParams yet with partial data - return dict to preserve
                return new_dict
        return new
    
    # Always merge field by field: new values override existing
    # Handle existing as dict or Pydantic object
    if isinstance(existing, dict):
        merged_dict = existing.copy()
    else:
        merged_dict = existing.model_dump()
    
    # Handle new as dict, PartialModelParams, or ModelParams
    if isinstance(new, dict):
        new_dict = {k: v for k, v in new.items() if v is not None}
    else:
        new_dict = new.model_dump(exclude_none=True)
    
    # Merge: new values override existing
    merged_dict.update(new_dict)
    
    # Try to create valid ModelParams
    try:
        return ComplexParams(**merged_dict)
    except Exception:
        # Still missing required fields - return merged dict to preserve partial data
        # This allows accumulation across multiple turns
        return merged_dict


async def apply_cross_phase_merge(
    state: Any,
    extracted: "CrossPhaseExtraction | DesignPhaseExtraction",
    complex_params: Optional["ComplexParams | Dict[str, Any]"] = None,
    user_message: str = ""
) -> Dict[str, Any]:
    """Apply cross-phase extraction to state with LLM-based smart merging.
    
    This is the main function assistants call to merge extracted hints
    into their state updates. Uses LLM to classify merge intent for robust decisions.
    
    Args:
        state: Current state object (supports both dict-like access and attribute access)
        extracted: CrossPhaseExtraction or DesignPhaseExtraction with newly extracted hints
        complex_params: Optional complex parameters to merge (for YOLOX etc.)
        user_message: Latest user message for intent classification
        
    Returns:
        Dict of state updates to return from assistant __call__
    """
    updates: Dict[str, Any] = {}
    
    # Helper to get value from state (supports both dict and object)
    def get_state_value(key: str, default: Any = None) -> Any:
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)
    
    # Merge architecture hints
    if extracted.pretrained_model is not None:
        updates["pretrained_model"] = merge_pretrained_model(
            get_state_value("pretrained_model"),
            extracted.pretrained_model,
            user_message
        )
    
    if extracted.manual_layers is not None:
        updates["manual_layers"] = await merge_manual_layers(
            get_state_value("manual_layers", []),
            extracted.manual_layers,
            user_message
        )
    
    # Merge training config hints
    if extracted.optimizer_config is not None:
        updates["optimizer_config"] = await merge_optimizer_config(
            get_state_value("optimizer_config"),
            extracted.optimizer_config,
            user_message
        )
    
    if extracted.loss_config is not None:
        updates["loss_config"] = await merge_loss_config(
            get_state_value("loss_config"),
            extracted.loss_config,
            user_message
        )
    
    if extracted.scheduler_config is not None:
        updates["scheduler_config"] = await merge_scheduler_config(
            get_state_value("scheduler_config"),
            extracted.scheduler_config,
            user_message
        )

    if complex_params is not None:
        updates["complex_params"] = await merge_complex_params(
            get_state_value("complex_params"),
            complex_params
        )
    else:
        cp_attr = getattr(extracted, "complex_params", None)
        if cp_attr is not None:
            updates["complex_params"] = await merge_complex_params(
                get_state_value("complex_params"),
                cp_attr
            )
    
    if extracted.model_params is not None:
        updates["model_params"] = await merge_model_params(
            get_state_value("model_params"),
            extracted.model_params
        )
    
    # Project hints (simple replacement, no LLM needed)
    # Only process project fields if extracted is CrossPhaseExtraction (not DesignPhaseExtraction)
    if isinstance(extracted, CrossPhaseExtraction):
        if extracted.project_name is not None:
            updates["project_name"] = extracted.project_name
        
        if extracted.output_dir is not None:
            updates["output_dir"] = extracted.output_dir
    
    return updates
