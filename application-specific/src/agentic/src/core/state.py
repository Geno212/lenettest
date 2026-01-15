"""
Neural Network Generator State Definition

This module defines the complete state structure for the neural network
generator workflow using LangGraph's supervisor pattern.

Design Philosophy:
- Store only PERSISTED artifacts (file paths, not contents)
- Store only COMPLETED information (not in-progress work)
- Store only DECISIONS made (not intermediate reasoning)
- Let assistants maintain working memory via messages

State Flow:
- Primary Assistant routes to specialized assistants
- Each assistant updates specific state fields
- Dialog state stack tracks active assistant
- Messages accumulate conversation history
"""

from collections.abc import MutableMapping, Iterator
from enum import Enum
from typing import Annotated, List, Dict, Optional, Any, Literal, Union
from langchain_core.messages import BaseMessage
import json
import ast

from pydantic import BaseModel, Field, field_validator
import operator



   


class MappingBaseModel(BaseModel, MutableMapping[str, Any]):
    """Base model that behaves like a mutable mapping for backward compatibility."""

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    # ------------------------------------------------------------------
    # Mapping protocol implementations
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        raise TypeError("State entries cannot be deleted")

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def items(self):
        return self._export().items()

    def keys(self):
        return self._export().keys()

    def values(self):
        return self._export().values()

    def copy(self):
        return self.__class__(**self._export())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _export(self) -> Dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()  # Pydantic v2 support
        return self.dict()  # Pydantic v1 fallback


class GraphPhase(str, Enum):
    """High-level workflow phases for the simplified graph."""

    PROJECT = "project"
    DESIGN = "design"
    EXECUTE = "execute"
    COMPLETE = "complete"


class ManualLayerConfig(BaseModel):
    """Flexible representation of a manually specified layer."""

    layer_type: str = Field(..., description="Layer identifier, e.g., Conv2d, Linear")
    params: Dict[str, Any] = Field(default_factory=dict, description="Layer parameters")
    position: Optional[int] = Field(
        None,
        description="Optional ordering hint when reconstructing the network"
    )

    @field_validator("params", mode="before")
    def ensure_params_is_dict(cls, v):
        """Coerce stringified params into a dict when possible.

        Accepts:
        - dict: returned as-is
        - JSON string: parsed via json.loads
        - Python literal string (e.g., "{'a': 1}"): parsed via ast.literal_eval

        Raises ValueError if unable to coerce to a dict.
        """
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            s = v.strip()
            # Try JSON first (standard for LLM output)
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            # Fallback to Python literal (safer than eval)
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            raise ValueError("'params' must be a dict or a JSON/Python string representing an object")
        # Any other types are invalid for params
        raise ValueError("'params' must be a mapping (dict) or a JSON/Python string representing an object")


class OptimizerConfig(BaseModel):
    """Optimizer definition with arbitrary parameter dictionary."""

    optimizer_type: str = Field(..., description="Optimizer name, e.g., Adam, SGD")
    params: Dict[str, Any] = Field(default_factory=dict, description="Optimizer kwargs")
    
    @field_validator("params", mode="before")
    def ensure_params_is_dict(cls, v):
        """Coerce stringified params into a dict when possible.

        Accepts:
        - dict: returned as-is
        - JSON string: parsed via json.loads
        - Python literal string (e.g., "{'a': 1}"): parsed via ast.literal_eval

        Raises ValueError if unable to coerce to a dict.
        """
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            s = v.strip()
            # Try JSON first (standard for LLM output)
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            # Fallback to Python literal (safer than eval)
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            raise ValueError("'params' must be a dict or a JSON/Python string representing an object")
        # Any other types are invalid for params
        raise ValueError("'params' must be a mapping (dict) or a JSON/Python string representing an object")


class LossFunctionConfig(BaseModel):
    """Loss function definition."""

    loss_type: str = Field(..., description="Loss name, e.g., CrossEntropyLoss")
    params: Dict[str, Any] = Field(default_factory=dict, description="Loss kwargs")
    
    @field_validator("params", mode="before")
    def ensure_params_is_dict(cls, v):
        """Coerce stringified params into a dict when possible.

        Accepts:
        - dict: returned as-is
        - JSON string: parsed via json.loads
        - Python literal string (e.g., "{'a': 1}"): parsed via ast.literal_eval

        Raises ValueError if unable to coerce to a dict.
        """
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            s = v.strip()
            # Try JSON first (standard for LLM output)
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            # Fallback to Python literal (safer than eval)
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            raise ValueError("'params' must be a dict or a JSON/Python string representing an object")
        # Any other types are invalid for params
        raise ValueError("'params' must be a mapping (dict) or a JSON/Python string representing an object")


class SchedulerConfig(BaseModel):
    """Learning rate scheduler definition."""

    scheduler_type: str = Field(..., description="Scheduler name, e.g., StepLR")
    params: Dict[str, Any] = Field(default_factory=dict, description="Scheduler kwargs")
    
    @field_validator("params", mode="before")
    def ensure_params_is_dict(cls, v):
        """Coerce stringified params into a dict when possible.

        Accepts:
        - dict: returned as-is
        - JSON string: parsed via json.loads
        - Python literal string (e.g., "{'a': 1}"): parsed via ast.literal_eval

        Raises ValueError if unable to coerce to a dict.
        """
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            s = v.strip()
            # Try JSON first (standard for LLM output)
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            # Fallback to Python literal (safer than eval)
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            raise ValueError("'params' must be a dict or a JSON/Python string representing an object")
        # Any other types are invalid for params
        raise ValueError("'params' must be a mapping (dict) or a JSON/Python string representing an object")


class ModelParams(BaseModel):
    """Core model/training parameters provided by the user."""

    height: int = Field(...,
        description="Input height (optional when square or inferred)")
    width: int = Field(..., description="Input width in pixels")
    channels: int = Field(..., description="Number of input channels")
    epochs: int | None = Field(
        None,
        ge=0,
        description=("Training epochs."),
    )
    target_accuracy: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Target accuracy (0-1)."
        ),
    )
    batch_size: int = Field(..., ge=0, description="Mini-batch size")
    device: str = Field(..., description="Training device: cpu/cuda/mps")
    dataset: str = Field(..., description="Dataset name")
    dataset_path: str = Field(..., description="Filesystem path to dataset")


class ComplexParams(BaseModel):
    """Advanced or optional training parameters."""

    data_workers: int = Field(..., description="Number of dataloader workers")
    eval_interval: int = Field(..., description="Evaluation cadence in epochs")
    warmup_epochs: int = Field(..., description="Warmup epochs before LR schedule")
    scheduler: str = Field(..., description="YOLOX scheduler strategy name. Available options: cos, warmcos, yoloxwarmcos, yoloxsemiwarmcos, multistep")
    num_classes: int= Field(..., description="Output classes for the task when using yolox")
    pretrained_weights: str = Field(..., description="Filesystem path to pretrained weights.")


class NNGeneratorState(MappingBaseModel):
    """
    Comprehensive Pydantic model for the Neural Network Generator workflow state.

    The state is organized into logical layers and exposes new structured
    sub-models for architecture and configuration artifacts. The model retains
    mapping semantics so existing code that treats the state like a dictionary
    continues to function without modification.
    """

    # ==================== CONVERSATION LAYER ====================

    messages: Annotated[List[BaseMessage], operator.add] = Field(
        default_factory=list,
        description="Conversation history including human, AI, and tool messages"
    )
    # Dialog state stack 

    # ==================== SESSION METADATA ====================

    session_id: str = Field("default", description="Unique MCP session identifier")
    current_phase: GraphPhase = Field(
        GraphPhase.PROJECT,
        description="High-level workflow phase controller"
    )

    # ==================== PROJECT LAYER ====================

    project_name: Optional[str] = Field(
        None,
        description="Project name as confirmed by the Project Manager"
    )
    project_path: Optional[str] = Field(
        None,
        description="Absolute path where project artifacts are stored"
    )

    # ==================== ARCHITECTURE LAYER ====================

    architecture_file: Optional[str] = Field(
        None,
        description="Path to the persisted architecture JSON file"
    )
    manual_layers: List[ManualLayerConfig] = Field(
        default_factory=list,
        description="Ordered list of manually specified layers"
    )
    pretrained_model: Optional[str] = Field(
        None,
        description="pretrained model name"
    )
    architecture_applied: bool = Field(False, description="Whether architecture node executed")

    # ==================== CONFIGURATION LAYER ====================

    optimizer_config: Optional[OptimizerConfig] = Field(
        None,
        description="Structured optimizer configuration"
    )
    loss_config: Optional[LossFunctionConfig] = Field(
        None,
        description="Structured loss function configuration"
    )
    scheduler_config: Optional[SchedulerConfig] = Field(
        None,
        description="Structured scheduler configuration"
    )
    model_params: Optional[Union[ModelParams, Dict[str, Any]]] = Field(
        None,
        description="Core training/model parameters (ModelParams when complete, dict when partial)"
    )
    complex_params: Optional[Union[ComplexParams, Dict[str, Any]]] = Field(
        None,
        description="Additional advanced configuration values "
                    "(ComplexParams when complete, dict when partial)"
    )
    training_params_applied: bool = Field(False, description="Whether config node executed")

    # ==================== GENERATED CODE LAYER ====================

    project_output: Optional[str] = Field(
        None,
        description="Root directory for generated code artifacts"
    )

    # ==================== TRAINING LAYER ====================

    training_runs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of completed training runs"
    )
    current_best_model: Dict[str, Any] = Field(
        default_factory=dict,
        description="Best model metadata tracked across runs"
    )
    trained_model_path: Optional[str] = Field(
        None,
        description="Path to the trained model file after training completes"
    )
    class_names_path: Optional[str] = Field(
        None,
        description="Path to class names file for inference"
    )

    # ==================== WORKFLOW CONTROL LAYER ====================

    completed_stages: List[str] = Field(
        default_factory=list,
        description="Ordered list of completed workflow stages"
    )
    current_stage: Optional[str] = Field(
        None,
        description="Stage currently being executed"
    )
    iteration: int = Field(0, ge=0, description="Current optimization iteration")
    max_iterations: int = Field(3, ge=1, description="Maximum optimization iterations")
    target_met: bool = Field(False, description="Whether the target metric was achieved")
    optimization_needed: bool = Field(False, description="If optimization loop is required")
    user_approvals: Dict[str, bool] = Field(
        default_factory=dict,
        description="User approvals gatekeeping sensitive stages"
    )
    design_confirmed: bool = Field(False, description="User confirmed design is ready")
    awaiting_design_confirmation: bool = Field(
        False,
        description="Whether system is waiting for user confirmation during design"
    )
    architecture_update_requested: bool = Field(
        False,
        description="User wants to update architecture after initial design"
    )
    training_params_update_requested: bool = Field(
        False,
        description="User wants to update training parameters after initial design"
    )
    
    run_number: int = Field(1, description="Current project run number")

    # ==================== ERROR HANDLING LAYER ====================

    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured records of runtime errors"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-critical warnings surfaced during workflow"
    )

    # ==================== CONTROL FLAGS ====================

    should_continue: bool = Field(True, description="Controls graph continuation")
    needs_user_input: bool = Field(False, description="Indicates awaiting user input")
    interrupted_for_approval: bool = Field(
        False,
        description="True when paused pending user approval"
    )

    # Node completion flags (for iterative refinement within same node)
    project_needs_completion: bool = Field(
        False,
        description="Project manager needs to run again (e.g., existing directory choice)"
    )
    architecture_needs_completion: bool = Field(
        False,
        description="Architecture node needs to run again (e.g., missing layer params)"
    )
    training_params_needs_completion: bool = Field(
        False,
        description="Config specialist needs to run again (e.g., missing optimizer params)"
    )
    
    # Project manager specific flags
    awaiting_existing_dir_choice: bool = Field(
        False,
        description="Project manager is waiting for user choice on existing directory"
    )
    
    # Architecture designer specific flags
    awaiting_arch_conflict_choice: bool = Field(
        False,
        description="Architecture designer is waiting for user to choose between pretrained and custom"
    )
    awaiting_valid_pretrained_model: bool = Field(
        False,
        description="Architecture designer is waiting for user to provide a valid pretrained model name"
    )
    awaiting_layer_params: bool = Field(
        False,
        description="Architecture designer is waiting for user to provide missing layer parameters"
    )
    
    # Post-training flags
    awaiting_new_design_choice: bool = Field(
        False,
        description="After training completes, waiting for user choice to create new NN or finish"
    )
    awaiting_test_image: bool = Field(
        False,
        description="After user chooses to test model, waiting for image path"
    )
    
    # RTL synthesis flags
    awaiting_rtl_synthesis: bool = Field(
        False,
        description="After user chooses to synthesize RTL, waiting for synthesis to complete"
    )
    awaiting_hls_config: bool = Field(
        False,
        description="Waiting for HLS configuration and C++ generation step"
    )
    awaiting_hls_verify: bool = Field(
        False,
        description="Waiting for HLS C++ verification step"
    )
    awaiting_rtl_build: bool = Field(
        False,
        description="Waiting for RTL synthesis/build step"
    )
    awaiting_direct_rtl_upload: bool = Field(
        False,
        description="User wants to provide existing trained .pt model for direct RTL synthesis"
    )
    pretrained_model_path: Optional[str] = Field(
        None,
        description="Path to user's existing trained .pt model file for direct RTL synthesis"
    )
    rtl_synthesis_complete: bool = Field(
        False,
        description="RTL synthesis has completed successfully"
    )
    rtl_output_path: Optional[str] = Field(
        None,
        description="Path to the synthesized RTL output directory"
    )
    rtl_synthesis_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for RTL synthesis (reuse_factor, clock_period, etc.)"
    )
    tried_dimensions: List[tuple] = Field(
        default_factory=list,
        description="List of input dimensions already tried during RTL synthesis (for auto-retry)"
    )

    # ==================== OUTPUT LAYER ====================

    final_report: Optional[str] = Field(
        None,
        description="Markdown report generated at workflow completion"
    )


# Type aliases for convenience
DialogStateValue = Literal[
    "project_manager",
    "architecture_designer",
    "configuration_specialist",
    "code_generator",
    "training_coordinator",
    "testing_specialist",
]
"""Type alias for valid dialog state values."""


# Default initial state factory
def create_initial_state(
    max_iterations: int = 3,
    session_id: Optional[str] = None
) -> NNGeneratorState:
    """
    Create initial state for a new workflow session.
    
    Args:
        max_iterations: Maximum optimization iterations
        session_id: Optional MCP session ID (will be fetched if not provided)
    
    Returns:
        NNGeneratorState with initial values
    
    Example:
        state = create_initial_state()
    """
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    
    # Session ID will be available from global MCP helper
    # We don't require it at initialization anymore
    if session_id is None:
        session_id = "default"  # Placeholder, will be updated by MCP

    return NNGeneratorState(
        # Conversation
        messages=[],
        # dialog_state removed - use messages and routing helpers for assistant stack

        # Session
        session_id=session_id,
        current_phase=GraphPhase.PROJECT,

        # Project
        project_name=None,
        project_path=None,
        run_number=1,

        
        # Architecture
        architecture_file=None,
        manual_layers=[],
        architecture_applied=False,
        pretrained_model=None,

        # Configuration
        optimizer_config=None,
        loss_config=None,
        scheduler_config=None,
        model_params=None,
    complex_params=None,
    training_params_applied=False,

        # Generated Code
        project_output=None,


        # Training
        training_runs=[],
        current_best_model={},
        trained_model_path=None,
        class_names_path=None,

        # Workflow Control
        completed_stages=[],
        current_stage=None,
        iteration=0,
        max_iterations=max_iterations,
        target_met=False,
        optimization_needed=False,
    user_approvals={},
    design_confirmed=False,
    awaiting_design_confirmation=False,
    architecture_update_requested=False,
    training_params_update_requested=False,

        # Error Handling
        errors=[],
        warnings=[],

        # Control Flags
        should_continue=True,
        needs_user_input=False,
    interrupted_for_approval=False,
    project_needs_completion=False,
    architecture_needs_completion=False,
    training_params_needs_completion=False,
    awaiting_existing_dir_choice=False,
    awaiting_arch_conflict_choice=False,
    awaiting_valid_pretrained_model=False,
    awaiting_layer_params=False,
    awaiting_new_design_choice=False,
    awaiting_test_image=False,

        # Output
        final_report=None,
    )


def create_fresh_design_state(state: NNGeneratorState) -> Dict[str, Any]:
    """
    Create state updates to reset for a new design.
    
    Preserves:
    - project_name
    - project_path
    - session_id
    - messages (conversation history)
    
    Resets everything else to initial values for a fresh design cycle.
    
    Args:
        state: Current state to extract preserved values from
        
    Returns:
        Dict of state updates to apply
    """
    return {
        "current_phase": GraphPhase.PROJECT,
        # Reset Project info
        "project_name": None,
        "project_path": None,
        
        # Reset architecture layer
        "architecture_file": None,
        "manual_layers": [],
        "pretrained_model": None,
        "architecture_applied": False,
        
        # Reset configuration layer
        "optimizer_config": None,
        "loss_config": None,
        "scheduler_config": None,
        "model_params": None,
        "complex_params": None,
        "training_params_applied": False,
        
        # Reset generated code layer
        "project_output": None,
        
        # Reset training layer
        "training_runs": [],
        "current_best_model": {},
        "trained_model_path": None,
        "class_names_path": None,
        
        # Reset workflow control layer
        "completed_stages": [],
        "current_stage": None,
        "iteration": 0,
        "target_met": False,
        "optimization_needed": False,
        "user_approvals": {},
        "design_confirmed": False,
        "awaiting_design_confirmation": False,
        "architecture_update_requested": False,
        "training_params_update_requested": False,
        "run_number": state.run_number + 1,
        
        # Reset error handling layer
        "errors": [],
        "warnings": [],
        
        # Reset control flags
        "should_continue": True,
        "needs_user_input": False,
        "interrupted_for_approval": False,
        "project_needs_completion": False,
        "architecture_needs_completion": False,
        "training_params_needs_completion": False,
        "awaiting_existing_dir_choice": False,
        "awaiting_arch_conflict_choice": False,
        "awaiting_valid_pretrained_model": False,
        "awaiting_layer_params": False,
        "awaiting_new_design_choice": False,
        "awaiting_test_image": False,
        
        # Reset output layer
        "final_report": None,
    }


# Export public API
__all__ = [
    "NNGeneratorState",
    "DialogStateValue",
    "create_initial_state",
    "create_fresh_design_state",
    "ManualLayerConfig",
    "OptimizerConfig",
    "LossFunctionConfig",
    "SchedulerConfig",
    "ModelParams",
    "ComplexParams",
    "GraphPhase",
]