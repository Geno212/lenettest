"""
Phase-based LangGraph definition for the Neural Network Generator.

This graph implements the scenario described in the design brief:
1. Extract intent from the initial user utterance
2. Walk through project, design, and execute phases
3. Confirm design before code generation and training
"""

import re
from typing import Any, Dict, Optional, List

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from pydantic import SecretStr

from .state import (
    NNGeneratorState,
    GraphPhase,
    create_fresh_design_state,
)
from .config import SystemConfig
from pydantic import BaseModel, Field
from src.agentic.src.assistants.project_manager import ProjectManager
from src.agentic.src.assistants.architecture_designer import ArchitectureDesigner
from src.agentic.src.assistants.configuration_specialist import ConfigurationSpecialist
from src.agentic.src.assistants.code_generator import CodeGenerator
from src.agentic.src.assistants.training_coordinator import TrainingCoordinator
from src.agentic.src.assistants.rtl_synthesis_coordinator import RTLSynthesisCoordinator
from src.agentic.src.assistants.cross_phase_extraction import (
    CrossPhaseExtraction,
    apply_cross_phase_merge,
)
from src.agentic.src.assistants.test_image_assistant import (
    test_image_node,
    extract_image_path_from_message,
)
from src.agentic.src.core import llm_context


# ---------------------------------------------------------------------------
# Intent extraction model
# ---------------------------------------------------------------------------

class InitialIntent(BaseModel):
    """User's intent at the beginning - detect if they want direct RTL synthesis with existing .pt file."""
    has_pretrained_model: bool = Field(
        False,
        description="User has an existing trained .pt model file"
    )
    wants_direct_rtl: bool = Field(
        False,
        description="User wants to skip training and go directly to RTL synthesis"
    )
    model_file_path: Optional[str] = Field(
        None,
        description="Path to the existing .pt model file if mentioned"
    )


class DesignUpdateIntent(BaseModel):
    """User's intent for updating design after both architecture and training are set."""
    wants_architecture_update: bool = Field(
        False,
        description="User wants to modify/update the architecture"
    )
    wants_training_update: bool = Field(
        False,
        description="User wants to modify/update training parameters"
    )
    wants_to_confirm: bool = Field(
        False,
        description="User is ready to confirm and proceed to code generation"
    )


class NewDesignIntent(BaseModel):
    """User's intent after training completes - test model, synthesize RTL, create new NN, or finish."""
    wants_to_test_model: bool = Field(
        False,
        description="User wants to test the trained model with an image"
    )
    wants_rtl_synthesis: bool = Field(
        False,
        description="User wants to synthesize the model to RTL hardware"
    )
    wants_new_design: bool = Field(
        False,
        description="User wants to create another neural network design"
    )
    wants_to_finish: bool = Field(
        False,
        description="User wants to finish/end the session"
    )


# Global LLMs (set during graph build)
_design_intent_llm = None
_cross_phase_llm = None


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _get_last_user_message(state: NNGeneratorState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) and message.content:
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text_value = item.get("text") or item.get("content")
                        if text_value:
                            parts.append(str(text_value))
                    else:
                        parts.append(str(item))
                return " ".join(parts)
            return str(content)
    return ""





def _append_stage(state: NNGeneratorState, stage: str) -> list:
    completed = list(state.get("completed_stages", []))
    if stage not in completed:
        completed.append(stage)
    return completed


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------
async def direct_rtl_upload_node(state: NNGeneratorState) -> Dict[str, Any]:
    """Handle user providing existing trained model for direct RTL synthesis.
    
    Supports:
    - Model directory (containing model.pt + model.py) - RECOMMENDED
    - Direct .pt/.pth file (legacy)
    """
    from langchain_core.messages import AIMessage
    import os
    import re
    from pathlib import Path
    
    latest_msg = _get_last_user_message(state)
    
    # Check if user provided model path in state already
    if state.get("pretrained_model_path") and os.path.exists(state.get("pretrained_model_path")):
        model_path = state.get("pretrained_model_path")
        return {
            "trained_model_path": model_path,
            "awaiting_direct_rtl_upload": False,  # Clear flag
            "awaiting_rtl_synthesis": True,
            "current_node": "direct_rtl_upload_node",  # Update graph
            "needs_user_input": False,
            "messages": [AIMessage(
                content=f"✅ Loaded existing model: `{os.path.basename(model_path)}`\n\nStarting RTL synthesis pipeline...",
                name="direct_rtl_upload"
            )],
            "completed_stages": _append_stage(state, "direct_model_upload"),
        }
    
    # Helper function to check if path is a valid model directory
    def is_model_directory(path):
        if not os.path.isdir(path):
            return False
        model_pt = os.path.join(path, 'model.pt')
        model_py = os.path.join(path, 'model.py')
        return os.path.exists(model_pt) and os.path.exists(model_py)
    
    # Helper function to check if path is a project with NN Generator structure
    def is_project_directory(path):
        """Check if path contains Manual_Output/, Pretrained_Output/, or YOLOX/ with correct structure."""
        if not os.path.isdir(path):
            return None
        for output_type in ['Manual_Output', 'Pretrained_Output', 'YOLOX']:
            model_pt = os.path.join(path, output_type, 'SystemC', 'Pt', 'model.pt')
            model_py = os.path.join(path, output_type, 'python', 'model.py')
            if os.path.exists(model_pt) and os.path.exists(model_py):
                return output_type
        return None
    
    # Try to extract directory paths first (new format with model.pt + model.py)
    dir_patterns = [
        r'for\s+([~/][\w\-./]+?)(?:\.|,|\s|$)',  # "for /path/to/model_dir"
        r'([~/][\w\-./]+?)(?:\s+Input|\s+input|\.\s|,\s|\s*$)',  # Unix paths before "Input shape"
        r'([A-Za-z]:[\\\/][\w\-\\\/. ]+?)(?:\s+Input|\s+input|\.\s|,\s|\s*$)',  # Windows paths
    ]
    
    for pattern in dir_patterns:
        match = re.search(pattern, latest_msg)
        if match:
            model_path = match.group(1).strip().rstrip('.')
            is_unix_path = model_path.startswith('/') or model_path.startswith('~')
            is_windows = os.name == 'nt'
            
            # On Windows, assume Unix paths are valid (for WSL/Remote)
            if is_windows and is_unix_path:
                return {
                    "trained_model_path": model_path,
                    "pretrained_model_path": model_path,
                    "awaiting_direct_rtl_upload": False,
                    "awaiting_rtl_synthesis": True,
                    "awaiting_new_design_choice": False,  # Clear post-training state
                    "awaiting_test_image": False,
                    "current_node": "direct_rtl_upload_node",
                    "needs_user_input": False,
                    "messages": [AIMessage(
                        content=f"✅ Model/Project directory: `{model_path}`\n\nStarting RTL synthesis pipeline...",
                        name="direct_rtl_upload"
                    )],
                    "completed_stages": _append_stage(state, "direct_model_upload"),
                }
            
            # Check if it's a model directory (simple format)
            if is_model_directory(model_path):
                return {
                    "trained_model_path": model_path,
                    "pretrained_model_path": model_path,
                    "awaiting_direct_rtl_upload": False,
                    "awaiting_rtl_synthesis": True,
                    "awaiting_new_design_choice": False,  # Clear post-training state
                    "awaiting_test_image": False,
                    "current_node": "direct_rtl_upload_node",
                    "needs_user_input": False,
                    "messages": [AIMessage(
                        content=f"✅ Model directory: `{model_path}`\n\n(Contains model.pt + model.py)\n\nStarting RTL synthesis pipeline...",
                        name="direct_rtl_upload"
                    )],
                    "completed_stages": _append_stage(state, "direct_model_upload"),
                }
            
            # Check if it's a project directory (NN Generator structure)
            output_type = is_project_directory(model_path)
            if output_type:
                return {
                    "trained_model_path": model_path,
                    "pretrained_model_path": model_path,
                    "awaiting_direct_rtl_upload": False,
                    "awaiting_rtl_synthesis": True,
                    "awaiting_new_design_choice": False,  # Clear post-training state
                    "awaiting_test_image": False,
                    "current_node": "direct_rtl_upload_node",
                    "needs_user_input": False,
                    "messages": [AIMessage(
                        content=f"✅ Project directory: `{model_path}`\n\n(Found {output_type} with model.pt + model.py)\n\nStarting RTL synthesis pipeline...",
                        name="direct_rtl_upload"
                    )],
                    "completed_stages": _append_stage(state, "direct_model_upload"),
                }
    
    # Try regex extraction for .pt/.pth files (legacy format)
    path_patterns = [
        r'([~/][\w\-./]+\.pt[h]?)',  # Unix/Linux paths
        r'([A-Za-z]:[\\\/][\w\-\\\/. ]+\.pt[h]?)',  # Windows paths
        r'(\.{1,2}[\/\\][\w\-\/\\. ]+\.pt[h]?)',  # Relative paths
        r'at\s+([^\s]+\.pt[h]?)',  # "at /path/to/model.pt"
    ]
    
    for pattern in path_patterns:
        match = re.search(pattern, latest_msg)
        if match:
            model_path = match.group(1).strip()
            # Allow path if it exists OR if it looks like a Unix path on Windows (assuming WSL/Remote)
            is_unix_path = model_path.startswith('/')
            is_windows = os.name == 'nt'
            
            if os.path.exists(model_path) or (is_windows and is_unix_path):
                return {
                    "trained_model_path": model_path,
                    "pretrained_model_path": model_path,
                    "awaiting_direct_rtl_upload": False,  # Clear flag
                    "awaiting_rtl_synthesis": True,
                    "awaiting_new_design_choice": False,  # Clear post-training state
                    "awaiting_test_image": False,
                    "current_node": "direct_rtl_upload_node",  # Update graph
                    "needs_user_input": False,
                    "messages": [AIMessage(
                        content=f"✅ Loaded existing model: `{os.path.basename(model_path)}`\n\nStarting RTL synthesis pipeline...",
                        name="direct_rtl_upload"
                    )],
                    "completed_stages": _append_stage(state, "direct_model_upload"),
                }
            else:
                return {
                    "messages": [AIMessage(
                        content=f"❌ Model not found at: `{model_path}`\n\nPlease verify the path and try again.",
                        name="direct_rtl_upload"
                    )],
                    "awaiting_rtl_synthesis": True,  # Keep RTL mode active
                    "awaiting_new_design_choice": False,
                    "awaiting_test_image": False,
                    "needs_user_input": True,
                }
    
    # Try LLM extraction as fallback
    if _design_intent_llm is None:
        return {
            "messages": [AIMessage(
                content="Please provide the path to your trained model:\n\n"
                        "**Option 1: Project directory** (if you trained through this UI)\n"
                        "Just provide the project path, e.g., `/home/user/my_project`\n\n"
                        "**Option 2: Model directory** (with model.pt + model.py)\n"
                        "e.g., `/home/user/models/my_model`",
                name="direct_rtl_upload"
            )],
            "awaiting_rtl_synthesis": True,  # Keep RTL mode active
            "awaiting_new_design_choice": False,  # Clear post-training state
            "awaiting_test_image": False,
            "needs_user_input": True,
        }
    
    prompt = f"""Extract the file path to a trained PyTorch model (.pt or .pth file) from the user's message.

User message: "{latest_msg}"

Look for:
- File paths containing .pt or .pth extensions
- Mentions of trained model locations
- Paths to model weights

Respond with structured output containing the file path if found."""
    
    try:
        structured_llm = _design_intent_llm.with_structured_output(InitialIntent)
        intent = await structured_llm.ainvoke(prompt)
        
        if isinstance(intent, dict):
            intent = InitialIntent.model_validate(intent)
        
        if isinstance(intent, InitialIntent) and intent.model_file_path:
            model_path = intent.model_file_path
            if os.path.exists(model_path):
                return {
                    "trained_model_path": model_path,
                    "pretrained_model_path": model_path,
                    "awaiting_direct_rtl_upload": False,  # Clear flag
                    "awaiting_rtl_synthesis": True,
                    "awaiting_new_design_choice": False,  # Clear post-training state
                    "awaiting_test_image": False,
                    "current_node": "direct_rtl_upload_node",  # Update graph
                    "needs_user_input": False,
                    "messages": [AIMessage(
                        content=f"✅ Loaded existing model: `{os.path.basename(model_path)}`\n\nStarting RTL synthesis pipeline...",
                        name="direct_rtl_upload"
                    )],
                    "completed_stages": _append_stage(state, "direct_model_upload"),
                }
            else:
                return {
                    "messages": [AIMessage(
                        content=f"❌ Model not found at: `{model_path}`\n\nPlease provide a valid path to your project or model directory.",
                        name="direct_rtl_upload"
                    )],
                    "awaiting_rtl_synthesis": True,  # Keep RTL mode active
                    "awaiting_new_design_choice": False,
                    "awaiting_test_image": False,
                    "needs_user_input": True,
                }
    except Exception as e:
        pass
    
    # If we couldn't extract path, ask user for it
    # Keep awaiting_rtl_synthesis True so we stay in RTL mode
    return {
        "messages": [AIMessage(
            content="Please provide the path to your trained model:\n\n"
                    "**Option 1: Project directory** (if you trained through this UI)\n"
                    "Just provide the project path, e.g., `/home/user/my_project`\n\n"
                    "**Option 2: Model directory** (with model.pt + model.py)\n"
                    "e.g., `/home/user/models/my_model`",
            name="direct_rtl_upload"
        )],
        "awaiting_rtl_synthesis": True,  # Keep RTL mode active
        "awaiting_new_design_choice": False,  # Clear post-training state
        "awaiting_test_image": False,
        "needs_user_input": True,
    }


def master_triage_router_node(state: NNGeneratorState):
    return {}

def master_triage_router_decision(state: NNGeneratorState) -> str:
    last_msg = _get_last_user_message(state).lower()
    
    # Check if user wants direct RTL synthesis with existing model
    if state.get("awaiting_direct_rtl_upload"):
        return "direct_rtl_upload_node"
    
    # Check if ready for RTL synthesis (from direct upload or post-training)
    # PRIORITY: RTL synthesis flags take precedence over post-training routing
    if state.get("awaiting_rtl_synthesis"):
        return "rtl_synthesis_node"
    
    # Check for intermediate RTL synthesis stages
    if state.get("awaiting_hls_config"):
        return "configure_hls_node"
        
    if state.get("awaiting_hls_verify"):
        if "verify" in last_msg or "check" in last_msg or "proceed" in last_msg or "continue" in last_msg:
            return "verify_hls_node"
        if "synthesize" in last_msg or "build" in last_msg:
             return "synthesize_rtl_node"
             
    if state.get("awaiting_rtl_build"):
        if "synthesize" in last_msg or "build" in last_msg or "generate" in last_msg or "proceed" in last_msg:
            return "synthesize_rtl_node"
    
    # DETECT RTL SYNTHESIS INTENT IN MESSAGE (even after training)
    # Look for keywords indicating RTL synthesis with a model path
    rtl_keywords = ['rtl synthesis', 'rtl', 'synthesize', 'synthesis', 'hls', 'catapult', 'verilog', 'vhdl']
    model_indicators = ['/home/', '~/', '.pt', '.pth', 'model', 'trained', '/ubuntu/', 'Manual_Output', 'Pretrained_Output', 'YOLOX']
    
    has_rtl_intent = any(kw in last_msg for kw in rtl_keywords)
    has_model_path = any(ind in last_msg for ind in model_indicators)
    
    if has_rtl_intent and has_model_path:
        # User wants direct RTL synthesis - route to direct_rtl_upload_node
        return "direct_rtl_upload_node"
    
    # If user provides a path with model indicators while awaiting_new_design_choice,
    # check if it looks like a project path (not an image path)
    if state.get("awaiting_new_design_choice"):
        # Check if message contains a path that looks like a model/project path (not image)
        is_project_path = has_model_path and not any(ext in last_msg for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp'])
        if is_project_path and ('rtl' in last_msg or 'synth' in last_msg):
            # Clear awaiting_new_design_choice and route to RTL
            return "direct_rtl_upload_node"
    
    # Check if awaiting new design choice (post-training)
    if state.get("awaiting_new_design_choice") or state.get("awaiting_test_image"):
        return "post_training_router"
    
    # Check if project node needs to complete (e.g., existing directory choice)
    if state.get("project_needs_completion"):
        return "create_project_node"
    
    phase = state.get("current_phase", GraphPhase.PROJECT)
    if phase == GraphPhase.PROJECT:
        return "create_project_node"
    if phase == GraphPhase.DESIGN:
        return "design_loop_router"
    if phase == GraphPhase.EXECUTE:
        # If code already generated, go directly to training
        completed_stages = state.get("completed_stages", [])
        if "code_generated" in completed_stages:
            return "train_node"
        return "generate_code_node"
    if phase == GraphPhase.COMPLETE:
        return END
    return END

async def design_loop_router_node(state: NNGeneratorState) -> Dict[str, Any]:
    """Parse user intent to determine routing - does NOT extract configuration details.
    
    Configuration extraction happens in individual assistants (architecture_designer, configuration_specialist).
    This router ONLY determines what the user wants to do (update architecture, update training, or confirm).
    """
    
    # If already confirmed, no need to parse
    if state.get("design_confirmed"):
        return {}
    
    latest_msg = _get_last_user_message(state)
    if not latest_msg.strip():
        return {}
    
    result: Dict[str, Any] = {
        "architecture_update_requested": False,
        "training_params_update_requested": False,
        "design_confirmed": False,
    }
    
    # Use LLM to parse user intent (NO extraction, just intent)
    if _design_intent_llm is None:
        return result
    
    # EARLY INTENT: After architecture is applied but training params not yet set
    # Ask: "proceed to training config or update architecture?"
    if state.get("architecture_applied") and not state.get("training_params_applied"):
        prompt = f"""Analyze user's intent after architecture creation.

User message: "{latest_msg}"

Determine if user wants to:
1. wants_architecture_update: Update/modify the architecture (add layers, change model, etc.)
2. wants_to_confirm: Proceed to training configuration

Respond with structured output."""
        
        try:
            structured_llm = _design_intent_llm.with_structured_output(DesignUpdateIntent)
            intent = await structured_llm.ainvoke(prompt)
            
            if isinstance(intent, dict):
                intent = DesignUpdateIntent.model_validate(intent)
            
            if isinstance(intent, DesignUpdateIntent):
                result.update({
                    "architecture_update_requested": intent.wants_architecture_update,
                    "design_confirmed": intent.wants_to_confirm,
                })
        except Exception:
            pass
        
        return result
    
    # FINAL INTENT: After both architecture and training params are set
    # Ask: "confirm design, update architecture, or update training params?"
    if state.get("architecture_applied") and state.get("training_params_applied"):
        prompt = f"""Analyze the user's message to determine their intent for the neural network design.

User message: "{latest_msg}"

Determine:
1. wants_architecture_update: Does the user want to change/modify the architecture ?
2. wants_training_update: Does the user want to change/modify training parameters (optimizer, learning rate, epochs, batch size, loss function, scheduler, model params, height, width, channels, device, complex params, etc.)?
3. wants_to_confirm: Is the user ready to confirm the design and proceed ?

Respond with structured output."""
        
        try:
            structured_llm = _design_intent_llm.with_structured_output(DesignUpdateIntent)
            intent = await structured_llm.ainvoke(prompt)
            
            if isinstance(intent, dict):
                intent = DesignUpdateIntent.model_validate(intent)
            
            if isinstance(intent, DesignUpdateIntent):
                result.update({
                    "architecture_update_requested": intent.wants_architecture_update,
                    "training_params_update_requested": intent.wants_training_update,
                    "design_confirmed": intent.wants_to_confirm,
                })
        except Exception:
            pass
    
    return result


def design_loop_decision(state: NNGeneratorState) -> str:
    # Check if architecture node needs to complete (e.g., missing params)
    if state.get("architecture_needs_completion"):
        return "design_arch_node"
    
    # Check if config node needs to complete
    if state.get("training_params_needs_completion"):
        return "config_params_node"
    
    # Initial setup: ensure both are set at least once
    if not state.get("architecture_applied"):
        return "design_arch_node"
    
    # Check if user wants to update either
    if state.get("architecture_update_requested"):
        return "design_arch_node"

    if not state.get("training_params_applied"):
        return "config_params_node"

    # Both are set - check if user wants to confirm (from intent parsing)
    if state.get("design_confirmed"):
        return "set_design_confirmed_node"
    

    
    if state.get("training_params_update_requested"):
        return "config_params_node"

    # No updates requested - ask for confirmation
    return "ask_design_confirmation_node"



def ask_design_confirmation_node(state: NNGeneratorState) -> Dict[str, Any]:
    from langchain_core.messages import AIMessage
    
    return {
        "messages": [AIMessage(
            content="Architecture and training parameters are ready. Let's proceed to code generation.",
            name="design_confirmation"
        )],
        "needs_user_input": True,
    }


def set_design_confirmed_node(state: NNGeneratorState) -> Dict[str, Any]:
    """Confirm design and transition to EXECUTE phase."""
    return {
        "design_confirmed": True,
        "architecture_update_requested": False,  # Reset flags
        "training_params_update_requested": False,  # Reset flags
        "needs_user_input": False,
        "current_phase": GraphPhase.EXECUTE,
        "completed_stages": _append_stage(state, "design_confirmed"),
    }


def ask_new_design_node(state: NNGeneratorState) -> Dict[str, Any]:
    """Ask user if they want to create another neural network after training completes."""
    from langchain_core.messages import AIMessage
    
    project_name = state.get("project_name", "your project")
    trained_model_path = state.get("trained_model_path")
    
    message_content = (
        f"🎉 **Training Complete!**\n\n"
        f"Your neural network for project `{project_name}` has been trained successfully.\n\n"
    )
    
    if trained_model_path:
        message_content += f"The trained model is saved at: `{trained_model_path}`\n\n"
    
    message_content += (
        f"Would you like to:\n"
        f"- **Test the trained model** with an image (use the Upload button or paste an image path)\n"
        f"- **Create another neural network**\n"
        f"- **Finish the session**\n\n"
        f"Your project files will be preserved at the current location."
    )
    
    return {
        "messages": [AIMessage(content=message_content, name="post_training")],
        "awaiting_new_design_choice": True,
        "needs_user_input": True,
        "completed_stages": _append_stage(state, "training_complete"),
    }


async def post_training_router_node(state: NNGeneratorState) -> Dict[str, Any]:
    """Parse user's choice after training - test image, new design, or finish."""
    # If not awaiting anything post-training, just return
    if not (state.get("awaiting_new_design_choice") or state.get("awaiting_test_image")):
        return {}
    
    latest_msg = _get_last_user_message(state)
    if not latest_msg.strip():
        return {}
    
    result: Dict[str, Any] = {
        "awaiting_new_design_choice": False,
    }

    # If already awaiting image, try to extract it and route to test_image_node
    if state.get("awaiting_test_image"):
        # Try to extract image path from message
        extracted_path = await extract_image_path_from_message(latest_msg)
        if extracted_path:
            # Image found - route to test_image_node with extracted path
            result["awaiting_test_image"] = True
            result["needs_user_input"] = False
            result["extracted_image_path"] = extracted_path
            return result
        else:
            # No image found - ask again
            from langchain_core.messages import AIMessage
            return {
                "messages": [AIMessage(
                    content="Please upload an image (Upload button) or paste an image file path to test the trained model.",
                    name="post_training_router"
                )],
                "awaiting_new_design_choice": False,
                "awaiting_test_image": True,
                "needs_user_input": True,
            }
    
    # Check if user provides an image path directly when choosing what to do after training
    extracted_path = await extract_image_path_from_message(latest_msg)
    if extracted_path:
        # User provided image path directly - route to test
        result["awaiting_test_image"] = True
        result["awaiting_new_design_choice"] = False
        result["needs_user_input"] = False
        result["extracted_image_path"] = extracted_path
        return result
    
    # Use LLM to parse user intent
    if _design_intent_llm is None:
        raise RuntimeError("Design intent LLM is not set in post-training router.")
    
    prompt = f"""Analyze the user's message to determine their intent after training completion.

User message: "{latest_msg}"

Determine:
1. wants_to_test_model: Does the user want to test/evaluate the trained model with an image ?
2. wants_rtl_synthesis: Does the user want to synthesize/convert the model to RTL hardware (Verilog/VHDL/Catapult) ?
3. wants_new_design: Does the user want to create another neural network ?
4. wants_to_finish: Does the user want to finish/end the session ?

If the user provides an image file path, that implies they want to test the model.
If the user mentions RTL, hardware, synthesis, Verilog, VHDL, or Catapult, they want RTL synthesis.

Respond with structured output."""
    
    try:
        structured_llm = _design_intent_llm.with_structured_output(NewDesignIntent)
        intent = await structured_llm.ainvoke(prompt)
        
        if isinstance(intent, dict):
            intent = NewDesignIntent.model_validate(intent)
        
        if isinstance(intent, NewDesignIntent):
            if intent.wants_to_test_model:
                # User wants to test the model - prompt for image
                from langchain_core.messages import AIMessage
                result["awaiting_test_image"] = True
                result["awaiting_new_design_choice"] = False
                result["needs_user_input"] = True
                result["messages"] = [AIMessage(
                    content="Please upload an image (Upload button) or paste an image file path to test the trained model.",
                    name="post_training_router"
                )]
            elif intent.wants_rtl_synthesis:
                # User wants RTL synthesis
                from langchain_core.messages import AIMessage
                result["awaiting_rtl_synthesis"] = True
                result["awaiting_new_design_choice"] = False
                result["needs_user_input"] = False
                result["messages"] = [AIMessage(
                    content="Starting RTL synthesis pipeline...",
                    name="post_training_router"
                )]
            elif intent.wants_new_design:
                # Reset state for new design
                from langchain_core.messages import AIMessage
                fresh_state = create_fresh_design_state(state)
                result.update(fresh_state)
                result["messages"] = [AIMessage(
                    content="Great! Let's design a new neural network. Please describe your architecture or choose a pretrained model.",
                    name="post_training_router"
                )]
            else:
                from langchain_core.messages import AIMessage
                result["current_phase"] = GraphPhase.COMPLETE
                result["messages"] = [AIMessage(
                    content="Thank you for using the Neural Network Generator! Your project files have been saved. Goodbye!",
                    name="post_training_router"
                )]
    except Exception:
        # On error, default to finish
        from langchain_core.messages import AIMessage
        result["current_phase"] = GraphPhase.COMPLETE
        result["messages"] = [AIMessage(
            content="Thank you for using the Neural Network Generator! Your project files have been saved. Goodbye!",
            name="post_training_router"
        )]
    
    return result


def post_training_decision(state: NNGeneratorState) -> str:
    """Decide where to route after post-training router processes user input."""
    # If a node explicitly needs user input, stop this run.
    if state.get("needs_user_input"):
        return END

    # Check if user wants to test with image
    if state.get("awaiting_test_image"):
        return "test_image_node"
    
    # Check if user wants RTL synthesis
    if state.get("awaiting_rtl_synthesis"):
        return "rtl_synthesis_node"
    
    phase = state.get("current_phase", GraphPhase.EXECUTE)
    
    # If phase changed to COMPLETE, end the graph
    if phase == GraphPhase.COMPLETE:
        return END
    
    # If phase changed to DESIGN, go back to design loop
    if phase == GraphPhase.DESIGN:
        return "design_loop_router"
    
    # Safeguard: if we're here with awaiting_new_design_choice still true,
    # it means the router didn't process anything meaningful - stop and wait
    if state.get("awaiting_new_design_choice"):
        return END
    
    # Default: end
    return END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _should_continue_or_end(state: NNGeneratorState) -> str:
    """Check if graph should continue or end based on needs_user_input."""
    if state.get("needs_user_input", False):
        return END
    return "master_triage_router"


async def build_graph(config: SystemConfig):
    """Build the LangGraph instance for the phased workflow."""

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=config.llm.temperature,
        api_key=SecretStr(config.llm.api_key) if config.llm.api_key else None,
    )
    
    # Set global LLMs for intent parsing and cross-phase extraction
    global _design_intent_llm, _cross_phase_llm
    _design_intent_llm = llm
    _cross_phase_llm = llm.with_structured_output(CrossPhaseExtraction)

    # Also set shared LLM handles for assistants/modules that are imported by the graph
    llm_context.set_llms(design_intent=llm, cross_phase=_cross_phase_llm)

    # Instantiate assistants
    project_manager = ProjectManager(llm)
    architecture_designer = ArchitectureDesigner(llm)
    configuration_specialist = ConfigurationSpecialist(llm)
    code_generator = CodeGenerator(llm)
    training_coordinator = TrainingCoordinator(llm)
    rtl_synthesis_coordinator = RTLSynthesisCoordinator(llm)

    builder = StateGraph(NNGeneratorState)

    builder.add_node("master_triage_router", master_triage_router_node)
    builder.add_node("create_project_node", project_manager)
    builder.add_node("direct_rtl_upload_node", direct_rtl_upload_node)
    builder.add_node("design_loop_router", design_loop_router_node)
    builder.add_node("design_arch_node", architecture_designer)
    builder.add_node("config_params_node", configuration_specialist)
    builder.add_node("ask_design_confirmation_node", ask_design_confirmation_node)
    builder.add_node("set_design_confirmed_node", set_design_confirmed_node)
    builder.add_node("generate_code_node", code_generator)
    builder.add_node("train_node", training_coordinator)
    builder.add_node("ask_new_design_node", ask_new_design_node)
    builder.add_node("post_training_router", post_training_router_node)
    builder.add_node("test_image_node", test_image_node)
    builder.add_node("rtl_synthesis_node", rtl_synthesis_coordinator)
    
    # New RTL synthesis sub-nodes
    builder.add_node("configure_hls_node", rtl_synthesis_coordinator.configure_hls)
    builder.add_node("verify_hls_node", rtl_synthesis_coordinator.verify_hls)
    builder.add_node("synthesize_rtl_node", rtl_synthesis_coordinator.synthesize_rtl)

    # Start directly at master triage router
    builder.add_edge(START, "master_triage_router")

    builder.add_conditional_edges(
        "master_triage_router",
        master_triage_router_decision,
        [
            "create_project_node", 
            "direct_rtl_upload_node", 
            "design_loop_router", 
            "generate_code_node", 
            "train_node", 
            "rtl_synthesis_node", 
            "configure_hls_node",
            "verify_hls_node",
            "synthesize_rtl_node",
            "post_training_router", 
            END
        ],
    )

    # Route from project node: check if needs user input
    builder.add_conditional_edges(
        "create_project_node",
        _should_continue_or_end,
        ["master_triage_router", END]
    )
    
    # Route from direct RTL upload node
    def _direct_rtl_should_continue(state: NNGeneratorState) -> str:
        if state.get("needs_user_input", False):
            return END
        if state.get("awaiting_rtl_synthesis"):
            return "rtl_synthesis_node"
        return "master_triage_router"
    
    builder.add_conditional_edges(
        "direct_rtl_upload_node",
        _direct_rtl_should_continue,
        ["rtl_synthesis_node", "master_triage_router", END]
    )

    builder.add_conditional_edges(
        "design_loop_router",
        design_loop_decision,
        [
            "design_arch_node",
            "config_params_node",
            "ask_design_confirmation_node",
            "set_design_confirmed_node",
        ],
    )

    # Design nodes: check if needs user input before continuing
    def _design_should_continue(state: NNGeneratorState) -> str:
        if state.get("needs_user_input", False):
            return END
        return "design_loop_router"
    
    builder.add_conditional_edges(
        "design_arch_node",
        _design_should_continue,
        ["design_loop_router", END]
    )
    builder.add_conditional_edges(
        "config_params_node",
        _design_should_continue,
        ["design_loop_router", END]
    )
    
    # After confirmation, route back to triage
    builder.add_conditional_edges(
        "set_design_confirmed_node",
        _should_continue_or_end,
        ["master_triage_router", END]
    )
    
    # After asking for confirmation, always END to wait for user
    builder.add_edge("ask_design_confirmation_node", END)

    # Code generation: check if needs user input before training
    def _code_gen_should_continue(state: NNGeneratorState) -> str:
        if state.get("needs_user_input", False):
            return END
        return "train_node"
    
    builder.add_conditional_edges(
        "generate_code_node",
        _code_gen_should_continue,
        ["train_node", END]
    )
    
    # After training, go to ask_new_design_node
    builder.add_edge("train_node", "ask_new_design_node")
    
    # After asking about new design, END to wait for user input
    builder.add_edge("ask_new_design_node", END)
    
    # Post-training router processes user's choice and routes accordingly
    builder.add_conditional_edges(
        "post_training_router",
        post_training_decision,
        ["test_image_node", "rtl_synthesis_node", "design_loop_router", END]
    )
    
    # After testing image, go directly to END (user must provide next input)
    # The master_triage_router will then route back to post_training_router when graph resumes
    builder.add_edge("test_image_node", END)
    
    # RTL synthesis node routing
    def _rtl_synthesis_should_continue(state: NNGeneratorState) -> str:
        # Debug logging
        print(f"[DEBUG] _rtl_synthesis_should_continue called:")
        print(f"  needs_user_input: {state.get('needs_user_input', False)}")
        print(f"  awaiting_rtl_synthesis: {state.get('awaiting_rtl_synthesis')}")
        print(f"  awaiting_hls_config: {state.get('awaiting_hls_config')}")
        
        # If needs user input, end current execution (wait for user response)
        if state.get("needs_user_input", False):
            print(f"[DEBUG] Routing to END (needs_user_input)")
            return END
        # If still awaiting RTL synthesis, stay in the node
        if state.get("awaiting_rtl_synthesis"):
            print(f"[DEBUG] Routing to rtl_synthesis_node")
            return "rtl_synthesis_node"
        # If ready for configuration
        if state.get("awaiting_hls_config"):
            print(f"[DEBUG] Routing to configure_hls_node")
            return "configure_hls_node"
        # Otherwise end (synthesis complete or error)
        print(f"[DEBUG] Routing to END (default)")
        return END
    
    builder.add_conditional_edges(
        "rtl_synthesis_node",
        _rtl_synthesis_should_continue,
        ["rtl_synthesis_node", "configure_hls_node", END]
    )
    
    # Configure HLS routing
    def _configure_hls_should_continue(state: NNGeneratorState) -> str:
        if state.get("needs_user_input", False):
            return END
        if state.get("awaiting_hls_verify"):
            return "verify_hls_node"
        return END

    builder.add_conditional_edges(
        "configure_hls_node",
        _configure_hls_should_continue,
        ["verify_hls_node", END]
    )

    # Verify HLS routing
    def _verify_hls_should_continue(state: NNGeneratorState) -> str:
        if state.get("needs_user_input", False):
            return END
        if state.get("awaiting_rtl_build"):
            return "synthesize_rtl_node"
        return END

    builder.add_conditional_edges(
        "verify_hls_node",
        _verify_hls_should_continue,
        ["synthesize_rtl_node", END]
    )

    # Synthesize RTL routing
    def _synthesize_rtl_should_continue(state: NNGeneratorState) -> str:
        # Always end after synthesis to wait for user
        return END

    builder.add_conditional_edges(
        "synthesize_rtl_node",
        _synthesize_rtl_should_continue,
        [END]
    )

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


async def create_graph(config_file: Optional[str] = None):
    """
    Create and initialize graph with configuration.
    
    Args:
        config_file: Optional path to config file
        
    Returns:
        Compiled LangGraph ready to use
        
    Example:
        graph = await create_graph()
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Build MNIST classifier")]},
            config={"configurable": {"thread_id": "1"}}
        )
    """
    from src.agentic.src.core.config import load_config
    from src.agentic.src.utils.langsmith_utils import configure_langsmith_tracing
    
    # Load configuration
    config = load_config(config_file)
    
    # Configure LangSmith tracing if enabled
    if config.langsmith.enabled or config.langsmith.api_key:
        configure_langsmith_tracing(
            api_key=config.langsmith.api_key,
            project_name=config.langsmith.project_name,
            endpoint=config.langsmith.endpoint,
            enabled=config.langsmith.enabled
        )
    
    return await build_graph(config)