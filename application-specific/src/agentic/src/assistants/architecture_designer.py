"""Architecture Designer - Designs neural network architectures (state-first)."""

from typing import Dict, Any, List, Optional, Literal
from langchain_core.runnables import RunnableConfig
import json
import ast
from langchain_core.messages import HumanMessage, AIMessage
from .base_assistant import BaseAssistant
from pathlib import Path
from src.agentic.src.logic.architecture_logic import ArchitectureDesignerLogic
from src.agentic.src.core.state import NNGeneratorState, ManualLayerConfig
from src.agentic.src.mcp import get_mcp_helper
from pydantic import BaseModel, Field, field_validator
from src.agentic.src.assistants.cross_phase_extraction import (
    DesignPhaseExtraction,
    ComplexParamsExtraction,
    apply_cross_phase_merge,
    set_intent_classifier_llm,
)
# --- Dynamic schema builder for explicit layer extraction ---
from src.agentic.src.utils.dynamic_schema_builder import (
    build_dynamic_extraction_model,
    explicit_to_manual_layer_configs,
)


class ArchitectureDesigner(BaseAssistant):
    """Designs neural network architectures using state + deterministic logic.

    New behavior:
    - Read pretrained_model and manual_layers from state first.
    - If neither present, attempt minimal structured extraction from last user message.
    - If still missing, ask the user with a minimal prompt (no tool binding).
    - If both present, ask the user to choose (conflict resolution) via a simple message.
    - Pretrained flow: validate and create architecture via MCP (no task_spec).
    - Custom flow: deferred (will be implemented later).
    """

    def __init__(self, llm):
        super().__init__(name="architecture_designer", llm=llm)
        self.logic = ArchitectureDesignerLogic()

        # Design phase extraction for ALL hints (architecture + training config)
        self.design_phase_llm = llm.with_structured_output(DesignPhaseExtraction)
        self.conflict_llm = llm.with_structured_output(ConflictResolution)
        self.valid_model_llm = llm.with_structured_output(ValidPretrainedModelChoice)
        self.missing_params_llm = llm.with_structured_output(MissingParamsUpdate)
        
        # Set the global LLM for intent classification
        set_intent_classifier_llm(llm)

    async def __call__(self, state: NNGeneratorState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Design neural network architecture using state-first logic."""
        # ------------------------------------------------------------------
        # 0) Extract ALL hints (architecture + training config) in one call
        # ------------------------------------------------------------------
        result = await self._extract_all_hints(state)
        extracted = result['extraction']
        complex_params = result['complex_params']
        cross_phase_updates = await apply_cross_phase_merge(
            state=state,
            extracted=extracted,
            complex_params=complex_params,
            user_message=self.get_last_user_message(state) or ""
        )
        if extracted.pretrained_model and extracted.pretrained_model.startswith("yolox"):
            extracted.pretrained_model = extracted.pretrained_model.replace("_", "-")
        
        project_path = state.get("project_path")

        # Check if project exists
        if not project_path:
            return {
                "messages": [self.format_message(
                    "❌ **Cannot Design Architecture**\n\n"
                    "Project has not been created yet.\n\n"
                    "Please create a project first."
                )],
                "needs_user_input": False,
                **cross_phase_updates  # Include training hints even on error
            }

        # Read from state first
        pretrained_model = state.get("pretrained_model")
        manual_layers = state.get("manual_layers") or []
        
        # Use merged manual_layers from cross_phase_updates if available
        # When awaiting_layer_params=True, the extraction should have filled in the missing params
        # so we SHOULD use cross_phase_updates.manual_layers if it has complete layers
        if cross_phase_updates.get("manual_layers"):
            manual_layers = cross_phase_updates.get("manual_layers")
        
        # Check if this is an update request (architecture already applied)
        is_update_mode = state.get("architecture_applied") and state.get("architecture_update_requested")
        
        if is_update_mode:
            # User wants to update existing architecture
            if extracted.pretrained_model or cross_phase_updates.get("pretrained_model"):
                # Update pretrained model
                new_pretrained = extracted.pretrained_model or cross_phase_updates.get("pretrained_model")
                result = await self._handle_pretrained_update(state, new_pretrained)
                return {**result, **cross_phase_updates, "architecture_update_requested": False}
            elif manual_layers:
                # Update manual layers (add new layers)
                result = await self._handle_manual_update(state, manual_layers)
                return {**result, **cross_phase_updates, "architecture_update_requested": False}
            else:
                # No clear update - ask user
                return {
                    "messages": [self.format_message(
                        "What would you like to update in the architecture?\n\n"
                        "- Add/modify layers\n"
                        "- Change pretrained model"
                    )],
                    "needs_user_input": True,
                    "architecture_needs_completion": True,
                    **cross_phase_updates
                }

        # Conflict: both provided
        if pretrained_model and manual_layers:
            result = await self._handle_conflict(state, pretrained_model, manual_layers)
            if state.get("architecture_update_requested"):
                return {**result, **cross_phase_updates, "architecture_update_requested": False}
            return {**result, **cross_phase_updates}

        # Pretrained path
        if pretrained_model:
            if pretrained_model.lower().startswith("yolox"):
                result = await self._handle_complex(state, pretrained_model)
            else:
                result = await self._handle_pretrained(state, pretrained_model)
            if state.get("architecture_update_requested"):
                return {**result, **cross_phase_updates, "architecture_update_requested": False}
            return {**result, **cross_phase_updates}

        # Custom path (deferred)
        if manual_layers:
            result = await self._handle_custom(state, manual_layers, cross_phase_updates)
            # Don't let cross_phase_updates overwrite manual_layers from result
            # Result has the correctly processed layers (with params extracted/validated)
            final_cross_phase = {k: v for k, v in cross_phase_updates.items() if k != "manual_layers"} if "manual_layers" in result else cross_phase_updates
            if state.get("architecture_update_requested"):
                return {**result, **final_cross_phase, "architecture_update_requested": False}
            return {**result, **final_cross_phase}
        
            
        # Check if extraction provided either architecture option
        if (extracted.pretrained_model and extracted.manual_layers) or (extracted.pretrained_model and manual_layers) or (extracted.manual_layers and pretrained_model):
            result = await self._handle_conflict(state, extracted.pretrained_model, extracted.manual_layers)
            if state.get("architecture_update_requested"):
                return {**result, **cross_phase_updates, "architecture_update_requested": False}
            return {**result, **cross_phase_updates}
        if extracted.pretrained_model:
            if extracted.pretrained_model.lower().startswith("yolox"):
                result = await self._handle_complex(state, extracted.pretrained_model)
            else:
                result = await self._handle_pretrained(state, extracted.pretrained_model)
            if state.get("architecture_update_requested"):
                return {**result, **cross_phase_updates, "architecture_update_requested": False}
            return {**result, **cross_phase_updates}
        if extracted.manual_layers:
            result = await self._handle_custom(state, extracted.manual_layers, cross_phase_updates)
            # Don't let cross_phase_updates overwrite manual_layers from result
            final_cross_phase = {k: v for k, v in cross_phase_updates.items() if k != "manual_layers"} if "manual_layers" in result else cross_phase_updates
            if state.get("architecture_update_requested"):
                return {**result, **final_cross_phase, "architecture_update_requested": False}
            return {**result, **final_cross_phase}

        # Minimal prompt to ask user for architecture choice
        return {
            "messages": [self.format_message(
                "Please specify your architecture choice.\n\n"
                "Options:\n"
                "- Pretrained model\n"
                "- Custom layers \n"
                "Provide only one of the two."
            )],
            "needs_user_input": True,
            "architecture_needs_completion": True,  # Return to this node after user provides choice
            **cross_phase_updates
        }
    
    async def _handle_conflict(
        self,
        state: NNGeneratorState,
        pretrained: str,
        layers: List[ManualLayerConfig]
    ) -> Dict[str, Any]:
        """Handle conflict when both pretrained and custom are provided.
        
        Uses LLM to parse user's choice between pretrained and custom architecture.
        """
        # Check if this is a return visit (user already responded)
        if state.get("awaiting_arch_conflict_choice"):
            latest_msg = self.get_last_user_message(state) or ""
            
            # Use LLM to parse user's choice
            choice_result = await self._parse_conflict_choice(latest_msg, pretrained, len(layers))
            
            if choice_result.choice == "pretrained":
                if pretrained.lower().startswith("yolox"):
                    result = await self._handle_complex(state, pretrained)
                else:
                    result = await self._handle_pretrained(state, pretrained)
                # Clear manual layers since user chose pretrained
                result["manual_layers"] = []
                result["awaiting_arch_conflict_choice"] = False
                result["awaiting_valid_pretrained_model"] = False
                result["awaiting_layer_params"] = False
                return result
            elif choice_result.choice == "custom":
                # Get cross_phase_updates from the result context if available
                cross_phase = {}
                result = await self._handle_custom(state, layers, cross_phase)
                # Clear pretrained model since user chose custom
                result["pretrained_model"] = None
                result["awaiting_arch_conflict_choice"] = False
                result["awaiting_valid_pretrained_model"] = False
                result["awaiting_layer_params"] = False
                return result
            else:
                # Unclear response - ask again
                return {
                    "messages": [self.format_message(
                        f"I didn't understand your choice.\n\n"
                        f"You've provided both:\n"
                        f"- Pretrained model: {pretrained}\n"
                        f"- Custom layers: {len(layers)} layers\n\n"
                        f"Please clearly state which you want to use:\n"
                        f"- To use the pretrained model, say: 'use pretrained'\n"
                        f"- To use custom layers, say: 'use custom'"
                    )],
                    "needs_user_input": True,
                    "architecture_needs_completion": True,
                    "awaiting_arch_conflict_choice": True,
                }
        
        # First visit: Ask user to choose
        return {
            "messages": [self.format_message(
                "You've provided both a pretrained model and custom layers.\n\n"
                f"Pretrained: {pretrained}\n"
                f"Custom layers: {len(layers)} entries\n\n"
                "Which would you like to use?"
            )],
            "needs_user_input": True,
            "architecture_needs_completion": True,
            "awaiting_arch_conflict_choice": True,  # Mark that we're waiting for this choice
        }
    
    async def _parse_conflict_choice(
        self,
        user_response: str,
        pretrained_name: str,
        num_layers: int
    ) -> "ConflictResolution":
        """Use LLM to parse user's choice between pretrained and custom."""
        prompt = (
            f"The user has provided both a pretrained model and "
            f"custom layers.\n\n"
            f"User was asked to choose between:\n"
            f"1. Use pretrained model\n"
            f"2. Use custom layers\n\n"
            f"User's response:\n{user_response}\n\n"
            f"Parse their intent:\n"
            f"- If they want pretrained, return choice='pretrained'\n"
            f"- If they want custom, return choice='custom'\n"
            f"- If unclear, return choice='unclear'"
        )
        
        try:
            result = await self.conflict_llm.ainvoke(prompt)
            if isinstance(result, ConflictResolution):
                return result
            if isinstance(result, dict):
                return ConflictResolution.model_validate(result)
        except Exception:
            return ConflictResolution(choice="unclear")
        
        return ConflictResolution(choice="unclear")
    
    async def _parse_valid_model_choice(
        self,
        user_response: str,
        available_models: List[str]
    ) -> "ValidPretrainedModelChoice":
        """Use LLM to extract valid pretrained model name from user response."""
        prompt = (
            f"The user was asked to provide a valid pretrained model name.\n\n"
            f"Available models: {', '.join(available_models)}\n\n"
            f"User's response:\n{user_response}\n\n"
            f"Extract the model name they chose. Return model_name field with the extracted name,\n"
            f"or None if you can't confidently identify a model name."
        )
        
        try:
            result = await self.valid_model_llm.ainvoke(prompt)
            if isinstance(result, ValidPretrainedModelChoice):
                return result
            if isinstance(result, dict):
                return ValidPretrainedModelChoice.model_validate(result)
        except Exception:
            return ValidPretrainedModelChoice(model_name=None)
        
        return ValidPretrainedModelChoice(model_name=None)
    
    async def _handle_pretrained(
        self,
        state: NNGeneratorState,
        pretrained_model: str,
    ) -> Dict[str, Any]:
        """Handle pretrained model path with LLM parsing for invalid models."""

        project_path = state.get("project_path")
        available_models = self.logic.get_available_pretrained_models()

       # Check if we're awaiting a valid model name from user
        if state.get("awaiting_valid_pretrained_model"):
            latest_msg = self.get_last_user_message(state) or ""
            
            # Use LLM to extract model name from user's response
            model_choice = await self._parse_valid_model_choice(latest_msg, available_models)
            
            if model_choice.model_name and self.logic.is_valid_pretrained_model(model_choice.model_name):
                # Valid model provided - proceed with creation
                pretrained_model = model_choice.model_name
                # Continue to creation below (don't return yet)
            elif model_choice.model_name:
                # Extracted but still invalid
                return {
                    "messages": [self.format_message(
                        f"The model '{model_choice.model_name}' is not valid.\n\n"
                        f"Available models: {', '.join(available_models)}\n\n"
                        f"Please choose one of the listed models."
                    )],
                    "needs_user_input": True,
                    "architecture_needs_completion": True,
                    "awaiting_valid_pretrained_model": True,
                }
            else:
                # Couldn't extract a model name
                return {
                    "messages": [self.format_message(
                        f"Please provide one of the available pretrained models:\n\n"
                        f"{', '.join(available_models)}"
                    )],
                    "needs_user_input": True,
                    "architecture_needs_completion": True,
                    "awaiting_valid_pretrained_model": True,
                }
        
        # First time or valid model - validate
        if not self.logic.is_valid_pretrained_model(pretrained_model):
            return {
                "messages": [self.format_message(
                    f"Unknown pretrained model: {pretrained_model}.\n\n"
                    f"Available models: {', '.join(available_models)}\n\n"
                    f"Which model would you like to use?"
                )],
                "needs_user_input": True,
                "architecture_needs_completion": True,
                "awaiting_valid_pretrained_model": True,  # Mark that we're waiting for valid model
            }

        # Create architecture via MCP (clear all awaiting flags)
        result_updates = {
            "awaiting_valid_pretrained_model": False,
            "awaiting_arch_conflict_choice": False,
            "awaiting_layer_params": False,
        }
        mcp = await get_mcp_helper()

        try:
            if not project_path:
                raise ValueError("Project path is not set; create a project before designing architecture.")

            arch_name = f"{pretrained_model}_architecture_{state.run_number}.json"
            arch_file_path = f"{project_path}/architectures/{arch_name}"

            await mcp.call("create_architecture", {
                "base_model": pretrained_model,
                "name": arch_name.replace('_architecture.json', ''),
                "output_dir": str(Path(project_path) / "architectures"),
                "run_number": state.run_number,
            })

            # Retrieve info and save
            await mcp.call("save_arch", {"file_path": arch_file_path})

            message = (
                "✅ **Architecture Created Successfully**\n\n"
                f"**Type:** Pretrained ({pretrained_model})\n"
                f"**File:** `{arch_file_path}`\n\n"
                "What would you like to do next?\n"
                "- Proceed to **training configuration**\n"
                "- **Update the architecture**"
            )

            return {
                "architecture_file": arch_file_path,
                "architecture_applied": True,
                "architecture_needs_completion": False,  # Clear completion flag
                "needs_user_input": True,  # Stop and wait for user acknowledgment
                "messages": [self.format_message(message)],
                **result_updates,  # Clear awaiting flags
                **self.update_stage(state, "architecture_designed"),
            }

        except Exception as e:
            return self.create_error_response(e, "architecture_design", "high")

    async def _handle_complex(
        self,
        state: NNGeneratorState,
        pretrained_model: str,
    ) -> Dict[str, Any]:
        """Handle complex pretrained model path with LLM parsing for invalid models."""

        project_path = state.get("project_path")
        available_models = self.logic.get_available_pretrained_models()

       # Check if we're awaiting a valid model name from user
        if state.get("awaiting_valid_pretrained_model"):
            latest_msg = self.get_last_user_message(state) or ""
            
            # Use LLM to extract model name from user's response
            model_choice = await self._parse_valid_model_choice(latest_msg, available_models)
            
            if model_choice.model_name and self.logic.is_valid_pretrained_model(model_choice.model_name):
                # Valid model provided - proceed with creation
                pretrained_model = model_choice.model_name
                # Continue to creation below (don't return yet)
            elif model_choice.model_name:
                # Extracted but still invalid
                return {
                    "messages": [self.format_message(
                        f"The model '{model_choice.model_name}' is not valid.\n\n"
                        f"Available models: {', '.join(available_models)}\n\n"
                        f"Please choose one of the listed models."
                    )],
                    "needs_user_input": True,
                    "architecture_needs_completion": True,
                    "awaiting_valid_pretrained_model": True,
                }
            else:
                # Couldn't extract a model name
                return {
                    "messages": [self.format_message(
                        f"Please provide one of the available pretrained models:\n\n"
                        f"{', '.join(available_models)}"
                    )],
                    "needs_user_input": True,
                    "architecture_needs_completion": True,
                    "awaiting_valid_pretrained_model": True,
                }
        
        # First time or valid model - validate
        if not self.logic.is_valid_pretrained_model(pretrained_model):
            return {
                "messages": [self.format_message(
                    f"Unknown pretrained model: {pretrained_model}.\n\n"
                    f"Available models: {', '.join(available_models)}\n\n"
                    f"Which model would you like to use?"
                )],
                "needs_user_input": True,
                "architecture_needs_completion": True,
                "awaiting_valid_pretrained_model": True,  # Mark that we're waiting for valid model
            }

        # Create architecture via MCP (clear all awaiting flags)
        result_updates = {
            "awaiting_valid_pretrained_model": False,
            "awaiting_arch_conflict_choice": False,
            "awaiting_layer_params": False,
        }
        mcp = await get_mcp_helper()

        try:
            if not project_path:
                raise ValueError("Project path is not set; create a project before designing architecture.")

            arch_file_path = f"{project_path}\\architectures\\{pretrained_model}_complex_architecture.json"

            await mcp.call("create_complex_architecture", {
                "name": pretrained_model,
                "output_dir": str(Path(project_path) / "architectures"),
                "model": pretrained_model
            })

            message = (
                "✅ **Architecture Created Successfully**\n\n"
                f"**Type:** Complex Pretrained ({pretrained_model})\n"
                f"**File:** `{arch_file_path}`\n\n"
                "What would you like to do next?\n"
                "- Proceed to **training configuration**\n"
                "- **Update the architecture**"
            )

            return {
                "architecture_file": arch_file_path,
                "architecture_applied": True,
                "architecture_needs_completion": False,  # Clear completion flag
                "needs_user_input": True,  # Stop and wait for user acknowledgment
                "messages": [self.format_message(message)],
                **result_updates,  # Clear awaiting flags
                **self.update_stage(state, "architecture_designed"),
            }

        except Exception as e:
            return self.create_error_response(e, "architecture_design", "high")
    
    async def _handle_pretrained_update(
        self,
        state: NNGeneratorState,
        new_pretrained_model: str,
    ) -> Dict[str, Any]:
        """Update existing architecture with new pretrained model.
        
        Uses MCP set_pretrained tool to update the pretrained configuration.
        """
        project_path = state.get("project_path")
        architecture_file = state.get("architecture_file")
        
        if not architecture_file:
            return {
                "messages": [self.format_message(
                    "❌ No architecture file found. Please create an architecture first."
                )],
                "needs_user_input": False,
            }
        
        available_models = self.logic.get_available_pretrained_models()
        
        # Validate new pretrained model
        if not self.logic.is_valid_pretrained_model(new_pretrained_model):
            return {
                "messages": [self.format_message(
                    f"❌ Invalid pretrained model: {new_pretrained_model}\n\n"
                    f"Available models: {', '.join(available_models)}"
                )],
                "needs_user_input": False,
            }
        
        mcp = await get_mcp_helper()
        
        try:
            # Update pretrained model via MCP (now handles saving internally)
            result = await mcp.call("set_pretrained", {
                "base_model": new_pretrained_model,
                "file_path": architecture_file,
            })
            
            if result.get("status") == "error":
                return {
                    "messages": [self.format_message(
                        f"❌ Failed to update pretrained model: {result.get('message')}"
                    )],
                    "needs_user_input": False,
                }
            
            # Determine if it was a complex model update
            is_complex = "depth" in result.get("details", {})
            model_type = "Complex Pretrained" if is_complex else "Pretrained"
            
            message = (
                "✅ **Architecture Updated Successfully**\n\n"
                f"**Type:** {model_type}\n"
                f"**Updated Model:** {new_pretrained_model}\n"
                f"**File:** `{architecture_file}`\n\n"
                "Ready to proceed. Would you like to configure training parameters or make more changes?"
            )
            
            return {
                "pretrained_model": new_pretrained_model,
                "architecture_applied": True,
                "architecture_needs_completion": False,
                "architecture_update_requested": False,
                "needs_user_input": True,
                "messages": [self.format_message(message)],
            }
        
        except Exception as e:
            return self.create_error_response(e, "architecture_update", "high")
    
    async def _handle_manual_update(
        self,
        state: NNGeneratorState,
        updated_layers: List[ManualLayerConfig],
    ) -> Dict[str, Any]:
        """Update existing architecture with new/modified layers.
        
        Uses MCP set_layers tool to replace all layers with the updated list.
        """
        project_path = state.get("project_path")
        architecture_file = state.get("architecture_file")
        
        if not architecture_file:
            return {
                "messages": [self.format_message(
                    "❌ No architecture file found. Please create an architecture first."
                )],
                "needs_user_input": False,
            }
        
        # Convert ManualLayerConfig to dict format for MCP
        layers_dict = []
        for idx, layer_cfg in enumerate(updated_layers):
            layer_dict = {
                "layer_type": layer_cfg.layer_type,
                "params": layer_cfg.params,
                "position": layer_cfg.position if layer_cfg.position is not None else idx
            }
            layers_dict.append(layer_dict)
        
        # Extract initial dimensions from state
        cross_phase_updates = {}
        initial_dimensions = self._extract_initial_dimensions(state, cross_phase_updates)
        
        # Track dimensions and infer missing params
        layers_dict = self._track_dimensions_through_layers(layers_dict, initial_dimensions)
        
        # Validate layers for missing required params
        layers_with_missing = []
        for idx, layer_dict in enumerate(layers_dict):
            validation = self.logic.get_missing_required_params(layer_dict)
            if validation.get("error"):
                return {
                    "messages": [self.format_message(
                        f"❌ Error in layer {idx}: {validation['error']}"
                    )],
                    "needs_user_input": False,
                }
            if validation.get("missing"):
                layers_with_missing.append({
                    "position": idx,
                    "layer_type": validation["layer_type"],
                    "missing": validation["missing"],
                })
        
        # If there are missing params, ask user
        if layers_with_missing:
            prompt_lines = ["Missing required parameters for layers:\n"]
            for item in layers_with_missing:
                prompt_lines.append(
                    f"- Layer {item['position']} ({item['layer_type']}): {', '.join(item['missing'])}"
                )
            prompt_lines.append(
                "\nPlease provide the missing parameters."
            )
            
            return {
                "messages": [self.format_message("\n".join(prompt_lines))],
                "needs_user_input": True,
                "architecture_needs_completion": True,
                "awaiting_layer_params": True,
                "manual_layers": updated_layers,
            }
        
        mcp = await get_mcp_helper()
        
        try:
            # Update layers via MCP set_layers tool
            result = await mcp.call("set_layers", {
                "layers_config": layers_dict,
                "file_path": architecture_file,
            })
            
            if result.get("status") == "error":
                return {
                    "messages": [self.format_message(
                        f"❌ Failed to update layers: {result.get('message')}"
                    )],
                    "needs_user_input": False,
                }
            
            message = (
                "✅ **Architecture Updated Successfully**\n\n"
                f"**Total Layers:** {len(layers_dict)}\n"
                f"**File:** `{architecture_file}`\n\n"
                "Ready to proceed. Would you like to configure training parameters or make more changes?"
            )
            
            return {
                "manual_layers": updated_layers,
                "architecture_applied": True,
                "architecture_needs_completion": False,
                "architecture_update_requested": False,
                "needs_user_input": True,
                "messages": [self.format_message(message)],
            }
        
        except Exception as e:
            return self.create_error_response(e, "architecture_update", "high")

    async def _handle_custom(
        self,
        state: NNGeneratorState,
        architecture_layers: List[ManualLayerConfig],
        cross_phase_updates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Handle custom architecture with layer validation and minimal prompts."""
        
        # Convert ManualLayerConfig to dict format for validation
        layers_dict = []
        for idx, layer_cfg in enumerate(architecture_layers):
            layer_dict = {
                "layer_type": layer_cfg.layer_type,
                "params": layer_cfg.params,
                "position": layer_cfg.position if layer_cfg.position is not None else idx
            }
            layers_dict.append(layer_dict)
        
        # Extract initial dimensions from state and cross_phase_updates
        if cross_phase_updates is None:
            cross_phase_updates = {}
        initial_dimensions = self._extract_initial_dimensions(state, cross_phase_updates)
        
        # Track dimensions through layers and infer missing params
        layers_dict = self._track_dimensions_through_layers(layers_dict, initial_dimensions)
        
        # Update the original architecture_layers with inferred params
        for idx, layer_dict in enumerate(layers_dict):
            if idx < len(architecture_layers):
                architecture_layers[idx].params = layer_dict["params"]
        
        # Validate each layer for missing required params
        layers_with_missing = []
        for idx, layer_dict in enumerate(layers_dict):
            validation = self.logic.get_missing_required_params(layer_dict)
            if validation.get("error"):
                return {
                    "messages": [self.format_message(
                        f"❌ Error in layer {idx}: {validation['error']}"
                    )],
                    "needs_user_input": False,
                }
            if validation.get("missing"):
                layers_with_missing.append({
                    "position": idx,
                    "layer_type": validation["layer_type"],
                    "missing": validation["missing"],
                })
        
        # If there are missing params, check if we're in return visit or first time
        if layers_with_missing:
            # Check if this is a return visit (user already responded)
            if state.get("awaiting_layer_params"):
                latest_msg = self.get_last_user_message(state) or ""
                
                # Try structured extraction for missing params
                # Build context about which param each layer needs
                layer_param_map = {}
                for item in layers_with_missing:
                    layer_param_map[item['position']] = {
                        'layer_type': item['layer_type'],
                        'missing': item['missing']
                    }
                
                prompt = (
                    "Extract layer parameter updates from the user's message according to your knowledge of Pytorch parameters of the layers.\n"
                    "The user is providing missing parameter values for neural network layers.\n\n"
                    f"Context - Missing parameters:\n"
                )
                for pos, info in layer_param_map.items():
                    prompt += f"- Layer {pos} ({info['layer_type']}): needs {', '.join(info['missing'])}\n"
                prompt += (
                    f"\nUser's response:\n{latest_msg}\n\n"
                    f"Parse the user's message. When the user provides 'layer N: VALUE', "
                    f"map VALUE to the missing parameter name for that layer and also extract any additional parameters if provided.\n\n"
                    f"Example:\n"
                    f"- 'layer 0: 32 * 7 * 7,16 , layer 1: 32 and use stride 2 for layer 1' where layer 0 needs in_channels and out_channels and layer 1 needs out_channels\n"
                    f"  -> [{{'position': 0, 'params': {{'in_channels': 1568 , 'out_channels': 16}}}}, {{'position': 1, 'params': {{'out_channels': 32, 'stride': 2}}}}]\n"
                    f"Return the layer_updates list with proper parameter names."
                )
                
                try:
                    update_obj = await self.missing_params_llm.ainvoke(prompt)
                    if isinstance(update_obj, MissingParamsUpdate) and update_obj.layer_updates:
                        # Apply updates to layers
                        for upd in update_obj.layer_updates:
                            # upd is now a LayerParamUpdate object
                            pos = upd.position
                            params = upd.params
                            if 0 <= pos < len(layers_dict):
                                # Infer parameter names from missing params if user just provided values
                                missing_for_layer = next(
                                    (item for item in layers_with_missing if item["position"] == pos),
                                    None
                                )
                                if missing_for_layer and len(params) == 1 and list(params.keys())[0].isdigit():
                                    # User provided just a number - map it to the first missing param
                                    value = list(params.values())[0]
                                    param_name = missing_for_layer["missing"][0]
                                    params = {param_name: value}
                                
                                layers_dict[pos]["params"].update(params)
                        
                        # Update the original architecture_layers list as well
                        for idx, layer_dict in enumerate(layers_dict):
                            if idx < len(architecture_layers):
                                architecture_layers[idx].params = layer_dict["params"]
                        
                        # Re-validate after updates
                        layers_with_missing = []
                        for idx, layer_dict in enumerate(layers_dict):
                            validation = self.logic.get_missing_required_params(layer_dict)
                            if validation.get("missing"):
                                layers_with_missing.append({
                                    "position": idx,
                                    "layer_type": validation["layer_type"],
                                    "missing": validation["missing"],
                                })
                except Exception as e:
                    # Log the error for debugging but don't crash
                    print(f"Error extracting missing params: {e}")
                    pass
            
            # If still missing after extraction (or first time), prompt user
            if layers_with_missing:
                prompt_lines = ["Missing required parameters for custom architecture:\n"]
                for item in layers_with_missing:
                    prompt_lines.append(
                        f"- Layer {item['position']} ({item['layer_type']}): {', '.join(item['missing'])}"
                    )
                prompt_lines.append(
                    "\nPlease provide the missing parameters for these layers in this format layer N: <param1>=<value1>, <param2>=<value2>."
                )
                
                # Update state with the modified layers (preserve any updates made so far)
                return {
                    "messages": [self.format_message("\n".join(prompt_lines))],
                    "needs_user_input": True,
                    "architecture_needs_completion": True,
                    "awaiting_layer_params": True,  # Mark that we're waiting for layer params
                    "manual_layers": architecture_layers,  # Persist updated layers back to state
                }
        
        # All params present, create architecture via MCP
        return await self._create_custom_architecture(state, layers_dict)

    def _get_last_ai_message_content(self, state: NNGeneratorState) -> Optional[str]:
        """Retrieve the content of the last AI message from the history."""
        messages = state.get("messages", [])
        # Iterate backwards to find the last AI Message, skipping the immediate user message at the end
        for message in reversed(messages[:-1]): 
            if isinstance(message, AIMessage) and message.content:
                if isinstance(message.content, str):
                    return message.content
                elif isinstance(message.content, list):
                    # Handle multimodal/complex content by joining text parts
                    parts = []
                    for item in message.content:
                        if isinstance(item, dict):
                            parts.append(item.get("text", ""))
                    return " ".join(parts)
        return None

    def _extract_initial_dimensions(self, state: NNGeneratorState, cross_phase_updates: Dict[str, Any]) -> Dict[str, Optional[int]]:
        """Extract initial image dimensions from state and cross_phase_updates.
        
        Returns dict with 'channels', 'width', 'height' (None if not found).
        """
        dimensions: Dict[str, Optional[int]] = {
            "channels": None,
            "width": None,
            "height": None,
        }
        
        def _get_param(obj, key: str):
            """Helper to get value from dict or Pydantic model."""
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj.get(key)
            # Pydantic model - use getattr
            return getattr(obj, key, None)
        
        # Check cross_phase_updates first (most recent)
        model_params = cross_phase_updates.get("model_params")
        if model_params:
            dimensions["channels"] = _get_param(model_params, "channels")
            dimensions["width"] = _get_param(model_params, "width")
            dimensions["height"] = _get_param(model_params, "height")
        
        # Fall back to state if not found
        if dimensions["channels"] is None or dimensions["width"] is None or dimensions["height"] is None:
            state_model_params = state.get("model_params")
            if state_model_params:
                if dimensions["channels"] is None:
                    dimensions["channels"] = _get_param(state_model_params, "channels")
                if dimensions["width"] is None:
                    dimensions["width"] = _get_param(state_model_params, "width")
                if dimensions["height"] is None:
                    dimensions["height"] = _get_param(state_model_params, "height")
        
        return dimensions

    def _track_dimensions_through_layers(
        self,
        layers_dict: List[Dict[str, Any]],
        initial_dimensions: Dict[str, Optional[int]]
    ) -> List[Dict[str, Any]]:
        """Track channels, width, and height through layers, inferring missing params.
        
        ONLY infers in_channels (for Conv) and in_features (for Linear) when possible.
        Does NOT use default values for other params - if any OTHER required param is missing,
        stops tracking and returns layers as-is to let validation handle it.
        
        Handles:
        - Conv layers: infer in_channels, track out_channels
        - Pooling layers: preserve channels, update width/height if params present
        - Activation layers (ReLU, etc.): preserve all dimensions
        - Linear layers: infer in_features from channels * width * height
        - Flatten: convert spatial dimensions to features
        
        Returns updated layers_dict with inferred parameters, or original if tracking must abort.
        """
        current_channels = initial_dimensions.get("channels")
        current_width = initial_dimensions.get("width")
        current_height = initial_dimensions.get("height")
        
        # Layers that preserve channels
        channel_preserving_layers = {
            "ReLU", "ReLU6", "LeakyReLU", "PReLU", "ELU", "GELU", "Sigmoid", "Tanh",
            "MaxPool2d", "MaxPool1d", "MaxPool3d",
            "AvgPool2d", "AvgPool1d", "AvgPool3d",
            "AdaptiveAvgPool2d", "AdaptiveMaxPool2d",
            "Dropout", "Dropout2d", "Dropout3d",
            "BatchNorm2d", "BatchNorm1d", "BatchNorm3d",
            "LayerNorm", "GroupNorm", "InstanceNorm2d"
        }
        
        for idx, layer_dict in enumerate(layers_dict):
            layer_type = layer_dict["layer_type"]
            params = layer_dict["params"]
            
            # Check if layer has all OTHER required params (excluding in_channels/in_features which we infer)
            validation = self.logic.get_missing_required_params(layer_dict)
            if validation.get("error"):
                # Unknown layer type - abort tracking
                return layers_dict
            
            missing_params = validation.get("missing", [])
            # Filter out in_channels/in_features since we're trying to infer those
            critical_missing = [p for p in missing_params if p not in ["in_channels", "in_features"]]
            
            if critical_missing:
                # Other required params are missing - abort tracking and let validation handle it
                return layers_dict
            
            # --- Conv2d/Conv1d/Conv3d ---
            if layer_type in ["Conv2d", "Conv1d", "Conv3d"]:
                # Infer in_channels for first layer or from previous layer
                if "in_channels" not in params:
                    if idx == 0 and current_channels is not None:
                        # First layer: use initial channels from model params
                        params["in_channels"] = current_channels
                    elif idx > 0 and current_channels is not None:
                        # Subsequent layer: use tracked channels
                        params["in_channels"] = current_channels
                
                # Update tracked channels if out_channels is specified
                if "out_channels" in params:
                    current_channels = params["out_channels"]
                
                # Update spatial dimensions for Conv2d (use PyTorch defaults if params not specified)
                if layer_type == "Conv2d" and current_width is not None and current_height is not None:
                    # kernel_size is required - only calculate if present
                    if "kernel_size" in params:
                        kernel_size = params["kernel_size"]
                        # stride and padding are optional - use PyTorch defaults
                        stride = params.get("stride", 1)  # PyTorch default: stride = 1
                        padding = params.get("padding", 0)  # PyTorch default: padding = 0
                        
                        if isinstance(stride, (list, tuple)):
                            stride_h, stride_w = stride[0], stride[1]
                        else:
                            stride_h = stride_w = stride
                        
                        if isinstance(kernel_size, (list, tuple)):
                            kernel_h, kernel_w = kernel_size[0], kernel_size[1]
                        else:
                            kernel_h = kernel_w = kernel_size
                        
                        if isinstance(padding, (list, tuple)):
                            padding_h, padding_w = padding[0], padding[1]
                        else:
                            padding_h = padding_w = padding
                        
                        # Calculate output dimensions
                        current_height = ((current_height + 2 * padding_h - kernel_h) // stride_h) + 1
                        current_width = ((current_width + 2 * padding_w - kernel_w) // stride_w) + 1
            
            # --- Pooling layers (2D) ---
            elif layer_type in ["MaxPool2d", "AvgPool2d"]:
                # Channels preserved
                # Only calculate dimensions if kernel_size is present (required param)
                if current_width is not None and current_height is not None and "kernel_size" in params:
                    kernel_size = params["kernel_size"]
                    # stride and padding are optional - check if present
                    stride = params.get("stride", kernel_size)  # PyTorch default: stride = kernel_size
                    padding = params.get("padding", 0)  # PyTorch default: padding = 0
                    
                    if isinstance(kernel_size, (list, tuple)):
                        kernel_h, kernel_w = kernel_size[0], kernel_size[1]
                    else:
                        kernel_h = kernel_w = kernel_size
                    
                    if isinstance(stride, (list, tuple)):
                        stride_h, stride_w = stride[0], stride[1]
                    else:
                        stride_h = stride_w = stride
                    
                    if isinstance(padding, (list, tuple)):
                        padding_h, padding_w = padding[0], padding[1]
                    else:
                        padding_h = padding_w = padding
                    
                    current_height = ((current_height + 2 * padding_h - kernel_h) // stride_h) + 1
                    current_width = ((current_width + 2 * padding_w - kernel_w) // stride_w) + 1
            
            # --- Adaptive pooling ---
            elif layer_type in ["AdaptiveAvgPool2d", "AdaptiveMaxPool2d"]:
                # Channels preserved, spatial dims set to output_size
                output_size = params.get("output_size")
                if output_size is not None:
                    if isinstance(output_size, (list, tuple)):
                        current_height, current_width = output_size[0], output_size[1]
                    else:
                        current_height = current_width = output_size
            
            # --- Flatten ---
            elif layer_type == "Flatten":
                # Convert spatial dimensions to feature dimension
                if current_channels is not None and current_width is not None and current_height is not None:
                    # After flatten, we have features = channels * width * height
                    flattened_features = current_channels * current_width * current_height
                    # Store for next linear layer
                    current_channels = flattened_features
                    current_width = None
                    current_height = None
            
            # --- Linear ---
            elif layer_type == "Linear":
                if "in_features" not in params:
                    # Calculate in_features from current state
                    if current_channels is not None:
                        if current_width is not None and current_height is not None:
                            # Following Conv layer (needs flattening)
                            params["in_features"] = current_channels * current_width * current_height
                            # After Linear, spatial dimensions are gone
                            current_width = None
                            current_height = None
                        else:
                            # Following another Linear or Flatten
                            params["in_features"] = current_channels
                
                # Update tracked features for next layer
                if "out_features" in params:
                    current_channels = params["out_features"]
                    current_width = None
                    current_height = None
            
            # --- Channel-preserving layers ---
            elif layer_type in channel_preserving_layers:
                # These layers don't change channels or spatial dimensions
                # (except some pooling which is handled above)
                pass
        
        return layers_dict

    async def _extract_all_hints(self, state: NNGeneratorState) -> Dict[str, Any]:
        """Extract ALL hints from user message (architecture + training config).
        Stage 1: normal extraction. Stage 2: YOLOX complex params if needed.
        Returns dict with 'extraction': DesignPhaseExtraction and 'complex_params': optional ComplexParams
        """
        latest_user = self.get_last_user_message(state) or ""
        if not latest_user.strip():
            return {'extraction': DesignPhaseExtraction(), 'complex_params': None}

        complex_params = None

        # ------------------------------------------------------------------
        # CONDITIONAL CONTEXT INJECTION
        # ------------------------------------------------------------------
        # We only inject context if the system specifically flagged that it was waiting 
        # for missing information (via architecture_needs_completion).
        context_prompt_section = ""
        
        # Add existing manual layers as context if present
        existing_layers = state.get("manual_layers")
        if existing_layers:
            layers_summary = []
            for idx, layer in enumerate(existing_layers):
                layer_type = layer.layer_type if hasattr(layer, 'layer_type') else layer.get('layer_type', 'Unknown')
                params = layer.params if hasattr(layer, 'params') else layer.get('params', {})
                layers_summary.append(f"  Layer {idx}: {layer_type} with params {params}")
            
            context_prompt_section += (
                f"\n\nEXISTING LAYERS IN STATE:\n"
                f"{chr(10).join(layers_summary)}\n"
                f"take these into account.\n"
            )
        
        if state.get("architecture_needs_completion"):
            last_ai_msg = self._get_last_ai_message_content(state)
            if last_ai_msg:
                context_prompt_section += (
                    f"\n\nCONTEXT (You previously said this to the user):\n"
                    f"\"\"\"{last_ai_msg}\"\"\"\n"
                    f"The user is likely answering the question asking for architecture choice, pretrained model selection, or missing layer parameters."
                    f"If the AI asked for 'model choice' and user says 'resnet18', map it to pretrained_model.\n"
                )

        # Stage 1: normal extraction
        prompt = (
            "You are extracting configuration information from a user message.\n\n"
            "From the user's message, extract ANY info about:\n"
            "- Architecture (pretrained model name(cannot be a file path), manual_layers)\n"
            "- Training configuration (image width, height and channels, device, dataset, dataset_path, batch size, num_epochs, optimizer, loss function, scheduler)\n\n"
            "Only extract fields you are confident about. Leave others as null.\n"
            "For model_params, extract ONLY the fields mentioned - all fields are optional.\n"
            "if any name of layer/optimizer/loss/scheduler/dataset/yolox model is mentioned but misspelled according to its name in pytorch, extract it as the correct name you know.\n"
            "Example of manual layer extraction (always include \"layer_type\", \"params\", and \"position\" and If a dimension (like in_features) is not explicitly stated in the text, DO NOT calculate it.). Return this section as valid JSON. Example:\n"
            "{"
            "  \"manual_layers\":["
            "    {"
            "      \"layer_type\":\"Conv2d\","
            "      \"params\": {\"in_channels\": 3,\"out_channels\":32, \"kernel_size\":5, \"stride\":2 },"
            "      \"position\": 0"
            "    },"
            "    {"
            "      \"layer_type\":\"Linear\","
            "      \"params\":{\"in_features\":128,\"out_features\":128, \"bias\":false},"
            "      \"position\": 1"
            "}"
            "]"
            "}"
            "IMPORTANT SPECIAL CASE - SCHEDULER: If the user explicitly states they do NOT want to use a scheduler, "
            "set scheduler_config with scheduler_type='None'. This is ONLY for scheduler, not other components.\n"
            f"{context_prompt_section}\n"
            "Example:\n"
            "User: 'I want to use resnet18 for classifying 224x224x3 images with batch size 32 and adam optimizer'\n"
            "Extracted fields:\n"
            "  - pretrained_model: 'resnet18'\n"
            "  - model_params: {{width: 224, height: 224, channels: 3, batch_size: 32}}\n"
            "  - optimizer_config: {{optimizer_type: 'adam'}}\n"
            "Return ONLY valid JSON that matches the DesignPhaseExtraction model.\n\n"
            f"User message:\n{latest_user}"
        )

        try:
            extracted = await self.design_phase_llm.ainvoke(prompt)
            if isinstance(extracted, dict):
                extracted = DesignPhaseExtraction.model_validate(extracted)
        except Exception:
            return {'extraction': DesignPhaseExtraction(), 'complex_params': None}

        # Stage 2: always extract complex params
        if self.llm is not None and not extracted.manual_layers:
            if extracted.pretrained_model:
                if not extracted.pretrained_model.lower().startswith("yolo"):
                    return {'extraction': extracted, 'complex_params': None}
            complex_llm = self.llm.with_structured_output(ComplexParamsExtraction)
            cp_prompt = (
                "You are extracting YOLOX-specific complex training parameters from the user message.\n\n"
                "Extract ONLY fields you see if available else output empty object\n"
                f"{context_prompt_section}\n"
                "pretrained_weights: null unless the user provides an explicit file path. NEVER put model names (like 'yolox-s', 'resnet50') here. NEVER put dataset names (like 'coco') here. It must look like a file location.\n"
                "if context irrelevant to YOLOX complex params, ignore it.\n\n"
                f"\n\nUser message:\n{latest_user}"
            )
            try:
                extra = await complex_llm.ainvoke(cp_prompt)
                if extra and extra.complex_params:
                    complex_params = extra.complex_params
            except Exception:
                pass

        return {'extraction': extracted, 'complex_params': complex_params}

    async def _extract_layer_parameters_refined(self, user_message: str, partial_layers: List[ManualLayerConfig]) -> DesignPhaseExtraction:
        """
        Stage 2: Given user message and partial manual_layers, dynamically build explicit param schema and extract all params.
        Returns a DesignPhaseExtraction with manual_layers fully populated (params as dict).
        """
        # 1. Get unique layer types in order (preserve occurrence order)
        # This ensures we only build schemas for the distinct layer types present,
        # while preserving the original order for per-instance mapping.
        seen = set()
        layer_types = []
        for layer in partial_layers:
            if layer.layer_type not in seen:
                seen.add(layer.layer_type)
                layer_types.append(layer.layer_type)
        # 2. Build dynamic extraction model
        DynamicModel = build_dynamic_extraction_model(layer_types)
        # 3. Prompt LLM to extract all params for each layer type
        # Include expected counts per type so the LLM returns ordered arrays
        # matching the number of occurrences of each layer type.
        counts: dict = {}
        for lt in layer_types:
            counts[lt] = sum(1 for l in partial_layers if l.layer_type == lt)

        prompt = (
            "Extract ALL parameters for the following neural network layers from the user's message.\n"
            "For each layer type, return an ordered JSON array with one entry per occurrence in the architecture (left-to-right order).\n"
            "Each entry should be a JSON object containing all parameters for that layer occurrence.\n"
            f"User message:\n{user_message}\n"
            f"Layer types (distinct, first-seen order): {layer_types}\n"
            f"Expected counts per type: {counts}\n"
            "Return a JSON object with one array field per layer type, each array having exactly the expected number of entries."
        )
        # 4. Run LLM extraction with explicit schema
        if self.llm is None:
            raise ValueError("LLM is not initialized for ArchitectureDesigner.")
        llm_explicit = self.llm.with_structured_output(DynamicModel)
        try:
            explicit_result = await llm_explicit.ainvoke(prompt)
            if isinstance(explicit_result, dict):
                explicit_result = DynamicModel.model_validate(explicit_result)
        except Exception:
            return DesignPhaseExtraction(manual_layers=partial_layers)
        # 5. Convert explicit schema output to ManualLayerConfig list
        manual_layers = explicit_to_manual_layer_configs(
            extracted=explicit_result.model_dump(exclude_none=True),
            partial_layers=partial_layers
        )
        # 6. Return new DesignPhaseExtraction with updated manual_layers
        return DesignPhaseExtraction(manual_layers=manual_layers)
    
    async def _create_custom_architecture(
        self,
        state: NNGeneratorState,
        layers_dict: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create custom architecture via MCP."""
        project_path = state.get("project_path")
        mcp = await get_mcp_helper()
        
        try:
            if not project_path:
                raise ValueError("Project path is not set")
            
            name_for_tool = "custom_architecture"
            arch_name = f"{name_for_tool}.json"
            
            # Create empty architecture
            create_result = await mcp.call("create_architecture", {
                "name": name_for_tool,
                "output_dir": str(Path(project_path) / "architectures"),
                "run_number": state.run_number,
            })
            
            if create_result.get("status") == "error":
                raise ValueError(f"Failed to create architecture: {create_result.get('message')}")
            
            # Define architecture file path for saving after each layer
            arch_file_path = f"{project_path}/architectures/{arch_name}"
            if isinstance(create_result, dict):
                details = create_result.get("details", {})
                created_path = details.get("file_path")
                if created_path:
                    arch_file_path = str(created_path)
            
            # Add layers one by one
            for idx, layer_dict in enumerate(layers_dict):
                layer_type = layer_dict["layer_type"]
                params = layer_dict.get("params", {})
                
                layer_result = await mcp.call("add_layer", {
                    "layer_type": layer_type,
                    "params": params,
                    "position": idx,
                    "file_path": arch_file_path,
                })
                
                if layer_result.get("status") == "error":
                    raise ValueError(f"Failed to add layer {idx} ({layer_type}): {layer_result.get('message')}")
            
            
            await mcp.call("save_arch", {"file_path": arch_file_path})
            message = (
                "✅ **Custom Architecture Created Successfully**\n\n"
                f"**Layers:** {len(layers_dict)}\n"
                f"**File:** `{arch_file_path}`\n\n"
                "What would you like to do next?\n"
                "- Proceed to **training configuration**\n"
                "- **Add more layers** or modify the architecture"
            )
            
            # Convert layers_dict back to ManualLayerConfig for state persistence
            complete_layers = [
                ManualLayerConfig(
                    layer_type=layer["layer_type"],
                    params=layer.get("params", {}),
                    position=layer.get("position", idx)
                )
                for idx, layer in enumerate(layers_dict)
            ]
            
            return {
                "architecture_file": arch_file_path,
                "architecture_applied": True,
                "architecture_needs_completion": False,  # Clear completion flag
                "awaiting_layer_params": False,  # Clear awaiting flag
                "awaiting_arch_conflict_choice": False,  # Clear conflict flag
                "awaiting_valid_pretrained_model": False,  # Clear model validation flag
                "manual_layers": complete_layers,  # Persist complete layers to state for future updates
                "needs_user_input": True,  # Stop and wait for user acknowledgment
                "messages": [self.format_message(message)],
                **self.update_stage(state, "architecture_designed"),
            }
        
        except Exception as e:
            return self.create_error_response(e, "architecture_design", "high")


class ConflictResolution(BaseModel):
    """Structured resolution for pretrained vs custom conflict."""
    choice: Literal["pretrained", "custom", "unclear"] = Field(
        description="User's choice: 'pretrained' to use pretrained model, 'custom' for manual layers, or 'unclear' if ambiguous"
    )


class ValidPretrainedModelChoice(BaseModel):
    """User's choice of valid pretrained model."""
    model_name: Optional[str] = Field(
        None,
        description="The pretrained model name provided by user (e.g., 'resnet18', 'yolox-s')"
    )


class LayerParamUpdate(BaseModel):
    """Single layer parameter update."""
    position: int = Field(description="Zero-based index of the layer to update")
    params: Dict[str, Any] = Field(description="Dictionary of parameter names to values (e.g., {'out_channels': 16})")
    
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

class MissingParamsUpdate(BaseModel):
    """Structured update for missing layer parameters."""
    layer_updates: Optional[List[LayerParamUpdate]] = Field(
        None,
        description="List of layer updates with position (layer index) and params (dict of param_name: value)"
    )