"""Configuration Specialist - State-first training configuration assistant.

New behavior (parallels `ArchitectureDesigner`):
1. Read existing configuration objects from state first (`model_params`, `optimizer_config`, `loss_config`, `scheduler_config`).
2. If all required pieces present and not yet applied -> apply directly via MCP (no extra prompting).
3. If some pieces missing -> attempt structured extraction from the latest user message.
4. If still missing -> minimally prompt user for JUST missing pieces using explicit schema examples.
5. On success -> set `training_params_applied=True` so the graph advances.

Required minimum to proceed: `model_params` (height,width,channels,epochs,batch_size,device,dataset,dataset_path), `optimizer_config`, `loss_config`.
Scheduler is optional.
"""

from typing import Dict, Any, Optional, cast, Union, List
import logging
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from .base_assistant import BaseAssistant
from src.agentic.src.core.state import (
    NNGeneratorState,
    ModelParams,
    OptimizerConfig,
    LossFunctionConfig,
    SchedulerConfig,
    ComplexParams,
)
from src.agentic.src.mcp import get_mcp_helper
from src.agentic.src.schemas.optimizers import get_optimizer_config
from src.agentic.src.schemas.losses import get_loss_config
from src.agentic.src.schemas.scheduler import get_scheduler_config
from src.agentic.src.assistants.cross_phase_extraction import (
    DesignPhaseExtraction,
    ComplexParamsExtraction,
    apply_cross_phase_merge,
    set_intent_classifier_llm,
)


class ConfigurationSpecialist(BaseAssistant):
    def __init__(self, llm):
        super().__init__(name="configuration_specialist", llm=llm)
        # Design phase extraction for ALL hints (architecture + training config)
        self.design_phase_llm = llm.with_structured_output(DesignPhaseExtraction)
        self.complex_llm = llm.with_structured_output(ComplexParamsExtraction)
        
        # Set the global LLM for intent classification
        set_intent_classifier_llm(llm)
        self.logger = logging.getLogger(__name__)

    async def __call__(self, state: NNGeneratorState, config: RunnableConfig | None = None) -> Dict[str, Any]:
        """Main entry: apply or collect training configuration."""
        
        # ------------------------------------------------------------------
        # 0) Extract ALL hints (architecture + training config) in one call
        # ------------------------------------------------------------------
        hints = await self._extract_all_hints(state)
        extracted = hints["extraction"]
        complex_params = hints.get("complex_params")
        cross_phase_updates = await apply_cross_phase_merge(
            state=state,
            extracted=extracted,
            complex_params=complex_params,
            user_message=self.get_last_user_message(state) or ""
        )
        
        # If architecture hints were extracted, set architecture_update_requested flag
        if extracted.pretrained_model or extracted.manual_layers:
            cross_phase_updates["architecture_update_requested"] = True

        # Ensure project & architecture are available before configuring
        if not state.get("project_path"):
            return {
                "messages": [self.format_message(
                    "❌ **Cannot Configure Training**\n\nProject has not been created yet. Please create a project first."
                )],
                "needs_user_input": False,
            }

        if not state.get("architecture_applied"):
            return {
                "messages": [self.format_message(
                    "⚠️ Architecture not applied yet. Please finalize architecture before setting training parameters."
                )],
                "needs_user_input": False,
                **cross_phase_updates  # Include architecture hints even on error
            }

        existing_model = state.get("model_params")
        existing_opt = state.get("optimizer_config")
        existing_loss = state.get("loss_config")
        existing_sched = state.get("scheduler_config")
        existing_complex = state.get("complex_params")
        
        # Always handle updates incrementally - no special "first time" vs "update" distinction
        # This handles both initial setup (providing configs gradually) and later changes
        result = await self._handle_incremental_update(
            state,
            extracted,
            cross_phase_updates,
            existing_model,
            existing_opt,
            existing_loss,
            existing_sched,
            existing_complex,
            complex_params
        )
        
        # Merge cross-phase updates (already included in result)
        return result

    def _missing_flags(
        self,
        model: Optional[Union[ModelParams, Dict[str, Any]]],
        optimizer: Optional[OptimizerConfig],
        loss: Optional[LossFunctionConfig],
        scheduler: Optional[SchedulerConfig],
        complex_params: Optional[Union[ComplexParams, Dict[str, Any]]],
        pretrained_model: Optional[str],
    ) -> tuple[bool, bool, bool, bool, bool, list[str], list[str]]:
        """Determine which major components are missing.

        Scheduler is treated as required just like model, optimizer, and loss.
        Returns: (model_missing, opt_missing, loss_missing, sched_missing, missing_model_params)
        """
        required_model_attrs = [
            "height", "width", "channels", "batch_size", "device", "dataset", "dataset_path"
        ]
        missing_model_params: list[str] = []

        if not model:
            # Require all base params, plus at least one of epochs/target_accuracy
            missing_model_params = required_model_attrs + ["epochs_or_target_accuracy"]
            model_missing = True
        else:
            # Check for missing or unset params (0 for ints, "" for strings, None for any)
            # Handle both dict (partial params) and ModelParams object
            for attr in required_model_attrs:
                if isinstance(model, dict):
                    value = model.get(attr)
                else:
                    value = getattr(model, attr, None)
                # Consider missing if: None, 0 (for numeric), or empty string
                if value is None or value == 0 or value == "":
                    missing_model_params.append(attr)

            # At least one of epochs or target_accuracy must be present
            if isinstance(model, dict):
                epochs_val = model.get("epochs")
                target_acc_val = model.get("target_accuracy")
            else:
                epochs_val = getattr(model, "epochs", None)
                target_acc_val = getattr(model, "target_accuracy", None)
            if (epochs_val is None or epochs_val == 0) and (target_acc_val is None or target_acc_val == 0):
                missing_model_params.append("epochs_or_target_accuracy")

            model_missing = len(missing_model_params) > 0
        
        opt_missing = not optimizer or not optimizer.optimizer_type
        loss_missing = not loss or not loss.loss_type
        sched_missing = not scheduler or not scheduler.scheduler_type

        # Complex params required only for YOLOX models
        requires_complex = bool(pretrained_model and pretrained_model.lower().startswith("yolox"))
        missing_complex_params: list[str] = []
        if not requires_complex:
            complex_missing = False
        else:
            required_complex_attrs = [
                "pretrained_weights", "num_classes", "scheduler", "warmup_epochs", "eval_interval", "data_workers"
            ]
            
            if not complex_params:
                missing_complex_params = required_complex_attrs
                complex_missing = True
            else:
                # Check for missing or unset params (0 for ints, "" for strings, None for any)
                # Handle both dict (partial params) and ModelParams object
                missing_complex_params = []
                for attr in required_complex_attrs:
                    if isinstance(complex_params, dict):
                        value = complex_params.get(attr)
                    else:
                        value = getattr(complex_params, attr, None)
                    # Consider missing if: None, 0 (for numeric), or empty string
                    if value is None or value == 0 or value == "":
                        missing_complex_params.append(attr)
                complex_missing = len(missing_complex_params) > 0

        return (
            model_missing,
            opt_missing,
            loss_missing,
            sched_missing,
            complex_missing,
            missing_model_params,
            missing_complex_params,
        )

    def _ensure_param_completeness(
        self,
        optimizer: Optional[OptimizerConfig],
        loss: Optional[LossFunctionConfig],
        scheduler: Optional[SchedulerConfig],
    ) -> tuple[Optional[OptimizerConfig], Optional[LossFunctionConfig], Optional[SchedulerConfig]]:
        """Fill in missing optimizer/loss/scheduler params using schema defaults.

        All optional params in the schema are treated as required for our flow; if
        the user didn't specify them, we populate them with sensible defaults.
        """
        # Optimizer params
        if optimizer is not None:
            opt_schema = get_optimizer_config(optimizer.optimizer_type) or {}
            opt_defaults = opt_schema.get("defaults", {})
            merged_opt_params: Dict[str, Any] = {**opt_defaults, **(optimizer.params or {})}
            optimizer.params = merged_opt_params

        # Loss params
        if loss is not None:
            loss_schema = get_loss_config(loss.loss_type) or {}
            loss_defaults = loss_schema.get("defaults", {})
            merged_loss_params: Dict[str, Any] = {**loss_defaults, **(loss.params or {})}
            loss.params = merged_loss_params

        # Scheduler params (if scheduler present)
        if scheduler is not None:
            sched_schema = get_scheduler_config(scheduler.scheduler_type) or {}
            sched_defaults = sched_schema.get("defaults", {})
            merged_sched_params: Dict[str, Any] = {**sched_defaults, **(scheduler.params or {})}
            scheduler.params = merged_sched_params

        return optimizer, loss, scheduler

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

    async def _extract_all_hints(self, state: NNGeneratorState) -> Dict[str, Any]:
        """Extract ALL hints from user message (architecture + training config).
        
        Uses DesignPhaseExtraction to capture everything in one call and returns a dict:
        `{'extraction': DesignPhaseExtraction, 'complex_params': Optional[ComplexParams|dict]}`
        """
        latest_user = self.get_last_user_message(state) or ""
        if not latest_user.strip():
            return {"extraction": DesignPhaseExtraction(), "complex_params": None}
        
        complex_params = None
        
        # ------------------------------------------------------------------
        # CONDITIONAL CONTEXT INJECTION
        # ------------------------------------------------------------------
        # We only inject context if the system specifically flagged that it was waiting 
        # for missing information (via training_params_needs_completion).
        context_prompt_section = ""
        
        if state.get("training_params_needs_completion"):
            last_ai_msg = self._get_last_ai_message_content(state)
            if last_ai_msg:
                context_prompt_section = (
                    f"\n\nCONTEXT (You previously said this to the user):\n"
                    f"\"\"\"{last_ai_msg}\"\"\"\n"
                    f"The user is likely answering the question 'Still missing: ...' from the context."
                    f"If the AI asked for 'optimizer' and user says 'adam', map it to optimizer.\n"
                )

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
            if isinstance(extracted, DesignPhaseExtraction):
                result = extracted
            if isinstance(extracted, dict):
                result = DesignPhaseExtraction.model_validate(extracted)
        except Exception:
            return {"extraction": DesignPhaseExtraction(), "complex_params": None}
        
        # Stage 2: always extract complex params
        if self.llm is not None and not state.get("manual_layers"):
            if state.get("pretrained_model"):
                extracted_pretrained_model = result.pretrained_model
                pretrained_modeL_in_state = state.get("pretrained_model")
                if extracted_pretrained_model is None:
                    extracted_pretrained_model = pretrained_modeL_in_state
                if extracted_pretrained_model and not extracted_pretrained_model.lower().startswith("yolo"):
                    return {'extraction': result, 'complex_params': None}
            complex_llm = self.llm.with_structured_output(ComplexParamsExtraction)
            cp_prompt = (
                "You are extracting YOLOX-specific complex training parameters from the user message.\n\n"
                "Extract ONLY fields you see if available else output empty object\n"
                f"{context_prompt_section}\n"
                "pretrained_weights: null unless the user provides an explicit file path. NEVER put model names (like 'yolox-s', 'resnet50') here. NEVER put dataset names (like 'coco') here. It must look like a file location.\n"
                f"\n\nUser message:\n{latest_user}"
            )
            try:
                extra = await complex_llm.ainvoke(cp_prompt)
                if extra and extra.complex_params:
                    complex_params = extra.complex_params
            except Exception:
                pass
        
        return {"extraction": result, "complex_params": complex_params}


    
    async def _handle_incremental_update(
        self,
        state: NNGeneratorState,
        extracted: DesignPhaseExtraction,
        cross_phase_updates: Dict[str, Any],
        existing_model: Optional[Union[ModelParams, Dict[str, Any]]],
        existing_opt: Optional[OptimizerConfig],
        existing_loss: Optional[LossFunctionConfig],
        existing_sched: Optional[SchedulerConfig],
        existing_complex: Optional[Union[ComplexParams, Dict[str, Any]]],
        complex_params: Optional["ComplexParams | Dict[str, Any]"] = None
    ) -> Dict[str, Any]:
        """Handle incremental updates to specific configuration pieces."""
        mcp = await get_mcp_helper()
        updated_parts = []
        state_updates: Dict[str, Any] = {"messages": []}
        architecture_file = state.get("architecture_file")
        pre_call_debug_text: Optional[str] = None
        
        try:
            # Use merged model_params from cross_phase_updates (already merged with existing)
            merged_model = cross_phase_updates.get("model_params")
            
            # Check if we have a complete ModelParams object
            if merged_model and isinstance(merged_model, ModelParams):
                mcp_params = {
                    "height": merged_model.height,
                    "width": merged_model.width,
                    "channels": merged_model.channels,
                    "batch_size": merged_model.batch_size,
                    "device": merged_model.device,
                    "dataset": merged_model.dataset,
                    "dataset_path": merged_model.dataset_path,
                    "file_path": architecture_file,
                }
                if merged_model.epochs is not None:
                    mcp_params["epochs"] = merged_model.epochs
                if merged_model.target_accuracy is not None:
                    mcp_params["target_accuracy"] = merged_model.target_accuracy
                mcp_response = await mcp.call("set_model_params", mcp_params)
                # Capture MCP response for debugging
                mcp_status_msg = f"[MCP Response] status={mcp_response.get('status')}, message={mcp_response.get('message')}, details={mcp_response.get('details')}"
                state_updates["messages"].append(self.format_message(mcp_status_msg))
                state_updates["model_params"] = merged_model
                existing_model = merged_model
                if extracted.model_params:  # Only report as updated if extraction provided new info
                    updated_parts.append("model parameters")
            elif merged_model and isinstance(merged_model, dict):
                # Partial model params (dict) - persist to state for accumulation across turns
                state_updates["model_params"] = merged_model
                existing_model = merged_model  # Update existing_model for _missing_flags check
                if extracted.model_params:
                    updated_parts.append("model parameters (partial)")
            elif existing_model and isinstance(existing_model, ModelParams):
                # Fallback: Apply complete model params from state if not already merged
                mcp_params = {
                    "height": existing_model.height,
                    "width": existing_model.width,
                    "channels": existing_model.channels,
                    "batch_size": existing_model.batch_size,
                    "device": existing_model.device,
                    "dataset": existing_model.dataset,
                    "dataset_path": existing_model.dataset_path,
                    "file_path": architecture_file,
                }
                if existing_model.epochs is not None:
                    mcp_params["epochs"] = existing_model.epochs
                if existing_model.target_accuracy is not None:
                    mcp_params["target_accuracy"] = existing_model.target_accuracy
                mcp_response = await mcp.call("set_model_params", mcp_params)

            
            # Use merged configs from cross_phase_updates (already merged with existing)
            merged_opt = cross_phase_updates.get("optimizer_config") or existing_opt
            merged_loss = cross_phase_updates.get("loss_config") or existing_loss
            merged_sched = cross_phase_updates.get("scheduler_config") or existing_sched
            merged_complex = cross_phase_updates.get("complex_params") or existing_complex

            # Ensure completeness for any present component
            if merged_opt or merged_loss or merged_sched:
                merged_opt, merged_loss, merged_sched = self._ensure_param_completeness(
                    merged_opt, merged_loss, merged_sched
                )

            # Apply all present components to MCP (eager apply)
            # This ensures first-time configs are applied immediately
            if merged_opt:
                result = await mcp.call("set_optimizer", {
                    "optimizer_type": merged_opt.optimizer_type,
                    "params": merged_opt.params or {},
                    "file_path": architecture_file,
                })
                if result.get("status") == "error":
                    raise ValueError(f"MCP failed to set optimizer: {result.get('message')}")
                state_updates["optimizer_config"] = merged_opt
                existing_opt = merged_opt
                if extracted.optimizer_config:  # Only add to updated_parts if newly provided
                    updated_parts.append("optimizer")

            if merged_loss:
                result = await mcp.call("set_loss_function", {
                    "loss_type": merged_loss.loss_type,
                    "params": merged_loss.params or {},
                    "file_path": architecture_file,
                })
                if result.get("status") == "error":
                    raise ValueError(f"MCP failed to set loss function: {result.get('message')}, details: {result.get('details')}")
                state_updates["loss_config"] = merged_loss
                existing_loss = merged_loss
                if extracted.loss_config:
                    updated_parts.append("loss function")

            if merged_sched:
                result = await mcp.call("set_scheduler", {
                    "scheduler_type": merged_sched.scheduler_type,
                    "params": merged_sched.params or {},
                    "file_path": architecture_file,
                })
                if result.get("status") == "error":
                    raise ValueError(f"MCP failed to set scheduler: {result.get('message')}")
                state_updates["scheduler_config"] = merged_sched
                existing_sched = merged_sched
                if extracted.scheduler_config:
                    updated_parts.append("scheduler")

            if merged_complex and isinstance(merged_complex, ComplexParams):

                mcp_response = await mcp.call("set_complex_params", {
                    "pretrained_weights": merged_complex.pretrained_weights,
                    "num_classes": merged_complex.num_classes,
                    "scheduler": merged_complex.scheduler,
                    "warmup_epochs": merged_complex.warmup_epochs,
                    "eval_interval": merged_complex.eval_interval,
                    "data_num_workers": merged_complex.data_workers,
                    "file_path": architecture_file,
                })

                state_updates["complex_params"] = merged_complex
                existing_complex = merged_complex
                if complex_params:  # Only report as updated if extraction provided new info
                    updated_parts.append("complex parameters")
            elif merged_complex and isinstance(merged_complex, dict):
                # Partial model params (dict) - persist to state for accumulation across turns
                state_updates["complex_params"] = merged_complex
                existing_complex = merged_complex  # Update existing_complex for _missing_flags check
                if complex_params:
                    updated_parts.append("complex parameters (partial)")
            elif existing_complex and isinstance(existing_complex, ComplexParams):
                # Fallback: Apply complete complex params from state if not already merged
                mcp_response = await mcp.call("set_complex_params", {
                    "pretrained_weights": existing_complex.pretrained_weights,
                    "num_classes": existing_complex.num_classes,
                    "scheduler": existing_complex.scheduler,
                    "warmup_epochs": existing_complex.warmup_epochs,
                    "eval_interval": existing_complex.eval_interval,
                    "data_num_workers": existing_complex.data_workers,
                    "file_path": architecture_file,
                })


            
            
            # Check if all required pieces are now present (including scheduler)
            model_missing, opt_missing, loss_missing, sched_missing, complex_missing, missing_model_params, missing_complex_params = self._missing_flags(
                existing_model,
                existing_opt,
                existing_loss,
                existing_sched,
                existing_complex,
                state.get("pretrained_model")
            )
            all_complete = not (model_missing or opt_missing or loss_missing or sched_missing or complex_missing)
            
            # Build response message (append instead of overwrite so earlier debug remains visible)
            if all_complete:
                # Embed the pre-call debug at the top if present so user sees exact params used
                debug_header = f"{pre_call_debug_text}\n\n" if pre_call_debug_text else ""
                message = (
                    debug_header +
                    f"✅ **Configuration Updated: {', '.join(updated_parts)}**\n\n"
                    "All required training parameters are now set. Ready for design confirmation." 
                )
                state_updates["training_params_applied"] = True
                state_updates["training_params_needs_completion"] = False  # Clear completion flag
                state_updates["completed_stages"] = self._append_stage(state, "training_params_applied")
            else:
                missing_list = []
                if model_missing:
                    if missing_model_params:
                        # Provide a more natural, human-readable message for missing params
                        human_names = {
                            "height": "image height (px)",
                            "width": "image width (px)",
                            "channels": "channels",
                            "batch_size": "batch size",
                            "device": "device",
                            "dataset": "dataset",
                            "dataset_path": "dataset path",
                        }

                        # If user needs to provide either epochs or a target accuracy, handle gracefully
                        if "epochs_or_target_accuracy" in missing_model_params:
                            # prepare the list of real missing params excluding the special union
                            remaining = [p for p in missing_model_params if p != "epochs_or_target_accuracy"]
                            if remaining:
                                pretty = [human_names.get(p, p) for p in remaining]
                                missing_list.append(
                                    "Model parameters missing — "
                                    f"{', '.join(pretty)}. "
                                    "Additionally, please provide either `num_epochs` or `target_accuracy` (at least one)."
                                )
                            else:
                                missing_list.append(
                                    "Model parameters missing — please provide either `num_epochs` or `target_accuracy` (at least one)."
                                )
                        else:
                            pretty = [human_names.get(p, p) for p in missing_model_params]
                            missing_list.append(f"Model parameters missing — {', '.join(pretty)}.")
                    else:
                        missing_list.append("model parameters")
                if opt_missing:
                    missing_list.append("optimizer")
                if loss_missing:
                    missing_list.append("loss function")
                if sched_missing:
                    missing_list.append("scheduler")
                if complex_missing:
                    if missing_complex_params:
                        missing_list.append(f"complex parameters ({', '.join(missing_complex_params)})")
                    else:
                        missing_list.append("complex parameters")
                debug_header = f"{pre_call_debug_text}\n\n" if pre_call_debug_text else ""
                message = (
                    debug_header +
                    f"✅ **Configuration Updated: {', '.join(updated_parts)}**\n\n"
                    f"Still missing: {', '.join(missing_list)}\n\n"
                    "Please provide the remaining configuration to complete setup."
                )
                state_updates["training_params_needs_completion"] = True  # Return to this node after user input

            # Append the status message (do NOT overwrite earlier debug messages)
            state_updates["messages"].append(self.format_message(message))
            # Always wait for user input after updating configuration
            state_updates["needs_user_input"] = True
            
            # Merge all cross-phase updates (including architecture hints if any)
            return {**state_updates, **cross_phase_updates}
            
        except Exception as e:
            return self.create_error_response(e, "incremental_configuration", "high")