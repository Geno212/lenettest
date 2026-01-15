# src/assistants/project_manager.py
"""Project Manager - Creates and manages project directories."""

from typing import Dict, Any, Optional, Literal
import os
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from .base_assistant import BaseAssistant
from src.agentic.src.logic.project_logic import ProjectManagerLogic
from src.agentic.src.core.state import NNGeneratorState, GraphPhase
from src.agentic.src.mcp import get_mcp_helper
from pydantic import BaseModel, Field
from src.agentic.src.assistants.cross_phase_extraction import (
    CrossPhaseExtraction,
    ComplexParamsExtraction,
    apply_cross_phase_merge,
    set_intent_classifier_llm,
)


class ProjectInfoUpdate(BaseModel):
    """Structured subset of project information that can be inferred.

    This is used when state doesn't already contain project_name/path and
    we want the LLM to infer them from the latest user message.
    """

    project_name: Optional[str] = None
    output_dir: Optional[str] = None


class ExistingDirectoryChoice(BaseModel):
    """User's choice when project directory already exists."""
    
    choice: Literal["use_existing", "create_new", "unclear"] = Field(
        description="User's decision: use_existing (use the existing project), "
        "create_new (create with a different name), or unclear (response was ambiguous)"
    )
    new_project_name: Optional[str] = Field(
        None,
        description="If choice is 'create_new', the new project name provided by user (if any)"
    )


class ProjectManager(BaseAssistant):
    """Manages project creation and organization.

    New flow:
    - Prefer project_name/project_path already stored in NNGeneratorState.
    - Otherwise, infer project info from tool params and latest user message
      using structured LLM output.
    - Validate the target path, handle existing directory conflicts, and
      create the project via MCP. On successful completion, advance
      current_phase to DESIGN.
    """

    def __init__(self, llm):
        super().__init__(name="project_manager", llm=llm)
        self.logic = ProjectManagerLogic()

        # LLM with structured output for project info extraction
        self.structured_project_llm = llm.with_structured_output(ProjectInfoUpdate)
        
        # LLM for cross-phase extraction (architecture + training hints)
        self.cross_phase_llm = llm.with_structured_output(CrossPhaseExtraction)
        self.complex_llm = llm.with_structured_output(ComplexParamsExtraction)
        
        # LLM for parsing existing directory choice
        self.existing_dir_choice_llm = llm.with_structured_output(ExistingDirectoryChoice)
        
        # Set the global LLM for intent classification
        set_intent_classifier_llm(llm)

    async def __call__(self, state: NNGeneratorState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Create and manage project structure."""
        # ------------------------------------------------------------------
        # 0a) Check if user wants direct RTL synthesis with existing model
        # ------------------------------------------------------------------
        latest_msg = self.get_last_user_message(state) or ""
        if self._detect_direct_rtl_intent(latest_msg):
            import re
            # Try to extract path immediately and store in state
            path_patterns = [
                r'(/[\w\-_/]+\.pt[h]?)',  # Unix/Linux paths like /home/ubuntu/model.pt
                r'(~[\w\-_/]+\.pt[h]?)',  # Home directory paths like ~/model.pt
                r'([A-Za-z]:[\\\/][\w\-_\\\/. ]+\.pt[h]?)',  # Windows paths
                r'(\.{1,2}[\/\\][\w\-_\/\\. ]+\.pt[h]?)',  # Relative paths
                r'at\s+([^\s]+\.pt[h]?)',  # "at /path/to/model.pt"
            ]
            
            extracted_path = None
            for pattern in path_patterns:
                match = re.search(pattern, latest_msg)
                if match:
                    extracted_path = match.group(1).strip()
                    break
            
            # Store path if found
            result = {
                "awaiting_direct_rtl_upload": True,
                "needs_user_input": False,  # Let next node handle it
            }
            if extracted_path:
                result["pretrained_model_path"] = extracted_path
            
            return result
        
        # ------------------------------------------------------------------
        # 0b) Extract cross-phase hints first (architecture + training config)
        #    This captures any architecture/training details mentioned early
        # ------------------------------------------------------------------
        hints = await self._extract_cross_phase_hints(state)
        extracted = hints['extraction']
        complex_params = hints['complex_params']
        cross_phase_updates = await apply_cross_phase_merge(
            state=state,
            extracted=extracted,
            complex_params=complex_params,
            user_message=self.get_last_user_message(state) or ""
        )
        
        # Helper to perform validation + creation given name/output_dir
        # output_dir can be None (use cwd), empty string (use cwd), or a specific path
        async def _validate_and_create(name: str, output_dir: Optional[str]) -> Dict[str, Any]:
            # Use current directory if output_dir is None or empty
            target_dir = output_dir if output_dir else os.getcwd()
            validation = self.logic.validate_project_path(target_dir, name)

            # Existing directory -> delegate to existing handler
            if validation["exists"]:
                result = await self._handle_existing_directory(
                    state, name, output_dir, validation, cross_phase_updates
                )
                # Ensure we move to DESIGN phase on successful project selection
                if not result.get("needs_user_input"):
                    result.setdefault("current_phase", GraphPhase.DESIGN)
                    # Merge cross-phase updates
                    result.update(cross_phase_updates)
                return result

            # Permission issues
            if not validation["writable"]:
                base_response = {
                    "messages": [self.format_message(
                        "❌ **Permission Error**\n\n"
                        f"Cannot write to directory: `{target_dir}`\n\n"
                        f"**Suggestions:**\n"
                        f"1. Check directory permissions\n"
                        f"2. Choose a different location\n"
                        f"3. Run with appropriate permissions\n\n"
                        f"Please provide a different output directory or fix permissions."
                    )],
                    "needs_user_input": True
                }
                # Merge cross-phase updates to persist extracted fields in state
                return {**base_response, **cross_phase_updates}

            # All checks passed - create project
            result = await self._create_project(state, name, output_dir, cross_phase_updates)
            # After successful creation, advance to DESIGN phase
            if not result.get("needs_user_input"):
                result.setdefault("current_phase", GraphPhase.DESIGN)
            return result

        # ------------------------------------------------------------------
        # 1) Check state first (for return visits after existing dir choice, etc.)
        # ------------------------------------------------------------------
        state_project_name = state.get("project_name")
        state_project_path = state.get("project_path")

        # Also check if cross-phase extraction found project_name
        if not state_project_name and cross_phase_updates.get("project_name"):
            state_project_name = cross_phase_updates["project_name"]
            state_project_path = cross_phase_updates.get("project_path")

        # If project_name already exists in state, use it
        # If project_path exists, extract output_dir from it; otherwise use None (defaults to cwd)
        if state_project_name:
            output_dir = os.path.dirname(state_project_path) if state_project_path else None
            return await _validate_and_create(state_project_name, output_dir)

        # ------------------------------------------------------------------
        # 2) Otherwise, try to infer from user messages
        # ------------------------------------------------------------------

        project_name = state_project_name
        output_dir = None  # None means use default (cwd)

        # If we still don't have a clear project_name, ask the LLM to extract
        # it (and optionally output_dir) from the latest user message.
        if not project_name:
            inferred = await self._extract_project_info_from_messages(state)
            if inferred.project_name:
                project_name = inferred.project_name
            if inferred.output_dir:
                output_dir = inferred.output_dir

        if not project_name:
            base_msg = {
                "messages": [self.format_message(
                    "Please provide a project name (output directory is optional).\n\n"
                )],
                "needs_user_input": True,
            }
            # Merge cross-phase updates to persist extracted fields in state
            return {**base_msg, **cross_phase_updates}

        # Sanitize name once we have it
        project_name = self.logic.sanitize_project_name(project_name)

        # output_dir is None by default (use cwd), or set if user provided it
        return await _validate_and_create(project_name, output_dir)


    async def _extract_cross_phase_hints(self, state: NNGeneratorState) -> Dict[str, Any]:
        """Extract architecture and training hints from user message (cross-phase extraction).
        
        Returns dict with 'extraction': CrossPhaseExtraction and 'complex_params': optional ComplexParams
        """
        latest_user = self.get_last_user_message(state) or ""
        if not latest_user.strip():
            return {'extraction': CrossPhaseExtraction(), 'complex_params': None}
        
        complex_params = None
        
        prompt = (
                    "Extract Neural Network configuration from the user message into valid JSON.\n\n"
                    "### TARGETS\n"
                    "- Project: project_name, output_dir\n"
                    "- Architecture: pretrained_model (valid name only, NO file paths), manual_layers\n"
                    "- Training: H/W/C, device, dataset/path, batch_size, epochs, optimizer, loss, scheduler\n\n"
                    "### RULES\n"
                    "1. Partial Updates: Extract ONLY confident fields. Leave others null.\n"
                    "2. Normalization: Correct typos to match standard PyTorch names (e.g., 'adam' -> 'Adam').\n"
                    "3. Scheduler Special Case: If explicitly declined, set scheduler_type='None'.\n"
                    "4. Manual Layers: Must include 'layer_type', 'params', and 'position'.\n"
                    "   Example: {\"manual_layers\": [{\"layer_type\": \"Conv2d\", \"params\": {\"in_channels\": 3}, \"position\": 0}]}\n\n"
                    "### EXAMPLES\n"
                    "User: 'Use resnet18 for 224x224x3 images, batch 32, adam opt'\n"
                    "Output: {\"pretrained_model\": \"resnet18\", \"model_params\": {\"width\": 224, \"height\": 224, \"channels\": 3, \"batch_size\": 32}, \"optimizer_config\": {\"optimizer_type\": \"Adam\"}}\n\n"
                    f"### INPUT\n{latest_user}"
                )
        
        try:
            extracted = await self.cross_phase_llm.ainvoke(prompt)
            if isinstance(extracted, CrossPhaseExtraction):
                result = extracted
            if isinstance(extracted, dict):
                result = CrossPhaseExtraction.model_validate(extracted)
        except Exception:
            return {'extraction': CrossPhaseExtraction(), 'complex_params': None}
        
        # Stage 2: always extract complex params
        if self.llm is not None and not result.manual_layers:
            if result.pretrained_model:
                if not result.pretrained_model.lower().startswith("yolo"):
                    return {'extraction': result, 'complex_params': None}
            complex_llm = self.llm.with_structured_output(ComplexParamsExtraction)
            cp_prompt = (
                "You are extracting YOLOX-specific complex training parameters from the user message.\n\n"
                "Extract ONLY fields you see if available else output empty object\n"
                "pretrained_weights: null unless the user provides an explicit file path. NEVER put model names (like 'yolox-s', 'resnet50') here. NEVER put dataset names (like 'coco') here. It must look like a file location.\n"
                f"\n\nUser message:\n{latest_user}"
            )
            try:
                extra = await complex_llm.ainvoke(cp_prompt)
                if extra and extra.complex_params:
                    complex_params = extra.complex_params
            except Exception:
                pass
        
        return {'extraction': result, 'complex_params': complex_params}
    
    async def _extract_project_info_from_messages(
        self, state: NNGeneratorState
    ) -> ProjectInfoUpdate:
        """Use the LLM to infer project_name/output_dir from conversation.
        """
        # Find latest human message content
        latest_user: Optional[str] = None
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, str):
                    latest_user = content
                    break
                latest_user = str(content)
                break

        if latest_user is None:
            latest_user = ""

        prompt = (
            "From the user's latest message, extract a project name and an "
            "optional output directory if they are clearly specified or can "
            "be confidently inferred. If not provided, leave fields as null.\n\n"
            f"User message:\n{latest_user}"
        )

        try:
            update_obj = await self.structured_project_llm.ainvoke(prompt)
            if isinstance(update_obj, ProjectInfoUpdate):
                return update_obj
            if isinstance(update_obj, dict):
                return ProjectInfoUpdate.model_validate(update_obj)
        except Exception:
            return ProjectInfoUpdate()

        return ProjectInfoUpdate()
    
    async def _handle_existing_directory(
        self,
        state: NNGeneratorState,
        project_name: str,
        output_dir: Optional[str],
        validation: Dict[str, Any],
        cross_phase_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle existing directory conflict using LLM to parse user's choice.
        
        Flow:
        1. First call: Prompt user for choice, set awaiting_existing_dir_choice=True
        2. Second call (after user responds): Use LLM to parse response and take action
        """
        base_dir = output_dir if output_dir else os.getcwd()
        abs_path = validation.get("absolute_path") or self.logic.get_project_path(base_dir, project_name)
        
        # Check if this is a return visit (user already responded)
        if state.get("awaiting_existing_dir_choice"):
            user_response = self.get_last_user_message(state) or ""
            
            # Use LLM to parse user's choice
            choice_result = await self._parse_existing_dir_choice(user_response, abs_path)
            
            # Case 1: User wants to use existing directory
            if choice_result.choice == "use_existing":
                # Get cross-phase updates
                hints = await self._extract_cross_phase_hints(state)
                extracted = hints['extraction']
                complex_params = hints['complex_params']
                cross_phase_updates = await apply_cross_phase_merge(
                    state=state,
                    extracted=extracted,
                    complex_params=complex_params,
                    user_message=user_response
                )
                
                # Set project info and advance to DESIGN phase
                result = {
                    "project_name": project_name,
                    "project_path": abs_path,
                    "project_needs_completion": False,  # Clear completion flag
                    "awaiting_existing_dir_choice": False,  # Clear awaiting flag
                    "needs_user_input": True,  # Wait for acknowledgment
                    "current_phase": GraphPhase.DESIGN,
                    "messages": [self.format_message(
                        f"✅ **Using Existing Project**\n\n"
                        f"**Name:** `{project_name}`\n"
                        f"**Location:** `{abs_path}`\n\n"
                        f"Project is ready! Next step: Design the architecture."
                    )],
                    **self.update_stage(state, "project_selected")
                }
                result.update(cross_phase_updates)
                return result
            
            # Case 2: User wants to create with new name
            elif choice_result.choice == "create_new":
                new_name = choice_result.new_project_name
                
                if not new_name:
                    # Ask for the new name
                    base_response = {
                        "messages": [self.format_message(
                            "Please provide the new project name.\n\n"
                            "Example: 'my_new_project' or 'create project called my_new_project'"
                        )],
                        "needs_user_input": True,
                        "project_needs_completion": True,
                        "awaiting_existing_dir_choice": True,
                    }
                    # Merge cross-phase updates to persist extracted fields in state
                    return {**base_response, **cross_phase_updates}
                
                new_name = self.logic.sanitize_project_name(new_name)
                
                # Get cross-phase updates
                hints = await self._extract_cross_phase_hints(state)
                extracted = hints['extraction']
                complex_params = hints['complex_params']
                cross_phase_updates = await apply_cross_phase_merge(
                    state=state,
                    extracted=extracted,
                    complex_params=complex_params,
                    user_message=user_response
                )
                
                # Validate and create with new name
                new_validation = self.logic.validate_project_path(base_dir, new_name)
                
                if new_validation["exists"]:
                    # Still exists with new name - inform user and ask again
                    base_response = {
                        "messages": [self.format_message(
                            f"❌ **Project Already Exists**\n\n"
                            f"A project named `{new_name}` already exists at: `{new_validation.get('absolute_path')}`\n\n"
                            f"Please choose a different project name."
                        )],
                        "needs_user_input": True,
                        "project_needs_completion": True,
                        "awaiting_existing_dir_choice": True,
                    }
                    # Merge cross-phase updates to persist extracted fields in state
                    return {**base_response, **cross_phase_updates}
                
                if not new_validation["writable"]:
                    base_response = {
                        "messages": [self.format_message(
                            f"❌ **Permission Error**\n\n"
                            f"Cannot write to directory: `{base_dir}`\n\n"
                            f"Please provide a different location or fix permissions."
                        )],
                        "needs_user_input": True,
                        "awaiting_existing_dir_choice": False,
                    }
                    # Merge cross-phase updates to persist extracted fields in state
                    return {**base_response, **cross_phase_updates}
                
                # Create project with new name
                return await self._create_project(state, new_name, output_dir, cross_phase_updates)
            
            # Case 3: Unclear response - ask again
            else:
                base_response = {
                    "messages": [self.format_message(
                        f"I didn't understand your response. The directory `{abs_path}` already exists.\n\n"
                        "Please reply clearly:\n"
                        "- To use the existing project, say: 'use existing' or 'use it'\n"
                        "- To create with a new name, say: 'create new name: my_project' or 'make a new one called my_project'"
                    )],
                    "needs_user_input": True,
                    "project_needs_completion": True,
                    "awaiting_existing_dir_choice": True,
                }
                # Merge cross-phase updates to persist extracted fields in state
                return {**base_response, **cross_phase_updates}
        
        # First visit: Ask user what to do
        base_response = {
            "messages": [self.format_message(
                "A project directory already exists at: `{}`.\n\n"
                "What would you like to do?\n"
                "- Use the existing project\n"
                "- Create a new project with a different name"
                .format(abs_path)
            )],
            "needs_user_input": True,
            "project_needs_completion": True,  # Return to this node after user input
            "awaiting_existing_dir_choice": True,  # Mark that we're waiting for this specific choice
            "project_name": project_name,  # Preserve project name for return visit
            "project_path": abs_path,  # Preserve project path for return visit
        }
        # Merge cross-phase updates to persist extracted fields in state
        return {**base_response, **cross_phase_updates}
    
    async def _parse_existing_dir_choice(
        self,
        user_response: str,
        existing_path: str
    ) -> ExistingDirectoryChoice:
        """Use LLM to parse user's choice about existing directory.
        
        Args:
            user_response: User's message
            existing_path: Path to the existing directory
            
        Returns:
            ExistingDirectoryChoice with parsed intent
        """
        prompt = (
            f"A project directory already exists at: {existing_path}\n\n"
            f"The user was asked to choose between:\n"
            f"1. Use the existing project directory\n"
            f"2. Create a new project with a different name\n\n"
            f"User's response:\n{user_response}\n\n"
            f"Parse the user's intent:\n"
            f"- If they want to use the existing directory, return choice='use_existing'\n"
            f"- If they want to create a new project, return choice='create_new' and extract the new project name if provided\n"
            f"- If the response is unclear or ambiguous, return choice='unclear'"
        )
        
        try:
            result = await self.existing_dir_choice_llm.ainvoke(prompt)
            if isinstance(result, ExistingDirectoryChoice):
                return result
            if isinstance(result, dict):
                return ExistingDirectoryChoice.model_validate(result)
        except Exception:
            return ExistingDirectoryChoice(choice="unclear", new_project_name=None)
        
        return ExistingDirectoryChoice(choice="unclear", new_project_name=None)
    
    def _detect_direct_rtl_intent(self, message: str) -> bool:
        """Detect if user wants to go directly to RTL synthesis with existing model."""
        if not message:
            return False
        
        msg_lower = message.lower()
        
        # Keywords indicating direct RTL synthesis
        rtl_keywords = ['rtl', 'synthesis', 'synthesize', 'verilog', 'vhdl', 'catapult', 'hardware']
        model_keywords = ['trained model', '.pt', '.pth', 'existing model', 'already trained', 'pre-trained model file']
        direct_keywords = ['directly', 'skip training', 'without training', 'already have', 'i have a']
        
        # Check for combinations
        has_rtl = any(keyword in msg_lower for keyword in rtl_keywords)
        has_model = any(keyword in msg_lower for keyword in model_keywords)
        has_direct = any(keyword in msg_lower for keyword in direct_keywords)
        
        # Strong indicator: mentions both RTL/synthesis AND existing model
        if has_rtl and (has_model or has_direct):
            return True
        
        # Also check for file path patterns (.pt or .pth files)
        if has_rtl and ('.pt' in msg_lower or '.pth' in msg_lower):
            return True
        
        return False
    
    async def _create_project(
        self,
        state: NNGeneratorState,
        project_name: str,
        output_dir: Optional[str],
        cross_phase_updates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create project via MCP tools."""
        # Get global MCP helper
        mcp = await get_mcp_helper()
        
        try:
            # Build MCP call parameters
            mcp_params = {
                "name": project_name,
                "description": f"Neural network project: {project_name}",
                "author": "User",
            }
            
            # Only add output_dir if explicitly provided (otherwise use cwd default)
            if output_dir:
                mcp_params["output_dir"] = output_dir
            
            # Call MCP tool
            result = await mcp.call("create_project", mcp_params)
            
            # Get project info - use output_dir if provided, otherwise use cwd
            target_dir = output_dir if output_dir else os.getcwd()
            project_path = self.logic.get_project_path(target_dir, project_name)
            
            # Verify creation
            try:
                project_info = await mcp.call("project_info", {
                    "path": project_path
                })
            except:
                project_info = None
            
            # Format success message
            # Pass the actual project name so the printed structure shows it
            structure = self.logic.get_project_structure(project_name)
            structure_str = self._format_structure(structure)
            
            # Build base response
            base_response = {
                "project_name": project_name,
                "project_path": project_path,
                "project_needs_completion": False,  # Clear completion flag
                "awaiting_existing_dir_choice": False,  # Clear awaiting flag (in case called from create_new path)
                "needs_user_input": True,  # Stop and wait for user acknowledgment
                "current_phase": GraphPhase.DESIGN,  # Ready for next phase
                "messages": [self.format_message(
                    f"✅ **Project Created Successfully**\n\n"
                    f"**Name:** `{project_name}`\n"
                    f"**Location:** `{project_path}`\n\n"
                    f"**Directory Structure:**\n"
                    f"```\n{structure_str}\n```\n\n"
                    f"Project is ready! Next step: Design the architecture."
                )],
                **self.update_stage(state, "project_created")
            }
            
            # Merge cross-phase hints if present
            if cross_phase_updates:
                base_response = {**base_response, **cross_phase_updates}
            
            return base_response
            
        except Exception as e:
            return self.create_error_response(e, "project_creation", "high")
    
    def _format_structure(self, structure: Dict[str, Any]) -> str:
        """Format project structure for display."""
        lines = [f"{structure['name']}/"]
        for subdir in structure.get("subdirs", []):
            lines.append(f"├── {subdir}/")
        return "\n".join(lines)