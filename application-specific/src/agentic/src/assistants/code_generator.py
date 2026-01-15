# src/assistants/code_generator.py
"""Code Generator - Generates PyTorch and SystemC code."""

from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableConfig
from .base_assistant import BaseAssistant
from src.agentic.src.core.state import NNGeneratorState
from src.agentic.src.mcp import get_mcp_helper
from pathlib import Path


class CodeGenerator(BaseAssistant):
    """
    Generates training and inference code.
    
    This is a TOOL-HEAVY assistant with minimal LLM decision-making.
    
    Responsibilities:
    - Verify prerequisites (architecture + config exist)
    - Generate PyTorch training code via MCP
    - Generate requirements.txt
    
    Decision-making:
    - Minimal - just prerequisite checking
    - Execution is deterministic (call generate_pytorch tool)
    """
    
    def __init__(self, llm):
        super().__init__(name="code_generator", llm=llm)
    
    async def __call__(self, state: NNGeneratorState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Generate code from architecture and configuration."""
        
        arch_file = state.get("architecture_file")
        optimizer_config = state.get("optimizer_config")
        loss_config = state.get("loss_config")
        model_params = state.get("model_params")
        project_path = state.get("project_path")
        
        # Check prerequisites
        if not arch_file:
            return {
                "messages": [self.format_message(
                    "❌ **Cannot Generate Code**\n\n"
                    "**Missing:** Architecture has not been designed yet.\n\n"
                    "Please design the neural network architecture, then I can generate the code\n\n"
                )],
                "needs_user_input": False
            }
        
        has_config = bool(
            optimizer_config
            or loss_config
            or model_params
        )

        if not has_config:
            return {
                "messages": [self.format_message(
                    "❌ **Cannot Generate Code**\n\n"
                    "**Missing:** Training configuration has not been set.\n\n"
                    "**Required Steps:**\n"
                    "1. Configure training parameters (optimizer, loss, epochs, etc.)\n"
                    "2. Then I can generate the code\n\n"
                    "Would you like me to help configure training?"
                )],
                "needs_user_input": False
            }
        
        if not project_path:
            return {
                "messages": [self.format_message(
                    "❌ **Cannot Generate Code**\n\n"
                    "**Missing:** Project has not been created.\n\n"
                    "This shouldn't happen (architecture requires project). Please report this issue."
                )],
                "needs_user_input": False
            }
        
        # All prerequisites met - generate code
        return await self._generate_code(state)
    
    async def _generate_code(
        self,
        state: NNGeneratorState
    ) -> Dict[str, Any]:
        """Generate code via MCP tools.
        
        The generate_pytorch tool automatically detects model type:
        - Manual: if no pretrained_model
        - YOLOX: if pretrained_model starts with "yolox"
        - Pretrained: if pretrained_model exists
        """
        
        # Read from state
        arch_file = state.get("architecture_file")
        project_path = state.get("project_path")
        
        # Get global MCP helper
        mcp = await get_mcp_helper()
        
        try:
            # Generate PyTorch code (tool handles model type detection)
            pytorch_result = await mcp.call("generate_pytorch", {
                "architecture_file": Path(str(arch_file)),
                "output_dir": Path(str(project_path)),
                "model_name": "NNModel",
                "include_requirements": True
            })

            # Extract project_output from result - this is the project root
            details = pytorch_result.get("details", {}) if isinstance(pytorch_result, dict) else {}
            project_output = details.get("output_dir")
            

            # Prepare display info
            generated_files = [("Project root", project_output)]
            
            # Show some generated files for user feedback
            gen_files = details.get("generated_files", [])
            if gen_files:
                generated_files.append(("Generated files", f"{len(gen_files)} files"))
            
            # Format success message
            message = "✅ **Code Generation Complete**\n\n"
            message += "**Generated Files:**\n"
            for name, path in generated_files:
                message += f"- **{name}**\n  `{path}`\n"
            

            
            message += "\n**Next Steps:**\n"
            message += "Ready to start training! The code is fully configured and ready to run.\n"
            message += "Say 'start training' when ready."
            
            result: Dict[str, Any] = {
                "messages": [self.format_message(message)],
                "needs_user_input": True,  # Stop and wait for user acknowledgment before training
                **self.update_stage(state, "code_generated")
            }
            # Persist project_output (project root) into state
            if project_output:
                result["project_output"] = project_output
            return result
            
        except Exception as e:
            return self.create_error_response(e, "code_generation", "high")