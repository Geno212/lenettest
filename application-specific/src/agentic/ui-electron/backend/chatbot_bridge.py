"""
Chatbot Bridge for Electron UI

This is a simplified bridge that directly mimics run_chatbot.py
Communication happens via JSON over stdin/stdout with Electron.

IMPORTANT: stdout is used ONLY for JSON messages.
All other output (prints, logs, debug messages) goes to stderr.
This prevents non-JSON text from corrupting the message stream.

The Electron app sends JSON messages like:
{"type": "message", "content": "user message here"}
{"type": "status"}
{"type": "quit"}

The bridge responds with JSON like:
{"type": "initialized", "data": {"session_id": "...", "thread_id": "..."}}
{"type": "response", "content": "...", "state": {...}}
{"type": "status", "data": {...}}
{"type": "error", "message": "..."}
"""

import asyncio
import json
import sys
import os
import warnings
from pathlib import Path

# Suppress deprecation warnings from Google protobuf
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google._upb._message")

# CRITICAL: Set UTF-8 encoding FIRST, before any other imports
# This prevents Unicode errors with emoji characters in Windows
if sys.platform == 'win32':
    # Set environment variable
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Reconfigure stdout and stderr to use UTF-8
    import codecs
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Save original stdout for JSON messages
_original_stdout = sys.stdout

# Redirect all stdout to stderr to prevent non-JSON output from polluting stdout
# We'll explicitly write JSON to _original_stdout
class StderrRedirector:
    """Redirects stdout to stderr, except for our explicit JSON messages"""
    def write(self, text):
        _original_stdout.flush()
        sys.stderr.write(text)
        sys.stderr.flush()
    
    def flush(self):
        sys.stderr.flush()

# Redirect stdout (libraries will print to stderr instead)
sys.stdout = StderrRedirector()

from dotenv import load_dotenv

# Add project root to Python path
# This file is at: src/agentic/ui-electron/backend/chatbot_bridge.py
# Project root is 4 levels up: ../../../../
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
backend_env = Path(__file__).parent / ".env"
if backend_env.exists():
    load_dotenv(backend_env)

from src.agentic.src.core.graph import create_graph
from src.agentic.src.core.state import create_initial_state
from src.agentic.src.utils.graph_utils import (
    create_user_message_input,
    format_graph_output,
    create_conversation_config
)
from src.agentic.src.mcp import get_mcp_helper, cleanup_global_mcp


class ChatbotBridge:
    """Bridge between Electron UI and the chatbot graph - mimics run_chatbot.py exactly"""
    
    def __init__(self):
        self.graph = None
        self.session_id = None
        self.thread_id = None
        self.first_message = True
        self.initial_state = None
        
    async def initialize(self):
        """Initialize the graph and MCP - same as run_chatbot.py"""
        try:
            # Create graph (initializes global MCP)
            self.graph = await create_graph()
            
            # Get MCP helper for session info
            mcp = await get_mcp_helper()
            self.session_id = mcp.client.session_id if mcp.client else "unknown"
            self.thread_id = f"mcp_{self.session_id[:8] if self.session_id else 'default'}"
            
            # Create initial state
            self.initial_state = create_initial_state()
            
            # Send initialization success
            self.send_message({
                "type": "initialized",
                "data": {
                    "session_id": self.session_id,
                    "thread_id": self.thread_id
                }
            })
            
        except UnicodeEncodeError as e:
            # Handle Unicode encoding errors specially
            error_msg = f"Unicode encoding error: {str(e)}. Try setting PYTHONIOENCODING=utf-8"
            self.send_message({
                "type": "error",
                "message": error_msg
            })
            raise
        except Exception as e:
            error_msg = f"Failed to initialize: {str(e)}"
            self.send_message({
                "type": "error",
                "message": error_msg
            })
            raise
    
    def _get_architecture_summary(self, state_dict):
        """Construct architecture summary from new state structure"""
        # Try to get existing summary (backward compatibility)
        if state_dict.get("architecture_summary"):
            return state_dict.get("architecture_summary")
            
        # Construct from new fields
        pretrained = state_dict.get("pretrained_model")
        manual_layers = state_dict.get("manual_layers", [])
        
        if pretrained:
            return {
                "type": "Pretrained",
                "base_model": pretrained,
                "parameters": "Unknown" # could be refined if available
            }
        elif manual_layers:
            return {
                "type": "Custom",
                "base_model": "Custom CNN",
                "parameters": f"{len(manual_layers)} layers"
            }
        return None

    def _get_dialog_state(self, state_dict):
        """Construct dialog state/active node from state dictionary"""
        # RTL synthesis flow - use status flags (most reliable)
        if state_dict.get("rtl_synthesis_complete"):
            return ["synthesize_rtl_node"]
        if state_dict.get("awaiting_rtl_build"):
            return ["verify_hls_node"]  # Verify done, waiting for build
        if state_dict.get("awaiting_hls_verify"):
            return ["configure_hls_node"]  # Config done, waiting for verify
        if state_dict.get("awaiting_hls_config"):
            return ["rtl_synthesis_node"]  # Parameters set, starting config
        if state_dict.get("awaiting_rtl_synthesis"):
            return ["direct_rtl_upload_node"]  # Model uploaded, collecting params
        
        # Check current_node as fallback
        current_node = state_dict.get("current_node")
        if current_node and isinstance(current_node, str) and current_node != "master_triage_router":
            return [current_node]
        
        # Try to get existing dialog_state (backward compatibility)
        dialog_state = state_dict.get("dialog_state")
        if dialog_state and isinstance(dialog_state, list) and len(dialog_state) > 0:
            # Don't return master_triage_router if we have other info
            if dialog_state != ["master_triage_router"]:
                return dialog_state
            
        current_stage = state_dict.get("current_stage")
        if not current_stage:
            return ["master_triage_router"]
            
        # Map stages to graph nodes for visualization
        stage_map = {
            "project_created": "create_project_node",
            "project_selected": "create_project_node",
            "architecture_defined": "design_arch_node",
            "configuration_set": "config_params_node",
            "design_confirmed": "set_design_confirmed_node",
            "code_generated": "generate_code_node",
            "training_started": "train_node",
            "training_completed": "train_node",
        }
        
        node_id = stage_map.get(current_stage)
        if node_id:
            return [node_id]
            
        return [current_stage]

    async def process_message(self, user_input: str):
        """Process user message - exactly like run_chatbot.py"""
        try:
            # Create input - same logic as run_chatbot.py
            if self.first_message:
                input_dict, config = create_user_message_input(
                    user_input,
                    thread_id=self.thread_id,
                    existing_state=self.initial_state
                )
                self.first_message = False
            else:
                input_dict, config = create_user_message_input(
                    user_input,
                    thread_id=self.thread_id
                )
            
            # Invoke graph - same as run_chatbot.py
            result = await self.graph.ainvoke(input_dict, config=config)
            
            # Format output - same as run_chatbot.py
            response = format_graph_output(result)
            
            # Construct dialog state from current_stage/phase
            dialog_state = self._get_dialog_state(result)
            
            # Send response
            self.send_message({
                "type": "response",
                "content": response,
                "state": {
                    "project_name": result.get("project_name"),
                    "project_path": result.get("project_path"),
                    "current_stage": result.get("current_stage"),
                    "current_node": result.get("current_node"),  # For graph visualization
                    "dialog_state": dialog_state,
                    "completed_stages": result.get("completed_stages", []),
                    "awaiting_test_image": result.get("awaiting_test_image", False),
                    "awaiting_new_design_choice": result.get("awaiting_new_design_choice", False),
                    "needs_user_input": result.get("needs_user_input", False),
                    # RTL synthesis status
                    "awaiting_rtl_synthesis": result.get("awaiting_rtl_synthesis", False),
                    "awaiting_hls_config": result.get("awaiting_hls_config", False),
                    "awaiting_hls_verify": result.get("awaiting_hls_verify", False),
                    "awaiting_rtl_build": result.get("awaiting_rtl_build", False),
                    "rtl_synthesis_config": result.get("rtl_synthesis_config", {}),
                    "rtl_synthesis_complete": result.get("rtl_synthesis_complete", False),
                    "rtl_output_path": result.get("rtl_output_path")
                }
            })
            
        except Exception as e:
            self.send_message({
                "type": "error",
                "message": str(e)
            })
    
    async def get_status(self):
        """Get current status - same as run_chatbot.py status command"""
        try:
            config = create_conversation_config(self.thread_id)
            state = self.graph.get_state(config)
            
            # Use helper to adapt new state structure
            arch_summary = self._get_architecture_summary(state.values)
            dialog_state = self._get_dialog_state(state.values)
            
            status_data = {
                "project_name": state.values.get("project_name", "Not created"),
                "current_stage": state.values.get("current_stage", "Initial"),
                "completed_stages": state.values.get("completed_stages", []),
                "architecture_summary": arch_summary,
                "training_runs": state.values.get("training_runs", []),
                "dialog_state": dialog_state
            }
            
            self.send_message({
                "type": "status",
                "data": status_data
            })
            
        except Exception as e:
            self.send_message({
                "type": "error",
                "message": f"Failed to get status: {str(e)}"
            })
    
    def send_message(self, message: dict):
        """Send JSON message to Electron via stdout (original stdout, not redirected)"""
        try:
            json_str = json.dumps(message)
            # Write to ORIGINAL stdout (not the redirected one)
            _original_stdout.write(json_str + '\n')
            _original_stdout.flush()
        except Exception as e:
            # Fallback error message
            error_json = json.dumps({"type": "error", "message": str(e)})
            _original_stdout.write(error_json + '\n')
            _original_stdout.flush()
    
    async def cleanup(self):
        """Cleanup - same as run_chatbot.py"""
        await cleanup_global_mcp()


async def main():
    """Main loop - reads JSON from stdin and processes messages"""
    bridge = ChatbotBridge()
    
    # Initialize
    try:
        await bridge.initialize()
    except Exception as e:
        bridge.send_message({
            "type": "error",
            "message": f"Failed to initialize bridge: {str(e)}"
        })
        import traceback
        traceback.print_exc()
        return
    
    # Read messages from stdin
    try:
        # Set stdin to unbuffered mode
        sys.stdin.reconfigure(encoding='utf-8', newline='\n')
        
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                message = json.loads(line)
                msg_type = message.get("type")
                
                if msg_type == "message":
                    user_input = message.get("content", "").strip()
                    if user_input:
                        await bridge.process_message(user_input)
                
                elif msg_type == "status":
                    await bridge.get_status()
                
                elif msg_type == "quit":
                    break
                
            except json.JSONDecodeError as e:
                bridge.send_message({
                    "type": "error",
                    "message": f"Invalid JSON: {str(e)}"
                })
            except Exception as e:
                bridge.send_message({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })
                import traceback
                traceback.print_exc()
    
    except KeyboardInterrupt:
        pass
    except Exception as e:
        bridge.send_message({
            "type": "error",
            "message": f"Fatal error: {str(e)}"
        })
        import traceback
        traceback.print_exc()
    finally:
        await bridge.cleanup()


if __name__ == "__main__":
    # Run the async main loop
    asyncio.run(main())
