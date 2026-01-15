# src/assistants/base_assistant.py
"""Base class for all assistants with common functionality."""

from typing import Dict, Any, Optional, List, cast
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from src.agentic.src.core.state import NNGeneratorState


class BaseAssistant:
    """
    Base class for all specialized assistants.
    
    Provides common functionality:
    - Message formatting
    - Tool message creation  
    - State updates
    - Error handling
    - Parameter extraction
    
    All specialized assistants inherit from this class.
    """
    
    def __init__(self, name: str, llm: Optional[Any] = None):
        """
        Initialize base assistant.
        
        Args:
            name: Name of the assistant (e.g., "project_manager")
            llm: Language model instance (optional, some assistants don't need LLM)
        """
        self.name = name
        self.llm = llm
    
    def format_message(self, content: str) -> AIMessage:
        """
        Create formatted AI message.
        
        Args:
            content: Message content
            
        Returns:
            AIMessage with assistant name
        """
        return AIMessage(content=content, name=self.name)
    
    def create_tool_message(self, content: str, tool_call_id: str) -> ToolMessage:
        """
        Create tool message for tool call results.
        
        Used when entering/exiting assistants or executing tools.
        
        Args:
            content: Tool result content
            tool_call_id: ID of the tool call being responded to
            
        Returns:
            ToolMessage
        """
        return ToolMessage(content=content, tool_call_id=tool_call_id, name=self.name)
    
    def extract_tool_params(self, state: NNGeneratorState) -> Dict[str, Any]:
        """
        Extract parameters from last tool call in state.
        
        When Primary Assistant calls ToRequirementsAnalyst(...), this extracts
        those parameters for the Requirements Analyst to use.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary of tool call arguments
        """
        if not state.get("messages"):
            return {}

        # Search backwards for the most recent message that contains a tool call
        # (previous implementation only checked the last message which may be
        # an assistant/AI message that came after the tool call).
        for message in reversed(state["messages"]):
            if hasattr(message, 'tool_calls') and message.tool_calls:
                # Return the args of the most recent tool call
                return message.tool_calls[0].get("args", {})

        return {}
    
    def get_tool_call_id(self, state: NNGeneratorState) -> Optional[str]:
        """
        Get tool call ID from last message.
        
        Used for creating ToolMessages that reference the original tool call.
        
        Args:
            state: Current workflow state
            
        Returns:
            Tool call ID or None
        """
        if not state.get("messages"):
            return None
        
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return last_message.tool_calls[0].get("id")
        
        return None
    
    def get_last_user_message(self, state: NNGeneratorState) -> Optional[str]:
        """
        Get the last user (human) message content.
        
        Useful for understanding what the user most recently said.
        
        Args:
            state: Current workflow state
            
        Returns:
            User message content or None
        """
        # Use the unified helper that mirrors core.graph._get_last_user_message
        return self._get_last_user_message(state)

    def _get_last_user_message(self, state: NNGeneratorState) -> str:
        """Get last human message and normalize to a string.

        This mirrors `_get_last_user_message` in `core.graph` so assistants
        and the graph itself treat human content the same way (stringified
        and joined when the content is a list/dict).
        """
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
    
    def create_error_response(
        self, 
        error: Exception, 
        stage: str,
        severity: str = "high"
    ) -> Dict[str, Any]:
        """
        Create standardized error response.
        
        Formats errors consistently and stores them in state.
        
        Args:
            error: The exception that occurred
            stage: Which stage the error occurred in
            severity: "low", "medium", "high", or "critical"
            
        Returns:
            State update dictionary with error and message
        """
        error_dict = {
            "timestamp": self._get_timestamp(),
            "stage": stage,
            "assistant": self.name,
            "error": str(error),
            "error_type": type(error).__name__,
            "severity": severity,
            "recoverable": severity in ["low", "medium"],
            "suggested_action": self._get_error_suggestion(error, stage)
        }
        
        return {
            "errors": [error_dict],
            "messages": [
                self.format_message(
                    f"❌ **Error in {stage}**\n\n"
                    f"**Error:** {str(error)}\n"
                    f"**Type:** {type(error).__name__}\n\n"
                    f"**Suggested Action:** {error_dict['suggested_action']}\n\n"
                    f"Please address this issue and try again."
                )
            ]
        }
    
    def update_stage(self, state: NNGeneratorState, stage: str) -> Dict[str, Any]:
        """
        Update current stage and add to completed stages.
        
        Tracks workflow progress.
        
        Args:
            state: Current workflow state
            stage: Stage that was just completed
            
        Returns:
            State update dictionary
        """
        completed = state.get("completed_stages", [])
        
        # Add to completed if not already there
        if stage not in completed:
            completed = completed + [stage]
        
        return {
            "current_stage": stage,
            "completed_stages": completed
        }

    def _append_stage(self, state: NNGeneratorState, stage: str) -> List[str]:
        """Return a copy of completed_stages with `stage` appended if missing.

        This utility matches the small helper in `core.graph` so wiring
        between graph and assistants keeps the same behavior.
        """
        completed = list(state.get("completed_stages", []))
        if stage not in completed:
            completed.append(stage)
        return completed

    def _user_confirmed_design(self, state: NNGeneratorState) -> bool:
        """Check if the latest human message confirms the design.

        Returns True when `awaiting_design_confirmation` is set and the last
        human message contains a confirmation word.
        """
        if not state.get("awaiting_design_confirmation"):
            return False
        text = self._get_last_user_message(state).strip().lower()
        if not text:
            return False
        confirmations = ["yes", "confirm", "ready", "proceed", "go ahead"]
        return any(word in text for word in confirmations)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _get_error_suggestion(self, error: Exception, stage: str) -> str:
        """
        Get suggested action for an error.
        
        Args:
            error: The exception
            stage: Stage where error occurred
            
        Returns:
            Suggestion string
        """
        error_type = type(error).__name__
        
        # Common error suggestions
        suggestions = {
            "FileNotFoundError": "Check that the file path exists and is accessible",
            "PermissionError": "Check file/directory permissions",
            "ConnectionError": "Check MCP server connection",
            "TimeoutError": "Operation timed out - try again or check system resources",
            "ValueError": "Check that input values are valid and in correct format",
            "KeyError": "Required field missing - check configuration",
            "TypeError": "Type mismatch - check parameter types"
        }
        
        return suggestions.get(error_type, "Review the error and try again")
    
    def __call__(self, state: NNGeneratorState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Main execution method - must be implemented by subclasses.
        
        This is called when the assistant is invoked in the graph.
        
        Args:
            state: Current workflow state
            config: Optional runnable configuration
            
        Returns:
            State update dictionary
            
        Raises:
            NotImplementedError: If subclass doesn't implement this
        """
        raise NotImplementedError(f"{self.name} must implement __call__ method")


class Assistant:
    """
    Wrapper class for LangChain runnable assistants.
    
    This implements the supervisor pattern's assistant loop:
    1. Invoke LLM with current state
    2. If response is empty, prompt for real output
    3. Repeat until valid response
    4. Return result
    
    Used for assistants that use LLM for decision-making.
    """
    
    def __init__(self, runnable):
        """
        Initialize assistant wrapper.
        
        Args:
            runnable: LangChain runnable (typically prompt | llm.bind_tools(...))
        """
        self.runnable = runnable
    
    def __call__(self, state: NNGeneratorState, config: Optional[RunnableConfig] = None):
        """
        Execute assistant with retry loop for valid responses.
        
        Args:
            state: Current workflow state
            config: Optional runnable configuration
            
        Returns:
            State update with messages
        """
        while True:
            # Invoke the runnable
            result = self.runnable.invoke(state, config=config)
            
            # Check if response is valid
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                # Empty response - add message prompting for real output
                messages = state["messages"] + [
                    result,
                    HumanMessage(content="Please respond with actual content or a tool call.")
                ]
                state = cast(NNGeneratorState, {**state, "messages": messages})
            else:
                # Valid response received
                break
        
        return {"messages": [result]}