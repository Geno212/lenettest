# src/utils/graph_utils.py
"""Utility functions for graph operations."""

from typing import Dict, Any, TypeVar, Optional
from langchain_core.messages import HumanMessage
from src.agentic.src.core.state import NNGeneratorState, create_initial_state
import uuid

T = TypeVar('T')


def safe_get(state: Dict[str, Any], key: str, default: T = None) -> T:
    """
    Safely get value from state, handling None values.
    
    Unlike dict.get(), this returns the default even when the key exists with None value.
    This is needed because our initial state explicitly sets many fields to None.
    
    Args:
        state: State dictionary
        key: Key to retrieve
        default: Default value if key is missing or value is None
        
    Returns:
        Value from state, or default if None or missing
        
    Example:
        state = {"task_specification": None}
        # Using dict.get() returns None (not the default!)
        spec = state.get("task_specification", {})  # Returns None
        # Using safe_get() returns the default
        spec = safe_get(state, "task_specification", {})  # Returns {}
    """
    value = state.get(key, default)
    return default if value is None else value


def create_conversation_config(thread_id: str = None) -> Dict[str, Any]:
    """
    Create configuration for graph invocation.
    
    The thread_id is used by the checkpointer to maintain conversation state.
    
    Args:
        thread_id: Optional thread ID. If None, generates new UUID.
        
    Returns:
        Configuration dictionary
        
    Example:
        config = create_conversation_config("user_123_session_1")
        graph.invoke(input_state, config=config)
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


def create_user_message_input(
    message: str,
    thread_id: str = None,
    existing_state: NNGeneratorState = None
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create input for graph invocation from user message.
    
    Args:
        message: User's message
        thread_id: Conversation thread ID
        existing_state: Optional existing state (for first message, pass initial state)
        
    Returns:
        Tuple of (input_dict, config_dict)
        
    Example:
        # First message
        initial_state = create_initial_state(session_id="sess_123")
        input_data, config = create_user_message_input(
            "Build MNIST classifier",
            thread_id="thread_1",
            existing_state=initial_state
        )
        result = graph.invoke(input_data, config=config)
        
        # Follow-up message (graph maintains state via checkpointer)
        input_data, config = create_user_message_input(
            "Use ResNet18",
            thread_id="thread_1"
        )
        result = graph.invoke(input_data, config=config)
    """
    # Create config
    config = create_conversation_config(thread_id)
    
    # Create input
    user_message = HumanMessage(content=message)
    
    if existing_state:
        # First message - include full state
        # Convert to regular dict to ensure proper handling
        input_dict = dict(existing_state) if not isinstance(existing_state, dict) else existing_state.copy()
        # Append user message to existing messages
        existing_messages = input_dict.get("messages", [])
        input_dict["messages"] = existing_messages + [user_message]
    else:
        # Follow-up message - graph will load state from checkpointer
        input_dict = {
            "messages": [user_message]
        }
    
    return input_dict, config


def format_graph_output(result: Dict[str, Any]) -> str:
    """
    Format graph output for display to user.
    
    Extracts the last AI message from the result.
    
    Args:
        result: Graph invocation result
        
    Returns:
        Formatted message string
    """
    if result is None:
        return "⚠️ No result returned from graph (execution may have failed)"
    
    messages = result.get("messages", [])
    
    if not messages:
        return "No response"
    
    # Get last AI message
    for message in reversed(messages):
        if hasattr(message, "content") and message.content:
            if hasattr(message, "type") and message.type == "ai":
                return message.content
    
    return "No response"


def visualize_graph(graph):
    """
    Generate visualization of the graph structure.
    
    Requires graphviz to be installed.
    
    Args:
        graph: Compiled LangGraph
        
    Returns:
        Graph visualization (can be displayed in Jupyter or saved)
        
    Example:
        graph = create_graph()
        viz = visualize_graph(graph)
        viz.render("graph_structure", format="png")
    """
    try:
        from IPython.display import Image, display
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception as e:
        print(f"Could not visualize graph: {e}")
        print("Make sure graphviz is installed: pip install graphviz")


def get_conversation_history(
    graph,
    thread_id: str,
    limit: int = 10
) -> list:
    """
    Retrieve conversation history from checkpointer.
    
    Args:
        graph: Compiled graph with checkpointer
        thread_id: Thread ID to retrieve
        limit: Maximum number of messages
        
    Returns:
        List of messages
    """
    config = create_conversation_config(thread_id)
    
    try:
        # Get state from checkpointer
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        
        # Return last N messages
        return messages[-limit:] if len(messages) > limit else messages
    except Exception as e:
        print(f"Could not retrieve history: {e}")
        return []