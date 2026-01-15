"""
LangSmith Tracing Utilities

This module provides helper functions for integrating LangSmith tracing
into the LangGraph-based Neural Network Generator workflow.

LangSmith provides:
- Detailed execution traces of agent workflows
- Performance monitoring and debugging
- Input/output logging for each step
- Error tracking and diagnostics
"""

import os
from typing import Optional, Dict, Any
from contextlib import contextmanager


def configure_langsmith_tracing(
    api_key: Optional[str] = None,
    project_name: str = "nn-generator-agentic",
    endpoint: str = "https://api.smith.langchain.com",
    enabled: bool = True
) -> bool:
    """
    Configure LangSmith tracing for LangGraph execution.
    
    This function sets the necessary environment variables that LangChain
    and LangGraph use to enable tracing.
    
    Args:
        api_key: LangSmith API key. If None, reads from LANGSMITH_API_KEY env var.
        project_name: Project name for organizing traces (default: "nn-generator-agentic")
        endpoint: LangSmith API endpoint (default: "https://api.smith.langchain.com")
        enabled: Whether to enable tracing (default: True)
    
    Returns:
        bool: True if tracing was successfully configured, False otherwise
    
    Example:
        from src.agentic.src.utils.langsmith_utils import configure_langsmith_tracing
        
        # Configure with API key
        configure_langsmith_tracing(
            api_key="your-api-key",
            project_name="my-nn-project"
        )
        
        # Or let it read from environment
        configure_langsmith_tracing()
    """
    if not enabled:
        # Disable tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False
    
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.environ.get("LANGSMITH_API_KEY")
    
    if api_key is None:
        print("⚠️  LangSmith tracing disabled: No API key provided")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False
    
    # Set environment variables for LangChain tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project_name
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    
    print(f"✅ LangSmith tracing enabled")
    print(f"   📊 Project: {project_name}")
    print(f"   🔗 View traces at: https://smith.langchain.com/")
    
    return True


def disable_langsmith_tracing():
    """
    Disable LangSmith tracing.
    
    This is useful for testing or when you want to temporarily disable
    tracing without changing configuration.
    
    Example:
        from src.agentic.src.utils.langsmith_utils import disable_langsmith_tracing
        disable_langsmith_tracing()
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("🚫 LangSmith tracing disabled")


@contextmanager
def trace_context(
    run_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None
):
    """
    Context manager for tracing a specific section of code.
    
    This provides additional context for LangSmith traces by setting
    run-specific metadata and tags.
    
    Args:
        run_name: Name for this trace run
        metadata: Additional metadata to attach to the trace
        tags: List of tags for categorizing the trace
    
    Example:
        from src.agentic.src.utils.langsmith_utils import trace_context
        
        with trace_context(
            run_name="test_mnist_classifier",
            metadata={"dataset": "MNIST", "architecture": "ResNet18"},
            tags=["test", "computer-vision"]
        ):
            # Your code here
            result = await graph.ainvoke(input_dict, config=config)
    """
    # Store original environment
    original_metadata = os.environ.get("LANGCHAIN_METADATA", "")
    original_tags = os.environ.get("LANGCHAIN_TAGS", "")
    
    try:
        # Set run-specific context
        if metadata:
            import json
            os.environ["LANGCHAIN_METADATA"] = json.dumps(metadata)
        
        if tags:
            os.environ["LANGCHAIN_TAGS"] = ",".join(tags)
        
        yield
    
    finally:
        # Restore original environment
        if original_metadata:
            os.environ["LANGCHAIN_METADATA"] = original_metadata
        else:
            os.environ.pop("LANGCHAIN_METADATA", None)
        
        if original_tags:
            os.environ["LANGCHAIN_TAGS"] = original_tags
        else:
            os.environ.pop("LANGCHAIN_TAGS", None)


def add_trace_metadata(metadata: Dict[str, Any]) -> None:
    """
    Add metadata to current trace.
    
    This allows you to attach additional information to traces dynamically
    during execution.
    
    Args:
        metadata: Dictionary of metadata key-value pairs
    
    Example:
        from src.agentic.src.utils.langsmith_utils import add_trace_metadata
        
        add_trace_metadata({
            "user_id": "user123",
            "session_type": "interactive",
            "model_complexity": "high"
        })
    """
    import json
    
    existing_metadata = os.environ.get("LANGCHAIN_METADATA", "{}")
    
    try:
        existing = json.loads(existing_metadata)
    except json.JSONDecodeError:
        existing = {}
    
    # Merge metadata
    existing.update(metadata)
    
    os.environ["LANGCHAIN_METADATA"] = json.dumps(existing)


def add_trace_tags(tags: list) -> None:
    """
    Add tags to current trace.
    
    Tags help categorize and filter traces in LangSmith.
    
    Args:
        tags: List of tag strings
    
    Example:
        from src.agentic.src.utils.langsmith_utils import add_trace_tags
        
        add_trace_tags(["production", "high-priority", "customer-facing"])
    """
    existing_tags = os.environ.get("LANGCHAIN_TAGS", "")
    
    if existing_tags:
        existing_list = [t.strip() for t in existing_tags.split(",")]
    else:
        existing_list = []
    
    # Add new tags (avoid duplicates)
    for tag in tags:
        if tag not in existing_list:
            existing_list.append(tag)
    
    os.environ["LANGCHAIN_TAGS"] = ",".join(existing_list)


def get_trace_url(run_id: Optional[str] = None) -> Optional[str]:
    """
    Get the LangSmith trace URL for a specific run.
    
    Args:
        run_id: The run ID from LangGraph execution. If None, returns project URL.
    
    Returns:
        URL string for viewing the trace, or None if tracing is disabled
    
    Example:
        url = get_trace_url(run_id="abc-123-def")
        print(f"View trace at: {url}")
    """
    if os.environ.get("LANGCHAIN_TRACING_V2") != "true":
        return None
    
    project = os.environ.get("LANGCHAIN_PROJECT", "default")
    
    if run_id:
        return f"https://smith.langchain.com/o/default/projects/p/{project}/r/{run_id}"
    else:
        return f"https://smith.langchain.com/o/default/projects/p/{project}"


def is_tracing_enabled() -> bool:
    """
    Check if LangSmith tracing is currently enabled.
    
    Returns:
        bool: True if tracing is enabled, False otherwise
    
    Example:
        if is_tracing_enabled():
            print("Traces will be sent to LangSmith")
    """
    return os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"


def print_tracing_status() -> None:
    """
    Print current LangSmith tracing configuration status.
    
    Useful for debugging and verifying configuration.
    
    Example:
        from src.agentic.src.utils.langsmith_utils import print_tracing_status
        print_tracing_status()
    """
    enabled = is_tracing_enabled()
    
    print("\n" + "="*60)
    print("📊 LangSmith Tracing Status")
    print("="*60)
    print(f"Enabled: {'✅ Yes' if enabled else '❌ No'}")
    
    if enabled:
        print(f"Project: {os.environ.get('LANGCHAIN_PROJECT', 'Not set')}")
        print(f"Endpoint: {os.environ.get('LANGCHAIN_ENDPOINT', 'Not set')}")
        print(f"API Key: {'🔑 Set' if os.environ.get('LANGCHAIN_API_KEY') else '❌ Not set'}")
        
        metadata = os.environ.get("LANGCHAIN_METADATA")
        if metadata:
            print(f"Metadata: {metadata}")
        
        tags = os.environ.get("LANGCHAIN_TAGS")
        if tags:
            print(f"Tags: {tags}")
        
        print(f"\n🔗 View traces at: {get_trace_url()}")
    else:
        print("Set LANGSMITH_API_KEY environment variable to enable tracing")
    
    print("="*60 + "\n")
