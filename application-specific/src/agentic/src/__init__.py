"""
Neural Network Generator - Agentic System with Supervisor Pattern

A comprehensive system for designing, training, and optimizing neural networks
using LangGraph's supervisor pattern with specialized assistants.

Main Components:
- Core: State management, graph construction, configuration
- Assistants: Specialized AI assistants for different tasks
- Logic: Deterministic logic modules for validation and inference
- Routing: Tools for inter-assistant communication
- Nodes: Graph nodes for workflow execution
- MCP: Integration with Model Context Protocol server
- Prompts: LLM prompt templates
- Utils: Utility functions
- Schemas: Data schemas and constants

Usage:
    from src.core import NNGeneratorState, build_graph
    from src.mcp import MCPClient
    
    # Initialize
    mcp_client = MCPClient(url="http://localhost:8000/mcp")
    graph = build_graph(mcp_client, llm)
    
    # Run workflow
    result = graph.invoke(initial_state)
"""

__version__ = "0.1.0"

# Version info
VERSION = __version__