# scripts/run_chatbot.py
"""
Main entry point for running the NN Generator chatbot.

Usage:
    python scripts/run_chatbot.py
"""

import asyncio
import logging
from typing import cast
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from src.agentic.src.core.graph import create_graph
from src.agentic.src.core.state import create_initial_state, NNGeneratorState
from src.agentic.src.utils.graph_utils import (
    create_user_message_input,
    format_graph_output,
    create_conversation_config
)
from src.agentic.src.mcp import get_mcp_helper, cleanup_global_mcp
import uuid


async def run_chatbot():
    """Run interactive chatbot session."""
    
    # Ensure logging is configured (only once)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    print("=" * 60)
    print("🤖 Neural Network Generator - Interactive Chat")
    print("=" * 60)
    print("\nInitializing system...")
    
    # Create graph (this also initializes global MCP)
    try:
        graph = await create_graph()
        print("✅ System and MCP initialized successfully!\n")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Get MCP helper for session info
    mcp = await get_mcp_helper()
    session_id = mcp.client.session_id if mcp.client else "unknown"
    thread_id = f"mcp_{session_id[:8] if session_id else 'default'}"
    
    print(f"📝 Session ID: {session_id}")
    print(f"🧵 Thread ID: {thread_id}")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("Type 'status' to see current workflow status.")
    print("Type 'help' for usage examples.\n")
    print("-" * 60)
    
    initial_state = create_initial_state()
    first_message = True
    
    while True:
        # Get user input
        try:
            user_input = input("\n👤 You: ").strip()
        except KeyboardInterrupt:
            await cleanup_global_mcp()
            print("\n\nGoodbye! 👋")
            break
        
        if not user_input:
            continue
        
        # Handle special commands
        if user_input.lower() in ['quit', 'exit']:
            await cleanup_global_mcp()
            print("\nGoodbye! 👋")
            break
        
        if user_input.lower() == 'help':
            print_help()
            continue
        
        if user_input.lower() == 'status':
            # Get current state and show status
            config = create_conversation_config(thread_id)
            try:
                state = graph.get_state(cast(RunnableConfig, config))
                print_status(state.values)
            except:
                print("No status available yet. Start a conversation first!")
            continue
        
        # Process message through graph
        try:
            # Create input
            if first_message:
                input_dict, config = create_user_message_input(
                    user_input,
                    thread_id=thread_id,
                    existing_state=initial_state
                )
                first_message = False
            else:
                input_dict, config = create_user_message_input(
                    user_input,
                    thread_id=thread_id
                )
            
            # Invoke graph
            print("\n🤖 Assistant: ", end="", flush=True)
            
            result = await graph.ainvoke(
                cast(NNGeneratorState, input_dict), 
                config=cast(RunnableConfig, config)
            )
            
            # Format and display output
            response = format_graph_output(result)
            print(response)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")


def print_help():
    """Print help message with examples."""
    print("\n" + "=" * 60)
    print("📚 HELP - Neural Network Generator")
    print("=" * 60)
    print("\n💡 EXAMPLE PROMPTS:\n")
    print("Getting Started:")
    print("  • 'Build a classifier for MNIST dataset with 95% accuracy'")
    print("  • 'Create an image classifier using ResNet18'")
    print("  • 'I want to train a CNN on CIFAR10'")
    print("\nArchitecture:")
    print("  • 'Use ResNet50 pretrained model'")
    print("  • 'Design a custom CNN with 3 conv layers'")
    print("  • 'Add dropout for regularization'")
    print("\nConfiguration:")
    print("  • 'Use Adam optimizer with learning rate 0.001'")
    print("  • 'Train for 50 epochs with batch size 32'")
    print("  • 'Set device to cuda'")
    print("\nTraining:")
    print("  • 'Start training'")
    print("  • 'Show training status'")
    print("  • 'Open TensorBoard'")
    print("\nOptimization:")
    print("  • 'My model isn't reaching target, help improve it'")
    print("  • 'Optimize the architecture'")
    print("\n🔧 COMMANDS:\n")
    print("  • status  - Show current workflow status")
    print("  • help    - Show this help message")
    print("  • quit    - Exit the chatbot")
    print("\n" + "=" * 60 + "\n")


def print_status(state: dict):
    """Print formatted status."""
    print("\n" + "=" * 60)
    print("📊 CURRENT STATUS")
    print("=" * 60)
    
    # Project
    project = state.get("project_name", "Not created")
    print(f"\n📁 Project: {project}")
    
    # Stage
    stage = state.get("current_stage", "Initial")
    print(f"📍 Current Stage: {stage}")
    
    # Completed
    completed = state.get("completed_stages", [])
    if completed:
        print(f"✅ Completed: {', '.join(completed)}")
    
    # Architecture
    arch = state.get("architecture_summary")
    if arch:
        print(f"\n🏗️  Architecture: {arch.get('type', 'unknown')}")
        if arch.get('base_model'):
            print(f"   Model: {arch['base_model']}")
        print(f"   Parameters: {arch.get('parameters', 0):,}")
    
    # Training
    runs = state.get("training_runs", [])
    if runs:
        latest = runs[-1]
        acc = latest.get("final_accuracy", 0) * 100
        print(f"\n📈 Latest Training: {acc:.1f}% accuracy")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(run_chatbot())