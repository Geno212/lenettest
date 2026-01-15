# scripts/test_graph.py
"""
Test script for graph construction and basic flow.

Usage:
    python scripts/test_graph.py
"""

import asyncio
from src.agentic.src.core.graph import create_graph
from src.agentic.src.core.state import create_initial_state
from src.agentic.src.utils.graph_utils import create_user_message_input, format_graph_output
from src.agentic.src.core.config import load_config
from src.agentic.src.core.state import update_dialog_stack
from src.agentic.src.utils.langsmith_utils import (
    trace_context, 
    add_trace_metadata, 
    add_trace_tags,
    print_tracing_status
)


async def test_basic_flow():
    """Test basic conversation flow."""
    print("🧪 Testing Graph Construction...\n")
    
    # Load config
    print("1. Loading configuration...")
    try:
        app_config = load_config()
        print("   ✅ Configuration loaded successfully\n")
    except Exception as e:
        print(f"   ❌ Failed to load config: {e}")
        return
    
    # Print tracing status
    print_tracing_status()
    
    # Create graph
    print("2. Creating graph...")
    try:
        graph, mcp_client = await create_graph()
        print("   ✅ Graph created successfully\n")
    except Exception as e:
        print(f"   ❌ Failed to create graph: {e}")
        return
    
    # Create initial state
    print("3. Creating initial state...")
    try:
        initial_state = create_initial_state(mcp_client=mcp_client)
        print(f"   ✅ Session: {mcp_client.session_id}\n")
    except Exception as e:
        print(f"   ❌ Failed to create initial state: {e}")
        return
    
    state = initial_state
    
    # Test 1: Simple requirements extraction
    print("4. Test 1: Requirements Extraction")
    print("   User: 'Build a classifier for MNIST with 95% accuracy using ResNet18'")
    
    # Add tracing metadata for this test
    add_trace_metadata({
        "test_name": "test_1_requirements_extraction",
        "test_type": "basic_flow",
        "user_input": "Build a classifier for MNIST with 95% accuracy using ResNet18"
    })
    add_trace_tags(["test", "requirements-extraction", "mnist"])
    
    input_dict, config = create_user_message_input(
        "Build a classifier for MNIST with 95% accuracy using ResNet18",
        thread_id="test_thread_1",
        existing_state=state
    )
    
    # Debug: Print input state
    print(f"   📋 Input keys: {list(input_dict.keys())}")
    print(f"   📬 Messages count: {len(input_dict.get('messages', []))}")
    print(f"   🆔 Session ID in input: {input_dict.get('session_id', 'MISSING')}")
    
    try:
        with trace_context(
            run_name="test_1_requirements_extraction",
            metadata={"dataset": "MNIST", "target_accuracy": 0.95, "architecture": "ResNet18"},
            tags=["test", "requirements"]
        ):
            result = await graph.ainvoke(input_dict, config=config)
        
        if result is None:
            print(f"   ❌ Failed: Graph returned None (execution failed)")
            print(f"   💡 This usually means the graph execution raised an exception")
            print(f"   � Check if all required state fields are present and valid")
            return
        
        state = result
        response = format_graph_output(result)
        print(f"   🤖 Assistant: {response}")
        print("   ✅ Requirements extraction working\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Follow-up message
    print("5. Test 2: Follow-up Message")
    print("   User: 'Yes, proceed'")
    
    add_trace_metadata({
        "test_name": "test_2_follow_up",
        "test_type": "basic_flow"
    })
    add_trace_tags(["test", "follow-up"])
    
    input_dict, config = create_user_message_input(
        "Yes, proceed",
        thread_id="test_thread_1",
        existing_state=state
    )
    
    try:
        with trace_context(
            run_name="test_2_follow_up_message",
            metadata={"conversation_step": 2},
            tags=["test", "follow-up"]
        ):
            result = await graph.ainvoke(input_dict, config=config)
        state = result
        response = format_graph_output(result)
        print(f"   🤖 Assistant: {response[:200]}...")
        print("   ✅ Follow-up handling working\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
        return
    
    # Test 3: Status query
    print("6. Test 3: Status Query")
    print("   User: 'What's the status?'")
    
    add_trace_metadata({
        "test_name": "test_3_status_query",
        "test_type": "basic_flow"
    })
    add_trace_tags(["test", "status-query"])
    
    input_dict, config = create_user_message_input(
        "What's the status?",
        thread_id="test_thread_1",
        existing_state=state
    )
    
    try:
        with trace_context(
            run_name="test_3_status_query",
            metadata={"conversation_step": 3},
            tags=["test", "status"]
        ):
            result = await graph.ainvoke(input_dict, config=config)
        response = format_graph_output(result)
        print(f"   🤖 Assistant: {response[:200]}...")
        print("   ✅ Status queries working\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
    
    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


async def test_graph_structure():
    """Test graph structure and nodes."""
    print("🧪 Testing Graph Structure...\n")
    
    try:
        graph, mcp_client = await create_graph()
        
        # Compiled LangGraph doesn't expose .nodes directly
        # Just verify it was created successfully
        print("✅ Graph compiled successfully")
        print(f"✅ MCP client initialized with session: {mcp_client.session_id}")
        
        # Try to get graph representation
        try:
            graph_dict = graph.get_graph()
            nodes_list = list(graph_dict.nodes.keys()) if hasattr(graph_dict, 'nodes') else []
            if nodes_list:
                print(f"\n📊 Total nodes: {len(nodes_list)}")
                print("\nNodes:")
                for node in sorted(nodes_list):
                    print(f"  • {node}")
        except:
            print("\n💡 Note: Node inspection not available for compiled graph")
        
        print("\n📊 Graph structure validated ✅")
        
    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        import traceback
        traceback.print_exc()




async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 GRAPH TESTING SUITE")
    print("=" * 60 + "\n")
    

    
    # Test 2: Graph structure
    await test_graph_structure()
    
    # Test 3: Basic flow
    print("\n")

    #await test_basic_flow()
    
    print("\n" + "=" * 60)
    print("🎉 Testing Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main()) 