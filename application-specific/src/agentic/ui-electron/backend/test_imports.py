"""
Test script to verify chatbot_bridge.py can import correctly
"""
import sys
from pathlib import Path

# Add project root to path (same as chatbot_bridge.py)
# This file is at: src/agentic/ui-electron/backend/test_imports.py
# Project root is 4 levels up
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"Test file location: {Path(__file__).resolve()}")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

try:
    print("\nTesting imports...")
    
    print("1. Importing create_graph...")
    from src.agentic.src.core.graph import create_graph
    print("   ✓ Success")
    
    print("2. Importing create_initial_state...")
    from src.agentic.src.core.state import create_initial_state
    print("   ✓ Success")
    
    print("3. Importing graph_utils...")
    from src.agentic.src.utils.graph_utils import (
        create_user_message_input,
        format_graph_output,
        create_conversation_config
    )
    print("   ✓ Success")
    
    print("4. Importing mcp...")
    from src.agentic.src.mcp import get_mcp_helper, cleanup_global_mcp
    print("   ✓ Success")
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
