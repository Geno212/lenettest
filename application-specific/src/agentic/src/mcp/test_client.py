"""Test script for the MCPHelper singleton-based client."""

import asyncio
import traceback
from .helper import get_mcp_helper, cleanup_global_mcp


async def main():
    # Obtain (and initialize) the global MCPHelper singleton
    mcp = await get_mcp_helper(url="http://127.0.0.1:8000/mcp")
    try:
        print("Using global MCP helper singleton...")
        print(f"Connected: {mcp.is_connected()}")

        # Get available tools through the helper
        tools = mcp.get_available_tools()
        print(f"\nAvailable tools: {len(tools)}")

        # Test calling a tool - get available optimizers
        print("\n" + "="*60)
        print("Testing: get_available_optimizers (via MCPHelper.call)")
        print("="*60)
        result = await mcp.call("get_available_optimizers", {})
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        if isinstance(result, dict) and 'optimizers' in result:
            print(f"Available optimizers: {result['optimizers']}")

        # Clean up global helper / connection
        print("\nCleaning up global MCP helper...")
        await cleanup_global_mcp()
        print("Cleanup complete.")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        # Ensure we attempt cleanup on error as well
        try:
            await cleanup_global_mcp()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())