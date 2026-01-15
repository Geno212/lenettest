# scripts/visualize_graph.py
"""
Visualize the graph structure.

Generates a visual representation of the workflow graph.

Usage:
    python scripts/visualize_graph.py
"""

import asyncio
from src.agentic.src.core.graph import create_graph


async def visualize():
    """Generate graph visualization."""
    print("🎨 Generating graph visualization...\n")
    
    try:
        # Create graph (create_graph returns a tuple: (compiled_graph, mcp_client))
        compiled_graph, mcp_client = await create_graph()

        # Get the underlying graph object once and reuse it
        inner_graph = compiled_graph.get_graph()

        # Generate mermaid diagram
        print("Graph structure (Mermaid syntax):\n")
        print(inner_graph.draw_mermaid())

        # Try to generate PNG if graphviz available
        try:
            png_data = inner_graph.draw_mermaid_png()

            # Save to file
            output_file = "graph_structure.png"
            with open(output_file, "wb") as f:
                f.write(png_data)

            print(f"\n✅ Graph visualization saved to: {output_file}")

        except Exception as e:
            print(f"\n⚠️  Could not generate PNG: {e}")
            print("Install graphviz to enable PNG generation:")
            print("  pip install graphviz")
            print("  # Also install system graphviz:")
            print("  # Ubuntu: sudo apt-get install graphviz")
            print("  # macOS: brew install graphviz")

        # Print node summary
        print("\n📊 Graph Summary:")
        try:
            total_nodes = len(inner_graph.nodes)
        except Exception:
            # Fallback if nodes attribute is missing or unusual
            total_nodes = "unknown"

        print(f"  Total nodes: {total_nodes}")
        print(f"  Entry point: START → master_triage_router")
        print(f"  Specialized assistants: 5 (project, architecture, config, code, training)")
        print(f"  Exit points: END")
        print(f"  Note: Each specialized assistant performs its own extraction with smart merging")
        
    except Exception as e:
        print(f"❌ Failed to visualize graph: {e}")


if __name__ == "__main__":
    asyncio.run(visualize())