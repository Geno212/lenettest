"""
MCP Helper for Assistants - Simplifies MCP usage in assistant classes.
"""

from typing import Optional
from .client import MCPClient


class MCPHelper:
    """
    Helper class to manage MCP client lifecycle for assistants.
    
    This provides a simple interface for assistants to use MCP tools
    without worrying about connection management.
    
    Usage in your assistant:
        class MyAssistant(BaseAssistant):
            def __init__(self, name: str, llm, mcp_url: str = "http://127.0.0.1:8000/mcp"):
                super().__init__(name, llm)
                self.mcp = MCPHelper(mcp_url)
            
            async def initialize(self):
                await self.mcp.connect()
            
            async def cleanup(self):
                await self.mcp.disconnect()
            
            async def some_method(self):
                result = await self.mcp.call("tool_name", {"arg": "value"})
                return result
    """
    
    def __init__(self, url: str = "http://127.0.0.1:8000/mcp"):
        """
        Initialize MCP helper.
        
        Args:
            url: MCP server URL
        """
        self.client: Optional[MCPClient] = None
        self.url = url
        self._is_connected = False
    
    async def connect(self):
        """
        Connect to MCP server and initialize session.
        
        Call this once when your assistant starts.
        """
        if self._is_connected:
            return
        
        self.client = MCPClient(url=self.url)
        await self.client.connect()
        await self.client.initialize()
        self._is_connected = True
    
    async def disconnect(self):
        """
        Disconnect from MCP server.
        
        Call this when your assistant is done or shutting down.
        """
        if not self._is_connected or not self.client:
            return
        
        await self.client.disconnect()
        self._is_connected = False
        self.client = None
    
    async def call(self, tool_name: str, arguments: dict):
        """
        Call an MCP tool.
        
        Args:
            tool_name: Name of the MCP tool to call
            arguments: Tool arguments as dictionary
            
        Returns:
            Tool result
            
        Raises:
            RuntimeError: If not connected
        """
        if not self._is_connected or not self.client:
            raise RuntimeError(
                f"MCP Helper not connected. Call await mcp.connect() first."
            )
        
        return await self.client.call_tool(tool_name, arguments)
    
    def is_connected(self) -> bool:
        """Check if MCP is connected."""
        return self._is_connected
    
    def get_available_tools(self):
        """Get list of available MCP tools."""
        if not self.client:
            return []
        return self.client.get_available_tools()


# Global singleton instance (optional, for simple use cases)
_global_mcp_helper: Optional[MCPHelper] = None


async def get_mcp_helper(url: str = "http://127.0.0.1:8000/mcp") -> MCPHelper:
    """
    Get global MCP helper instance (singleton pattern).
    
    This creates one shared MCP connection for all assistants.
    Useful if you want to share a single connection across multiple assistants.
    
    Usage:
        mcp = await get_mcp_helper()
        result = await mcp.call("tool_name", {})
    
    Args:
        url: MCP server URL
        
    Returns:
        Global MCPHelper instance
    """
    global _global_mcp_helper
    
    if _global_mcp_helper is None:
        _global_mcp_helper = MCPHelper(url)
        await _global_mcp_helper.connect()
    
    return _global_mcp_helper


async def cleanup_global_mcp():
    """
    Cleanup global MCP helper.
    
    Call this when your application is shutting down.
    """
    global _global_mcp_helper
    
    if _global_mcp_helper:
        await _global_mcp_helper.disconnect()
        _global_mcp_helper = None
