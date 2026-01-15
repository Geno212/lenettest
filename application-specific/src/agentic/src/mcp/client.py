"""MCP Client for executing tools on the MCP server."""

from typing import Dict, Any, Optional, List
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
import asyncio


class MCPClient:
    """
    Wrapper for MCP client with error handling and retry logic.
    
    Usage:
        async with MCPClient(url="http://localhost:8000/mcp") as client:
            await client.initialize()
            result = await client.call_tool("project_create", {"name": "test"})
    """
    
    def __init__(
        self, 
        url: str = "http://127.0.0.1:8000/mcp",
        timeout: int = 300,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        self.url = url
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Use a long SSE read timeout to support long-running tool calls without client timeouts
        self.transport = StreamableHttpTransport(url=url, sse_read_timeout=60 * 60 * 6)
        self.client = Client(self.transport)
        self.session_id: Optional[str] = None
        self.available_tools: List[Dict[str, Any]] = []
        self._context_manager = None
    
    async def __aenter__(self):
        """Enter the async context manager."""
        self._context_manager = await self.client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager."""
        if self._context_manager:
            return await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def connect(self):
        """
        Manually open a persistent connection.
        
        Use this when you need a long-lived client (e.g., in an assistant).
        Must call disconnect() when done.
        
        Example:
            client = MCPClient()
            await client.connect()
            await client.initialize()
            result = await client.call_tool("tool_name", {})
            await client.disconnect()
        """
        if not self._context_manager:
            self._context_manager = await self.client.__aenter__()
        return self
    
    async def disconnect(self):
        """
        Manually close the persistent connection.
        
        Should be called after connect() when you're done with the client.
        """
        if self._context_manager:
            try:
                await self.close()  # Close MCP session first
            except:
                pass
            await self.client.__aexit__(None, None, None)
            self._context_manager = None
    
    async def initialize(self) -> Optional[str]:
        """
        Initialize MCP session and list available tools.
        
        NOTE: Must be called within an active context (async with MCPClient...).
        """
        # Ping to check connection
        await self.client.ping()
        
        # List available tools
        tools = await self.client.list_tools()
        self.available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in tools
        ]
        
        # Initialize session - call directly without nested context
        result = await self.client.call_tool("initialize_session", {})
        
        # Extract session_id from structured_content
        if hasattr(result, 'structured_content') and result.structured_content:
            self.session_id = result.structured_content.get("session_id")
        elif isinstance(result, dict):
            self.session_id = result.get("session_id")
        
        return self.session_id
    
    async def call_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        retry: bool = True
    ) -> Any:
        """
        Call an MCP tool with retry logic.
        
        NOTE: Must be called within an active context (async with MCPClient...).
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            retry: Whether to retry on failure
            
        Returns:
            Tool result as dictionary
        """
        attempts = self.retry_attempts if retry else 1
        
        for attempt in range(attempts):
            try:
                # No nested context - use existing connection
                result = await self.client.call_tool(tool_name, arguments)
                
                # Extract structured content if available
                if hasattr(result, 'structured_content'):
                    return result.structured_content
                
                # Otherwise return text content
                return {"content": result.content if hasattr(result, 'content') else str(result)}
                    
            except Exception as e:
                if attempt < attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    # Last attempt failed
                    raise MCPToolError(f"Tool {tool_name} failed after {attempts} attempts: {str(e)}")
    
    async def close(self) -> None:
        """
        Close the MCP session.
        
        NOTE: Must be called within an active context (async with MCPClient...).
        """
        if self.session_id:
            try:
                await self.call_tool("close_session", {}, retry=False)
            except:
                pass  # Ignore errors on close
            finally:
                self.session_id = None
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        return self.available_tools
    
    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self.session_id is not None


class MCPToolError(Exception):
    """Exception raised when MCP tool execution fails."""
    pass