# MCP Integration Guide for Assistants

## Why It Now Works

**The Problem Before:**

- Each method created its own `async with client:` context
- Contexts were opened and closed for each call
- **Session state was lost between calls** (session_id, initialization, etc.)

**The Solution:**

- **ONE persistent connection** stays open
- All operations use the same connection
- Session state is maintained throughout

## Quick Start

### For Your Assistants (Recommended)

```python
from src.agentic.src.mcp import get_mcp_helper

class MyAssistant(BaseAssistant):
    async def __call__(self, state, config=None):
        # Get global MCP connection (auto-connects)
        mcp = await get_mcp_helper()

        # Call tools
        result = await mcp.call("tool_name", {"arg": "value"})

        # Use result...
        return {"messages": [self.format_message(str(result))]}
```

### Application Startup/Shutdown

```python
from src.agentic.src.mcp import get_mcp_helper, cleanup_global_mcp

async def startup():
    """Call when your app starts."""
    mcp = await get_mcp_helper()
    print("✓ MCP connected")

async def shutdown():
    """Call when your app shuts down."""
    await cleanup_global_mcp()
    print("✓ MCP disconnected")
```

## All Usage Patterns

### Pattern 1: Shared Global Connection (RECOMMENDED)

**Best for:** Multi-assistant applications (like yours)

```python
from src.agentic.src.mcp import get_mcp_helper

async def some_function():
    mcp = await get_mcp_helper()
    result = await mcp.call("tool_name", {})
```

**Pros:**

- One connection for everything
- Simple to use
- No connection overhead

**Cons:**

- Global state

---

### Pattern 2: Context Manager

**Best for:** Simple scripts, one-off operations

```python
from src.agentic.src.mcp import MCPClient

async with MCPClient() as client:
    await client.initialize()
    result = await client.call_tool("tool_name", {})
    await client.close()
```

**Pros:**

- Automatic cleanup
- Clear scope

**Cons:**

- New connection each time

---

### Pattern 3: Manual Connection

**Best for:** Long-lived services with explicit lifecycle

```python
from src.agentic.src.mcp import MCPClient

client = MCPClient()
await client.connect()
await client.initialize()

# Use many times
result1 = await client.call_tool("tool1", {})
result2 = await client.call_tool("tool2", {})

await client.disconnect()
```

**Pros:**

- Full control
- Persistent connection

**Cons:**

- Manual cleanup required

---

### Pattern 4: Per-Assistant Connection

**Best for:** Isolated assistants that need their own connection

```python
from src.agentic.src.mcp import MCPHelper

class MyAssistant:
    def __init__(self):
        self.mcp = MCPHelper()

    async def initialize(self):
        await self.mcp.connect()

    async def cleanup(self):
        await self.mcp.disconnect()

    async def work(self):
        result = await self.mcp.call("tool", {})
```

**Pros:**

- Isolated connections
- Easy to test

**Cons:**

- More connections (overhead)

## Available MCP Tools

Once connected, you can call these tools:

```python
mcp = await get_mcp_helper()

# Get available tools
tools = mcp.get_available_tools()

# Session management
await mcp.call("initialize_session", {})
await mcp.call("close_session", {})

# Get options
await mcp.call("get_available_optimizers", {})
await mcp.call("get_available_loss_functions", {})
await mcp.call("get_available_metrics", {})

# Project operations
await mcp.call("project_create", {"name": "my_project"})
await mcp.call("project_list", {})
await mcp.call("project_load", {"name": "my_project"})

# And many more... (34 tools total)
```

## Integration Checklist

- [ ] Import MCP helper in your assistant
- [ ] Call `get_mcp_helper()` in async methods
- [ ] Use `mcp.call(tool_name, args)` to execute tools
- [ ] Add `cleanup_global_mcp()` to application shutdown
- [ ] Test with MCP server running

## Common Mistakes to Avoid

❌ **DON'T** create new client for each call:

```python
# BAD - Creates new connection each time
async def bad():
    client = MCPClient()
    await client.connect()
    result = await client.call_tool("tool", {})
    await client.disconnect()  # Loses session!
```

✅ **DO** reuse connection:

```python
# GOOD - Uses persistent connection
async def good():
    mcp = await get_mcp_helper()
    result = await mcp.call("tool", {})
```

---

❌ **DON'T** forget to connect:

```python
# BAD - Not connected
client = MCPClient()
result = await client.call_tool("tool", {})  # ERROR!
```

✅ **DO** ensure connection:

```python
# GOOD
client = MCPClient()
await client.connect()
await client.initialize()
result = await client.call_tool("tool", {})
```

---

❌ **DON'T** nest context managers:

```python
# BAD - Already inside context
async with client:
    # This is inside a context already
    result = await client.call_tool(...)  # Don't create another context!
```

✅ **DO** use existing context:

```python
# GOOD
async with client:
    await client.initialize()
    result = await client.call_tool(...)  # Uses existing context
```

## Testing

Run the examples:

```bash
# Test basic client
python -m src.agentic.src.mcp.test_client

# Test all patterns
python -m src.agentic.src.mcp.usage_examples

# See integration examples
python -m src.agentic.src.mcp.integration_examples
```

## Troubleshooting

**Error: "Must be called within an active context"**

- You're calling methods without connecting first
- Solution: Use `await client.connect()` or context manager

**Error: "NoneType object is not subscriptable"**

- Connection was lost between calls
- Solution: Use persistent connection (Pattern 1 or 3)

**Result is None or empty**

- Session state was lost
- Solution: Keep connection open for all operations

**Connection refused**

- MCP server not running
- Solution: Start your MCP server first
