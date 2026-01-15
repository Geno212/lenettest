# 🔌 Agentic Graph Integration

## Overview

The Electron UI is **fully integrated** with the agentic graph system in `src/agentic/`. It uses the exact same graph, assistants, and tools as the CLI version.

## Architecture

```
┌─────────────────────────────────────┐
│     Electron UI (Desktop App)       │
│     frontend/index.html             │
└──────────────┬──────────────────────┘
               │ HTTP + WebSocket
┌──────────────▼──────────────────────┐
│     Flask Backend                    │
│     backend/server.py               │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Agentic Graph (Shared Code)       │
│  src/agentic/src/core/graph.py     │
│  - Primary Assistant                │
│  - Requirements Analyst             │
│  - Project Manager                  │
│  - Architecture Designer            │
│  - Configuration Specialist         │
│  - Code Generator                   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         MCP Server                  │
│  src/cli/mcp_server/main.py        │
│  - Project Tools                    │
│  - Architecture Tools               │
│  - Config Tools                     │
│  - Code Generation Tools            │
└─────────────────────────────────────┘
```

## Integration Points

### 1. Graph Creation
**Location:** `backend/server.py` line 57

```python
from src.agentic.src.core.graph import create_graph
graph = await create_graph()
```

This uses the **exact same** `create_graph()` function as:
- CLI chatbot (`src/agentic/src/scripts/run_chatbot.py`)
- Test scripts (`src/agentic/src/scripts/test_graph.py`)

### 2. Message Processing
**Location:** `backend/server.py` line 134

```python
from src.agentic.src.utils.graph_utils import create_user_message_input

input_dict, config = create_user_message_input(
    message,
    thread_id=thread_id
)

async for event in graph.astream(input_dict, config):
    # Process events...
```

Uses the same helper functions as CLI.

### 3. State Management
**Location:** `backend/server.py` line 165

```python
final_state = graph.get_state(config)
current_state = final_state.values
```

State includes all fields from `src/agentic/src/core/state.py`:
- `dialog_state` - Current assistant stack
- `project_name`, `project_path` - Project info
- `architecture_path` - Architecture file
- `config_path` - Configuration file
- `training_status` - Training progress
- `generated_code_paths` - Generated files
- And all other state fields

### 4. Tool Execution
**Location:** `backend/server.py` line 146

The graph automatically executes MCP tools through:
```python
from src.agentic.src.mcp import get_mcp_helper

mcp = await get_mcp_helper()
```

All tools from the MCP server are available:
- `create_project`, `list_projects`, `load_project`
- `create_architecture`, `add_layer`, `list_layers`
- `set_model_params`, `set_optimizer`, `set_loss_function`
- `generate_pytorch`, `list_templates`

## Configuration

### API Keys
**Location:** `backend/.env`

The backend loads environment variables using `python-dotenv`:

```python
# backend/server.py line 15
from dotenv import load_dotenv
load_dotenv(backend_dir / ".env")
```

This feeds into the config system:
```python
# src/agentic/src/core/config.py line 276
def _load_from_env(self):
    if "OPENAI_API_KEY" in os.environ:
        self.llm.api_key = os.environ["OPENAI_API_KEY"]
```

### LLM Configuration
**Location:** `src/agentic/src/core/config.py`

Environment variables control the LLM:
- `OPENAI_API_KEY` - Your API key (required)
- `LLM_PROVIDER` - Provider: openai/anthropic (default: openai)
- `LLM_MODEL` - Model name (default: gpt-4)
- `LLM_TEMPERATURE` - Temperature 0-1 (default: 0.0)

### MCP Server URL
**Location:** `backend/.env` or uses default

```env
MCP_SERVER_URL=http://127.0.0.1:8000/mcp
```

Default: `http://127.0.0.1:8000/mcp`

## UI Graph Visualization

**Location:** `frontend/js/graph.js`

The graph visualization matches the actual LangGraph structure:

### Nodes (from graph.js line 52-65)
```javascript
'START'                           // Entry point
'primary_assistant'               // Primary assistant node
'enter_requirements_analyst'      // Enter requirements specialist
'requirements_analyst'            // Requirements analyst node
'enter_project_manager'           // Enter project manager
'project_manager'                 // Project manager node
'enter_architecture_designer'     // Enter architecture designer
'architecture_designer'           // Architecture designer node
'enter_configuration_specialist'  // Enter config specialist
'configuration_specialist'        // Config specialist node
'enter_code_generator'            // Enter code generator
'code_generator'                  // Code generator node
'leave_skill'                     // Leave skill/return to primary
'END'                             // End point
```

These **exactly match** the nodes in `src/agentic/src/core/graph.py`:
- Line 104: `builder.add_node("primary_assistant", ...)`
- Line 115: `builder.add_node("requirements_analyst", ...)`
- Line 120: `builder.add_node("project_manager", ...)`
- Line 125: `builder.add_node("architecture_designer", ...)`
- Line 130: `builder.add_node("configuration_specialist", ...)`
- Line 135: `builder.add_node("code_generator", ...)`

### Graph Updates
**Location:** `frontend/js/graph.js` line 176

```javascript
updateActiveNode(nodeName) {
    this.activeNode = nodeName;
    this.visitedNodes.add(nodeName);
    this.draw();
}
```

Called when state updates from backend:
```javascript
// frontend/js/state.js line 31
const activeAssistant = state.dialog_state?.[state.dialog_state.length - 1];
graphManager.updateActiveNode(activeAssistant);
```

## State Monitoring

**Location:** `frontend/js/state.js`

The UI displays all state fields:

### Current Assistant
```javascript
// Line 49
const activeAssistant = state.dialog_state?.[state.dialog_state.length - 1];
```

Maps to `NNGeneratorState.dialog_state` from `src/agentic/src/core/state.py`

### Project Information
```javascript
// Line 64-68
state.project_name || 'Not created'
state.project_path || 'N/A'
```

Maps to:
- `NNGeneratorState.project_name`
- `NNGeneratorState.project_path`

### Architecture Information
```javascript
// Line 78-82
state.architecture_path || 'Not created'
state.architecture_name || 'N/A'
```

Maps to:
- `NNGeneratorState.architecture_path`
- `NNGeneratorState.architecture_name`

### Configuration Information
```javascript
// Line 92-97
state.config_path || 'Not created'
```

Maps to:
- `NNGeneratorState.config_path`

### Training Status
```javascript
// Line 107-111
state.training_status || 'Not started'
```

Maps to:
- `NNGeneratorState.training_status`

## Tool Call Tracking

**Location:** `backend/server.py` line 146

```python
if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
    for tool_call in last_message.tool_calls:
        tool_info = {
            'name': tool_call.get('name', 'unknown'),
            'args': tool_call.get('args', {}),
            'timestamp': datetime.now().isoformat()
        }
        socketio.emit('tool_call', tool_info)
```

**Location:** `frontend/js/chat.js` line 94

```javascript
addToolCall(toolCall) {
    const toolDiv = document.createElement('div');
    toolDiv.className = 'tool-call';
    toolDiv.innerHTML = `
        <strong>🔧 ${toolCall.name}</strong>
        <pre>${JSON.stringify(toolCall.args, null, 2)}</pre>
    `;
    this.chatMessages.appendChild(toolDiv);
}
```

## Real-Time Updates

**Location:** `backend/server.py` + `frontend/js/api.js`

### WebSocket Events

1. **processing_started**
   - Backend emits: Line 132
   - Frontend receives: `api.js` line 56

2. **tool_call**
   - Backend emits: Line 156
   - Frontend receives: `api.js` line 60

3. **processing_complete**
   - Backend emits: Line 172
   - Frontend receives: `api.js` line 64

## Verification

### Backend Uses Exact Graph
```python
# backend/server.py line 35
from src.agentic.src.core.graph import create_graph
from src.agentic.src.utils.graph_utils import (
    create_user_message_input,
    create_conversation_config
)
from src.agentic.src.mcp import get_mcp_helper, cleanup_global_mcp
```

All imports are from `src/agentic/src/`, ensuring 100% code reuse.

### Same MCP Client
```python
# backend/server.py line 59
mcp = await get_mcp_helper()
```

Uses the global MCP helper from `src/agentic/src/mcp/__init__.py`, same as CLI.

### Same Configuration
```python
# src/agentic/src/core/graph.py line 267
config = load_config(config_file)
```

Both UI and CLI use `SystemConfig.load()` which:
1. Loads from environment variables
2. Uses same defaults
3. Configures same LLM
4. Connects to same MCP server

## Testing Integration

To verify the integration is working:

1. **Check Graph Initialization**
   ```
   Flask Backend window should show:
   "Graph initialized successfully!"
   ```

2. **Check State Updates**
   - Send message in UI
   - Watch State Monitor update
   - Verify dialog_state changes

3. **Check Tool Execution**
   - Create a project
   - Watch tool call appear in chat
   - Verify project created in filesystem

4. **Check Node Transitions**
   - Send message
   - Watch graph highlight different nodes
   - Verify matches dialog_state in State Monitor

## Common Issues

### "Graph not initialized"
**Cause:** Backend failed to create graph
**Check:**
1. MCP server running on port 8000
2. OPENAI_API_KEY set in backend/.env
3. Backend console for error messages

### Graph not updating
**Cause:** dialog_state not being tracked
**Check:**
1. State Monitor → Current tab → Dialog State
2. Should show array like `["primary_assistant"]`
3. If empty, graph routing has issues

### Tools not executing
**Cause:** MCP client not connected
**Check:**
1. MCP server running
2. Backend console: "Session ID: ..."
3. Try: `curl http://127.0.0.1:8000/mcp`

## Summary

✅ **100% Integration** - Uses exact same graph code as CLI
✅ **Same Assistants** - All 6 assistants from agentic folder
✅ **Same Tools** - All MCP tools available
✅ **Same State** - Full NNGeneratorState tracking
✅ **Same Config** - Shares environment variables and config
✅ **Real-Time** - WebSocket updates for live feedback

The UI is a **view layer** on top of the existing agentic system, not a separate implementation.
