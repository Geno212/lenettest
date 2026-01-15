# Simplified Electron UI Architecture

## Overview

The Electron UI has been **simplified** to directly mimic the behavior of `run_chatbot.py`. The Flask server middleware has been removed, and the Electron app now communicates directly with a Python bridge process via stdio.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Electron UI                            │
│  ┌──────────────┐    ┌──────────────┐   ┌──────────────┐  │
│  │   Frontend   │◄───┤  Preload.js  │◄──┤   Main.js    │  │
│  │  (HTML/CSS/  │    │    (IPC      │   │  (Electron   │  │
│  │  JavaScript) │    │    Bridge)   │   │   Process)   │  │
│  └──────────────┘    └──────────────┘   └───────┬──────┘  │
└──────────────────────────────────────────────────│─────────┘
                                                    │ stdio
                                                    │ (JSON)
┌───────────────────────────────────────────────────▼─────────┐
│              chatbot_bridge.py                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Exactly mimics run_chatbot.py:                      │  │
│  │  • create_graph()                                    │  │
│  │  • get_mcp_helper()                                  │  │
│  │  • create_initial_state()                            │  │
│  │  • create_user_message_input()                       │  │
│  │  • graph.ainvoke()                                   │  │
│  │  • format_graph_output()                             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                                │
                                │ MCP Protocol
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                      MCP Server                             │
│  • Handles tool calls                                       │
│  • Manages project state                                    │
│  • File system operations                                   │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. **chatbot_bridge.py** (NEW - Replaces Flask server)
- **Direct equivalent of `run_chatbot.py`**
- Communicates via **JSON over stdin/stdout**
- Uses the **exact same functions**:
  - `create_graph()`
  - `create_initial_state()`
  - `create_user_message_input()`
  - `graph.ainvoke()`
  - `format_graph_output()`
- No HTTP/WebSocket overhead
- Simpler, more direct communication

**Message Protocol:**

Input (from Electron):
```json
{"type": "message", "content": "user message"}
{"type": "status"}
{"type": "quit"}
```

Output (to Electron):
```json
{"type": "initialized", "data": {"session_id": "...", "thread_id": "..."}}
{"type": "response", "content": "...", "state": {...}}
{"type": "status", "data": {...}}
{"type": "error", "message": "..."}
```

### 2. **electron/main.js** (UPDATED)
- Spawns `chatbot_bridge.py` as a child process
- Communicates via **stdin/stdout** pipes
- Parses JSON messages from bridge
- Forwards messages to renderer via IPC

**Key Changes:**
- Removed `startPythonBackend()` and `startMCPServer()` functions
- Added `startChatbotBridge()` function
- Simplified process management

### 3. **electron/preload.js** (UPDATED)
- Exposes bridge communication to frontend
- New event handlers:
  - `onBridgeMessage()` - receives messages from bridge
  - `onBridgeError()` - receives errors
  - `onBridgeClosed()` - notified when bridge closes

### 4. **frontend/js/api.js** (SIMPLIFIED)
- Removed all HTTP/WebSocket code
- Removed axios and socket.io dependencies
- Direct IPC communication with main process
- Simplified event system

**Before (Complex):**
- Flask server on port 5000
- Socket.io for WebSocket communication
- HTTP requests for API calls
- Complex connection management

**After (Simple):**
- Direct stdio communication
- Simple JSON message passing
- No ports, no HTTP, no WebSocket

### 5. **frontend/js/app.js** (UPDATED)
- Removed health check polling
- Bridge sends `initialized` message when ready
- Simplified connection status handling

### 6. **frontend/js/chat.js** (NO CHANGES NEEDED)
- Already uses the correct abstractions
- Works with new simplified API

## Comparison with run_chatbot.py

### run_chatbot.py (Terminal)
```python
# Initialize
graph = await create_graph()
mcp = await get_mcp_helper()
thread_id = f"mcp_{session_id[:8]}"
initial_state = create_initial_state()

# Process message
if first_message:
    input_dict, config = create_user_message_input(
        user_input, thread_id, existing_state=initial_state
    )
else:
    input_dict, config = create_user_message_input(
        user_input, thread_id
    )

result = await graph.ainvoke(input_dict, config=config)
response = format_graph_output(result)
print(response)
```

### chatbot_bridge.py (Electron)
```python
# Initialize
self.graph = await create_graph()
mcp = await get_mcp_helper()
self.thread_id = f"mcp_{session_id[:8]}"
self.initial_state = create_initial_state()

# Process message
if self.first_message:
    input_dict, config = create_user_message_input(
        user_input, thread_id, existing_state=self.initial_state
    )
else:
    input_dict, config = create_user_message_input(
        user_input, thread_id
    )

result = await self.graph.ainvoke(input_dict, config=config)
response = format_graph_output(result)
self.send_message({"type": "response", "content": response})
```

**They are IDENTICAL!** Just different output methods.

## Benefits of New Architecture

### 1. **Simplicity**
- ❌ No Flask server
- ❌ No HTTP endpoints
- ❌ No WebSocket connections
- ❌ No port conflicts
- ✅ Just one Python process
- ✅ Direct stdio communication

### 2. **Reliability**
- No network overhead
- No connection timeouts
- No reconnection logic needed
- Process lifecycle tied to Electron

### 3. **Exact Behavior Match**
- Uses the **exact same code** as `run_chatbot.py`
- No translation between HTTP/WebSocket and graph
- Same state management
- Same error handling

### 4. **Easier Debugging**
- All bridge output goes to Electron console
- No need to check multiple servers
- Single process to debug

### 5. **Fewer Dependencies**
- No Flask
- No Flask-CORS
- No Flask-SocketIO
- No axios (frontend)
- No socket.io (frontend)

## Running the Application

### Start Everything:
```bash
cd src/agentic/ui-electron
launch.bat
```

This will:
1. Start the MCP server (in a separate window)
2. Start Electron (which starts the chatbot bridge internally)

### What Runs Where:
- **MCP Server**: Separate process (started by `launch.bat`)
- **Chatbot Bridge**: Child process of Electron (started automatically)
- **Electron UI**: Main Electron app

## File Changes Summary

### New Files:
- `backend/chatbot_bridge.py` - Direct bridge (replaces server.py)

### Updated Files:
- `electron/main.js` - Removed Flask, added bridge spawning
- `electron/preload.js` - Updated IPC API
- `frontend/js/api.js` - Removed HTTP/WebSocket, added IPC
- `frontend/js/app.js` - Simplified connection handling
- `launch.bat` - Removed Flask server startup

### Deprecated Files (No longer used):
- `backend/server.py` - Replaced by chatbot_bridge.py
- `start_backend.bat` - No longer needed

## Troubleshooting

### Bridge Not Starting
Check Electron DevTools console for bridge startup errors:
- Python path issues
- Missing dependencies
- Import errors

### No Response to Messages
Check if bridge received message:
- Electron DevTools → Console → Look for "Bridge message:"
- Check bridge stderr output

### MCP Connection Issues
Bridge will show error if MCP server is not running:
```json
{"type": "error", "message": "Failed to initialize: ..."}
```

Make sure MCP server is running before starting Electron.

## Development

### Testing the Bridge Standalone
You can test the bridge directly:

```bash
cd d:\Siemens GP\Application-Specific-Deep-Learning-Accelerator-Designer
venv\Scripts\python.exe -m src.agentic.ui-electron.backend.chatbot_bridge
```

Then type JSON messages:
```json
{"type": "message", "content": "Build a CNN for MNIST"}
```

Press Enter to send. The bridge will respond with JSON.

### Debugging
1. Open Electron DevTools (View → Toggle Developer Tools)
2. Check Console for bridge messages
3. All Python print() statements go to Electron console
4. Use `console.log()` in frontend for UI debugging

## Future Enhancements

Possible improvements:
1. Add streaming support (send partial responses as they're generated)
2. Add progress indicators (show which node is active)
3. Add tool call visualization
4. Add state persistence (save/load sessions)

All of these can be done by extending the bridge protocol with new message types.
