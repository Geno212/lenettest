# Quick Start Guide - Simplified Electron UI

## What Changed?

The Electron UI now **directly mimics `run_chatbot.py`** without any Flask server middleware:

- ❌ **Removed**: Flask server, HTTP, WebSocket, axios, socket.io
- ✅ **Added**: Direct Python bridge via stdio (same code as run_chatbot.py)
- ✅ **Result**: Simpler, faster, more reliable

## How to Run

### 1. Start the Application

```bash
cd "d:\Siemens GP\Application-Specific-Deep-Learning-Accelerator-Designer\src\agentic\ui-electron"
launch.bat
```

This will:
1. **Start MCP Server** (in a separate terminal window)
2. **Start Electron UI** (which internally starts the chatbot bridge)

### 2. Wait for Initialization

The splash screen will show while:
- MCP server starts up
- Chatbot bridge initializes
- Graph is created

### 3. Start Chatting

Once you see the welcome message, you can start typing prompts:

**Example Prompts:**
- "Build a classifier for MNIST dataset with 95% accuracy"
- "Create an image classifier using ResNet18"
- "I want to train a CNN on CIFAR10"

**Special Commands:**
- `help` - Show help and examples
- `status` - Show current workflow status  
- `clear` - Clear chat history

## Architecture Comparison

### Old Architecture (Complex)
```
Frontend → axios/HTTP → Flask Server → WebSocket → Graph
                ↓
          Socket.io Events
```

### New Architecture (Simple)
```
Frontend → IPC → Electron Main → stdio/JSON → chatbot_bridge.py → Graph
                                               (same as run_chatbot.py)
```

## What's the Same?

The **chatbot_bridge.py** uses the **exact same code** as `run_chatbot.py`:

```python
# Both use:
graph = await create_graph()
mcp = await get_mcp_helper()
initial_state = create_initial_state()

input_dict, config = create_user_message_input(...)
result = await graph.ainvoke(input_dict, config=config)
response = format_graph_output(result)
```

The only difference is:
- `run_chatbot.py` → prints to terminal
- `chatbot_bridge.py` → sends JSON to Electron

## Files Modified

### New Files
- `backend/chatbot_bridge.py` - Direct bridge (replaces server.py)
- `SIMPLIFIED_ARCHITECTURE.md` - Architecture documentation
- `QUICK_START_SIMPLIFIED.md` - This guide

### Updated Files
- `electron/main.js` - Spawns bridge process instead of connecting to Flask
- `electron/preload.js` - IPC bridge API
- `frontend/js/api.js` - Removed HTTP/WebSocket, uses IPC
- `frontend/js/app.js` - Simplified initialization
- `frontend/index.html` - Removed external script tags
- `package.json` - Removed axios, socket.io dependencies
- `launch.bat` - Removed Flask server startup

### Deprecated Files (Still present but not used)
- `backend/server.py` - Old Flask server
- `start_backend.bat` - Started Flask server

## Troubleshooting

### Issue: "Initializing..." forever

**Cause**: Chatbot bridge failed to start or MCP server not running

**Solution**:
1. Open Electron DevTools (View → Toggle Developer Tools)
2. Check Console for errors
3. Look for bridge startup messages or errors
4. Ensure MCP server is running in its window

### Issue: No response to messages

**Cause**: Bridge not receiving messages or crashed

**Solution**:
1. Check Electron DevTools Console
2. Look for "Bridge message:" logs
3. Check if bridge stderr shows errors
4. Restart the application

### Issue: Import errors or module not found

**Cause**: Python environment issues

**Solution**:
1. Ensure virtual environment is activated
2. Check that all dependencies are installed
3. Verify Python path in `electron/main.js`

## Testing the Bridge Standalone

You can test the bridge directly without Electron:

```bash
cd "d:\Siemens GP\Application-Specific-Deep-Learning-Accelerator-Designer"
venv\Scripts\activate
python -m src.agentic.ui-electron.backend.chatbot_bridge
```

Then type JSON messages:
```json
{"type": "message", "content": "Build a CNN for MNIST"}
```

Press Enter. The bridge will respond with JSON.

Type `Ctrl+C` to exit.

## Debugging

### Electron Console
- Open with `Ctrl+Shift+I` or View → Toggle Developer Tools
- Shows all JavaScript errors and logs
- Shows bridge stdout/stderr output

### Python Errors
All Python errors from the bridge appear in the Electron console with the prefix "Bridge error:"

### Message Flow
You can see the complete message flow:
1. User types → `api.sendMessage()`
2. Frontend → `window.electronAPI.sendMessage()`
3. Preload → `ipcRenderer.invoke('send-message')`
4. Main → `chatbotBridge.stdin.write(JSON)`
5. Bridge → `graph.ainvoke()`
6. Bridge → `bridge.stdout.write(JSON)`
7. Main → `mainWindow.webContents.send('bridge-message')`
8. Preload → `ipcRenderer.on('bridge-message')`
9. Frontend → `api.onBridgeMessage()`
10. Chat → `chatManager.addAssistantMessage()`

## Next Steps

Once running successfully:

1. **Try the example prompts** in the help menu
2. **Check the graph tab** to see workflow visualization
3. **Monitor the status tab** to see project state
4. **Use the status command** to see detailed information

## Need Help?

- Check `SIMPLIFIED_ARCHITECTURE.md` for detailed architecture info
- Check `TROUBLESHOOTING.md` for common issues
- Check Electron DevTools Console for errors
- Check MCP Server window for MCP-related errors

## Comparison with Terminal

| Feature | run_chatbot.py | Electron UI |
|---------|----------------|-------------|
| Input | Terminal stdin | Chat input box |
| Output | Terminal stdout | Chat messages |
| Commands | Type directly | Type in chat |
| Status | `status` command | Status tab + command |
| Help | `help` command | Help command + menu |
| Graph | Text description | Visual graph |
| Session | Single session | Persistent session |
| Multi-window | Single terminal | Tabs + panels |

The Electron UI is essentially a **graphical version** of the terminal chatbot with the **exact same backend logic**.
