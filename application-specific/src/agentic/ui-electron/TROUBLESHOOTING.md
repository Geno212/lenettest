# Troubleshooting Guide: Electron GUI After launch.bat

## Your Implementation is Correct! ✅

Your `server.py` **already uses `ainvoke`** just like `run_chatbot.py` does. The core logic is identical.

See `IMPLEMENTATION_COMPARISON.md` for detailed comparison.

---

## Common Issues After Starting with launch.bat

### Issue 1: MCP Server Not Connecting

**Symptoms:**
- Backend shows: "Failed to initialize graph"
- MCP Server window shows errors
- Backend stuck at "Initializing graph..."

**Solution:**
```powershell
# Check MCP Server window - look for:
✅ "Server log: Created manager for session [session_id]"

# If not showing, restart MCP Server:
cd "d:\Siemens GP\Application-Specific-Deep-Learning-Accelerator-Designer"
venv\Scripts\python.exe -m src.cli.mcp_server.main
```

---

### Issue 2: Backend Can't Connect to MCP

**Symptoms:**
- Backend shows: "Failed to initialize graph"
- Frontend shows: "Connection Failed"

**Check:**
1. MCP Server must be running FIRST
2. Wait 10 seconds after MCP starts before starting backend
3. Check MCP server console for session creation

**Solution:**
```bat
# Restart in correct order:
1. Start MCP Server first
2. Wait 10 seconds
3. Start Backend
4. Start Electron
```

---

### Issue 3: Environment Variables Missing

**Symptoms:**
- Backend starts but no API calls work
- Error: "OPENAI_API_KEY not found"

**Solution:**
Create `.env` file in `backend/` folder:

```bash
# backend/.env
OPENAI_API_KEY=your-api-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key-here  # Optional
LANGCHAIN_PROJECT=agentic-nn-generator
```

---

### Issue 4: Frontend Can't Connect to Backend

**Symptoms:**
- Electron opens but shows "Connection Failed"
- Status: "Connecting... (10/10)" then fails

**Check:**
```powershell
# Test backend manually:
curl http://127.0.0.1:5000/api/health

# Should return:
{"status":"ok","graph_initialized":true,...}
```

**Solution:**
- Ensure Flask backend is running on port 5000
- Check Windows Firewall isn't blocking
- Verify backend window shows: "Starting Flask server on http://127.0.0.1:5000"

---

### Issue 5: Backend Running but Graph Not Initialized

**Symptoms:**
- Backend window shows: "Failed to initialize graph"
- Health check returns: `"graph_initialized": false`

**Root Causes:**
1. MCP Server not running
2. MCP Server not ready when backend started
3. Path issues preventing MCP connection

**Solution:**
```bat
# 1. Stop everything
# 2. Start ONLY MCP Server
cd "d:\Siemens GP\Application-Specific-Deep-Learning-Accelerator-Designer"
venv\Scripts\python.exe -m src.cli.mcp_server.main

# 3. Wait for:
✅ "Server log: Created manager for session..."

# 4. THEN start backend in new window:
cd "d:\Siemens GP\Application-Specific-Deep-Learning-Accelerator-Designer\src\agentic\ui-electron\backend"
..\..\..\..\venv\Scripts\python.exe server.py

# 5. Look for:
✅ "Graph initialized successfully!"
✅ "Starting Flask server on http://127.0.0.1:5000"

# 6. THEN start Electron:
cd "d:\Siemens GP\Application-Specific-Deep-Learning-Accelerator-Designer\src\agentic\ui-electron"
npm start
```

---

## Quick Diagnostic Checklist

Run this checklist when things aren't working:

- [ ] **MCP Server running?** Check its window for session creation message
- [ ] **Backend connected to MCP?** Look for "Graph initialized successfully!"
- [ ] **Flask server started?** Should show "Starting Flask server..."
- [ ] **Backend health OK?** Visit http://127.0.0.1:5000/api/health in browser
- [ ] **Electron can connect?** Status badge should show "Connected"
- [ ] **Environment vars set?** Check `backend/.env` exists with OPENAI_API_KEY

---

## Manual Testing

### Test 1: Verify MCP Server
```powershell
cd "d:\Siemens GP\Application-Specific-Deep-Learning-Accelerator-Designer"
venv\Scripts\python.exe -m src.cli.mcp_server.main
```

**Expected output:**
```
Server log: Created manager for session b7b5c891650c467ab003a64f14f76c98
```

### Test 2: Verify run_chatbot.py Works
```powershell
cd "d:\Siemens GP\Application-Specific-Deep-Learning-Accelerator-Designer"
venv\Scripts\python.exe -m src.agentic.src.scripts.run_chatbot
```

**Expected output:**
```
🤖 Neural Network Generator - Interactive Chat
Initializing system...
✅ System and MCP initialized successfully!
📝 Session ID: [session-id]
🧵 Thread ID: mcp_[session-id]
```

If this works, your backend should also work!

### Test 3: Verify Backend
```powershell
cd "d:\Siemens GP\Application-Specific-Deep-Learning-Accelerator-Designer\src\agentic\ui-electron\backend"
..\..\..\..\venv\Scripts\python.exe server.py
```

**Expected output:**
```
Graph initialized successfully!
Session ID: [session-id]
Thread ID: mcp_[session-id]
Starting Flask server on http://127.0.0.1:5000
```

### Test 4: Test API
Open browser to: http://127.0.0.1:5000/api/health

**Expected response:**
```json
{
  "status": "ok",
  "graph_initialized": true,
  "session_id": "[session-id]",
  "thread_id": "mcp_[first-8-chars]",
  "is_processing": false
}
```

---

## Understanding the Startup Sequence

```
┌─────────────────┐
│  1. MCP Server  │ ← Must start FIRST
│   (Port: ???)   │   Creates session
└────────┬────────┘
         │
         │ Wait 10 seconds
         │
         ▼
┌─────────────────┐
│  2. Backend     │ ← Connects to MCP
│  (Port: 5000)   │   Creates graph
└────────┬────────┘
         │
         │ Wait 5 seconds
         │
         ▼
┌─────────────────┐
│  3. Electron    │ ← Connects to Backend
│   (Frontend)    │   Shows UI
└─────────────────┘
```

**Critical:** Each step depends on the previous step being fully initialized!

---

## Log Files to Check

If issues persist, check these logs:

1. **MCP Server Console** - Shows session creation and tool calls
2. **Backend Console** - Shows graph initialization and API requests  
3. **Electron Console** (F12 in app) - Shows frontend errors
4. **Browser Network Tab** (F12) - Shows API call failures

---

## Most Common Fix

**90% of issues are timing-related:**

The `launch.bat` waits 10 seconds for MCP, but sometimes it needs more time.

**Manual startup solves this:**
1. Start MCP manually → wait for session message
2. Start Backend manually → wait for "Graph initialized"
3. Start Electron → should connect immediately

---

## Still Not Working?

If you've verified all the above and it's still not working, provide:

1. **MCP Server console output** - First 20 lines
2. **Backend console output** - First 30 lines
3. **Browser console (F12)** - Any red errors
4. **Health check response** - From http://127.0.0.1:5000/api/health

This will help diagnose the specific issue!
