# 🧪 UI Testing Guide

## ⚙️ First-Time Setup (Required!)

### 1. Set Your OpenAI API Key

**BEFORE starting the app, you MUST configure your API key:**

```powershell
# Navigate to backend folder
cd src\agentic\ui-electron\backend

# Copy the example .env file
copy .env.example .env

# Edit the .env file and add your API key
notepad .env
```

In the `.env` file, replace `your_openai_api_key_here` with your actual key:
```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxx
```

**Without this, the backend will fail to start!**

---

## Quick Start Testing

### 2. Start the Application

```powershell
cd src\agentic\ui-electron
.\launch.bat
```

**What to expect:**
- 3 windows will open: MCP Server, Flask Backend, Electron UI
- Wait ~5 seconds for all services to start
- Electron window should show chat interface

---

## 🎯 Test Scenarios

### Test 1: Simple Greeting (Verify Connection)

**Type:**
```
Hello
```

**Expected Result:**
- ✅ Message appears in chat with "You:" prefix
- ✅ Response from assistant appears
- ✅ Graph shows "primary_assistant" node highlighted in blue
- ✅ State Monitor → Dialog State shows `["primary_assistant"]`

---

### Test 2: Create a Simple CNN Project

**Type:**
```
Create a CNN for MNIST classification
```

**Expected Result:**
- ✅ Assistant analyzes your request
- ✅ Graph highlights different nodes as it moves through workflow:
  - `primary_assistant` → `enter_requirements_analyst` → `requirements_analyst`
  - Then `enter_project_manager` → `project_manager`
- ✅ State Monitor shows:
  - **Current Assistant**: Changes as workflow progresses
  - **Project Name**: Shows created project name (e.g., "mnist_cnn")
  - **Dialog State**: Shows stack of active assistants
- ✅ Tool calls appear in chat (e.g., `create_project`)
- ✅ Chat shows project creation confirmation

---

### Test 3: Design Architecture

**Type:**
```
Design a simple CNN with 3 conv layers
```

**Expected Result:**
- ✅ Graph moves to `architecture_designer` node (orange)
- ✅ State Monitor → Architecture shows file path
- ✅ Tool call: `create_architecture` or similar
- ✅ Assistant describes the architecture

---

### Test 4: Configure Training

**Type:**
```
Set batch size to 32 and learning rate to 0.001
```

**Expected Result:**
- ✅ Graph moves to `configuration_specialist` node (purple)
- ✅ State Monitor → Configuration shows config details
- ✅ Tool call: `create_config` or similar
- ✅ Confirmation message in chat

---

### Test 5: Generate Code

**Type:**
```
Generate the training code
```

**Expected Result:**
- ✅ Graph moves to `code_generator` node (teal)
- ✅ Tool call: `generate_code`
- ✅ Code generation confirmation
- ✅ File paths shown in chat

---

### Test 6: Use Pretrained Model

**Type:**
```
Use ResNet-18 pretrained on ImageNet
```

**Expected Result:**
- ✅ Assistant routes to architecture designer
- ✅ Tool call: `create_pretrained_architecture`
- ✅ Shows pretrained model selection
- ✅ State updates with architecture file

---

## 🎨 UI Component Tests

### Chat Interface
- ✅ **Scrolling**: Auto-scrolls to bottom on new messages
- ✅ **Message bubbles**: Your messages (right, blue), Assistant (left, gray)
- ✅ **Tool calls**: Yellow boxes showing tool name and arguments
- ✅ **Errors**: Red boxes if something fails
- ✅ **Input field**: Type and press Enter or click Send

### Graph Visualization
- ✅ **Active node**: Blue highlight on current node
- ✅ **Visited nodes**: Blue stroke on visited nodes
- ✅ **Visited paths**: Blue edges showing path taken
- ✅ **Zoom in/out**: Click + and - buttons
- ✅ **Reset**: Click reset button to center graph

### State Monitor
- ✅ **Tabs**: Click between Current, Project, Architecture, Config, Training
- ✅ **Refresh**: Click refresh icon to update state
- ✅ **Auto-update**: State updates automatically on tool calls
- ✅ **Current tab**: Shows active assistant and dialog state
- ✅ **Project tab**: Shows project name and path
- ✅ **Architecture tab**: Shows architecture file and model info
- ✅ **Config tab**: Shows training configuration
- ✅ **Training tab**: Shows training status and metrics

---

## 🔍 What to Watch For

### Graph Behavior
1. **Node Highlighting**:
   - Active node should be blue
   - Visited nodes have blue outline
   - Inactive nodes are gray

2. **Path Visualization**:
   - Blue edges connect visited nodes
   - Watch the flow: START → Primary → Specialists → Leave → END

3. **Node Names** (as shown in graph):
   - `START` - Entry point
   - `Primary` - Primary assistant
   - `Enter Req` - Enter requirements analyst
   - `Requirements` - Requirements analyst
   - `Enter Proj` - Enter project manager
   - `Project` - Project manager
   - `Enter Arch` - Enter architecture designer
   - `Architecture` - Architecture designer
   - `Enter Config` - Enter configuration specialist
   - `Config` - Configuration specialist
   - `Enter Code` - Enter code generator
   - `Code Gen` - Code generator
   - `Leave` - Leave skill
   - `END` - End point

### State Updates
1. **Dialog State**: Stack showing current conversation flow
   - Example: `["primary_assistant", "requirements_analyst"]`

2. **Project Info**: After project creation
   - Project name (e.g., "mnist_cnn")
   - Project path (full file path)

3. **Architecture Info**: After architecture design
   - Architecture file path
   - Model type (CNN, pretrained, etc.)

4. **Config Info**: After configuration
   - Batch size
   - Learning rate
   - Optimizer settings

### Tool Calls
Watch for these tool names in chat:
- `create_project` - Creates new project
- `create_architecture` - Designs architecture
- `create_pretrained_architecture` - Uses pretrained model
- `create_config` - Sets training configuration
- `generate_code` - Generates code files
- `list_projects` - Shows existing projects
- `list_architectures` - Shows architecture options

---

## 🚨 Common Issues & Fixes

### "Connection Failed" Error
**Cause**: Backend not running
**Fix**:
1. Check if Flask Backend window shows "Running on http://127.0.0.1:5000"
2. Check if MCP Server window shows "Server running"
3. Restart launch.bat

### Graph Not Updating
**Cause**: WebSocket disconnected
**Fix**:
1. Check browser console (F12) for errors
2. Click refresh icon in State Monitor
3. Restart Electron app

### State Monitor Shows "No Data"
**Cause**: No state yet or not initialized
**Fix**:
1. Send a message to initialize conversation
2. Click refresh icon
3. Check that tool calls are completing successfully

### Messages Not Sending
**Cause**: Backend error or initialization issue
**Fix**:
1. Check Flask Backend window for Python errors
2. Check MCP Server window for errors
3. Look at Electron DevTools console (F12)

---

## 📊 Complete Workflow Test

Try this complete workflow to test everything:

```
1. "Hello" 
   → Verify connection works

2. "Create a project called test_cnn for image classification"
   → Watch graph move to Project node
   → Check Project tab in State Monitor

3. "Design a CNN with 2 conv layers and 2 dense layers"
   → Watch graph move to Architecture node
   → Check Architecture tab

4. "Configure training with batch size 64 and learning rate 0.0001"
   → Watch graph move to Config node
   → Check Config tab

5. "Generate the code"
   → Watch graph move to Code Gen node
   → Check for file paths in chat

6. "Thank you"
   → Should return to primary assistant
```

---

## 🎓 Advanced Testing

### Test Error Handling
```
Invalid project name with spaces and special chars!@#
```
**Expected**: Error message displayed, graph stays at current node

### Test State Persistence
1. Create a project
2. Close and restart Electron app
3. Type: "What's my current project?"
**Expected**: Should remember project (if backend still running)

### Test Multiple Projects
```
Create project mnist_cnn
Create project cifar_classifier
```
**Expected**: Both projects created, state updates correctly

---

## 📝 Testing Checklist

Before reporting success, verify:

- [ ] All 3 services start without errors
- [ ] Chat interface sends and receives messages
- [ ] Graph highlights active nodes correctly
- [ ] Graph shows visited paths in blue
- [ ] Zoom in/out and reset work
- [ ] State Monitor tabs all work
- [ ] Refresh button updates state
- [ ] Tool calls appear in chat
- [ ] Project creation works
- [ ] Architecture design works
- [ ] Configuration works
- [ ] Code generation works
- [ ] Error messages display properly
- [ ] WebSocket updates work (real-time)

---

## 🎯 Quick Test Commands

Copy-paste these for quick testing:

```
# Test 1: Connection
Hello

# Test 2: Project
Create a CNN project called my_test

# Test 3: Architecture  
Design a simple CNN

# Test 4: Config
Set batch size to 32

# Test 5: Generate
Generate the code

# Test 6: Pretrained
Use ResNet-18 pretrained model

# Test 7: List
What projects do I have?
```

---

**Happy Testing! 🚀**

If everything works, you should see:
- ✅ Smooth chat interaction
- ✅ Graph animating through nodes
- ✅ State updating in real-time
- ✅ Tool calls executing successfully
- ✅ Projects/architectures/configs being created
