# 🚀 Quick Start Guide

## Step 1: Configure API Key

```powershell
# Go to ui-electron folder
cd src\agentic\ui-electron

# Run setup script (installs dependencies + opens .env editor)
.\setup.bat
```

When notepad opens, add your OpenAI API key:
```env
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

Save and close notepad.

---

## Step 2: Start Everything

```powershell
.\launch.bat
```

Wait for 3 windows to open:
1. **MCP Server** - Shows "Server running on http://127.0.0.1:8000"
2. **Flask Backend** - Shows "Graph initialized successfully!"
3. **Electron UI** - Desktop app window

---

## Step 3: Test It

In the Electron app, type:

```
Hello
```

You should see:
- ✅ Your message appears (right side, blue)
- ✅ Assistant responds (left side, gray)
- ✅ Graph highlights "Primary" node in blue
- ✅ State Monitor shows dialog_state: ["primary_assistant"]

---

## Step 4: Create a Project

```
Create a CNN for MNIST classification
```

Watch:
- ✅ Graph nodes light up as it progresses
- ✅ Tool calls appear (yellow boxes)
- ✅ State Monitor updates with project name
- ✅ Assistant confirms project creation

---

## That's It!

📖 **More test commands**: See `TESTING_GUIDE.md`
🔌 **How it works**: See `INTEGRATION.md`
❓ **Issues**: Check `README.md` troubleshooting section

---

## File Locations

- **API Key**: `backend/.env`
- **Launcher**: `launch.bat`
- **Setup**: `setup.bat`
