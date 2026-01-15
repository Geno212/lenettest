# Siemens Neural Network Generator - Electron UI

Desktop application for the Agentic Neural Network Generator with Siemens branding.

## ✨ Features

- 🎨 **Siemens Branding** - Professional corporate design with Siemens teal colors
- 🌓 **Dark/Light Themes** - Toggle between themes with saved preferences
- 💬 **Claude-Like Chat** - Smooth, professional chat interface
- 📊 **Live Graph Visualization** - Real-time workflow graph updates
- 🚀 **Splash Screen** - Beautiful loading screen with Siemens logo

## 🚀 Quick Start

### First Time Setup

```bash
# 1. Navigate to this directory
cd src/agentic/ui-electron

# 2. Install Node.js dependencies
npm install

# 3. Install Python dependencies
cd backend
pip install -r requirements.txt
cd ..
```

### Running the Application

**Easy Way (Windows):**
```powershell
.\launch.bat
```

This automatically starts:
- MCP Server
- Flask Backend
- Electron UI

**Manual Way:**
```bash
# Terminal 1: MCP Server
cd src/cli/mcp_server
python main.py

# Terminal 2: Flask Backend
cd src/agentic/ui-electron/backend
python server.py

# Terminal 3: Electron UI
cd src/agentic/ui-electron
npm start
```

## 📂 Project Structure

```
ui-electron/
├── launch.bat           # Windows launcher
├── package.json         # Node.js config
├── electron/            # Electron main process
├── frontend/            # HTML/CSS/JS UI
└── backend/             # Flask server
    └── server.py        # Python backend
```

## 🔧 Configuration

**IMPORTANT: Set your OpenAI API key before starting!**

Create a `.env` file in the `backend/` directory:

```bash
cd src/agentic/ui-electron/backend
copy .env.example .env
```

Edit `backend/.env` and add your API key:

```env
# REQUIRED: Your OpenAI API Key
OPENAI_API_KEY=sk-your-actual-api-key-here

# OPTIONAL: Model configuration (defaults shown)
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.0
```

**Quick setup:**
```powershell
# Navigate to backend folder
cd src\agentic\ui-electron\backend

# Copy the example file
copy .env.example .env

# Edit .env file and add your API key
notepad .env
```

## 🚨 Troubleshooting

**Connection Failed?**
1. Make sure MCP server is running on port 8000
2. Make sure Flask backend is running on port 5000
3. Check firewall settings

**UI Won't Start?**
1. Run `npm install` in ui-electron directory
2. Run `pip install -r backend/requirements.txt`
3. Check Node.js is v16+ with `node --version`
