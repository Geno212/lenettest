/**
 * Electron Main Process
 * 
 * This file handles the main Electron process, window creation,
 * and communication with the Python backend.
 */

const { app, BrowserWindow, ipcMain, Menu } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

// Check if --gpu flag is passed
const useGPU = process.argv.includes('--gpu');

// Configure GPU based on platform and flag
if (process.platform === 'linux' && useGPU) {
  console.log('🚀 GPU Acceleration: ENABLED (NVIDIA Tesla T4)');
  app.commandLine.appendSwitch('enable-gpu-rasterization');
  app.commandLine.appendSwitch('enable-native-gpu-memory-buffers');
  app.commandLine.appendSwitch('enable-zero-copy');
  app.commandLine.appendSwitch('ignore-gpu-blocklist');
} else if (process.platform === 'linux' && !useGPU) {
  console.log('💻 GPU Acceleration: DISABLED (Software Rendering)');
  console.log('   Use "npm run start:gpu" to enable GPU acceleration');
  app.commandLine.appendSwitch('disable-gpu');
  app.commandLine.appendSwitch('disable-gpu-compositing');
} else {
  // Keep GPU disabled on Windows/Mac to avoid crashes
  console.log('💻 GPU Acceleration: DISABLED (Platform: ' + process.platform + ')');
  app.commandLine.appendSwitch('disable-gpu');
  app.commandLine.appendSwitch('disable-gpu-compositing');
  app.commandLine.appendSwitch('disable-software-rasterizer');
}

// Define project root globally
const projectRoot = path.join(__dirname, '../../../../');

let mainWindow;
let splashWindow;
let chatbotBridge; // Direct Python bridge process (no Flask)

// Create splash screen
function createSplashScreen() {
  splashWindow = new BrowserWindow({
    width: 500,
    height: 400,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
    resizable: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  splashWindow.loadFile(path.join(__dirname, '../frontend/splash.html'));
  splashWindow.center();

  // Store the time when splash was created
  splashWindow.createdAt = Date.now();
}

// Create the main window
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 600,
    icon: path.join(__dirname, '../assets/icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    backgroundColor: '#ffffff',
    show: false
  });

  // Load the app
  mainWindow.loadFile(path.join(__dirname, '../frontend/index.html'));

  // Show window when ready and close splash
  mainWindow.once('ready-to-show', () => {
    // Ensure splash is shown for at least 3 seconds
    const splashMinTime = 3000; // 3 seconds minimum
    const elapsed = Date.now() - (splashWindow ? splashWindow.createdAt : Date.now());
    const remainingTime = Math.max(0, splashMinTime - elapsed);

    setTimeout(() => {
      if (splashWindow && !splashWindow.isDestroyed()) {
        splashWindow.close();
      }
      mainWindow.show();
      mainWindow.focus();
    }, remainingTime);
  });

  // Log any load errors
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    console.error('Failed to load:', errorCode, errorDescription);
  });

  // Log console messages from renderer
  mainWindow.webContents.on('console-message', (event, level, message, line, sourceId) => {
    console.log(`Renderer console [${level}]:`, message);
  });

  // Create application menu
  createMenu();

  // Open DevTools automatically to see any errors
  mainWindow.webContents.openDevTools();

  // Handle window close
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Create application menu
function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'New Session',
          accelerator: 'CmdOrCtrl+N',
          click: () => {
            mainWindow.webContents.send('menu-new-session');
          }
        },
        { type: 'separator' },
        {
          label: 'Save Chat',
          accelerator: 'CmdOrCtrl+S',
          click: () => {
            mainWindow.webContents.send('menu-save-chat');
          }
        },
        {
          label: 'Load Chat',
          accelerator: 'CmdOrCtrl+O',
          click: () => {
            mainWindow.webContents.send('menu-load-chat');
          }
        },
        { type: 'separator' },
        {
          label: 'Exit',
          accelerator: 'CmdOrCtrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'View',
      submenu: [
        {
          label: 'Clear Chat',
          accelerator: 'CmdOrCtrl+W',
          click: () => {
            mainWindow.webContents.send('menu-clear-chat');
          }
        },
        {
          label: 'Reset Graph',
          click: () => {
            mainWindow.webContents.send('menu-reset-graph');
          }
        },
        { type: 'separator' },
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'Documentation',
          click: () => {
            mainWindow.webContents.send('menu-show-docs');
          }
        },
        {
          label: 'About',
          click: () => {
            mainWindow.webContents.send('menu-show-about');
          }
        }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// Start the chatbot bridge - direct connection like run_chatbot.py
function startChatbotBridge() {
  const bridgeScript = path.join(__dirname, '../backend/chatbot_bridge.py');

  // Use system Python on Linux, venv on Windows
  const pythonPath = process.platform === 'win32'
    ? path.join(projectRoot, 'trial', 'Scripts', 'python.exe')
    : '/usr/bin/python3';

  console.log('Starting chatbot bridge:', pythonPath, bridgeScript);

  chatbotBridge = spawn(pythonPath, ['-u', bridgeScript], {
    cwd: projectRoot,
    stdio: ['pipe', 'pipe', 'pipe'],
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      PYTHONIOENCODING: 'utf-8'  // Force UTF-8 encoding on Windows
    }
  });

  // Buffer for incomplete JSON messages
  let outputBuffer = '';

  chatbotBridge.stdout.on('data', (data) => {
    outputBuffer += data.toString();

    // Try to parse complete JSON messages
    const lines = outputBuffer.split('\n');
    outputBuffer = lines.pop(); // Keep incomplete line in buffer

    for (const line of lines) {
      if (line.trim()) {
        try {
          const message = JSON.parse(line);
          console.log('Bridge message:', message);

          // Forward to renderer
          if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('bridge-message', message);
          }
        } catch (e) {
          console.error('Failed to parse bridge message:', line, e);
        }
      }
    }
  });

  chatbotBridge.stderr.on('data', (data) => {
    const message = data.toString();

    // Log to console for debugging only - DO NOT send to UI
    // User requested: no error messages in the response
    console.log('Bridge stderr:', message);
  });

  chatbotBridge.on('close', (code) => {
    console.log(`Chatbot bridge exited with code ${code}`);
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('bridge-closed', code);
    }
  });

  chatbotBridge.on('error', (err) => {
    console.error('Failed to start bridge:', err);
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('bridge-error', err.message);
    }
  });
}

// App lifecycle
app.whenReady().then(() => {
  // Show splash screen first
  createSplashScreen();

  // Start the chatbot bridge (it will connect to MCP server started by launch.bat)
  startChatbotBridge();

  // Create main window after a delay (let splash be visible)
  setTimeout(() => {
    createWindow();
  }, 1000); // Wait 1 second before even creating the window

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createSplashScreen();
      setTimeout(() => {
        createWindow();
      }, 1000);
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('quit', () => {
  // Cleanup bridge process
  if (chatbotBridge && !chatbotBridge.killed) {
    chatbotBridge.stdin.write(JSON.stringify({ type: 'quit' }) + '\n');
    chatbotBridge.kill();
  }
});

// IPC handlers
ipcMain.handle('send-message', async (event, message) => {
  // Send message to Python bridge via stdin
  if (chatbotBridge && !chatbotBridge.killed) {
    try {
      const jsonMessage = JSON.stringify({ type: 'message', content: message }) + '\n';
      chatbotBridge.stdin.write(jsonMessage);
      return { success: true };
    } catch (error) {
      console.error('Failed to send message to bridge:', error);
      return { success: false, error: error.message };
    }
  } else {
    return { success: false, error: 'Bridge not running' };
  }
});

ipcMain.handle('get-status', async (event) => {
  // Request status from Python bridge
  if (chatbotBridge && !chatbotBridge.killed) {
    try {
      const jsonMessage = JSON.stringify({ type: 'status' }) + '\n';
      chatbotBridge.stdin.write(jsonMessage);
      return { success: true };
    } catch (error) {
      console.error('Failed to request status:', error);
      return { success: false, error: error.message };
    }
  } else {
    return { success: false, error: 'Bridge not running' };
  }
});

ipcMain.handle('upload-image', async (event, fileData) => {
  // Handle image upload
  try {
    // Validate file data
    if (!fileData || !fileData.name || !fileData.data) {
      return { success: false, error: 'Invalid file data' };
    }

    // Check if it's an image
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp'];
    if (!allowedTypes.includes(fileData.type)) {
      return { success: false, error: 'Invalid file type. Please upload an image file.' };
    }

    // Create uploads directory if it doesn't exist
    const uploadsDir = path.join(projectRoot, 'uploads');
    if (!fs.existsSync(uploadsDir)) {
      fs.mkdirSync(uploadsDir, { recursive: true });
    }

    // Generate safe filename
    const timestamp = Date.now();
    const safeName = fileData.name.replace(/[^a-zA-Z0-9.-]/g, '_');
    const filename = `${timestamp}_${safeName}`;
    const filePath = path.join(uploadsDir, filename);

    // Convert base64 data URL to buffer and save
    const base64Data = fileData.data.replace(/^data:image\/\w+;base64,/, '');
    const buffer = Buffer.from(base64Data, 'base64');
    fs.writeFileSync(filePath, buffer);

    console.log(`Image uploaded: ${filePath}`);

    return {
      success: true,
      filename: filename,
      file_path: filePath
    };
  } catch (error) {
    console.error('Upload error:', error);
    return { success: false, error: error.message };
  }
});
