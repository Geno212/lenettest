/**
 * Preload Script
 * 
 * This script runs before the renderer process and exposes
 * safe APIs to the frontend.
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Send message to chatbot bridge
  sendMessage: (message) => ipcRenderer.invoke('send-message', message),
  
  // Get current status
  getStatus: () => ipcRenderer.invoke('get-status'),
  
  // Upload image
  uploadImage: (filePath) => ipcRenderer.invoke('upload-image', filePath),
  
  // Bridge events
  onBridgeMessage: (callback) => ipcRenderer.on('bridge-message', (event, data) => callback(data)),
  onBridgeError: (callback) => ipcRenderer.on('bridge-error', (event, data) => callback(data)),
  onBridgeClosed: (callback) => ipcRenderer.on('bridge-closed', (event, data) => callback(data)),
  
  // Menu actions
  onNewSession: (callback) => ipcRenderer.on('menu-new-session', callback),
  onSaveChat: (callback) => ipcRenderer.on('menu-save-chat', callback),
  onLoadChat: (callback) => ipcRenderer.on('menu-load-chat', callback),
  onClearChat: (callback) => ipcRenderer.on('menu-clear-chat', callback),
  onResetGraph: (callback) => ipcRenderer.on('menu-reset-graph', callback),
  onShowDocs: (callback) => ipcRenderer.on('menu-show-docs', callback),
  onShowAbout: (callback) => ipcRenderer.on('menu-show-about', callback),
  
  // Remove listeners
  removeListener: (channel, callback) => ipcRenderer.removeListener(channel, callback)
});
