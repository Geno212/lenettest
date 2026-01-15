/**
 * API Module
 * Handles communication with the Python chatbot bridge via Electron IPC
 * This directly mimics run_chatbot.py behavior
 */

class API {
    constructor() {
        this.connected = false;
        this.sessionId = null;
        this.threadId = null;
        this.initializeBridgeListeners();
    }

    initializeBridgeListeners() {
        // Listen for bridge messages
        window.electronAPI.onBridgeMessage((message) => {
            console.log('Bridge message:', message);
            
            switch (message.type) {
                case 'initialized':
                    this.connected = true;
                    this.sessionId = message.data.session_id;
                    this.threadId = message.data.thread_id;
                    if (this.onBackendConnected) {
                        this.onBackendConnected(message.data);
                    }
                    break;
                
                case 'response':
                    // Assistant response
                    if (this.onProcessingComplete) {
                        this.onProcessingComplete({
                            response: message.content,
                            state: message.state
                        });
                    }
                    break;
                
                case 'status':
                    // Status response
                    if (this.onStatusReceived) {
                        this.onStatusReceived(message.data);
                    }
                    break;
                
                case 'error':
                    if (this.onProcessingError) {
                        this.onProcessingError({ error: message.message });
                    }
                    break;
            }
        });

        // Listen for bridge errors (actual errors, not info/debug messages)
        window.electronAPI.onBridgeError((error) => {
            console.error('Bridge error:', error);
            // Only show actual errors, not info messages
            if (this.onProcessingError) {
                this.onProcessingError({ error });
            }
        });

        // Listen for bridge closed
        window.electronAPI.onBridgeClosed((code) => {
            console.log('Bridge closed with code:', code);
            this.connected = false;
            if (this.onConnectionChange) {
                this.onConnectionChange(false);
            }
        });
    }

    async sendMessage(message) {
        // Send message via Electron IPC
        try {
            if (this.onProcessingStarted) {
                this.onProcessingStarted({ message });
            }
            
            const result = await window.electronAPI.sendMessage(message);
            return result;
        } catch (error) {
            console.error('Send message failed:', error);
            throw error;
        }
    }

    async getStatus() {
        // Request status via Electron IPC
        try {
            const result = await window.electronAPI.getStatus();
            return result;
        } catch (error) {
            console.error('Get status failed:', error);
            throw error;
        }
    }

    // Event callbacks (to be overridden)
    onConnectionChange(connected) {
        // Override this
    }

    onBackendConnected(data) {
        // Override this
    }

    onProcessingStarted(data) {
        // Override this
    }

    onProcessingComplete(data) {
        // Override this
    }

    onProcessingError(data) {
        // Override this
    }
    
    onStatusReceived(data) {
        // Override this
    }
}

// Export singleton instance
const api = new API();
