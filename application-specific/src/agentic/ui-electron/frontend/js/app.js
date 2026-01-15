/**
 * Main Application Module
 * Coordinates all UI components and backend communication
 */

class App {
    constructor() {
        this.currentTheme = localStorage.getItem('theme') || 'light';
        this.initialize();
    }

    async initialize() {
        console.log('Initializing application...');
        
        // Apply saved theme
        this.applyTheme(this.currentTheme);
        
        // Setup theme toggle
        this.setupThemeToggle();
        
        // Setup tab navigation
        this.setupTabs();
        
        // Setup API event handlers
        this.setupAPIHandlers();
        
        // Check backend health
        await this.checkBackendHealth();
        
        console.log('Application initialized');
    }

    setupThemeToggle() {
        const themeToggle = document.getElementById('theme-toggle');
        themeToggle.addEventListener('click', () => {
            this.currentTheme = this.currentTheme === 'light' ? 'dark' : 'light';
            this.applyTheme(this.currentTheme);
            localStorage.setItem('theme', this.currentTheme);
        });
    }

    applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        // Redraw graph with new theme colors
        if (window.graphManager) {
            setTimeout(() => graphManager.draw(), 100);
        }
    }

    setupTabs() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabName = button.getAttribute('data-tab');
                
                // Remove active class from all
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Add active class to selected
                button.classList.add('active');
                document.getElementById(`tab-${tabName}`).classList.add('active');
                
                // Redraw graph if graph tab is selected
                if (tabName === 'graph') {
                    setTimeout(() => graphManager.draw(), 100);
                }
            });
        });
    }

    setupAPIHandlers() {
        // Connection change
        api.onConnectionChange = (connected) => {
            const statusBadge = document.getElementById('connection-status');
            if (connected) {
                statusBadge.textContent = 'Connected';
                statusBadge.classList.add('connected');
                statusBadge.classList.remove('error');
            } else {
                statusBadge.textContent = 'Disconnected';
                statusBadge.classList.remove('connected');
                statusBadge.classList.add('error');
            }
        };
        
        // Backend connected (bridge initialized)
        api.onBackendConnected = (data) => {
            console.log('Bridge initialized:', data);
            const statusBadge = document.getElementById('connection-status');
            statusBadge.textContent = 'Connected';
            statusBadge.classList.add('connected');
            statusBadge.classList.remove('error');
            
            const sessionId = data.session_id || 'unknown';
            const sessionElement = document.getElementById('session-id');
            if (sessionElement) {
                sessionElement.textContent = sessionId.substring(0, 12);
            }
            
            // Add welcome message
            chatManager.addSystemMessage(
                `<strong>🤖 Neural Network Generator - Interactive Chat</strong><br>` +
                `<br>Session ID: ${sessionId}` +
                `<br>Thread ID: ${data.thread_id}` +
                `<br><br>Type <code>help</code> for usage examples.`
            );
        };
        
        // Processing started
        api.onProcessingStarted = (data) => {
            console.log('Processing message:', data.message);
            chatManager.updateProcessingStatus('Processing...');
        };
        
        // Processing complete - response from bridge
        api.onProcessingComplete = (data) => {
            if (data.response) {
                chatManager.addAssistantMessage(data.response);
            }
            if (data.state) {
                stateManager.updateState(data.state);
                
                // Update graph based on dialog_state
                // Filter out non-string entries (like empty arrays)
                if (data.state.dialog_state && data.state.dialog_state.length > 0) {
                    const validStack = data.state.dialog_state.filter(
                        item => typeof item === 'string' && item.length > 0
                    );
                    if (validStack.length > 0) {
                        const activeNode = validStack[validStack.length - 1];
                        graphManager.setActiveNode(activeNode);
                    } else {
                        // No valid entries, default to master triage router
                        graphManager.setActiveNode('master_triage_router');
                    }
                }
            }
            chatManager.setProcessing(false);
        };

        // Processing error
        api.onProcessingError = (data) => {
            console.error('Processing error:', data);
            chatManager.removeThinkingIndicator();
            chatManager.addErrorMessage(data.error || 'An error occurred while processing your message');
            chatManager.setProcessing(false);
        };
        
        // Status received
        api.onStatusReceived = (data) => {
            console.log('Status received:', data);
            stateManager.updateState(data);
            chatManager.showStatus();
        };
    }

    async checkBackendHealth() {
        // No need for health check anymore - bridge will send 'initialized' message
        // Just set initial status
        const statusBadge = document.getElementById('connection-status');
        statusBadge.textContent = 'Initializing...';
        statusBadge.classList.remove('connected', 'error');
        
        console.log('Waiting for bridge initialization...');
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new App();
});
