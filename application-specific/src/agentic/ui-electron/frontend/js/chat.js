/**
 * Chat Module
 * Handles chat interface and messages (Claude-like style)
 */

class ChatManager {
    constructor() {
        this.messagesContainer = document.getElementById('chat-messages');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('btn-send');
        
        this.isProcessing = false;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Send message on button click
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Send message on Enter key (without Shift)
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey && !this.isProcessing) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = this.messageInput.scrollHeight + 'px';
        });
    }

    async sendMessage(overrideMessage = null) {
        const message = (overrideMessage !== null && overrideMessage !== undefined)
            ? String(overrideMessage).trim()
            : this.messageInput.value.trim();
        
        if (!message || this.isProcessing) {
            return;
        }
        
        // Clear input and reset height (only for manual typing)
        if (overrideMessage === null || overrideMessage === undefined) {
            this.messageInput.value = '';
            this.messageInput.style.height = 'auto';
        }
        
        // Add user message
        this.addUserMessage(message);
        
        // Check for special commands (like terminal chatbot)
        const command = message.toLowerCase();
        
        if (command === 'help') {
            this.showHelp();
            return;
        }
        
        if (command === 'status') {
            this.showStatus();
            return;
        }
        
        if (command === 'clear') {
            this.clearChat();
            return;
        }
        
        // Update UI state
        this.setProcessing(true);
        
        // Add thinking indicator
        this.addThinkingIndicator();
        
        try {
            // Send to backend
            const result = await api.sendMessage(message);
            
            // Response will come via WebSocket
            // So we don't need to handle it here
            
        } catch (error) {
            this.removeThinkingIndicator();
            this.addErrorMessage(error.message || 'Failed to send message');
            this.setProcessing(false);
        }
    }

    addUserMessage(content) {
        const messageWrapper = document.createElement('div');
        messageWrapper.className = 'message-wrapper user-message';
        messageWrapper.innerHTML = `
            <div class="message-content">
                <div class="message-header">
                    <span class="user-label">You</span>
                </div>
                <div class="message-text">${this.formatContent(content)}</div>
            </div>
        `;
        
        this.messagesContainer.appendChild(messageWrapper);
        this.scrollToBottom();
    }

    addAssistantMessage(content) {
        // Remove thinking indicator if present
        this.removeThinkingIndicator();
        
        const messageWrapper = document.createElement('div');
        messageWrapper.className = 'message-wrapper assistant-message';
        messageWrapper.innerHTML = `
            <div class="message-content">
                <div class="message-header">
                    <span class="assistant-label">Neural Network Generator</span>
                </div>
                <div class="message-text">${this.formatContent(content)}</div>
            </div>
        `;
        
        this.messagesContainer.appendChild(messageWrapper);
        this.scrollToBottom();
        
        // Set processing to false
        this.setProcessing(false);
    }

    addThinkingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'thinking-indicator';
        indicator.id = 'thinking-indicator';
        indicator.innerHTML = `
            <span>Thinking</span>
            <div class="thinking-dots">
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
            </div>
        `;
        
        this.messagesContainer.appendChild(indicator);
        this.scrollToBottom();
    }

    removeThinkingIndicator() {
        const indicator = document.getElementById('thinking-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    addToolCall(toolCall) {
        const toolDiv = document.createElement('div');
        toolDiv.className = 'tool-call';
        toolDiv.innerHTML = `
            <strong>Tool: ${this.escapeHtml(toolCall.name)}</strong>
            <pre>${this.escapeHtml(JSON.stringify(toolCall.args, null, 2))}</pre>
        `;
        
        this.messagesContainer.appendChild(toolDiv);
        this.scrollToBottom();
    }

    addErrorMessage(content) {
        this.removeThinkingIndicator();
        
        const messageWrapper = document.createElement('div');
        messageWrapper.className = 'message-wrapper assistant-message';
        messageWrapper.innerHTML = `
            <div class="message-content">
                <div class="message-header">
                    <span class="assistant-label">System</span>
                </div>
                <div class="message-text" style="background: var(--error); color: white;">
                    <strong>Error:</strong> ${this.escapeHtml(content)}
                </div>
            </div>
        `;
        
        this.messagesContainer.appendChild(messageWrapper);
        this.scrollToBottom();
    }

    formatContent(content) {
        // Escape HTML
        let formatted = this.escapeHtml(content);
        
        // Convert newlines to <br>
        formatted = formatted.replace(/\n/g, '<br>');
        
        // Convert **bold**
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Convert *italic*
        formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Convert `code`
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        return formatted;
    }

    setProcessing(processing) {
        this.isProcessing = processing;
        this.sendButton.disabled = processing;
        this.messageInput.disabled = processing;
        
        const statusMessage = document.getElementById('status-message');
        const progressBar = document.getElementById('progress-bar');
        
        if (processing) {
            if (statusMessage) statusMessage.textContent = 'Processing...';
            if (progressBar) progressBar.style.display = 'block';
        } else {
            if (statusMessage) statusMessage.textContent = 'Ready';
            if (progressBar) progressBar.style.display = 'none';
        }
    }
    
    updateProcessingStatus(status) {
        const statusMessage = document.getElementById('status-message');
        if (statusMessage) {
            statusMessage.textContent = status;
        }
    }

    getCurrentTime() {
        const now = new Date();
        return now.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            second: '2-digit'
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    scrollToBottom() {
        setTimeout(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }, 100);
    }

    showHelp() {
        const helpContent = `
<strong>HELP - Neural Network Generator</strong>
<br><br>
<strong>EXAMPLE PROMPTS:</strong>
<br><br>
<em>Getting Started:</em>
<br>• "Build a classifier for MNIST dataset with 95% accuracy"
<br>• "Create an image classifier using ResNet18"
<br>• "I want to train a CNN on CIFAR10"
<br><br>
<em>Architecture:</em>
<br>• "Use ResNet50 pretrained model"
<br>• "Design a custom CNN with 3 conv layers"
<br>• "Add dropout for regularization"
<br><br>
<em>Configuration:</em>
<br>• "Use Adam optimizer with learning rate 0.001"
<br>• "Train for 50 epochs with batch size 32"
<br>• "Set device to cuda"
<br><br>
<em>Training:</em>
<br>• "Start training"
<br>• "Show training status"
<br>• "Open TensorBoard"
<br><br>
<em>Optimization:</em>
<br>• "My model isn't reaching target, help improve it"
<br>• "Optimize the architecture"
<br><br>
<strong>COMMANDS:</strong>
<br>• <code>status</code> - Show current workflow status
<br>• <code>help</code> - Show this help message
<br>• <code>clear</code> - Clear chat history
        `;
        
        this.addSystemMessage(helpContent);
    }

    showStatus() {
        if (!window.stateManager || !window.stateManager.state) {
            this.addSystemMessage('No status available yet. Start a conversation first!');
            return;
        }
        
        const state = window.stateManager.state;
        
        let statusContent = '<strong>CURRENT STATUS</strong><br><br>';
        
        // Project
        const project = state.project_name || 'Not created';
        statusContent += `<strong>Project:</strong> ${this.escapeHtml(project)}<br>`;
        
        // Stage
        const stage = state.current_stage || 'Initial';
        statusContent += `<strong>Current Stage:</strong> ${this.escapeHtml(stage)}<br>`;
        
        // Completed
        const completed = state.completed_stages || [];
        if (completed.length > 0) {
            statusContent += `<strong>Completed:</strong> ${completed.join(', ')}<br>`;
        }
        
        // Architecture
        const arch = state.architecture_summary;
        if (arch) {
            statusContent += `<br><strong>Architecture:</strong> ${this.escapeHtml(arch.type || 'unknown')}<br>`;
            if (arch.base_model) {
                statusContent += `   Model: ${this.escapeHtml(arch.base_model)}<br>`;
            }
            if (arch.parameters) {
                statusContent += `   Parameters: ${arch.parameters.toLocaleString()}<br>`;
            }
        }
        
        // Training
        const runs = state.training_runs || [];
        if (runs.length > 0) {
            const latest = runs[runs.length - 1];
            const acc = (latest.final_accuracy || 0) * 100;
            statusContent += `<br><strong>Latest Training:</strong> ${acc.toFixed(1)}% accuracy<br>`;
        }
        
        this.addSystemMessage(statusContent);
    }

    clearChat() {
        this.messagesContainer.innerHTML = '';
        this.addSystemMessage('Chat cleared. Type <code>help</code> for usage examples.');
    }

    addSystemMessage(content) {
        const messageWrapper = document.createElement('div');
        messageWrapper.className = 'message-wrapper assistant-message';
        messageWrapper.innerHTML = `
            <div class="message-content">
                <div class="message-header">
                    <span class="assistant-label">System</span>
                </div>
                <div class="message-text">${content}</div>
            </div>
        `;
        
        this.messagesContainer.appendChild(messageWrapper);
        this.scrollToBottom();
    }
}

// Initialize chat manager
const chatManager = new ChatManager();
