/**
 * State Module
 * Handles state monitoring and display
 */

class StateManager {
    constructor() {
        this.currentState = {};
        this.initializeEventListeners();
        this.initializeUploadButton();
    }

    initializeUploadButton() {
        // Ensure upload button is disabled on startup
        const uploadBtn = document.getElementById('btn-upload');
        if (uploadBtn) {
            uploadBtn.disabled = true;
            uploadBtn.title = 'Upload is enabled when testing a trained model';
        }
    }

    // Helper to safely set text content
    safeSetText(elementId, text) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = text;
        }
    }

    initializeEventListeners() {
        const refreshBtn = document.getElementById('btn-refresh-state');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.refreshState();
            });
        }
    }

    async refreshState() {
        try {
            const result = await api.getState();
            if (result.success) {
                this.updateState(result.state);
            }
        } catch (error) {
            console.error('Failed to refresh state:', error);
        }
    }

    updateState(state) {
        this.currentState = state;
        
        // Update current assistant
        const dialogStack = state.dialog_state || [];
        // Filter out non-string entries (like empty arrays)
        const validStack = dialogStack.filter(item => typeof item === 'string' && item.length > 0);
        
        if (validStack.length > 0) {
            const current = validStack[validStack.length - 1];
            this.safeSetText('current-assistant', this.formatAssistantName(current));
        } else {
            this.safeSetText('current-assistant', 'Master Triage Router');
        }
        
        // Update dialog stack
        if (validStack.length > 0) {
            const stackText = validStack
                .map(s => this.formatAssistantName(s))
                .join(' → ');
            this.safeSetText('dialog-stack', stackText);
        } else {
            this.safeSetText('dialog-stack', 'Empty');
        }
        
        // Update project info
        const projectInfo = document.getElementById('project-info');
        if (projectInfo) {
            if (state.project_name) {
                let projectText = `<b>Name:</b> ${state.project_name}`;
                if (state.project_path) {
                    projectText += `<br><b>Path:</b> ${state.project_path}`;
                }
                projectInfo.innerHTML = projectText;
            } else {
                projectInfo.textContent = 'No project loaded';
            }
        }
        
        // Update architecture info
        const archInfo = document.getElementById('architecture-info');
        if (archInfo) {
            if (state.architecture_name || state.architecture_path) {
                let archText = '';
                if (state.architecture_name) {
                    archText += `<b>Name:</b> ${state.architecture_name}<br>`;
                }
                if (state.architecture_path) {
                    archText += `<b>Path:</b> ${state.architecture_path}`;
                }
                archInfo.innerHTML = archText;
            } else {
                archInfo.textContent = 'No architecture defined';
            }
        }
        
        // Update configuration
        const configInfo = document.getElementById('config-info');
        if (configInfo) {
            if (state.config_path) {
                configInfo.innerHTML = `<b>Path:</b> ${state.config_path}`;
            } else {
                configInfo.textContent = 'No configuration set';
            }
        }
        
        // Update generated code
        const codeInfo = document.getElementById('code-info');
        if (codeInfo) {
            if (state.generated_code_paths) {
                let codeText = '<b>Generated Files:</b><br>';
                for (const [key, path] of Object.entries(state.generated_code_paths)) {
                    codeText += `• ${key}: ${path}<br>`;
                }
                codeInfo.innerHTML = codeText;
            } else {
                codeInfo.textContent = 'No code generated';
            }
        }
        
        // Update training status
        const trainingInfo = document.getElementById('training-info');
        if (trainingInfo) {
            if (state.training_status) {
                let trainingText = `<b>Status:</b> ${state.training_status}`;
                if (state.training_results) {
                    trainingText += `<br><b>Results:</b> ${state.training_results}`;
                }
                trainingInfo.innerHTML = trainingText;
            } else {
                trainingInfo.textContent = 'Not started';
            }
        }
        
        // Update RTL synthesis status
        const rtlInfo = document.getElementById('rtl-info');
        if (rtlInfo) {
            const rtlConfig = state.rtl_synthesis_config || {};
            if (state.rtl_synthesis_complete) {
                let rtlText = '<b>Status:</b> ✅ Complete';
                if (state.rtl_output_path) {
                    rtlText += `<br><b>Output:</b> ${state.rtl_output_path}`;
                }
                rtlInfo.innerHTML = rtlText;
            } else if (state.awaiting_rtl_synthesis || state.awaiting_hls_config || state.awaiting_hls_verify || state.awaiting_rtl_build) {
                let rtlText = '<b>Status:</b> 🔄 In Progress';
                if (state.awaiting_hls_config) rtlText += ' (HLS Config)';
                else if (state.awaiting_hls_verify) rtlText += ' (Verifying)';
                else if (state.awaiting_rtl_build) rtlText += ' (Synthesizing)';
                if (rtlConfig.reuse_factor) rtlText += `<br><b>Reuse:</b> ${rtlConfig.reuse_factor}`;
                if (rtlConfig.io_type) rtlText += `<br><b>IO:</b> ${rtlConfig.io_type}`;
                if (rtlConfig.precision) rtlText += `<br><b>Precision:</b> ${rtlConfig.precision}`;
                rtlInfo.innerHTML = rtlText;
            } else if (Object.keys(rtlConfig).length > 0) {
                let rtlText = '<b>Status:</b> Configured';
                if (rtlConfig.reuse_factor) rtlText += `<br><b>Reuse:</b> ${rtlConfig.reuse_factor}`;
                if (rtlConfig.io_type) rtlText += `<br><b>IO:</b> ${rtlConfig.io_type}`;
                rtlInfo.innerHTML = rtlText;
            } else {
                rtlInfo.textContent = 'Not started';
            }
        }

        // Enable upload when at post-training menu or explicitly awaiting image
        const uploadBtn = document.getElementById('btn-upload');
        if (uploadBtn) {
            // Enable if awaiting new design choice (post-training menu) OR awaiting test image
            const enabled = !!(state.awaiting_new_design_choice || state.awaiting_test_image);
            uploadBtn.disabled = !enabled;
            uploadBtn.title = enabled
                ? 'Upload Image to Test Model'
                : 'Upload is enabled after training completes';
        }
        
        // Update graph visualization - prefer current_node over dialog_state
        const currentNode = state.current_node;
        if (currentNode && typeof currentNode === 'string') {
            graphManager.setActiveNode(currentNode);
        } else if (validStack.length > 0) {
            graphManager.setActiveNode(validStack[validStack.length - 1]);
        } else {
            graphManager.setActiveNode('master_triage_router');
        }
    }

    formatAssistantName(name) {
        // Handle non-string inputs safely
        if (!name || typeof name !== 'string') {
            return 'Unknown';
        }
        
        return name
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    clearState() {
        this.currentState = {};
        this.safeSetText('current-assistant', 'None');
        this.safeSetText('dialog-stack', 'Empty');
        this.safeSetText('project-info', 'No project loaded');
        this.safeSetText('architecture-info', 'No architecture defined');
        this.safeSetText('config-info', 'No configuration set');
        this.safeSetText('code-info', 'No code generated');
        this.safeSetText('training-info', 'Not started');
    }
}

// Initialize state manager
const stateManager = new StateManager();
