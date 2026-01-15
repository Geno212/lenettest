/**
 * Image Upload Handler
 * Handles image upload for model testing
 */

class UploadManager {
    constructor() {
        this.setupUploadButton();
    }

    setupUploadButton() {
        const uploadButton = document.getElementById('btn-upload');
        const fileInput = document.getElementById('image-upload');
        
        if (!uploadButton || !fileInput) {
            console.error('Upload elements not found');
            return;
        }

        // Click upload button to trigger file input
        uploadButton.addEventListener('click', () => {
            fileInput.click();
        });

        // Handle file selection
        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            // Validate file type
            if (!file.type.startsWith('image/')) {
                chatManager.addErrorMessage('Please select a valid image file.');
                return;
            }

            await this.uploadImage(file);
            
            // Reset file input
            fileInput.value = '';
        });
    }

    async uploadImage(file) {
        try {
            chatManager.updateProcessingStatus('Uploading image...');

            // Read file as data URL
            const reader = new FileReader();
            
            const fileData = await new Promise((resolve, reject) => {
                reader.onload = (e) => resolve(e.target.result);
                reader.onerror = (e) => reject(new Error('Failed to read file'));
                reader.readAsDataURL(file);
            });

            // Send file to main process via IPC
            const result = await window.electronAPI.uploadImage({
                name: file.name,
                type: file.type,
                data: fileData
            });

            if (!result.success) {
                throw new Error(result.error || 'Upload failed');
            }

            // Display uploaded image info
            chatManager.addSystemMessage(
                `📷 **Image Uploaded**<br>` +
                `File: ${result.filename}<br>` +
                `Path: <code>${result.file_path}</code>`
            );

            // Send the file path as a message to the graph
            chatManager.sendMessage(result.file_path);

        } catch (error) {
            console.error('Upload error:', error);
            chatManager.addErrorMessage(`Upload failed: ${error.message}`);
            chatManager.setProcessing(false);
        }
    }
}

// Initialize upload manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.uploadManager = new UploadManager();
});
