document.addEventListener('DOMContentLoaded', function() {
    // Tab switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            
            // Remove active class from all buttons and contents
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to selected button and content
            btn.classList.add('active');
            document.getElementById(`${tab}-tab`).classList.add('active');
        });
    });
    
    // Temperature slider display
    const temperatureSlider = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperature-value');
    
    temperatureSlider.addEventListener('input', () => {
        temperatureValue.textContent = temperatureSlider.value;
    });
    
    // Check model status on load
    checkModelStatus();
    
    // Form submissions
    const downloadForm = document.getElementById('download-form');
    const uploadForm = document.getElementById('upload-form');
    const inferenceForm = document.getElementById('inference-form');
    
    // Model download
    downloadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const modelName = document.getElementById('model-name').value.trim();
        
        if (!modelName) {
            addLogEntry('Please enter a model name or path', 'error');
            return;
        }
        
        // Show progress container
        const progressContainer = document.getElementById('download-progress-container');
        progressContainer.style.display = 'block';
        
        // Reset progress bar
        const progressBar = document.getElementById('download-progress-bar');
        const progressText = document.getElementById('download-progress-text');
        progressBar.style.width = '0%';
        progressText.textContent = 'Starting download...';
        
        addLogEntry(`Starting download of model: ${modelName}...`);
        
        try {
            const formData = new FormData();
            formData.append('model_name', modelName);
            formData.append('background', true);
            
            const response = await fetch('/download-model', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                addLogEntry(result.message);
                // Start polling for progress
                startProgressPolling();
            } else {
                addLogEntry(result.message, 'error');
                progressContainer.style.display = 'none';
            }
        } catch (error) {
            addLogEntry(`Error: ${error.message}`, 'error');
            progressContainer.style.display = 'none';
        }
    });
    
    // Progress polling for model download
    let progressInterval = null;
    
    function startProgressPolling() {
        if (progressInterval) {
            clearInterval(progressInterval);
        }
        
        progressInterval = setInterval(checkDownloadProgress, 1000);
    }
    
    async function checkDownloadProgress() {
        try {
            const response = await fetch('/download-progress');
            const data = await response.json();
            
            const progressBar = document.getElementById('download-progress-bar');
            const progressText = document.getElementById('download-progress-text');
            const progressContainer = document.getElementById('download-progress-container');
            
            // Update progress bar
            progressBar.style.width = `${data.progress}%`;
            progressText.textContent = data.status;
            
            // Handle completion or error
            if (data.progress === 100 || data.error) {
                clearInterval(progressInterval);
                
                if (data.error) {
                    addLogEntry(`Download error: ${data.error}`, 'error');
                } else {
                    addLogEntry(`Download of ${data.model_name} completed successfully`, 'success');
                    checkModelStatus();
                    
                    // Hide progress bar after a delay
                    setTimeout(() => {
                        progressContainer.style.display = 'none';
                    }, 3000);
                }
            }
        } catch (error) {
            addLogEntry(`Error checking download progress: ${error.message}`, 'error');
            clearInterval(progressInterval);
        }
    }
    
    // Model upload
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById('model-file');
        
        if (!fileInput.files || fileInput.files.length === 0) {
            addLogEntry('Please select a file to upload', 'error');
            return;
        }
        
        const file = fileInput.files[0];
        
        if (!file.name.endsWith('.gguf')) {
            addLogEntry('Only .gguf files are supported', 'error');
            return;
        }
        
        addLogEntry(`Uploading model: ${file.name}...`);
        toggleLoading(true);
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/upload-model', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                addLogEntry(result.message, 'success');
                checkModelStatus();
            } else {
                addLogEntry(result.message, 'error');
            }
        } catch (error) {
            addLogEntry(`Error: ${error.message}`, 'error');
        } finally {
            toggleLoading(false);
        }
    });
    
    // Inference
    inferenceForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const prompt = document.getElementById('prompt').value.trim();
        const temperature = document.getElementById('temperature').value;
        const nPredict = document.getElementById('n-predict').value;
        const conversation = document.getElementById('conversation').checked;
        
        if (!prompt) {
            addLogEntry('Please enter a prompt', 'error');
            return;
        }
        
        addLogEntry(`Generating response with temperature ${temperature} and max tokens ${nPredict}...`);
        toggleLoading(true);
        
        // Clear previous content and show loading in the output area
        const outputElement = document.getElementById('output');
        outputElement.innerHTML = '<p class="loading">Generating text, please wait...</p>';
        
        try {
            const formData = new FormData();
            formData.append('prompt', prompt);
            formData.append('temperature', temperature);
            formData.append('n_predict', nPredict);
            formData.append('conversation', conversation);
            
            const response = await fetch('/generate', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                displayOutput(result.output);
                addLogEntry('Text generation completed', 'success');
            } else {
                displayError(result.message);
                addLogEntry(result.message, 'error');
            }
        } catch (error) {
            displayError(`Error: ${error.message}`);
            addLogEntry(`Error: ${error.message}`, 'error');
        } finally {
            toggleLoading(false);
        }
    });
    
    // Popular model links
    document.querySelectorAll('.model-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const model = this.dataset.model;
            document.getElementById('model-name').value = model;
            addLogEntry(`Selected model: ${model}`);
        });
    });
});

// Function to check model status
async function checkModelStatus() {
    try {
        const response = await fetch('/model-status');
        const result = await response.json();
        
        const statusText = document.getElementById('model-status-text');
        
        if (result.model_loaded && result.current_model) {
            const modelName = result.current_model.split('/').pop();
            statusText.textContent = `Model loaded: ${modelName}`;
            statusText.className = 'success';
            addLogEntry(`Model status: ${modelName} is loaded and ready`, 'success');
        } else {
            statusText.textContent = `No model loaded`;
            statusText.className = '';
        }
    } catch (error) {
        addLogEntry(`Error checking model status: ${error.message}`, 'error');
    }
}

// Function to add log entry
function addLogEntry(message, type = '') {
    const logs = document.getElementById('logs');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    
    const timestamp = new Date().toLocaleTimeString();
    entry.textContent = `[${timestamp}] ${message}`;
    
    logs.appendChild(entry);
    logs.scrollTop = logs.scrollHeight;
}

// Function to display output
function displayOutput(output) {
    const outputElement = document.getElementById('output');
    outputElement.innerHTML = '';
    
    const outputText = document.createElement('pre');
    outputText.textContent = output;
    
    outputElement.appendChild(outputText);
}

// Function to display error in the output area
function displayError(message) {
    const outputElement = document.getElementById('output');
    outputElement.innerHTML = '';
    
    const errorContainer = document.createElement('div');
    errorContainer.className = 'error-message';
    
    const errorIcon = document.createElement('i');
    errorIcon.className = 'fas fa-exclamation-triangle';
    errorContainer.appendChild(errorIcon);
    
    const errorText = document.createElement('p');
    errorText.textContent = message;
    errorContainer.appendChild(errorText);
    
    if (message.includes("compiler tools")) {
        const helpLink = document.createElement('a');
        helpLink.href = "https://visualstudio.microsoft.com/downloads/";
        helpLink.textContent = "Download Visual Studio Build Tools";
        helpLink.target = "_blank";
        helpLink.className = 'help-link';
        errorContainer.appendChild(helpLink);
    }
    
    outputElement.appendChild(errorContainer);
}

// Function to toggle loading state
function toggleLoading(isLoading) {
    const buttons = document.querySelectorAll('.btn');
    
    buttons.forEach(btn => {
        btn.disabled = isLoading;
    });
    
    if (isLoading) {
        document.body.style.cursor = 'wait';
    } else {
        document.body.style.cursor = 'default';
    }
} 