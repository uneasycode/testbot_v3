// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    // Handle response type selection in the edit form
    const responseTypeRadios = document.querySelectorAll('input[name="type"]');
    const responseContents = document.querySelectorAll('.response-content');
    
    function updateResponseContentVisibility() {
        const selectedType = document.querySelector('input[name="type"]:checked').value;
        
        responseContents.forEach(content => {
            content.style.display = 'none';
        });
        
        document.getElementById(`${selectedType}-content`).style.display = 'block';
    }
    
    if (responseTypeRadios.length > 0) {
        responseTypeRadios.forEach(radio => {
            radio.addEventListener('change', updateResponseContentVisibility);
        });
        
        // Initialize display
        updateResponseContentVisibility();
    }
    
    // Handle media file upload (images)
    const uploadImageForm = document.getElementById('upload-image-form');
    if (uploadImageForm) {
        uploadImageForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const statusElement = document.getElementById('image-upload-status');
            
            statusElement.innerHTML = '<div class="alert alert-info">Uploading...</div>';
            
            fetch('/upload/media', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    statusElement.innerHTML = `<div class="alert alert-success">${data.message}</div>`;
                    
                    // Update the image select dropdown
                    const imageSelect = document.getElementById('content-image');
                    if (imageSelect) {
                        const option = document.createElement('option');
                        option.value = data.filename;
                        option.text = data.filename;
                        option.selected = true;
                        imageSelect.appendChild(option);
                    }
                    
                    // Reset the form
                    uploadImageForm.reset();
                } else {
                    statusElement.innerHTML = `<div class="alert alert-danger">${data.error || 'Upload failed'}</div>`;
                }
            })
            .catch(error => {
                statusElement.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            });
        });
    }
    
    // Handle media file upload (audio)
    const uploadAudioForm = document.getElementById('upload-audio-form');
    if (uploadAudioForm) {
        uploadAudioForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const statusElement = document.getElementById('audio-upload-status');
            
            statusElement.innerHTML = '<div class="alert alert-info">Uploading...</div>';
            
            fetch('/upload/media', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    statusElement.innerHTML = `<div class="alert alert-success">${data.message}</div>`;
                    
                    // Update the audio select dropdown
                    const audioSelect = document.getElementById('content-audio');
                    if (audioSelect) {
                        const option = document.createElement('option');
                        option.value = data.filename;
                        option.text = data.filename;
                        option.selected = true;
                        audioSelect.appendChild(option);
                    }
                    
                    // Reset the form
                    uploadAudioForm.reset();
                } else {
                    statusElement.innerHTML = `<div class="alert alert-danger">${data.error || 'Upload failed'}</div>`;
                }
            })
            .catch(error => {
                statusElement.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            });
        });
    }
    
    // Handle media deletion
    const deleteMediaButtons = document.querySelectorAll('.delete-media');
    if (deleteMediaButtons.length > 0) {
        deleteMediaButtons.forEach(button => {
            button.addEventListener('click', function() {
                const mediaType = this.dataset.type;
                const filename = this.dataset.filename;
                
                if (confirm(`Are you sure you want to delete ${filename}?`)) {
                    fetch(`/api/media/${mediaType}/${filename}`, {
                        method: 'DELETE'
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            // Remove the media item from the UI
                            this.parentElement.parentElement.remove();
                        } else {
                            alert(`Error: ${data.error || 'Failed to delete file'}`);
                        }
                    })
                    .catch(error => {
                        alert(`Error: ${error.message}`);
                    });
                }
            });
        });
    }
    
    // Confirmation for response deletion
    window.confirmDelete = function(keyword) {
        const confirmationModal = new bootstrap.Modal(document.getElementById('confirmationModal'));
        const confirmButton = document.getElementById('confirm-delete-btn');
        
        confirmButton.onclick = function() {
            const form = document.getElementById('delete-form');
            form.action = `/delete/${keyword}`;
            form.method = 'POST';
            form.submit();
            confirmationModal.hide();
        };
        
        confirmationModal.show();
    };
    
    // Check bot status periodically
    function checkBotStatus() {
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                const statusElement = document.getElementById('bot-status');
                if (statusElement) {
                    if (data.status === 'running') {
                        statusElement.className = 'badge bg-success me-2';
                        statusElement.innerHTML = '<i class="fas fa-circle"></i> Running';
                    } else {
                        statusElement.className = 'badge bg-danger me-2';
                        statusElement.innerHTML = '<i class="fas fa-circle"></i> Stopped';
                    }
                }
            })
            .catch(error => {
                console.error('Error checking bot status:', error);
            });
    }
    
    // Check status initially
    checkBotStatus();
    
    // Then check every 30 seconds
    setInterval(checkBotStatus, 30000);
});