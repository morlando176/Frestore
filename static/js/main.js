/**
 * Main JavaScript for Foreskin Tracker
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips if any
    initTooltips();
    
    // Add event listeners for image deletion if on gallery page
    setupImageDeletion();
    
    // Setup image modal viewing if on gallery page
    setupImageViewer();
    
    // Setup form validation if on upload page
    setupFormValidation();
});

/**
 * Initialize tooltips
 */
function initTooltips() {
    const tooltips = document.querySelectorAll('[data-tooltip]');
    
    tooltips.forEach(tooltip => {
        tooltip.addEventListener('mouseenter', function() {
            const tooltipText = this.getAttribute('data-tooltip');
            
            const tooltipEl = document.createElement('div');
            tooltipEl.className = 'tooltip';
            tooltipEl.textContent = tooltipText;
            
            document.body.appendChild(tooltipEl);
            
            const rect = this.getBoundingClientRect();
            tooltipEl.style.top = rect.bottom + 10 + 'px';
            tooltipEl.style.left = rect.left + (rect.width / 2) - (tooltipEl.offsetWidth / 2) + 'px';
            
            setTimeout(() => {
                tooltipEl.classList.add('visible');
            }, 10);
        });
        
        tooltip.addEventListener('mouseleave', function() {
            const tooltipEl = document.querySelector('.tooltip');
            if (tooltipEl) {
                tooltipEl.classList.remove('visible');
                
                tooltipEl.addEventListener('transitionend', function() {
                    if (tooltipEl.parentNode) {
                        tooltipEl.parentNode.removeChild(tooltipEl);
                    }
                });
            }
        });
    });
}

/**
 * Setup image deletion functionality
 */
function setupImageDeletion() {
    const deleteButtons = document.querySelectorAll('.btn-delete');
    
    if (deleteButtons.length === 0) return;
    
    deleteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            const galleryItem = this.closest('.gallery-item');
            const imagePath = galleryItem.querySelector('img').getAttribute('src');
            const imageName = imagePath.split('/').pop();
            
            if (confirm(`Are you sure you want to delete this image: ${imageName}?`)) {
                // Send delete request to server
                fetch(`/delete-image/${imageName}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Requested-With': 'XMLHttpRequest'
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Remove the gallery item from the DOM
                        galleryItem.remove();
                        
                        // Show success message
                        showNotification('Image deleted successfully', 'success');
                        
                        // If no more images, reload the page to show empty state
                        if (document.querySelectorAll('.gallery-item').length === 0) {
                            window.location.reload();
                        }
                    } else {
                        showNotification('Error deleting image: ' + data.message, 'error');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    showNotification('Error deleting image', 'error');
                });
            }
        });
    });
}

/**
 * Setup image viewer modal
 */
function setupImageViewer() {
    const viewButtons = document.querySelectorAll('.btn-view');
    
    if (viewButtons.length === 0) return;
    
    // Create modal elements if they don't exist
    if (!document.getElementById('image-modal')) {
        const modal = document.createElement('div');
        modal.id = 'image-modal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <span class="close-modal">&times;</span>
                <img id="modal-image" src="" alt="Full size image">
                <div class="modal-caption"></div>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Close modal when clicking the X
        document.querySelector('.close-modal').addEventListener('click', function() {
            document.getElementById('image-modal').style.display = 'none';
        });
        
        // Close modal when clicking outside the image
        window.addEventListener('click', function(e) {
            if (e.target === document.getElementById('image-modal')) {
                document.getElementById('image-modal').style.display = 'none';
            }
        });
    }
    
    // Add click event to view buttons
    viewButtons.forEach(button => {
        button.addEventListener('click', function() {
            const galleryItem = this.closest('.gallery-item');
            const imageSrc = galleryItem.querySelector('img').getAttribute('src');
            const imageDate = galleryItem.querySelector('.image-date').textContent;
            
            document.getElementById('modal-image').src = imageSrc;
            document.querySelector('.modal-caption').textContent = `Date: ${imageDate}`;
            document.getElementById('image-modal').style.display = 'block';
        });
    });
}

/**
 * Setup form validation for upload form
 */
function setupFormValidation() {
    const uploadForm = document.querySelector('.upload-form');
    
    if (!uploadForm) return;
    
    uploadForm.addEventListener('submit', function(e) {
        const fileInput = document.getElementById('file');
        
        if (!fileInput.files.length) {
            e.preventDefault();
            showNotification('Please select a file to upload', 'error');
            return;
        }
        
        const file = fileInput.files[0];
        const fileType = file.type;
        const validTypes = ['image/jpeg', 'image/png', 'image/gif'];
        
        if (!validTypes.includes(fileType)) {
            e.preventDefault();
            showNotification('Please select a valid image file (JPEG, PNG, or GIF)', 'error');
            return;
        }
        
        // File size validation (max 5MB)
        if (file.size > 5 * 1024 * 1024) {
            e.preventDefault();
            showNotification('File size must be less than 5MB', 'error');
            return;
        }
        
        // Show loading state
        const submitBtn = uploadForm.querySelector('button[type="submit"]');
        submitBtn.disabled = true;
        submitBtn.innerHTML = 'Uploading...';
        
        // Add a loading indicator
        const loadingIndicator = document.createElement('div');
        loadingIndicator.className = 'loading-indicator';
        uploadForm.appendChild(loadingIndicator);
    });
}

/**
 * Show notification message
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Add to DOM
    document.body.appendChild(notification);
    
    // Trigger animation
    setTimeout(() => {
        notification.classList.add('visible');
    }, 10);
    
    // Remove after delay
    setTimeout(() => {
        notification.classList.remove('visible');
        
        notification.addEventListener('transitionend', function() {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        });
    }, 5000);
}

/**
 * Add CSS styles for elements created by JS
 */
(function addDynamicStyles() {
    const style = document.createElement('style');
    style.textContent = `
        /* Modal styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            overflow: auto;
        }
        
        .modal-content {
            position: relative;
            margin: 5% auto;
            padding: 20px;
            max-width: 90%;
            max-height: 90vh;
            text-align: center;
        }
        
        .close-modal {
            position: absolute;
            top: -30px;
            right: 0;
            color: #fff;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        
        #modal-image {
            max-width: 100%;
            max-height: 80vh;
            border: 2px solid #fff;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.3);
        }
        
        .modal-caption {
            margin-top: 15px;
            color: #fff;
            font-size: 16px;
        }
        
        /* Tooltip styles */
        .tooltip {
            position: absolute;
            background-color: #333;
            color: #fff;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 14px;
            z-index: 100;
            opacity: 0;
            transition: opacity 0.3s;
            pointer-events: none;
        }
        
        .tooltip:after {
            content: '';
            position: absolute;
            bottom: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: transparent transparent #333 transparent;
        }
        
        .tooltip.visible {
            opacity: 1;
        }
        
        /* Notification styles */
        .notification {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px 20px;
            background-color: #333;
            color: #fff;
            border-radius: 4px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
            z-index: 1000;
            transform: translateY(100px);
            opacity: 0;
            transition: transform 0.3s, opacity 0.3s;
        }
        
        .notification.visible {
            transform: translateY(0);
            opacity: 1;
        }
        
        .notification.success {
            background-color: #28a745;
        }
        
        .notification.error {
            background-color: #dc3545;
        }
        
        .notification.info {
            background-color: #4a90e2;
        }
        
        /* Loading indicator */
        .loading-indicator {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-left: 10px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(style);
})();
