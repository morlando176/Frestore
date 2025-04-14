/**
 * Handles client-side logic for image preview, upload,
 * and displaying results from the U-Net analysis endpoint.
 */

// --- Get references to HTML elements ---
const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const analyzeButton = document.getElementById('analyzeButton');
const resultDisplay = document.getElementById('resultDisplay');
const loadingIndicator = document.getElementById('loadingIndicator');

// --- State variable ---
let selectedFile = null;

// --- Event Listener for File Input Change ---
imageUpload.addEventListener('change', (event) => {
    // Get the selected file
    const files = event.target.files;
    if (files && files.length > 0) {
        selectedFile = files[0];

        // Basic file type check (optional, server validates too)
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
        if (!allowedTypes.includes(selectedFile.type)) {
            displayError('Invalid file type. Please select a PNG, JPG, or JPEG image.');
            resetUI();
            return;
        }

        // Use FileReader to display a preview
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block'; // Show preview
        }
        reader.readAsDataURL(selectedFile); // Read file as Data URL

        analyzeButton.disabled = false; // Enable the analyze button
        clearResult(); // Clear previous results/errors

    } else {
        // No file selected or selection cleared
        resetUI();
    }
});

// --- Event Listener for Analyze Button Click ---
analyzeButton.addEventListener('click', async () => {
    if (!selectedFile) {
        displayError('Please select an image file first.');
        return;
    }

    // --- UI updates for processing ---
    analyzeButton.disabled = true; // Disable button during processing
    loadingIndicator.style.display = 'block'; // Show loading text
    clearResult();

    // --- Prepare data for sending ---
    const formData = new FormData();
    // 'image' key must match the key expected by the Flask backend (request.files['image'])
    formData.append('image', selectedFile);

    // --- Send data to Flask backend ---
    try {
        console.log("Sending image to /analyze endpoint..."); // Debug log
        const response = await fetch('/analyze', { // Ensure URL matches your Flask route
            method: 'POST',
            body: formData,
            // Headers are usually set automatically by browser for FormData
        });

        console.log("Received response from server:", response.status); // Debug log

        // Check if the request was successful
        if (!response.ok) {
            // Try to get error message from server response body
            let errorMsg = `Server Error: ${response.status} ${response.statusText}`;
            try {
                 const errorData = await response.json();
                 // Use server's error message if available
                 errorMsg = `Error: ${errorData.error || response.statusText}`;
            } catch (e) {
                // If response body isn't JSON or empty, use the status text
                console.log("Could not parse error response as JSON.");
            }
            // Throw an error to be caught by the catch block
            throw new Error(errorMsg);
        }

        // Parse the successful JSON response
        const data = await response.json();
        console.log("Parsed JSON response:", data); // Debug log

        // --- Display the result ---
        if (data.coverage_pct !== undefined) {
            resultDisplay.textContent = `Calculated Coverage: ${data.coverage_pct.toFixed(2)}%`;
            resultDisplay.classList.remove('error');
        } else if (data.error) {
            // Handle errors reported by the server in the JSON response
             displayError(`Analysis Error: ${data.error}`);
        } else {
            // Handle unexpected successful response format
             displayError('Received an unexpected response from the server.');
        }

    } catch (error) {
        // Handle network errors or errors thrown from response check
        console.error('Fetch Error:', error);
        displayError(`Network or processing error: ${error.message}`);
    } finally {
        // --- UI cleanup after processing ---
        loadingIndicator.style.display = 'none'; // Hide loading text
        analyzeButton.disabled = false; // Re-enable button
        // Optional: Clear the file input after analysis?
        // imageUpload.value = null;
        // resetUI();
    }
});

// --- Helper Functions ---
function displayError(message) {
    resultDisplay.textContent = message;
    resultDisplay.classList.add('error');
}

function clearResult() {
     resultDisplay.textContent = '';
     resultDisplay.classList.remove('error');
}

function resetUI() {
    imagePreview.style.display = 'none';
    imagePreview.src = '#';
    analyzeButton.disabled = true;
    selectedFile = null;
    clearResult();
    imageUpload.value = null; // Clear the file input selection
}

// --- Initial setup ---
// Disable button initially
analyzeButton.disabled = true;

console.log("Analyzer JS loaded."); // Debug log

