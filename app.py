# -*- coding: utf-8 -*-
"""
Main Flask application file for the Foreskin Restoration Tracker.
Includes routes for the U-Net analyzer component.
"""

import os
import uuid # To generate unique temporary filenames
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    flash, # Example import if you add flashing
    redirect, # Example import if you add redirects
    url_for # Example import if needed elsewhere
)
from werkzeug.utils import secure_filename

# --- Import custom modules ---
# Import the analysis function from unet_predictor.py
# Ensure unet_predictor.py is in the same directory or adjust import path
try:
    from unet_predictor import run_unet_analysis
except ImportError:
    print("❌ ERROR: Could not import 'run_unet_analysis' from 'unet_predictor'.")
    print("   Ensure 'unet_predictor.py' exists and has no import errors itself.")
    # Define a dummy function to allow app to potentially still run for other routes
    def run_unet_analysis(image_path):
        print("WARNING: Using dummy run_unet_analysis function.")
        return None

# --- App Initialization ---
app = Flask(__name__)
# It's good practice to set a secret key for session management, flash messages etc.
# Replace 'your secret key' with a real random secret key.
app.config['SECRET_KEY'] = '96000286064aebe4ba8d643acb94e4e050e2eec4bba30c06'
# Define allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- Helper Function ---
def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---

# Example route for a potential homepage
@app.route('/')
def home():
    """Renders the homepage."""
    # Replace with your actual homepage template or logic
    return "Welcome to the Foreskin Restoration Tracker! <a href='/analyzer'>Go to Analyzer</a>"

@app.route('/analyzer')
def analyzer_page_route():
    """Renders the HTML page for the U-Net analyzer."""
    print("Rendering analyzer page...")
    return render_template('analyzer_page.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Receives uploaded image via POST request, runs U-Net analysis,
    deletes the temporary image, and returns the coverage result as JSON.
    """
    print("Received request to /analyze") # Debug print

    # --- Basic request validation ---
    if 'image' not in request.files:
        print("No image file part in request")
        return jsonify({'error': 'No image file part in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        print("No image selected")
        return jsonify({'error': 'No selected file'}), 400

    # --- File validation and temporary saving ---
    if file and allowed_file(file.filename):
        # Secure the filename and generate a unique temp name
        filename = secure_filename(file.filename)
        temp_filename = str(uuid.uuid4()) + "_" + filename

        # Define path for temporary storage (using Flask instance folder is safer)
        # Ensure the instance folder exists
        try:
            temp_dir = os.path.join(app.instance_path, 'temp_uploads')
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, temp_filename)
        except OSError as e:
             print(f"❌ ERROR creating temporary directory {temp_dir}: {e}")
             return jsonify({'error': 'Server configuration error (cannot create temp dir)'}), 500


        print(f"Saving temporary file to: {temp_image_path}")
        try:
            # Save the uploaded file temporarily
            file.save(temp_image_path)
            print("Temporary file saved. Running analysis...")

            # --- Run the ONNX analysis ---
            coverage_result = run_unet_analysis(temp_image_path)
            print(f"Analysis result: {coverage_result}")

            # --- Return result ---
            if coverage_result is not None:
                return jsonify({'coverage_pct': coverage_result})
            else:
                # run_unet_analysis likely printed an error
                return jsonify({'error': 'Analysis failed on server'}), 500

        except Exception as e:
            # Catch potential errors during saving or analysis function call
            print(f"❌ Error during file saving or analysis call: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            return jsonify({'error': 'Server error during processing'}), 500

        finally:
            # --- CRITICAL: Ensure temporary file is deleted ---
            # This block executes whether the try block succeeded or failed
            if os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                    print(f"✅ Successfully deleted temporary file: {temp_image_path}")
                except Exception as del_e:
                    # Log deletion error but don't necessarily fail the request
                    print(f"⚠️ ERROR deleting temporary file {temp_image_path}: {del_e}")
            # else:
            #      print(f"Temporary file {temp_image_path} not found for deletion (might have failed saving).")

    else:
        # File not present or not allowed extension
        print(f"File type not allowed or file missing: {file.filename}")
        return jsonify({'error': 'File type not allowed or file missing'}), 400

# --- App Execution ---
if __name__ == '__main__':
    # Ensure instance folder exists when running directly
    # This is where temporary files are stored in this example
    try:
        os.makedirs(app.instance_path, exist_ok=True)
        print(f"Instance path: {app.instance_path}")
    except OSError as e:
        print(f"Could not create instance path: {e}")
        # Decide if you want to exit or continue without instance path functionality

    # Run the Flask development server
    # host='0.0.0.0' makes it accessible on your network (use with caution)
    # Use debug=True only for development, not production
    app.run(debug=True, host='127.0.0.1', port=5000)

