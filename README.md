# Foreskin Tracker

A Flask web application for tracking foreskin restoration progress using U-Net image segmentation.

## Features

- Upload images to track foreskin restoration progress
- Process images using a U-Net deep learning model
- View a gallery of processed images to track progress over time

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/foreskin_tracker.git
   cd foreskin_tracker
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Place your trained U-Net model in the `models` directory:
   ```
   models/unet_model.pth
   ```

## Usage

1. Run the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Use the upload form to submit images for processing
4. View your progress in the gallery

## Project Structure

```
foreskin_tracker/
├── venv/                   # Virtual environment directory
├── models/                 # To store your trained model files
│   └── unet_model.pth      # Your trained U-Net model file
├── static/                 # For static files (CSS, JavaScript, UI images)
│   ├── css/
│   │   └── main.css        # Primary CSS file
│   ├── js/
│   │   └── main.js         # Primary JavaScript file
│   └── images/             # Static UI images (logos, icons, etc.)
├── templates/              # For HTML templates that Flask will render
│   ├── base.html           # Base template for common layout
│   ├── index.html          # Main page template
│   ├── upload.html         # Template for the upload form
│   └── gallery.html        # Template to display progress photos
├── uploads/                # Temporary storage for user uploads
├── tests/                  # For application tests
│   └── test_app.py
├── app.py                  # Main Flask application file
├── unet_predictor.py       # Python module containing U-Net prediction functionality
├── config.py               # Configuration settings
├── requirements.txt        # List of Python package dependencies
├── .gitignore              # Tells Git which files/directories to ignore
└── README.md               # Project description and setup instructions
```

## Training Your Own Model

This application uses a pre-trained U-Net model for foreskin segmentation. If you want to train your own model:

1. Collect a dataset of foreskin images with corresponding segmentation masks
2. Use the U-Net architecture in `unet_predictor.py` to train your model
3. Save the trained model to `models/unet_model.pth`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# forestore
