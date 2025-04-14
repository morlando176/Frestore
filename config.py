import os
import secrets

class Config:
    """Configuration settings for the Flask application."""
    
    # Flask settings
    DEBUG = True
    SECRET_KEY = secrets.token_hex(16)
    
    # File upload settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    
    # Model settings
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'unet_model.pth')
    
    # Database settings (if needed in the future)
    # DATABASE_URI = 'sqlite:///foreskin_tracker.db'
