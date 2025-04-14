import os
import sys
import unittest
from io import BytesIO
from PIL import Image
import numpy as np

# Add parent directory to path to import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app
import config

class FlaskAppTestCase(unittest.TestCase):
    """Test cases for the Flask application."""
    
    def setUp(self):
        """Set up test client and other test variables."""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'test_uploads')
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        self.client = app.test_client()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test upload directory
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    
    def test_index_page(self):
        """Test that the index page loads correctly."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<!DOCTYPE html>', response.data)
    
    def test_upload_page(self):
        """Test that the upload page loads correctly."""
        response = self.client.get('/upload')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<!DOCTYPE html>', response.data)
    
    def test_gallery_page(self):
        """Test that the gallery page loads correctly."""
        response = self.client.get('/gallery')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<!DOCTYPE html>', response.data)
    
    def test_upload_file(self):
        """Test uploading a file."""
        # Create a test image
        img = Image.new('RGB', (100, 100), color='red')
        img_io = BytesIO()
        img.save(img_io, 'JPEG')
        img_io.seek(0)
        
        # Mock file upload
        response = self.client.post(
            '/upload',
            data={
                'file': (img_io, 'test.jpg')
            },
            content_type='multipart/form-data',
            follow_redirects=True
        )
        
        # Should redirect to gallery after upload
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'gallery', response.data)

if __name__ == '__main__':
    unittest.main()
