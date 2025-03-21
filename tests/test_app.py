import os
import pytest
import json
import io
from PIL import Image
import numpy as np
from app.app import app as flask_app

@pytest.fixture
def app():
    """Create a test Flask application."""
    # Create a test environment
    flask_app.config.update({
        "TESTING": True,
    })
    return flask_app

@pytest.fixture
def client(app):
    """Create a test client for the Flask application."""
    return app.test_client()

def test_home_endpoint(client):
    """Test the home endpoint returns the expected status and message."""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["message"] == "House Segmentation API is running"

def test_favicon_endpoint(client):
    """Test the favicon endpoint returns a 204 status code."""
    response = client.get('/favicon.ico')
    assert response.status_code == 204

def test_predict_without_api_key(client):
    """Test that the predict endpoint requires an API key."""
    # Create a test image
    img = Image.new('RGB', (100, 100), color='red')
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    
    response = client.post(
        '/predict',
        data={'image': (img_io, 'test.jpg')},
        content_type='multipart/form-data'
    )
    assert response.status_code == 401
    data = json.loads(response.data)
    assert "error" in data
    assert "Unauthorized" in data["error"]

def test_predict_with_valid_api_key(client, monkeypatch):
    """Test that the predict endpoint works with a valid API key."""
    # Mock the API key validation
    monkeypatch.setenv("API_KEY", "test_api_key")
    
    # Create a test image
    img = Image.new('RGB', (100, 100), color='red')
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    
    # Create a mock model prediction
    def mock_predict(self, image):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:150, 50:150] = 1  # Create a simple square mask
        metrics = {"iou": 0.85, "dice": 0.90}
        return mask, metrics
    
    # Apply the mock
    from app.segmentation_model import SegmentationModel
    monkeypatch.setattr(SegmentationModel, "predict", mock_predict)
    
    response = client.post(
        '/predict',
        data={'image': (img_io, 'test.jpg')},
        headers={'X-API-Key': 'test_api_key'},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "mask" in data
    assert "metrics" in data
    assert "iou" in data["metrics"]
    assert "dice" in data["metrics"]

def test_predict_without_image(client, monkeypatch):
    """Test that the predict endpoint requires an image file."""
    # Mock the API key validation
    monkeypatch.setenv("API_KEY", "test_api_key")
    
    response = client.post(
        '/predict',
        data={},
        headers={'X-API-Key': 'test_api_key'},
        content_type='multipart/form-data'
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "Missing image file" in data["error"]

def test_predict_with_invalid_image(client, monkeypatch):
    """Test that the predict endpoint handles invalid image files."""
    # Mock the API key validation
    monkeypatch.setenv("API_KEY", "test_api_key")
    
    # Create invalid image data
    invalid_data = io.BytesIO(b"not an image")
    invalid_data.seek(0)
    
    response = client.post(
        '/predict',
        data={'image': (invalid_data, 'invalid.jpg')},
        headers={'X-API-Key': 'test_api_key'},
        content_type='multipart/form-data'
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data