import os
import pytest
import json
import io
from PIL import Image
import numpy as np
from app import app as flask_app

@pytest.fixture
def app():
    # Create a test environment
    flask_app.config.update({
        "TESTING": True,
    })
    return flask_app

@pytest.fixture
def client(app):
    return app.test_client()

def test_home_endpoint(client):
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["message"] == "House Segmentation API is running"

def test_favicon_endpoint(client):
    response = client.get('/favicon.ico')
    assert response.status_code == 204

def test_predict_without_api_key(client):
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

def test_predict_with_valid_api_key(client, monkeypatch):
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
    from segmentation_model import SegmentationModel
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