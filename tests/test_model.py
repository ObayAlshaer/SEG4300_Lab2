import os
import pytest
import numpy as np
import torch
from PIL import Image
from app.segmentation_model import UNet, SegmentationModel

def test_unet_forward():
    """Test that UNet forward pass works with expected input shapes."""
    # Initialize model
    model = UNet(n_channels=3, n_classes=1)
    
    # Create a test input tensor
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    x = torch.randn(batch_size, channels, height, width)
    
    # Test forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, 1, height, width)

def test_unet_architecture():
    """Test UNet architecture components."""
    model = UNet(n_channels=3, n_classes=1)
    
    # Check if model has the expected components
    assert hasattr(model, 'inc')
    assert hasattr(model, 'down1')
    assert hasattr(model, 'down2')
    assert hasattr(model, 'down3')
    assert hasattr(model, 'down4')
    assert hasattr(model, 'up1')
    assert hasattr(model, 'up2')
    assert hasattr(model, 'up3')
    assert hasattr(model, 'up4')
    assert hasattr(model, 'outc')

def test_segmentation_model_init():
    """Test SegmentationModel initialization."""
    model = SegmentationModel()
    assert model is not None
    assert hasattr(model, 'model')
    assert hasattr(model, 'transform')
    assert hasattr(model, 'device')

def test_segmentation_model_predict():
    """Test SegmentationModel prediction."""
    # Initialize model
    model = SegmentationModel()
    
    # Create a test image
    image = Image.new('RGB', (300, 300), color='red')
    
    # Run prediction
    mask, metrics = model.predict(image)
    
    # Check output types and shapes
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (256, 256)  # Default output shape with the preprocessing
    assert isinstance(metrics, dict)
    assert "iou" in metrics
    assert "dice" in metrics

def test_segmentation_model_with_custom_path():
    """Test SegmentationModel with a custom model path."""
    # This test should be skipped if the model file doesn't exist
    model_path = "tests/test_data/dummy_model.pth"
    
    # Check if the test model exists, skip if not
    if not os.path.exists(model_path):
        pytest.skip(f"Test model file {model_path} not found")
    
    model = SegmentationModel(model_path=model_path)
    assert model is not None

def test_confidence_calculation():
    """Test confidence calculation methods."""
    model = SegmentationModel()
    
    # Create a dummy output tensor
    batch_size = 1
    height = 256
    width = 256
    output = torch.zeros((batch_size, 1, height, width))
    
    # Set some regions to high confidence positive predictions
    output[0, 0, 50:150, 50:150] = 5.0  # High sigmoid value
    
    # Set some regions to high confidence negative predictions
    output[0, 0, 200:250, 200:250] = -5.0  # Low sigmoid value
    
    # Calculate confidence
    iou_conf = model._calculate_confidence(output)
    dice_conf = model._calculate_dice_confidence(output)
    
    # Check that the values are within expected range (0-1)
    assert 0 <= iou_conf <= 1
    assert 0 <= dice_conf <= 1