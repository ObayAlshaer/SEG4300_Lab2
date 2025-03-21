from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import torch
from PIL import Image
import numpy as np
import io

# Import your segmentation model
from segmentation_model import SegmentationModel

# Load environment variables from .env file
load_dotenv()

# Access secrets
MODEL_PATH = os.getenv("MODEL_PATH")
API_KEY = os.getenv("API_KEY")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

app = Flask(__name__)

# Initialize the model
model = SegmentationModel(model_path=MODEL_PATH)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "House Segmentation API is running"}), 200

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content response for favicon requests

@app.route('/predict', methods=['POST'])
def predict():
    # Check API key for authentication
    request_api_key = request.headers.get('X-API-Key')
    if request_api_key != API_KEY:
        return jsonify({"error": "Unauthorized access"}), 401

    if 'image' not in request.files:
        return jsonify({"error": "Missing image file"}), 400
    
    # Read image file
    image_file = request.files['image']
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Perform segmentation
    mask, metrics = model.predict(image)
    
    # Convert mask to list format for JSON serialization
    mask_list = mask.tolist() if isinstance(mask, np.ndarray) else mask
    
    result = {
        "mask": mask_list,
        "metrics": {
            "iou": float(metrics["iou"]),
            "dice": float(metrics["dice"])
        }
    }
    
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=DEBUG_MODE)