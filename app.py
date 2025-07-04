from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import io
import base64
from ultralytics import YOLO
import logging
import numpy as np

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
try:
    model = YOLO('best.pt')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

# Food database
FOOD_DATABASE = {
    'kokis': {
        'description': 'Traditional Sri Lankan crispy snack',
        'ingredients': ['Rice flour', 'coconut milk', 'sugar', 'turmeric'],
        'origin': 'Sri Lanka',
        'calories': 150
    },
    'konda_kewum': {
        'description': 'Traditional Sri Lankan sweet made with rice flour and treacle',
        'ingredients': ['Rice flour', 'palm treacle', 'coconut', 'cardamom'],
        'origin': 'Sri Lanka',
        'calories': 280
    }
}

@app.before_request
def log_request():
    logger.info(f"Incoming request: {request.method} {request.path}")
    logger.debug(f"Headers: {dict(request.headers)}")
    logger.debug(f"Body size: {request.content_length} bytes")

def process_prediction(results):
    """Process YOLOv8 results with robust error handling"""
    predictions = []
    
    for result in results:
        if not hasattr(result, 'boxes'):
            continue
            
        try:
            for box in result.boxes:
                predictions.append({
                    'class': str(result.names[int(box.cls)]),
                    'confidence': round(float(box.conf), 4),
                    'bbox': [round(float(x), 1) for x in box.xyxy[0].tolist()]
                })
        except Exception as e:
            logger.warning(f"Skipping invalid box: {str(e)}")
            continue
    
    return predictions

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Missing image data"}), 400

        # Process image
        try:
            image_base64 = data['image'].split(",")[-1]  # Remove data: prefix
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            return jsonify({"error": "Invalid image data"}), 400

        # Run inference
        results = model(image, conf=0.25)  # Lower confidence threshold
        predictions = process_prediction(results)
        
        logger.info(f"Returning {len(predictions)} predictions")
        if predictions:
            logger.debug(f"Sample prediction: {predictions[0]}")

        return jsonify(predictions)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)