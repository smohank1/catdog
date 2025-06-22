from flask import Flask, request, jsonify, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable to store the model
model = None

def load_catdog_model():
    """Load the pre-trained cat vs dog model"""
    global model
    try:
        model = load_model('catdogmodel0603.h5')
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    """Preprocess image for model prediction"""
    try:
        # Resize image to 64x64
        img = img.resize((64, 64))
        
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to array
        img_array = image.img_to_array(img)
        
        # Reshape for single prediction (1, 64, 64, 3)
        img_array = img_array.reshape(1, 64, 64, 3)
        
        # Normalize pixel values (if your model was trained with normalization)
        # Uncomment the line below if your model expects normalized inputs
        # img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def predict_image(img_array):
    """Make prediction on preprocessed image"""
    try:
        if model is None:
            raise Exception("Model not loaded")
        
        # Make prediction
        result = model.predict(img_array)
        
        # Get prediction confidence
        confidence = float(result[0][0])
        
        # Classify based on threshold
        if confidence >= 0.5:
            prediction = 'dog'
            confidence_percentage = confidence * 100
        else:
            prediction = 'cat'
            confidence_percentage = (1 - confidence) * 100
        
        return {
            'prediction': prediction,
            'confidence': round(confidence_percentage, 2),
            'raw_output': float(confidence)
        }
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")

@app.route('/')
def home():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload an image file.'}), 400
        
        # Read and process the image
        img = Image.open(file.stream)
        
        # Preprocess image
        img_array = preprocess_image(img)
        
        # Make prediction
        result = predict_image(img_array)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'message': f"This is a {result['prediction']} with {result['confidence']:.1f}% confidence"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """API endpoint for programmatic access"""
    try:
        # Handle both file upload and base64 encoded images
        if 'file' in request.files:
            file = request.files['file']
            img = Image.open(file.stream)
        elif 'image_data' in request.json:
            # Handle base64 encoded image
            image_data = request.json['image_data']
            img_data = base64.b64decode(image_data.split(',')[1])
            img = Image.open(io.BytesIO(img_data))
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess and predict
        img_array = preprocess_image(img)
        result = predict_image(img_array)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'tensorflow_version': tf.__version__
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    if load_catdog_model():
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please ensure 'catdogmodel0603.h5' is in the same directory.")
        print("Application will not start without the model file.")