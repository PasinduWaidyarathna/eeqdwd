from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
#MODEL_PATH = 'cat_dog_final.keras'
MODEL_PATH = 'cat_dog_final.keras'
model = load_model(MODEL_PATH)

IMG_SIZE = 125

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img = Image.open(file).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize

        prediction = model.predict(img_array)[0][0]
        label = 'dog' if prediction > 0.5 else 'cat'
        confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)

        return jsonify({
            'prediction': label,
            'confidence': round(confidence, 3)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check
@app.route('/', methods=['GET'])
def index():
    return "Cat vs Dog Model API is running."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
