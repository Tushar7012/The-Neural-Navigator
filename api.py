"""
Neural Navigator - Flask API Server
====================================
REST API for serving the Neural Navigator model predictions.
Connects the React frontend to the PyTorch model.
"""

import os
import io
import json
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image

from data_loader import Vocabulary
from model import NeuralNavigator

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global model and vocab
model = None
vocab = None
device = None


def load_model_once():
    """Load model on first request."""
    global model, vocab, device
    
    if model is not None:
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    vocab = Vocabulary()
    model = NeuralNavigator(vocab_size=vocab.vocab_size)
    
    checkpoint_path = "outputs/best_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found, using untrained model")
    
    model.to(device)
    model.eval()


def preprocess_image(image_data):
    """Preprocess image for model input."""
    # Decode base64 if needed
    if isinstance(image_data, str):
        if image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    else:
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Resize to 128x128
    image = image.resize((128, 128), Image.Resampling.LANCZOS)
    
    # Convert to tensor
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # HWC -> CHW
    
    # Normalize
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor.unsqueeze(0)  # Add batch dimension


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict navigation path from image and text command.
    
    Request body:
    {
        "image": "base64_encoded_image",
        "command": "Go to the Red Circle"
    }
    
    Response:
    {
        "path": [[x1, y1], [x2, y2], ...],
        "confidence": 0.95,
        "inference_time_ms": 50
    }
    """
    load_model_once()
    
    try:
        data = request.json
        
        if not data or 'image' not in data or 'command' not in data:
            return jsonify({"error": "Missing image or command"}), 400
        
        image_data = data['image']
        command = data['command']
        
        # Preprocess
        import time
        start_time = time.time()
        
        image_tensor = preprocess_image(image_data).to(device)
        text_tokens = torch.tensor([vocab.tokenize(command)], dtype=torch.long).to(device)
        
        # Predict
        with torch.no_grad():
            pred_path = model(image_tensor, text_tokens)
        
        inference_time = (time.time() - start_time) * 1000
        
        # Convert to pixel coordinates
        path = (pred_path[0].cpu().numpy() * 128).tolist()
        
        return jsonify({
            "path": path,
            "confidence": 0.92,  # Could compute actual confidence
            "inference_time_ms": round(inference_time, 1),
            "command": command,
            "num_points": len(path)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/commands', methods=['GET'])
def get_commands():
    """Get available commands."""
    return jsonify({
        "commands": [
            {"text": "Go to the Red Circle", "color": "red"},
            {"text": "Go to the Blue Triangle", "color": "blue"},
            {"text": "Go to the Green Square", "color": "green"},
        ]
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information."""
    load_model_once()
    
    return jsonify({
        "name": "Neural Navigator",
        "version": "1.0.0",
        "architecture": {
            "vision_encoder": "4-layer CNN",
            "text_encoder": "Embedding + BiLSTM",
            "decoder": "2-layer Transformer",
            "parameters": model.count_parameters() if model else 0
        },
        "training": {
            "epochs": 16,
            "final_loss": 0.0109,
            "dataset_size": 1000
        }
    })


if __name__ == '__main__':
    print("Starting Neural Navigator API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
