# Neural Navigator Assignment

Build a neural network that acts like a "Smart GPS." Given a 2D map image and a text command, the model should generate a path to the specified target.

## Dataset Structure

```
data/                          # Training dataset (1000 samples)
├── images/                    # 128x128 PNG images
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── annotations/               # JSON annotations per sample
│   ├── 000000.json
│   └── ...
└── metadata.json              # Dataset summary

test_data/                     # Test dataset (100 samples)
├── images/
├── annotations/
└── metadata.json
```

## Data Format

### Images
- Size: 128×128 pixels, RGB
- Content: White background with colored shapes
- Shapes: Red Circle, Blue Triangle, Green Square

### Training Annotations (`data/annotations/*.json`)
```json
{
  "id": 0,
  "image_file": "000000.png",
  "text": "Go to the Red Circle",
  "path": [[64.0, 64.0], [67.6, 62.7], ..., [97.0, 53.0]],
  "target": {
    "shape": "Circle",
    "color": "Red",
    "position": [97, 53]
  }
}
```

| Field | Description |
|-------|-------------|
| `text` | Navigation instruction |
| `path` | 10 (x, y) coordinates from image center to target |
| `target` | Target shape info with position |

### Test Annotations (`test_data/annotations/*.json`)
```json
{
  "id": 0,
  "image_file": "000000.png",
  "text": "Go to the Green Square",
  "target": {
    "shape": "Square",
    "color": "Green"
  }
}
```


## Tasks

### 1. Data Loader
Write a custom PyTorch/JAX DataLoader that:
- Loads images and annotations
- Returns clean batches of tensors ready for training

### 2. Model Architecture
Build a Multi-Modal Transformer with:
- **Vision Encoder**: CNN or Linear Patch Projection
- **Text Encoder**: Learnable embeddings for text tokens
- **Fusion**: Concatenate image and text features
- **Decoder**: Output sequence of 10 (x, y) coordinates

**Constraint**: Do not use pre-made libraries like Hugging Face Trainer.

### 3. Training Loop
- Loss: MSE between predicted and ground truth path coordinates
- Log training loss curves (should decrease visibly)

### 4. Prediction Script (`predict.py`)
- Input: Test image + text command
- Output: Save a `.png` with predicted path drawn on the image


## Submission

Submit a Github Repo containing:
- `data_loader.py`
- `model.py`
- `train.py`
- `predict.py`
- `requirements.txt`
- Training loss curves
- Sample prediction images from test set
