---
description: How to run the Neural Navigator training and inference pipeline
---

# Neural Navigator Workflow

This workflow describes how to train and run inference with the Neural Navigator model.

## Prerequisites

1. Ensure Python 3.8+ is installed
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. For GPU support (recommended):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## Step 1: Verify Data Loader

// turbo
Test that the data loader works correctly:
```bash
python data_loader.py
```

Expected output:
- Vocabulary size: 11
- Training samples: 1000
- Test samples: 100
- Batch shapes verified

## Step 2: Verify Model

// turbo
Test that the model architecture is correct:
```bash
python model.py
```

Expected output:
- Model architecture summary
- Total parameters: ~10.5 million
- Output shape: (batch, 10, 2)

## Step 3: Train the Model

Train with default settings:
```bash
python train.py --epochs 50 --batch-size 32 --lr 0.001 --output-dir outputs
```

Or with custom settings:
```bash
python train.py --epochs 100 --batch-size 16 --lr 0.0005
```

Outputs:
- `outputs/best_model.pth` - Best checkpoint
- `outputs/training_loss.png` - Loss curves
- `outputs/training_history.json` - Training metrics

## Step 4: Run Predictions

Generate predictions on test data:
```bash
python predict.py --checkpoint outputs/best_model.pth --num-samples 20
```

Outputs:
- `predictions/prediction_*.png` - Individual visualizations
- `predictions/prediction_grid.png` - Grid comparison
- `predictions/predictions.json` - Predicted coordinates

## Step 5: Push to GitHub

```bash
git add .
git commit -m "Update model and predictions"
git push origin master
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python train.py --batch-size 16
```

### Unicode Encoding Error on Windows
The project uses ASCII-safe characters. If you add new print statements, avoid Unicode arrows/symbols.

### Model Not Found
Ensure training completed and `outputs/best_model.pth` exists.
