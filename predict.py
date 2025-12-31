"""
Neural Navigator - Prediction Script
=====================================
Inference script for running predictions on test images.
Visualizes predicted paths overlaid on the original images.
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional

from data_loader import Vocabulary, NavigatorDataset, create_data_loaders
from model import NeuralNavigator


def load_model(
    checkpoint_path: str,
    vocab_size: int = 11,
    device: torch.device = None,
) -> NeuralNavigator:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        vocab_size: Size of vocabulary
        device: Device to load model on
        
    Returns:
        Loaded NeuralNavigator model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = NeuralNavigator(vocab_size=vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    return model


def predict_single(
    model: NeuralNavigator,
    image: torch.Tensor,
    text_tokens: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Predict path for a single image-text pair.
    
    Args:
        model: Trained model
        image: Image tensor (1, 3, 128, 128)
        text_tokens: Token tensor (1, seq_len)
        device: Device to run on
        
    Returns:
        Predicted path as numpy array (10, 2) in pixel coordinates
    """
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        text_tokens = text_tokens.to(device)
        
        # Predict normalized coordinates
        pred_path = model(image, text_tokens)
        
        # Convert to pixel coordinates
        pred_path = pred_path.cpu().numpy()[0] * 128
    
    return pred_path


def visualize_prediction(
    image_path: str,
    text_command: str,
    predicted_path: np.ndarray,
    target_path: Optional[np.ndarray] = None,
    target_position: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize predicted path on the original image.
    
    Args:
        image_path: Path to original image
        text_command: Text command used
        predicted_path: Predicted path coordinates (N, 2)
        target_path: Ground truth path (optional)
        target_position: Target shape position (optional)
        save_path: Path to save visualization
        show: Whether to display the plot
    """
    # Load original image
    image = Image.open(image_path).convert("RGB")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    
    # Plot predicted path
    pred_x = predicted_path[:, 0]
    pred_y = predicted_path[:, 1]
    
    # Draw predicted path as line with markers
    ax.plot(
        pred_x, pred_y,
        color="blue",
        linewidth=2,
        marker="o",
        markersize=6,
        markerfacecolor="cyan",
        markeredgecolor="blue",
        label="Predicted Path",
        zorder=3,
    )
    
    # Draw start point
    ax.scatter(
        pred_x[0], pred_y[0],
        s=150,
        color="green",
        marker="*",
        zorder=4,
        label="Start",
    )
    
    # Draw end point
    ax.scatter(
        pred_x[-1], pred_y[-1],
        s=150,
        color="red",
        marker="X",
        zorder=4,
        label="Predicted End",
    )
    
    # Plot ground truth path if available
    if target_path is not None:
        gt_x = target_path[:, 0]
        gt_y = target_path[:, 1]
        ax.plot(
            gt_x, gt_y,
            color="lime",
            linewidth=2,
            linestyle="--",
            marker="s",
            markersize=5,
            markerfacecolor="yellow",
            markeredgecolor="lime",
            label="Ground Truth",
            zorder=2,
        )
    
    # Mark target position if available
    if target_position is not None:
        ax.scatter(
            target_position[0], target_position[1],
            s=200,
            color="magenta",
            marker="P",
            zorder=5,
            label="Target Position",
        )
    
    # Add title and legend
    ax.set_title(f'Command: "{text_command}"', fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 128)
    ax.set_ylim(128, 0)  # Flip y-axis to match image coordinates
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def run_predictions(
    model: NeuralNavigator,
    test_loader,
    vocab: Vocabulary,
    device: torch.device,
    output_dir: str,
    num_samples: int = 10,
) -> dict:
    """
    Run predictions on test dataset and save visualizations.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        vocab: Vocabulary instance
        device: Device to run on
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        
    Returns:
        Dictionary with evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    predictions = []
    total_distance = 0.0
    num_evaluated = 0
    
    print(f"\nRunning predictions on {num_samples} test samples...")
    
    with torch.no_grad():
        sample_idx = 0
        for batch in test_loader:
            if sample_idx >= num_samples:
                break
            
            images = batch["image"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            texts = batch["text"]
            image_files = batch["image_file"]
            
            # Get predictions
            pred_paths = model(images, text_tokens)
            pred_paths = pred_paths.cpu().numpy() * 128
            
            # Process each sample in batch
            for i in range(len(images)):
                if sample_idx >= num_samples:
                    break
                
                pred_path = pred_paths[i]
                text = texts[i]
                image_file = image_files[i]
                
                # Get target position if available
                target_pos = None
                if "target_position" in batch:
                    target_pos = batch["target_position"][i].numpy() * 128
                
                # Get ground truth path if available
                gt_path = None
                if "path" in batch:
                    gt_path = batch["path"][i].numpy() * 128
                
                # Calculate distance to target (final point)
                if target_pos is not None:
                    final_pred = pred_path[-1]
                    dist = np.sqrt(np.sum((final_pred - target_pos) ** 2))
                    total_distance += float(dist)
                    num_evaluated += 1
                
                # Create visualization
                image_path = os.path.join(
                    test_loader.dataset.images_dir, image_file
                )
                save_path = os.path.join(
                    output_dir, f"prediction_{sample_idx:04d}.png"
                )
                
                visualize_prediction(
                    image_path=image_path,
                    text_command=text,
                    predicted_path=pred_path,
                    target_path=gt_path,
                    target_position=target_pos,
                    save_path=save_path,
                    show=False,
                )
                
                predictions.append({
                    "image_file": image_file,
                    "text": text,
                    "predicted_path": pred_path.tolist(),
                    "target_position": target_pos.tolist() if target_pos is not None else None,
                })
                
                sample_idx += 1
    
    # Calculate metrics
    avg_distance = total_distance / num_evaluated if num_evaluated > 0 else 0.0
    
    results = {
        "num_samples": sample_idx,
        "average_distance_to_target": avg_distance,
        "predictions": predictions,
    }
    
    # Save predictions
    results_path = os.path.join(output_dir, "predictions.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nPredictions complete!")
    print(f"  Samples evaluated: {sample_idx}")
    print(f"  Average distance to target: {avg_distance:.2f} pixels")
    print(f"  Results saved to: {results_path}")
    
    return results


def create_comparison_grid(
    predictions_dir: str,
    output_path: str,
    num_images: int = 9,
) -> None:
    """
    Create a grid of prediction visualizations for the README.
    
    Args:
        predictions_dir: Directory containing prediction images
        output_path: Path to save the grid image
        num_images: Number of images to include in grid
    """
    # Get prediction images
    pred_files = sorted([
        f for f in os.listdir(predictions_dir)
        if f.startswith("prediction_") and f.endswith(".png")
    ])[:num_images]
    
    if not pred_files:
        print("No prediction images found!")
        return
    
    # Calculate grid dimensions
    n = len(pred_files)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, pred_file in enumerate(pred_files):
        row = idx // cols
        col = idx % cols
        
        img = Image.open(os.path.join(predictions_dir, pred_file))
        axes[row, col].imshow(img)
        axes[row, col].axis("off")
        axes[row, col].set_title(f"Sample {idx + 1}", fontsize=10)
    
    # Hide empty subplots
    for idx in range(len(pred_files), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis("off")
    
    plt.suptitle("Neural Navigator - Sample Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Comparison grid saved to {output_path}")


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Run Neural Navigator predictions")
    parser.add_argument(
        "--checkpoint", type=str, default="outputs/best_model.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test-dir", type=str, default="test_data",
        help="Test data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="predictions",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--num-samples", type=int, default=20,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for inference"
    )
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create vocabulary and data loader
    vocab = Vocabulary()
    test_dataset = NavigatorDataset(
        data_dir=args.test_dir,
        vocab=vocab,
        is_training=False,
    )
    
    from torch.utils.data import DataLoader
    from data_loader import collate_fn
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model
    model = load_model(args.checkpoint, vocab_size=vocab.vocab_size, device=device)
    
    # Run predictions
    results = run_predictions(
        model=model,
        test_loader=test_loader,
        vocab=vocab,
        device=device,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )
    
    # Create comparison grid
    grid_path = os.path.join(args.output_dir, "prediction_grid.png")
    create_comparison_grid(args.output_dir, grid_path, num_images=9)
    
    print("\nPrediction script complete!")


if __name__ == "__main__":
    main()
