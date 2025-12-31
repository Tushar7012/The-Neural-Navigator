"""
Neural Navigator - Training Script
===================================
Training pipeline for the Neural Navigator model.
Includes training loop, validation, loss logging, and visualization.
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from data_loader import create_data_loaders, Vocabulary
from model import NeuralNavigator


class Trainer:
    """
    Trainer class for Neural Navigator model.
    
    Handles:
    - Training loop with progress tracking
    - Validation evaluation
    - Loss logging and visualization
    - Model checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        test_loader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        output_dir: str = "outputs",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Loss function - MSE for coordinate regression
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = None  # Will be set in train()
        
        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float("inf")
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
        )
        
        for batch in progress_bar:
            # Move data to device
            images = batch["image"].to(self.device)
            text_tokens = batch["text_tokens"].to(self.device)
            target_paths = batch["path"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_paths = self.model(images, text_tokens)
            
            # Compute loss
            loss = self.criterion(predicted_paths, target_paths)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Evaluate on validation/test set.
        
        Returns:
            Tuple of (average loss, mean euclidean distance in pixels)
        """
        self.model.eval()
        total_loss = 0.0
        total_distance = 0.0
        num_batches = 0
        num_samples = 0
        
        for batch in self.test_loader:
            images = batch["image"].to(self.device)
            text_tokens = batch["text_tokens"].to(self.device)
            
            # Test data might not have path annotations
            # Use target position if available, otherwise skip loss calculation
            if "path" in batch:
                target_paths = batch["path"].to(self.device)
                predicted_paths = self.model(images, text_tokens)
                
                loss = self.criterion(predicted_paths, target_paths)
                total_loss += loss.item()
                
                # Calculate mean euclidean distance (in pixels)
                # Convert from [0,1] back to pixel coordinates
                pred_pixels = predicted_paths * 128
                target_pixels = target_paths * 128
                distances = torch.sqrt(
                    ((pred_pixels - target_pixels) ** 2).sum(dim=-1)
                )
                total_distance += distances.mean().item() * images.size(0)
                num_samples += images.size(0)
            else:
                # For test data without paths, we can evaluate final point accuracy
                if "target_position" in batch:
                    target_pos = batch["target_position"].to(self.device)
                    predicted_paths = self.model(images, text_tokens)
                    
                    # Compare final predicted point with target position
                    final_pred = predicted_paths[:, -1, :]  # Last point
                    
                    distances = torch.sqrt(
                        ((final_pred * 128 - target_pos * 128) ** 2).sum(dim=-1)
                    )
                    total_distance += distances.mean().item() * images.size(0)
                    num_samples += images.size(0)
            
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_distance = total_distance / num_samples if num_samples > 0 else 0.0
        
        return avg_loss, avg_distance
    
    def train(
        self,
        num_epochs: int = 100,
        save_every: int = 10,
        early_stopping_patience: int = 15,
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of training epochs
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            Dictionary with training history
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"{'='*60}\n")
        
        # Set up scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # Train one epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_distance = self.validate()
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Print progress
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Dist: {val_distance:.2f}px | "
                f"LR: {current_lr:.2e}"
            )
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint("best_model.pth")
                print(f"  --> New best model saved!")
            else:
                patience_counter += 1
            
            # Save periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
        
        # Save training curves
        self.plot_training_curves()
        
        # Save final checkpoint
        self.save_checkpoint("final_model.pth")
        
        # Save training history
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "total_epochs": len(self.train_losses),
            "training_time_minutes": total_time / 60,
        }
        
        with open(os.path.join(self.output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.output_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }, path)
    
    def plot_training_curves(self):
        """Plot and save training loss curves."""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.plot(epochs, self.train_losses, "b-", label="Training Loss", linewidth=2)
        if self.val_losses:
            plt.plot(epochs, self.val_losses, "r-", label="Validation Loss", linewidth=2)
        
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.title("Neural Navigator - Training Loss Curves", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add annotations
        min_train_loss = min(self.train_losses)
        min_val_loss = min(self.val_losses) if self.val_losses else 0
        plt.annotate(
            f"Best Train: {min_train_loss:.4f}",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
        )
        if self.val_losses:
            plt.annotate(
                f"Best Val: {min_val_loss:.4f}",
                xy=(0.02, 0.92),
                xycoords="axes fraction",
                fontsize=10,
                verticalalignment="top",
            )
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "training_loss.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        print(f"\nTraining curves saved to {self.output_dir}/training_loss.png")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Neural Navigator model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--data-dir", type=str, default="data", help="Training data directory")
    parser.add_argument("--test-dir", type=str, default="test_data", help="Test data directory")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, test_loader, vocab = create_data_loaders(
        train_dir=args.data_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = NeuralNavigator(vocab_size=vocab.vocab_size)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
    )
    
    # Train model
    history = trainer.train(
        num_epochs=args.epochs,
        save_every=20,
        early_stopping_patience=15,
    )
    
    print("\nTraining complete!")
    print(f"Final training loss: {history['train_losses'][-1]:.4f}")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    print(f"Model saved to: {args.output_dir}/best_model.pth")


if __name__ == "__main__":
    main()
