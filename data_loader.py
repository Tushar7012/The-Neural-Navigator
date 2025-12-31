"""
Neural Navigator - Data Loader Module
=====================================
Custom PyTorch DataLoader for the Neural Navigator dataset.
Handles image loading, text tokenization, and path coordinate preprocessing.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional


class Vocabulary:
    """
    Simple vocabulary class for text tokenization.
    Maps words to indices and vice versa.
    """
    
    def __init__(self):
        # Define vocabulary for navigation commands
        # Commands follow pattern: "Go to the [Color] [Shape]"
        self.word2idx = {
            "<PAD>": 0,
            "<UNK>": 1,
            "go": 2,
            "to": 3,
            "the": 4,
            "red": 5,
            "blue": 6,
            "green": 7,
            "circle": 8,
            "triangle": 9,
            "square": 10,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self.max_length = 5  # "Go to the Red Circle" = 5 tokens
    
    def tokenize(self, text: str) -> List[int]:
        """
        Convert text to list of token indices.
        
        Args:
            text: Input text command (e.g., "Go to the Red Circle")
            
        Returns:
            List of token indices
        """
        tokens = text.lower().split()
        indices = []
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx["<UNK>"])
        
        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices.extend([self.word2idx["<PAD>"]] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]
            
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Convert token indices back to text."""
        words = [self.idx2word.get(idx, "<UNK>") for idx in indices]
        return " ".join(words)


class NavigatorDataset(Dataset):
    """
    Custom PyTorch Dataset for the Neural Navigator task.
    
    Loads:
    - Images: 128x128 RGB PNG files
    - Text: Navigation commands (e.g., "Go to the Red Circle")
    - Paths: Sequence of 10 (x, y) coordinates (training only)
    
    Args:
        data_dir: Path to data directory (e.g., "data" or "test_data")
        vocab: Vocabulary instance for text tokenization
        is_training: Whether this is training data (has path annotations)
        transform: Optional image transforms
    """
    
    def __init__(
        self,
        data_dir: str,
        vocab: Vocabulary,
        is_training: bool = True,
        transform=None
    ):
        self.data_dir = data_dir
        self.vocab = vocab
        self.is_training = is_training
        self.transform = transform
        
        self.images_dir = os.path.join(data_dir, "images")
        self.annotations_dir = os.path.join(data_dir, "annotations")
        
        # Load metadata
        metadata_path = os.path.join(data_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        self.samples = self.metadata["samples"]
        self.image_size = self.metadata["dataset_info"]["image_size"]
        self.num_path_points = self.metadata["dataset_info"]["num_path_points"]
        
        # Image normalization parameters
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
            - image: Normalized image tensor (3, 128, 128)
            - text_tokens: Token indices tensor (max_length,)
            - path: Path coordinates tensor (10, 2) - normalized to [0, 1]
            - target_position: Target position (2,) - only for test data
        """
        sample_info = self.samples[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, sample_info["image_file"])
        image = Image.open(image_path).convert("RGB")
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Normalize image
        image = (image - self.mean) / self.std
        
        # Tokenize text
        text = sample_info["text"]
        text_tokens = torch.tensor(self.vocab.tokenize(text), dtype=torch.long)
        
        result = {
            "image": image,
            "text_tokens": text_tokens,
            "text": text,
            "image_file": sample_info["image_file"],
        }
        
        # Load path from annotation file (training data has paths, test data doesn't)
        annotation_path = os.path.join(
            self.annotations_dir, sample_info["annotation_file"]
        )
        with open(annotation_path, "r") as f:
            annotation = json.load(f)
        
        if self.is_training and "path" in annotation:
            # Path coordinates normalized to [0, 1]
            path = torch.tensor(annotation["path"], dtype=torch.float32)
            path = path / self.image_size  # Normalize to [0, 1]
            result["path"] = path
        
        # Include target information
        if "target" in annotation:
            result["target"] = annotation["target"]
            if "position" in annotation["target"]:
                target_pos = torch.tensor(
                    annotation["target"]["position"], dtype=torch.float32
                ) / self.image_size
                result["target_position"] = target_pos
        
        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to batch samples.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary with stacked tensors
    """
    collated = {
        "image": torch.stack([s["image"] for s in batch]),
        "text_tokens": torch.stack([s["text_tokens"] for s in batch]),
        "text": [s["text"] for s in batch],
        "image_file": [s["image_file"] for s in batch],
    }
    
    # Stack paths if available
    if "path" in batch[0]:
        collated["path"] = torch.stack([s["path"] for s in batch])
    
    # Stack target positions if available
    if "target_position" in batch[0]:
        collated["target_position"] = torch.stack([s["target_position"] for s in batch])
    
    return collated


def create_data_loaders(
    train_dir: str = "data",
    test_dir: str = "test_data",
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, Vocabulary]:
    """
    Create training and test data loaders.
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, test_loader, vocabulary)
    """
    vocab = Vocabulary()
    
    train_dataset = NavigatorDataset(
        data_dir=train_dir,
        vocab=vocab,
        is_training=True,
    )
    
    test_dataset = NavigatorDataset(
        data_dir=test_dir,
        vocab=vocab,
        is_training=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return train_loader, test_loader, vocab


if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader...")
    
    train_loader, test_loader, vocab = create_data_loaders(
        batch_size=4, num_workers=0
    )
    
    print(f"\nVocabulary size: {vocab.vocab_size}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Get a sample batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Text tokens: {batch['text_tokens'].shape}")
    print(f"  Paths: {batch['path'].shape}")
    
    print(f"\nSample texts: {batch['text'][:2]}")
    print(f"Sample path (first point): {batch['path'][0, 0]}")
    
    # Test tokenization
    test_text = "Go to the Blue Triangle"
    tokens = vocab.tokenize(test_text)
    print(f"\nTokenization test:")
    print(f"  Text: '{test_text}'")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded: '{vocab.decode(tokens)}'")
    
    print("\nData loader test passed!")
