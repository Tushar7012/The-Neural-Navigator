"""
Neural Navigator - Model Architecture
======================================
Multi-modal neural network for path prediction from images and text commands.

Architecture:
- Vision Encoder: CNN-based feature extractor
- Text Encoder: Embedding + LSTM
- Path Decoder: Transformer-based sequence generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer decoder.
    Adds position information to the input embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class VisionEncoder(nn.Module):
    """
    CNN-based vision encoder for extracting features from map images.
    
    Architecture:
    - 3 convolutional layers with batch normalization and ReLU
    - AdaptiveAvgPool to fixed spatial size
    - Linear projection to feature dimension
    
    Input: (batch, 3, 128, 128)
    Output: (batch, feature_dim)
    """
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv block 1: 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Conv block 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Conv block 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Conv block 4: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Project to feature dimension
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (batch, 3, 128, 128)
        Returns:
            Feature vector of shape (batch, feature_dim)
        """
        x = self.conv_layers(x)
        x = self.pool(x)
        x = self.fc(x)
        return x


class TextEncoder(nn.Module):
    """
    Text encoder using embeddings and LSTM.
    
    Architecture:
    - Learnable word embeddings
    - Bidirectional LSTM
    - Linear projection to feature dimension
    
    Input: (batch, seq_len) token indices
    Output: (batch, feature_dim)
    """
    
    def __init__(
        self,
        vocab_size: int = 11,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        feature_dim: int = 256,
        num_layers: int = 1,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0,
        )
        
        # Bidirectional LSTM outputs 2 * hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices of shape (batch, seq_len)
        Returns:
            Feature vector of shape (batch, feature_dim)
        """
        # Get embeddings
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # Pass through LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Concatenate final hidden states from both directions
        # hidden shape: (num_layers * 2, batch, hidden_dim)
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Project to feature dimension
        features = self.fc(hidden_cat)
        return features


class PathDecoder(nn.Module):
    """
    Transformer-based decoder for generating path coordinates.
    
    Architecture:
    - Learnable query embeddings for each path point
    - Positional encoding
    - Transformer decoder layers
    - Linear projection to (x, y) coordinates
    
    Input: Fused features (batch, feature_dim)
    Output: Path coordinates (batch, num_points, 2)
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_points: int = 10,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_points = num_points
        self.feature_dim = feature_dim
        
        # Learnable query embeddings for each path point
        self.query_embed = nn.Parameter(torch.randn(num_points, feature_dim))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(feature_dim, max_len=num_points)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        
        # Output projection to coordinates
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 2),
            nn.Sigmoid(),  # Output in [0, 1] range
        )
    
    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory: Fused features of shape (batch, feature_dim)
        Returns:
            Path coordinates of shape (batch, num_points, 2)
        """
        batch_size = memory.size(0)
        
        # Expand memory to sequence format for transformer
        # (batch, 1, feature_dim)
        memory = memory.unsqueeze(1)
        
        # Expand query embeddings for batch
        # (batch, num_points, feature_dim)
        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.pos_encoder(queries)
        
        # Decode path
        decoded = self.transformer_decoder(queries, memory)
        
        # Project to coordinates
        path = self.output_proj(decoded)
        
        return path


class NeuralNavigator(nn.Module):
    """
    Complete Neural Navigator model combining vision, text, and path prediction.
    
    Architecture:
    1. Vision Encoder (CNN) -> visual features
    2. Text Encoder (LSTM) -> text features
    3. Feature Fusion (concatenation + projection)
    4. Path Decoder (Transformer) -> (x, y) coordinates
    
    Input:
    - image: (batch, 3, 128, 128)
    - text_tokens: (batch, seq_len)
    
    Output:
    - path: (batch, 10, 2) normalized coordinates
    """
    
    def __init__(
        self,
        vocab_size: int = 11,
        vision_feature_dim: int = 256,
        text_feature_dim: int = 256,
        fusion_dim: int = 512,
        num_path_points: int = 10,
        num_decoder_heads: int = 4,
        num_decoder_layers: int = 2,
    ):
        super().__init__()
        
        # Encoders
        self.vision_encoder = VisionEncoder(feature_dim=vision_feature_dim)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size, feature_dim=text_feature_dim
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(vision_feature_dim + text_feature_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(0.1),
        )
        
        # Path decoder
        self.path_decoder = PathDecoder(
            feature_dim=fusion_dim,
            num_points=num_path_points,
            num_heads=num_decoder_heads,
            num_layers=num_decoder_layers,
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, image: torch.Tensor, text_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the complete model.
        
        Args:
            image: Image tensor of shape (batch, 3, 128, 128)
            text_tokens: Token indices of shape (batch, seq_len)
            
        Returns:
            Predicted path of shape (batch, 10, 2)
        """
        # Encode image and text
        vision_features = self.vision_encoder(image)  # (batch, vision_dim)
        text_features = self.text_encoder(text_tokens)  # (batch, text_dim)
        
        # Fuse features
        combined = torch.cat([vision_features, text_features], dim=1)
        fused = self.fusion(combined)  # (batch, fusion_dim)
        
        # Decode path
        path = self.path_decoder(fused)  # (batch, num_points, 2)
        
        return path
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing Neural Navigator model...")
    
    # Create model
    model = NeuralNavigator(vocab_size=11)
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal trainable parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    image = torch.randn(batch_size, 3, 128, 128)
    text_tokens = torch.randint(0, 11, (batch_size, 5))
    
    print(f"\nInput shapes:")
    print(f"  Image: {image.shape}")
    print(f"  Text tokens: {text_tokens.shape}")
    
    # Forward pass
    with torch.no_grad():
        path = model(image, text_tokens)
    
    print(f"\nOutput shape: {path.shape}")
    print(f"Output range: [{path.min().item():.4f}, {path.max().item():.4f}]")
    print(f"Sample path point: {path[0, 0]}")
    
    print("\nModel test passed!")
