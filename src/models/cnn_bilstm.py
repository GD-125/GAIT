# models/cnn_bilstm.py
"""
1D-CNN + BiLSTM Model for Gait Detection
Combines convolutional layers for local feature extraction with bidirectional LSTM 
for temporal sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CNN_BiLSTM_GaitDetector(nn.Module):
    """
    1D-CNN + BiLSTM architecture for binary gait detection.
    
    Architecture:
        Input (batch, seq_len, features)
        ↓
        Conv1D Blocks (3 layers with residual connections)
        ↓
        BiLSTM Layers (2 layers)
        ↓
        Global Average Pooling
        ↓
        Fully Connected Layers
        ↓
        Binary Output (Gait / Non-Gait)
    """
    
    def __init__(self,
                 input_features: int = 38,
                 seq_length: int = 128,
                 conv_filters: list = [64, 128, 256],
                 kernel_sizes: list = [5, 5, 5],
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 2,
                 fc_hidden_sizes: list = [256, 128],
                 dropout: float = 0.3,
                 use_batch_norm: bool = True,
                 use_residual: bool = True):
        """
        Initialize the CNN-BiLSTM model.
        
        Args:
            input_features: Number of input features (38 for your dataset)
            seq_length: Sequence length (window size)
            conv_filters: List of filter sizes for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            lstm_hidden_size: Hidden size for LSTM layers
            lstm_num_layers: Number of BiLSTM layers
            fc_hidden_sizes: List of hidden sizes for FC layers
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections in CNN
        """
        super(CNN_BiLSTM_GaitDetector, self).__init__()
        
        self.input_features = input_features
        self.seq_length = seq_length
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.use_residual = use_residual
        
        # ============ CNN Layers ============
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.residual_projections = nn.ModuleList() if use_residual else None
        
        in_channels = input_features
        for i, (out_channels, kernel_size) in enumerate(zip(conv_filters, kernel_sizes)):
            # Conv1D layer
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # Same padding
                    bias=not use_batch_norm
                )
            )
            
            # Batch normalization
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(out_channels))
            
            # Residual projection if input/output dims don't match
            if use_residual and in_channels != out_channels:
                self.residual_projections.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1)
                )
            else:
                self.residual_projections.append(None) if use_residual else None
            
            in_channels = out_channels
        
        self.conv_dropout = nn.Dropout(dropout)
        
        # ============ BiLSTM Layers ============
        self.lstm = nn.LSTM(
            input_size=conv_filters[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.lstm_dropout = nn.Dropout(dropout)
        
        # ============ Attention Mechanism (Optional) ============
        self.attention = nn.Linear(lstm_hidden_size * 2, 1)
        
        # ============ Fully Connected Layers ============
        self.fc_layers = nn.ModuleList()
        
        # First FC layer takes BiLSTM output
        fc_input_size = lstm_hidden_size * 2  # *2 for bidirectional
        
        for fc_hidden_size in fc_hidden_sizes:
            self.fc_layers.append(nn.Linear(fc_input_size, fc_hidden_size))
            fc_input_size = fc_hidden_size
        
        # Output layer (binary classification)
        self.output_layer = nn.Linear(fc_input_size, 1)
        
        self.fc_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_features)

        Returns:
            Output tensor of shape (batch_size, 1) with raw logits (use with BCEWithLogitsLoss)
        """
        batch_size = x.size(0)
        
        # ============ CNN Feature Extraction ============
        # Transpose for Conv1D: (batch, features, seq_length)
        x = x.transpose(1, 2)
        
        for i, conv_layer in enumerate(self.conv_layers):
            identity = x
            
            # Convolution
            x = conv_layer(x)
            
            # Batch normalization
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # Activation
            x = F.relu(x)
            
            # Residual connection
            if self.use_residual:
                if self.residual_projections[i] is not None:
                    identity = self.residual_projections[i](identity)
                x = x + identity
            
            # Max pooling (reduce temporal dimension)
            if i < len(self.conv_layers) - 1:  # Don't pool after last conv
                x = F.max_pool1d(x, kernel_size=2)
        
        x = self.conv_dropout(x)
        
        # Transpose back for LSTM: (batch, seq_length, features)
        x = x.transpose(1, 2)
        
        # ============ BiLSTM Temporal Modeling ============
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch, seq_length, lstm_hidden_size * 2)
        
        lstm_out = self.lstm_dropout(lstm_out)
        
        # ============ Attention Mechanism ============
        # Calculate attention weights
        attention_weights = torch.tanh(self.attention(lstm_out))
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention (weighted sum)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        # context_vector shape: (batch, lstm_hidden_size * 2)
        
        # Alternative: Use last hidden states from both directions
        # h_n shape: (num_layers * 2, batch, lstm_hidden_size)
        # forward_hidden = h_n[-2, :, :]
        # backward_hidden = h_n[-1, :, :]
        # context_vector = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # ============ Fully Connected Layers ============
        x = context_vector
        
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = F.relu(x)
            x = self.fc_dropout(x)
        
        # Output layer (returns logits for use with BCEWithLogitsLoss)
        x = self.output_layer(x)

        return x
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings before the final classification layer.
        Useful for visualization and explainability.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature embeddings
        """
        batch_size = x.size(0)
        
        # CNN layers
        x = x.transpose(1, 2)
        for i, conv_layer in enumerate(self.conv_layers):
            identity = x
            x = conv_layer(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            if self.use_residual and self.residual_projections[i] is not None:
                identity = self.residual_projections[i](identity)
                x = x + identity
            if i < len(self.conv_layers) - 1:
                x = F.max_pool1d(x, kernel_size=2)
        
        x = self.conv_dropout(x)
        x = x.transpose(1, 2)
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Attention
        attention_weights = torch.tanh(self.attention(lstm_out))
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # FC layers (excluding final output layer)
        x = context_vector
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = F.relu(x)
            x = self.fc_dropout(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_model_summary(self):
        """Print model architecture summary."""
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*70)
        print(f"Model: CNN-BiLSTM Gait Detector")
        print(f"\nInput Shape: (batch_size, {self.seq_length}, {self.input_features})")
        print(f"\n--- CNN Layers ---")
        for i, conv_layer in enumerate(self.conv_layers):
            print(f"  Conv1D-{i+1}: {conv_layer.in_channels} → {conv_layer.out_channels} "
                  f"(kernel={conv_layer.kernel_size[0]})")
        
        print(f"\n--- BiLSTM Layers ---")
        print(f"  BiLSTM: {self.lstm_num_layers} layers × {self.lstm_hidden_size} hidden units")
        print(f"  Output: {self.lstm_hidden_size * 2} (bidirectional)")
        
        print(f"\n--- Fully Connected Layers ---")
        prev_size = self.lstm_hidden_size * 2
        for i, fc_layer in enumerate(self.fc_layers):
            print(f"  FC-{i+1}: {fc_layer.in_features} → {fc_layer.out_features}")
        print(f"  Output: {self.output_layer.in_features} → 1 (binary)")
        
        print(f"\nTotal Parameters: {self.count_parameters():,}")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Example usage and testing
    print("CNN-BiLSTM Gait Detector Model\n")
    
    # Create model
    model = CNN_BiLSTM_GaitDetector(
        input_features=38,
        seq_length=128,
        conv_filters=[64, 128, 256],
        kernel_sizes=[5, 5, 5],
        lstm_hidden_size=128,
        lstm_num_layers=2,
        fc_hidden_sizes=[256, 128],
        dropout=0.3
    )
    
    # Print model summary
    model.print_model_summary()
    
    # Test forward pass
    batch_size = 16
    dummy_input = torch.randn(batch_size, 128, 38)
    
    print("Testing forward pass...")
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test embeddings extraction
    embeddings = model.get_embeddings(dummy_input)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    
    print("\n✓ Model test successful!")
