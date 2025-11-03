"""
Neurological Disease Classification Model
Multi-class classifier for identifying specific neurological diseases from gait patterns.

Supported Diseases:
- Parkinson's Disease
- Huntington's Disease
- Cerebral Palsy
- Multiple Sclerosis
- Ataxia
- Normal/Healthy Gait
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class DiseaseClassifier(nn.Module):
    """
    Deep learning model for neurological disease classification.

    Architecture: CNN + Transformer + Fully Connected Layers
    - CNN layers extract local temporal features from gait patterns
    - Transformer captures long-range dependencies and attention
    - FC layers perform final classification
    """

    def __init__(self,
                 input_features: int = 38,
                 seq_length: int = 128,
                 num_diseases: int = 6,
                 conv_filters: list = [64, 128, 256],
                 transformer_heads: int = 8,
                 transformer_layers: int = 4,
                 transformer_dim: int = 256,
                 fc_hidden_sizes: list = [512, 256],
                 dropout: float = 0.3,
                 use_batch_norm: bool = True):
        """
        Initialize disease classifier.

        Args:
            input_features: Number of input features (sensors)
            seq_length: Sequence length (time steps)
            num_diseases: Number of disease classes
            conv_filters: List of conv filter sizes
            transformer_heads: Number of attention heads
            transformer_layers: Number of transformer layers
            transformer_dim: Transformer embedding dimension
            fc_hidden_sizes: List of FC layer sizes
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(DiseaseClassifier, self).__init__()

        self.input_features = input_features
        self.seq_length = seq_length
        self.num_diseases = num_diseases
        self.use_batch_norm = use_batch_norm

        # Disease mapping
        self.disease_names = [
            "Parkinson's Disease",
            "Huntington's Disease",
            "Cerebral Palsy",
            "Multiple Sclerosis",
            "Ataxia",
            "Normal/Healthy Gait"
        ]

        # ==================== CNN Feature Extraction ====================
        self.conv_layers = nn.ModuleList()
        in_channels = input_features

        for filters in conv_filters:
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, filters, kernel_size=5, padding=2),
                nn.BatchNorm1d(filters) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(filters, filters, kernel_size=3, padding=1),
                nn.BatchNorm1d(filters) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            self.conv_layers.append(conv_block)
            in_channels = filters

        # Calculate sequence length after conv layers
        self.conv_out_length = seq_length
        for _ in conv_filters:
            self.conv_out_length = self.conv_out_length // 2

        # ==================== Transformer Encoder ====================
        # Project CNN features to transformer dimension
        self.feature_projection = nn.Linear(conv_filters[-1], transformer_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.conv_out_length, transformer_dim)
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        # ==================== Attention Pooling ====================
        self.attention = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.Tanh(),
            nn.Linear(transformer_dim // 2, 1)
        )

        # ==================== Fully Connected Layers ====================
        fc_layers = []
        in_features = transformer_dim

        for hidden_size in fc_hidden_sizes:
            fc_layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_features = hidden_size

        self.fc_layers = nn.Sequential(*fc_layers)

        # Output layer for multi-class classification
        self.output_layer = nn.Linear(fc_hidden_sizes[-1], num_diseases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_length, input_features)

        Returns:
            Class probabilities (batch_size, num_diseases)
        """
        batch_size = x.size(0)

        # Transpose for Conv1d: (batch, features, seq_length)
        x = x.transpose(1, 2)

        # CNN feature extraction
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Transpose back: (batch, seq_length, features)
        x = x.transpose(1, 2)

        # Project to transformer dimension
        x = self.feature_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Attention pooling
        attention_weights = F.softmax(self.attention(x), dim=1)
        x = torch.sum(attention_weights * x, dim=1)

        # Fully connected layers
        x = self.fc_layers(x)

        # Output layer
        logits = self.output_layer(x)
        probabilities = F.softmax(logits, dim=1)

        return probabilities

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with class labels and probabilities.

        Args:
            x: Input tensor

        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
            predicted_classes = torch.argmax(probs, dim=1)

        return predicted_classes, probs

    def get_disease_name(self, class_index: int) -> str:
        """Get disease name from class index."""
        return self.disease_names[class_index]

    def print_model_summary(self):
        """Print model architecture summary."""
        print("\n" + "="*80)
        print("NEUROLOGICAL DISEASE CLASSIFIER - MODEL SUMMARY")
        print("="*80)
        print(f"Input Features: {self.input_features}")
        print(f"Sequence Length: {self.seq_length}")
        print(f"Number of Disease Classes: {self.num_diseases}")
        print(f"\nDisease Classes:")
        for idx, name in enumerate(self.disease_names):
            print(f"  [{idx}] {name}")

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("="*80 + "\n")


class SeverityEstimator(nn.Module):
    """
    Disease Severity Estimation Model.
    Estimates disease severity as a percentage (0-100%).

    Architecture: CNN + BiLSTM + Fully Connected
    """

    def __init__(self,
                 input_features: int = 38,
                 seq_length: int = 128,
                 conv_filters: list = [64, 128, 256],
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 2,
                 fc_hidden_sizes: list = [256, 128],
                 dropout: float = 0.3):
        """
        Initialize severity estimator.

        Args:
            input_features: Number of input features
            seq_length: Sequence length
            conv_filters: Conv filter sizes
            lstm_hidden_size: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            fc_hidden_sizes: FC layer sizes
            dropout: Dropout probability
        """
        super(SeverityEstimator, self).__init__()

        self.input_features = input_features
        self.seq_length = seq_length

        # ==================== CNN Layers ====================
        self.conv_layers = nn.ModuleList()
        in_channels = input_features

        for filters in conv_filters:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, filters, kernel_size=5, padding=2),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            in_channels = filters

        # ==================== BiLSTM Layers ====================
        self.bilstm = nn.LSTM(
            input_size=conv_filters[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )

        # ==================== Fully Connected Layers ====================
        fc_layers = []
        in_features = lstm_hidden_size * 2  # Bidirectional

        for hidden_size in fc_hidden_sizes:
            fc_layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_features = hidden_size

        self.fc_layers = nn.Sequential(*fc_layers)

        # Output layer for severity (0-1, will be converted to 0-100%)
        self.output_layer = nn.Sequential(
            nn.Linear(fc_hidden_sizes[-1], 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_length, input_features)

        Returns:
            Severity scores (batch_size, 1) in range [0, 1]
        """
        # Transpose for Conv1d
        x = x.transpose(1, 2)

        # CNN feature extraction
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Transpose back for LSTM
        x = x.transpose(1, 2)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)

        # Take last time step
        x = lstm_out[:, -1, :]

        # FC layers
        x = self.fc_layers(x)

        # Output severity
        severity = self.output_layer(x)

        return severity

    def predict_severity_percentage(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict severity as percentage.

        Args:
            x: Input tensor

        Returns:
            Severity percentages (0-100)
        """
        self.eval()
        with torch.no_grad():
            severity_normalized = self.forward(x)
            severity_percentage = severity_normalized * 100.0

        return severity_percentage

    def print_model_summary(self):
        """Print model architecture summary."""
        print("\n" + "="*80)
        print("DISEASE SEVERITY ESTIMATOR - MODEL SUMMARY")
        print("="*80)
        print(f"Input Features: {self.input_features}")
        print(f"Sequence Length: {self.seq_length}")
        print(f"Output: Severity Percentage (0-100%)")

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("="*80 + "\n")


if __name__ == "__main__":
    print("Testing Neurological Disease Classifier...")

    # Test Disease Classifier
    print("\n" + "="*80)
    print("1. DISEASE CLASSIFIER TEST")
    print("="*80)

    classifier = DiseaseClassifier(
        input_features=38,
        seq_length=128,
        num_diseases=6
    )
    classifier.print_model_summary()

    # Test forward pass
    batch_size = 16
    test_input = torch.randn(batch_size, 128, 38)

    print("Testing forward pass...")
    output = classifier(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output probabilities sum: {output.sum(dim=1)[0]:.4f} (should be ~1.0)")

    # Test prediction
    predicted_classes, probs = classifier.predict(test_input)
    print(f"\nSample prediction:")
    print(f"  Predicted class: {predicted_classes[0].item()}")
    print(f"  Disease: {classifier.get_disease_name(predicted_classes[0].item())}")
    print(f"  Confidence: {probs[0, predicted_classes[0]].item():.2%}")

    # Test Severity Estimator
    print("\n" + "="*80)
    print("2. SEVERITY ESTIMATOR TEST")
    print("="*80)

    severity_model = SeverityEstimator(
        input_features=38,
        seq_length=128
    )
    severity_model.print_model_summary()

    print("Testing forward pass...")
    severity_output = severity_model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Severity output shape: {severity_output.shape}")
    print(f"Severity range: [{severity_output.min():.3f}, {severity_output.max():.3f}]")

    # Test severity prediction
    severity_percentage = severity_model.predict_severity_percentage(test_input)
    print(f"\nSample severity predictions:")
    for i in range(min(5, batch_size)):
        print(f"  Sample {i+1}: {severity_percentage[i].item():.1f}%")

    print("\n" + "="*80)
    print("All tests completed successfully!")
    print("="*80)
