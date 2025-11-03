# models/model_utils.py
"""
Model Utility Functions for Gait Detection System
Helper functions for model operations, checkpointing, and management.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from datetime import datetime


def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   accuracy: float,
                   filepath: str,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   additional_info: Optional[Dict] = None):
    """
    Save model checkpoint with training state.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        filepath: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        additional_info: Any additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str,
                   model: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   device: str = 'cpu') -> Dict:
    """
    Load model checkpoint and restore training state.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load model onto
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✓ Checkpoint loaded from {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    print(f"  Accuracy: {checkpoint.get('accuracy', 'N/A'):.4f}")
    
    return checkpoint


def save_model_weights_only(model: nn.Module, filepath: str):
    """
    Save only model weights (lighter than full checkpoint).
    
    Args:
        model: PyTorch model
        filepath: Path to save weights
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), filepath)
    print(f"✓ Model weights saved to {filepath}")


def load_model_weights_only(model: nn.Module, filepath: str, device: str = 'cpu'):
    """
    Load only model weights.
    
    Args:
        model: PyTorch model
        filepath: Path to weights file
        device: Device to load onto
    """
    model.load_state_dict(torch.load(filepath, map_location=device, weights_only=False))
    print(f"✓ Model weights loaded from {filepath}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def freeze_layers(model: nn.Module, layer_names: list):
    """
    Freeze specific layers in the model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False
            print(f"  Froze: {name}")


def unfreeze_layers(model: nn.Module, layer_names: list):
    """
    Unfreeze specific layers in the model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True
            print(f"  Unfroze: {name}")


def get_model_size(model: nn.Module) -> float:
    """
    Calculate model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in megabytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def export_to_onnx(model: nn.Module,
                   filepath: str,
                   input_shape: Tuple[int, ...],
                   device: str = 'cpu'):
    """
    Export model to ONNX format for deployment.
    
    Args:
        model: PyTorch model
        filepath: Path to save ONNX model
        input_shape: Input shape (batch_size, seq_length, features)
        device: Device model is on
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Model exported to ONNX: {filepath}")


def print_model_info(model: nn.Module, input_shape: Optional[Tuple] = None):
    """
    Print comprehensive model information.
    
    Args:
        model: PyTorch model
        input_shape: Optional input shape for testing
    """
    print(f"\n{'='*70}")
    print("MODEL INFORMATION")
    print(f"{'='*70}")
    
    # Parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Model size
    size_mb = get_model_size(model)
    print(f"Model Size: {size_mb:.2f} MB")
    
    # Test forward pass if input shape provided
    if input_shape is not None:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(input_shape)
            try:
                output = model(dummy_input)
                print(f"\nTest Forward Pass:")
                print(f"  Input shape: {dummy_input.shape}")
                print(f"  Output shape: {output.shape}")
                print(f"  ✓ Forward pass successful")
            except Exception as e:
                print(f"  ✗ Forward pass failed: {str(e)}")
    
    print(f"{'='*70}\n")


def save_training_config(config: Dict, filepath: str):
    """
    Save training configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save config
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✓ Training config saved to {filepath}")


def load_training_config(filepath: str) -> Dict:
    """
    Load training configuration from JSON file.
    
    Args:
        filepath: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    print(f"✓ Training config loaded from {filepath}")
    return config


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set to {seed}")


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        torch.device object
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"✓ Using device: {device}")
    
    return device


def move_to_device(data, device: torch.device):
    """
    Move data to specified device (handles nested structures).
    
    Args:
        data: Data to move (tensor, list, tuple, or dict)
        device: Target device
        
    Returns:
        Data on the specified device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    else:
        return data


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training when validation loss doesn't improve for patience epochs.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, current_value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⚠ Early stopping triggered after {self.counter} epochs without improvement")
                return True
        
        return False


if __name__ == "__main__":
    # Example usage
    from cnn_bilstm import CNN_BiLSTM_GaitDetector
    
    # Create model
    model = CNN_BiLSTM_GaitDetector(
        input_features=38,
        seq_length=128
    )
    
    # Print model info
    print_model_info(model, input_shape=(16, 128, 38))
    
    # Set device
    device = get_device()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Early stopping example
    early_stopping = EarlyStopping(patience=5, mode='min')
    
    print("\nTesting early stopping...")
    losses = [0.5, 0.4, 0.35, 0.34, 0.34, 0.33, 0.33, 0.33, 0.33, 0.33]
    for epoch, loss in enumerate(losses):
        if early_stopping(loss):
            print(f"  Would stop at epoch {epoch}")
            break
        print(f"  Epoch {epoch}: loss={loss:.3f}, counter={early_stopping.counter}")
