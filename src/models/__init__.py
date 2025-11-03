# models/__init__.py
"""
Models Module for Gait Detection System

Provides CNN-BiLSTM architecture and utility functions for model management.

Example usage:
    from models import CNN_BiLSTM_GaitDetector, save_checkpoint, load_checkpoint
    
    # Create model
    model = CNN_BiLSTM_GaitDetector(
        input_features=38,
        seq_length=128,
        conv_filters=[64, 128, 256],
        lstm_hidden_size=128
    )
    
    # Train model...
    
    # Save checkpoint
    save_checkpoint(model, optimizer, epoch, loss, accuracy, 'checkpoints/model.pt')
"""

from .cnn_bilstm import CNN_BiLSTM_GaitDetector
from .model_utils import (
    save_checkpoint,
    load_checkpoint,
    save_model_weights_only,
    load_model_weights_only,
    count_parameters,
    freeze_layers,
    unfreeze_layers,
    get_model_size,
    export_to_onnx,
    print_model_info,
    save_training_config,
    load_training_config,
    set_seed,
    get_device,
    move_to_device,
    EarlyStopping
)

__all__ = [
    'CNN_BiLSTM_GaitDetector',
    'save_checkpoint',
    'load_checkpoint',
    'save_model_weights_only',
    'load_model_weights_only',
    'count_parameters',
    'freeze_layers',
    'unfreeze_layers',
    'get_model_size',
    'export_to_onnx',
    'print_model_info',
    'save_training_config',
    'load_training_config',
    'set_seed',
    'get_device',
    'move_to_device',
    'EarlyStopping'
]

__version__ = '1.0.0'
