# training/__init__.py
"""
Training Module for Gait Detection System

Provides training, validation, and metrics calculation functionality.

Example usage:
    from training import GaitDetectorTrainer, ModelValidator, MetricsCalculator
    from training import create_data_loaders, calculate_class_weights
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=32
    )
    
    # Setup trainer
    trainer = GaitDetectorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )
    
    # Train
    trainer.train(num_epochs=50)
    
    # Validate
    validator = ModelValidator(model, device)
    results = validator.test(X_test, y_test)
"""

from .trainer import GaitDetectorTrainer, create_data_loaders
from .validator import ModelValidator
from .metrics import MetricsCalculator, calculate_class_weights

__all__ = [
    'GaitDetectorTrainer',
    'create_data_loaders',
    'ModelValidator',
    'MetricsCalculator',
    'calculate_class_weights'
]

__version__ = '1.0.0'
