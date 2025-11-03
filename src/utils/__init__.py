# utils/__init__.py
"""
Utilities Module for Gait Detection System

Provides configuration management and visualization utilities.

Example usage:
    from utils import Config, get_default_config, Visualizer
    
    # Configuration
    config = get_default_config()
    config.print_config()
    config.save_yaml('configs/config.yaml')
    
    # Visualization
    visualizer = Visualizer(save_dir='results/plots')
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(cm)
"""

from .config import (
    Config,
    PreprocessingConfig,
    ModelConfig,
    TrainingConfig,
    PathConfig,
    ExplainabilityConfig,
    get_default_config
)
from .visualization import Visualizer

__all__ = [
    'Config',
    'PreprocessingConfig',
    'ModelConfig',
    'TrainingConfig',
    'PathConfig',
    'ExplainabilityConfig',
    'get_default_config',
    'Visualizer'
]

__version__ = '1.0.0'
