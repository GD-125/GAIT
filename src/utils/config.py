# utils/config.py
"""
Configuration Management Module for Gait Detection System
Centralizes all hyperparameters and configuration settings.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    sampling_rate: float = 100.0
    window_size: int = 128
    overlap: float = 0.5
    normalization_method: str = 'zscore'  # 'zscore', 'minmax', 'robust'
    filter_type: str = 'sensor_specific'  # 'sensor_specific', 'bandpass', 'lowpass'
    balance_method: str = 'undersample'  # 'undersample', 'oversample', 'none'
    outlier_threshold: float = 3.0
    missing_value_method: str = 'interpolate'  # 'interpolate', 'forward_fill', 'mean'


@dataclass
class ModelConfig:
    """Configuration for CNN-BiLSTM model architecture."""
    input_features: int = 38
    seq_length: int = 128
    conv_filters: list = None
    kernel_sizes: list = None
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    fc_hidden_sizes: list = None
    dropout: float = 0.3
    use_batch_norm: bool = True
    use_residual: bool = True
    
    def __post_init__(self):
        if self.conv_filters is None:
            self.conv_filters = [64, 128, 256]
        if self.kernel_sizes is None:
            self.kernel_sizes = [5, 5, 5]
        if self.fc_hidden_sizes is None:
            self.fc_hidden_sizes = [256, 128]


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    scheduler: str = 'reduce_on_plateau'  # 'reduce_on_plateau', 'cosine', 'step', 'none'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 15
    gradient_clip_max_norm: float = 1.0
    use_class_weights: bool = False
    train_val_split: float = 0.2
    random_seed: int = 42


@dataclass
class PathConfig:
    """Configuration for file paths."""
    data_raw_dir: str = 'data/raw'
    data_processed_dir: str = 'data/processed'
    data_splits_dir: str = 'data/splits'
    checkpoint_dir: str = 'checkpoints'
    results_dir: str = 'results'
    logs_dir: str = 'results/logs'
    plots_dir: str = 'results/plots'
    shap_dir: str = 'results/shap_outputs'


@dataclass
class ExplainabilityConfig:
    """Configuration for SHAP explainability."""
    max_background_samples: int = 100
    num_test_samples: int = 20
    top_k_features: int = 20
    generate_force_plots: bool = True
    num_force_plot_samples: int = 3


class Config:
    """
    Main configuration class that combines all sub-configurations.
    """
    
    def __init__(self,
                 preprocessing: Optional[PreprocessingConfig] = None,
                 model: Optional[ModelConfig] = None,
                 training: Optional[TrainingConfig] = None,
                 paths: Optional[PathConfig] = None,
                 explainability: Optional[ExplainabilityConfig] = None):
        """
        Initialize configuration.
        
        Args:
            preprocessing: Preprocessing configuration
            model: Model architecture configuration
            training: Training configuration
            paths: Path configuration
            explainability: Explainability configuration
        """
        self.preprocessing = preprocessing or PreprocessingConfig()
        self.model = model or ModelConfig()
        self.training = training or TrainingConfig()
        self.paths = paths or PathConfig()
        self.explainability = explainability or ExplainabilityConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'preprocessing': asdict(self.preprocessing),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'paths': asdict(self.paths),
            'explainability': asdict(self.explainability)
        }
    
    def save_yaml(self, filepath: str):
        """
        Save configuration to YAML file.
        
        Args:
            filepath: Path to save YAML file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        
        print(f"✓ Configuration saved to {filepath}")
    
    def save_json(self, filepath: str):
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        
        print(f"✓ Configuration saved to {filepath}")
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            Config object
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            preprocessing=PreprocessingConfig(**config_dict.get('preprocessing', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            paths=PathConfig(**config_dict.get('paths', {})),
            explainability=ExplainabilityConfig(**config_dict.get('explainability', {}))
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> 'Config':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Config object
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            preprocessing=PreprocessingConfig(**config_dict.get('preprocessing', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            paths=PathConfig(**config_dict.get('paths', {})),
            explainability=ExplainabilityConfig(**config_dict.get('explainability', {}))
        )
    
    def print_config(self):
        """Print formatted configuration."""
        print(f"\n{'='*70}")
        print("CONFIGURATION SUMMARY")
        print(f"{'='*70}\n")
        
        print("--- Preprocessing ---")
        for key, value in asdict(self.preprocessing).items():
            print(f"  {key}: {value}")
        
        print("\n--- Model Architecture ---")
        for key, value in asdict(self.model).items():
            print(f"  {key}: {value}")
        
        print("\n--- Training ---")
        for key, value in asdict(self.training).items():
            print(f"  {key}: {value}")
        
        print("\n--- Paths ---")
        for key, value in asdict(self.paths).items():
            print(f"  {key}: {value}")
        
        print("\n--- Explainability ---")
        for key, value in asdict(self.explainability).items():
            print(f"  {key}: {value}")
        
        print(f"\n{'='*70}\n")
    
    def create_directories(self):
        """Create all necessary directories."""
        dirs_to_create = [
            self.paths.data_raw_dir,
            self.paths.data_processed_dir,
            self.paths.data_splits_dir,
            self.paths.checkpoint_dir,
            self.paths.results_dir,
            self.paths.logs_dir,
            self.paths.plots_dir,
            self.paths.shap_dir
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print("✓ All directories created successfully")


def get_default_config() -> Config:
    """
    Get default configuration for the gait detection system.
    
    Returns:
        Default Config object
    """
    return Config(
        preprocessing=PreprocessingConfig(
            sampling_rate=100.0,
            window_size=128,
            overlap=0.5,
            normalization_method='zscore',
            filter_type='sensor_specific',
            balance_method='undersample'
        ),
        model=ModelConfig(
            input_features=38,
            seq_length=128,
            conv_filters=[64, 128, 256],
            kernel_sizes=[5, 5, 5],
            lstm_hidden_size=128,
            lstm_num_layers=2,
            fc_hidden_sizes=[256, 128],
            dropout=0.3
        ),
        training=TrainingConfig(
            num_epochs=50,
            batch_size=32,
            learning_rate=0.001,
            early_stopping_patience=15,
            random_seed=42
        ),
        paths=PathConfig(),
        explainability=ExplainabilityConfig()
    )


if __name__ == "__main__":
    # Example usage
    print("Configuration Management Example\n")
    
    # Get default configuration
    config = get_default_config()
    
    # Print configuration
    config.print_config()
    
    # Save to YAML
    config.save_yaml('configs/config.yaml')
    
    # Save to JSON
    config.save_json('configs/config.json')
    
    # Create directories
    config.create_directories()
    
    # Load from YAML
    loaded_config = Config.from_yaml('configs/config.yaml')
    print("✓ Configuration loaded successfully")
    
    # Modify and save
    loaded_config.training.num_epochs = 100
    loaded_config.training.batch_size = 64
    loaded_config.save_yaml('configs/config_modified.yaml')
