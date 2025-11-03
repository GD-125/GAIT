"""
Data Normalization Module for Gait Detection System
Handles normalization and standardization of sensor data.
"""

import numpy as np
import pickle
from typing import Tuple, Optional, Dict
from pathlib import Path


class DataNormalizer:
    """
    Normalizes sensor data using various methods (Z-score, Min-Max, Robust).
    Stores normalization parameters for consistent inference.
    """
    
    def __init__(self, method: str = 'zscore'):
        """
        Initialize the data normalizer.
        
        Args:
            method: Normalization method ('zscore', 'minmax', 'robust')
        """
        self.method = method
        self.params = {}
        self.is_fitted = False
        
    def fit_zscore(self, data: np.ndarray) -> Dict:
        """
        Calculate Z-score normalization parameters.
        
        Args:
            data: Input data array (samples x features)
            
        Returns:
            Dictionary with mean and std for each feature
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        # Avoid division by zero
        std[std == 0] = 1.0
        
        return {'mean': mean, 'std': std}
    
    def fit_minmax(self, data: np.ndarray) -> Dict:
        """
        Calculate Min-Max normalization parameters.
        
        Args:
            data: Input data array (samples x features)
            
        Returns:
            Dictionary with min and max for each feature
        """
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        
        # Avoid division by zero
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        
        return {'min': min_val, 'max': max_val, 'range': range_val}
    
    def fit_robust(self, data: np.ndarray) -> Dict:
        """
        Calculate Robust normalization parameters (using median and IQR).
        Less sensitive to outliers than Z-score.
        
        Args:
            data: Input data array (samples x features)
            
        Returns:
            Dictionary with median and IQR for each feature
        """
        median = np.median(data, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        iqr = q75 - q25
        
        # Avoid division by zero
        iqr[iqr == 0] = 1.0
        
        return {'median': median, 'iqr': iqr, 'q25': q25, 'q75': q75}
    
    def fit(self, data: np.ndarray) -> 'DataNormalizer':
        """
        Fit the normalizer to the data (calculate normalization parameters).
        
        Args:
            data: Input data array (samples x features)
            
        Returns:
            Self for method chaining
        """
        print(f"\n{'='*60}")
        print(f"FITTING {self.method.upper()} NORMALIZER")
        print(f"{'='*60}")
        print(f"Data shape: {data.shape}")
        
        if self.method == 'zscore':
            self.params = self.fit_zscore(data)
            print(f"Mean range: [{self.params['mean'].min():.2f}, {self.params['mean'].max():.2f}]")
            print(f"Std range: [{self.params['std'].min():.2f}, {self.params['std'].max():.2f}]")
            
        elif self.method == 'minmax':
            self.params = self.fit_minmax(data)
            print(f"Min range: [{self.params['min'].min():.2f}, {self.params['min'].max():.2f}]")
            print(f"Max range: [{self.params['max'].min():.2f}, {self.params['max'].max():.2f}]")
            
        elif self.method == 'robust':
            self.params = self.fit_robust(data)
            print(f"Median range: [{self.params['median'].min():.2f}, {self.params['median'].max():.2f}]")
            print(f"IQR range: [{self.params['iqr'].min():.2f}, {self.params['iqr'].max():.2f}]")
            
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        self.is_fitted = True
        print(f"✓ Normalizer fitted successfully!")
        print(f"{'='*60}\n")
        
        return self
    
    def transform_zscore(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Z-score normalization: (X - mean) / std
        
        Args:
            data: Input data array
            
        Returns:
            Normalized data array
        """
        normalized = (data - self.params['mean']) / self.params['std']
        return normalized
    
    def transform_minmax(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Min-Max normalization: (X - min) / (max - min)
        Scales data to [0, 1] range.
        
        Args:
            data: Input data array
            
        Returns:
            Normalized data array
        """
        normalized = (data - self.params['min']) / self.params['range']
        return normalized
    
    def transform_robust(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Robust normalization: (X - median) / IQR
        
        Args:
            data: Input data array
            
        Returns:
            Normalized data array
        """
        normalized = (data - self.params['median']) / self.params['iqr']
        return normalized
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted normalization parameters.
        
        Args:
            data: Input data array (samples x features)
            
        Returns:
            Normalized data array
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform. Call fit() first.")
        
        print(f"Applying {self.method} normalization...")
        
        if self.method == 'zscore':
            normalized = self.transform_zscore(data)
        elif self.method == 'minmax':
            normalized = self.transform_minmax(data)
        elif self.method == 'robust':
            normalized = self.transform_robust(data)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        print(f"✓ Data normalized successfully!")
        print(f"  Range: [{normalized.min():.2f}, {normalized.max():.2f}]")
        
        return normalized
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the normalizer and transform data in one step.
        
        Args:
            data: Input data array (samples x features)
            
        Returns:
            Normalized data array
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Reverse the normalization to get original scale.
        
        Args:
            data: Normalized data array
            
        Returns:
            Data in original scale
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse transform.")
        
        if self.method == 'zscore':
            original = (data * self.params['std']) + self.params['mean']
        elif self.method == 'minmax':
            original = (data * self.params['range']) + self.params['min']
        elif self.method == 'robust':
            original = (data * self.params['iqr']) + self.params['median']
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        return original
    
    def save(self, filepath: str):
        """
        Save normalization parameters to disk.
        
        Args:
            filepath: Path to save the parameters
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted normalizer.")
        
        save_dict = {
            'method': self.method,
            'params': self.params,
            'is_fitted': self.is_fitted
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"✓ Normalizer saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load normalization parameters from disk.
        
        Args:
            filepath: Path to load the parameters from
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.method = save_dict['method']
        self.params = save_dict['params']
        self.is_fitted = save_dict['is_fitted']
        
        print(f"✓ Normalizer loaded from {filepath}")
    
    def get_statistics(self, data: np.ndarray) -> Dict:
        """
        Get statistical summary of normalized data.
        
        Args:
            data: Normalized data array
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'median': np.median(data, axis=0),
            'q25': np.percentile(data, 25, axis=0),
            'q75': np.percentile(data, 75, axis=0)
        }
        return stats
    
    def print_statistics(self, data: np.ndarray, name: str = "Data"):
        """
        Print formatted statistics of the data.
        
        Args:
            data: Data array
            name: Name for display
        """
        stats = self.get_statistics(data)
        
        print(f"\n{'='*60}")
        print(f"{name.upper()} STATISTICS")
        print(f"{'='*60}")
        print(f"Mean: [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}]")
        print(f"Std:  [{stats['std'].min():.4f}, {stats['std'].max():.4f}]")
        print(f"Min:  [{stats['min'].min():.4f}, {stats['min'].max():.4f}]")
        print(f"Max:  [{stats['max'].min():.4f}, {stats['max'].max():.4f}]")
        print(f"Median: [{stats['median'].min():.4f}, {stats['median'].max():.4f}]")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    from data_loader import GaitDataLoader
    from cleaner import DataCleaner
    from filter import SignalFilter
    
    # Load, clean, and filter data
    loader = GaitDataLoader(data_dir="../data/raw")
    features, labels = loader.load_single_file("../data/raw/HuGaDB_v2_various_01_00.csv")
    
    cleaner = DataCleaner()
    features_cleaned, _ = cleaner.clean_data(features)
    
    filter_obj = SignalFilter(sampling_rate=100.0)
    features_filtered = filter_obj.apply_sensor_specific_filters(features_cleaned, 
                                                                   loader.sensor_columns)
    
    # Normalize data
    normalizer = DataNormalizer(method='zscore')
    
    # Print statistics before normalization
    normalizer.print_statistics(features_filtered, "Before Normalization")
    
    # Fit and transform
    features_normalized = normalizer.fit_transform(features_filtered)
    
    # Print statistics after normalization
    normalizer.print_statistics(features_normalized, "After Normalization")
    
    # Save normalizer for later use
    normalizer.save("../checkpoints/normalizer.pkl")
