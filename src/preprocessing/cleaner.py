"""
Data Cleaning Module for Gait Detection System
Handles missing values, outliers, and data quality checks.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from scipy import stats


class DataCleaner:
    """
    Cleans sensor data by handling missing values and outliers.
    """
    
    def __init__(self, 
                 missing_threshold: float = 0.1,
                 outlier_method: str = 'iqr',
                 outlier_threshold: float = 3.0):
        """
        Initialize the data cleaner.
        
        Args:
            missing_threshold: Maximum allowed proportion of missing values per feature
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
            outlier_threshold: Threshold for outlier detection (IQR multiplier or z-score)
        """
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.cleaning_stats = {}
    
    def check_missing_values(self, data: np.ndarray) -> Dict:
        """
        Check for missing values in the dataset.
        
        Args:
            data: Input data array (samples x features)
            
        Returns:
            Dictionary with missing value statistics
        """
        missing_mask = np.isnan(data) | np.isinf(data)
        missing_per_feature = missing_mask.sum(axis=0)
        missing_per_sample = missing_mask.sum(axis=1)
        
        stats = {
            'total_missing': missing_mask.sum(),
            'missing_percentage': (missing_mask.sum() / data.size) * 100,
            'features_with_missing': (missing_per_feature > 0).sum(),
            'samples_with_missing': (missing_per_sample > 0).sum(),
            'missing_per_feature': missing_per_feature,
            'missing_per_sample': missing_per_sample
        }
        
        return stats
    
    def handle_missing_values(self, 
                             data: np.ndarray,
                             method: str = 'interpolate') -> np.ndarray:
        """
        Handle missing values using specified method.
        
        Args:
            data: Input data array (samples x features)
            method: Imputation method ('interpolate', 'forward_fill', 'mean', 'median', 'remove')
            
        Returns:
            Cleaned data array
        """
        # Check for missing values
        missing_stats = self.check_missing_values(data)
        
        if missing_stats['total_missing'] == 0:
            print("✓ No missing values found")
            return data
        
        print(f"⚠ Found {missing_stats['total_missing']} missing values "
              f"({missing_stats['missing_percentage']:.2f}%)")
        
        data_cleaned = data.copy()
        
        if method == 'interpolate':
            # Linear interpolation for time-series data
            df = pd.DataFrame(data)
            df = df.interpolate(method='linear', axis=0, limit_direction='both')
            data_cleaned = df.values
            
        elif method == 'forward_fill':
            # Forward fill (carry last observation forward)
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill').fillna(method='bfill')
            data_cleaned = df.values
            
        elif method == 'mean':
            # Mean imputation
            col_means = np.nanmean(data, axis=0)
            for i in range(data.shape[1]):
                data_cleaned[np.isnan(data[:, i]), i] = col_means[i]
                
        elif method == 'median':
            # Median imputation
            col_medians = np.nanmedian(data, axis=0)
            for i in range(data.shape[1]):
                data_cleaned[np.isnan(data[:, i]), i] = col_medians[i]
                
        elif method == 'remove':
            # Remove samples with missing values
            valid_samples = ~np.isnan(data).any(axis=1)
            data_cleaned = data[valid_samples]
            print(f"  Removed {(~valid_samples).sum()} samples with missing values")
            
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        
        print(f"✓ Missing values handled using '{method}' method")
        
        return data_cleaned
    
    def detect_outliers_iqr(self, data: np.ndarray) -> np.ndarray:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            data: Input data array (samples x features)
            
        Returns:
            Boolean mask indicating outliers (True = outlier)
        """
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        
        return outlier_mask
    
    def detect_outliers_zscore(self, data: np.ndarray) -> np.ndarray:
        """
        Detect outliers using Z-score method.
        
        Args:
            data: Input data array (samples x features)
            
        Returns:
            Boolean mask indicating outliers (True = outlier)
        """
        z_scores = np.abs(stats.zscore(data, axis=0, nan_policy='omit'))
        outlier_mask = z_scores > self.outlier_threshold
        
        return outlier_mask
    
    def handle_outliers(self, 
                       data: np.ndarray,
                       method: str = 'clip') -> np.ndarray:
        """
        Handle outliers using specified method.
        
        Args:
            data: Input data array (samples x features)
            method: Handling method ('clip', 'remove', 'winsorize', 'none')
            
        Returns:
            Cleaned data array
        """
        if method == 'none':
            print("✓ Skipping outlier handling")
            return data
        
        # Detect outliers
        if self.outlier_method == 'iqr':
            outlier_mask = self.detect_outliers_iqr(data)
        elif self.outlier_method == 'zscore':
            outlier_mask = self.detect_outliers_zscore(data)
        else:
            raise ValueError(f"Unknown outlier detection method: {self.outlier_method}")
        
        num_outliers = outlier_mask.sum()
        outlier_percentage = (num_outliers / data.size) * 100
        
        print(f"⚠ Detected {num_outliers} outliers ({outlier_percentage:.2f}%) "
              f"using {self.outlier_method} method")
        
        data_cleaned = data.copy()
        
        if method == 'clip':
            # Clip outliers to acceptable range
            if self.outlier_method == 'iqr':
                Q1 = np.percentile(data, 25, axis=0)
                Q3 = np.percentile(data, 75, axis=0)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR
                data_cleaned = np.clip(data, lower_bound, upper_bound)
            else:  # zscore
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                lower_bound = mean - self.outlier_threshold * std
                upper_bound = mean + self.outlier_threshold * std
                data_cleaned = np.clip(data, lower_bound, upper_bound)
            print(f"✓ Outliers clipped to acceptable range")
            
        elif method == 'remove':
            # Remove samples with any outlier
            samples_with_outliers = outlier_mask.any(axis=1)
            data_cleaned = data[~samples_with_outliers]
            print(f"  Removed {samples_with_outliers.sum()} samples with outliers")
            
        elif method == 'winsorize':
            # Winsorize: replace outliers with boundary values
            for i in range(data.shape[1]):
                col_data = data[:, i]
                p5, p95 = np.percentile(col_data, [5, 95])
                data_cleaned[:, i] = np.clip(col_data, p5, p95)
            print(f"✓ Outliers winsorized to 5th-95th percentile range")
            
        else:
            raise ValueError(f"Unknown outlier handling method: {method}")
        
        return data_cleaned
    
    def clean_data(self, 
                   data: np.ndarray,
                   missing_method: str = 'interpolate',
                   outlier_method: str = 'clip') -> Tuple[np.ndarray, Dict]:
        """
        Complete data cleaning pipeline.
        
        Args:
            data: Input data array (samples x features)
            missing_method: Method for handling missing values
            outlier_method: Method for handling outliers
            
        Returns:
            Tuple of (cleaned_data, cleaning_statistics)
        """
        print(f"\n{'='*60}")
        print("DATA CLEANING PIPELINE")
        print(f"{'='*60}")
        print(f"Input shape: {data.shape}")
        
        # Step 1: Handle missing values
        print("\n[1/2] Handling Missing Values...")
        data_cleaned = self.handle_missing_values(data, method=missing_method)
        
        # Step 2: Handle outliers
        print("\n[2/2] Handling Outliers...")
        data_cleaned = self.handle_outliers(data_cleaned, method=outlier_method)
        
        # Collect statistics
        self.cleaning_stats = {
            'original_shape': data.shape,
            'cleaned_shape': data_cleaned.shape,
            'samples_removed': data.shape[0] - data_cleaned.shape[0],
            'missing_method': missing_method,
            'outlier_method': outlier_method
        }
        
        print(f"\n{'='*60}")
        print(f"Cleaning Complete!")
        print(f"Original shape: {data.shape}")
        print(f"Cleaned shape: {data_cleaned.shape}")
        print(f"Samples removed: {self.cleaning_stats['samples_removed']}")
        print(f"{'='*60}\n")
        
        return data_cleaned, self.cleaning_stats


if __name__ == "__main__":
    # Example usage
    from data_loader import GaitDataLoader
    
    # Load data
    loader = GaitDataLoader(data_dir="../data/raw")
    features, labels = loader.load_single_file("../data/raw/HuGaDB_v2_various_01_00.csv")
    
    # Clean data
    cleaner = DataCleaner(outlier_method='iqr', outlier_threshold=3.0)
    features_cleaned, stats = cleaner.clean_data(features, 
                                                  missing_method='interpolate',
                                                  outlier_method='clip')
