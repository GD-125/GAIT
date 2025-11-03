"""
Data Loader Module for Gait Detection System
Handles loading and parsing of CSV sensor data files.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import glob
from pathlib import Path


class GaitDataLoader:
    """
    Loads and parses CSV files containing sensor data for gait detection.
    
    Expected columns:
    - Accelerometer: right/left foot, shin, thigh (x, y, z)
    - Gyroscope: right/left foot, shin, thigh (x, y, z)
    - EMG: right, left
    - Activity: label column
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.sensor_columns = self._define_sensor_columns()
        
    def _define_sensor_columns(self) -> Dict[str, List[str]]:
        """Define all sensor column names based on dataset format."""
        
        # Accelerometer columns
        accel_cols = []
        for location in ['right_foot', 'right_shin', 'right_thigh', 
                        'left_foot', 'left_shin', 'left_thigh']:
            for axis in ['x', 'y', 'z']:
                accel_cols.append(f'accelerometer_{location}_{axis}')
        
        # Gyroscope columns
        gyro_cols = []
        for location in ['right_foot', 'right_shin', 'right_thigh',
                        'left_foot', 'left_shin', 'left_thigh']:
            for axis in ['x', 'y', 'z']:
                gyro_cols.append(f'gyroscope_{location}_{axis}')
        
        # EMG columns
        emg_cols = ['EMG_right', 'EMG_left']
        
        # All feature columns
        all_features = accel_cols + gyro_cols + emg_cols
        
        return {
            'accelerometer': accel_cols,
            'gyroscope': gyro_cols,
            'emg': emg_cols,
            'all_features': all_features,
            'label': 'activity'
        }
    
    def load_single_file(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        try:
            # Read CSV file
            df = pd.read_csv(filepath)
            
            # Remove unnamed index column if present
            if 'Unnamed: 0' in df.columns or '' in df.columns:
                df = df.drop(columns=['Unnamed: 0', ''], errors='ignore')
            
            # Extract features (all sensor data)
            features = df[self.sensor_columns['all_features']].values
            
            # Extract labels
            labels = df[self.sensor_columns['label']].values
            
            print(f"✓ Loaded {filepath}: {features.shape[0]} samples, {features.shape[1]} features")
            
            return features, labels
            
        except Exception as e:
            print(f"✗ Error loading {filepath}: {str(e)}")
            raise
    
    def load_multiple_files(self, file_pattern: str = "*.csv") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load multiple CSV files from the data directory.
        
        Args:
            file_pattern: Glob pattern for file matching (default: "*.csv")
            
        Returns:
            Tuple of (all_features, all_labels, filenames)
        """
        # Find all CSV files
        file_paths = sorted(glob.glob(str(self.data_dir / file_pattern)))
        
        if not file_paths:
            raise FileNotFoundError(f"No files found matching {file_pattern} in {self.data_dir}")
        
        print(f"\n{'='*60}")
        print(f"Loading {len(file_paths)} files from {self.data_dir}")
        print(f"{'='*60}\n")
        
        all_features = []
        all_labels = []
        filenames = []
        
        for filepath in file_paths:
            features, labels = self.load_single_file(filepath)
            all_features.append(features)
            all_labels.append(labels)
            filenames.append(os.path.basename(filepath))
        
        # Concatenate all data
        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels)
        
        print(f"\n{'='*60}")
        print(f"Total loaded: {all_features.shape[0]} samples, {all_features.shape[1]} features")
        print(f"{'='*60}\n")
        
        return all_features, all_labels, filenames
    
    def get_dataset_info(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Get summary statistics about the loaded dataset.
        
        Args:
            features: Feature array
            labels: Label array
            
        Returns:
            Dictionary with dataset information
        """
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        
        info = {
            'num_samples': features.shape[0],
            'num_features': features.shape[1],
            'num_accelerometer_features': len(self.sensor_columns['accelerometer']),
            'num_gyroscope_features': len(self.sensor_columns['gyroscope']),
            'num_emg_features': len(self.sensor_columns['emg']),
            'unique_activities': unique_labels.tolist(),
            'activity_counts': dict(zip(unique_labels, label_counts)),
            'feature_columns': self.sensor_columns['all_features']
        }
        
        return info
    
    def print_dataset_info(self, features: np.ndarray, labels: np.ndarray):
        """Print formatted dataset information."""
        info = self.get_dataset_info(features, labels)
        
        print(f"\n{'='*60}")
        print("DATASET INFORMATION")
        print(f"{'='*60}")
        print(f"Total Samples: {info['num_samples']}")
        print(f"Total Features: {info['num_features']}")
        print(f"  - Accelerometer: {info['num_accelerometer_features']} features")
        print(f"  - Gyroscope: {info['num_gyroscope_features']} features")
        print(f"  - EMG: {info['num_emg_features']} features")
        print(f"\nActivity Distribution:")
        for activity, count in info['activity_counts'].items():
            percentage = (count / info['num_samples']) * 100
            print(f"  - {activity}: {count} samples ({percentage:.1f}%)")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    loader = GaitDataLoader(data_dir="../data/raw")
    
    # Load single file
    features, labels = loader.load_single_file("../data/raw/HuGaDB_v2_various_01_00.csv")
    
    # Print dataset info
    loader.print_dataset_info(features, labels)
