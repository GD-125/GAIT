"""
Complete Preprocessing Pipeline for Gait Detection System
Integrates all preprocessing steps into a single, easy-to-use pipeline.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from .data_loader import GaitDataLoader
from .cleaner import DataCleaner
from .filter import SignalFilter
from .normalizer import DataNormalizer
from .segmentation import DataSegmenter


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline that chains all preprocessing steps.
    """
    
    def __init__(self,
                 sampling_rate: float = 100.0,
                 window_size: int = 128,
                 overlap: float = 0.5,
                 normalization_method: str = 'zscore',
                 filter_type: str = 'sensor_specific',
                 balance_method: str = 'undersample'):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            sampling_rate: Sampling frequency in Hz
            window_size: Number of samples per window
            overlap: Overlap ratio between windows (0.0 to 0.99)
            normalization_method: 'zscore', 'minmax', or 'robust'
            filter_type: 'sensor_specific', 'bandpass', or 'lowpass'
            balance_method: 'undersample', 'oversample', or 'none'
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        self.normalization_method = normalization_method
        self.filter_type = filter_type
        self.balance_method = balance_method
        
        # Initialize components
        self.loader = None
        self.cleaner = DataCleaner(outlier_method='iqr', outlier_threshold=3.0)
        self.filter = SignalFilter(sampling_rate=sampling_rate)
        self.normalizer = DataNormalizer(method=normalization_method)
        self.segmenter = DataSegmenter(window_size=window_size, 
                                      overlap=overlap, 
                                      sampling_rate=sampling_rate)
        
        self.is_fitted = False
        
    def preprocess_single_file(self, 
                               filepath: str,
                               data_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a single CSV file through the complete pipeline.
        
        Args:
            filepath: Path to CSV file
            data_dir: Directory containing the file (optional)
            
        Returns:
            Tuple of (windowed_data, binary_labels)
        """
        if data_dir is None:
            data_dir = str(Path(filepath).parent)
        
        # Initialize loader if needed
        if self.loader is None:
            self.loader = GaitDataLoader(data_dir)
        
        print(f"\n{'='*80}")
        print("PREPROCESSING PIPELINE - SINGLE FILE")
        print(f"{'='*80}\n")
        print(f"File: {filepath}")
        
        # Step 1: Load data
        print(f"\n[STEP 1/6] Loading data...")
        features, labels = self.loader.load_single_file(filepath)
        
        # Step 2: Clean data
        print(f"\n[STEP 2/6] Cleaning data...")
        features, _ = self.cleaner.clean_data(features, 
                                              missing_method='interpolate',
                                              outlier_method='clip')
        
        # Step 3: Filter signals
        print(f"\n[STEP 3/6] Filtering signals...")
        if self.filter_type == 'sensor_specific':
            features = self.filter.apply_sensor_specific_filters(features, 
                                                                 self.loader.sensor_columns)
        else:
            features = self.filter.apply_filter(features, method=self.filter_type)
        
        # Step 4: Normalize data
        print(f"\n[STEP 4/6] Normalizing data...")
        if not self.is_fitted:
            features = self.normalizer.fit_transform(features)
            self.is_fitted = True
        else:
            features = self.normalizer.transform(features)
        
        # Step 5: Create binary labels
        print(f"\n[STEP 5/6] Creating binary labels...")
        binary_labels = self.segmenter.create_binary_labels(labels)
        
        # Step 6: Segment into windows
        print(f"\n[STEP 6/6] Segmenting into windows...")
        windowed_data, windowed_labels, _ = self.segmenter.segment_data(features, 
                                                                         binary_labels)
        
        print(f"\n{'='*80}")
        print("PREPROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Output shape: {windowed_data.shape}")
        print(f"Labels shape: {windowed_labels.shape}")
        print(f"{'='*80}\n")
        
        return windowed_data, windowed_labels
    
    def preprocess_multiple_files(self,
                                  data_dir: str,
                                  file_pattern: str = "*.csv",
                                  max_files: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess multiple CSV files through the complete pipeline.
        
        Args:
            data_dir: Directory containing CSV files
            file_pattern: Glob pattern for file matching
            max_files: Maximum number of files to process (None = all)
            
        Returns:
            Tuple of (all_windowed_data, all_binary_labels)
        """
        self.loader = GaitDataLoader(data_dir)
        
        # Find all files
        import glob
        file_paths = sorted(glob.glob(str(Path(data_dir) / file_pattern)))
        
        if max_files is not None:
            file_paths = file_paths[:max_files]
        
        print(f"\n{'='*80}")
        print("PREPROCESSING PIPELINE - MULTIPLE FILES")
        print(f"{'='*80}")
        print(f"Processing {len(file_paths)} files from {data_dir}")
        print(f"{'='*80}\n")
        
        all_windowed_data = []
        all_windowed_labels = []
        
        for i, filepath in enumerate(file_paths):
            print(f"\n{'='*80}")
            print(f"FILE {i+1}/{len(file_paths)}: {Path(filepath).name}")
            print(f"{'='*80}")
            
            try:
                windowed_data, windowed_labels = self.preprocess_single_file(filepath, data_dir)
                all_windowed_data.append(windowed_data)
                all_windowed_labels.append(windowed_labels)
                print(f"✓ File {i+1} processed successfully")
            except Exception as e:
                print(f"✗ Error processing file {i+1}: {str(e)}")
                continue
        
        # Concatenate all data
        all_windowed_data = np.vstack(all_windowed_data)
        all_windowed_labels = np.concatenate(all_windowed_labels)
        
        # Balance dataset if requested
        if self.balance_method != 'none':
            print(f"\nBalancing dataset...")
            all_windowed_data, all_windowed_labels = self.segmenter.balance_dataset(
                all_windowed_data, 
                all_windowed_labels,
                method=self.balance_method
            )
        
        print(f"\n{'='*80}")
        print("ALL FILES PROCESSED")
        print(f"{'='*80}")
        print(f"Total windows: {len(all_windowed_data)}")
        print(f"Final shape: {all_windowed_data.shape}")
        print(f"Labels shape: {all_windowed_labels.shape}")
        
        # Print class distribution
        unique, counts = np.unique(all_windowed_labels, return_counts=True)
        print(f"\nFinal Class Distribution:")
        for label, count in zip(unique, counts):
            label_name = "Gait" if label == 1 else "Non-Gait"
            print(f"  {label_name} ({label}): {count} samples ({count/len(all_windowed_labels)*100:.1f}%)")
        print(f"{'='*80}\n")
        
        return all_windowed_data, all_windowed_labels
    
    def save_preprocessed_data(self,
                              windowed_data: np.ndarray,
                              windowed_labels: np.ndarray,
                              output_dir: str):
        """
        Save preprocessed data and pipeline parameters.
        
        Args:
            windowed_data: Windowed feature data
            windowed_labels: Binary labels
            output_dir: Directory to save outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving preprocessed data to {output_dir}...")
        
        # Save data
        np.save(output_path / 'windowed_data.npy', windowed_data)
        np.save(output_path / 'windowed_labels.npy', windowed_labels)
        
        # Save normalizer
        self.normalizer.save(output_path / 'normalizer.pkl')
        
        # Save pipeline configuration
        config = {
            'sampling_rate': self.sampling_rate,
            'window_size': self.window_size,
            'overlap': self.overlap,
            'normalization_method': self.normalization_method,
            'filter_type': self.filter_type,
            'balance_method': self.balance_method,
            'data_shape': windowed_data.shape,
            'num_features': windowed_data.shape[2]
        }
        
        with open(output_path / 'pipeline_config.pkl', 'wb') as f:
            pickle.dump(config, f)
        
        print(f"✓ Data saved successfully!")
        print(f"  - windowed_data.npy: {windowed_data.shape}")
        print(f"  - windowed_labels.npy: {windowed_labels.shape}")
        print(f"  - normalizer.pkl")
        print(f"  - pipeline_config.pkl")
    
    def load_preprocessed_data(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load preprocessed data and configuration.
        
        Args:
            data_dir: Directory containing preprocessed data
            
        Returns:
            Tuple of (windowed_data, windowed_labels, config)
        """
        data_path = Path(data_dir)
        
        print(f"\nLoading preprocessed data from {data_dir}...")
        
        # Load data
        windowed_data = np.load(data_path / 'windowed_data.npy')
        windowed_labels = np.load(data_path / 'windowed_labels.npy')
        
        # Load configuration
        with open(data_path / 'pipeline_config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        # Load normalizer
        self.normalizer.load(data_path / 'normalizer.pkl')
        self.is_fitted = True
        
        print(f"✓ Data loaded successfully!")
        print(f"  Data shape: {windowed_data.shape}")
        print(f"  Labels shape: {windowed_labels.shape}")
        
        return windowed_data, windowed_labels, config


if __name__ == "__main__":
    # Example 1: Process single file
    pipeline = PreprocessingPipeline(
        sampling_rate=100.0,
        window_size=128,
        overlap=0.5,
        normalization_method='zscore',
        filter_type='sensor_specific',
        balance_method='none'
    )
    
    windowed_data, windowed_labels = pipeline.preprocess_single_file(
        filepath="../data/raw/HuGaDB_v2_various_01_00.csv"
    )
    
    # Save preprocessed data
    pipeline.save_preprocessed_data(windowed_data, windowed_labels, 
                                    output_dir="../data/processed")
    
    # Example 2: Process multiple files
    # pipeline = PreprocessingPipeline(
    #     sampling_rate=100.0,
    #     window_size=128,
    #     overlap=0.5,
    #     normalization_method='zscore',
    #     balance_method='undersample'
    # )
    # 
    # all_data, all_labels = pipeline.preprocess_multiple_files(
    #     data_dir="../data/raw",
    #     max_files=10  # Process first 10 files
    # )
    # 
    # pipeline.save_preprocessed_data(all_data, all_labels, 
    #                                 output_dir="../data/processed")
