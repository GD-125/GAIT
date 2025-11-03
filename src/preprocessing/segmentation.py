"""
Data Segmentation Module for Gait Detection System
Segments time-series data into fixed-length windows for model input.
"""

import numpy as np
from typing import Tuple, List, Optional
from collections import Counter


class DataSegmenter:
    """
    Segments continuous time-series data into fixed-length windows.
    Supports overlapping and non-overlapping windows with label handling.
    """
    
    def __init__(self, 
                 window_size: int = 128,
                 overlap: float = 0.5,
                 sampling_rate: float = 100.0):
        """
        Initialize the data segmenter.
        
        Args:
            window_size: Number of samples per window
            overlap: Overlap ratio between consecutive windows (0.0 to 0.99)
            sampling_rate: Sampling frequency in Hz
        """
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.step_size = int(window_size * (1 - overlap))
        
        # Calculate window duration in seconds
        self.window_duration = window_size / sampling_rate
        
        print(f"\n{'='*60}")
        print("SEGMENTATION CONFIGURATION")
        print(f"{'='*60}")
        print(f"Window size: {window_size} samples ({self.window_duration:.2f} seconds)")
        print(f"Overlap: {overlap*100:.0f}%")
        print(f"Step size: {self.step_size} samples")
        print(f"Sampling rate: {sampling_rate} Hz")
        print(f"{'='*60}\n")
    
    def segment_data(self, 
                     data: np.ndarray,
                     labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
        """
        Segment data into fixed-length windows.
        
        Args:
            data: Input data array (samples x features)
            labels: Label array (samples,) - optional
            
        Returns:
            Tuple of (windowed_data, windowed_labels, window_indices)
            - windowed_data: (num_windows, window_size, num_features)
            - windowed_labels: (num_windows,) - most common label per window
            - window_indices: Starting indices of each window
        """
        num_samples, num_features = data.shape
        
        # Calculate number of windows
        num_windows = (num_samples - self.window_size) // self.step_size + 1
        
        if num_windows <= 0:
            raise ValueError(f"Data length ({num_samples}) is too short for window size ({self.window_size})")
        
        print(f"Segmenting data into windows...")
        print(f"  Total samples: {num_samples}")
        print(f"  Number of windows: {num_windows}")
        
        # Initialize arrays
        windowed_data = np.zeros((num_windows, self.window_size, num_features))
        windowed_labels = None
        if labels is not None:
            windowed_labels = np.zeros(num_windows, dtype=labels.dtype)
        window_indices = []
        
        # Create windows
        for i in range(num_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            
            # Extract window
            windowed_data[i] = data[start_idx:end_idx, :]
            window_indices.append(start_idx)
            
            # Assign label (most common label in window)
            if labels is not None:
                window_labels = labels[start_idx:end_idx]
                windowed_labels[i] = self._get_majority_label(window_labels)
        
        print(f"✓ Segmentation complete!")
        print(f"  Output shape: {windowed_data.shape}")
        if windowed_labels is not None:
            print(f"  Labels shape: {windowed_labels.shape}")
        
        return windowed_data, windowed_labels, window_indices
    
    def _get_majority_label(self, labels: np.ndarray) -> str:
        """
        Get the most common label in a window.
        
        Args:
            labels: Array of labels in a window
            
        Returns:
            Most frequent label
        """
        if len(labels) == 0:
            return None
        
        # Count occurrences
        label_counts = Counter(labels)
        
        # Return most common
        return label_counts.most_common(1)[0][0]
    
    def create_binary_labels(self, 
                            labels: np.ndarray,
                            gait_activities: List[str] = None) -> np.ndarray:
        """
        Convert multi-class activity labels to binary (Gait vs Non-Gait).
        
        Args:
            labels: Original activity labels
            gait_activities: List of activities considered as gait
                           If None, uses common gait patterns
            
        Returns:
            Binary labels (1 = Gait, 0 = Non-Gait)
        """
        if gait_activities is None:
            # Default gait activities (common patterns in gait datasets)
            gait_activities = [
                'walk', 'walking', 'walk_normal', 'walk_slow', 'walk_fast',
                'jog', 'jogging', 'run', 'running',
                'stairs_up', 'stairs_down', 'upstairs', 'downstairs'
            ]
        
        print(f"\n{'='*60}")
        print("CREATING BINARY LABELS")
        print(f"{'='*60}")
        print(f"Gait activities: {gait_activities}")
        
        # Convert to binary
        binary_labels = np.zeros(len(labels), dtype=np.int32)
        
        for i, label in enumerate(labels):
            # Check if label contains any gait activity keyword (case-insensitive)
            label_lower = str(label).lower()
            if any(activity in label_lower for activity in gait_activities):
                binary_labels[i] = 1
        
        # Count distribution
        num_gait = np.sum(binary_labels == 1)
        num_non_gait = np.sum(binary_labels == 0)
        
        print(f"\nBinary Label Distribution:")
        print(f"  Gait (1): {num_gait} samples ({num_gait/len(binary_labels)*100:.1f}%)")
        print(f"  Non-Gait (0): {num_non_gait} samples ({num_non_gait/len(binary_labels)*100:.1f}%)")
        print(f"{'='*60}\n")
        
        return binary_labels
    
    def segment_multiple_sequences(self,
                                   data_list: List[np.ndarray],
                                   labels_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment multiple sequences (e.g., from different files) into windows.
        
        Args:
            data_list: List of data arrays
            labels_list: List of label arrays
            
        Returns:
            Tuple of (all_windowed_data, all_windowed_labels)
        """
        all_windowed_data = []
        all_windowed_labels = []
        
        print(f"\n{'='*60}")
        print(f"SEGMENTING {len(data_list)} SEQUENCES")
        print(f"{'='*60}\n")
        
        for i, (data, labels) in enumerate(zip(data_list, labels_list)):
            print(f"[{i+1}/{len(data_list)}] Processing sequence...")
            windowed_data, windowed_labels, _ = self.segment_data(data, labels)
            all_windowed_data.append(windowed_data)
            all_windowed_labels.append(windowed_labels)
        
        # Concatenate all windows
        all_windowed_data = np.vstack(all_windowed_data)
        all_windowed_labels = np.concatenate(all_windowed_labels)
        
        print(f"\n{'='*60}")
        print(f"SEGMENTATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total windows: {len(all_windowed_data)}")
        print(f"Final shape: {all_windowed_data.shape}")
        print(f"{'='*60}\n")
        
        return all_windowed_data, all_windowed_labels
    
    def balance_dataset(self,
                       windowed_data: np.ndarray,
                       windowed_labels: np.ndarray,
                       method: str = 'undersample') -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance the dataset by handling class imbalance.
        
        Args:
            windowed_data: Windowed data array
            windowed_labels: Windowed label array
            method: Balancing method ('undersample', 'oversample', 'none')
            
        Returns:
            Tuple of (balanced_data, balanced_labels)
        """
        if method == 'none':
            return windowed_data, windowed_labels
        
        print(f"\n{'='*60}")
        print(f"BALANCING DATASET - {method.upper()}")
        print(f"{'='*60}")
        
        # Get class distribution
        unique_labels, counts = np.unique(windowed_labels, return_counts=True)
        print(f"Original distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count} samples")
        
        if method == 'undersample':
            # Undersample majority class
            min_count = counts.min()
            
            balanced_data = []
            balanced_labels = []
            
            for label in unique_labels:
                # Get indices for this class
                indices = np.where(windowed_labels == label)[0]
                
                # Randomly sample min_count samples
                np.random.seed(42)
                selected_indices = np.random.choice(indices, size=min_count, replace=False)
                
                balanced_data.append(windowed_data[selected_indices])
                balanced_labels.append(windowed_labels[selected_indices])
            
            balanced_data = np.vstack(balanced_data)
            balanced_labels = np.concatenate(balanced_labels)
            
        elif method == 'oversample':
            # Oversample minority class
            max_count = counts.max()
            
            balanced_data = []
            balanced_labels = []
            
            for label in unique_labels:
                # Get indices for this class
                indices = np.where(windowed_labels == label)[0]
                
                # Randomly sample with replacement to reach max_count
                np.random.seed(42)
                selected_indices = np.random.choice(indices, size=max_count, replace=True)
                
                balanced_data.append(windowed_data[selected_indices])
                balanced_labels.append(windowed_labels[selected_indices])
            
            balanced_data = np.vstack(balanced_data)
            balanced_labels = np.concatenate(balanced_labels)
            
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        # Shuffle the balanced dataset
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(balanced_data))
        balanced_data = balanced_data[shuffle_idx]
        balanced_labels = balanced_labels[shuffle_idx]
        
        print(f"\nBalanced distribution:")
        unique_labels, counts = np.unique(balanced_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count} samples")
        print(f"✓ Balancing complete!")
        print(f"{'='*60}\n")
        
        return balanced_data, balanced_labels


if __name__ == "__main__":
    # Example usage
    from data_loader import GaitDataLoader
    from cleaner import DataCleaner
    from filter import SignalFilter
    from normalizer import DataNormalizer
    
    # Load and preprocess data
    loader = GaitDataLoader(data_dir="../data/raw")
    features, labels = loader.load_single_file("../data/raw/HuGaDB_v2_various_01_00.csv")
    
    cleaner = DataCleaner()
    features_cleaned, _ = cleaner.clean_data(features)
    
    filter_obj = SignalFilter(sampling_rate=100.0)
    features_filtered = filter_obj.apply_sensor_specific_filters(features_cleaned, 
                                                                   loader.sensor_columns)
    
    normalizer = DataNormalizer(method='zscore')
    features_normalized = normalizer.fit_transform(features_filtered)
    
    # Segment data
    segmenter = DataSegmenter(window_size=128, overlap=0.5, sampling_rate=100.0)
    
    # Create binary labels
    binary_labels = segmenter.create_binary_labels(labels)
    
    # Segment into windows
    windowed_data, windowed_labels, indices = segmenter.segment_data(features_normalized, 
                                                                      binary_labels)
    
    # Balance dataset
    balanced_data, balanced_labels = segmenter.balance_dataset(windowed_data, 
                                                                windowed_labels,
                                                                method='undersample')
    
    print(f"Final dataset ready for training!")
    print(f"  Data shape: {balanced_data.shape}")
    print(f"  Labels shape: {balanced_labels.shape}")
