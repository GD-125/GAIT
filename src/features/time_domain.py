# features/time_domain.py
"""
Time-Domain Feature Extraction Module for Gait Detection System
Extracts statistical and temporal features from windowed sensor data.
"""

import numpy as np
from typing import Dict, List, Optional
from scipy import stats
from scipy.signal import find_peaks


class TimeDomainFeatureExtractor:
    """
    Extracts time-domain features from windowed sensor signals.
    Useful for enhancing the model with hand-crafted features.
    """
    
    def __init__(self, feature_list: Optional[List[str]] = None):
        """
        Initialize the feature extractor.
        
        Args:
            feature_list: List of features to extract. If None, extracts all features.
                Available: ['mean', 'std', 'var', 'min', 'max', 'range', 'rms',
                           'skewness', 'kurtosis', 'zero_crossing_rate', 'peak_count',
                           'energy', 'entropy']
        """
        if feature_list is None:
            self.feature_list = [
                'mean', 'std', 'var', 'min', 'max', 'range', 'rms',
                'skewness', 'kurtosis', 'zero_crossing_rate', 'peak_count',
                'energy', 'entropy'
            ]
        else:
            self.feature_list = feature_list
        
        self.feature_names = []
    
    def extract_mean(self, signal: np.ndarray) -> float:
        """Mean value of the signal."""
        return np.mean(signal)
    
    def extract_std(self, signal: np.ndarray) -> float:
        """Standard deviation of the signal."""
        return np.std(signal)
    
    def extract_var(self, signal: np.ndarray) -> float:
        """Variance of the signal."""
        return np.var(signal)
    
    def extract_min(self, signal: np.ndarray) -> float:
        """Minimum value of the signal."""
        return np.min(signal)
    
    def extract_max(self, signal: np.ndarray) -> float:
        """Maximum value of the signal."""
        return np.max(signal)
    
    def extract_range(self, signal: np.ndarray) -> float:
        """Range (max - min) of the signal."""
        return np.max(signal) - np.min(signal)
    
    def extract_rms(self, signal: np.ndarray) -> float:
        """Root Mean Square of the signal."""
        return np.sqrt(np.mean(signal**2))
    
    def extract_skewness(self, signal: np.ndarray) -> float:
        """Skewness (asymmetry) of the signal distribution."""
        return stats.skew(signal)
    
    def extract_kurtosis(self, signal: np.ndarray) -> float:
        """Kurtosis (tailedness) of the signal distribution."""
        return stats.kurtosis(signal)
    
    def extract_zero_crossing_rate(self, signal: np.ndarray) -> float:
        """Rate of sign changes in the signal."""
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        return len(zero_crossings) / len(signal)
    
    def extract_peak_count(self, signal: np.ndarray, prominence: float = 0.5) -> int:
        """Number of peaks in the signal."""
        peaks, _ = find_peaks(signal, prominence=prominence)
        return len(peaks)
    
    def extract_energy(self, signal: np.ndarray) -> float:
        """Energy of the signal (sum of squared values)."""
        return np.sum(signal**2)
    
    def extract_entropy(self, signal: np.ndarray, bins: int = 10) -> float:
        """Shannon entropy of the signal."""
        hist, _ = np.histogram(signal, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        return -np.sum(hist * np.log2(hist))
    
    def extract_single_channel(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract all features from a single channel/signal.
        
        Args:
            signal: 1D array representing one sensor channel
            
        Returns:
            Array of extracted features
        """
        features = []
        
        for feature_name in self.feature_list:
            if feature_name == 'mean':
                features.append(self.extract_mean(signal))
            elif feature_name == 'std':
                features.append(self.extract_std(signal))
            elif feature_name == 'var':
                features.append(self.extract_var(signal))
            elif feature_name == 'min':
                features.append(self.extract_min(signal))
            elif feature_name == 'max':
                features.append(self.extract_max(signal))
            elif feature_name == 'range':
                features.append(self.extract_range(signal))
            elif feature_name == 'rms':
                features.append(self.extract_rms(signal))
            elif feature_name == 'skewness':
                features.append(self.extract_skewness(signal))
            elif feature_name == 'kurtosis':
                features.append(self.extract_kurtosis(signal))
            elif feature_name == 'zero_crossing_rate':
                features.append(self.extract_zero_crossing_rate(signal))
            elif feature_name == 'peak_count':
                features.append(self.extract_peak_count(signal))
            elif feature_name == 'energy':
                features.append(self.extract_energy(signal))
            elif feature_name == 'entropy':
                features.append(self.extract_entropy(signal))
        
        return np.array(features)
    
    def extract_window(self, window: np.ndarray) -> np.ndarray:
        """
        Extract features from a single window (all channels).
        
        Args:
            window: 2D array of shape (time_steps, num_channels)
            
        Returns:
            1D array of all extracted features
        """
        all_features = []
        
        # Extract features for each channel
        for channel_idx in range(window.shape[1]):
            signal = window[:, channel_idx]
            channel_features = self.extract_single_channel(signal)
            all_features.extend(channel_features)
        
        return np.array(all_features)
    
    def extract_batch(self, windowed_data: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Extract features from a batch of windows.
        
        Args:
            windowed_data: 3D array of shape (num_windows, time_steps, num_channels)
            verbose: Print progress information
            
        Returns:
            2D array of shape (num_windows, num_features)
        """
        num_windows = windowed_data.shape[0]
        num_channels = windowed_data.shape[2]
        num_features_per_channel = len(self.feature_list)
        total_features = num_channels * num_features_per_channel
        
        if verbose:
            print(f"\n{'='*60}")
            print("EXTRACTING TIME-DOMAIN FEATURES")
            print(f"{'='*60}")
            print(f"Input shape: {windowed_data.shape}")
            print(f"Features per channel: {num_features_per_channel}")
            print(f"Total features: {total_features}")
        
        # Extract features for all windows
        features_batch = np.zeros((num_windows, total_features))
        
        for i in range(num_windows):
            features_batch[i] = self.extract_window(windowed_data[i])
            
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{num_windows} windows...")
        
        if verbose:
            print(f"✓ Feature extraction complete!")
            print(f"Output shape: {features_batch.shape}")
            print(f"{'='*60}\n")
        
        return features_batch
    
    def get_feature_names(self, channel_names: List[str]) -> List[str]:
        """
        Generate feature names for all extracted features.
        
        Args:
            channel_names: List of channel/sensor names
            
        Returns:
            List of feature names
        """
        feature_names = []
        
        for channel_name in channel_names:
            for feature_name in self.feature_list:
                feature_names.append(f"{channel_name}_{feature_name}")
        
        self.feature_names = feature_names
        return feature_names
    
    def extract_with_names(self, 
                          windowed_data: np.ndarray,
                          channel_names: List[str]) -> tuple:
        """
        Extract features and return with feature names.
        
        Args:
            windowed_data: 3D array of shape (num_windows, time_steps, num_channels)
            channel_names: List of channel names
            
        Returns:
            Tuple of (features_array, feature_names_list)
        """
        features = self.extract_batch(windowed_data)
        feature_names = self.get_feature_names(channel_names)
        
        return features, feature_names
    
    def save_features(self, features: np.ndarray, filepath: str):
        """
        Save extracted features to disk.
        
        Args:
            features: Feature array
            filepath: Path to save the features
        """
        np.save(filepath, features)
        print(f"✓ Features saved to {filepath}")
    
    def load_features(self, filepath: str) -> np.ndarray:
        """
        Load features from disk.
        
        Args:
            filepath: Path to load features from
            
        Returns:
            Feature array
        """
        features = np.load(filepath)
        print(f"✓ Features loaded from {filepath}")
        return features


class StatisticalFeatureExtractor:
    """
    Additional statistical features for enhanced representation.
    """
    
    @staticmethod
    def extract_percentiles(signal: np.ndarray, percentiles: List[int] = [25, 50, 75]) -> np.ndarray:
        """Extract percentile values from signal."""
        return np.percentile(signal, percentiles)
    
    @staticmethod
    def extract_iqr(signal: np.ndarray) -> float:
        """Extract Interquartile Range."""
        q75, q25 = np.percentile(signal, [75, 25])
        return q75 - q25
    
    @staticmethod
    def extract_mad(signal: np.ndarray) -> float:
        """Extract Median Absolute Deviation."""
        return np.median(np.abs(signal - np.median(signal)))
    
    @staticmethod
    def extract_correlation_between_axes(window: np.ndarray, axis1_idx: int, axis2_idx: int) -> float:
        """
        Calculate correlation between two sensor axes.
        Useful for capturing coordination patterns.
        """
        if window.shape[1] <= max(axis1_idx, axis2_idx):
            return 0.0
        
        corr = np.corrcoef(window[:, axis1_idx], window[:, axis2_idx])[0, 1]
        return corr if not np.isnan(corr) else 0.0


if __name__ == "__main__":
    # Example usage
    print("Time-Domain Feature Extraction Example\n")
    
    # Create synthetic windowed data
    num_windows = 100
    window_size = 128
    num_channels = 38  # Your dataset has 38 features
    
    windowed_data = np.random.randn(num_windows, window_size, num_channels)
    
    # Extract features
    extractor = TimeDomainFeatureExtractor(
        feature_list=['mean', 'std', 'rms', 'energy', 'zero_crossing_rate']
    )
    
    features = extractor.extract_batch(windowed_data)
    
    print(f"Input shape: {windowed_data.shape}")
    print(f"Extracted features shape: {features.shape}")
    print(f"Features per channel: {len(extractor.feature_list)}")
    
    # Generate feature names
    channel_names = [f"sensor_{i}" for i in range(num_channels)]
    feature_names = extractor.get_feature_names(channel_names)
    print(f"\nTotal feature names: {len(feature_names)}")
    print(f"Example names: {feature_names[:5]}")
