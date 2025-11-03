"""
Signal Filtering Module for Gait Detection System
Applies noise reduction filters to sensor data.
"""

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, medfilt
from typing import Tuple, Optional


class SignalFilter:
    """
    Applies various filtering techniques to remove noise from sensor signals.
    Particularly useful for accelerometer, gyroscope, and EMG data.
    """
    
    def __init__(self, 
                 sampling_rate: float = 100.0,
                 filter_type: str = 'butterworth'):
        """
        Initialize the signal filter.
        
        Args:
            sampling_rate: Sampling frequency in Hz (default: 100 Hz)
            filter_type: Type of filter ('butterworth', 'savgol', 'median')
        """
        self.sampling_rate = sampling_rate
        self.filter_type = filter_type
        
    def butterworth_lowpass(self, 
                           data: np.ndarray,
                           cutoff_freq: float = 20.0,
                           order: int = 4) -> np.ndarray:
        """
        Apply Butterworth low-pass filter to remove high-frequency noise.
        
        Args:
            data: Input data array (samples x features)
            cutoff_freq: Cutoff frequency in Hz
            order: Filter order (higher = steeper cutoff)
            
        Returns:
            Filtered data array
        """
        # Calculate normalized cutoff frequency (Nyquist frequency = sampling_rate/2)
        nyquist_freq = 0.5 * self.sampling_rate
        normal_cutoff = cutoff_freq / nyquist_freq
        
        # Design Butterworth filter
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        # Apply filter to each feature (column)
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            # Use filtfilt for zero-phase filtering (no time delay)
            filtered_data[:, i] = filtfilt(b, a, data[:, i])
        
        return filtered_data
    
    def butterworth_bandpass(self,
                            data: np.ndarray,
                            lowcut: float = 0.5,
                            highcut: float = 20.0,
                            order: int = 4) -> np.ndarray:
        """
        Apply Butterworth band-pass filter to retain specific frequency range.
        Useful for isolating gait-related frequencies (typically 0.5-20 Hz).
        
        Args:
            data: Input data array (samples x features)
            lowcut: Low cutoff frequency in Hz
            highcut: High cutoff frequency in Hz
            order: Filter order
            
        Returns:
            Filtered data array
        """
        nyquist_freq = 0.5 * self.sampling_rate
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        
        # Design Butterworth bandpass filter
        b, a = butter(order, [low, high], btype='band', analog=False)
        
        # Apply filter to each feature
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = filtfilt(b, a, data[:, i])
        
        return filtered_data
    
    def savitzky_golay_filter(self,
                             data: np.ndarray,
                             window_length: int = 11,
                             polyorder: int = 3) -> np.ndarray:
        """
        Apply Savitzky-Golay filter for smoothing while preserving signal features.
        
        Args:
            data: Input data array (samples x features)
            window_length: Length of the filter window (must be odd)
            polyorder: Order of polynomial used to fit samples
            
        Returns:
            Filtered data array
        """
        # Ensure window length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        # Ensure window length is greater than polyorder
        if window_length <= polyorder:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1
        
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = savgol_filter(data[:, i], 
                                                window_length, 
                                                polyorder)
        
        return filtered_data
    
    def median_filter(self,
                     data: np.ndarray,
                     kernel_size: int = 5) -> np.ndarray:
        """
        Apply median filter to remove spike noise.
        Particularly effective for EMG signals.
        
        Args:
            data: Input data array (samples x features)
            kernel_size: Size of the median filter kernel (must be odd)
            
        Returns:
            Filtered data array
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = medfilt(data[:, i], kernel_size=kernel_size)
        
        return filtered_data
    
    def moving_average_filter(self,
                             data: np.ndarray,
                             window_size: int = 5) -> np.ndarray:
        """
        Apply simple moving average filter for smoothing.
        
        Args:
            data: Input data array (samples x features)
            window_size: Size of the moving average window
            
        Returns:
            Filtered data array
        """
        filtered_data = np.zeros_like(data)
        
        for i in range(data.shape[1]):
            # Use convolution for moving average
            kernel = np.ones(window_size) / window_size
            filtered_data[:, i] = np.convolve(data[:, i], kernel, mode='same')
        
        return filtered_data
    
    def apply_filter(self,
                    data: np.ndarray,
                    method: str = 'butterworth',
                    **kwargs) -> np.ndarray:
        """
        Apply the specified filter to the data.
        
        Args:
            data: Input data array (samples x features)
            method: Filter method ('butterworth', 'bandpass', 'savgol', 'median', 'moving_average')
            **kwargs: Additional parameters for the specific filter
            
        Returns:
            Filtered data array
        """
        print(f"\n{'='*60}")
        print(f"APPLYING {method.upper()} FILTER")
        print(f"{'='*60}")
        print(f"Input shape: {data.shape}")
        
        if method == 'butterworth' or method == 'lowpass':
            cutoff_freq = kwargs.get('cutoff_freq', 20.0)
            order = kwargs.get('order', 4)
            filtered_data = self.butterworth_lowpass(data, cutoff_freq, order)
            print(f"Cutoff frequency: {cutoff_freq} Hz")
            print(f"Filter order: {order}")
            
        elif method == 'bandpass':
            lowcut = kwargs.get('lowcut', 0.5)
            highcut = kwargs.get('highcut', 20.0)
            order = kwargs.get('order', 4)
            filtered_data = self.butterworth_bandpass(data, lowcut, highcut, order)
            print(f"Frequency range: {lowcut}-{highcut} Hz")
            print(f"Filter order: {order}")
            
        elif method == 'savgol':
            window_length = kwargs.get('window_length', 11)
            polyorder = kwargs.get('polyorder', 3)
            filtered_data = self.savitzky_golay_filter(data, window_length, polyorder)
            print(f"Window length: {window_length}")
            print(f"Polynomial order: {polyorder}")
            
        elif method == 'median':
            kernel_size = kwargs.get('kernel_size', 5)
            filtered_data = self.median_filter(data, kernel_size)
            print(f"Kernel size: {kernel_size}")
            
        elif method == 'moving_average':
            window_size = kwargs.get('window_size', 5)
            filtered_data = self.moving_average_filter(data, window_size)
            print(f"Window size: {window_size}")
            
        else:
            raise ValueError(f"Unknown filter method: {method}")
        
        print(f"✓ Filtering complete!")
        print(f"Output shape: {filtered_data.shape}")
        print(f"{'='*60}\n")
        
        return filtered_data
    
    def apply_sensor_specific_filters(self, 
                                      data: np.ndarray,
                                      sensor_columns: dict) -> np.ndarray:
        """
        Apply optimized filters for different sensor types.
        
        Args:
            data: Input data array (samples x features)
            sensor_columns: Dictionary mapping sensor types to column indices
            
        Returns:
            Filtered data array
        """
        print(f"\n{'='*60}")
        print("APPLYING SENSOR-SPECIFIC FILTERS")
        print(f"{'='*60}\n")
        
        filtered_data = data.copy()
        
        # Accelerometer: Bandpass filter (0.5-20 Hz for gait)
        if 'accelerometer' in sensor_columns:
            accel_indices = [i for i, col in enumerate(sensor_columns['all_features']) 
                           if 'accelerometer' in col]
            print(f"[1/3] Filtering {len(accel_indices)} accelerometer channels...")
            filtered_data[:, accel_indices] = self.butterworth_bandpass(
                data[:, accel_indices], lowcut=0.5, highcut=20.0, order=4)
            print("✓ Accelerometer filtering complete (0.5-20 Hz bandpass)")
        
        # Gyroscope: Bandpass filter (0.5-20 Hz)
        if 'gyroscope' in sensor_columns:
            gyro_indices = [i for i, col in enumerate(sensor_columns['all_features'])
                          if 'gyroscope' in col]
            print(f"\n[2/3] Filtering {len(gyro_indices)} gyroscope channels...")
            filtered_data[:, gyro_indices] = self.butterworth_bandpass(
                data[:, gyro_indices], lowcut=0.5, highcut=20.0, order=4)
            print("✓ Gyroscope filtering complete (0.5-20 Hz bandpass)")
        
        # EMG: Higher frequency range (20-450 Hz) + median filter for spikes
        if 'emg' in sensor_columns:
            emg_indices = [i for i, col in enumerate(sensor_columns['all_features'])
                         if 'EMG' in col]
            print(f"\n[3/3] Filtering {len(emg_indices)} EMG channels...")
            # First apply bandpass for EMG frequency range
            filtered_data[:, emg_indices] = self.butterworth_bandpass(
                data[:, emg_indices], lowcut=20.0, highcut=min(450.0, self.sampling_rate/2.5), order=4)
            # Then apply median filter to remove spikes
            filtered_data[:, emg_indices] = self.median_filter(
                filtered_data[:, emg_indices], kernel_size=3)
            print("✓ EMG filtering complete (20-450 Hz bandpass + median filter)")
        
        print(f"\n{'='*60}")
        print("All sensor-specific filters applied successfully!")
        print(f"{'='*60}\n")
        
        return filtered_data


if __name__ == "__main__":
    # Example usage
    from data_loader import GaitDataLoader
    from cleaner import DataCleaner
    
    # Load and clean data
    loader = GaitDataLoader(data_dir="../data/raw")
    features, labels = loader.load_single_file("../data/raw/HuGaDB_v2_various_01_00.csv")
    
    cleaner = DataCleaner()
    features_cleaned, _ = cleaner.clean_data(features)
    
    # Apply filtering
    filter_obj = SignalFilter(sampling_rate=100.0)
    
    # Option 1: Apply sensor-specific filters (recommended)
    features_filtered = filter_obj.apply_sensor_specific_filters(
        features_cleaned, 
        loader.sensor_columns
    )
    
    # Option 2: Apply uniform filter
    # features_filtered = filter_obj.apply_filter(features_cleaned, 
    #                                             method='bandpass',
    #                                             lowcut=0.5,
    #                                             highcut=20.0)
