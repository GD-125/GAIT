"""
Preprocessing Module for Gait Detection System

This module provides comprehensive data preprocessing capabilities including:
- Data loading and parsing
- Data cleaning (missing values, outliers)
- Signal filtering (Butterworth, Savitzky-Golay, Median)
- Normalization (Z-score, Min-Max, Robust)
- Window segmentation and labeling

Example usage:
    from preprocessing import GaitDataLoader, DataCleaner, SignalFilter, DataNormalizer, DataSegmenter
    
    # Load data
    loader = GaitDataLoader("data/raw")
    features, labels = loader.load_single_file("data.csv")
    
    # Preprocess
    cleaner = DataCleaner()
    features = cleaner.clean_data(features)[0]
    
    filter_obj = SignalFilter(sampling_rate=100.0)
    features = filter_obj.apply_sensor_specific_filters(features, loader.sensor_columns)
    
    normalizer = DataNormalizer(method='zscore')
    features = normalizer.fit_transform(features)
    
    # Segment
    segmenter = DataSegmenter(window_size=128, overlap=0.5)
    binary_labels = segmenter.create_binary_labels(labels)
    windowed_data, windowed_labels, _ = segmenter.segment_data(features, binary_labels)
"""

from .data_loader import GaitDataLoader
from .cleaner import DataCleaner
from .filter import SignalFilter
from .normalizer import DataNormalizer
from .segmentation import DataSegmenter

__all__ = [
    'GaitDataLoader',
    'DataCleaner',
    'SignalFilter',
    'DataNormalizer',
    'DataSegmenter'
]

__version__ = '1.0.0'
