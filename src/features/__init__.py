# features/__init__.py
"""
Feature Extraction Module for Gait Detection System

Provides time-domain feature extraction capabilities for sensor data.

Example usage:
    from features import TimeDomainFeatureExtractor
    
    extractor = TimeDomainFeatureExtractor(
        feature_list=['mean', 'std', 'rms', 'energy']
    )
    features = extractor.extract_batch(windowed_data)
"""

from .time_domain import TimeDomainFeatureExtractor, StatisticalFeatureExtractor

__all__ = [
    'TimeDomainFeatureExtractor',
    'StatisticalFeatureExtractor'
]

__version__ = '1.0.0'
