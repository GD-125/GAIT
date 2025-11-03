# src/preprocessing/advanced_cleaner.py
"""
Advanced Data Cleaning for Gait Detection
Implements robust outlier detection and noise reduction to prevent overfitting.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy import stats
from scipy.signal import medfilt


class AdvancedDataCleaner:
    """
    Advanced data cleaning with multiple outlier detection methods.
    Helps reduce noise that can cause overfitting.
    """

    def __init__(self,
                 outlier_method: str = 'iqr',
                 iqr_multiplier: float = 3.0,
                 z_score_threshold: float = 4.0,
                 isolation_contamination: float = 0.01,
                 apply_median_filter: bool = True,
                 median_kernel_size: int = 3):
        """
        Initialize advanced cleaner.

        Args:
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation', 'ensemble')
            iqr_multiplier: IQR multiplier for outlier detection (3.0 is conservative)
            z_score_threshold: Z-score threshold for outlier detection
            isolation_contamination: Contamination parameter for isolation forest
            apply_median_filter: Whether to apply median filtering for spike removal
            median_kernel_size: Kernel size for median filter (odd number)
        """
        self.outlier_method = outlier_method
        self.iqr_multiplier = iqr_multiplier
        self.z_score_threshold = z_score_threshold
        self.isolation_contamination = isolation_contamination
        self.apply_median_filter = apply_median_filter
        self.median_kernel_size = median_kernel_size

    def detect_outliers_iqr(self, data: np.ndarray) -> np.ndarray:
        """
        Detect outliers using IQR method (robust to extreme values).

        Args:
            data: Input data (samples, features)

        Returns:
            Boolean mask (True = outlier)
        """
        q25 = np.percentile(data, 25, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        iqr = q75 - q25

        lower_bound = q25 - self.iqr_multiplier * iqr
        upper_bound = q75 + self.iqr_multiplier * iqr

        # Mark as outlier if ANY feature is out of bounds
        outlier_mask = np.any((data < lower_bound) | (data > upper_bound), axis=1)

        return outlier_mask

    def detect_outliers_zscore(self, data: np.ndarray) -> np.ndarray:
        """
        Detect outliers using modified Z-score method.

        Args:
            data: Input data (samples, features)

        Returns:
            Boolean mask (True = outlier)
        """
        # Use median and MAD for robustness
        median = np.median(data, axis=0)
        mad = np.median(np.abs(data - median), axis=0)

        # Avoid division by zero
        mad[mad == 0] = 1e-10

        # Modified z-score
        modified_z_scores = 0.6745 * (data - median) / mad

        # Mark as outlier if ANY feature has high z-score
        outlier_mask = np.any(np.abs(modified_z_scores) > self.z_score_threshold, axis=1)

        return outlier_mask

    def detect_outliers_isolation_forest(self, data: np.ndarray) -> np.ndarray:
        """
        Detect outliers using Isolation Forest (for complex patterns).

        Args:
            data: Input data (samples, features)

        Returns:
            Boolean mask (True = outlier)
        """
        try:
            from sklearn.ensemble import IsolationForest

            # Limit features to prevent overfitting
            n_features = min(data.shape[1], 10)
            selected_features = np.random.choice(data.shape[1], n_features, replace=False)

            iso_forest = IsolationForest(
                contamination=self.isolation_contamination,
                random_state=42,
                n_estimators=100
            )

            predictions = iso_forest.fit_predict(data[:, selected_features])
            outlier_mask = predictions == -1

            return outlier_mask

        except ImportError:
            print("Warning: scikit-learn not available, falling back to IQR method")
            return self.detect_outliers_iqr(data)

    def detect_outliers_ensemble(self, data: np.ndarray) -> np.ndarray:
        """
        Detect outliers using ensemble of methods (most robust).

        Args:
            data: Input data (samples, features)

        Returns:
            Boolean mask (True = outlier)
        """
        # Get outlier masks from multiple methods
        iqr_outliers = self.detect_outliers_iqr(data)
        zscore_outliers = self.detect_outliers_zscore(data)

        # Majority voting (outlier if 2+ methods agree)
        outlier_votes = iqr_outliers.astype(int) + zscore_outliers.astype(int)

        # Mark as outlier if at least 2 methods agree
        ensemble_mask = outlier_votes >= 2

        return ensemble_mask

    def remove_spikes(self, data: np.ndarray) -> np.ndarray:
        """
        Remove sudden spikes using median filtering.

        Args:
            data: Input data (samples, features)

        Returns:
            Filtered data
        """
        if not self.apply_median_filter:
            return data

        filtered_data = data.copy()

        # Apply median filter to each feature
        for i in range(data.shape[1]):
            filtered_data[:, i] = medfilt(data[:, i], kernel_size=self.median_kernel_size)

        return filtered_data

    def clip_extreme_values(self, data: np.ndarray, percentile: float = 99.5) -> np.ndarray:
        """
        Clip extreme values to reduce impact of outliers.

        Args:
            data: Input data (samples, features)
            percentile: Percentile for clipping (99.5 = clip top/bottom 0.5%)

        Returns:
            Clipped data
        """
        lower = np.percentile(data, 100 - percentile, axis=0)
        upper = np.percentile(data, percentile, axis=0)

        clipped_data = np.clip(data, lower, upper)

        return clipped_data

    def clean_data(self,
                   data: np.ndarray,
                   labels: Optional[np.ndarray] = None,
                   remove_outliers: bool = True,
                   clip_extremes: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """
        Perform comprehensive data cleaning.

        Args:
            data: Input data (samples, features)
            labels: Optional labels
            remove_outliers: Whether to remove outlier samples
            clip_extremes: Whether to clip extreme values

        Returns:
            Tuple of (cleaned_data, cleaned_labels, cleaning_report)
        """
        print(f"\n{'='*60}")
        print("ADVANCED DATA CLEANING")
        print(f"{'='*60}")
        print(f"Input shape: {data.shape}")

        original_samples = len(data)
        cleaned_data = data.copy()
        cleaned_labels = labels.copy() if labels is not None else None

        # Step 1: Remove spikes
        if self.apply_median_filter:
            print(f"\nStep 1: Removing spikes with median filter...")
            cleaned_data = self.remove_spikes(cleaned_data)
            print(f"✓ Spikes removed")

        # Step 2: Detect and remove outliers
        if remove_outliers:
            print(f"\nStep 2: Detecting outliers using {self.outlier_method} method...")

            if self.outlier_method == 'iqr':
                outlier_mask = self.detect_outliers_iqr(cleaned_data)
            elif self.outlier_method == 'zscore':
                outlier_mask = self.detect_outliers_zscore(cleaned_data)
            elif self.outlier_method == 'isolation':
                outlier_mask = self.detect_outliers_isolation_forest(cleaned_data)
            elif self.outlier_method == 'ensemble':
                outlier_mask = self.detect_outliers_ensemble(cleaned_data)
            else:
                raise ValueError(f"Unknown outlier method: {self.outlier_method}")

            n_outliers = np.sum(outlier_mask)
            outlier_percentage = (n_outliers / original_samples) * 100

            print(f"✓ Found {n_outliers} outliers ({outlier_percentage:.2f}%)")

            # Remove outliers
            if n_outliers > 0:
                cleaned_data = cleaned_data[~outlier_mask]
                if cleaned_labels is not None:
                    cleaned_labels = cleaned_labels[~outlier_mask]

                print(f"✓ Outliers removed")
        else:
            n_outliers = 0
            outlier_percentage = 0.0

        # Step 3: Clip extreme values
        if clip_extremes:
            print(f"\nStep 3: Clipping extreme values...")
            cleaned_data = self.clip_extreme_values(cleaned_data)
            print(f"✓ Extreme values clipped")

        # Generate report
        final_samples = len(cleaned_data)
        removed_samples = original_samples - final_samples

        report = {
            'original_samples': original_samples,
            'final_samples': final_samples,
            'removed_samples': removed_samples,
            'removal_percentage': (removed_samples / original_samples) * 100,
            'outliers_detected': n_outliers,
            'outlier_percentage': outlier_percentage,
            'method': self.outlier_method
        }

        print(f"\n{'='*60}")
        print("CLEANING REPORT")
        print(f"{'='*60}")
        print(f"Original samples: {original_samples}")
        print(f"Final samples: {final_samples}")
        print(f"Removed: {removed_samples} ({report['removal_percentage']:.2f}%)")
        print(f"{'='*60}\n")

        return cleaned_data, cleaned_labels, report


if __name__ == "__main__":
    # Test advanced cleaner
    print("Testing Advanced Data Cleaner\n")

    # Create synthetic data with outliers
    np.random.seed(42)
    normal_data = np.random.randn(1000, 38)

    # Add outliers
    n_outliers = 50
    outlier_indices = np.random.choice(1000, n_outliers, replace=False)
    normal_data[outlier_indices] = np.random.randn(n_outliers, 38) * 10  # Extreme values

    labels = np.random.randint(0, 2, 1000)

    # Test cleaner
    cleaner = AdvancedDataCleaner(
        outlier_method='ensemble',
        iqr_multiplier=3.0,
        apply_median_filter=True
    )

    cleaned_data, cleaned_labels, report = cleaner.clean_data(
        normal_data,
        labels,
        remove_outliers=True,
        clip_extremes=True
    )

    print(f"✓ Cleaning complete!")
    print(f"  Original shape: {normal_data.shape}")
    print(f"  Cleaned shape: {cleaned_data.shape}")
    print(f"  Removed: {report['removed_samples']} samples")
