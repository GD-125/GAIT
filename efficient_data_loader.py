"""
Efficient Data Loader with Caching, Parallel Processing, and Data Quality Filtering
Optimized for handling large numbers of CSV files with potentially irrelevant data.
"""

import numpy as np
import pandas as pd
import pickle
import hashlib
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')


class DataQualityFilter:
    """Filter out files with irrelevant or low-quality data."""

    def __init__(self,
                 min_samples: int = 100,
                 max_missing_ratio: float = 0.3,
                 min_variance_threshold: float = 1e-6):
        """
        Initialize data quality filter.

        Args:
            min_samples: Minimum number of samples required
            max_missing_ratio: Maximum ratio of missing values allowed
            min_variance_threshold: Minimum variance to consider data meaningful
        """
        self.min_samples = min_samples
        self.max_missing_ratio = max_missing_ratio
        self.min_variance_threshold = min_variance_threshold

    def check_file_quality(self, filepath: str, sensor_columns: List[str]) -> Tuple[bool, str]:
        """
        Check if a file meets quality criteria.

        Args:
            filepath: Path to CSV file
            sensor_columns: List of expected sensor column names

        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Read file with minimal parsing
            df = pd.read_csv(filepath, usecols=sensor_columns + ['activity'], nrows=None)

            # Check 1: Minimum samples
            if len(df) < self.min_samples:
                return False, f"Insufficient samples: {len(df)} < {self.min_samples}"

            # Check 2: Missing values ratio
            missing_ratio = df[sensor_columns].isnull().sum().sum() / (len(df) * len(sensor_columns))
            if missing_ratio > self.max_missing_ratio:
                return False, f"Too many missing values: {missing_ratio:.2%}"

            # Check 3: Check if data has meaningful variance (not all zeros/constants)
            variances = df[sensor_columns].var()
            low_variance_cols = (variances < self.min_variance_threshold).sum()
            if low_variance_cols > len(sensor_columns) * 0.5:  # More than 50% low variance
                return False, f"Low variance data: {low_variance_cols}/{len(sensor_columns)} columns"

            # Check 4: Check if activity labels exist
            if 'activity' not in df.columns or df['activity'].isnull().all():
                return False, "Missing or invalid activity labels"

            return True, "Valid"

        except Exception as e:
            return False, f"Error reading file: {str(e)}"


class EfficientDataCache:
    """Cache system for preprocessed data with file integrity checking."""

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize cache system.

        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.pkl"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)

    def _get_file_hash(self, filepath: str) -> str:
        """Get hash of file for cache key."""
        filepath = Path(filepath)
        # Hash based on filename and modification time
        key_string = f"{filepath.name}_{filepath.stat().st_mtime}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, filepath: str, pipeline_config: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get cached data if available and valid.

        Args:
            filepath: Original file path
            pipeline_config: Preprocessing configuration

        Returns:
            Tuple of (windowed_data, labels) or None if not cached
        """
        file_hash = self._get_file_hash(filepath)
        config_hash = hashlib.md5(str(pipeline_config).encode()).hexdigest()
        cache_key = f"{file_hash}_{config_hash}"

        if cache_key in self.metadata:
            cache_path = self.cache_dir / f"{cache_key}.npz"
            if cache_path.exists():
                try:
                    data = np.load(cache_path)
                    return data['windowed_data'], data['labels']
                except:
                    # Cache corrupted, remove it
                    del self.metadata[cache_key]
                    cache_path.unlink()

        return None

    def set(self, filepath: str, pipeline_config: Dict,
            windowed_data: np.ndarray, labels: np.ndarray):
        """
        Cache processed data.

        Args:
            filepath: Original file path
            pipeline_config: Preprocessing configuration
            windowed_data: Preprocessed windowed data
            labels: Labels
        """
        file_hash = self._get_file_hash(filepath)
        config_hash = hashlib.md5(str(pipeline_config).encode()).hexdigest()
        cache_key = f"{file_hash}_{config_hash}"

        cache_path = self.cache_dir / f"{cache_key}.npz"
        np.savez_compressed(cache_path, windowed_data=windowed_data, labels=labels)

        self.metadata[cache_key] = {
            'filepath': str(filepath),
            'cached_at': pd.Timestamp.now().isoformat(),
            'shape': windowed_data.shape
        }
        self._save_metadata()

    def clear(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.npz"):
            cache_file.unlink()
        self.metadata = {}
        self._save_metadata()

    def get_cache_info(self) -> Dict:
        """Get information about cached data."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.npz"))
        return {
            'num_cached_files': len(self.metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }


def process_single_file_worker(args):
    """
    Worker function for parallel file processing.

    Args:
        args: Tuple of (filepath, pipeline, cache, quality_filter, sensor_columns)

    Returns:
        Tuple of (filepath, windowed_data, labels, status_message)
    """
    filepath, pipeline_dict, cache_dir, quality_filter_dict, sensor_columns = args

    try:
        # Reconstruct objects (can't pickle complex objects easily)
        from preprocessing.pipeline import PreprocessingPipeline

        # Check cache first
        cache = EfficientDataCache(cache_dir)
        cached_data = cache.get(filepath, pipeline_dict)
        if cached_data is not None:
            return (filepath, cached_data[0], cached_data[1], "cached")

        # Check data quality
        quality_filter = DataQualityFilter(**quality_filter_dict)
        is_valid, reason = quality_filter.check_file_quality(filepath, sensor_columns)
        if not is_valid:
            return (filepath, None, None, f"skipped: {reason}")

        # Process file
        pipeline = PreprocessingPipeline(**pipeline_dict)
        windowed_data, labels = pipeline.preprocess_single_file(filepath)

        # Cache the result
        cache.set(filepath, pipeline_dict, windowed_data, labels)

        return (filepath, windowed_data, labels, "processed")

    except Exception as e:
        return (filepath, None, None, f"error: {str(e)}")


class EfficientGaitDataLoader:
    """
    Efficient data loader with parallel processing, caching, and quality filtering.
    """

    def __init__(self,
                 data_dir: str,
                 cache_dir: str = "data/cache",
                 enable_cache: bool = True,
                 enable_quality_filter: bool = True,
                 max_workers: Optional[int] = None):
        """
        Initialize efficient data loader.

        Args:
            data_dir: Directory containing CSV files
            cache_dir: Directory for caching preprocessed data
            enable_cache: Whether to use caching
            enable_quality_filter: Whether to filter low-quality files
            max_workers: Number of parallel workers (None = CPU count)
        """
        self.data_dir = Path(data_dir)
        self.enable_cache = enable_cache
        self.enable_quality_filter = enable_quality_filter
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)

        # Initialize components
        self.cache = EfficientDataCache(cache_dir) if enable_cache else None
        self.quality_filter = DataQualityFilter() if enable_quality_filter else None

        # Sensor columns definition
        self.sensor_columns = self._define_sensor_columns()

    def _define_sensor_columns(self) -> List[str]:
        """Define all sensor column names."""
        accel_cols = []
        for location in ['right_foot', 'right_shin', 'right_thigh',
                        'left_foot', 'left_shin', 'left_thigh']:
            for axis in ['x', 'y', 'z']:
                accel_cols.append(f'accelerometer_{location}_{axis}')

        gyro_cols = []
        for location in ['right_foot', 'right_shin', 'right_thigh',
                        'left_foot', 'left_shin', 'left_thigh']:
            for axis in ['x', 'y', 'z']:
                gyro_cols.append(f'gyroscope_{location}_{axis}')

        emg_cols = ['EMG_right', 'EMG_left']

        return accel_cols + gyro_cols + emg_cols

    def load_files_parallel(self,
                          file_pattern: str = "*.csv",
                          max_files: Optional[int] = None,
                          pipeline_config: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load and preprocess multiple files in parallel.

        Args:
            file_pattern: Glob pattern for file matching
            max_files: Maximum number of files to process
            pipeline_config: Preprocessing pipeline configuration

        Returns:
            Tuple of (windowed_data, labels, stats)
        """
        # Find all files
        file_paths = sorted(self.data_dir.glob(file_pattern))
        if max_files:
            file_paths = file_paths[:max_files]

        if not file_paths:
            raise FileNotFoundError(f"No files found matching {file_pattern} in {self.data_dir}")

        # Default pipeline config
        if pipeline_config is None:
            pipeline_config = {
                'sampling_rate': 100.0,
                'window_size': 128,
                'overlap': 0.5,
                'normalization_method': 'zscore',
                'filter_type': 'sensor_specific',
                'balance_method': 'none'
            }

        quality_filter_config = {
            'min_samples': 100,
            'max_missing_ratio': 0.3,
            'min_variance_threshold': 1e-6
        }

        print(f"\n{'='*80}")
        print("EFFICIENT PARALLEL DATA LOADING")
        print(f"{'='*80}")
        print(f"Files to process: {len(file_paths)}")
        print(f"Parallel workers: {self.max_workers}")
        print(f"Cache enabled: {self.enable_cache}")
        print(f"Quality filter enabled: {self.enable_quality_filter}")
        print(f"{'='*80}\n")

        # Prepare arguments for parallel processing
        args_list = [
            (str(fp), pipeline_config, str(self.cache.cache_dir) if self.cache else None,
             quality_filter_config, self.sensor_columns)
            for fp in file_paths
        ]

        # Process files in parallel
        all_windowed_data = []
        all_labels = []
        stats = {
            'processed': 0,
            'cached': 0,
            'skipped': 0,
            'errors': 0,
            'skipped_files': [],
            'error_files': []
        }

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_single_file_worker, args) for args in args_list]

            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                filepath, windowed_data, labels, status = future.result()

                if windowed_data is not None:
                    all_windowed_data.append(windowed_data)
                    all_labels.append(labels)

                    if status == "cached":
                        stats['cached'] += 1
                    else:
                        stats['processed'] += 1
                else:
                    if status.startswith("skipped"):
                        stats['skipped'] += 1
                        stats['skipped_files'].append((Path(filepath).name, status))
                    else:
                        stats['errors'] += 1
                        stats['error_files'].append((Path(filepath).name, status))

        if not all_windowed_data:
            raise ValueError("No valid data files were processed!")

        # Concatenate all data
        print("\nCombining all data...")
        all_windowed_data = np.vstack(all_windowed_data)
        all_labels = np.concatenate(all_labels)

        # Balance dataset if specified in pipeline config
        if pipeline_config.get('balance_method', 'none') != 'none':
            print(f"\nBalancing dataset using {pipeline_config['balance_method']}...")
            from preprocessing.segmentation import DataSegmenter
            segmenter = DataSegmenter(window_size=pipeline_config['window_size'],
                                     overlap=pipeline_config['overlap'])
            all_windowed_data, all_labels = segmenter.balance_dataset(
                all_windowed_data, all_labels, method=pipeline_config['balance_method']
            )

        # Print summary
        print(f"\n{'='*80}")
        print("LOADING COMPLETE")
        print(f"{'='*80}")
        print(f"Successfully processed: {stats['processed'] + stats['cached']} files")
        print(f"  - Newly processed: {stats['processed']}")
        print(f"  - Loaded from cache: {stats['cached']}")
        print(f"  - Skipped (quality): {stats['skipped']}")
        print(f"  - Errors: {stats['errors']}")
        print(f"\nFinal dataset:")
        print(f"  - Total windows: {len(all_windowed_data)}")
        print(f"  - Shape: {all_windowed_data.shape}")

        # Print class distribution
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"\nClass Distribution:")
        for label, count in zip(unique, counts):
            label_name = "Gait" if label == 1 else "Non-Gait"
            print(f"  {label_name}: {count} ({count/len(all_labels)*100:.1f}%)")

        if stats['skipped'] > 0:
            print(f"\nSkipped files (quality issues): {stats['skipped']}")
            for filename, reason in stats['skipped_files'][:5]:  # Show first 5
                print(f"  - {filename}: {reason}")
            if len(stats['skipped_files']) > 5:
                print(f"  ... and {len(stats['skipped_files']) - 5} more")

        print(f"{'='*80}\n")

        return all_windowed_data, all_labels, stats

    def get_cache_info(self):
        """Print cache information."""
        if self.cache:
            info = self.cache.get_cache_info()
            print(f"\nCache Information:")
            print(f"  Cached files: {info['num_cached_files']}")
            print(f"  Total size: {info['total_size_mb']:.2f} MB")
            print(f"  Cache directory: {info['cache_dir']}")

    def clear_cache(self):
        """Clear all cached data."""
        if self.cache:
            self.cache.clear()
            print("Cache cleared!")


if __name__ == "__main__":
    # Example usage
    loader = EfficientGaitDataLoader(
        data_dir="data/raw",
        cache_dir="data/cache",
        enable_cache=True,
        enable_quality_filter=True,
        max_workers=4  # Adjust based on your CPU
    )

    # Load data with parallel processing
    windowed_data, labels, stats = loader.load_files_parallel(
        file_pattern="*.csv",
        max_files=50,  # Process first 50 files
        pipeline_config={
            'sampling_rate': 100.0,
            'window_size': 128,
            'overlap': 0.5,
            'normalization_method': 'zscore',
            'filter_type': 'sensor_specific',
            'balance_method': 'undersample'
        }
    )

    # Show cache info
    loader.get_cache_info()

    # Save processed data
    np.save('data/processed/efficient_windowed_data.npy', windowed_data)
    np.save('data/processed/efficient_windowed_labels.npy', labels)
