"""
Data Quality Checker and Filter
Analyzes CSV files to identify and filter out irrelevant or low-quality data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob
from tqdm import tqdm
import json


class DataQualityChecker:
    """
    Comprehensive data quality checker for gait detection dataset.
    """

    def __init__(self, data_dir: str):
        """
        Initialize quality checker.

        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.sensor_columns = self._define_sensor_columns()
        self.quality_report = {}

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

    def check_single_file(self, filepath: str) -> Dict:
        """
        Perform comprehensive quality check on a single file.

        Args:
            filepath: Path to CSV file

        Returns:
            Dictionary with quality metrics
        """
        try:
            df = pd.read_csv(filepath)

            # Remove unnamed columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            # Check if required columns exist
            missing_columns = set(self.sensor_columns + ['activity']) - set(df.columns)
            if missing_columns:
                return {
                    'valid': False,
                    'error': f"Missing columns: {missing_columns}",
                    'filename': Path(filepath).name
                }

            # Extract features and labels
            features = df[self.sensor_columns]
            labels = df['activity']

            # Quality metrics
            quality_metrics = {
                'filename': Path(filepath).name,
                'valid': True,
                'num_samples': len(df),
                'num_features': len(self.sensor_columns),

                # Missing values
                'missing_values_count': features.isnull().sum().sum(),
                'missing_ratio': features.isnull().sum().sum() / (len(df) * len(self.sensor_columns)),
                'columns_with_missing': features.columns[features.isnull().any()].tolist(),

                # Zero values (might indicate sensor failure)
                'zero_columns': [],
                'low_variance_columns': [],

                # Label distribution
                'label_distribution': labels.value_counts().to_dict(),
                'unique_labels': labels.nunique(),

                # Statistical properties
                'feature_means': features.mean().to_dict(),
                'feature_stds': features.std().to_dict(),
                'feature_ranges': (features.max() - features.min()).to_dict(),

                # Data quality scores
                'completeness_score': 1 - features.isnull().sum().sum() / (len(df) * len(self.sensor_columns)),
                'variance_score': 0.0,
                'label_diversity_score': 0.0,
                'overall_quality_score': 0.0,

                # Issues detected
                'issues': []
            }

            # Check for columns with all zeros
            zero_cols = features.columns[(features == 0).all()].tolist()
            quality_metrics['zero_columns'] = zero_cols
            if zero_cols:
                quality_metrics['issues'].append(f"Columns with all zeros: {zero_cols}")

            # Check for low variance columns (constant or near-constant)
            variances = features.var()
            low_var_threshold = 1e-6
            low_var_cols = variances[variances < low_var_threshold].index.tolist()
            quality_metrics['low_variance_columns'] = low_var_cols
            if low_var_cols:
                quality_metrics['issues'].append(f"Low variance columns: {len(low_var_cols)}")

            # Variance score (what % of columns have meaningful variance)
            quality_metrics['variance_score'] = 1 - (len(low_var_cols) / len(self.sensor_columns))

            # Label diversity score
            if len(labels.unique()) > 1:
                # Calculate entropy of label distribution
                label_probs = labels.value_counts(normalize=True)
                entropy = -(label_probs * np.log2(label_probs)).sum()
                max_entropy = np.log2(len(label_probs))
                quality_metrics['label_diversity_score'] = entropy / max_entropy if max_entropy > 0 else 0
            else:
                quality_metrics['label_diversity_score'] = 0
                quality_metrics['issues'].append("Only one label present")

            # Check for outliers (values beyond 5 standard deviations)
            z_scores = np.abs((features - features.mean()) / features.std())
            extreme_outliers = (z_scores > 5).sum().sum()
            if extreme_outliers > len(df) * 0.1:  # More than 10% outliers
                quality_metrics['issues'].append(f"High number of outliers: {extreme_outliers}")

            # Check sample length
            if len(df) < 100:
                quality_metrics['issues'].append(f"Very short sequence: {len(df)} samples")

            # Calculate overall quality score (0-100)
            weights = {
                'completeness': 0.3,
                'variance': 0.4,
                'label_diversity': 0.3
            }

            overall_score = (
                weights['completeness'] * quality_metrics['completeness_score'] +
                weights['variance'] * quality_metrics['variance_score'] +
                weights['label_diversity'] * quality_metrics['label_diversity_score']
            ) * 100

            quality_metrics['overall_quality_score'] = overall_score

            # Determine if file passes quality threshold
            quality_threshold = 50  # Minimum score to be considered valid
            if overall_score < quality_threshold:
                quality_metrics['valid'] = False
                quality_metrics['issues'].append(f"Low quality score: {overall_score:.1f}/100")

            # Check missing ratio
            if quality_metrics['missing_ratio'] > 0.3:
                quality_metrics['valid'] = False
                quality_metrics['issues'].append(f"High missing ratio: {quality_metrics['missing_ratio']:.1%}")

            return quality_metrics

        except Exception as e:
            return {
                'filename': Path(filepath).name,
                'valid': False,
                'error': str(e),
                'issues': [f"Failed to load: {str(e)}"]
            }

    def check_all_files(self, file_pattern: str = "*.csv",
                       max_files: Optional[int] = None,
                       save_report: bool = True) -> Dict:
        """
        Check quality of all files in the directory.

        Args:
            file_pattern: Glob pattern for file matching
            max_files: Maximum number of files to check
            save_report: Whether to save report to JSON

        Returns:
            Dictionary with overall quality report
        """
        file_paths = sorted(self.data_dir.glob(file_pattern))
        if max_files:
            file_paths = file_paths[:max_files]

        print(f"\n{'='*80}")
        print("DATA QUALITY CHECK")
        print(f"{'='*80}")
        print(f"Directory: {self.data_dir}")
        print(f"Files to check: {len(file_paths)}")
        print(f"{'='*80}\n")

        valid_files = []
        invalid_files = []
        all_quality_scores = []

        for filepath in tqdm(file_paths, desc="Checking files"):
            quality_metrics = self.check_single_file(str(filepath))

            if quality_metrics.get('valid', False):
                valid_files.append(quality_metrics)
                all_quality_scores.append(quality_metrics['overall_quality_score'])
            else:
                invalid_files.append(quality_metrics)

            self.quality_report[quality_metrics['filename']] = quality_metrics

        # Generate summary report
        report = {
            'summary': {
                'total_files': len(file_paths),
                'valid_files': len(valid_files),
                'invalid_files': len(invalid_files),
                'validity_rate': len(valid_files) / len(file_paths) if file_paths else 0,
                'average_quality_score': np.mean(all_quality_scores) if all_quality_scores else 0,
                'min_quality_score': np.min(all_quality_scores) if all_quality_scores else 0,
                'max_quality_score': np.max(all_quality_scores) if all_quality_scores else 0,
            },
            'valid_files': [f['filename'] for f in valid_files],
            'invalid_files_details': invalid_files,
            'quality_distribution': {
                'excellent (80-100)': sum(1 for s in all_quality_scores if s >= 80),
                'good (60-80)': sum(1 for s in all_quality_scores if 60 <= s < 80),
                'fair (40-60)': sum(1 for s in all_quality_scores if 40 <= s < 60),
                'poor (0-40)': sum(1 for s in all_quality_scores if s < 40),
            }
        }

        # Print summary
        self.print_summary(report)

        # Save report
        if save_report:
            report_path = self.data_dir / 'data_quality_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4, default=str)
            print(f"\n✓ Quality report saved to: {report_path}")

            # Save list of valid files for easy loading
            valid_files_path = self.data_dir / 'valid_files.txt'
            with open(valid_files_path, 'w') as f:
                for filename in report['valid_files']:
                    f.write(f"{filename}\n")
            print(f"✓ Valid files list saved to: {valid_files_path}")

        return report

    def print_summary(self, report: Dict):
        """Print formatted summary of quality report."""
        summary = report['summary']

        print(f"\n{'='*80}")
        print("QUALITY CHECK SUMMARY")
        print(f"{'='*80}")
        print(f"Total Files: {summary['total_files']}")
        print(f"Valid Files: {summary['valid_files']} ({summary['validity_rate']:.1%})")
        print(f"Invalid Files: {summary['invalid_files']}")
        print(f"\nQuality Scores:")
        print(f"  Average: {summary['average_quality_score']:.1f}/100")
        print(f"  Range: {summary['min_quality_score']:.1f} - {summary['max_quality_score']:.1f}")
        print(f"\nQuality Distribution:")
        for category, count in report['quality_distribution'].items():
            print(f"  {category}: {count} files")

        if report['invalid_files_details']:
            print(f"\nTop Issues in Invalid Files:")
            issue_counts = {}
            for file_detail in report['invalid_files_details'][:10]:  # Show first 10
                for issue in file_detail.get('issues', []):
                    issue_key = issue.split(':')[0]  # Get issue type
                    issue_counts[issue_key] = issue_counts.get(issue_key, 0) + 1

            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {issue}: {count} files")

        print(f"{'='*80}\n")

    def get_valid_file_paths(self) -> List[str]:
        """
        Get list of valid file paths.

        Returns:
            List of valid file paths
        """
        valid_files = []
        for filename, metrics in self.quality_report.items():
            if metrics.get('valid', False):
                valid_files.append(str(self.data_dir / filename))
        return sorted(valid_files)

    def filter_files_by_quality(self, min_quality_score: float = 50.0) -> List[str]:
        """
        Filter files by minimum quality score.

        Args:
            min_quality_score: Minimum quality score (0-100)

        Returns:
            List of file paths that meet the quality threshold
        """
        filtered_files = []
        for filename, metrics in self.quality_report.items():
            if metrics.get('valid', False) and \
               metrics.get('overall_quality_score', 0) >= min_quality_score:
                filtered_files.append(str(self.data_dir / filename))
        return sorted(filtered_files)


def quick_scan_files(data_dir: str, max_files: Optional[int] = None):
    """
    Quick scan of files to identify obvious issues.

    Args:
        data_dir: Directory containing CSV files
        max_files: Maximum number of files to scan
    """
    checker = DataQualityChecker(data_dir)
    report = checker.check_all_files(max_files=max_files)

    return checker, report


if __name__ == "__main__":
    # Example usage
    import sys

    # Check data directory
    data_dir = "data/raw"

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print(f"Checking data quality in: {data_dir}")

    # Run quality check
    checker = DataQualityChecker(data_dir)
    report = checker.check_all_files(max_files=None, save_report=True)

    # Get list of valid files
    valid_files = checker.get_valid_file_paths()
    print(f"\n✓ Found {len(valid_files)} valid files")

    # Get high-quality files only
    high_quality_files = checker.filter_files_by_quality(min_quality_score=70.0)
    print(f"✓ Found {len(high_quality_files)} high-quality files (score >= 70)")
