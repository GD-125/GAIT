"""
Efficient analyzer for datasets with 40 columns
Optimized for large-scale gait detection datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class EfficientDatasetAnalyzer:
    """Memory-efficient analyzer for 40-column datasets."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.column_stats = {}
        self.file_summaries = []
        
    def quick_file_scan(self, file_path: Path) -> Dict:
        """Quick scan of a single file without loading full data."""
        try:
            # Read only first few rows to get structure
            df_sample = pd.read_csv(file_path, nrows=5)
            
            # Get full row count efficiently
            row_count = sum(1 for _ in open(file_path)) - 1  # Exclude header
            
            return {
                'file': file_path.name,
                'rows': row_count,
                'columns': len(df_sample.columns),
                'column_names': df_sample.columns.tolist(),
                'dtypes': df_sample.dtypes.to_dict(),
                'file_size_mb': file_path.stat().st_size / (1024 * 1024)
            }
        except Exception as e:
            return {'file': file_path.name, 'error': str(e)}
    
    def batch_analyze_datasets(self, file_pattern: str = "*.csv", 
                               max_files: Optional[int] = None) -> pd.DataFrame:
        """Analyze multiple datasets efficiently."""
        csv_files = list(self.data_dir.glob(file_pattern))
        
        if max_files:
            csv_files = csv_files[:max_files]
        
        print(f"📁 Found {len(csv_files)} CSV files")
        print("🔍 Scanning files...")
        
        summaries = []
        for file in tqdm(csv_files, desc="Scanning"):
            summary = self.quick_file_scan(file)
            summaries.append(summary)
        
        self.file_summaries = summaries
        df_summary = pd.DataFrame(summaries)
        
        return df_summary
    
    def analyze_column_statistics(self, file_paths: List[Path], 
                                  sample_size: int = 10000) -> Dict:
        """Analyze statistics across all columns efficiently using sampling."""
        print(f"\n📊 Analyzing column statistics (sampling {sample_size} rows per file)...")
        
        all_stats = {}
        
        for file in tqdm(file_paths[:20], desc="Processing"):  # Limit to 20 files for speed
            try:
                # Sample data instead of loading everything
                df = pd.read_csv(file, nrows=sample_size)
                
                for col in df.columns:
                    if col not in all_stats:
                        all_stats[col] = {
                            'values': [],
                            'missing_counts': 0,
                            'total_rows': 0
                        }
                    
                    all_stats[col]['values'].extend(df[col].dropna().values)
                    all_stats[col]['missing_counts'] += df[col].isna().sum()
                    all_stats[col]['total_rows'] += len(df)
                    
            except Exception as e:
                print(f"⚠ Error processing {file.name}: {str(e)}")
                continue
        
        # Calculate aggregated statistics
        column_stats = {}
        for col, data in all_stats.items():
            values = np.array(data['values'])
            column_stats[col] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'missing_pct': (data['missing_counts'] / data['total_rows']) * 100,
                'total_samples': len(values)
            }
        
        self.column_stats = column_stats
        return column_stats
    
    def detect_column_types(self) -> Dict[str, List[str]]:
        """Categorize columns by sensor type."""
        categories = {
            'accelerometer': [],
            'gyroscope': [],
            'emg': [],
            'angle': [],
            'other': []
        }
        
        for col in self.column_stats.keys():
            col_lower = col.lower()
            if 'acc' in col_lower or 'accel' in col_lower:
                categories['accelerometer'].append(col)
            elif 'gyro' in col_lower or 'gyr' in col_lower:
                categories['gyroscope'].append(col)
            elif 'emg' in col_lower or 'muscle' in col_lower:
                categories['emg'].append(col)
            elif 'angle' in col_lower or 'deg' in col_lower:
                categories['angle'].append(col)
            else:
                categories['other'].append(col)
        
        return categories
    
    def generate_analysis_report(self, output_dir: str = "analysis_results"):
        """Generate comprehensive analysis report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n📝 Generating analysis report...")
        
        # 1. File summary
        df_files = pd.DataFrame(self.file_summaries)
        df_files.to_csv(output_path / "file_summary.csv", index=False)
        
        # 2. Column statistics
        df_stats = pd.DataFrame(self.column_stats).T
        df_stats.to_csv(output_path / "column_statistics.csv")
        
        # 3. Column categories
        categories = self.detect_column_types()
        with open(output_path / "column_categories.txt", 'w') as f:
            for cat, cols in categories.items():
                f.write(f"\n{cat.upper()} ({len(cols)} columns):\n")
                for col in cols:
                    f.write(f"  - {col}\n")
        
        # 4. Visualization
        self.create_visualizations(output_path)
        
        print(f"✅ Report saved to: {output_path}")
        return output_path
    
    def create_visualizations(self, output_dir: Path):
        """Create visualization plots."""
        
        # Plot 1: File size distribution
        df_files = pd.DataFrame(self.file_summaries)
        if 'file_size_mb' in df_files.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df_files['file_size_mb'].dropna(), bins=30, edgecolor='black')
            plt.xlabel('File Size (MB)')
            plt.ylabel('Frequency')
            plt.title('Dataset File Size Distribution')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "file_size_distribution.png", dpi=300)
            plt.close()
        
        # Plot 2: Column value ranges
        if self.column_stats:
            df_stats = pd.DataFrame(self.column_stats).T
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Mean values
            df_stats['mean'].plot(kind='bar', ax=axes[0, 0], color='steelblue')
            axes[0, 0].set_title('Mean Values by Column')
            axes[0, 0].set_xlabel('Columns')
            axes[0, 0].set_ylabel('Mean')
            axes[0, 0].tick_params(axis='x', rotation=90, labelsize=6)
            
            # Standard deviation
            df_stats['std'].plot(kind='bar', ax=axes[0, 1], color='coral')
            axes[0, 1].set_title('Standard Deviation by Column')
            axes[0, 1].set_xlabel('Columns')
            axes[0, 1].set_ylabel('Std Dev')
            axes[0, 1].tick_params(axis='x', rotation=90, labelsize=6)
            
            # Min-Max ranges
            ax = axes[1, 0]
            x = np.arange(len(df_stats))
            ax.scatter(x, df_stats['min'], alpha=0.6, label='Min', s=20)
            ax.scatter(x, df_stats['max'], alpha=0.6, label='Max', s=20)
            ax.set_title('Value Ranges (Min-Max)')
            ax.set_xlabel('Column Index')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Missing data percentage
            df_stats['missing_pct'].plot(kind='bar', ax=axes[1, 1], color='indianred')
            axes[1, 1].set_title('Missing Data Percentage')
            axes[1, 1].set_xlabel('Columns')
            axes[1, 1].set_ylabel('Missing %')
            axes[1, 1].tick_params(axis='x', rotation=90, labelsize=6)
            
            plt.tight_layout()
            plt.savefig(output_dir / "column_statistics_overview.png", dpi=300)
            plt.close()
    
    def get_preprocessing_recommendations(self) -> Dict:
        """Get recommendations for preprocessing based on analysis."""
        recommendations = {
            'normalization': [],
            'filtering': [],
            'feature_engineering': [],
            'data_quality': []
        }
        
        if not self.column_stats:
            return recommendations
        
        for col, stats in self.column_stats.items():
            # Check value ranges
            if stats['max'] - stats['min'] > 100:
                recommendations['normalization'].append(
                    f"{col}: Large range ({stats['min']:.2f} to {stats['max']:.2f}) - Apply Z-score normalization"
                )
            
            # Check for noise (high std relative to mean)
            if stats['mean'] != 0 and stats['std'] / abs(stats['mean']) > 2:
                recommendations['filtering'].append(
                    f"{col}: High variance (std/mean = {stats['std']/abs(stats['mean']):.2f}) - Consider Butterworth filter"
                )
            
            # Check missing data
            if stats['missing_pct'] > 5:
                recommendations['data_quality'].append(
                    f"{col}: {stats['missing_pct']:.1f}% missing data - Consider imputation or removal"
                )
        
        # Feature engineering suggestions
        categories = self.detect_column_types()
        if len(categories['accelerometer']) >= 3:
            recommendations['feature_engineering'].append(
                "Accelerometer data: Compute magnitude, jerk, and frequency features"
            )
        if len(categories['gyroscope']) >= 3:
            recommendations['feature_engineering'].append(
                "Gyroscope data: Extract angular velocity magnitude and rate of change"
            )
        
        return recommendations


def main():
    """Main execution function."""
    
    print("="*70)
    print("🚀 EFFICIENT 40-COLUMN DATASET ANALYZER")
    print("="*70)
    
    # Configuration
    data_dir = input("\nEnter data directory path [data/raw]: ").strip() or "data/raw"
    max_files = input("Max files to analyze (Enter for all): ").strip()
    max_files = int(max_files) if max_files else None
    
    # Initialize analyzer
    analyzer = EfficientDatasetAnalyzer(data_dir)
    
    # Step 1: Quick scan
    print("\n" + "="*70)
    print("STEP 1: Quick File Scan")
    print("="*70)
    df_summary = analyzer.batch_analyze_datasets(max_files=max_files)
    
    print("\n📋 Dataset Summary:")
    print(f"  Total files: {len(df_summary)}")
    print(f"  Total size: {df_summary['file_size_mb'].sum():.2f} MB")
    print(f"  Avg rows per file: {df_summary['rows'].mean():.0f}")
    print(f"  Avg columns: {df_summary['columns'].mean():.0f}")
    
    # Step 2: Column analysis
    print("\n" + "="*70)
    print("STEP 2: Column Statistics Analysis")
    print("="*70)
    
    file_paths = list(Path(data_dir).glob("*.csv"))
    column_stats = analyzer.analyze_column_statistics(file_paths)
    
    print(f"\n✅ Analyzed {len(column_stats)} columns")
    
    # Step 3: Column categorization
    print("\n" + "="*70)
    print("STEP 3: Column Categorization")
    print("="*70)
    categories = analyzer.detect_column_types()
    for cat, cols in categories.items():
        if cols:
            print(f"\n{cat.upper()}: {len(cols)} columns")
            for col in cols[:5]:  # Show first 5
                print(f"  - {col}")
            if len(cols) > 5:
                print(f"  ... and {len(cols)-5} more")
    
    # Step 4: Recommendations
    print("\n" + "="*70)
    print("STEP 4: Preprocessing Recommendations")
    print("="*70)
    recommendations = analyzer.get_preprocessing_recommendations()
    
    for category, items in recommendations.items():
        if items:
            print(f"\n📌 {category.upper()}:")
            for item in items[:3]:  # Show top 3
                print(f"  • {item}")
            if len(items) > 3:
                print(f"  ... and {len(items)-3} more")
    
    # Step 5: Generate report
    print("\n" + "="*70)
    print("STEP 5: Generate Report")
    print("="*70)
    
    generate = input("\nGenerate full report with visualizations? (y/n) [y]: ").strip().lower()
    if generate != 'n':
        output_dir = analyzer.generate_analysis_report()
        print(f"\n✅ Complete analysis saved to: {output_dir}")
        print("\nGenerated files:")
        print("  1. file_summary.csv - Overview of all files")
        print("  2. column_statistics.csv - Detailed column stats")
        print("  3. column_categories.txt - Categorized columns")
        print("  4. file_size_distribution.png - File size visualization")
        print("  5. column_statistics_overview.png - Column stats plots")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
