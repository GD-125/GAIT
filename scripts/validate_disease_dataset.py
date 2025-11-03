"""
Validate and visualize the generated disease dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import seaborn as sns


def load_dataset(data_dir='data/disease_data'):
    """Load generated disease dataset."""
    data_path = Path(data_dir)

    X = np.load(data_path / 'disease_data.npy')
    y = np.load(data_path / 'disease_labels.npy')
    severity = np.load(data_path / 'disease_severity.npy')

    with open(data_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    return X, y, severity, metadata


def visualize_dataset(X, y, severity, metadata, output_dir='data/disease_data/validation'):
    """Create validation visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    disease_names = [metadata['diseases'][str(i)]['name'] for i in range(6)]
    disease_codes = [metadata['diseases'][str(i)]['code'] for i in range(6)]

    # 1. Class distribution
    plt.figure(figsize=(12, 6))
    counts = [np.sum(y == i) for i in range(6)]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#95E1D3']

    plt.bar(disease_codes, counts, color=colors, edgecolor='black', linewidth=1.5)
    plt.xlabel('Disease Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.title('Disease Dataset Class Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'class_distribution.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved class distribution plot")
    plt.close()

    # 2. Severity distribution per disease
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(6):
        disease_severity = severity[y == i]
        axes[i].hist(disease_severity, bins=30, color=colors[i], edgecolor='black', alpha=0.7)
        axes[i].set_xlabel('Severity (%)', fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].set_title(f'{disease_names[i]}', fontsize=11, fontweight='bold')
        axes[i].grid(axis='y', alpha=0.3)
        axes[i].axvline(disease_severity.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {disease_severity.mean():.1f}%')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(output_path / 'severity_distributions.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved severity distribution plots")
    plt.close()

    # 3. Sample time series for each disease
    fig, axes = plt.subplots(6, 1, figsize=(14, 18))

    for i in range(6):
        # Get first sample of this disease
        sample_idx = np.where(y == i)[0][0]
        sample_data = X[sample_idx]

        # Plot first 6 features (right foot accel x,y,z and gyro x,y,z)
        time = np.arange(len(sample_data)) / metadata['sampling_rate']

        axes[i].plot(time, sample_data[:, 0], label='Accel X', alpha=0.7)
        axes[i].plot(time, sample_data[:, 1], label='Accel Y', alpha=0.7)
        axes[i].plot(time, sample_data[:, 2], label='Accel Z', alpha=0.7)

        axes[i].set_xlabel('Time (s)', fontsize=10)
        axes[i].set_ylabel('Acceleration', fontsize=10)
        axes[i].set_title(f'{disease_names[i]} - Sample Gait Pattern (Right Foot)', fontsize=11, fontweight='bold')
        axes[i].legend(loc='upper right')
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'sample_time_series.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved sample time series plots")
    plt.close()

    # 4. Feature statistics heatmap
    feature_means = np.zeros((6, 38))
    for i in range(6):
        disease_samples = X[y == i]
        feature_means[i] = disease_samples.reshape(-1, 38).mean(axis=0)

    plt.figure(figsize=(16, 8))
    sns.heatmap(feature_means.T, cmap='RdYlBu_r', cbar_kws={'label': 'Mean Value'},
                xticklabels=disease_codes, yticklabels=[f'F{i}' for i in range(38)])
    plt.xlabel('Disease Class', fontsize=12, fontweight='bold')
    plt.ylabel('Feature Index', fontsize=12, fontweight='bold')
    plt.title('Mean Feature Values per Disease Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'feature_heatmap.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved feature heatmap")
    plt.close()

    # 5. Data statistics summary
    print("\n" + "="*80)
    print("DATASET VALIDATION SUMMARY")
    print("="*80)
    print(f"\nDataset Shape: {X.shape}")
    print(f"Labels Shape: {y.shape}")
    print(f"Severity Shape: {severity.shape}")
    print(f"\nWindow Size: {metadata['window_size']}")
    print(f"Number of Features: {metadata['num_features']}")
    print(f"Sampling Rate: {metadata['sampling_rate']} Hz")

    print(f"\nClass Distribution:")
    for i in range(6):
        count = np.sum(y == i)
        percentage = (count / len(y)) * 100
        print(f"  {i}: {disease_codes[i]:6s} ({disease_names[i]:25s}) - {count:4d} samples ({percentage:5.1f}%)")

    print(f"\nSeverity Statistics:")
    print(f"  Overall Range: {severity.min():.2f}% - {severity.max():.2f}%")
    print(f"  Overall Mean: {severity.mean():.2f}%")
    print(f"  Overall Std: {severity.std():.2f}%")

    print(f"\nPer-Disease Severity:")
    for i in range(6):
        disease_severity = severity[y == i]
        print(f"  {disease_codes[i]:6s}: Mean={disease_severity.mean():5.1f}%, Std={disease_severity.std():5.1f}%")

    print(f"\nData Value Ranges:")
    print(f"  Overall Min: {X.min():.2f}")
    print(f"  Overall Max: {X.max():.2f}")
    print(f"  Overall Mean: {X.mean():.2f}")
    print(f"  Overall Std: {X.std():.2f}")

    # Check for NaN or Inf
    has_nan = np.isnan(X).any()
    has_inf = np.isinf(X).any()
    print(f"\nData Quality Checks:")
    print(f"  Contains NaN: {has_nan}")
    print(f"  Contains Inf: {has_inf}")

    if not has_nan and not has_inf:
        print("  [OK] Dataset is clean!")
    else:
        print("  [WARNING] Dataset contains invalid values!")

    print("\n" + "="*80)
    print(f"Validation plots saved to: {output_path}")
    print("="*80 + "\n")

    # Save statistics to file
    stats = {
        'dataset_shape': list(X.shape),
        'class_distribution': {disease_codes[i]: int(np.sum(y == i)) for i in range(6)},
        'severity_stats': {
            'overall': {
                'min': float(severity.min()),
                'max': float(severity.max()),
                'mean': float(severity.mean()),
                'std': float(severity.std())
            },
            'per_disease': {
                disease_codes[i]: {
                    'mean': float(severity[y == i].mean()),
                    'std': float(severity[y == i].std())
                }
                for i in range(6)
            }
        },
        'data_value_ranges': {
            'min': float(X.min()),
            'max': float(X.max()),
            'mean': float(X.mean()),
            'std': float(X.std())
        },
        'quality_checks': {
            'has_nan': bool(has_nan),
            'has_inf': bool(has_inf),
            'is_valid': bool(not has_nan and not has_inf)
        }
    }

    with open(output_path / 'validation_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"Validation statistics saved to: {output_path / 'validation_stats.json'}\n")


def main():
    """Main validation function."""
    print("\n" + "="*80)
    print("DISEASE DATASET VALIDATION")
    print("="*80 + "\n")

    # Load dataset
    print("Loading dataset...")
    X, y, severity, metadata = load_dataset('data/disease_data')
    print(f"[OK] Dataset loaded: {X.shape}\n")

    # Visualize and validate
    print("Generating validation visualizations...")
    visualize_dataset(X, y, severity, metadata, 'data/disease_data/validation')

    print("\nValidation complete!")


if __name__ == "__main__":
    main()
