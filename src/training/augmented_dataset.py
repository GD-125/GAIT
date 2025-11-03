# src/training/augmented_dataset.py
"""
Augmented Dataset for Training with Real-time Augmentation
Integrates data augmentation into the training pipeline to prevent overfitting.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, List
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.augmentation import TimeSeriesAugmenter, MixUp, HUGADB_3D_SENSORS


class AugmentedGaitDataset(Dataset):
    """
    Dataset with real-time data augmentation.
    Applies random augmentations during training to prevent overfitting.
    """

    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 augmenter: Optional[TimeSeriesAugmenter] = None,
                 augmentation_prob: float = 0.7,
                 use_mixup: bool = False,
                 mixup_alpha: float = 0.2,
                 train_mode: bool = True):
        """
        Initialize augmented dataset.

        Args:
            data: Input data (samples, seq_length, features)
            labels: Labels (samples,)
            augmenter: TimeSeriesAugmenter instance
            augmentation_prob: Probability of applying augmentation
            use_mixup: Whether to use MixUp augmentation
            mixup_alpha: MixUp alpha parameter
            train_mode: If True, apply augmentation; if False, no augmentation
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
        self.train_mode = train_mode
        self.augmentation_prob = augmentation_prob
        self.use_mixup = use_mixup

        # Initialize augmenter
        if augmenter is None and train_mode:
            self.augmenter = TimeSeriesAugmenter(
                noise_factor=0.01,         # 1% noise
                scaling_factor=0.15,       # ±15% scaling
                rotation_angle=10.0,       # ±10 degrees rotation
                time_warp_factor=0.2,      # 20% time warping
                magnitude_warp_factor=0.2  # 20% magnitude warping
            )
        else:
            self.augmenter = augmenter

        # Initialize MixUp
        if use_mixup and train_mode:
            self.mixup = MixUp(alpha=mixup_alpha)
        else:
            self.mixup = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get item with optional augmentation.

        Args:
            idx: Sample index

        Returns:
            Tuple of (data, label)
        """
        data = self.data[idx].numpy()
        label = self.labels[idx].item()

        # Apply augmentation during training
        if self.train_mode and self.augmenter is not None:
            if np.random.random() < self.augmentation_prob:
                data, label = self.augmenter.augment(
                    data,
                    label,
                    sensor_3d_indices=HUGADB_3D_SENSORS
                )

        # Apply MixUp (requires another sample)
        if self.train_mode and self.mixup is not None and np.random.random() < 0.3:
            # Get random second sample
            idx2 = np.random.randint(0, len(self))
            data2 = self.data[idx2].numpy()
            label2 = self.labels[idx2].item()

            # Apply mixup
            data, label = self.mixup(data, label, data2, label2)

        return torch.FloatTensor(data), torch.FloatTensor([label])

    def set_train_mode(self, train_mode: bool):
        """Toggle training mode (augmentation on/off)."""
        self.train_mode = train_mode


class BalancedAugmentedDataset(AugmentedGaitDataset):
    """
    Augmented dataset with class balancing.
    Oversamples minority class with augmentation to prevent overfitting.
    """

    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 augmenter: Optional[TimeSeriesAugmenter] = None,
                 augmentation_prob: float = 0.7,
                 balance_ratio: float = 1.0,
                 train_mode: bool = True):
        """
        Initialize balanced augmented dataset.

        Args:
            data: Input data
            labels: Labels
            augmenter: Augmenter instance
            augmentation_prob: Augmentation probability
            balance_ratio: Target ratio of minority to majority class (1.0 = balanced)
            train_mode: Training mode flag
        """
        super().__init__(data, labels, augmenter, augmentation_prob, train_mode=train_mode)

        # Calculate class distribution
        unique, counts = np.unique(labels, return_counts=True)
        self.class_counts = dict(zip(unique, counts))

        # Determine minority/majority classes
        if len(unique) == 2:
            if counts[0] < counts[1]:
                self.minority_class = unique[0]
                self.majority_class = unique[1]
            else:
                self.minority_class = unique[1]
                self.majority_class = unique[0]

            # Calculate how many minority samples to generate
            majority_count = self.class_counts[self.majority_class]
            minority_count = self.class_counts[self.minority_class]
            target_minority_count = int(majority_count * balance_ratio)
            self.oversample_count = max(0, target_minority_count - minority_count)

            # Get minority class indices
            self.minority_indices = np.where(labels == self.minority_class)[0]

            print(f"\n✓ Balanced Dataset:")
            print(f"  Majority class ({self.majority_class}): {majority_count} samples")
            print(f"  Minority class ({self.minority_class}): {minority_count} samples")
            print(f"  Will generate {self.oversample_count} augmented minority samples")
        else:
            self.oversample_count = 0
            self.minority_indices = np.array([])

    def __len__(self):
        # Return original length + oversampled count
        return len(self.labels) + (self.oversample_count if self.train_mode else 0)

    def __getitem__(self, idx):
        # If idx is within original data range
        if idx < len(self.labels):
            return super().__getitem__(idx)

        # Otherwise, generate augmented minority sample
        else:
            # Pick random minority sample
            minority_idx = np.random.choice(self.minority_indices)
            data = self.data[minority_idx].numpy()
            label = self.labels[minority_idx].item()

            # Always augment oversampled data
            if self.augmenter is not None:
                data, label = self.augmenter.augment(
                    data,
                    label,
                    sensor_3d_indices=HUGADB_3D_SENSORS
                )

            return torch.FloatTensor(data), torch.FloatTensor([label])


if __name__ == "__main__":
    # Test augmented dataset
    print("Testing Augmented Dataset\n")

    # Create dummy data
    np.random.seed(42)
    data = np.random.randn(100, 128, 38)  # 100 samples, 128 timesteps, 38 features
    labels = np.random.randint(0, 2, 100)

    # Test basic augmented dataset
    print("="*60)
    print("Testing AugmentedGaitDataset")
    print("="*60)

    dataset = AugmentedGaitDataset(
        data, labels,
        augmentation_prob=0.7,
        use_mixup=True,
        train_mode=True
    )

    print(f"Dataset size: {len(dataset)}")

    # Get sample
    sample_data, sample_label = dataset[0]
    print(f"Sample shape: {sample_data.shape}")
    print(f"Label shape: {sample_label.shape}")

    # Test balanced dataset
    print("\n" + "="*60)
    print("Testing BalancedAugmentedDataset")
    print("="*60)

    # Create imbalanced data
    imbalanced_labels = np.array([0]*30 + [1]*70)
    imbalanced_data = np.random.randn(100, 128, 38)

    balanced_dataset = BalancedAugmentedDataset(
        imbalanced_data,
        imbalanced_labels,
        balance_ratio=1.0,
        train_mode=True
    )

    print(f"Original dataset size: {len(imbalanced_labels)}")
    print(f"Balanced dataset size: {len(balanced_dataset)}")

    print("\n✓ Augmented dataset tests passed!")
