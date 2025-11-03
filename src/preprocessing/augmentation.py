# src/preprocessing/augmentation.py
"""
Time Series Data Augmentation for Gait Detection
Prevents overfitting by creating variations of training data.
"""

import numpy as np
import torch
from typing import Tuple, Optional


class TimeSeriesAugmenter:
    """
    Augmentation techniques for time series sensor data.
    Helps prevent overfitting by creating realistic variations.
    """

    def __init__(self,
                 noise_factor: float = 0.005,
                 scaling_factor: float = 0.1,
                 rotation_angle: float = 5.0,
                 time_warp_factor: float = 0.2,
                 magnitude_warp_factor: float = 0.2):
        """
        Initialize augmenter with augmentation parameters.

        Args:
            noise_factor: Gaussian noise std (0.005 = 0.5% noise)
            scaling_factor: Random scaling range (0.1 = ±10%)
            rotation_angle: Max rotation in degrees for 3D sensors
            time_warp_factor: Time warping strength
            magnitude_warp_factor: Magnitude warping strength
        """
        self.noise_factor = noise_factor
        self.scaling_factor = scaling_factor
        self.rotation_angle = rotation_angle
        self.time_warp_factor = time_warp_factor
        self.magnitude_warp_factor = magnitude_warp_factor

    def add_noise(self, data: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to the data.

        Args:
            data: Input data (seq_length, features)

        Returns:
            Augmented data
        """
        noise = np.random.normal(0, self.noise_factor, data.shape)
        return data + noise

    def scale(self, data: np.ndarray) -> np.ndarray:
        """
        Random scaling of the data.

        Args:
            data: Input data (seq_length, features)

        Returns:
            Scaled data
        """
        scale_factor = 1 + np.random.uniform(-self.scaling_factor, self.scaling_factor)
        return data * scale_factor

    def time_warp(self, data: np.ndarray) -> np.ndarray:
        """
        Apply time warping (speed up/slow down segments).

        Args:
            data: Input data (seq_length, features)

        Returns:
            Time-warped data
        """
        seq_length = data.shape[0]

        # Random warping points
        num_knots = 4
        knots = np.random.choice(seq_length, num_knots, replace=False)
        knots = np.sort(knots)

        # Random time shifts
        time_shifts = np.random.uniform(-self.time_warp_factor,
                                       self.time_warp_factor, num_knots)

        # Interpolate warped indices
        original_indices = np.arange(seq_length)
        warped_indices = original_indices.copy().astype(float)

        for i, (knot, shift) in enumerate(zip(knots, time_shifts)):
            # Apply smooth warping around each knot
            distance = np.abs(original_indices - knot) / (seq_length / num_knots)
            weight = np.exp(-distance)
            warped_indices += weight * shift * (seq_length / num_knots)

        # Clip and interpolate
        warped_indices = np.clip(warped_indices, 0, seq_length - 1)

        # Interpolate data at warped indices
        warped_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            warped_data[:, i] = np.interp(original_indices, warped_indices, data[:, i])

        return warped_data

    def magnitude_warp(self, data: np.ndarray) -> np.ndarray:
        """
        Apply magnitude warping (smooth random scaling over time).

        Args:
            data: Input data (seq_length, features)

        Returns:
            Magnitude-warped data
        """
        seq_length = data.shape[0]

        # Create smooth random curve
        num_knots = 4
        knot_values = 1 + np.random.uniform(-self.magnitude_warp_factor,
                                            self.magnitude_warp_factor, num_knots)

        # Interpolate to full sequence length
        knot_indices = np.linspace(0, seq_length - 1, num_knots)
        all_indices = np.arange(seq_length)
        smooth_curve = np.interp(all_indices, knot_indices, knot_values)

        # Apply warping
        return data * smooth_curve[:, np.newaxis]

    def rotation_3d(self, data: np.ndarray, sensor_indices: list) -> np.ndarray:
        """
        Apply random 3D rotation to accelerometer/gyroscope data.

        Args:
            data: Input data (seq_length, features)
            sensor_indices: List of (x, y, z) index tuples for 3D sensors

        Returns:
            Rotated data
        """
        data_aug = data.copy()

        # Random rotation angles (in radians)
        angles = np.random.uniform(-self.rotation_angle, self.rotation_angle, 3)
        angles = np.radians(angles)

        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])

        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])

        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])

        # Combined rotation
        R = Rz @ Ry @ Rx

        # Apply rotation to each 3D sensor
        for indices in sensor_indices:
            if len(indices) == 3:
                sensor_data = data[:, indices].T  # (3, seq_length)
                rotated = R @ sensor_data
                data_aug[:, indices] = rotated.T

        return data_aug

    def permutation(self, data: np.ndarray, num_segments: int = 4) -> np.ndarray:
        """
        Randomly permute segments (for non-temporal features).

        Args:
            data: Input data (seq_length, features)
            num_segments: Number of segments to create

        Returns:
            Permuted data
        """
        seq_length = data.shape[0]
        segment_size = seq_length // num_segments

        # Split into segments
        segments = []
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size if i < num_segments - 1 else seq_length
            segments.append(data[start:end])

        # Randomly permute segments
        np.random.shuffle(segments)

        return np.concatenate(segments, axis=0)

    def augment(self,
                data: np.ndarray,
                label: float,
                techniques: list = None,
                sensor_3d_indices: list = None) -> Tuple[np.ndarray, float]:
        """
        Apply random augmentation techniques.

        Args:
            data: Input data (seq_length, features)
            label: Label (unchanged)
            techniques: List of techniques to apply (default: all except permutation)
            sensor_3d_indices: Indices for 3D sensors for rotation

        Returns:
            Augmented (data, label)
        """
        if techniques is None:
            # Default augmentation pipeline
            techniques = ['noise', 'scale', 'magnitude_warp']

            # Add time_warp with 50% probability
            if np.random.random() < 0.5:
                techniques.append('time_warp')

            # Add rotation if 3D sensor indices provided
            if sensor_3d_indices and np.random.random() < 0.3:
                techniques.append('rotation')

        data_aug = data.copy()

        for technique in techniques:
            if technique == 'noise':
                data_aug = self.add_noise(data_aug)
            elif technique == 'scale':
                data_aug = self.scale(data_aug)
            elif technique == 'time_warp':
                data_aug = self.time_warp(data_aug)
            elif technique == 'magnitude_warp':
                data_aug = self.magnitude_warp(data_aug)
            elif technique == 'rotation' and sensor_3d_indices:
                data_aug = self.rotation_3d(data_aug, sensor_3d_indices)
            elif technique == 'permutation':
                data_aug = self.permutation(data_aug)

        return data_aug, label

    def augment_batch(self,
                     batch_data: np.ndarray,
                     batch_labels: np.ndarray,
                     augmentation_prob: float = 0.5,
                     sensor_3d_indices: list = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment a batch of data.

        Args:
            batch_data: (batch_size, seq_length, features)
            batch_labels: (batch_size,)
            augmentation_prob: Probability of augmenting each sample
            sensor_3d_indices: Indices for 3D sensors

        Returns:
            Augmented (batch_data, batch_labels)
        """
        batch_size = batch_data.shape[0]
        augmented_data = batch_data.copy()

        for i in range(batch_size):
            if np.random.random() < augmentation_prob:
                augmented_data[i], _ = self.augment(
                    batch_data[i],
                    batch_labels[i],
                    sensor_3d_indices=sensor_3d_indices
                )

        return augmented_data, batch_labels


class MixUp:
    """
    MixUp augmentation for time series data.
    Mixes two samples to create synthetic training data.
    """

    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp.

        Args:
            alpha: Beta distribution parameter (0.2 is conservative)
        """
        self.alpha = alpha

    def __call__(self,
                 data1: np.ndarray,
                 label1: float,
                 data2: np.ndarray,
                 label2: float) -> Tuple[np.ndarray, float]:
        """
        Apply MixUp to two samples.

        Args:
            data1: First sample
            label1: First label
            data2: Second sample
            label2: Second label

        Returns:
            Mixed (data, label)
        """
        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Mix data and labels
        mixed_data = lam * data1 + (1 - lam) * data2
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_data, mixed_label


# Sensor indices for HuGaDB dataset (38 features)
# Format: [x, y, z] indices for each 3D sensor
HUGADB_3D_SENSORS = [
    [0, 1, 2],      # Accel ankle
    [3, 4, 5],      # Gyro ankle
    [6, 7, 8],      # Accel chest
    [9, 10, 11],    # Gyro chest
    [12, 13, 14],   # Accel right pocket
    [15, 16, 17],   # Gyro right pocket
    [18, 19, 20],   # Accel right lower arm
    [21, 22, 23],   # Gyro right lower arm
    [24, 25, 26],   # Accel right upper arm
    [27, 28, 29],   # Gyro right upper arm
    [30, 31, 32],   # Accel right wrist
    [33, 34, 35],   # Gyro right wrist
]


if __name__ == "__main__":
    # Test augmentation
    print("Testing Time Series Augmentation\n")

    # Create dummy data
    seq_length = 128
    num_features = 38
    data = np.random.randn(seq_length, num_features)
    label = 1.0

    # Initialize augmenter
    augmenter = TimeSeriesAugmenter(
        noise_factor=0.005,
        scaling_factor=0.1,
        rotation_angle=5.0
    )

    # Test individual techniques
    print("Testing individual augmentation techniques:")
    print(f"Original data shape: {data.shape}")
    print(f"Original data range: [{data.min():.3f}, {data.max():.3f}]\n")

    # Noise
    noisy = augmenter.add_noise(data)
    print(f"✓ Noise: range [{noisy.min():.3f}, {noisy.max():.3f}]")

    # Scaling
    scaled = augmenter.scale(data)
    print(f"✓ Scaling: range [{scaled.min():.3f}, {scaled.max():.3f}]")

    # Time warp
    warped = augmenter.time_warp(data)
    print(f"✓ Time warp: range [{warped.min():.3f}, {warped.max():.3f}]")

    # Magnitude warp
    mag_warped = augmenter.magnitude_warp(data)
    print(f"✓ Magnitude warp: range [{mag_warped.min():.3f}, {mag_warped.max():.3f}]")

    # 3D rotation
    rotated = augmenter.rotation_3d(data, HUGADB_3D_SENSORS)
    print(f"✓ 3D rotation: range [{rotated.min():.3f}, {rotated.max():.3f}]")

    # Full augmentation pipeline
    aug_data, aug_label = augmenter.augment(
        data, label,
        sensor_3d_indices=HUGADB_3D_SENSORS
    )
    print(f"\n✓ Full pipeline: range [{aug_data.min():.3f}, {aug_data.max():.3f}]")

    # Test batch augmentation
    batch_data = np.random.randn(16, seq_length, num_features)
    batch_labels = np.random.randint(0, 2, 16)

    aug_batch_data, aug_batch_labels = augmenter.augment_batch(
        batch_data, batch_labels,
        augmentation_prob=0.5,
        sensor_3d_indices=HUGADB_3D_SENSORS
    )

    print(f"\n✓ Batch augmentation: {aug_batch_data.shape}")
    print("\nAugmentation module ready for use!")
