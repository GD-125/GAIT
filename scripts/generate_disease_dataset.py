"""
Synthetic Neurological Disease Dataset Generator

Generates realistic gait data for 6 neurological conditions:
1. Parkinson's Disease - Characterized by shuffling gait, reduced arm swing, freezing
2. Huntington's Disease - Characterized by irregular, jerky movements (chorea)
3. Cerebral Palsy - Characterized by spastic, scissoring gait patterns
4. Multiple Sclerosis - Characterized by variable gait, balance issues, foot drop
5. Ataxia - Characterized by wide-based, unsteady gait, coordination issues
6. Normal/Healthy Gait - Regular, coordinated walking patterns

Each dataset includes:
- 38 sensor features (accelerometer, gyroscope, EMG from 6 body locations)
- Disease labels
- Severity levels (mild, moderate, severe)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
import json
from datetime import datetime
from tqdm import tqdm


class NeurologicalDiseaseDataGenerator:
    """Generate synthetic neurological disease gait data."""

    def __init__(self, sampling_rate: int = 100, seed: int = 42):
        """
        Initialize data generator.

        Args:
            sampling_rate: Samples per second
            seed: Random seed for reproducibility
        """
        self.sampling_rate = sampling_rate
        self.seed = seed
        np.random.seed(seed)

        # Disease configurations
        self.disease_patterns = self._define_disease_patterns()

        # Sensor locations
        self.sensor_locations = [
            'right_foot', 'right_shin', 'right_thigh',
            'left_foot', 'left_shin', 'left_thigh'
        ]

        # Severity levels
        self.severity_levels = {
            'mild': (0.2, 0.4),
            'moderate': (0.4, 0.7),
            'severe': (0.7, 0.95)
        }

    def _define_disease_patterns(self) -> Dict:
        """Define characteristic patterns for each disease."""
        return {
            0: {  # Parkinson's Disease
                'name': "Parkinson's Disease",
                'characteristics': {
                    'stride_length': 0.6,  # Reduced (shuffling)
                    'step_frequency': 1.3,  # Increased (small rapid steps)
                    'arm_swing': 0.3,  # Reduced
                    'movement_variability': 0.4,  # Reduced
                    'tremor_frequency': 5.0,  # 4-6 Hz tremor
                    'rigidity': 0.7,  # High rigidity
                    'postural_instability': 0.6
                }
            },
            1: {  # Huntington's Disease
                'name': "Huntington's Disease",
                'characteristics': {
                    'stride_length': 0.85,  # Variable
                    'step_frequency': 0.9,  # Variable
                    'arm_swing': 1.4,  # Excessive, jerky
                    'movement_variability': 1.8,  # Very high (chorea)
                    'irregularity': 0.9,  # Highly irregular
                    'chorea_intensity': 0.8,  # Involuntary movements
                    'coordination': 0.4  # Poor coordination
                }
            },
            2: {  # Cerebral Palsy
                'name': "Cerebral Palsy",
                'characteristics': {
                    'stride_length': 0.7,  # Reduced
                    'step_frequency': 0.8,  # Slower
                    'spasticity': 0.9,  # High muscle tone
                    'scissoring': 0.7,  # Leg crossing
                    'toe_walking': 0.8,  # Forefoot contact
                    'asymmetry': 0.7,  # Left-right differences
                    'muscle_cocontraction': 0.8
                }
            },
            3: {  # Multiple Sclerosis
                'name': "Multiple Sclerosis",
                'characteristics': {
                    'stride_length': 0.8,  # Slightly reduced
                    'step_frequency': 0.9,  # Slightly reduced
                    'foot_drop': 0.6,  # Difficulty lifting foot
                    'balance_issues': 0.7,  # Poor balance
                    'fatigue_effect': 0.8,  # Worsens over time
                    'sensory_loss': 0.6,  # Reduced proprioception
                    'variability': 0.7  # Inconsistent patterns
                }
            },
            4: {  # Ataxia
                'name': "Ataxia",
                'characteristics': {
                    'stride_length': 0.7,  # Variable
                    'step_frequency': 0.75,  # Slower
                    'base_width': 1.8,  # Wide-based gait
                    'coordination': 0.3,  # Very poor
                    'balance': 0.3,  # Poor balance
                    'movement_smoothness': 0.4,  # Jerky
                    'overshooting': 0.8,  # Dysmetria
                }
            },
            5: {  # Normal/Healthy
                'name': "Normal/Healthy Gait",
                'characteristics': {
                    'stride_length': 1.0,  # Normal
                    'step_frequency': 1.0,  # Normal (1.8-2.0 steps/sec)
                    'arm_swing': 1.0,  # Normal
                    'movement_variability': 0.15,  # Low variability
                    'symmetry': 0.95,  # High symmetry
                    'smoothness': 0.95,  # Smooth movements
                    'balance': 1.0  # Good balance
                }
            }
        }

    def generate_base_gait_cycle(self, duration: float = 1.0) -> np.ndarray:
        """
        Generate a base normal gait cycle.

        Args:
            duration: Duration in seconds

        Returns:
            Array of shape (num_samples, 38) representing one gait cycle
        """
        num_samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, num_samples)

        data = np.zeros((num_samples, 38))

        # Generate sinusoidal gait patterns for each sensor
        for i, location in enumerate(self.sensor_locations):
            base_idx = i * 6  # 6 values per location (3 accel + 3 gyro)

            # Phase shift for left vs right
            phase = 0 if 'right' in location else np.pi

            # Accelerometer (X, Y, Z)
            # Vertical movement (Y) - strongest signal
            data[:, base_idx + 1] = 1000 * np.sin(2 * np.pi * 2 * t + phase)

            # Forward-backward (X)
            data[:, base_idx] = 500 * np.sin(2 * np.pi * 2 * t + phase + np.pi/4)

            # Lateral (Z)
            data[:, base_idx + 2] = 300 * np.sin(2 * np.pi * 2 * t + phase + np.pi/2)

            # Gyroscope (X, Y, Z) - rotational movements
            data[:, base_idx + 3] = 100 * np.sin(2 * np.pi * 2 * t + phase)
            data[:, base_idx + 4] = 150 * np.sin(2 * np.pi * 2 * t + phase + np.pi/3)
            data[:, base_idx + 5] = 80 * np.sin(2 * np.pi * 2 * t + phase + np.pi/6)

        # EMG signals (last 2 columns)
        data[:, 36] = 200 * np.abs(np.sin(2 * np.pi * 2 * t))  # Right EMG
        data[:, 37] = 200 * np.abs(np.sin(2 * np.pi * 2 * t + np.pi))  # Left EMG

        return data

    def apply_disease_pattern(self,
                             base_data: np.ndarray,
                             disease_id: int,
                             severity: float = 0.5) -> np.ndarray:
        """
        Apply disease-specific modifications to base gait data.

        Args:
            base_data: Base gait cycle data
            disease_id: Disease class (0-5)
            severity: Severity level (0.0-1.0)

        Returns:
            Modified data with disease characteristics
        """
        data = base_data.copy()
        pattern = self.disease_patterns[disease_id]['characteristics']

        if disease_id == 0:  # Parkinson's
            # Reduce stride length (amplitude reduction)
            data *= pattern['stride_length'] + (1 - pattern['stride_length']) * (1 - severity)

            # Add tremor (4-6 Hz)
            tremor_freq = pattern['tremor_frequency']
            t = np.arange(len(data)) / self.sampling_rate
            tremor = severity * 50 * np.sin(2 * np.pi * tremor_freq * t)
            data += tremor[:, np.newaxis]

            # Reduce arm swing (upper body sensors)
            for i in [2, 5]:  # Thigh sensors
                data[:, i*6:(i+1)*6] *= pattern['arm_swing']

            # Add rigidity (reduce variability)
            noise_reduction = pattern['rigidity'] * severity
            data += np.random.normal(0, 30 * (1 - noise_reduction), data.shape)

        elif disease_id == 1:  # Huntington's
            # Add choreic movements (irregular, jerky)
            irregular_movements = severity * 200 * np.random.randn(*data.shape)
            data += irregular_movements

            # Increase variability
            data *= (1 + pattern['movement_variability'] * severity * np.random.randn(*data.shape) * 0.3)

            # Add sudden jerks
            num_jerks = int(10 * severity)
            for _ in range(num_jerks):
                jerk_idx = np.random.randint(0, len(data))
                jerk_magnitude = severity * 300
                data[max(0, jerk_idx-5):min(len(data), jerk_idx+5)] += jerk_magnitude * np.random.randn(1, 38)

        elif disease_id == 2:  # Cerebral Palsy
            # Add spasticity (increased muscle tone)
            spasticity_effect = pattern['spasticity'] * severity
            data *= (1 + 0.3 * spasticity_effect)

            # Add asymmetry (left-right differences)
            left_indices = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37]
            data[:, left_indices] *= (1 - pattern['asymmetry'] * severity * 0.3)

            # Add toe walking (increased foot/shin acceleration)
            for i in [3, 4]:  # Left foot and shin
                data[:, i*6:(i+1)*6] *= (1 + pattern['toe_walking'] * severity * 0.4)

            # Add scissoring (lateral movement in thighs)
            for i in [2, 5]:  # Thigh sensors
                data[:, i*6+2] += severity * 100 * np.sin(np.linspace(0, 4*np.pi, len(data)))

        elif disease_id == 3:  # Multiple Sclerosis
            # Add foot drop (reduced foot dorsiflexion)
            foot_indices = [0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23]
            data[:, foot_indices] *= (1 - pattern['foot_drop'] * severity * 0.4)

            # Add balance issues (increased sway)
            balance_noise = severity * 80 * np.random.randn(*data.shape)
            data += balance_noise

            # Simulate fatigue (gradual decline)
            fatigue_curve = np.linspace(1.0, 1.0 - pattern['fatigue_effect'] * severity * 0.3, len(data))
            data *= fatigue_curve[:, np.newaxis]

            # Add variability
            data += severity * 60 * np.random.randn(*data.shape)

        elif disease_id == 4:  # Ataxia
            # Increase base width (lateral spreading)
            lateral_indices = [2, 8, 14, 20, 26, 32]  # Z-axis accelerometers
            data[:, lateral_indices] *= (pattern['base_width'])

            # Add poor coordination (irregular patterns)
            coordination_noise = severity * 150 * np.random.randn(*data.shape)
            data += coordination_noise

            # Add overshooting/dysmetria
            overshoot_indices = np.random.choice(len(data), int(len(data) * 0.2 * severity))
            data[overshoot_indices] *= (1 + pattern['overshooting'] * severity * 0.5)

            # Reduce smoothness (add jerkiness)
            data += severity * 100 * np.diff(np.vstack([data[0], data]), axis=0)

        elif disease_id == 5:  # Normal/Healthy
            # Just add minimal natural variability
            data += np.random.normal(0, 20, data.shape)

        return data

    def generate_disease_dataset(self,
                                  disease_id: int,
                                  num_samples: int = 1000,
                                  severity_distribution: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate dataset for a specific disease.

        Args:
            disease_id: Disease class (0-5)
            num_samples: Number of samples to generate
            severity_distribution: Distribution of severity levels
                                  e.g., {'mild': 0.3, 'moderate': 0.5, 'severe': 0.2}

        Returns:
            Tuple of (data, labels, severities)
        """
        if severity_distribution is None:
            severity_distribution = {'mild': 0.3, 'moderate': 0.5, 'severe': 0.2}

        # Calculate samples per severity level
        severity_samples = {
            level: int(num_samples * ratio)
            for level, ratio in severity_distribution.items()
        }

        all_data = []
        all_labels = []
        all_severities = []

        disease_name = self.disease_patterns[disease_id]['name']
        print(f"\nGenerating {num_samples} samples for {disease_name}...")

        for severity_level, num in tqdm(severity_samples.items(), desc="Severity Levels"):
            severity_range = self.severity_levels[severity_level]

            for _ in range(num):
                # Random severity within range
                severity = np.random.uniform(*severity_range)

                # Generate base gait cycle
                duration = np.random.uniform(0.8, 1.2)  # Vary cycle duration
                base_data = self.generate_base_gait_cycle(duration)

                # Apply disease pattern
                disease_data = self.apply_disease_pattern(base_data, disease_id, severity)

                all_data.append(disease_data)
                all_labels.append(disease_id)
                all_severities.append(severity * 100)  # Convert to percentage

        return np.array(all_data), np.array(all_labels), np.array(all_severities)

    def generate_complete_dataset(self,
                                   samples_per_disease: int = 1000,
                                   output_dir: str = 'data/disease_data/synthetic') -> Dict:
        """
        Generate complete dataset for all diseases.

        Args:
            samples_per_disease: Samples per disease class
            output_dir: Output directory

        Returns:
            Dictionary with dataset statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("NEUROLOGICAL DISEASE DATASET GENERATION")
        print("="*80)
        print(f"Samples per disease: {samples_per_disease}")
        print(f"Total samples: {samples_per_disease * 6}")
        print(f"Output directory: {output_dir}")
        print("="*80)

        all_data = []
        all_labels = []
        all_severities = []

        # Generate data for each disease
        for disease_id in range(6):
            data, labels, severities = self.generate_disease_dataset(
                disease_id=disease_id,
                num_samples=samples_per_disease
            )
            all_data.append(data)
            all_labels.append(labels)
            all_severities.append(severities)

        # Combine all data
        X = np.vstack(all_data)
        y = np.concatenate(all_labels)
        severity = np.concatenate(all_severities)

        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        severity = severity[indices]

        # Save datasets
        print("\nSaving datasets...")
        np.save(output_path / 'disease_data.npy', X)
        np.save(output_path / 'disease_labels.npy', y)
        np.save(output_path / 'disease_severity.npy', severity)

        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'total_samples': len(X),
            'samples_per_disease': samples_per_disease,
            'num_diseases': 6,
            'diseases': [self.disease_patterns[i]['name'] for i in range(6)],
            'data_shape': X.shape,
            'sampling_rate': self.sampling_rate,
            'severity_range': [float(severity.min()), float(severity.max())],
            'class_distribution': {
                self.disease_patterns[i]['name']: int(np.sum(y == i))
                for i in range(6)
            }
        }

        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"\n✓ Dataset saved successfully!")
        print(f"  - Data shape: {X.shape}")
        print(f"  - Labels shape: {y.shape}")
        print(f"  - Severity shape: {severity.shape}")
        print(f"\nClass Distribution:")
        for disease_name, count in metadata['class_distribution'].items():
            print(f"  - {disease_name}: {count} samples")

        print("\n" + "="*80)
        print("DATASET GENERATION COMPLETED!")
        print("="*80 + "\n")

        return metadata


def main():
    """Main function to generate disease dataset."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate Neurological Disease Dataset')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Samples per disease class (default: 1000)')
    parser.add_argument('--output', type=str, default='data/disease_data/synthetic',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Generate dataset
    generator = NeurologicalDiseaseDataGenerator(seed=args.seed)
    metadata = generator.generate_complete_dataset(
        samples_per_disease=args.samples,
        output_dir=args.output
    )

    print(f"\nDataset files saved to: {args.output}")
    print("Files created:")
    print("  - disease_data.npy")
    print("  - disease_labels.npy")
    print("  - disease_severity.npy")
    print("  - metadata.json")


if __name__ == "__main__":
    main()
