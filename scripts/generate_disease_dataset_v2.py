"""
Enhanced Neurological Disease Dataset Generator (V2)
Uses REAL HuGaDB data as base and applies disease-specific transformations

Generates realistic gait data for 6 neurological conditions:
1. Parkinson's Disease - Shuffling gait, reduced arm swing, tremor
2. Huntington's Disease - Irregular, jerky movements (chorea)
3. Cerebral Palsy - Spastic, scissoring gait
4. Multiple Sclerosis - Variable gait, balance issues, foot drop
5. Ataxia - Wide-based, unsteady gait
6. Normal/Healthy Gait - From actual HuGaDB data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import json
from datetime import datetime
from tqdm import tqdm
import glob


class EnhancedDiseaseDataGenerator:
    """Generate disease gait data from real HuGaDB baseline."""

    def __init__(self, hugadb_dir: str, sampling_rate: int = 100, seed: int = 42):
        """
        Initialize generator with real HuGaDB data.

        Args:
            hugadb_dir: Path to HuGaDB raw CSV files
            sampling_rate: Samples per second
            seed: Random seed
        """
        self.hugadb_dir = Path(hugadb_dir)
        self.sampling_rate = sampling_rate
        self.seed = seed
        np.random.seed(seed)

        # Load HuGaDB data
        print("Loading HuGaDB dataset...")
        self.hugadb_data = self._load_hugadb_data()
        print(f"[OK] Loaded {len(self.hugadb_data)} samples from HuGaDB")

        # Disease patterns
        self.disease_patterns = self._define_disease_patterns()

        # Severity levels
        self.severity_levels = {
            'mild': (0.2, 0.4),
            'moderate': (0.4, 0.7),
            'severe': (0.7, 0.95)
        }

        # Feature indices for targeted modifications
        self._define_feature_indices()

    def _load_hugadb_data(self) -> List[np.ndarray]:
        """Load all HuGaDB CSV files."""
        csv_files = list(self.hugadb_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.hugadb_dir}")

        all_data = []

        for csv_file in tqdm(csv_files[:50], desc="Loading HuGaDB files"):  # Limit to 50 files
            try:
                df = pd.read_csv(csv_file)

                # Remove index and activity columns
                feature_cols = [col for col in df.columns
                               if col not in ['Unnamed: 0', 'activity', 'label']]

                if len(feature_cols) == 38:
                    data = df[feature_cols].values.astype(np.float64)
                    all_data.append(data)
            except Exception as e:
                print(f"Warning: Could not load {csv_file.name}: {e}")
                continue

        return all_data

    def _define_feature_indices(self):
        """Define feature indices for targeted modifications."""
        # Accelerometer and gyroscope indices for each body part
        self.feature_map = {
            'right_foot': {'accel': [0, 1, 2], 'gyro': [3, 4, 5]},
            'right_shin': {'accel': [6, 7, 8], 'gyro': [9, 10, 11]},
            'right_thigh': {'accel': [12, 13, 14], 'gyro': [15, 16, 17]},
            'left_foot': {'accel': [18, 19, 20], 'gyro': [21, 22, 23]},
            'left_shin': {'accel': [24, 25, 26], 'gyro': [27, 28, 29]},
            'left_thigh': {'accel': [30, 31, 32], 'gyro': [33, 34, 35]},
            'emg': [36, 37]
        }

    def _define_disease_patterns(self) -> Dict:
        """Define disease characteristics."""
        return {
            0: {  # Parkinson's Disease
                'name': "Parkinson's Disease",
                'code': 'PD',
                'characteristics': {
                    'stride_amplitude_reduction': 0.65,  # Reduced stride
                    'step_frequency_increase': 1.15,  # Faster steps
                    'tremor_frequency': 5.0,  # 4-6 Hz resting tremor
                    'tremor_amplitude': 60.0,
                    'rigidity_factor': 0.75,  # Reduced variability
                    'arm_swing_reduction': 0.4,  # Reduced upper limb
                    'shuffling_factor': 0.7  # Reduced foot clearance
                }
            },
            1: {  # Huntington's Disease
                'name': "Huntington's Disease",
                'code': 'HD',
                'characteristics': {
                    'chorea_intensity': 0.8,  # Involuntary movements
                    'movement_irregularity': 1.6,  # High variability
                    'jerky_movements': 0.85,
                    'coordination_loss': 0.5,
                    'random_acceleration_bursts': 250.0
                }
            },
            2: {  # Cerebral Palsy
                'name': "Cerebral Palsy",
                'code': 'CP',
                'characteristics': {
                    'spasticity': 0.85,  # Increased muscle tone
                    'asymmetry': 0.7,  # Left-right imbalance
                    'toe_walking': 0.75,  # Forefoot strike
                    'scissoring': 0.65,  # Hip adduction
                    'cocontraction': 0.8  # Simultaneous muscle activation
                }
            },
            3: {  # Multiple Sclerosis
                'name': "Multiple Sclerosis",
                'code': 'MS',
                'characteristics': {
                    'foot_drop': 0.6,  # Reduced dorsiflexion
                    'fatigue_factor': 0.75,  # Progressive decline
                    'balance_instability': 0.7,
                    'sensory_ataxia': 0.65,
                    'variability': 0.7
                }
            },
            4: {  # Ataxia
                'name': "Ataxia",
                'code': 'ATAX',
                'characteristics': {
                    'base_width_increase': 1.7,  # Wide stance
                    'coordination_loss': 0.35,  # Poor timing
                    'overshooting': 0.8,  # Dysmetria
                    'irregular_rhythm': 0.85,
                    'balance_deficit': 0.3
                }
            },
            5: {  # Normal/Healthy
                'name': "Normal/Healthy",
                'code': 'NORM',
                'characteristics': {
                    'natural_variability': 0.05  # Minimal noise
                }
            }
        }

    def get_random_hugadb_sample(self, window_size: int = 128) -> np.ndarray:
        """
        Extract random window from HuGaDB data.

        Args:
            window_size: Number of timesteps

        Returns:
            Array of shape (window_size, 38)
        """
        # Select random file
        file_data = self.hugadb_data[np.random.randint(0, len(self.hugadb_data))]

        # Extract random window
        if len(file_data) >= window_size:
            start_idx = np.random.randint(0, len(file_data) - window_size + 1)
            return file_data[start_idx:start_idx + window_size]
        else:
            # Pad if too short
            padding = window_size - len(file_data)
            return np.vstack([file_data, np.zeros((padding, 38))])

    def apply_parkinsons_pattern(self, data: np.ndarray, severity: float) -> np.ndarray:
        """Apply Parkinson's disease transformations."""
        modified = data.copy()
        pattern = self.disease_patterns[0]['characteristics']

        # 1. Reduce stride amplitude (shuffling gait)
        reduction = pattern['stride_amplitude_reduction'] + (1 - pattern['stride_amplitude_reduction']) * (1 - severity)
        modified[:, :36] *= reduction

        # 2. Add resting tremor (4-6 Hz)
        t = np.arange(len(data)) / self.sampling_rate
        tremor_freq = pattern['tremor_frequency']
        tremor_amp = pattern['tremor_amplitude'] * severity
        tremor = tremor_amp * np.sin(2 * np.pi * tremor_freq * t)

        # Apply tremor mainly to hands/arms (thigh sensors as proxy)
        for part in ['right_thigh', 'left_thigh']:
            for axis in self.feature_map[part]['accel']:
                modified[:, axis] += tremor * np.random.uniform(0.8, 1.2)

        # 3. Reduce arm swing (upper body rigidity)
        for part in ['right_thigh', 'left_thigh']:
            modified[:, self.feature_map[part]['accel']] *= pattern['arm_swing_reduction']

        # 4. Add rigidity (reduced natural variability)
        rigidity = pattern['rigidity_factor'] * severity
        natural_noise = np.random.normal(0, 15 * (1 - rigidity), modified.shape)
        modified += natural_noise

        # 5. Reduce foot clearance (shuffling)
        for part in ['right_foot', 'left_foot']:
            # Reduce vertical acceleration
            modified[:, self.feature_map[part]['accel'][1]] *= pattern['shuffling_factor']

        return modified

    def apply_huntingtons_pattern(self, data: np.ndarray, severity: float) -> np.ndarray:
        """Apply Huntington's disease (chorea) transformations."""
        modified = data.copy()
        pattern = self.disease_patterns[1]['characteristics']

        # 1. Add choreic movements (random jerky motions)
        chorea_noise = severity * pattern['random_acceleration_bursts'] * np.random.randn(*data.shape)
        modified += chorea_noise

        # 2. Add sudden involuntary movements
        num_jerks = int(15 * severity)
        for _ in range(num_jerks):
            jerk_idx = np.random.randint(5, len(data) - 5)
            jerk_magnitude = severity * 350 * np.random.randn(1, 38)
            modified[jerk_idx-3:jerk_idx+3] += jerk_magnitude

        # 3. Increase overall movement variability
        variability_factor = 1 + pattern['movement_irregularity'] * severity * np.random.randn(*data.shape) * 0.25
        modified *= np.clip(variability_factor, 0.3, 2.0)

        # 4. Add irregular rhythm
        irregular_scaling = 1 + severity * 0.3 * np.sin(np.linspace(0, 8*np.pi, len(data)))
        modified *= irregular_scaling[:, np.newaxis]

        return modified

    def apply_cerebral_palsy_pattern(self, data: np.ndarray, severity: float) -> np.ndarray:
        """Apply Cerebral Palsy transformations."""
        modified = data.copy()
        pattern = self.disease_patterns[2]['characteristics']

        # 1. Add spasticity (increased muscle tone)
        spasticity_factor = 1 + pattern['spasticity'] * severity * 0.35
        modified[:, :36] *= spasticity_factor

        # 2. Add asymmetry (one side affected more)
        asymmetry = pattern['asymmetry'] * severity
        # Reduce left side movement
        for part in ['left_foot', 'left_shin', 'left_thigh']:
            for key in ['accel', 'gyro']:
                modified[:, self.feature_map[part][key]] *= (1 - asymmetry * 0.4)

        # 3. Toe walking (increased forefoot contact)
        for part in ['right_foot', 'left_foot']:
            # Increase acceleration in toe-off phase
            modified[:, self.feature_map[part]['accel'][1]] *= (1 + pattern['toe_walking'] * severity * 0.5)

        # 4. Scissoring gait (increased lateral hip movement)
        for part in ['right_thigh', 'left_thigh']:
            # Add lateral oscillation
            lateral_idx = self.feature_map[part]['accel'][2]  # Z-axis
            scissoring = severity * 120 * np.sin(np.linspace(0, 6*np.pi, len(data)))
            modified[:, lateral_idx] += scissoring

        # 5. Co-contraction (simultaneous agonist-antagonist activation)
        modified[:, 36:] *= (1 + pattern['cocontraction'] * severity * 0.6)  # Increase EMG

        return modified

    def apply_multiple_sclerosis_pattern(self, data: np.ndarray, severity: float) -> np.ndarray:
        """Apply Multiple Sclerosis transformations."""
        modified = data.copy()
        pattern = self.disease_patterns[3]['characteristics']

        # 1. Foot drop (reduced ankle dorsiflexion)
        for part in ['right_foot', 'left_foot']:
            foot_drop_factor = 1 - pattern['foot_drop'] * severity * 0.45
            modified[:, self.feature_map[part]['accel'][1]] *= foot_drop_factor

        # 2. Progressive fatigue (decline over time)
        fatigue_curve = np.linspace(1.0, 1.0 - pattern['fatigue_factor'] * severity * 0.35, len(data))
        modified *= fatigue_curve[:, np.newaxis]

        # 3. Balance instability
        balance_noise = severity * 95 * np.random.randn(*data.shape)
        modified += balance_noise

        # 4. Sensory ataxia (increased variability)
        sensory_noise = severity * 70 * np.random.randn(*data.shape)
        modified += sensory_noise

        # 5. Variable gait pattern
        variable_scaling = 1 + severity * 0.2 * np.random.randn(len(data), 1)
        modified *= variable_scaling

        return modified

    def apply_ataxia_pattern(self, data: np.ndarray, severity: float) -> np.ndarray:
        """Apply Ataxia transformations."""
        modified = data.copy()
        pattern = self.disease_patterns[4]['characteristics']

        # 1. Wide-based gait (increased lateral movement)
        for part in ['right_foot', 'right_shin', 'left_foot', 'left_shin']:
            lateral_idx = self.feature_map[part]['accel'][2]  # Z-axis
            modified[:, lateral_idx] *= pattern['base_width_increase']

        # 2. Poor coordination (irregular timing)
        coordination_noise = severity * 180 * np.random.randn(*data.shape)
        modified += coordination_noise

        # 3. Dysmetria (overshooting movements)
        overshoot_indices = np.random.choice(len(data), int(len(data) * 0.25 * severity), replace=False)
        modified[overshoot_indices] *= (1 + pattern['overshooting'] * severity * 0.6)

        # 4. Irregular rhythm and timing
        irregular_phase = np.cumsum(1 + severity * 0.3 * np.random.randn(len(data)))
        irregular_phase = (irregular_phase - irregular_phase.min()) / (irregular_phase.max() - irregular_phase.min())

        # 5. Add jerkiness (reduced smoothness)
        jerk = np.diff(modified, axis=0, prepend=modified[0:1])
        modified += severity * 110 * jerk

        return modified

    def apply_disease_pattern(self, data: np.ndarray, disease_id: int, severity: float) -> np.ndarray:
        """
        Apply disease-specific pattern to base data.

        Args:
            data: Base HuGaDB data (window_size, 38)
            disease_id: Disease class 0-5
            severity: Severity level 0.0-1.0

        Returns:
            Modified data with disease characteristics
        """
        if disease_id == 0:
            return self.apply_parkinsons_pattern(data, severity)
        elif disease_id == 1:
            return self.apply_huntingtons_pattern(data, severity)
        elif disease_id == 2:
            return self.apply_cerebral_palsy_pattern(data, severity)
        elif disease_id == 3:
            return self.apply_multiple_sclerosis_pattern(data, severity)
        elif disease_id == 4:
            return self.apply_ataxia_pattern(data, severity)
        elif disease_id == 5:
            # Normal/Healthy - just add minimal natural variability
            return data + np.random.normal(0, 18, data.shape)
        else:
            raise ValueError(f"Invalid disease_id: {disease_id}")

    def generate_disease_samples(self,
                                 disease_id: int,
                                 num_samples: int = 1000,
                                 window_size: int = 128,
                                 severity_distribution: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate samples for a specific disease.

        Args:
            disease_id: Disease class 0-5
            num_samples: Number of samples
            window_size: Window size (timesteps)
            severity_distribution: Severity distribution

        Returns:
            (data, labels, severities)
        """
        if severity_distribution is None:
            severity_distribution = {'mild': 0.3, 'moderate': 0.5, 'severe': 0.2}

        disease_name = self.disease_patterns[disease_id]['name']
        print(f"\nGenerating {num_samples} samples for {disease_name}...")

        all_data = []
        all_labels = []
        all_severities = []

        # Calculate samples per severity
        severity_samples = {
            level: int(num_samples * ratio)
            for level, ratio in severity_distribution.items()
        }

        for severity_level, num in tqdm(severity_samples.items(), desc="Severity levels"):
            severity_range = self.severity_levels[severity_level]

            for _ in range(num):
                # Get real HuGaDB sample
                base_data = self.get_random_hugadb_sample(window_size)

                # Random severity
                severity = np.random.uniform(*severity_range)

                # Apply disease pattern
                disease_data = self.apply_disease_pattern(base_data, disease_id, severity)

                all_data.append(disease_data)
                all_labels.append(disease_id)
                all_severities.append(severity * 100)

        return np.array(all_data), np.array(all_labels), np.array(all_severities)

    def generate_complete_dataset(self,
                                  samples_per_disease: int = 1000,
                                  window_size: int = 128,
                                  output_dir: str = 'data/disease_data') -> Dict:
        """
        Generate complete multi-class disease dataset.

        Args:
            samples_per_disease: Samples per disease
            window_size: Window size
            output_dir: Output directory

        Returns:
            Dataset metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("ENHANCED DISEASE DATASET GENERATION (Using Real HuGaDB Data)")
        print("="*80)
        print(f"Samples per disease: {samples_per_disease}")
        print(f"Window size: {window_size}")
        print(f"Total samples: {samples_per_disease * 6}")
        print(f"Output: {output_dir}")
        print("="*80)

        all_data = []
        all_labels = []
        all_severities = []

        # Generate for each disease
        for disease_id in range(6):
            data, labels, severities = self.generate_disease_samples(
                disease_id=disease_id,
                num_samples=samples_per_disease,
                window_size=window_size
            )
            all_data.append(data)
            all_labels.append(labels)
            all_severities.append(severities)

        # Combine and shuffle
        X = np.vstack(all_data)
        y = np.concatenate(all_labels)
        severity = np.concatenate(all_severities)

        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        severity = severity[indices]

        # Save
        print("\nSaving dataset...")
        np.save(output_path / 'disease_data.npy', X)
        np.save(output_path / 'disease_labels.npy', y)
        np.save(output_path / 'disease_severity.npy', severity)

        # Metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'generator_version': '2.0',
            'base_dataset': 'HuGaDB',
            'total_samples': int(len(X)),
            'samples_per_disease': samples_per_disease,
            'window_size': window_size,
            'num_features': 38,
            'num_diseases': 6,
            'diseases': {
                i: {
                    'name': self.disease_patterns[i]['name'],
                    'code': self.disease_patterns[i]['code'],
                    'sample_count': int(np.sum(y == i))
                }
                for i in range(6)
            },
            'data_shape': list(X.shape),
            'sampling_rate': self.sampling_rate,
            'severity_range': [float(severity.min()), float(severity.max())],
        }

        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"\n[OK] Dataset generated successfully!")
        print(f"  Data shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")
        print(f"  Severity shape: {severity.shape}")
        print(f"\nClass Distribution:")
        for i in range(6):
            disease_name = self.disease_patterns[i]['name']
            count = np.sum(y == i)
            print(f"  {i}: {disease_name} - {count} samples")

        print("\n" + "="*80)
        print("GENERATION COMPLETE!")
        print("="*80 + "\n")

        return metadata


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Disease Dataset Generator')
    parser.add_argument('--hugadb_dir', type=str, default='../data/raw',
                       help='HuGaDB data directory')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Samples per disease')
    parser.add_argument('--window_size', type=int, default=128,
                       help='Window size')
    parser.add_argument('--output', type=str, default='../data/disease_data',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Generate
    generator = EnhancedDiseaseDataGenerator(
        hugadb_dir=args.hugadb_dir,
        seed=args.seed
    )

    metadata = generator.generate_complete_dataset(
        samples_per_disease=args.samples,
        window_size=args.window_size,
        output_dir=args.output
    )

    print(f"\nDataset saved to: {args.output}")
    print("\nFiles created:")
    print("  - disease_data.npy")
    print("  - disease_labels.npy")
    print("  - disease_severity.npy")
    print("  - metadata.json")


if __name__ == "__main__":
    main()
