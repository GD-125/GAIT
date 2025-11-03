"""
Synthetic Gait Sensor Data Generator
=====================================
Generates realistic sensor data for testing gait detection models.
Simulates 6-location wearable sensor data (accelerometer + gyroscope + EMG).

Usage:
    python scripts/synthetic_sensor_generator.py --patients 5 --duration 60 --output data/synthetic/

For doctors: This simulates the exact sensor setup needed for real data collection.
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
import json

class GaitSensorSimulator:
    """
    Simulates realistic gait sensor data based on biomechanical principles.

    Sensor Locations (6 total):
    - Right: Foot, Shin, Thigh
    - Left: Foot, Shin, Thigh

    Each location has:
    - 3-axis accelerometer (X, Y, Z) in m/s²
    - 3-axis gyroscope (X, Y, Z) in deg/s

    Additional:
    - 2 EMG channels (right, left) in μV
    """

    def __init__(self, sampling_rate=100):
        """
        Args:
            sampling_rate: Hz (default 100 Hz = 10ms per sample)
        """
        self.fs = sampling_rate
        self.dt = 1.0 / sampling_rate

        # Activity definitions
        self.activities = {
            'walking': {
                'gait_class': 1,
                'cadence_range': (90, 120),  # steps/minute
                'acceleration_magnitude': (8000, 15000),  # raw sensor units
                'gyro_magnitude': (100, 300),
                'emg_magnitude': (80, 150)
            },
            'jogging': {
                'gait_class': 1,
                'cadence_range': (140, 180),
                'acceleration_magnitude': (12000, 25000),
                'gyro_magnitude': (200, 500),
                'emg_magnitude': (150, 250)
            },
            'running': {
                'gait_class': 1,
                'cadence_range': (160, 200),
                'acceleration_magnitude': (15000, 35000),
                'gyro_magnitude': (300, 700),
                'emg_magnitude': (200, 350)
            },
            'stairs_up': {
                'gait_class': 1,
                'cadence_range': (60, 90),
                'acceleration_magnitude': (10000, 20000),
                'gyro_magnitude': (150, 400),
                'emg_magnitude': (120, 220)
            },
            'stairs_down': {
                'gait_class': 1,
                'cadence_range': (70, 100),
                'acceleration_magnitude': (9000, 18000),
                'gyro_magnitude': (120, 350),
                'emg_magnitude': (100, 180)
            },
            'sitting': {
                'gait_class': 0,
                'cadence_range': (0, 5),
                'acceleration_magnitude': (1000, 3000),
                'gyro_magnitude': (5, 30),
                'emg_magnitude': (10, 40)
            },
            'standing': {
                'gait_class': 0,
                'cadence_range': (0, 5),
                'acceleration_magnitude': (2000, 5000),
                'gyro_magnitude': (10, 50),
                'emg_magnitude': (20, 60)
            },
            'lying': {
                'gait_class': 0,
                'cadence_range': (0, 2),
                'acceleration_magnitude': (500, 2000),
                'gyro_magnitude': (2, 20),
                'emg_magnitude': (5, 25)
            }
        }

    def generate_gait_cycle(self, activity, duration_seconds):
        """Generate one activity segment with realistic gait patterns"""

        n_samples = int(duration_seconds * self.fs)
        activity_params = self.activities[activity]

        # Calculate gait frequency (Hz)
        cadence = np.random.uniform(*activity_params['cadence_range'])  # steps/min
        gait_freq = cadence / 60.0  # steps/sec (Hz)

        # Time vector
        t = np.linspace(0, duration_seconds, n_samples)

        # Create data dictionary
        data = {}

        # Generate for each sensor location
        locations = ['right_foot', 'right_shin', 'right_thigh',
                     'left_foot', 'left_shin', 'left_thigh']

        for loc in locations:
            # Determine phase offset for left vs right
            phase_offset = np.pi if 'left' in loc else 0

            # Location-specific amplitude modifiers
            if 'foot' in loc:
                acc_mod = 1.2
                gyro_mod = 1.5
            elif 'shin' in loc:
                acc_mod = 1.0
                gyro_mod = 1.0
            else:  # thigh
                acc_mod = 0.8
                gyro_mod = 0.7

            # Generate accelerometer data (3 axes)
            acc_base = np.random.uniform(*activity_params['acceleration_magnitude'])

            # X-axis: forward-backward (main gait direction)
            acc_x = acc_base * acc_mod * np.sin(2 * np.pi * gait_freq * t + phase_offset)
            acc_x += np.random.normal(0, acc_base * 0.1, n_samples)  # noise

            # Y-axis: lateral movement
            acc_y = acc_base * 0.6 * acc_mod * np.sin(2 * np.pi * gait_freq * t + phase_offset + np.pi/4)
            acc_y += np.random.normal(0, acc_base * 0.08, n_samples)

            # Z-axis: vertical (gravity + movement)
            gravity_component = 9800 if activity_params['gait_class'] == 0 else 8000
            acc_z = gravity_component + acc_base * 0.5 * acc_mod * np.sin(2 * np.pi * gait_freq * t + phase_offset + np.pi/2)
            acc_z += np.random.normal(0, acc_base * 0.1, n_samples)

            data[f'accelerometer_{loc}_x'] = acc_x.astype(int)
            data[f'accelerometer_{loc}_y'] = acc_y.astype(int)
            data[f'accelerometer_{loc}_z'] = acc_z.astype(int)

            # Generate gyroscope data (3 axes)
            gyro_base = np.random.uniform(*activity_params['gyro_magnitude'])

            # Rotation around each axis
            gyro_x = gyro_base * gyro_mod * np.sin(2 * np.pi * gait_freq * t + phase_offset + np.pi/6)
            gyro_x += np.random.normal(0, gyro_base * 0.15, n_samples)

            gyro_y = gyro_base * 0.7 * gyro_mod * np.cos(2 * np.pi * gait_freq * t + phase_offset)
            gyro_y += np.random.normal(0, gyro_base * 0.12, n_samples)

            gyro_z = gyro_base * 0.5 * gyro_mod * np.sin(2 * np.pi * gait_freq * t + phase_offset + np.pi/3)
            gyro_z += np.random.normal(0, gyro_base * 0.1, n_samples)

            data[f'gyroscope_{loc}_x'] = gyro_x.astype(int)
            data[f'gyroscope_{loc}_y'] = gyro_y.astype(int)
            data[f'gyroscope_{loc}_z'] = gyro_z.astype(int)

        # Generate EMG data (muscle activation)
        emg_base = np.random.uniform(*activity_params['emg_magnitude'])

        # Right EMG
        emg_right = emg_base * np.abs(np.sin(2 * np.pi * gait_freq * t))
        emg_right += np.random.normal(0, emg_base * 0.2, n_samples)
        data['EMG_right'] = np.clip(emg_right, 0, 500).astype(int)

        # Left EMG (phase-shifted)
        emg_left = emg_base * np.abs(np.sin(2 * np.pi * gait_freq * t + np.pi))
        emg_left += np.random.normal(0, emg_base * 0.2, n_samples)
        data['EMG_left'] = np.clip(emg_left, 0, 500).astype(int)

        # Activity label
        data['activity'] = [activity] * n_samples

        return pd.DataFrame(data)

    def generate_patient_session(self, patient_id, session_id, activity_sequence=None,
                                 duration_per_activity=10):
        """
        Generate a complete patient session with multiple activities.

        Args:
            patient_id: Patient identifier (1-100)
            session_id: Session number (0-20)
            activity_sequence: List of activities, or None for random
            duration_per_activity: Seconds per activity

        Returns:
            DataFrame with complete session data
        """

        if activity_sequence is None:
            # Default mixed activity sequence
            activity_sequence = [
                'sitting', 'standing', 'walking', 'standing',
                'jogging', 'walking', 'stairs_up', 'stairs_down',
                'walking', 'sitting'
            ]

        print(f"Generating Patient {patient_id:02d} Session {session_id:02d}")
        print(f"  Activities: {' -> '.join(activity_sequence)}")

        session_data = []
        for activity in activity_sequence:
            segment = self.generate_gait_cycle(activity, duration_per_activity)
            session_data.append(segment)

        # Concatenate all segments
        full_session = pd.concat(session_data, ignore_index=True)

        # Reorder columns to match HuGaDB format
        column_order = []
        for loc in ['right_foot', 'right_shin', 'right_thigh',
                    'left_foot', 'left_shin', 'left_thigh']:
            column_order.extend([
                f'accelerometer_{loc}_x',
                f'accelerometer_{loc}_y',
                f'accelerometer_{loc}_z',
                f'gyroscope_{loc}_x',
                f'gyroscope_{loc}_y',
                f'gyroscope_{loc}_z'
            ])
        column_order.extend(['EMG_right', 'EMG_left', 'activity'])

        return full_session[column_order]

    def generate_dataset(self, num_patients=5, sessions_per_patient=3,
                        output_dir='data/synthetic', variety='mixed'):
        """
        Generate a complete synthetic dataset.

        Args:
            num_patients: Number of patients to simulate
            sessions_per_patient: Sessions per patient
            output_dir: Where to save CSV files
            variety: 'mixed', 'gait_only', 'non_gait_only', or 'custom'

        Returns:
            metadata dictionary
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Activity sequences for different varieties
        sequences = {
            'mixed': [
                ['sitting', 'standing', 'walking', 'jogging', 'walking', 'sitting'],
                ['standing', 'walking', 'stairs_up', 'stairs_down', 'walking', 'standing'],
                ['lying', 'sitting', 'standing', 'walking', 'running', 'walking', 'sitting'],
            ],
            'gait_only': [
                ['walking'] * 6,
                ['jogging', 'walking', 'jogging', 'walking'],
                ['walking', 'stairs_up', 'stairs_down', 'walking'],
            ],
            'non_gait_only': [
                ['sitting', 'standing', 'sitting', 'lying'],
                ['standing', 'sitting', 'standing'],
                ['lying', 'sitting', 'lying'],
            ]
        }

        metadata = {
            'generation_date': datetime.now().isoformat(),
            'num_patients': num_patients,
            'sessions_per_patient': sessions_per_patient,
            'sampling_rate_hz': self.fs,
            'variety': variety,
            'files': []
        }

        file_count = 0
        for patient_id in range(1, num_patients + 1):
            for session_id in range(sessions_per_patient):
                # Select activity sequence
                if variety in sequences:
                    activity_seq = sequences[variety][session_id % len(sequences[variety])]
                else:
                    activity_seq = None  # use default

                # Generate session
                session_df = self.generate_patient_session(
                    patient_id,
                    session_id,
                    activity_sequence=activity_seq,
                    duration_per_activity=10
                )

                # Save to CSV
                filename = f"synthetic_patient_{patient_id:02d}_session_{session_id:02d}.csv"
                filepath = output_path / filename
                session_df.to_csv(filepath, index=False)

                # Record metadata
                metadata['files'].append({
                    'filename': filename,
                    'patient_id': patient_id,
                    'session_id': session_id,
                    'num_samples': len(session_df),
                    'duration_seconds': len(session_df) / self.fs,
                    'activities': list(session_df['activity'].unique())
                })

                file_count += 1
                print(f"  ✓ Saved: {filename} ({len(session_df)} samples)")

        # Save metadata
        metadata_file = output_path / 'dataset_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Dataset generation complete!")
        print(f"  Total files: {file_count}")
        print(f"  Output directory: {output_path.absolute()}")
        print(f"  Metadata: {metadata_file.absolute()}")

        return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic gait sensor data for testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5 patients, 3 sessions each, mixed activities
  python scripts/synthetic_sensor_generator.py --patients 5 --sessions 3

  # Generate gait-only data for testing gait detection
  python scripts/synthetic_sensor_generator.py --patients 10 --variety gait_only

  # Generate large dataset with custom output
  python scripts/synthetic_sensor_generator.py --patients 20 --sessions 5 --output data/test_dataset/

Activity Types:
  Gait: walking, jogging, running, stairs_up, stairs_down
  Non-Gait: sitting, standing, lying
        """
    )

    parser.add_argument('--patients', type=int, default=5,
                       help='Number of patients to simulate (default: 5)')
    parser.add_argument('--sessions', type=int, default=3,
                       help='Sessions per patient (default: 3)')
    parser.add_argument('--output', type=str, default='data/synthetic',
                       help='Output directory (default: data/synthetic)')
    parser.add_argument('--variety', type=str, default='mixed',
                       choices=['mixed', 'gait_only', 'non_gait_only'],
                       help='Type of activities to generate (default: mixed)')
    parser.add_argument('--sampling-rate', type=int, default=100,
                       help='Sampling rate in Hz (default: 100)')

    args = parser.parse_args()

    print("=" * 70)
    print("SYNTHETIC GAIT SENSOR DATA GENERATOR")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Patients: {args.patients}")
    print(f"  Sessions per patient: {args.sessions}")
    print(f"  Activity variety: {args.variety}")
    print(f"  Sampling rate: {args.sampling_rate} Hz")
    print(f"  Output: {args.output}")
    print("=" * 70)
    print()

    # Create simulator
    simulator = GaitSensorSimulator(sampling_rate=args.sampling_rate)

    # Generate dataset
    metadata = simulator.generate_dataset(
        num_patients=args.patients,
        sessions_per_patient=args.sessions,
        output_dir=args.output,
        variety=args.variety
    )

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print(f"1. Verify data: Check {args.output}/ for CSV files")
    print(f"2. Test with model: python efficient_run.py --data-dir {args.output}")
    print(f"3. Upload to dashboard: Run dashboard and upload any CSV file")
    print("=" * 70)


if __name__ == "__main__":
    main()
