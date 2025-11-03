# tests/test_pipeline.py
"""
Integration Tests for Complete Gait Detection Pipeline
Tests the full workflow from data loading to model prediction.
"""

import unittest
import numpy as np
import torch
import tempfile
import os
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import (
    GaitDataLoader,
    DataCleaner,
    SignalFilter,
    DataNormalizer,
    DataSegmenter
)
from preprocessing.pipeline import PreprocessingPipeline
from models import CNN_BiLSTM_GaitDetector, get_device
from training import create_data_loaders, MetricsCalculator
from features import TimeDomainFeatureExtractor


class TestPreprocessingPipeline(unittest.TestCase):
    """Test complete preprocessing pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv = self._create_test_csv()
        
    def _create_test_csv(self):
        """Create a temporary CSV file for testing."""
        num_samples = 500
        data = {
            '': range(num_samples),
            'activity': ['walk'] * 250 + ['sit'] * 250
        }
        
        # Add sensor columns
        for location in ['right_foot', 'right_shin', 'right_thigh', 
                        'left_foot', 'left_shin', 'left_thigh']:
            for axis in ['x', 'y', 'z']:
                data[f'accelerometer_{location}_{axis}'] = np.random.randint(-1000, 1000, num_samples)
                data[f'gyroscope_{location}_{axis}'] = np.random.randint(-500, 500, num_samples)
        
        data['EMG_right'] = np.random.randint(0, 100, num_samples)
        data['EMG_left'] = np.random.randint(0, 100, num_samples)
        
        df = pd.DataFrame(data)
        filepath = os.path.join(self.temp_dir, 'test_data.csv')
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        pipeline = PreprocessingPipeline(
            sampling_rate=100.0,
            window_size=128,
            overlap=0.5,
            normalization_method='zscore',
            filter_type='sensor_specific',
            balance_method='none'
        )
        
        # Run preprocessing
        windowed_data, windowed_labels = pipeline.preprocess_single_file(
            self.test_csv, data_dir=self.temp_dir
        )
        
        # Check output shapes
        self.assertEqual(windowed_data.shape[1], 128)  # Window size
        self.assertEqual(windowed_data.shape[2], 38)   # Features
        self.assertEqual(len(windowed_labels), len(windowed_data))
        
        # Check labels are binary
        self.assertTrue(np.all((windowed_labels == 0) | (windowed_labels == 1)))
        
        # Check normalization (mean ≈ 0)
        mean_per_feature = windowed_data.mean(axis=(0, 1))
        self.assertTrue(np.allclose(mean_per_feature, 0, atol=0.5))
    
    def test_pipeline_save_load(self):
        """Test saving and loading preprocessed data."""
        pipeline = PreprocessingPipeline(
            sampling_rate=100.0,
            window_size=128,
            overlap=0.5
        )
        
        # Preprocess
        windowed_data, windowed_labels = pipeline.preprocess_single_file(
            self.test_csv, data_dir=self.temp_dir
        )
        
        # Save
        output_dir = os.path.join(self.temp_dir, 'processed')
        pipeline.save_preprocessed_data(windowed_data, windowed_labels, output_dir)
        
        # Check files exist
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'windowed_data.npy')))
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'windowed_labels.npy')))
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'normalizer.pkl')))
        
        # Load
        loaded_data, loaded_labels, config = pipeline.load_preprocessed_data(output_dir)
        
        # Verify loaded data matches
        self.assertTrue(np.array_equal(windowed_data, loaded_data))
        self.assertTrue(np.array_equal(windowed_labels, loaded_labels))
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic data
        self.num_samples = 200
        self.window_size = 128
        self.num_features = 38
        
        self.X_train = np.random.randn(150, self.window_size, self.num_features)
        self.y_train = np.random.randint(0, 2, 150)
        self.X_val = np.random.randn(30, self.window_size, self.num_features)
        self.y_val = np.random.randint(0, 2, 30)
        self.X_test = np.random.randn(20, self.window_size, self.num_features)
        self.y_test = np.random.randint(0, 2, 20)
    
    def test_data_loader_creation(self):
        """Test creating PyTorch data loaders."""
        train_loader, val_loader = create_data_loaders(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            batch_size=16
        )
        
        self.assertGreater(len(train_loader), 0)
        self.assertGreater(len(val_loader), 0)
        
        # Test iterating through loader
        for batch_X, batch_y in train_loader:
            self.assertEqual(batch_X.shape[1], self.window_size)
            self.assertEqual(batch_X.shape[2], self.num_features)
            break
    
    def test_model_training_step(self):
        """Test single training step."""
        device = torch.device('cpu')
        model = CNN_BiLSTM_GaitDetector(
            input_features=self.num_features,
            seq_length=self.window_size
        ).to(device)
        
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create data loader
        train_loader, _ = create_data_loaders(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            batch_size=16
        )
        
        # Training step
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Check loss is finite
            self.assertTrue(torch.isfinite(loss))
            break
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        device = torch.device('cpu')
        model = CNN_BiLSTM_GaitDetector(
            input_features=self.num_features,
            seq_length=self.window_size
        ).to(device)
        
        model.eval()
        
        # Convert test data to tensors
        X_test_tensor = torch.FloatTensor(self.X_test).to(device)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            predictions = (outputs > 0.5).float().cpu().numpy()
        
        # Check predictions shape
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Check predictions are binary
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))
        
        # Calculate metrics
        calculator = MetricsCalculator()
        metrics = calculator.calculate_all_metrics(
            y_true=self.y_test,
            y_pred=predictions.flatten(),
            y_prob=outputs.cpu().numpy().flatten()
        )
        
        # Check metrics exist
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
    
    def test_feature_extraction_integration(self):
        """Test feature extraction with model."""
        # Extract time-domain features
        extractor = TimeDomainFeatureExtractor(
            feature_list=['mean', 'std', 'rms']
        )
        
        features = extractor.extract_batch(self.X_train, verbose=False)
        
        # Check output shape
        expected_features = self.num_features * 3  # 3 features per channel
        self.assertEqual(features.shape, (len(self.X_train), expected_features))


class TestDataFlowIntegration(unittest.TestCase):
    """Test data flow through entire system."""
    
    def test_preprocessing_to_training(self):
        """Test data flow from preprocessing to training."""
        # Create synthetic raw data
        raw_data = np.random.randn(1000, 38)
        raw_labels = np.array(['walk'] * 500 + ['sit'] * 500)
        
        # Step 1: Clean
        cleaner = DataCleaner()
        cleaned_data, _ = cleaner.clean_data(raw_data, 
                                             missing_method='interpolate',
                                             outlier_method='clip')
        
        # Step 2: Filter
        filter_obj = SignalFilter(sampling_rate=100.0)
        filtered_data = filter_obj.apply_filter(cleaned_data, method='lowpass', cutoff_freq=20.0)
        
        # Step 3: Normalize
        normalizer = DataNormalizer(method='zscore')
        normalized_data = normalizer.fit_transform(filtered_data)
        
        # Step 4: Segment
        segmenter = DataSegmenter(window_size=128, overlap=0.5)
        binary_labels = segmenter.create_binary_labels(raw_labels)
        windowed_data, windowed_labels, _ = segmenter.segment_data(
            normalized_data, binary_labels
        )
        
        # Step 5: Create data loaders
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            windowed_data, windowed_labels, test_size=0.2, random_state=42
        )
        
        train_loader, val_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size=16
        )
        
        # Step 6: Test with model
        device = torch.device('cpu')
        model = CNN_BiLSTM_GaitDetector(
            input_features=38,
            seq_length=128
        ).to(device)
        
        # Forward pass
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Check outputs are valid
            self.assertEqual(outputs.shape[0], inputs.shape[0])
            self.assertTrue(torch.all(outputs >= 0))
            self.assertTrue(torch.all(outputs <= 1))
            break
    
    def test_reproducibility(self):
        """Test that pipeline produces reproducible results."""
        # Set seed
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create data
        data = np.random.randn(500, 38)
        
        # Normalize twice
        normalizer1 = DataNormalizer(method='zscore')
        result1 = normalizer1.fit_transform(data)
        
        normalizer2 = DataNormalizer(method='zscore')
        result2 = normalizer2.fit_transform(data)
        
        # Results should be identical
        self.assertTrue(np.allclose(result1, result2))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
