# tests/test_model.py
"""
Unit Tests for Model Module
Tests CNN-BiLSTM model architecture, forward pass, and utility functions.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import (
    CNN_BiLSTM_GaitDetector,
    save_checkpoint,
    load_checkpoint,
    save_model_weights_only,
    load_model_weights_only,
    count_parameters,
    get_device,
    EarlyStopping
)


class TestCNN_BiLSTM_GaitDetector(unittest.TestCase):
    """Test cases for CNN-BiLSTM model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = CNN_BiLSTM_GaitDetector(
            input_features=38,
            seq_length=128,
            conv_filters=[64, 128, 256],
            kernel_sizes=[5, 5, 5],
            lstm_hidden_size=128,
            lstm_num_layers=2,
            fc_hidden_sizes=[256, 128],
            dropout=0.3
        )
        self.batch_size = 16
        self.seq_length = 128
        self.input_features = 38
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertEqual(self.model.input_features, 38)
        self.assertEqual(self.model.seq_length, 128)
    
    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        dummy_input = torch.randn(self.batch_size, self.seq_length, self.input_features)
        output = self.model(dummy_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Check output range [0, 1] due to sigmoid
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
    
    def test_model_parameters(self):
        """Test model has trainable parameters."""
        total_params = self.model.count_parameters()
        self.assertGreater(total_params, 0)
    
    def test_get_embeddings(self):
        """Test embedding extraction."""
        dummy_input = torch.randn(self.batch_size, self.seq_length, self.input_features)
        embeddings = self.model.get_embeddings(dummy_input)
        
        # Embeddings should be from the last FC layer
        expected_size = self.model.fc_layers[-1].out_features
        self.assertEqual(embeddings.shape, (self.batch_size, expected_size))
    
    def test_different_input_sizes(self):
        """Test model handles different batch sizes."""
        for batch_size in [1, 8, 32]:
            dummy_input = torch.randn(batch_size, self.seq_length, self.input_features)
            output = self.model(dummy_input)
            self.assertEqual(output.shape, (batch_size, 1))
    
    def test_model_on_device(self):
        """Test model can be moved to device."""
        device = get_device()
        model = self.model.to(device)
        dummy_input = torch.randn(self.batch_size, self.seq_length, self.input_features).to(device)
        
        output = model(dummy_input)
        self.assertEqual(output.device, device)


class TestModelUtils(unittest.TestCase):
    """Test cases for model utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = CNN_BiLSTM_GaitDetector(input_features=38, seq_length=128)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.temp_dir = tempfile.mkdtemp()
    
    def test_count_parameters(self):
        """Test parameter counting."""
        total_params, trainable_params = count_parameters(self.model)
        
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)  # All params should be trainable
    
    def test_save_load_checkpoint(self):
        """Test checkpoint saving and loading."""
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint.pt')
        
        # Save checkpoint
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=5,
            loss=0.5,
            accuracy=0.85,
            filepath=checkpoint_path
        )
        
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Create new model and optimizer
        new_model = CNN_BiLSTM_GaitDetector(input_features=38, seq_length=128)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        
        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path, new_model, new_optimizer)
        
        self.assertEqual(checkpoint['epoch'], 5)
        self.assertEqual(checkpoint['loss'], 0.5)
        self.assertEqual(checkpoint['accuracy'], 0.85)
    
    def test_save_load_weights_only(self):
        """Test saving and loading model weights only."""
        weights_path = os.path.join(self.temp_dir, 'test_weights.pt')
        
        # Save weights
        save_model_weights_only(self.model, weights_path)
        self.assertTrue(os.path.exists(weights_path))
        
        # Create new model and load weights
        new_model = CNN_BiLSTM_GaitDetector(input_features=38, seq_length=128)
        load_model_weights_only(new_model, weights_path)
        
        # Compare model parameters
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
    
    def test_early_stopping(self):
        """Test early stopping mechanism."""
        early_stopping = EarlyStopping(patience=3, mode='min')
        
        # Simulate improving losses
        losses = [0.5, 0.4, 0.3, 0.3, 0.3, 0.3]
        
        for i, loss in enumerate(losses):
            should_stop = early_stopping(loss)
            if i < 5:
                self.assertFalse(should_stop)
            else:
                self.assertTrue(should_stop)  # Should stop after 3 epochs without improvement
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)


class TestModelTraining(unittest.TestCase):
    """Test cases for model training components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = CNN_BiLSTM_GaitDetector(input_features=38, seq_length=128)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def test_loss_computation(self):
        """Test loss computation."""
        batch_size = 16
        dummy_input = torch.randn(batch_size, 128, 38)
        dummy_labels = torch.randint(0, 2, (batch_size, 1)).float()
        
        # Forward pass
        outputs = self.model(dummy_input)
        loss = self.criterion(outputs, dummy_labels)
        
        # Loss should be a scalar
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)
    
    def test_backward_pass(self):
        """Test backward pass and gradient computation."""
        batch_size = 16
        dummy_input = torch.randn(batch_size, 128, 38)
        dummy_labels = torch.randint(0, 2, (batch_size, 1)).float()
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(dummy_input)
        loss = self.criterion(outputs, dummy_labels)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_optimizer_step(self):
        """Test optimizer updates parameters."""
        batch_size = 16
        dummy_input = torch.randn(batch_size, 128, 38)
        dummy_labels = torch.randint(0, 2, (batch_size, 1)).float()
        
        # Store initial parameters
        initial_params = [p.clone() for p in self.model.parameters()]
        
        # Training step
        self.optimizer.zero_grad()
        outputs = self.model(dummy_input)
        loss = self.criterion(outputs, dummy_labels)
        loss.backward()
        self.optimizer.step()
        
        # Check parameters changed
        params_changed = False
        for p_initial, p_current in zip(initial_params, self.model.parameters()):
            if not torch.allclose(p_initial, p_current):
                params_changed = True
                break
        
        self.assertTrue(params_changed)


class TestModelArchitectureVariants(unittest.TestCase):
    """Test different model architecture configurations."""
    
    def test_different_conv_filters(self):
        """Test model with different convolution filters."""
        model = CNN_BiLSTM_GaitDetector(
            input_features=38,
            seq_length=128,
            conv_filters=[32, 64],
            kernel_sizes=[3, 3]
        )
        
        dummy_input = torch.randn(8, 128, 38)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (8, 1))
    
    def test_different_lstm_config(self):
        """Test model with different LSTM configuration."""
        model = CNN_BiLSTM_GaitDetector(
            input_features=38,
            seq_length=128,
            lstm_hidden_size=64,
            lstm_num_layers=1
        )
        
        dummy_input = torch.randn(8, 128, 38)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (8, 1))
    
    def test_no_residual_connections(self):
        """Test model without residual connections."""
        model = CNN_BiLSTM_GaitDetector(
            input_features=38,
            seq_length=128,
            use_residual=False
        )
        
        dummy_input = torch.randn(8, 128, 38)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (8, 1))
    
    def test_no_batch_norm(self):
        """Test model without batch normalization."""
        model = CNN_BiLSTM_GaitDetector(
            input_features=38,
            seq_length=128,
            use_batch_norm=False
        )
        
        dummy_input = torch.randn(8, 128, 38)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (8, 1))


class TestModelEvalMode(unittest.TestCase):
    """Test model behavior in evaluation mode."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = CNN_BiLSTM_GaitDetector(
            input_features=38,
            seq_length=128,
            dropout=0.5
        )
    
    def test_train_mode(self):
        """Test model in training mode."""
        self.model.train()
        self.assertTrue(self.model.training)
    
    def test_eval_mode(self):
        """Test model in evaluation mode."""
        self.model.eval()
        self.assertFalse(self.model.training)
    
    def test_deterministic_output_eval(self):
        """Test model produces deterministic output in eval mode."""
        self.model.eval()
        dummy_input = torch.randn(8, 128, 38)
        
        with torch.no_grad():
            output1 = self.model(dummy_input)
            output2 = self.model(dummy_input)
        
        # Outputs should be identical in eval mode
        self.assertTrue(torch.allclose(output1, output2))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
