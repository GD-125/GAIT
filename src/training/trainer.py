# training/trainer.py
"""
Training Module for Gait Detection System
Handles the complete training loop with logging, checkpointing, and visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import time
from tqdm import tqdm
import json

from models.model_utils import (
    save_checkpoint, 
    EarlyStopping,
    get_device,
    set_seed
)
from training.metrics import MetricsCalculator


class GaitDetectorTrainer:
    """
    Trainer class for CNN-BiLSTM gait detection model.
    Handles training loop, validation, logging, and checkpointing.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 early_stopping_patience: int = 15,
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'results/logs'):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            scheduler: Learning rate scheduler (optional)
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='min'
        )
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calculator = MetricsCalculator()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.start_epoch = 0
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            # Move data to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{running_loss/(batch_idx+1):.4f}'
            })
        
        # Calculate epoch statistics
        avg_loss = running_loss / len(self.train_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        return avg_loss, accuracy
    
    def validate(self, epoch: int) -> Tuple[float, float, Dict]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, accuracy, metrics_dict)
        """
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            
            for inputs, labels in pbar:
                # Move data to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                predictions = (outputs > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(outputs.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        avg_loss = running_loss / len(self.val_loader)
        
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels).flatten()
        all_probabilities = np.array(all_probabilities).flatten()
        
        metrics = self.metrics_calculator.calculate_all_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            y_prob=all_probabilities
        )
        
        return avg_loss, metrics['accuracy'], metrics
    
    def train(self, num_epochs: int, save_best_only: bool = True):
        """
        Complete training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_best_only: Whether to save only the best model
        """
        print(f"\n{'='*80}")
        print("STARTING TRAINING")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Epochs: {num_epochs}")
        print(f"{'='*80}\n")
        
        training_start_time = time.time()
        
        for epoch in range(self.start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch + 1)
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate(epoch + 1)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch + 1}/{num_epochs} SUMMARY")
            print(f"{'='*80}")
            print(f"Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
            print(f"\nTrain - Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
            print(f"\nValidation Metrics:")
            print(f"  Precision: {val_metrics['precision']:.4f}")
            print(f"  Recall: {val_metrics['recall']:.4f}")
            print(f"  F1-Score: {val_metrics['f1']:.4f}")
            print(f"  AUC-ROC: {val_metrics['auc_roc']:.4f}")
            print(f"{'='*80}\n")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                
                checkpoint_path = self.checkpoint_dir / 'best_model.pt'
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    loss=val_loss,
                    accuracy=val_acc,
                    filepath=checkpoint_path,
                    scheduler=self.scheduler,
                    additional_info={
                        'metrics': val_metrics,
                        'train_loss': train_loss,
                        'train_acc': train_acc
                    }
                )
                print(f"✓ New best model saved! (Val Loss: {val_loss:.4f})\n")
            
            # Save checkpoint every N epochs (if not save_best_only)
            if not save_best_only and (epoch + 1) % 5 == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    loss=val_loss,
                    accuracy=val_acc,
                    filepath=checkpoint_path,
                    scheduler=self.scheduler
                )
            
            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"\n⚠ Early stopping triggered!")
                print(f"Best Val Loss: {self.best_val_loss:.4f}")
                print(f"Best Val Acc: {self.best_val_acc:.4f}")
                break
        
        # Training completed
        total_time = time.time() - training_start_time
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Total Time: {total_time/60:.2f} minutes")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Val Acc: {self.best_val_acc:.4f}")
        print(f"{'='*80}\n")
        
        # Save training history
        self.save_history()
    
    def save_history(self):
        """Save training history to JSON file."""
        history_path = self.log_dir / 'training_history.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        print(f"✓ Training history saved to {history_path}")
    
    def load_history(self):
        """Load training history from JSON file."""
        history_path = self.log_dir / 'training_history.json'
        
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.history = json.load(f)
            print(f"✓ Training history loaded from {history_path}")
        else:
            print(f"⚠ No training history found at {history_path}")


def create_data_loaders(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       batch_size: int = 32,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch data loaders from numpy arrays.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✓ Data loaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    from models import CNN_BiLSTM_GaitDetector, get_device
    from sklearn.model_selection import train_test_split
    
    # Load preprocessed data (example)
    # windowed_data = np.load('data/processed/windowed_data.npy')
    # windowed_labels = np.load('data/processed/windowed_labels.npy')
    
    # Create dummy data for testing
    windowed_data = np.random.randn(1000, 128, 38)
    windowed_labels = np.random.randint(0, 2, 1000)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        windowed_data, windowed_labels, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=32
    )
    
    # Initialize model
    device = get_device()
    model = CNN_BiLSTM_GaitDetector(input_features=38, seq_length=128)
    
    # Setup training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Create trainer
    trainer = GaitDetectorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        early_stopping_patience=10
    )
    
    # Train
    trainer.train(num_epochs=50)
