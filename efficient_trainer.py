"""
Optimized Trainer with Mixed Precision, Gradient Accumulation, and Memory Optimization
For efficient training on large datasets with limited resources.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import time
from tqdm import tqdm
import json
import gc

from models.model_utils import (
    save_checkpoint,
    EarlyStopping,
    get_device,
    set_seed
)
from training.metrics import MetricsCalculator


class MemoryEfficientDataset(Dataset):
    """
    Memory-efficient dataset that loads data in chunks.
    Useful when dataset is too large to fit in memory.
    """

    def __init__(self, data_path: str, labels_path: str,
                 load_in_memory: bool = True):
        """
        Initialize dataset.

        Args:
            data_path: Path to .npy file containing data
            labels_path: Path to .npy file containing labels
            load_in_memory: If True, load all data in memory (faster but more memory)
        """
        self.data_path = data_path
        self.labels_path = labels_path
        self.load_in_memory = load_in_memory

        if load_in_memory:
            self.data = np.load(data_path)
            self.labels = np.load(labels_path)
        else:
            # Memory-mapped array for large datasets
            self.data = np.load(data_path, mmap_mode='r')
            self.labels = np.load(labels_path, mmap_mode='r')

        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Convert to tensor on-the-fly to save memory
        return (
            torch.FloatTensor(self.data[idx]),
            torch.FloatTensor([self.labels[idx]])
        )


class EfficientGaitTrainer:
    """
    Optimized trainer with mixed precision training, gradient accumulation,
    and memory-efficient data loading.
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 use_amp: bool = True,
                 gradient_accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0,
                 early_stopping_patience: int = 15,
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'results/logs',
                 save_memory: bool = False):
        """
        Initialize the efficient trainer.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            scheduler: Learning rate scheduler
            use_amp: Use automatic mixed precision (faster on GPU)
            gradient_accumulation_steps: Accumulate gradients for effective larger batch size
            max_grad_norm: Maximum gradient norm for clipping
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            save_memory: Enable aggressive memory saving mode
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.use_amp = use_amp and device.type == 'cuda'
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_memory = save_memory

        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler() if self.use_amp else None

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
            'learning_rates': [],
            'epoch_times': []
        }

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.start_epoch = 0

        # Calculate effective batch size
        self.effective_batch_size = train_loader.batch_size * gradient_accumulation_steps

        print(f"\n{'='*80}")
        print("EFFICIENT TRAINER INITIALIZED")
        print(f"{'='*80}")
        print(f"Device: {device}")
        print(f"Mixed Precision: {'Enabled' if self.use_amp else 'Disabled'}")
        print(f"Batch Size: {train_loader.batch_size}")
        print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
        print(f"Effective Batch Size: {self.effective_batch_size}")
        print(f"Memory Saving Mode: {'Enabled' if save_memory else 'Disabled'}")
        print(f"{'='*80}\n")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch with optimizations.

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
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)

            # Mixed precision forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Statistics
            running_loss += loss.item() * self.gradient_accumulation_steps
            predictions = (outputs > 0.0).float()  # Use 0.0 threshold for logits

            all_predictions.extend(predictions.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())

            # Clear cache if memory saving mode
            if self.save_memory and batch_idx % 50 == 0:
                torch.cuda.empty_cache() if self.device.type == 'cuda' else None

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'avg_loss': f'{running_loss/(batch_idx+1):.4f}'
            })

        # Calculate epoch statistics
        avg_loss = running_loss / len(self.train_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float, Dict]:
        """
        Validate the model with optimizations.

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

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')

        for batch_idx, (inputs, labels) in enumerate(pbar):
            # Move data to device
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)

            # Mixed precision inference
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            predictions = (outputs > 0.0).float()  # Use 0.0 threshold for logits
            probabilities = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

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

    def train(self, num_epochs: int, save_best_only: bool = True,
              log_interval: int = 1):
        """
        Complete optimized training loop.

        Args:
            num_epochs: Number of epochs to train
            save_best_only: Whether to save only the best model
            log_interval: Log metrics every N epochs
        """
        print(f"\n{'='*80}")
        print("STARTING EFFICIENT TRAINING")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Effective batch size: {self.effective_batch_size}")
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

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Update history
            self.history['train_loss'].append(float(train_loss))
            self.history['train_acc'].append(float(train_acc))
            self.history['val_loss'].append(float(val_loss))
            self.history['val_acc'].append(float(val_acc))
            self.history['learning_rates'].append(float(current_lr))
            self.history['epoch_times'].append(float(epoch_time))

            # Overfitting detection
            loss_gap = val_loss - train_loss
            acc_gap = train_acc - val_acc
            is_overfitting = loss_gap > 0.15 or acc_gap > 0.10

            # Print epoch summary (every log_interval epochs)
            if (epoch + 1) % log_interval == 0:
                print(f"\n{'='*80}")
                print(f"EPOCH {epoch + 1}/{num_epochs} SUMMARY")
                print(f"{'='*80}")
                print(f"Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
                print(f"\nTrain - Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
                print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

                # Overfitting warning
                if is_overfitting:
                    print(f"\n⚠ WARNING: Possible overfitting detected!")
                    print(f"  Loss gap: {loss_gap:.4f} (train-val)")
                    print(f"  Acc gap: {acc_gap:.4f} (train-val)")
                    print(f"  Consider: reducing model complexity, adding regularization, or data augmentation")
                else:
                    print(f"\n✓ Model generalizing well (Loss gap: {loss_gap:.4f}, Acc gap: {acc_gap:.4f})")

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
                        'train_acc': train_acc,
                        'use_amp': self.use_amp,
                        'gradient_accumulation_steps': self.gradient_accumulation_steps
                    }
                )
                print(f"✓ New best model saved! (Val Loss: {val_loss:.4f})\n")

            # Save checkpoint every 5 epochs (if not save_best_only)
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

            # Memory cleanup
            if self.save_memory:
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"\n⚠ Early stopping triggered!")
                print(f"Best Val Loss: {self.best_val_loss:.4f}")
                print(f"Best Val Acc: {self.best_val_acc:.4f}")
                break

        # Training completed
        total_time = time.time() - training_start_time
        avg_epoch_time = np.mean(self.history['epoch_times'])

        print(f"\n{'='*80}")
        print("TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Total Time: {total_time/60:.2f} minutes")
        print(f"Average Epoch Time: {avg_epoch_time:.2f}s")
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


def create_efficient_data_loaders(X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 X_val: np.ndarray,
                                 y_val: np.ndarray,
                                 batch_size: int = 32,
                                 num_workers: int = 4,
                                 pin_memory: bool = True,
                                 prefetch_factor: int = 2) -> Tuple[DataLoader, DataLoader]:
    """
    Create optimized PyTorch data loaders.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        batch_size: Batch size
        num_workers: Number of data loading workers (use 0 on Windows if issues)
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker

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

    # Determine optimal settings
    use_cuda = torch.cuda.is_available()
    pin_memory = pin_memory and use_cuda

    # Create data loaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

    print(f"✓ Optimized data loaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Num workers: {num_workers}")
    print(f"  Pin memory: {pin_memory}")

    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    from models import CNN_BiLSTM_GaitDetector, get_device
    from sklearn.model_selection import train_test_split

    # Load preprocessed data
    windowed_data = np.load('data/processed/efficient_windowed_data.npy')
    windowed_labels = np.load('data/processed/efficient_windowed_labels.npy')

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        windowed_data, windowed_labels, test_size=0.2, random_state=42, stratify=windowed_labels
    )

    # Create efficient data loaders
    train_loader, val_loader = create_efficient_data_loaders(
        X_train, y_train, X_val, y_val,
        batch_size=32,
        num_workers=4,  # Set to 0 on Windows if you encounter issues
        pin_memory=True
    )

    # Initialize model
    device = get_device()
    model = CNN_BiLSTM_GaitDetector(input_features=38, seq_length=128)

    # Setup training (BCEWithLogitsLoss is safe for AMP)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    # Create efficient trainer
    trainer = EfficientGaitTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        use_amp=True,  # Enable mixed precision
        gradient_accumulation_steps=2,  # Effective batch size = 32 * 2 = 64
        early_stopping_patience=300
    )

    # Train
    trainer.train(num_epochs=24, save_best_only=True)
