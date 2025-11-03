# training/validator.py
"""
Validation Module for Gait Detection System
Handles model evaluation, testing, and performance analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from training.metrics import MetricsCalculator


class ModelValidator:
    """
    Validator class for evaluating trained gait detection models.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 threshold: float = 0.0):
        """
        Initialize the validator.

        Args:
            model: Trained PyTorch model
            device: Device to run evaluation on
            threshold: Classification threshold for logits (default: 0.0, equivalent to 0.5 for probabilities)
        """
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        self.metrics_calculator = MetricsCalculator()
    
    def validate(self,
                data_loader: DataLoader,
                criterion: Optional[nn.Module] = None,
                return_predictions: bool = False) -> Dict:
        """
        Validate model on given data loader.
        
        Args:
            data_loader: Data loader for validation
            criterion: Loss function (optional)
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary containing metrics and optionally predictions
        """
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        print(f"\n{'='*60}")
        print("RUNNING VALIDATION")
        print(f"{'='*60}")
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc='Validating')
            
            for inputs, labels in pbar:
                # Move data to device
                inputs = inputs.to(self.device)
                labels_device = labels.to(self.device).float().unsqueeze(1)
                
                # Forward pass (model returns logits)
                outputs = self.model(inputs)

                # Calculate loss if criterion provided
                if criterion is not None:
                    loss = criterion(outputs, labels_device)
                    running_loss += loss.item()

                # Get predictions (use threshold on logits)
                predictions = (outputs > self.threshold).float()

                # Get probabilities (apply sigmoid to logits)
                probabilities = torch.sigmoid(outputs)

                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update progress
                if criterion is not None:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels).flatten()
        all_probabilities = np.array(all_probabilities).flatten()
        
        # Calculate all metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            y_prob=all_probabilities
        )
        
        # Add loss if calculated
        if criterion is not None:
            metrics['loss'] = running_loss / len(data_loader)
        
        # Print results
        self._print_validation_results(metrics)
        
        # Prepare return dictionary
        results = {'metrics': metrics}
        
        if return_predictions:
            results['predictions'] = all_predictions
            results['labels'] = all_labels
            results['probabilities'] = all_probabilities
        
        return results
    
    def test(self,
            X_test: np.ndarray,
            y_test: np.ndarray,
            batch_size: int = 32) -> Dict:
        """
        Test model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size for testing
            
        Returns:
            Dictionary containing test metrics
        """
        from torch.utils.data import TensorDataset, DataLoader
        
        # Create test loader
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        print(f"\n{'='*60}")
        print("TESTING MODEL")
        print(f"{'='*60}")
        print(f"Test samples: {len(X_test)}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")
        
        # Run validation
        results = self.validate(test_loader, return_predictions=True)
        
        return results
    
    def predict(self,
               X: np.ndarray,
               batch_size: int = 32,
               return_probabilities: bool = False) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features (num_samples, seq_length, num_features)
            batch_size: Batch size for inference
            return_probabilities: Return probabilities instead of binary predictions
            
        Returns:
            Predictions array
        """
        self.model.eval()
        
        from torch.utils.data import TensorDataset, DataLoader
        
        # Create data loader
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        predictions = []
        
        with torch.no_grad():
            for (inputs,) in tqdm(data_loader, desc='Predicting'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)  # Returns logits

                if return_probabilities:
                    # Apply sigmoid to get probabilities
                    probs = torch.sigmoid(outputs)
                    predictions.extend(probs.cpu().numpy())
                else:
                    preds = (outputs > self.threshold).float()
                    predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions).flatten()
    
    def evaluate_at_different_thresholds(self,
                                        data_loader: DataLoader,
                                        thresholds: list = None) -> Dict:
        """
        Evaluate model at different classification thresholds.

        Args:
            data_loader: Data loader
            thresholds: List of probability thresholds to evaluate (default: [0.3, 0.4, 0.5, 0.6, 0.7])

        Returns:
            Dictionary mapping thresholds to metrics
        """
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

        print(f"\n{'='*60}")
        print("EVALUATING AT DIFFERENT THRESHOLDS")
        print(f"{'='*60}\n")

        # Get predictions and labels
        self.model.eval()
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc='Getting predictions'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)  # Returns logits

                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(outputs)

                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_labels = np.array(all_labels).flatten()
        all_probabilities = np.array(all_probabilities).flatten()
        
        # Evaluate at each threshold
        results = {}
        
        for threshold in thresholds:
            all_predictions = (all_probabilities > threshold).astype(float)
            
            metrics = self.metrics_calculator.calculate_all_metrics(
                y_true=all_labels,
                y_pred=all_predictions,
                y_prob=all_probabilities
            )
            
            results[threshold] = metrics
            
            print(f"Threshold: {threshold:.2f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}\n")
        
        print(f"{'='*60}\n")
        
        return results
    
    def find_optimal_threshold(self,
                              data_loader: DataLoader,
                              metric: str = 'f1') -> Tuple[float, Dict]:
        """
        Find optimal classification threshold based on specified metric.

        Args:
            data_loader: Data loader
            metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')

        Returns:
            Tuple of (optimal_threshold, metrics_at_optimal)
        """
        print(f"\n{'='*60}")
        print(f"FINDING OPTIMAL THRESHOLD (optimizing {metric.upper()})")
        print(f"{'='*60}\n")

        # Get predictions
        self.model.eval()
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc='Getting predictions'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)  # Returns logits

                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(outputs)

                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_labels = np.array(all_labels).flatten()
        all_probabilities = np.array(all_probabilities).flatten()
        
        # Search thresholds
        thresholds = np.arange(0.1, 0.95, 0.05)
        best_score = 0.0
        best_threshold = 0.5
        best_metrics = None
        
        for threshold in thresholds:
            all_predictions = (all_probabilities > threshold).astype(float)
            
            metrics = self.metrics_calculator.calculate_all_metrics(
                y_true=all_labels,
                y_pred=all_predictions,
                y_prob=all_probabilities
            )
            
            score = metrics[metric]
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = metrics
        
        print(f"✓ Optimal threshold found: {best_threshold:.2f}")
        print(f"  {metric.capitalize()}: {best_score:.4f}")
        print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"  Precision: {best_metrics['precision']:.4f}")
        print(f"  Recall: {best_metrics['recall']:.4f}")
        print(f"  F1-Score: {best_metrics['f1']:.4f}")
        print(f"{'='*60}\n")
        
        return best_threshold, best_metrics
    
    def _print_validation_results(self, metrics: Dict):
        """Print formatted validation results."""
        print(f"\n{'='*60}")
        print("VALIDATION RESULTS")
        print(f"{'='*60}")
        
        if 'loss' in metrics:
            print(f"Loss: {metrics['loss']:.4f}")
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  TN: {cm[0, 0]:<6} FP: {cm[0, 1]:<6}")
        print(f"  FN: {cm[1, 0]:<6} TP: {cm[1, 1]:<6}")
        
        print(f"\nPer-Class Metrics:")
        print(f"  Non-Gait (0):")
        print(f"    Precision: {metrics['precision_per_class'][0]:.4f}")
        print(f"    Recall: {metrics['recall_per_class'][0]:.4f}")
        print(f"    F1-Score: {metrics['f1_per_class'][0]:.4f}")
        print(f"  Gait (1):")
        print(f"    Precision: {metrics['precision_per_class'][1]:.4f}")
        print(f"    Recall: {metrics['recall_per_class'][1]:.4f}")
        print(f"    F1-Score: {metrics['f1_per_class'][1]:.4f}")
        
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    from models import CNN_BiLSTM_GaitDetector, get_device, load_checkpoint
    
    # Create dummy test data
    X_test = np.random.randn(200, 128, 38)
    y_test = np.random.randint(0, 2, 200)
    
    # Initialize model and device
    device = get_device()
    model = CNN_BiLSTM_GaitDetector(input_features=38, seq_length=128)
    
    # Load trained weights (if available)
    # load_checkpoint('checkpoints/best_model.pt', model, device=device)
    
    # Create validator
    validator = ModelValidator(model=model, device=device, threshold=0.0)
    
    # Test model
    results = validator.test(X_test, y_test, batch_size=32)
    
    # Find optimal threshold
    from torch.utils.data import TensorDataset, DataLoader
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    optimal_threshold, metrics = validator.find_optimal_threshold(test_loader, metric='f1')
