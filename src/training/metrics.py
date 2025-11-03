# training/metrics.py
"""
Metrics Module for Gait Detection System
Comprehensive metrics calculation for binary classification.
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef
)


class MetricsCalculator:
    """
    Calculator for binary classification metrics.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.class_names = ['Non-Gait', 'Gait']
    
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy."""
        return accuracy_score(y_true, y_pred)
    
    def calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision (positive predictive value)."""
        return precision_score(y_true, y_pred, zero_division=0)
    
    def calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall (sensitivity, true positive rate)."""
        return recall_score(y_true, y_pred, zero_division=0)
    
    def calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1-score (harmonic mean of precision and recall)."""
        return f1_score(y_true, y_pred, zero_division=0)
    
    def calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        # Ensure confusion matrix is 2x2
        if cm.shape != (2, 2):
            # If only one class present, return 0.0
            return 0.0
        
        tn = cm[0, 0]
        fp = cm[0, 1]
        
        if (tn + fp) == 0:
            return 0.0
        
        return tn / (tn + fp)
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix (always returns 2x2 for binary classification)."""
        return confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    def calculate_auc_roc(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Area Under ROC Curve."""
        try:
            # Check if we have both classes
            if len(np.unique(y_true)) < 2:
                return 0.0
            return roc_auc_score(y_true, y_prob)
        except Exception as e:
            return 0.0
    
    def calculate_auc_pr(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Area Under Precision-Recall Curve."""
        try:
            # Check if we have both classes
            if len(np.unique(y_true)) < 2:
                return 0.0
            return average_precision_score(y_true, y_prob)
        except Exception as e:
            return 0.0
    
    def calculate_cohen_kappa(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Cohen's Kappa score."""
        return cohen_kappa_score(y_true, y_pred)
    
    def calculate_mcc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Matthews Correlation Coefficient."""
        return matthews_corrcoef(y_true, y_pred)
    
    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate precision, recall, and F1 for each class.
        
        Returns:
            Dictionary with per-class metrics
        """
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
        
        # Ensure arrays have length 2
        if len(precision_per_class) < 2:
            precision_per_class = np.array([0.0, 0.0])
        if len(recall_per_class) < 2:
            recall_per_class = np.array([0.0, 0.0])
        if len(f1_per_class) < 2:
            f1_per_class = np.array([0.0, 0.0])
        
        return {
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }
    
    def calculate_all_metrics(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_prob: np.ndarray = None) -> Dict:
        """
        Calculate all available metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (binary)
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary with all metrics
        """
        # Ensure we have both classes for proper metric calculation
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        
        metrics = {
            'accuracy': self.calculate_accuracy(y_true, y_pred),
            'precision': self.calculate_precision(y_true, y_pred),
            'recall': self.calculate_recall(y_true, y_pred),
            'f1': self.calculate_f1(y_true, y_pred),
            'specificity': self.calculate_specificity(y_true, y_pred),
            'confusion_matrix': self.calculate_confusion_matrix(y_true, y_pred),
            'cohen_kappa': self.calculate_cohen_kappa(y_true, y_pred),
            'mcc': self.calculate_mcc(y_true, y_pred)
        }
        
        # Add per-class metrics
        per_class = self.calculate_per_class_metrics(y_true, y_pred)
        metrics.update(per_class)
        
        # Add probability-based metrics if available
        if y_prob is not None and len(unique_true) > 1:
            try:
                metrics['auc_roc'] = self.calculate_auc_roc(y_true, y_prob)
                metrics['auc_pr'] = self.calculate_auc_pr(y_true, y_prob)
            except:
                # If AUC calculation fails (e.g., only one class present)
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
        else:
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
        
        return metrics
    
    def get_roc_curve_data(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """
        Get ROC curve data for plotting.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary with fpr, tpr, thresholds, and auc
        """
        try:
            # Check if we have both classes
            if len(np.unique(y_true)) < 2:
                return {
                    'fpr': np.array([0, 1]),
                    'tpr': np.array([0, 1]),
                    'thresholds': np.array([0, 1]),
                    'auc': 0.0
                }
            
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            
            return {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': auc
            }
        except Exception as e:
            return {
                'fpr': np.array([0, 1]),
                'tpr': np.array([0, 1]),
                'thresholds': np.array([0, 1]),
                'auc': 0.0
            }
    
    def get_pr_curve_data(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """
        Get Precision-Recall curve data for plotting.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary with precision, recall, thresholds, and auc_pr
        """
        try:
            # Check if we have both classes
            if len(np.unique(y_true)) < 2:
                return {
                    'precision': np.array([1, 0]),
                    'recall': np.array([0, 1]),
                    'thresholds': np.array([0]),
                    'auc_pr': 0.0
                }
            
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            auc_pr = average_precision_score(y_true, y_prob)
            
            return {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds,
                'auc_pr': auc_pr
            }
        except Exception as e:
            return {
                'precision': np.array([1, 0]),
                'recall': np.array([0, 1]),
                'thresholds': np.array([0]),
                'auc_pr': 0.0
            }
    
    def print_metrics_report(self, metrics: Dict):
        """
        Print formatted metrics report.
        
        Args:
            metrics: Dictionary of metrics
        """
        print(f"\n{'='*70}")
        print("METRICS REPORT")
        print(f"{'='*70}")
        
        # Main metrics
        print(f"\n--- Primary Metrics ---")
        print(f"Accuracy:   {metrics['accuracy']:.4f}")
        print(f"Precision:  {metrics['precision']:.4f}")
        print(f"Recall:     {metrics['recall']:.4f}")
        print(f"F1-Score:   {metrics['f1']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        
        # AUC metrics
        if 'auc_roc' in metrics:
            print(f"\n--- AUC Metrics ---")
            print(f"AUC-ROC:    {metrics['auc_roc']:.4f}")
            print(f"AUC-PR:     {metrics['auc_pr']:.4f}")
        
        # Confusion Matrix
        print(f"\n--- Confusion Matrix ---")
        cm = metrics['confusion_matrix']
        print(f"              Predicted")
        print(f"            Non-Gait  Gait")
        print(f"Actual Non-Gait  {cm[0,0]:<6} {cm[0,1]:<6}")
        print(f"       Gait      {cm[1,0]:<6} {cm[1,1]:<6}")
        
        # Derived metrics from confusion matrix
        cm = metrics['confusion_matrix']
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print(f"\n--- Confusion Matrix Values ---")
            print(f"True Negatives:  {tn}")
            print(f"False Positives: {fp}")
            print(f"False Negatives: {fn}")
            print(f"True Positives:  {tp}")
        else:
            print(f"\n--- Confusion Matrix Values ---")
            print(f"Warning: Confusion matrix shape is {cm.shape} (expected 2x2)")
            print(f"This may occur if only one class is present in predictions.")
        
        # Per-class metrics
        print(f"\n--- Per-Class Metrics ---")
        if len(metrics['precision_per_class']) >= 2:
            for i, class_name in enumerate(self.class_names):
                print(f"\n{class_name}:")
                print(f"  Precision: {metrics['precision_per_class'][i]:.4f}")
                print(f"  Recall:    {metrics['recall_per_class'][i]:.4f}")
                print(f"  F1-Score:  {metrics['f1_per_class'][i]:.4f}")
        else:
            print("  Warning: Per-class metrics not available (only one class present)")
        
        # Additional metrics
        print(f"\n--- Additional Metrics ---")
        print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        print(f"MCC:           {metrics['mcc']:.4f}")
        
        print(f"{'='*70}\n")
    
    def compare_models(self, metrics_dict: Dict[str, Dict]) -> None:
        """
        Compare metrics across multiple models.
        
        Args:
            metrics_dict: Dictionary mapping model names to their metrics
        """
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}\n")
        
        # Get metric names
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        
        # Print header
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC-ROC':<12}")
        print("-" * 80)
        
        # Print each model's metrics
        for model_name, metrics in metrics_dict.items():
            print(f"{model_name:<20} ", end="")
            for metric in metric_names:
                value = metrics.get(metric, 0.0)
                print(f"{value:<12.4f} ", end="")
            print()
        
        print(f"{'='*80}\n")


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Label array
        
    Returns:
        Dictionary mapping class index to weight
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    weights = {}
    for cls, count in zip(unique_classes, counts):
        weight = total / (len(unique_classes) * count)
        weights[int(cls)] = weight
    
    print(f"\nClass Weights (for handling imbalance):")
    for cls, weight in weights.items():
        class_name = "Non-Gait" if cls == 0 else "Gait"
        print(f"  {class_name} ({cls}): {weight:.4f}")
    
    return weights


if __name__ == "__main__":
    # Example usage
    print("Metrics Calculator Example\n")
    
    # Create dummy predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.randint(0, 2, 1000)
    y_prob = np.random.rand(1000)
    
    # Calculate metrics
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(y_true, y_pred, y_prob)
    
    # Print report
    calculator.print_metrics_report(metrics)
    
    # Get ROC curve data
    roc_data = calculator.get_roc_curve_data(y_true, y_prob)
    print(f"ROC Curve - AUC: {roc_data['auc']:.4f}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_true)