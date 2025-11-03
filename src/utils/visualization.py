# utils/visualization.py
"""
Visualization Module for Gait Detection System
Plotting utilities for training curves, metrics, confusion matrices, and data exploration.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class Visualizer:
    """
    Comprehensive visualization utilities for gait detection system.
    """
    
    def __init__(self, save_dir: str = 'results/plots'):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_history(self,
                             history: Dict,
                             save_path: Optional[str] = None):
        """
        Plot training and validation loss/accuracy curves.
        
        Args:
            history: Training history dictionary with keys:
                    'train_loss', 'train_acc', 'val_loss', 'val_acc'
            save_path: Path to save plot (optional)
        """
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(loc='lower right', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self,
                            cm: np.ndarray,
                            class_names: List[str] = ['Non-Gait', 'Gait'],
                            save_path: Optional[str] = None):
        """
        Plot confusion matrix with annotations.
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray')
        
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        # Add percentage annotations
        total = np.sum(cm)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=9, color='darkred')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self,
                      fpr: np.ndarray,
                      tpr: np.ndarray,
                      auc: float,
                      save_path: Optional[str] = None):
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rate
            tpr: True positive rate
            auc: Area under curve
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=(8, 8))
        
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve plot saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self,
                                    precision: np.ndarray,
                                    recall: np.ndarray,
                                    auc_pr: float,
                                    save_path: Optional[str] = None):
        """
        Plot Precision-Recall curve.
        
        Args:
            precision: Precision values
            recall: Recall values
            auc_pr: Area under PR curve
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=(8, 8))
        
        plt.plot(recall, precision, 'b-', linewidth=2, 
                label=f'PR Curve (AUC = {auc_pr:.4f})')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ PR curve plot saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self,
                               metrics_dict: Dict[str, Dict],
                               save_path: Optional[str] = None):
        """
        Plot bar chart comparing metrics across different models/experiments.
        
        Args:
            metrics_dict: Dictionary mapping model names to metrics
            save_path: Path to save plot (optional)
        """
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        model_names = list(metrics_dict.keys())
        
        x = np.arange(len(metric_names))
        width = 0.8 / len(model_names)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, model_name in enumerate(model_names):
            metrics = metrics_dict[model_name]
            values = [metrics.get(m, 0) for m in metric_names]
            offset = width * (i - len(model_names)/2 + 0.5)
            ax.bar(x + offset, values, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Metrics comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_sensor_data(self,
                        data: np.ndarray,
                        labels: np.ndarray,
                        sensor_names: List[str],
                        num_samples: int = 3,
                        save_path: Optional[str] = None):
        """
        Plot raw sensor data for visual inspection.
        
        Args:
            data: Windowed data (num_windows, seq_length, num_features)
            labels: Labels for each window
            sensor_names: Names of sensor channels
            num_samples: Number of samples to plot
            save_path: Path to save plot (optional)
        """
        num_sensors = min(6, len(sensor_names))  # Plot first 6 sensors
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            sample_data = data[i]  # (seq_length, num_features)
            label = 'Gait' if labels[i] == 1 else 'Non-Gait'
            
            for j in range(num_sensors):
                axes[i].plot(sample_data[:, j], label=sensor_names[j], linewidth=1.5, alpha=0.7)
            
            axes[i].set_xlabel('Time Steps', fontsize=10)
            axes[i].set_ylabel('Sensor Value', fontsize=10)
            axes[i].set_title(f'Sample {i+1} - Label: {label}', fontsize=12, fontweight='bold')
            axes[i].legend(loc='upper right', fontsize=8, ncol=2)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Sensor data plot saved to {save_path}")
        
        plt.show()
    
    def plot_class_distribution(self,
                               labels: np.ndarray,
                               class_names: List[str] = ['Non-Gait', 'Gait'],
                               save_path: Optional[str] = None):
        """
        Plot class distribution as pie chart and bar chart.
        
        Args:
            labels: Label array
            class_names: Names of classes
            save_path: Path to save plot (optional)
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pie chart
        colors = ['#ff9999', '#66b3ff']
        axes[0].pie(counts, labels=class_names, autopct='%1.1f%%', 
                   colors=colors, startangle=90, textprops={'fontsize': 12})
        axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        axes[1].bar(class_names, counts, color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Class Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            axes[1].text(i, count + max(counts)*0.02, str(count), 
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Class distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_learning_rate_schedule(self,
                                    history: Dict,
                                    save_path: Optional[str] = None):
        """
        Plot learning rate schedule over epochs.
        
        Args:
            history: Training history with 'learning_rates' key
            save_path: Path to save plot (optional)
        """
        if 'learning_rates' not in history:
            print("⚠ No learning rate data found in history")
            return
        
        epochs = range(1, len(history['learning_rates']) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['learning_rates'], 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Learning rate schedule plot saved to {save_path}")
        
        plt.show()
    
    def create_comprehensive_report(self,
                                   history: Dict,
                                   metrics: Dict,
                                   roc_data: Dict,
                                   pr_data: Dict):
        """
        Create comprehensive visualization report with all plots.
        
        Args:
            history: Training history
            metrics: Evaluation metrics
            roc_data: ROC curve data (fpr, tpr, auc)
            pr_data: PR curve data (precision, recall, auc_pr)
        """
        print(f"\n{'='*60}")
        print("GENERATING COMPREHENSIVE VISUALIZATION REPORT")
        print(f"{'='*60}\n")
        
        # 1. Training history
        print("[1/5] Plotting training history...")
        self.plot_training_history(history, 
                                   save_path=self.save_dir / 'training_history.png')
        
        # 2. Confusion matrix
        print("[2/5] Plotting confusion matrix...")
        self.plot_confusion_matrix(metrics['confusion_matrix'],
                                   save_path=self.save_dir / 'confusion_matrix.png')
        
        # 3. ROC curve
        print("[3/5] Plotting ROC curve...")
        self.plot_roc_curve(roc_data['fpr'], roc_data['tpr'], roc_data['auc'],
                           save_path=self.save_dir / 'roc_curve.png')
        
        # 4. PR curve
        print("[4/5] Plotting Precision-Recall curve...")
        self.plot_precision_recall_curve(pr_data['precision'], pr_data['recall'], 
                                        pr_data['auc_pr'],
                                        save_path=self.save_dir / 'pr_curve.png')
        
        # 5. Learning rate schedule
        print("[5/5] Plotting learning rate schedule...")
        self.plot_learning_rate_schedule(history,
                                        save_path=self.save_dir / 'lr_schedule.png')
        
        print(f"\n{'='*60}")
        print(f"✓ Comprehensive report generated!")
        print(f"Output directory: {self.save_dir}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    print("Visualization Module Example\n")
    
    visualizer = Visualizer(save_dir='results/plots')
    
    # Create dummy data
    history = {
        'train_loss': np.random.uniform(0.6, 0.3, 30),
        'train_acc': np.random.uniform(0.7, 0.95, 30),
        'val_loss': np.random.uniform(0.65, 0.35, 30),
        'val_acc': np.random.uniform(0.65, 0.90, 30),
        'learning_rates': np.linspace(0.001, 0.0001, 30)
    }
    
    # Make losses decreasing
    history['train_loss'] = np.sort(history['train_loss'])[::-1] + np.random.normal(0, 0.02, 30)
    history['val_loss'] = np.sort(history['val_loss'])[::-1] + np.random.normal(0, 0.02, 30)
    
    # Plot training history
    visualizer.plot_training_history(history)
    
    # Plot confusion matrix
    cm = np.array([[450, 50], [30, 470]])
    visualizer.plot_confusion_matrix(cm)
    
    # Plot class distribution
    labels = np.random.randint(0, 2, 1000)
    visualizer.plot_class_distribution(labels)
