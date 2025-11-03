# explainability/lime_explainer.py
"""
LIME Explainability Module for Gait Detection System
Provides model interpretability using LIME (Local Interpretable Model-agnostic Explanations).
More stable alternative to SHAP for complex time-series models.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class GaitLIMEExplainer:
    """
    LIME-based explainability for gait detection models.
    Explains predictions by perturbing features and observing output changes.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained PyTorch model
            device: Device model is on
            feature_names: List of feature names for visualization
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Generate default feature names if not provided
        if feature_names is None:
            self.feature_names = self._generate_default_feature_names()
        else:
            self.feature_names = feature_names
        
        self.explanations = []
        self.feature_importance_global = None
    
    def _generate_default_feature_names(self, num_features: int = 38) -> List[str]:
        """Generate default feature names based on sensor configuration."""
        feature_names = []
        
        # Accelerometer features (18 features)
        for location in ['right_foot', 'right_shin', 'right_thigh', 
                        'left_foot', 'left_shin', 'left_thigh']:
            for axis in ['x', 'y', 'z']:
                feature_names.append(f'accel_{location}_{axis}')
        
        # Gyroscope features (18 features)
        for location in ['right_foot', 'right_shin', 'right_thigh',
                        'left_foot', 'left_shin', 'left_thigh']:
            for axis in ['x', 'y', 'z']:
                feature_names.append(f'gyro_{location}_{axis}')
        
        # EMG features (2 features)
        feature_names.extend(['EMG_right', 'EMG_left'])
        
        return feature_names
    
    def _predict_fn(self, samples: np.ndarray) -> np.ndarray:
        """
        Prediction function for LIME.
        
        Args:
            samples: Perturbed samples (num_samples, seq_length, features)
            
        Returns:
            Predictions (num_samples, 2) for [non-gait, gait] probabilities
        """
        self.model.eval()
        
        with torch.no_grad():
            # Convert to tensor
            samples_tensor = torch.FloatTensor(samples).to(self.device)
            
            # Get predictions
            outputs = self.model(samples_tensor).cpu().numpy()
            
            # Convert to binary classification probabilities
            if len(outputs.shape) == 1 or outputs.shape[1] == 1:
                # Single output (sigmoid)
                outputs = outputs.flatten()
                probs = np.column_stack([1 - outputs, outputs])
            else:
                # Multi-class output (already probabilities)
                probs = outputs
            
            return probs
    
    def _perturb_sample(self,
                       sample: np.ndarray,
                       num_samples: int = 1000,
                       std_scale: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create perturbed versions of a sample for LIME.
        
        Args:
            sample: Original sample (seq_length, features)
            num_samples: Number of perturbed samples to generate
            std_scale: Standard deviation scale for perturbations
            
        Returns:
            Perturbed samples and binary mask showing which features were perturbed
        """
        seq_length, num_features = sample.shape
        
        # Create perturbation masks (1 = keep original, 0 = perturb)
        masks = np.random.binomial(1, 0.5, size=(num_samples, num_features))
        
        # Calculate feature-wise standard deviations
        feature_stds = np.std(sample, axis=0) * std_scale
        feature_stds = np.where(feature_stds == 0, 0.01, feature_stds)  # Avoid zero std
        
        # Generate perturbed samples
        perturbed_samples = np.zeros((num_samples, seq_length, num_features))
        
        for i in range(num_samples):
            perturbed = sample.copy()
            
            for feat_idx in range(num_features):
                if masks[i, feat_idx] == 0:  # Perturb this feature
                    # Add Gaussian noise across time dimension
                    noise = np.random.normal(0, feature_stds[feat_idx], seq_length)
                    perturbed[:, feat_idx] += noise
            
            perturbed_samples[i] = perturbed
        
        return perturbed_samples, masks
    
    def explain_sample(self,
                      sample: np.ndarray,
                      sample_idx: int = 0,
                      num_samples: int = 1000) -> Dict:
        """
        Explain a single prediction using LIME.

        Args:
            sample: Sample to explain (seq_length, features)
            sample_idx: Index for tracking
            num_samples: Number of perturbed samples

        Returns:
            Dictionary containing explanation details
        """
        print(f"\nExplaining sample {sample_idx}...")

        # Get original prediction
        with torch.no_grad():
            original_tensor = torch.FloatTensor(sample).unsqueeze(0).to(self.device)
            original_pred = self.model(original_tensor).cpu().numpy()[0]

            if len(original_pred.shape) == 0:
                original_pred = float(original_pred)
            else:
                original_pred = float(original_pred[0])

        print(f"  Original prediction: {original_pred:.4f} ({'Gait' if original_pred > 0.5 else 'Non-Gait'})")

        # Generate perturbed samples
        perturbed_samples, masks = self._perturb_sample(sample, num_samples)

        # Get predictions for perturbed samples
        perturbed_preds = self._predict_fn(perturbed_samples)[:, 1]  # Get "gait" probability

        # Fit linear model to explain local decision boundary
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        # Weight samples by distance to original (using mask)
        distances = np.sum(1 - masks, axis=1)
        weights = np.exp(-distances / (masks.shape[1] * 0.25))

        # Fit ridge regression
        scaler = StandardScaler()
        masks_scaled = scaler.fit_transform(masks)

        ridge = Ridge(alpha=1.0)
        ridge.fit(masks_scaled, perturbed_preds, sample_weight=weights)

        # Get feature importance (coefficients)
        feature_importance = ridge.coef_

        # Get intercept (base value)
        base_value = ridge.intercept_

        # Create explanation dictionary
        explanation = {
            'sample_idx': sample_idx,
            'prediction': original_pred,
            'predicted_class': 'Gait' if original_pred > 0.5 else 'Non-Gait',
            'base_value': base_value,
            'feature_importance': feature_importance,
            'feature_names': self.feature_names[:len(feature_importance)],
            'r2_score': ridge.score(masks_scaled, perturbed_preds, sample_weight=weights)
        }

        self.explanations.append(explanation)

        print(f"  Local model R² score: {explanation['r2_score']:.4f}")
        print(f"  Top 3 influential features:")

        # Show top features
        abs_importance = np.abs(feature_importance)
        top_indices = np.argsort(abs_importance)[-3:][::-1]

        for idx in top_indices:
            feat_name = explanation['feature_names'][idx]
            importance = feature_importance[idx]
            print(f"    - {feat_name}: {importance:+.4f}")

        return explanation

    def explain_single_sample(self,
                             sample_data: np.ndarray,
                             num_features: int = 15,
                             num_samples: int = 500):
        """
        Simplified explanation method for single sample (for dashboard).

        Args:
            sample_data: Sample to explain (seq_length, features)
            num_features: Number of features to return
            num_samples: Number of perturbations

        Returns:
            Explanation object with as_map() method
        """
        # Create explanation using existing method
        explanation = self.explain_sample(sample_data, sample_idx=0, num_samples=num_samples)

        # Create mock LIME explanation object for compatibility
        class MockLIMEExplanation:
            def __init__(self, feature_importance, feature_names):
                self.feature_importance = feature_importance
                self.feature_names = feature_names

            def as_map(self):
                # Return map for class 1 (Gait)
                # Format: {class_id: [(feature_idx, weight), ...]}
                importance_tuples = [(i, float(w)) for i, w in enumerate(self.feature_importance)]
                # Sort by absolute importance
                importance_tuples.sort(key=lambda x: abs(x[1]), reverse=True)
                return {1: importance_tuples}

        return MockLIMEExplanation(explanation['feature_importance'], explanation['feature_names'])
    
    def explain_samples(self,
                       test_data: np.ndarray,
                       num_samples_to_explain: int = 10,
                       num_perturbations: int = 1000) -> List[Dict]:
        """
        Explain multiple samples.
        
        Args:
            test_data: Test data (num_samples, seq_length, features)
            num_samples_to_explain: Number of samples to explain
            num_perturbations: Number of perturbations per sample
            
        Returns:
            List of explanation dictionaries
        """
        print(f"\n{'='*60}")
        print(f"GENERATING LIME EXPLANATIONS")
        print(f"{'='*60}")
        
        num_to_explain = min(num_samples_to_explain, len(test_data))
        
        self.explanations = []
        
        for i in range(num_to_explain):
            explanation = self.explain_sample(
                sample=test_data[i],
                sample_idx=i,
                num_samples=num_perturbations
            )
        
        # Calculate global feature importance
        self._calculate_global_importance()
        
        print(f"\n{'='*60}")
        print(f"✓ Explained {num_to_explain} samples!")
        print(f"{'='*60}\n")
        
        return self.explanations
    
    def _calculate_global_importance(self):
        """Calculate global feature importance from all explanations."""
        if not self.explanations:
            return
        
        # Average absolute importance across all explanations
        all_importance = np.array([exp['feature_importance'] for exp in self.explanations])
        self.feature_importance_global = np.mean(np.abs(all_importance), axis=0)
    
    def get_feature_importance(self) -> Dict:
        """
        Get global feature importance rankings.
        
        Returns:
            Dictionary of feature names and importance scores
        """
        if self.feature_importance_global is None:
            return {}
        
        importance_dict = {}
        for i, importance in enumerate(self.feature_importance_global):
            if i < len(self.feature_names):
                importance_dict[self.feature_names[i]] = float(importance)
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True))
        
        return importance_dict
    
    def plot_feature_importance(self,
                               top_k: int = 20,
                               save_path: Optional[str] = None):
        """
        Plot global feature importance.
        
        Args:
            top_k: Number of top features to plot
            save_path: Path to save plot
        """
        try:
            importance_dict = self.get_feature_importance()
            
            if not importance_dict:
                print("⚠ No feature importance data available")
                return
            
            # Get top-k features
            top_features = list(importance_dict.items())[:top_k]
            features, importance = zip(*top_features)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(features)), importance, color='coral')
            plt.yticks(range(len(features)), features, fontsize=8)
            plt.xlabel('Mean Absolute LIME Coefficient', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.title(f'Top {len(features)} Most Important Features (LIME)', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Feature importance plot saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"⚠ Error plotting feature importance: {str(e)}")
            plt.close()
    
    def plot_single_explanation(self,
                               explanation_idx: int = 0,
                               top_k: int = 10,
                               save_path: Optional[str] = None):
        """
        Plot explanation for a single sample.
        
        Args:
            explanation_idx: Index in explanations list
            top_k: Number of features to show
            save_path: Path to save plot
        """
        if explanation_idx >= len(self.explanations):
            print(f"⚠ Explanation index {explanation_idx} out of range")
            return
        
        try:
            exp = self.explanations[explanation_idx]
            
            # Get top features by absolute importance
            importance = exp['feature_importance']
            abs_importance = np.abs(importance)
            top_indices = np.argsort(abs_importance)[-top_k:]
            
            # Sort by actual importance for waterfall effect
            sorted_indices = top_indices[np.argsort(importance[top_indices])]
            
            values = importance[sorted_indices]
            features = [exp['feature_names'][i] for i in sorted_indices]
            
            # Create waterfall plot
            plt.figure(figsize=(12, 6))
            
            colors = ['red' if v < 0 else 'green' for v in values]
            y_pos = np.arange(len(features))
            
            plt.barh(y_pos, values, color=colors, alpha=0.7)
            plt.yticks(y_pos, features)
            plt.xlabel('LIME Coefficient (Impact on Gait Probability)', fontsize=11)
            plt.title(f'LIME Explanation for Sample {exp["sample_idx"]}\n' + 
                     f'Prediction: {exp["prediction"]:.3f} ({exp["predicted_class"]}) | ' +
                     f'R² Score: {exp["r2_score"]:.3f}',
                     fontsize=12, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Explanation plot saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"⚠ Error plotting explanation: {str(e)}")
            import traceback
            traceback.print_exc()
            plt.close()
    
    def plot_comparison_chart(self,
                             num_samples: int = 5,
                             save_path: Optional[str] = None):
        """
        Plot comparison of predictions and top features across multiple samples.
        
        Args:
            num_samples: Number of samples to compare
            save_path: Path to save plot
        """
        if len(self.explanations) == 0:
            print("⚠ No explanations available")
            return
        
        try:
            num_samples = min(num_samples, len(self.explanations))
            
            fig, axes = plt.subplots(num_samples, 1, figsize=(14, 4 * num_samples))
            
            if num_samples == 1:
                axes = [axes]
            
            for i, ax in enumerate(axes):
                if i >= len(self.explanations):
                    break
                
                exp = self.explanations[i]
                importance = exp['feature_importance']
                
                # Get top 8 features
                abs_importance = np.abs(importance)
                top_indices = np.argsort(abs_importance)[-8:]
                sorted_indices = top_indices[np.argsort(importance[top_indices])]
                
                values = importance[sorted_indices]
                features = [exp['feature_names'][idx] for idx in sorted_indices]
                
                colors = ['red' if v < 0 else 'green' for v in values]
                y_pos = np.arange(len(features))
                
                ax.barh(y_pos, values, color=colors, alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features, fontsize=9)
                ax.set_xlabel('Impact on Prediction', fontsize=10)
                ax.set_title(f'Sample {exp["sample_idx"]} - Pred: {exp["prediction"]:.3f} ({exp["predicted_class"]})',
                           fontsize=11, fontweight='bold')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                ax.grid(axis='x', alpha=0.3)
            
            plt.suptitle('LIME Explanations Comparison', fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Comparison chart saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"⚠ Error creating comparison chart: {str(e)}")
            plt.close()
    
    def generate_explanation_report(self,
                                   test_data: np.ndarray,
                                   num_samples_to_explain: int = 10,
                                   output_dir: str = 'results/lime_outputs'):
        """
        Generate comprehensive LIME explanation report.
        
        Args:
            test_data: Test data for explanation
            num_samples_to_explain: Number of samples to explain
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("GENERATING LIME EXPLANATION REPORT")
        print(f"{'='*60}\n")
        
        # 1. Generate explanations
        print("[1/5] Generating explanations...")
        self.explain_samples(test_data, num_samples_to_explain)
        
        # 2. Global feature importance
        print("[2/5] Creating global feature importance plot...")
        self.plot_feature_importance(
            top_k=20,
            save_path=output_path / 'feature_importance.png'
        )
        
        # 3. Individual explanations
        print("[3/5] Creating individual explanation plots...")
        num_individual = min(3, len(self.explanations))
        for i in range(num_individual):
            self.plot_single_explanation(
                explanation_idx=i,
                top_k=10,
                save_path=output_path / f'explanation_sample_{i}.png'
            )
        
        # 4. Comparison chart
        print("[4/5] Creating comparison chart...")
        self.plot_comparison_chart(
            num_samples=min(5, len(self.explanations)),
            save_path=output_path / 'explanations_comparison.png'
        )
        
        # 5. Save text report
        print("[5/5] Saving text report...")
        importance_dict = self.get_feature_importance()
        
        with open(output_path / 'lime_report.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("LIME EXPLAINABILITY REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("GLOBAL FEATURE IMPORTANCE\n")
            f.write("-"*60 + "\n")
            for rank, (feature, importance) in enumerate(importance_dict.items(), 1):
                f.write(f"{rank:3d}. {feature:40s} {importance:.6f}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("INDIVIDUAL SAMPLE EXPLANATIONS\n")
            f.write("="*60 + "\n\n")
            
            for exp in self.explanations:
                f.write(f"Sample {exp['sample_idx']}:\n")
                f.write(f"  Prediction: {exp['prediction']:.4f} ({exp['predicted_class']})\n")
                f.write(f"  R² Score: {exp['r2_score']:.4f}\n")
                f.write(f"  Top contributing features:\n")
                
                importance = exp['feature_importance']
                abs_imp = np.abs(importance)
                top_idx = np.argsort(abs_imp)[-5:][::-1]
                
                for idx in top_idx:
                    f.write(f"    {exp['feature_names'][idx]:40s} {importance[idx]:+.6f}\n")
                
                f.write("\n")
        
        print(f"\n{'='*60}")
        print(f"✓ LIME report generated successfully!")
        print(f"Output directory: {output_path}")
        print(f"Files created:")
        print(f"  - feature_importance.png")
        print(f"  - explanation_sample_*.png (x{num_individual})")
        print(f"  - explanations_comparison.png")
        print(f"  - lime_report.txt")
        print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    from models import CNN_BiLSTM_GaitDetector, get_device
    
    # Setup
    device = get_device()
    model = CNN_BiLSTM_GaitDetector(input_features=38, seq_length=128)
    
    # Create dummy data
    test_data = np.random.randn(20, 128, 38)
    
    # Initialize explainer
    explainer = GaitLIMEExplainer(model=model, device=device)
    
    # Generate report
    explainer.generate_explanation_report(
        test_data=test_data,
        num_samples_to_explain=10,
        output_dir='results/lime_outputs'
    )